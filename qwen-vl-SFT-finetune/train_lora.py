"""
Qwen3-VL LoRA training for PhysX-CoT with SAM features

        V2  
  1.    cot_finetune_v3        sam_feature <overall> <geometry_l_k>    
  2. SAMProjector 256   D_hidden   D_llm    MLP   SAM3 RoI       LLM     
  3. SAMEmbeddingInjector embed_tokens forward hook   <sam_feat>          
  4.      token <sam_feat> <overall>/<overall> <geometry_l_k>   
  5. <sam_feat>     labels     -100     loss 

SAM        
  -             <sam_feat_l_k>         token <sam_feat>
  -     embed_tokens hook     forward   
      SAMProjector(sam_feats[k])     k   <sam_feat>     embedding
  - Qwen3-VL special tokens are augmented with projected SAM features.

           
  torchrun --nproc_per_node=2 train_lora.py --config scripts/lora_config.json
"""

import os
import re
import json
import random
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, TaskType

from callbacks import (
    CheckpointCompletionCallback,
    CheckpointEvalCallback,
    PeriodicSnapshotCallback,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


#                                                                
#     Token   
#                                                                

_NEW_SPECIAL_TOKENS: List[str] = [
    "<sam_feat>",
    "<think>",    "</think>",
    "<overall>",  "</overall>",
    *[f"<geometry_l_{k}>"  for k in range(24)],
    *[f"</geometry_l_{k}>" for k in range(24)],
]

#    <sam_feat_l_0>              <sam_feat_l_{part_id}>       
_SAM_FEAT_PAT = re.compile(r"<sam_feat_l_\d+>")


def _get_visual_merger(model: nn.Module) -> nn.Module:
    current = model
    for _ in range(6):
        visual = getattr(current, "visual", None)
        if visual is not None and hasattr(visual, "merger"):
            return visual.merger
        nested = getattr(current, "model", None)
        if nested is not None and nested is not current:
            current = nested
            continue
        base_model = getattr(current, "base_model", None)
        if base_model is not None and base_model is not current:
            current = base_model
            continue
        break
    raise AttributeError("Qwen3-VL visual merger was not found.")


def _get_text_hidden_size(model: nn.Module) -> int:
    current = model
    for _ in range(6):
        config = getattr(current, "config", None)
        if config is not None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None and hasattr(text_config, "hidden_size"):
                return int(text_config.hidden_size)
            if hasattr(config, "hidden_size"):
                return int(config.hidden_size)
        base_model = getattr(current, "base_model", None)
        if base_model is not None and base_model is not current:
            current = base_model
            continue
        nested = getattr(current, "model", None)
        if nested is not None and nested is not current:
            current = nested
            continue
        break
    raise AttributeError("Unable to resolve the Qwen3-VL text hidden size.")


#                                                                
#       BatchSampler
#                                                                

def _estimate_sample_length(sample: dict) -> int:
    """            token                    """
    return sum(len(t.get("value", "")) for t in sample.get("conversations", []))


class LengthBucketBatchSampler:
    """
    DDP-aware      BatchSampler 

                   batch          padding    

       
      1.       token            
      2.   batches_per_bucket   world_size   batch_size       
                                  
      3.               seed   epoch             
      4. DDP   rank     batch         rank (mod world_size)   batch 
                       batch    

       
      batches_per_bucket :         all-replica batch    100 
                                      padding           
                                              DataLoader 
    """

    def __init__(
        self,
        lengths:            List[int],
        batch_size:         int,
        num_replicas:       int  = 1,
        rank:               int  = 0,
        batches_per_bucket: int  = 100,
        seed:               int  = 42,
        drop_last:          bool = True,
    ):
        self.lengths          = lengths
        self.batch_size       = batch_size
        self.num_replicas     = num_replicas
        self.rank             = rank
        #       =                batch size
        self.bucket_size      = batches_per_bucket * num_replicas * batch_size
        self.seed             = seed
        self.drop_last        = drop_last
        self.epoch            = 0

    # Trainer     epoch           batch_sampler      
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _make_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        #            
        sorted_idx = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        #   bucket_size   
        buckets = [
            sorted_idx[i: i + self.bucket_size]
            for i in range(0, len(sorted_idx), self.bucket_size)
        ]
        #              
        for b in buckets:
            rng.shuffle(b)
        rng.shuffle(buckets)

        flat = [i for b in buckets for i in b]

        #     (world_size   batch_size)         DDP    batch    
        keep = (len(flat) // (self.num_replicas * self.batch_size)) * (
            self.num_replicas * self.batch_size
        )
        flat = flat[:keep]

        return [flat[i: i + self.batch_size] for i in range(0, len(flat), self.batch_size)]

    def __iter__(self):
        all_batches = self._make_batches()
        #   rank     rank, rank+W, rank+2W     batch      
        for batch in all_batches[self.rank:: self.num_replicas]:
            yield batch

    def __len__(self) -> int:
        return len(self.lengths) // (self.num_replicas * self.batch_size)


#                                                                
#  SAM        MLP 
#                                                                

class SAMProjector(nn.Module):
    """
    SAM3 RoI      [D_sam=256]   [D_hidden]   [D_llm]

      SAM3 encoder   RoI      256   float32    
    Qwen3-VL text hidden size is read from the checkpoint configuration.

       Linear   GELU   Linear
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 3584):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim]     [..., out_dim]"""
        return self.net(x)


#                                                                
#  embed_tokens Forward Hook SAM      
#                                                                

class SAMEmbeddingInjector:
    """
        embed_tokens   forward hook 

      model.forward()    embed_tokens(input_ids)        
      <sam_feat>     embedding     SAMProjector     

         <sam_feat> token       l_0 l_1 l_2...      
      npz                   

            GPU              
            output.clone() + index_copy    in-place     tensor 
    """

    def __init__(self, projector: SAMProjector, sam_token_id: int):
        self.projector    = projector
        self.sam_token_id = sam_token_id
        self._pending_feats: Optional[torch.Tensor] = None  # [B, N_parts, D_sam]

    def set_batch(self, sam_feats: Optional[torch.Tensor]) -> None:
        """    model.forward()          batch   SAM      """
        self._pending_feats = sam_feats

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor,   # [B, L, D_llm]
    ) -> torch.Tensor:
        if self._pending_feats is None:
            return output

        input_ids = inputs[0]                        # [B, L]
        sam_feats = self._pending_feats
        self._pending_feats = None                   #             

        #     LLM     
        projected = self.projector(
            sam_feats.to(device=output.device, dtype=output.dtype)
        )                                            # [B, N_parts, D_llm]

        # index_copy         output in-place   
        output = output.clone()
        B = input_ids.size(0)
        for b in range(B):
            positions = (input_ids[b] == self.sam_token_id).nonzero(as_tuple=True)[0]
            n = min(len(positions), projected.size(1))
            if n > 0:
                output[b] = output[b].index_copy(
                    0, positions[:n], projected[b, :n]
                )
        return output


#                                                                
#      
#                                                                

@dataclass
class ScriptArguments:
    model_path: str = field(
        default="Qwen/Qwen3-VL-8B-Instruct"
    )
    data_path: str = field(
        default="./data/renders",
        metadata={"help": "Directory containing rendered training images."},
    )
    annotation_path: str = field(
        default="./data/train.jsonl",
        metadata={"help": "PhysX-CoT training annotations in JSONL format."},
    )
    eval_annotation_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional validation annotations in JSONL format."},
    )
    #    SAM                                                    
    sam_feature_dir: Optional[str] = field(
        default="./data/sam_features",
        metadata={
            "help": "Directory containing cached SAM features."
        },
    )
    sam_proj_hidden: int = field(
        default=512,
        metadata={"help": "SAMProjector MLP        256   sam_proj_hidden   D_llm "},
    )
    #    LoRA                                                     
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    #                                                          
    max_length: int = field(default=8704)
    #                                                         
    max_pixels: int = field(default=200704)   # 448 x 448
    min_pixels: int = field(default=200704)   # 448 x 448
    #    Checkpoint                                         
    eval_image: Optional[str] = field(
        default=None,
        metadata={"help": "checkpoint                   "},
    )
    eval_device: str = field(default="cpu")
    eval_max_new_tokens: int = field(default=2048)
    eval_out_root: str = field(default="./checkpoint_evals")
    #               save_total_limit rotate           
    periodic_save_steps: int = field(
        default=0,
        metadata={
            "help": "           checkpoint      "
                    "{output_dir}/periodic_snapshots/      0      "
        },
    )
    #         BatchSampler                                    
    bucket_batches: int = field(
        default=20,
        metadata={
            "help": "LengthBucketBatchSampler   batches_per_bucket       "
                    "   padding     loss                  "
                    "   20   DDP   2 + micro-batch 2        "
        },
    )


#                                                                
#     
#                                                                

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class PhysXCoTDataset(Dataset):
    """
    CoT V3               cache     

         2      
      Turn 1  Human : <image> + overall_cot_prompt
              GPT   : <think> </think><overall> </overall>
      Turn 2  Human : geometry_prompt   part_id 
              GPT   : <geometry_l_k> </geometry_l_k>

    label masking    
      -    user turn   assistant turn           -100
      - <sam_feat>    SAM         -100     loss
      - <think> </think>   <overall> </overall>     loss CoT      
    """

    def __init__(
        self,
        samples:         List[Dict],
        data_root:       str,
        processor,
        max_length:      int           = 8704,
        sam_feature_dir: Optional[str] = None,
    ):
        self.samples         = samples
        self.data_root       = data_root
        self.processor       = processor
        self.max_length      = max_length
        self.sam_feature_dir = sam_feature_dir or ""

        #       token ID           
        tok = processor.tokenizer
        self._im_start_id  = tok.convert_tokens_to_ids("<|im_start|>")
        self._assistant_id = tok.convert_tokens_to_ids("assistant")
        self._sam_token_id = tok.convert_tokens_to_ids("<sam_feat>")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._process_sample(self.samples[idx])

    #    SAM                                                 

    def _load_sam_feats(self, sample: Dict) -> Optional[torch.Tensor]:
        """    sam_feature      NPZ    [N_parts, 256] float32   None """
        if not self.sam_feature_dir or not sample.get("sam_feature"):
            return None
        npz_path = os.path.join(self.sam_feature_dir, sample["sam_feature"])
        try:
            data     = np.load(npz_path)
            part_ids = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
            feats    = [torch.tensor(data[k], dtype=torch.float32) for k in part_ids]
            return torch.stack(feats, dim=0)   # [N_parts, 256]
        except Exception as exc:
            logger.warning(f"SAM        ({npz_path}): {exc}")
            return None

    #                                                       

    def _process_sample(self, sample: Dict) -> Dict[str, Any]:
        convs     = sample["conversations"]
        messages  = []
        image_obj = None

        for turn in convs:
            role    = "user" if turn["from"] == "human" else "assistant"
            content = _SAM_FEAT_PAT.sub("<sam_feat>", turn["value"])

            if "<image>" in content and image_obj is None:
                img_path = os.path.join(self.data_root, sample.get("image", ""))
                try:
                    image_obj = Image.open(img_path).convert("RGB")
                    image_obj = ImageOps.pad(
                        image_obj,
                        (448, 448),
                        method=Image.Resampling.BICUBIC,
                        color=(0, 0, 0),
                    )
                except Exception:
                    image_obj = Image.new("RGB", (448, 448), color=0)

                messages.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": image_obj},
                        {"type": "text",  "text": content.replace("<image>", "").strip()},
                    ],
                })
            else:
                messages.append({"role": role, "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=text,
            images=[image_obj] if image_obj is not None else None,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {
            k: v if k == "image_grid_thw" else v.squeeze(0)
            for k, v in inputs.items()
        }

        input_ids = inputs["input_ids"]
        labels    = input_ids.clone()

        #    assistant turn label masking                        
        #   assistant turn      loss <|im_start|>assistant\n   token      -100
        in_assistant = False
        i = 0
        while i < len(input_ids):
            tid = input_ids[i].item()
            if tid == self._im_start_id:
                if (i + 1 < len(input_ids)
                        and input_ids[i + 1].item() == self._assistant_id):
                    in_assistant = True
                    labels[i] = labels[i + 1] = -100
                    i += 2
                    if i < len(input_ids):       # mask '\n' after 'assistant'
                        labels[i] = -100
                        i += 1
                else:
                    if in_assistant:
                        in_assistant = False
                    labels[i] = -100
                    i += 1
            elif not in_assistant:
                labels[i] = -100
                i += 1
            else:
                i += 1

        #    <sam_feat>      loss                             
        sam_pos = (input_ids == self._sam_token_id).nonzero(as_tuple=True)[0]
        labels[sam_pos] = -100

        inputs["labels"]    = labels
        inputs["sam_feats"] = self._load_sam_feats(sample)
        return inputs


#                                                                
#  Collator
#                                                                

class VLCollator:
    """
       padding collator 

       V2     
           sam_feats               
        batch["sam_feats"]      [B, max_parts, 256]  float32
        batch["sam_feat_masks"] [B, max_parts]        bool True=   
    """

    def __init__(self, pad_token_id: int, max_length: int = 8704):
        self.pad_id     = pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, Any]:
        #     sam_feats       /   key   
        sam_feats_list = [f.pop("sam_feats", None) for f in features]

        text_keys = ["input_ids", "attention_mask", "labels"]
        batch: Dict[str, Any] = {}

        for key in text_keys:
            seqs = [f[key] for f in features if key in f]
            if not seqs:
                continue
            seqs    = [s[:self.max_length] for s in seqs]
            max_len = max(s.size(0) for s in seqs)
            padded  = []
            for s in seqs:
                pad_val = (
                    self.pad_id if key == "input_ids"
                    else (0 if key == "attention_mask" else -100)
                )
                pad = torch.full((max_len - s.size(0),), pad_val, dtype=s.dtype)
                padded.append(torch.cat([s, pad], dim=0))
            batch[key] = torch.stack(padded, dim=0)

        for key in ["pixel_values", "image_grid_thw"]:
            vals = [f[key] for f in features if key in f]
            if vals:
                batch[key] = torch.cat(vals, dim=0)

        #    SAM      pad   batch                     
        valid = [sf for sf in sam_feats_list if sf is not None]
        if valid:
            max_parts = max(sf.size(0) for sf in valid)
            feat_dim  = valid[0].size(1)          # 256
            B         = len(features)
            padded_sf = torch.zeros(B, max_parts, feat_dim, dtype=torch.float32)
            feat_mask = torch.zeros(B, max_parts,           dtype=torch.bool)
            for b, sf in enumerate(sam_feats_list):
                if sf is not None:
                    n = sf.size(0)
                    padded_sf[b, :n] = sf
                    feat_mask[b, :n] = True
            batch["sam_feats"]      = padded_sf   # [B, max_parts, 256]
            batch["sam_feat_masks"] = feat_mask   # [B, max_parts]

        return batch


#                                                                
#      Trainer SAM      
#                                                                

class SAMPhysXTrainer(Trainer):
    """
       Trainer     compute_loss     SAMEmbeddingInjector
      embed_tokens hook      batch   SAM      

    SAMProjector     model.sam_projector       
              Trainer   optimizer   DDP    
    """

    def __init__(
        self,
        sam_injector: SAMEmbeddingInjector,
        bucket_batches: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sam_injector   = sam_injector
        self.bucket_batches = int(bucket_batches)

    def get_train_dataloader(self) -> DataLoader:
        """
             DataLoader    LengthBucketBatchSampler 

                       batch      padding    
              shuffle     20-40%     padding     
        """
        dataset = self.train_dataset
        if not hasattr(dataset, "samples"):
            logger.warning("train_dataset   .samples         DataLoader ")
            return super().get_train_dataloader()

        lengths = [_estimate_sample_length(s) for s in dataset.samples]
        logger.info(
            "LengthBucketBatchSampler     %d    world_size=%d rank=%d "
            "per_device_batch=%d",
            len(lengths), self.args.world_size, self.args.process_index,
            self.args.per_device_train_batch_size,
        )

        batch_sampler = LengthBucketBatchSampler(
            lengths            = lengths,
            batch_size         = self.args.per_device_train_batch_size,
            num_replicas       = self.args.world_size,
            rank               = self.args.process_index,
            batches_per_bucket = self.bucket_batches,
            seed               = self.args.seed,
            drop_last          = True,
        )

        num_workers = self.args.dataloader_num_workers
        return DataLoader(
            dataset,
            batch_sampler      = batch_sampler,
            num_workers        = num_workers,
            collate_fn         = self.data_collator,
            pin_memory         = self.args.dataloader_pin_memory,
            persistent_workers = num_workers > 0,
            prefetch_factor    = 2 if num_workers > 0 else None,
        )

    def compute_loss(
        self,
        model:          nn.Module,
        inputs:         Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        sam_feats = inputs.pop("sam_feats",      None)
        inputs.pop("sam_feat_masks", None)

        #   hook    SAM    embed_tokens forward         
        self.sam_injector.set_batch(sam_feats)

        outputs = model(**inputs)
        loss    = outputs.loss
        return (loss, outputs) if return_outputs else loss


#                                                                
#       
#                                                                

def train() -> None:
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    #                                                       
    logger.info(f"     {script_args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        script_args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.config.use_cache = False

    #    Processor +    Token                               
    processor = AutoProcessor.from_pretrained(
        script_args.model_path,
        local_files_only=True,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    #             token           
    new_tokens = [t for t in _NEW_SPECIAL_TOKENS if t not in processor.tokenizer.get_vocab()]
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
        logger.info(
            f"   {len(new_tokens)}     token        {len(processor.tokenizer)}"
        )
    else:
        logger.info("   token           ")

    sam_token_id = processor.tokenizer.convert_tokens_to_ids("<sam_feat>")

    #    LoRA                                                  
    # modules_to_save=["embed_tokens", "lm_head"]      
    #       resize_token_embeddings       <think>/<overall>/
    # <geometry_l_k>/<sam_feat>   special token        base model
    #               modules_to_save LoRA      adapter 
    #               lm_head                
    #   argmax         token      token    U+FFFD     
    target_modules = [m.strip() for m in script_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Vision Merger      
    visual_merger = _get_visual_merger(model)
    for p in visual_merger.parameters():
        p.requires_grad_(True)

    #    SAM Projector    MLP                               
    llm_hidden_dim = _get_text_hidden_size(model)
    sam_projector  = SAMProjector(
        in_dim     = 256,
        hidden_dim = script_args.sam_proj_hidden,
        out_dim    = llm_hidden_dim,
    )
    #     model          Trainer optimizer + DDP     
    model.sam_projector = sam_projector

    model.print_trainable_parameters()
    logger.info(
        "SAMProjector 256   %d   %d     %s",
        script_args.sam_proj_hidden,
        llm_hidden_dim,
        f"{sum(p.numel() for p in sam_projector.parameters()):,}",
    )

    #    embed_tokens Hook                                     
    sam_injector = SAMEmbeddingInjector(sam_projector, sam_token_id)
    hook_handle  = model.get_input_embeddings().register_forward_hook(sam_injector)

    #                                                         
    samples = load_jsonl(script_args.annotation_path)
    max_len = script_args.max_length

    dataset = PhysXCoTDataset(
        samples         = samples,
        data_root       = script_args.data_path,
        processor       = processor,
        max_length      = max_len,
        sam_feature_dir = script_args.sam_feature_dir,
    )
    eval_dataset = None
    if script_args.eval_annotation_path:
        eval_samples = load_jsonl(script_args.eval_annotation_path)
        eval_dataset = PhysXCoTDataset(
            samples=eval_samples,
            data_root=script_args.data_path,
            processor=processor,
            max_length=max_len,
            sam_feature_dir=script_args.sam_feature_dir,
        )

    collator = VLCollator(
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        max_length   = max_len,
    )

    #    Callbacks                                              
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    #          
    #   1) Completion         processor / merger / sam_projector 
    #           Eval worker    checkpoint      
    #   2) Snapshot    Completion                 
    #   3) Eval                   worker 
    callbacks: List[TrainerCallback] = [
        CheckpointCompletionCallback(
            processor     = processor,
            sam_projector = sam_projector,
            visual_merger = visual_merger,
        ),
    ]

    if script_args.periodic_save_steps > 0:
        callbacks.append(
            PeriodicSnapshotCallback(
                every_n_steps = script_args.periodic_save_steps,
            )
        )
        logger.info(
            "PeriodicSnapshotCallback       %d        periodic_snapshots/",
            script_args.periodic_save_steps,
        )

    if script_args.eval_image:
        physx_root = str(pathlib.Path(__file__).parent.parent)
        callbacks.append(
            CheckpointEvalCallback(
                eval_image          = script_args.eval_image,
                base_model          = script_args.model_path,
                physx_root          = physx_root,
                eval_out_root       = script_args.eval_out_root,
                eval_device         = script_args.eval_device,
                eval_max_new_tokens = script_args.eval_max_new_tokens,
                skip_auto_sam       = True,
            )
        )
        logger.info(
            "CheckpointEvalCallback     image=%s device=%s",
            script_args.eval_image, script_args.eval_device,
        )

    #    SAMPhysXTrainer                                         
    trainer = SAMPhysXTrainer(
        sam_injector   = sam_injector,
        bucket_batches = script_args.bucket_batches,
        model          = model,
        args           = training_args,
        train_dataset  = dataset,
        eval_dataset   = eval_dataset,
        data_collator  = collator,
        callbacks      = callbacks,
    )

    #                                                         
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logger.info("      checkpoint        ...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    #       LoRA adapter + Vision Merger + SAM Projector     
    if trainer.args.should_save:
        model.save_pretrained(training_args.output_dir)

        merger_state = {
            n: p.detach().cpu()
            for n, p in visual_merger.named_parameters()
        }
        torch.save(merger_state, os.path.join(training_args.output_dir, "merger_weights.pt"))

        torch.save(
            sam_projector.state_dict(),
            os.path.join(training_args.output_dir, "sam_projector.pt"),
        )

        processor.save_pretrained(training_args.output_dir)
        logger.info(f"       {training_args.output_dir}")

    hook_handle.remove()


if __name__ == "__main__":
    train()
