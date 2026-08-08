"""
trainer/model_setup.py

Qwen3-VL + SFT + SAM + GRPO LoRA

               

     1. AutoProcessor.from_pretrained(base_model)
   
     2. add_tokens([45   SFT special token])          SFT adapter  
          <sam_feat>, <think>, </think>, <overall>, </overall>,
          <geometry_l_{0..19}>, </geometry_l_{0..19}>
   
     3. Qwen3VLForConditionalGeneration.from_pretrained(base_model)
   
     4. model.resize_token_embeddings(len(tokenizer))
            base   embed_tokens / lm_head     SFT      
   
     5. PeftModel.from_pretrained(model, sft_adapter_path)
                 LoRA delta q/k/v/o/MLP 
                SFT          embed_tokens / lm_head
             modules_to_save      ModulesToSaveWrapper 
   
     6. model.merge_and_unload()
            LoRA delta     base   attention/MLP  
            ModulesToSaveWrapper       SFT      embed_tokens / lm_head
                 Qwen3VLForConditionalGeneration
   
     7.    merger_weights.pt     model.visual.merger
   
     8.    SAMProjector +    sam_projector.pt
             model.sam_projector   nn.Module   /dtype    
   
     9. SAMEmbeddingInjector     model.get_input_embeddings()
          hook   PeftModel            nn.Embedding       
   
     10.     get_peft_model(model, GRPO LoRA)
             q/k/v/o/MLP    r=16      embed_tokens / lm_head
           SFT      GRPO      
   
     11.      vision tower / merger / sam_projector / embed_tokens / lm_head
           ModelConfig.freeze_*      

     
      SFT    / Special Token / hook               
         SAM Projector    SFT            flag    
         GRPO LoRA     embed_tokens / lm_head    modules_to_save
                       
    device_map        trainer      DDP / FSDP 
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    PeftModel = None
    get_peft_model = None

from configs import ModelConfig

from .sam_injection import SAMEmbeddingInjector, SAMProjector, load_sam_projector_state

LOGGER = logging.getLogger(__name__)


#      SFT     45   special token                                          

NEW_SPECIAL_TOKENS: List[str] = [
    "<sam_feat>",
    "<think>",   "</think>",
    "<overall>", "</overall>",
    *[f"<geometry_l_{k}>"  for k in range(24)],
    *[f"</geometry_l_{k}>" for k in range(24)],
]


#                                                                           


@dataclass
class SAMSetupConfig:
    """SAM                     ModelConfig          """

    sft_adapter_path: Optional[str] = None
    """SFT        adapter_config.json / adapter_model.safetensors /
       sam_projector.pt / merger_weights.pt"""

    merger_weights_path: Optional[str] = None
    """     merger_weights.pt   None      sft_adapter_path/merger_weights.pt """

    sam_projector_path: Optional[str] = None
    """     sam_projector.pt   None      sft_adapter_path/sam_projector.pt """

    sam_proj_hidden: int = 512
    """SAMProjector MLP          SFT       512  """

    sam_in_dim: int = 256
    """SAM3 RoI      """

    freeze_sam_projector: bool = True
    """     SAM Projector    RL      SFT         """

    freeze_merger: bool = True
    """     visual.merger SFT      """

    freeze_embed_tokens: bool = True
    """     embed_tokens SFT     special token      """

    freeze_lm_head: bool = True
    """     lm_head   embed_tokens     """

    local_files_only: bool = False
    """                """


#                                                                          


@dataclass
class SetupResult:
    model: nn.Module
    processor: object
    sam_projector: SAMProjector
    sam_injector: SAMEmbeddingInjector
    sam_token_id: int
    hook_handle: object
    """register_forward_hook            trainer       .remove() """


#                                                                            


def setup_model_and_processor(
    model_cfg: ModelConfig,
    sam_cfg: SAMSetupConfig,
) -> SetupResult:
    """     11             GRPOTrainer   model + processor """

    if PeftModel is None or LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is not installed. Please run: pip install peft")

    torch_dtype = _resolve_torch_dtype(model_cfg.torch_dtype)

    #    1) processor   
    processor_name = model_cfg.processor_name_or_path or model_cfg.model_name_or_path
    LOGGER.info("[1/11] Loading processor: %s", processor_name)
    processor = AutoProcessor.from_pretrained(
        processor_name,
        trust_remote_code=model_cfg.trust_remote_code,
        local_files_only=sam_cfg.local_files_only,
    )

    #    2) special token         
    new_tokens = [t for t in NEW_SPECIAL_TOKENS if t not in processor.tokenizer.get_vocab()]
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
        LOGGER.info("[2/11] Added %d new special tokens; vocab   %d",
                    len(new_tokens), len(processor.tokenizer))
    else:
        LOGGER.info("[2/11] All special tokens already present.")

    sam_token_id = processor.tokenizer.convert_tokens_to_ids("<sam_feat>")
    if sam_token_id is None or sam_token_id < 0:
        raise RuntimeError("Failed to register <sam_feat> token.")

    #    3) base model   
    LOGGER.info("[3/11] Loading base model: %s", model_cfg.model_name_or_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_cfg.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.trust_remote_code,
        attn_implementation=model_cfg.attn_implementation,
        local_files_only=sam_cfg.local_files_only,
    )
    model.config.use_cache = False

    #    4) resize embeddings   SFT            
    LOGGER.info("[4/11] Resizing token embeddings   %d", len(processor.tokenizer))
    model.resize_token_embeddings(len(processor.tokenizer))

    #    5+6)    SFT adapter       
    if sam_cfg.sft_adapter_path:
        LOGGER.info("[5/11] Loading SFT adapter: %s", sam_cfg.sft_adapter_path)
        model = PeftModel.from_pretrained(
            model, sam_cfg.sft_adapter_path, is_trainable=False
        )
        LOGGER.info("[6/11] Merging SFT LoRA delta into base; "
                    "modules_to_save (embed_tokens/lm_head) will be retained.")
        model = model.merge_and_unload()
        # merge_and_unload returns Qwen3VLForConditionalGeneration.
    else:
        LOGGER.warning("[5-6/11] sft_adapter_path is empty; skipping SFT merge. "
                       "GRPO will start from raw base model   strongly NOT recommended.")

    #    7) merger_weights.pt   
    merger_path = sam_cfg.merger_weights_path or _default_merger_path(sam_cfg.sft_adapter_path)
    if merger_path and os.path.isfile(merger_path):
        LOGGER.info("[7/11] Loading visual.merger weights: %s", merger_path)
        merger_state = torch.load(merger_path, map_location="cpu")
        merger = _get_visual_merger(model)
        if merger is None:
            raise AttributeError("Qwen3-VL visual merger was not found.")
        missing, unexpected = merger.load_state_dict(merger_state, strict=False)
        if missing or unexpected:
            LOGGER.warning("merger load mismatch: missing=%s, unexpected=%s",
                           missing, unexpected)
    else:
        LOGGER.warning("[7/11] merger_weights.pt not found at %s; "
                       "merger will keep base-model defaults (no SFT update).",
                       merger_path)

    #    8) SAMProjector +    sam_projector.pt   
    llm_hidden = _get_text_hidden_size(model)
    sam_projector = SAMProjector(
        in_dim=sam_cfg.sam_in_dim,
        hidden_dim=sam_cfg.sam_proj_hidden,
        out_dim=llm_hidden,
    )
    proj_path = sam_cfg.sam_projector_path or _default_proj_path(sam_cfg.sft_adapter_path)
    if proj_path and os.path.isfile(proj_path):
        LOGGER.info("[8/11] Loading sam_projector weights: %s", proj_path)
        load_sam_projector_state(sam_projector, proj_path)
    else:
        LOGGER.warning("[8/11] sam_projector.pt not found at %s; "
                       "projector will start from random init.", proj_path)

    sam_projector.to(dtype=torch_dtype)
    model.sam_projector = sam_projector  #          .to(device)   

    #    9) hook      
    sam_injector = SAMEmbeddingInjector(sam_projector, sam_token_id)
    hook_handle = model.get_input_embeddings().register_forward_hook(sam_injector)
    LOGGER.info("[9/11] Registered SAM injection hook on embed_tokens.")

    #    10) GRPO LoRA   hook        hook       nn.Embedding        
    if model_cfg.use_lora:
        peft_config = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            target_modules=list(model_cfg.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        LOGGER.info("[10/11] GRPO LoRA enabled: r=%d alpha=%d targets=%s",
                    model_cfg.lora_r, model_cfg.lora_alpha,
                    list(model_cfg.lora_target_modules))
    else:
        LOGGER.info("[10/11] GRPO LoRA disabled (full fine-tune mode).")

    #    11)        
    _apply_freeze_policy(model, model_cfg, sam_cfg)
    LOGGER.info("[11/11] Freeze policy applied.")

    if model_cfg.gradient_checkpointing:
        #   PEFT        PeftModel     base 
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    _log_trainable_parameters(model)

    return SetupResult(
        model=model,
        processor=processor,
        sam_projector=sam_projector,
        sam_injector=sam_injector,
        sam_token_id=sam_token_id,
        hook_handle=hook_handle,
    )


#                                                                             


def _resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_str}")
    return mapping[key]


def _default_merger_path(sft_dir: Optional[str]) -> Optional[str]:
    return str(Path(sft_dir) / "merger_weights.pt") if sft_dir else None


def _default_proj_path(sft_dir: Optional[str]) -> Optional[str]:
    return str(Path(sft_dir) / "sam_projector.pt") if sft_dir else None


def _get_visual_module(model: nn.Module) -> Optional[nn.Module]:
    current = model
    for _ in range(6):
        visual = getattr(current, "visual", None)
        if visual is not None:
            return visual
        nested = getattr(current, "model", None)
        if nested is not None and nested is not current:
            current = nested
            continue
        base_model = getattr(current, "base_model", None)
        if base_model is not None and base_model is not current:
            current = base_model
            continue
        break
    return None


def _get_visual_merger(model: nn.Module) -> Optional[nn.Module]:
    visual = _get_visual_module(model)
    return getattr(visual, "merger", None) if visual is not None else None


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


def _apply_freeze_policy(
    model: nn.Module,
    model_cfg: ModelConfig,
    sam_cfg: SAMSetupConfig,
) -> None:
    """        

          PEFT      
        model.base_model.model.visual...    PeftModel     
        model.visual...                          
    """
    base = _unwrap_peft(model)
    visual = _get_visual_module(base)

    # vision tower   backbone + merger 
    if model_cfg.freeze_vision_tower:
        if visual is not None:
            for param in visual.parameters():
                param.requires_grad = False

    # merger         vision_tower             merger 
    merger = _get_visual_merger(base)
    if sam_cfg.freeze_merger and merger is not None:
        for p in merger.parameters():
            p.requires_grad = False

    # SAM Projector
    if hasattr(base, "sam_projector"):
        for p in base.sam_projector.parameters():
            p.requires_grad = not sam_cfg.freeze_sam_projector

    # embed_tokens SFT modules_to_save     
    if sam_cfg.freeze_embed_tokens:
        emb = base.get_input_embeddings()
        for p in emb.parameters():
            p.requires_grad = False

    # lm_head
    if sam_cfg.freeze_lm_head and hasattr(base, "lm_head"):
        for p in base.lm_head.parameters():
            p.requires_grad = False


def _unwrap_peft(model: nn.Module) -> nn.Module:
    """   PEFT               """
    inner = model
    for _ in range(4):  #       PeftModel + base_model
        if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
            inner = inner.base_model.model
        else:
            break
    return inner


def _log_trainable_parameters(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (100.0 * train / total) if total > 0 else 0.0
    LOGGER.info("Trainable params: %s / %s (%.4f%%)",
                f"{train:,}", f"{total:,}", pct)
