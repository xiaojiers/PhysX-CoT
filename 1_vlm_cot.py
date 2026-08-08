"""PhysX-CoT Structured Physical CoT inference.

Turn 1 emits the ordered state trajectory (parts, 2D/3D grounding, relations,
coarse geometry, and surface cues). Turn 2 emits local voxel occupancy as RLE.
The decoder and SimReady stages consume these machine-parseable outputs.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


#                                                                
#       & Prompt
#                                                                

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# External model components are intentionally not bundled with the release.
# Set these variables when using local checkpoints; the base model may also be
# a Hugging Face identifier understood by ``from_pretrained``.
DEFAULT_ADAPTER = os.environ.get(
    "PHYSX_COT_ADAPTER", os.path.join(_PROJ_ROOT, "pretrain", "vlm")
)
DEFAULT_BASE_MODEL = os.environ.get(
    "PHYSX_COT_BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct"
)
DEFAULT_SAM_FEATURE_DIR = os.environ.get(
    "PHYSX_COT_SAM_FEATURE_DIR", os.path.join(_PROJ_ROOT, "dataset", "sam_feature")
)
DEFAULT_SAM3_ROOT = os.environ.get(
    "PHYSX_COT_SAM3_ROOT", os.path.join(_PROJ_ROOT, "external", "sam3")
)
DEFAULT_SAM3_CHECKPOINT = os.environ.get(
    "PHYSX_COT_SAM3_CHECKPOINT",
    os.path.join(DEFAULT_SAM3_ROOT, "checkpoints", "sam3.pt"),
)

# Turn1 Prompt                  cot_finetune_v3   human turn 
_DEFAULT_TURN1_PROMPT = (
    "Analyze the 3D physical object in the image and output its complete physical asset description.\n\n"
    "First, reason step by step inside <think>:\n"
    "Step 1: Count the total number of independent structural parts (`part_count`).\n"
    "Step 2: For each part, record its 2D image bounding range `bbox_2d` = "
    "[x_min, x_max, y_min, y_max] (normalized 0~1), its 3D voxel bounding range "
    "`bbox_3d` = [x_min, x_max, y_min, y_max, z_min, z_max] in the canonical "
    "32 32 32 voxel space (both use the same min/max vertex format), and the SAM "
    "visual feature token `sam_feat` = <sam_feat_l_{part_id}> which encodes the "
    "region appearance from the SAM3 encoder.\n"
    "Step 3: For each part, describe the relative 3D position of its directly "
    "adjacent parts using discrete direction labels "
    "(top/bottom/left/right/front/back/center). Non-adjacent parts are not recorded.\n"
    "Step 4: Identify each part's dominant geometric primitive (`shape_label`: "
    "cuboid/cylinder/sphere/complex), its major axis orientation (`major_axis`: "
    "x/y/z), and its aspect ratio (`aspect_ratio`: "
    "very_flat/flat/balanced/tall/elongated).\n"
    "Step 5: Assess each part's surface perceptual properties: `hardness` "
    "(soft/semi_rigid/rigid), `roughness` (smooth/textured/rough), `reflectivity` "
    "(matte/glossy/highly_reflective), and `transparency` "
    "(opaque/translucent/transparent).\n\n"
    "Then output the structured physical description inside <overall>."
)

# Turn2 prompt can be supplied with ``--geo_prompt_file`` when using a custom
# geometry template.
_DEFAULT_TURN2_PROMPT = (
    "Based on the `bbox_3d` of `l_{part_id}` from Step 2, generate its local 3D "
    "voxel occupancy as a 1D run-length encoded sequence. Encoding: "
    "local_id = (x-x_min)*(dy*dz) + (y-y_min)*dz + (z-z_min), where "
    "dy=y_max-y_min+1, dz=z_max-z_min+1 (derived from bbox_3d). Merge maximal "
    "consecutive runs (e.g. 0 1-5 36-41 ...). Wrap the result in "
    "<geometry_l_{part_id}>.</geometry_l_{part_id}>."
)

#            token adapter   tokenizer                 
_EXPECTED_SPECIAL_TOKENS: List[str] = [
    "<sam_feat>",
    "<think>", "</think>",
    "<overall>", "</overall>",
    *[f"<geometry_l_{k}>"  for k in range(24)],
    *[f"</geometry_l_{k}>" for k in range(24)],
]
MAX_PARTS = 24


#                                                                
#  SAM Projector        train_lora.py       
#                                                                

class SAMProjector(nn.Module):
    """256   hidden   D_llm    MLP          """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 3584):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAMInferenceInjector:
    """
        embed_tokens forward hook 

         SAMEmbeddingInjector       
             batch set    sam_feats hook            
                    SAM      _feats         
                   N   forward         
    """

    def __init__(self, projector: SAMProjector, sam_token_id: int):
        self.projector = projector
        self.sam_token_id = sam_token_id
        self._feats: Optional[torch.Tensor] = None  # [N_parts, 256] float32 / cpu

    def set_feats(self, feats: Optional[torch.Tensor]) -> None:
        self._feats = feats

    def clear(self) -> None:
        self._feats = None

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor,   # [B, L, D_llm]
    ) -> torch.Tensor:
        if self._feats is None:
            return output

        input_ids = inputs[0]                    # [B, L]
        projected = self.projector(
            self._feats.to(device=output.device, dtype=output.dtype)
        )                                        # [N_parts, D_llm]

        output = output.clone()
        B = input_ids.size(0)
        for b in range(B):
            positions = (input_ids[b] == self.sam_token_id).nonzero(as_tuple=True)[0]
            n = min(len(positions), projected.size(0))
            if n > 0:
                output[b] = output[b].index_copy(0, positions[:n], projected[:n])
        return output


#                                                                
#  SAM3           Two-pass          
#                                                                

class SAM3FeatureExtractor:
    """
            SAM3 RoI feature extractor is provided by ``dataset/catch_sam_feature.py``.

      1. processor.set_image(image)   backbone_out       
      2.     add_geometric_prompt(box)    encoder   hook    memory
      3. RoI Align    encoder memory   [256] float32

               extract()    import sam3       
                OOD                 

      VLM       
        sam3_device      VLM device        ~3-5 GB    
        set_image / add_geometric_prompt      bfloat16 autocast   /    
    """

    def __init__(
        self,
        sam3_root: str,
        checkpoint: str,
        device: str = "cuda:0",
        roi_output_size: int = 3,
    ):
        self.sam3_root      = Path(sam3_root)
        self.checkpoint     = checkpoint
        self.device         = device
        self.roi_output_size = roi_output_size

        self._model     = None
        self._processor = None
        self._enc_memory:         Optional[torch.Tensor] = None
        self._enc_spatial_shapes: Optional[torch.Tensor] = None

    #                                                      
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.sam3_root.is_dir():
            raise FileNotFoundError(f"SAM3         : {self.sam3_root}")
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f"SAM3      : {self.checkpoint}")

        if str(self.sam3_root) not in sys.path:
            sys.path.insert(0, str(self.sam3_root))

        #    import         sam3     
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        LOGGER.info("     SAM3   : %s (device=%s)", self.checkpoint, self.device)
        self._model = build_sam3_image_model(
            checkpoint_path=self.checkpoint,
            load_from_HF=False,
            device=self.device,
        )
        # build_sam3_image_model    _setup_device_and_mode   `if device == "cuda"`
        #           "cuda:0" / "cuda:1"       .cuda()        CPU 
        #      .to(device)      cuda / cuda:N / cpu        
        self._model = self._model.to(self.device)
        self._processor = Sam3Processor(self._model, device=self.device)

    #    encoder forward hook                                  
    def _hook_fn(self, module, inputs, output) -> None:
        self._enc_memory         = output["memory"].detach()
        self._enc_spatial_shapes = output["spatial_shapes"].detach()

    #    RoI Align                                           
    def _roi_pool_enc(
        self,
        memory:         torch.Tensor,   # [ HW, N, d]
        spatial_shapes: torch.Tensor,   # [num_levels, 2]
        box_cxcywh:     List[float],
    ) -> np.ndarray:
        import torchvision.ops as tvops

        enc_h, enc_w = spatial_shapes[0].tolist()
        N, d = memory.shape[1], memory.shape[2]
        feat_map = (
            memory.permute(1, 2, 0)
                  .reshape(N, d, enc_h, enc_w)
                  .float()
        )

        cx, cy, w, h = box_cxcywh
        x1 = max((cx - w / 2) * enc_w, 0.0)
        y1 = max((cy - h / 2) * enc_h, 0.0)
        x2 = min((cx + w / 2) * enc_w, enc_w - 1e-3)
        y2 = min((cy + h / 2) * enc_h, enc_h - 1e-3)

        box = torch.tensor([[x1, y1, x2, y2]],
                           dtype=torch.float32, device=feat_map.device)
        roi = tvops.roi_align(feat_map, [box],
                              output_size=self.roi_output_size, aligned=True)
        return roi.mean(dim=[-2, -1]).squeeze(0).cpu().float().numpy()

    #                                                     
    @torch.inference_mode()
    def extract(
        self,
        image:  Image.Image,
        bboxes_cxcywh: Dict[str, List[float]],
    ) -> Dict[str, np.ndarray]:
        """
        Parameters
        ----------
        image          : PIL.Image (RGB)      SAM3 processor     resize 
        bboxes_cxcywh  : {'l_0': [cx, cy, w, h], ...}         0~1

        Returns
        -------
        {'l_0': np.ndarray[256] float32, ...}
        """
        self._ensure_loaded()
        device_type = str(self._model.device.type)

        # Step 1: backbone_fpn                
        with torch.autocast(device_type, dtype=torch.bfloat16):
            base_state = self._processor.set_image(image)
        backbone_out = base_state["backbone_out"]

        feats: Dict[str, np.ndarray] = {}

        # Step 2:     encoder       backbone_out 
        for pid, box in bboxes_cxcywh.items():
            part_state = {
                "backbone_out":    backbone_out,
                "original_height": base_state["original_height"],
                "original_width":  base_state["original_width"],
            }
            handle = self._model.transformer.encoder.register_forward_hook(self._hook_fn)
            try:
                with torch.autocast(device_type, dtype=torch.bfloat16):
                    self._processor.add_geometric_prompt(
                        box=box, label=True, state=part_state,
                    )
            finally:
                handle.remove()

            if self._enc_memory is not None:
                feats[pid] = self._roi_pool_enc(
                    self._enc_memory, self._enc_spatial_shapes, box,
                )
                self._enc_memory = None
                self._enc_spatial_shapes = None

        return feats


#                                                                
#       LoRA + Merger + SAM Projector 
#                                                                

def _get_model_device(model) -> str:
    """
               CUDA    model.device   device_map          cpu  
    """
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, (str, torch.device)) and str(v).startswith("cuda"):
                return str(v)
    p = next(model.parameters())
    return str(p.device) if p.is_cuda else "cuda:0"


def _get_visual_merger(model) -> Optional[nn.Module]:
    """Return the Qwen3-VL visual merger through plain or PEFT wrappers."""
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
    return None


def _get_text_hidden_size(model) -> int:
    """Resolve the language hidden size from Qwen3-VL and PEFT configs."""
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


def _ensure_special_tokens(tokenizer, model) -> None:
    """
       base model   embed_tokens / lm_head           

         
        Qwen3-VL uses a multimodal tokenizer whose vocabulary can differ
        between the base checkpoint and a PhysX-CoT adapter.
              `model.resize_token_embeddings(len(tokenizer))`       
           151710  LoRA adapter   embed/lm_head          
             base    152064    adapter     shape mismatch

          tokenizer     token       base   embedding   
        `len(tokenizer)`            
    """
    # 1.               token     adapter   tokenizer       
    missing = [t for t in _EXPECTED_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if missing:
        LOGGER.warning("Tokenizer    %d     token    %s", len(missing), missing)
        tokenizer.add_tokens(missing, special_tokens=True)

    # 2.      base embedding   tokenizer       adapter        
    current_vocab = model.get_input_embeddings().weight.shape[0]
    target_vocab  = len(tokenizer)
    if current_vocab != target_vocab:
        LOGGER.info(
            "Resize base embedding: %d   %d       adapter       ",
            current_vocab, target_vocab,
        )
        model.resize_token_embeddings(target_vocab)


def load_model_and_processor(
    adapter_path: str,
    base_model: str,
    merge_weights: bool,
    min_pixels: int,
    max_pixels: int,
    device: str = "cuda:0",
    sam_proj_hidden: int = 512,
) -> Tuple[object, AutoProcessor, SAMInferenceInjector, object]:
    """
       base + LoRA adapter + Vision Merger + SAM Projector    SAM    hook 

    Returns
    -------
    (model, processor, sam_injector, hook_handle)
    """
    LOGGER.info("   processor: %s", adapter_path)
    processor = AutoProcessor.from_pretrained(
        adapter_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )

    # FlashAttention2     CUDA CPU       eager    HF   raise
    # ValueError      callback     cpu            GPU 
    attn_impl = "flash_attention_2" if str(device).lower().startswith("cuda") else "eager"
    LOGGER.info("   base model: %s (device=%s, attn=%s)", base_model, device, attn_impl)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=device,
        trust_remote_code=True,
    )

    #          base model    resize            LoRA     shape mismatch
    _ensure_special_tokens(processor.tokenizer, base)

    LOGGER.info("   LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)

    #    Vision Merger           LoRA               
    merger_path = os.path.join(adapter_path, "merger_weights.pt")
    merger = _get_visual_merger(model)
    if merger is not None and os.path.isfile(merger_path):
        state = torch.load(merger_path, map_location="cpu")
        missing, unexpected = merger.load_state_dict(state, strict=False)
        LOGGER.info(
            "Vision Merger     (%d   ) missing=%d, unexpected=%d",
            len(state), len(missing), len(unexpected),
        )
    else:
        LOGGER.warning("Vision Merger      : %s", merger_path)

    if merge_weights:
        LOGGER.info("   LoRA     base model...")
        model = model.merge_and_unload()
        model.eval()

    #    SAM Projector                                            
    # PeftModel   config     base      hidden_size
    llm_hidden_dim = _get_text_hidden_size(model)

    projector = SAMProjector(
        in_dim=256, hidden_dim=sam_proj_hidden, out_dim=llm_hidden_dim,
    )
    sp_path = os.path.join(adapter_path, "sam_projector.pt")
    if os.path.isfile(sp_path):
        state = torch.load(sp_path, map_location="cpu")
        projector.load_state_dict(state)
        LOGGER.info("SAM Projector    : %s", sp_path)
    else:
        LOGGER.warning(
            "SAM Projector      : %s <sam_feat>                 smoke test ",
            sp_path,
        )

    target_device = _get_model_device(model)
    projector = projector.to(device=target_device, dtype=torch.bfloat16)
    projector.eval()

    #       embed_tokens hook                                   
    sam_token_id = processor.tokenizer.convert_tokens_to_ids("<sam_feat>")
    injector = SAMInferenceInjector(projector, sam_token_id)
    hook_handle = model.get_input_embeddings().register_forward_hook(injector)

    #           CPU offload
    if hasattr(model, "hf_device_map"):
        LOGGER.info("Device map: %s", model.hf_device_map)
    if target_device.startswith("cuda"):
        mem = torch.cuda.memory_allocated(target_device) / 1024 ** 3
        LOGGER.info("GPU %s     : %.2f GB", target_device, mem)

    return model, processor, injector, hook_handle


#                                                                
#  SAM      &     
#                                                                

def load_sam_feats(npz_path: str) -> torch.Tensor:
    """NPZ   [N_parts, 256] float32   l_0, l_1, ...      """
    data = np.load(npz_path)
    keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
    feats = [torch.tensor(data[k], dtype=torch.float32) for k in keys]
    return torch.stack(feats, dim=0)


def resolve_sam_feature_path(
    image_path: str,
    explicit: Optional[str],
    sam_feature_dir: Optional[str],
) -> Optional[str]:
    """
               SAM    npz 
      1.    --sam_feature       dir +    stem      
      2. --sam_feature_dir            
         image     renders_cond/{obj_id}_/{img_id}.png
            npz     {sam_feature_dir}/{obj_id}/{img_id}.npz
      3.       image.png   image.npz     
          None 
    """
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        if os.path.isdir(explicit):
            cand = os.path.join(explicit, Path(image_path).stem + ".npz")
            if os.path.isfile(cand):
                return cand
        LOGGER.warning("   --sam_feature      : %s", explicit)

    img_p = Path(image_path)

    if sam_feature_dir and os.path.isdir(sam_feature_dir):
        parent = img_p.parent.name
        obj_id = parent.rstrip("_") if parent.endswith("_") else parent
        cand = Path(sam_feature_dir) / obj_id / f"{img_p.stem}.npz"
        if cand.is_file():
            return str(cand)

    sibling = img_p.with_suffix(".npz")
    if sibling.is_file():
        return str(sibling)

    return None


#                                                                
#       &   
#                                                                

def build_turn1_messages(image: Image.Image, user_prompt: str) -> list:
    """Turn1    user turn   image + v3 CoT    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_prompt},
            ],
        },
    ]


def build_turn2_messages(
    turn1_messages: list,
    turn1_response: str,
    part_id: int,
    geometry_prompt_template: str,
) -> list:
    """Turn2   Turn1         assistant    +    user    """
    question = geometry_prompt_template.replace("{part_id}", str(part_id))
    return turn1_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": turn1_response}]},
        {"role": "user",      "content": [{"type": "text", "text": question}]},
    ]


#                  CoT V3 tag          
_CRITICAL_TAG_LITERALS: List[str] = [
    "<sam_feat>",
    "<think>", "</think>",
    "<overall>", "</overall>",
    *[f"<geometry_l_{k}>"  for k in range(24)],
    *[f"</geometry_l_{k}>" for k in range(24)],
]


def _build_critical_id_to_content(tokenizer) -> Dict[int, str]:
    """
       {token_id: literal_str}       decode     critical tag
               

      value             _CRITICAL_TAG_LITERALS       
      tokenizer.added_tokens_decoder   AddedToken.content       
          tokenizers/transformers              
      key   tokenizer.convert_tokens_to_ids              
    """
    unk_id = tokenizer.unk_token_id
    mapping: Dict[int, str] = {}
    for literal in _CRITICAL_TAG_LITERALS:
        tid = tokenizer.convert_tokens_to_ids(literal)
        if tid is None or tid == unk_id:
            LOGGER.warning("tokenizer     critical tag: %s      added_tokens    ", literal)
            continue
        mapping[int(tid)] = literal
    return mapping


def _collect_all_added_token_ids(tokenizer) -> set:
    """
       tokenizer     added-token ID      Qwen      token +        

      critical   added token   <|im_end|>/<|endoftext|>/<|image_pad|>  
    decode            byte-level      
    """
    ids: set = set()
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tid in decoder.keys():
        try:
            ids.add(int(tid))
        except (TypeError, ValueError):
            continue
    for tok in getattr(tokenizer, "all_special_tokens", []) or []:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                ids.add(int(tid))
        except Exception:
            continue
    return ids


def _robust_decode(tokenizer, token_ids) -> str:
    """
       decode            
        critical tag (<think> / <overall> / <geometry_l_k> / <sam_feat>):
                               tokenizer      
           added special token (<|im_end|> / <|endoftext|> / vision pad  ):
                   byte-level decoder 
           BPE token:
              buffer    tokenizer.decode(skip_special_tokens=True)
             buffer     added token byte-level       
    """
    ids: List[int] = [int(x) for x in token_ids]
    critical_map = _build_critical_id_to_content(tokenizer)
    added_ids    = _collect_all_added_token_ids(tokenizer)

    parts:  List[str] = []
    buffer: List[int] = []

    def _flush():
        if not buffer:
            return
        text = tokenizer.decode(
            buffer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        parts.append(text)
        buffer.clear()

    for tid in ids:
        if tid in critical_map:
            _flush()
            parts.append(critical_map[tid])
        elif tid in added_ids:
            _flush()
        else:
            buffer.append(tid)
    _flush()

    return "".join(parts)


@torch.inference_mode()
def _generate(
    model,
    processor: AutoProcessor,
    messages: list,
    max_new_tokens: int,
    temperature: float,
) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    target_device = _get_model_device(model)
    inputs = {
        k: v.to(target_device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
    )
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], generated_ids)
    ]
    #    decode    <think>/<overall>/<geometry_l_k>/<sam_feat>  
    # critical tag                    
    return _robust_decode(processor.tokenizer, trimmed[0])


#                                                                
#  Turn1      CoT V3 
#                                                                

_BBOX3D_PAT = re.compile(
    r"Part\s+`(l_\d+)`\s*:[^\n]*?`bbox_3d`\s*=\s*\[([^\]]+)\]",
)
_BBOX2D_PAT = re.compile(
    r"Part\s+`(l_\d+)`\s*:[^\n]*?`bbox_2d`\s*=\s*\[([^\]]+)\]",
)
_PART_COUNT_PAT = re.compile(r"core parts:\s*(\d+)", re.IGNORECASE)
_IM_END_PAT = re.compile(r"<\|im_end\|>\s*$")


def _strip_trailing_eos(text: str) -> str:
    """   generate          <|im_end|> /    """
    return _IM_END_PAT.sub("", text).strip()


def parse_turn1_output(raw: str) -> Dict[str, Optional[str]]:
    """
       Turn1    <think>...</think><overall>...</overall>

      
    -------
    {
      "think"     : <think>...</think>         tag    None
      "overall"   : <overall>...</overall>         tag    None
      "raw_clean" :      <|im_end|>       
      "raw"       :    generate   
    }
    """
    clean = _strip_trailing_eos(raw)

    think = None
    if "<think>" in clean and "</think>" in clean:
        think = clean.split("<think>", 1)[1].split("</think>", 1)[0].strip()

    overall = None
    if "<overall>" in clean and "</overall>" in clean:
        overall = clean.split("<overall>", 1)[1].split("</overall>", 1)[0].strip()
    elif "<overall>" in clean:
        #         <overall>                   
        overall = clean.split("<overall>", 1)[1].strip()

    #      tag         bad_words_ids    /         tag /
    # generate   max_new_tokens                       
    #   Step 1 ... part_count ...   Step 2 Part l_0/l_1 ...   Step 3 inter-part
    #     Step 4 primitive/surface    tag      overall Parts: l_0 ... l_k
    #        Parts: > Overall > Object category         think 
    #      tag              
    if think is None and overall is None and clean:
        overall_anchors = ["\nParts:", "\nparts:", "Parts:", "parts:",
                           "\nObject category", "\nobject_category",
                           "\nOverall", "\noverall"]
        split_idx = -1
        for anchor in overall_anchors:
            p = clean.find(anchor)
            if p > 0:
                split_idx = p
                break
        if split_idx > 0:
            think   = clean[:split_idx].strip() or None
            overall = clean[split_idx:].lstrip("\n").strip() or None
    else:
            think = clean.strip() or None

    return {
        "think":     think,
        "overall":   overall,
        "raw_clean": clean,
        "raw":       raw,
    }


def parse_bbox_2d_from_think(think_text: Optional[str]) -> Dict[str, List[float]]:
    """
      <think> Step 2       bbox_2d = [x_min, x_max, y_min, y_max]     0~1 
         SAM3       [cx, cy, w, h]    

       {'l_0': [cx, cy, w, h], ...}   bbox           
    """
    if not think_text:
        return {}
    out: Dict[str, List[float]] = {}
    for m in _BBOX2D_PAT.finditer(think_text):
        pid = m.group(1)
        try:
            nums = [float(v.strip()) for v in m.group(2).split(",")]
        except ValueError:
            continue
        if len(nums) != 4:
            continue
        x_min, x_max, y_min, y_max = nums
        #          /         [0,1]   
        w, h = x_max - x_min, y_max - y_min
        if w <= 0 or h <= 0 or not (0 <= x_min <= 1 and 0 <= y_max <= 1):
            continue
        out[pid] = [
            (x_min + x_max) / 2,   # cx
            (y_min + y_max) / 2,   # cy
            w,
            h,
        ]
    return out


def parse_bbox_3d_from_think(think_text: Optional[str]) -> Dict[str, List[int]]:
    """
      <think> Step 2       bbox_3d 
       {"l_0": [x_min,x_max,y_min,y_max,z_min,z_max], ...}
    """
    if not think_text:
        return {}
    out: Dict[str, List[int]] = {}
    for m in _BBOX3D_PAT.finditer(think_text):
        pid = m.group(1)
        try:
            nums = [int(round(float(v.strip()))) for v in m.group(2).split(",")]
        except ValueError:
            continue
        if len(nums) == 6:
            out[pid] = nums
    return out


def parse_part_count(
    think_text: Optional[str],
    overall_text: Optional[str],
    bbox_3d_map: Dict[str, List[int]],
) -> int:
    """
                             Step 1/2   overall Parts       
      1. Step 1    "core parts: N"
      2. bbox_3d_map    l_k      + 1
      3. <overall>        l_0, l_1, ...
    """
    candidates = []

    if think_text:
        m = _PART_COUNT_PAT.search(think_text)
        if m:
            candidates.append(int(m.group(1)))

    if bbox_3d_map:
        ids = [int(k.split("_")[1]) for k in bbox_3d_map.keys()]
        candidates.append(max(ids) + 1)

    if overall_text:
        count = 0
        while f"l_{count}" in overall_text:
            count += 1
        if count > 0:
            candidates.append(count)

    return max(candidates) if candidates else 0


#                                                                
#  Turn2         RLE + bbox_3d       xyz 
#                                                                

def extract_geometry_payload(raw: str, part_id: int) -> str:
    """
      Turn2       <geometry_l_{part_id}>...</geometry_l_{part_id}>     RLE    
      wrapper            tag          EOS      
    """
    clean = _strip_trailing_eos(raw)
    open_tag = f"<geometry_l_{part_id}>"
    close_tag = f"</geometry_l_{part_id}>"

    if open_tag in clean:
        inner = clean.split(open_tag, 1)[1]
        if close_tag in inner:
            inner = inner.split(close_tag, 1)[0]
        return inner.strip()
    return clean


def _rle_to_local_ids(s: str) -> np.ndarray:
    """'0 1-5 36-41'   np.int64    id          """
    if not s.strip():
        return np.array([], dtype=np.int64)
    out = []
    for tok in s.split():
        if "-" in tok:
            try:
                a, b = map(int, tok.split("-", 1))
            except ValueError:
                continue
            if a > b:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            try:
                out.append(int(tok))
            except ValueError:
                continue
    if not out:
        return np.array([], dtype=np.int64)
    return np.unique(np.array(out, dtype=np.int64))


def local_ids_to_global_xyz(
    local_ids: np.ndarray,
    bbox_3d: List[int],
    grid_size: int = 32,
) -> np.ndarray:
    """
    bbox_3d = [x_min, x_max, y_min, y_max, z_min, z_max]
    local_id = (x-x_min)*(dy*dz) + (y-y_min)*dz + (z-z_min)
       [N, 3]    xyz clip   [0, grid_size-1]         
    """
    if local_ids.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    x_min, x_max, y_min, y_max, z_min, z_max = bbox_3d
    dy = max(1, y_max - y_min + 1)
    dz = max(1, z_max - z_min + 1)
    dyz = dy * dz
    capacity = (max(1, x_max - x_min + 1)) * dyz

    #          local_id     bbox     
    ids = np.clip(local_ids, 0, capacity - 1)

    x = x_min + ids // dyz
    rem = ids % dyz
    y = y_min + rem // dz
    z = z_min + rem % dz
    pts = np.stack([x, y, z], axis=1)
    return np.clip(pts, 0, grid_size - 1).astype(np.int64)


#                                                                
#  Turn2           geometry 
#                                                                

@torch.inference_mode()
def infer_geometry(
    model,
    processor: AutoProcessor,
    turn1_messages: list,
    turn1_response: str,
    n_parts: int,
    bbox_3d_map: Dict[str, List[int]],
    geometry_prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    save_dir: str,
    save_ply: bool,
    grid_size: int = 32,
) -> List[np.ndarray]:
    """          RLE       xyz     .npy / .ply """
    all_voxels: List[np.ndarray] = []

    for part_id in range(n_parts):
        pid_key = f"l_{part_id}"
        bbox = bbox_3d_map.get(pid_key)
        if bbox is None:
            LOGGER.warning(
                "  [%s]    bbox_3d         [0..%d]      ",
                pid_key, grid_size - 1,
            )
            bbox = [0, grid_size - 1] * 3

        LOGGER.info("     %s / l_%d bbox_3d=%s", pid_key, n_parts - 1, bbox)

        messages = build_turn2_messages(
            turn1_messages, turn1_response, part_id, geometry_prompt_template,
        )
        raw = _generate(
            model, processor, messages,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        rle_text = extract_geometry_payload(raw, part_id)
        LOGGER.info("  [%s] RLE   120   : %s", pid_key, rle_text[:120])

        local_ids = _rle_to_local_ids(rle_text)
        voxels    = local_ids_to_global_xyz(local_ids, bbox, grid_size=grid_size)
        all_voxels.append(voxels)

        #       RLE        xyz    PLY
        Path(os.path.join(save_dir, f"coord_{part_id}.txt")).write_text(
            rle_text, encoding="utf-8",
        )
        np.save(os.path.join(save_dir, f"ind_{part_id}.npy"), voxels)

        if save_ply and len(voxels) > 0:
            try:
                import trimesh
                pc = trimesh.points.PointCloud(voxels)
                pc.export(os.path.join(save_dir, f"ind_{part_id}.ply"))
            except ImportError:
                LOGGER.warning("trimesh        .ply    ")

    if all_voxels:
        concat = np.concatenate(all_voxels, axis=0) if all_voxels else np.zeros((0, 3), dtype=np.int64)
        np.save(os.path.join(save_dir, "allind.npy"), concat)
        LOGGER.info(
            "        : %d   ,    %d   ",
            n_parts, len(concat),
        )

    return all_voxels


#                                                                
#      
#                                                                

def save_turn1_result(parsed: Dict, save_dir: str, stem: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    Path(os.path.join(save_dir, f"{stem}_cot.txt")).write_text(
        parsed["raw_clean"], encoding="utf-8",
    )
    if parsed["think"] is not None:
        Path(os.path.join(save_dir, f"{stem}_think.txt")).write_text(
            parsed["think"], encoding="utf-8",
        )
    if parsed["overall"] is not None:
        Path(os.path.join(save_dir, f"{stem}_overall.txt")).write_text(
            parsed["overall"], encoding="utf-8",
        )


def print_turn1_parsed(parsed: Dict, bbox_3d_map: Dict, image_name: str, n_parts: int) -> None:
    sep = " " * 70
    LOGGER.info("\n%s\n  : %s   |        : %d\n%s",
                sep, image_name, n_parts, sep)
    if parsed["think"]:
        LOGGER.info("[<think>]\n%s", parsed["think"])
    else:
        LOGGER.warning("[<think>]      ")
    if parsed["overall"]:
        LOGGER.info("[<overall>]\n%s", parsed["overall"])
    else:
        LOGGER.warning("[<overall>]      ")
    if bbox_3d_map:
        LOGGER.info("[bbox_3d   ] %s", bbox_3d_map)
    else:
        LOGGER.warning("[bbox_3d]          bbox_3d Turn2           ")


#                                                                
#     /       
#                                                                

def _acquire_sam_feats(
    image:                Image.Image,
    image_path:           str,
    sam_injector:         SAMInferenceInjector,
    sam_feature_explicit: Optional[str],
    sam_feature_dir:      Optional[str],
    sam_extractor:        Optional[SAM3FeatureExtractor],
    model,
    processor:            AutoProcessor,
    turn1_prompt:         str,
    max_new_tokens_pass1: int,
    temperature:          float,
    save_dir:             str,
    stem:                 str,
    save_sam_feature:     bool,
) -> Tuple[bool, Dict]:
    """
            SAM       sam_injector 
      1.   /       npz load_sam_feats 
      2. Two-pass      
           Pass 1   SAM   Turn1      bbox_2d
             SAM3      [N_parts, 256]
                   {stem}_sam.npz      
      3.          SAM fallback      embedding 

    Returns
    -------
    (sam_ready, pass1_cache)
      sam_ready  :        SAM   
      pass1_cache:     Two-pass    {'raw', 'parsed', 'bbox_2d_map'}    
    """
    #             
    sam_injector.clear()

    #       1     npz                                    
    sam_path = resolve_sam_feature_path(
        image_path=image_path,
        explicit=sam_feature_explicit,
        sam_feature_dir=sam_feature_dir,
    )
    if sam_path:
        try:
            feats = load_sam_feats(sam_path)
            sam_injector.set_feats(feats)
            LOGGER.info(
                "SAM           : %s ([%d, %d])",
                sam_path, feats.size(0), feats.size(1),
            )
            return True, {}
        except Exception as exc:
            LOGGER.warning("SAM        (%s): %s", sam_path, exc)

    #       2 Two-pass                                  
    if sam_extractor is None:
        LOGGER.info(
            "       SAM      --auto_extract_sam          SAM   ",
        )
        return False, {}

    LOGGER.info("   Two-pass Pass 1   SAM       bbox_2d ...")
    turn1_messages = build_turn1_messages(image, turn1_prompt)
    raw_pass1 = _generate(
        model, processor, turn1_messages,
        max_new_tokens=max_new_tokens_pass1, temperature=temperature,
    )
    parsed_pass1 = parse_turn1_output(raw_pass1)
    bbox_2d_map  = parse_bbox_2d_from_think(parsed_pass1["think"])

    if not bbox_2d_map:
        LOGGER.warning("Pass 1        bbox_2d        SAM         SAM   ")
        return False, {"raw": raw_pass1, "parsed": parsed_pass1, "bbox_2d_map": {}}

    LOGGER.info(
        "Pass 1      %d     bbox_2d    SAM3    RoI    ...",
        len(bbox_2d_map),
    )
    try:
        feats_dict = sam_extractor.extract(image, bbox_2d_map)
    except Exception as exc:
        LOGGER.warning("SAM3       : %s      SAM   ", exc)
        return False, {"raw": raw_pass1, "parsed": parsed_pass1, "bbox_2d_map": bbox_2d_map}

    if not feats_dict:
        LOGGER.warning("SAM3              SAM   ")
        return False, {"raw": raw_pass1, "parsed": parsed_pass1, "bbox_2d_map": bbox_2d_map}

    part_keys = sorted(feats_dict.keys(), key=lambda k: int(k.split("_")[1]))
    feats = torch.stack(
        [torch.from_numpy(feats_dict[k]).float() for k in part_keys], dim=0,
    )
    sam_injector.set_feats(feats)
    LOGGER.info(
        "SAM            : parts=%s, shape=[%d, %d]",
        part_keys, feats.size(0), feats.size(1),
    )

    if save_sam_feature:
        sam_out_path = os.path.join(save_dir, f"{stem}_sam.npz")
        np.savez_compressed(sam_out_path, **feats_dict)
        LOGGER.info("SAM      : %s         --sam_feature      ", sam_out_path)

    return True, {
        "raw":         raw_pass1,
        "parsed":      parsed_pass1,
        "bbox_2d_map": bbox_2d_map,
    }


def infer_single(
    model,
    processor: AutoProcessor,
    sam_injector: SAMInferenceInjector,
    image_path: str,
    output_dir: str,
    turn1_prompt: str,
    turn2_prompt_template: str,
    max_new_tokens_turn1: int,
    max_new_tokens_turn2: int,
    max_new_tokens_pass1: int,
    temperature: float,
    remove_bg: bool,
    save_ply: bool,
    sam_feature_explicit: Optional[str],
    sam_feature_dir: Optional[str],
    sam_extractor: Optional[SAM3FeatureExtractor],
    save_sam_feature: bool,
    grid_size: int,
) -> Dict:
    #                                                        
    image = Image.open(image_path).convert("RGB").resize((512, 512), Image.LANCZOS)
    if remove_bg:
        try:
            from rembg import remove as rembg_remove
            image = rembg_remove(image)
        except ImportError:
            LOGGER.warning("rembg           ")

    stem     = Path(image_path).stem
    save_dir = os.path.join(output_dir, stem)
    os.makedirs(save_dir, exist_ok=True)

    #       SAM                                       
    sam_ready, pass1_cache = _acquire_sam_feats(
        image                = image,
        image_path           = image_path,
        sam_injector         = sam_injector,
        sam_feature_explicit = sam_feature_explicit,
        sam_feature_dir      = sam_feature_dir,
        sam_extractor        = sam_extractor,
        model                = model,
        processor            = processor,
        turn1_prompt         = turn1_prompt,
        max_new_tokens_pass1 = max_new_tokens_pass1,
        temperature          = temperature,
        save_dir             = save_dir,
        stem                 = stem,
        save_sam_feature     = save_sam_feature,
    )
    if not sam_ready:
        LOGGER.warning(
            "      SAM fallback    <sam_feat>       embedding         ",
        )

    try:
        #    Pass 2 Turn 1                                        
        turn1_messages = build_turn1_messages(image, turn1_prompt)
        raw_turn1 = _generate(
            model, processor, turn1_messages,
            max_new_tokens=max_new_tokens_turn1, temperature=temperature,
        )
        parsed = parse_turn1_output(raw_turn1)
        bbox_3d_map = parse_bbox_3d_from_think(parsed["think"])
        n_parts = parse_part_count(parsed["think"], parsed["overall"], bbox_3d_map)
        if n_parts > MAX_PARTS:
            raise ValueError(
                f"Predicted part_count={n_parts} exceeds the supported maximum "
                f"of {MAX_PARTS}."
            )

        save_turn1_result(parsed, save_dir, stem)
        print_turn1_parsed(parsed, bbox_3d_map, image_path, n_parts)

        #    Pass 2 Turn 2     geometry                       
        if n_parts == 0:
            LOGGER.warning("No parts detected; skipping local geometry generation")
            return {
                **parsed,
                "n_parts": 0,
                "bbox_3d_map": {},
                "sam_ready": sam_ready,
                "pass1_cache": pass1_cache,
            }

        LOGGER.info("Generating local geometry for %d parts", n_parts)
        infer_geometry(
            model=model,
            processor=processor,
            turn1_messages=turn1_messages,
            turn1_response=parsed["raw_clean"],
            n_parts=n_parts,
            bbox_3d_map=bbox_3d_map,
            geometry_prompt_template=turn2_prompt_template,
            max_new_tokens=max_new_tokens_turn2,
            temperature=temperature,
            save_dir=save_dir,
            save_ply=save_ply,
            grid_size=grid_size,
        )

        return {
            **parsed,
            "n_parts":     n_parts,
            "bbox_3d_map": bbox_3d_map,
            "sam_ready":   sam_ready,
            "pass1_cache": pass1_cache,
        }

    finally:
        sam_injector.clear()
        LOGGER.info("Saved inference outputs to %s", save_dir)


#                                                                
#  Prompt           
#                                                                

def _load_or_default(path: Optional[str], default: str, name: str) -> str:
    if path and os.path.isfile(path):
        content = Path(path).read_text(encoding="utf-8").strip()
        LOGGER.info("%s      : %s", name, path)
        return content
    if path:
        LOGGER.warning("%s      : %s        prompt", name, path)
    return default


#                                                                
#  CLI
#                                                                

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhysX-CoT inference with Qwen3-VL and local 3D geometry"
    )

    #                                                           
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER,
            help="Path to the PhysX-CoT LoRA adapter checkpoint.")
    parser.add_argument("--base_model",   type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--image",        type=str, default=None, help="      ")
    parser.add_argument("--image_dir",    type=str, default=None, help="          ")
    parser.add_argument("--output_dir",   type=str, default="./cot_results",
                        help="        ")

    #    SAM                                                
    parser.add_argument("--sam_feature",     type=str, default=None,
                        help="   SAM    npz          ")
    parser.add_argument("--sam_feature_dir", type=str, default=DEFAULT_SAM_FEATURE_DIR,
                        help="SAM         obj_id/img_id.npz        ")
    parser.add_argument("--sam_proj_hidden", type=int, default=512,
                        help="SAMProjector                 512 ")

    #    SAM         Two-pass                            
    parser.add_argument("--auto_extract_sam", dest="auto_extract_sam",
                        action="store_true", default=True,
                        help="    npz        Two-pass      SAM         ")
    parser.add_argument("--no_auto_extract_sam", dest="auto_extract_sam",
                        action="store_false",
                        help="   Two-pass     npz       SAM fallback")
    parser.add_argument("--sam3_root",        type=str, default=DEFAULT_SAM3_ROOT,
                        help="sam3-main         catch_sam_feature.py    ")
    parser.add_argument("--sam3_checkpoint",  type=str, default=DEFAULT_SAM3_CHECKPOINT,
                        help="SAM3       ")
    parser.add_argument("--sam3_device",      type=str, default=None,
                        help="SAM3            --device       ~3-5 GB    ")
    parser.add_argument("--sam3_roi_size",    type=int, default=3,
                        help="RoI Align        catch_sam_feature.py      ")
    parser.add_argument("--max_new_tokens_pass1", type=int, default=2048,
                        help="Two-pass   Pass 1   max_new_tokens      Step1+Step2    2048 ")
    parser.add_argument("--save_sam_feature", dest="save_sam_feature",
                        action="store_true", default=True,
                        help="      SAM       {output_dir}/{stem}/{stem}_sam.npz      ")
    parser.add_argument("--no_save_sam_feature", dest="save_sam_feature",
                        action="store_false",
                        help="   SAM       ")

    #                                                         
    parser.add_argument("--merge_weights", action="store_true", default=False,
                        help="      LoRA                adapter ")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="   device    device_map='auto' CPU offload        ")

    #                                                         
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="            turn1/turn2 ")
    parser.add_argument("--max_new_tokens_turn1", type=int, default=None,
                        help="Turn1 CoT+overall      token         --max_new_tokens")
    parser.add_argument("--max_new_tokens_turn2", type=int, default=None,
                        help="Turn2     geometry      token         --max_new_tokens")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0=     >0     ")
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--grid_size",  type=int, default=32,
                        help="Voxel          32   canonical 3D        ")

    #                                                        
    parser.add_argument("--remove_bg", action="store_true", default=False)
    parser.add_argument("--save_ply",  action="store_true", default=False,
                        help="         .ply       trimesh ")

    #    Legacy checkpoint                                    
    #     checkpoint   embed_tokens/lm_head     
    #  LoRA   modules_to_save   legacy run      45   special token
    #   lm_head          argmax          id       
    #   1. generate      bad_words_ids     45   id   logit 
    #   2. parse_turn1_output     <think>/<overall>           
    #      bbox_2d / bbox_3d / primitive        tag        
    #        modules_to_save             legacy checkpoint 
    #         the current CoT runner 
    parser.add_argument("--skip_new_special_tokens", action="store_true", default=False,
                        help="       special token     bad_words_ids  "
                             "     embed_tokens/lm_head     modules_to_save "
                             "  legacy checkpoint          False ")

    #    Prompt              v3                    
    parser.add_argument("--prompt_file",     type=str, default=None,
                        help="   Turn1 prompt                     v3 prompt")
    parser.add_argument("--geo_prompt_file", type=str, default=None,
                        help="   Turn2 prompt            {part_id}     ")

    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("    --image   --image_dir")

    turn1_prompt = _load_or_default(args.prompt_file,     _DEFAULT_TURN1_PROMPT, "Turn1 prompt")
    turn2_prompt = _load_or_default(args.geo_prompt_file, _DEFAULT_TURN2_PROMPT, "Turn2 prompt")

    max_new_tokens_turn1 = args.max_new_tokens_turn1 or args.max_new_tokens
    max_new_tokens_turn2 = args.max_new_tokens_turn2 or args.max_new_tokens

    model, processor, sam_injector, hook_handle = load_model_and_processor(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        merge_weights=args.merge_weights,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        device=args.device,
        sam_proj_hidden=args.sam_proj_hidden,
    )

    #                special token                      
    # bad_words_ids     generate   logit processor          
    #    id input_ids    <sam_feat>       SAM hook         
    if args.skip_new_special_tokens:
        banned_ids: List[int] = []
        unk_id = processor.tokenizer.unk_token_id
        for literal in _CRITICAL_TAG_LITERALS:
            tid = processor.tokenizer.convert_tokens_to_ids(literal)
            if tid is None or tid == unk_id:
                continue
            banned_ids.append(int(tid))
        if banned_ids:
            model.generation_config.bad_words_ids = [[tid] for tid in banned_ids]
            LOGGER.warning(
                "    %d         special token id      :   10   = %s",
                len(banned_ids), banned_ids[:10],
            )
        else:
            LOGGER.warning("--skip_new_special_tokens                id")

    #    SAM3                                        
    sam_extractor: Optional[SAM3FeatureExtractor] = None
    if args.auto_extract_sam:
        sam3_device = args.sam3_device or args.device
        sam_extractor = SAM3FeatureExtractor(
            sam3_root       = args.sam3_root,
            checkpoint      = args.sam3_checkpoint,
            device          = sam3_device,
            roi_output_size = args.sam3_roi_size,
        )
        LOGGER.info(
            "Two-pass SAM         (sam3_root=%s, device=%s)",
            args.sam3_root, sam3_device,
        )
    else:
        LOGGER.info("Two-pass SAM         --no_auto_extract_sam ")

    try:
        image_paths: List[str] = []
        if args.image:
            image_paths.append(args.image)
        if args.image_dir:
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            image_paths.extend(
                str(p) for p in sorted(Path(args.image_dir).iterdir())
                if p.suffix.lower() in exts
            )

        LOGGER.info("Found %d input image(s)", len(image_paths))
        for img_path in image_paths:
            infer_single(
                model=model,
                processor=processor,
                sam_injector=sam_injector,
                image_path=img_path,
                output_dir=args.output_dir,
                turn1_prompt=turn1_prompt,
                turn2_prompt_template=turn2_prompt,
                max_new_tokens_turn1=max_new_tokens_turn1,
                max_new_tokens_turn2=max_new_tokens_turn2,
                max_new_tokens_pass1=args.max_new_tokens_pass1,
                temperature=args.temperature,
                remove_bg=args.remove_bg,
                save_ply=args.save_ply,
                sam_feature_explicit=args.sam_feature,
                sam_feature_dir=args.sam_feature_dir,
                sam_extractor=sam_extractor,
                save_sam_feature=args.save_sam_feature,
                grid_size=args.grid_size,
            )
    finally:
        hook_handle.remove()


if __name__ == "__main__":
    main()
