"""
callbacks/checkpoint_completion.py

Trainer    _save_checkpoint    LoRA adapter   optimizer state 
    processor / tokenizer / Vision Merger / SAM Projector 

  callback     checkpoint       rank 0    
  - processor.save_pretrained(ckpt)     config.json / tokenizer / preprocessor
  - merger_weights.pt                   Vision Merger     
  - sam_projector.pt                    SAMProjector state_dict

         1_vlm_cot.py /          checkpoint-XXX    
              
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class CheckpointCompletionCallback(TrainerCallback):
    """    checkpoint       processor + merger + sam_projector """

    def __init__(
        self,
        processor,
        sam_projector: nn.Module,
        visual_merger: Optional[nn.Module],
    ):
        self.processor     = processor
        self.sam_projector = sam_projector
        self.visual_merger = visual_merger

    def _checkpoint_dir(self, args, state) -> Optional[Path]:
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        return ckpt if ckpt.is_dir() else None

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        ckpt_dir = self._checkpoint_dir(args, state)
        if ckpt_dir is None:
            return control

        # processor / tokenizer / preprocessor_config
        try:
            self.processor.save_pretrained(str(ckpt_dir))
        except Exception as exc:
            logger.warning("[Completion] processor.save_pretrained   : %s", exc)

        # Vision Merger                 LoRA adapter   
        if self.visual_merger is not None:
            try:
                merger_state = {
                    n: p.detach().cpu()
                    for n, p in self.visual_merger.named_parameters()
                }
                torch.save(merger_state, ckpt_dir / "merger_weights.pt")
            except Exception as exc:
                logger.warning("[Completion] merger_weights.pt     : %s", exc)

        # SAM Projector    256   hidden   D_llm    MLP 
        try:
            projector_state = {
                k: v.detach().cpu()
                for k, v in self.sam_projector.state_dict().items()
            }
            torch.save(projector_state, ckpt_dir / "sam_projector.pt")
        except Exception as exc:
            logger.warning("[Completion] sam_projector.pt     : %s", exc)

        logger.info("[Completion] checkpoint    : %s", ckpt_dir)
        return control
