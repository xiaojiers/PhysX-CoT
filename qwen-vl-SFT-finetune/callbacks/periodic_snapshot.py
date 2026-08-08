"""
callbacks/periodic_snapshot.py

HuggingFace Trainer   save_total_limit / load_best_model_at_end    
  "   "   checkpoint              

  callback    N   periodic_save_steps     checkpoint  
**   **      {output_dir}/periodic_snapshots/checkpoint-XXX 
   
  -              inode Trainer                
  -      save_total_limit     snapshot     
  -                         

       on_save         CompletionCallback         
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class PeriodicSnapshotCallback(TrainerCallback):
    """  every_n_steps    checkpoint      periodic_snapshots/ """

    def __init__(self, every_n_steps: int):
        if every_n_steps <= 0:
            raise ValueError("every_n_steps    > 0")
        self.every_n_steps = every_n_steps

    @staticmethod
    def _hardlink_tree(src: Path, dst: Path) -> None:
        """            mkdir     os.link  """
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                PeriodicSnapshotCallback._hardlink_tree(item, target)
            elif target.exists():
                continue
            else:
                try:
                    os.link(item, target)
                except OSError:
                    #     /            copy2    mtime 
                    shutil.copy2(item, target)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        step = state.global_step
        if step <= 0 or step % self.every_n_steps != 0:
            return control

        src = Path(args.output_dir) / f"checkpoint-{step}"
        if not src.is_dir():
            logger.warning("[Snapshot]   checkpoint       : %s", src)
            return control

        dst = Path(args.output_dir) / "periodic_snapshots" / f"checkpoint-{step}"
        if dst.exists():
            logger.info("[Snapshot]         : %s", dst)
            return control

        try:
            self._hardlink_tree(src, dst)
            logger.info("[Snapshot]     step=%d   %s", step, dst)
        except Exception as exc:
            logger.warning("[Snapshot]      step=%d: %s", step, exc)

        return control
