"""
callbacks/checkpoint_eval.py

    checkpoint              +        

     
  -       subprocess.Popen   wait()          
  - GPU       --eval_device    cpu / cuda:<idx>  worker   
            CUDA_VISIBLE_DEVICES       rank     
  -   rank-0    DDP                
  -                         zombie 
  - SAM    skip_auto_sam=True     SAM            SAM
                
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

# worker      ../scripts/checkpoint_eval_worker.py 
_WORKER = str(Path(__file__).parent.parent / "scripts" / "checkpoint_eval_worker.py")


class CheckpointEvalCallback(TrainerCallback):
    """   Trainer    checkpoint                 """

    def __init__(
        self,
        eval_image: str,
        base_model: str,
        physx_root: str,
        eval_out_root: str,
        eval_device: str = "cpu",
        eval_max_new_tokens: int = 2048,
        skip_auto_sam: bool = True,
    ):
        self.eval_image         = eval_image
        self.base_model         = base_model
        self.physx_root         = physx_root
        self.eval_out_root      = eval_out_root
        self.eval_device        = eval_device
        self.max_new_tokens     = eval_max_new_tokens
        self.skip_auto_sam      = skip_auto_sam
        self._procs: List[subprocess.Popen] = []

    def _reap_finished(self) -> None:
        """             zombie """
        alive: List[subprocess.Popen] = []
        for p in self._procs:
            ret = p.poll()
            if ret is None:
                alive.append(p)
            else:
                logger.info("[CheckpointEval] pid=%d        =%d", p.pid, ret)
        self._procs = alive

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_world_process_zero:
            return control

        self._reap_finished()

        #      CPU        > save               
        #           eval worker                     
        if self._procs:
            alive_pids = [p.pid for p in self._procs]
            logger.info(
                "[CheckpointEval] step=%d           %d   worker     (pids=%s)",
                state.global_step, len(self._procs), alive_pids,
            )
            return control

        step            = state.global_step
        checkpoint_dir  = os.path.join(args.output_dir, f"checkpoint-{step}")
        eval_out        = os.path.join(self.eval_out_root, f"checkpoint-{step}")

        # trainer_state.json      checkpoint     Trainer      
        state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if not os.path.isfile(state_path):
            state_path = os.path.join(args.output_dir, "trainer_state.json")

        os.makedirs(eval_out, exist_ok=True)

        cmd: List[str] = [
            sys.executable, _WORKER,
            "--checkpoint",      checkpoint_dir,
            "--image",           self.eval_image,
            "--base_model",      self.base_model,
            "--physx_root",      self.physx_root,
            "--output_dir",      eval_out,
            "--device",          self.eval_device,
            "--max_new_tokens",  str(self.max_new_tokens),
            "--state_path",      state_path,
        ]
        if self.skip_auto_sam:
            cmd.append("--skip_auto_sam")

        log_file = os.path.join(eval_out, "eval.log")
        fout = open(log_file, "w")
        proc = subprocess.Popen(cmd, stdout=fout, stderr=subprocess.STDOUT)
        self._procs.append(proc)

        logger.info(
            "[CheckpointEval] step=%d         (pid=%d)   : %s",
            step, proc.pid, log_file,
        )
        return control
