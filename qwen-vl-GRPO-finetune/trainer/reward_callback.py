"""
trainer/reward_callback.py

GRPO Reward         callback 

   
        logging_steps    on_log     GRPORewardWrapper      
       main / sub / info             trainer       logs dict 
                              
              /        health metrics      

   
        on_log       on_step_end    trainer logging      
           
      wrapper                      reward/*    
       wandb / tensorboard    NaN 
       key      
        reward/total
        reward/main/R_loc, R_coarse, R_detail, R_phys, P, R_main
        reward/sub/R_count, R_bbox2d, ..., D_local, D_global, ...
        reward/info/has_overall_ratio, pred_part_count_avg, ...
        reward/health/window_size, sam_cache_hit_ratio
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from transformers import TrainerCallback

LOGGER = logging.getLogger(__name__)


class RewardComponentLoggingCallback(TrainerCallback):
    """  GRPORewardWrapper         trainer   logs """

    def __init__(
        self,
        reward_wrapper,
        sam_loader: Optional[object] = None,
        log_main: bool = True,
        log_sub: bool = True,
        log_info: bool = True,
        reset_after_log: bool = True,
    ):
        self.wrapper          = reward_wrapper
        self.sam_loader       = sam_loader
        self.log_main         = bool(log_main)
        self.log_sub          = bool(log_sub)
        self.log_info         = bool(log_info)
        self.reset_after_log  = bool(reset_after_log)

    def on_log(self, args, state, control, logs: Optional[Dict[str, Any]] = None, **kwargs):
        if logs is None:
            return control

        wrapper = self.wrapper
        if wrapper is None:
            return control

        #         main / sub / info   
        if self.log_main:
            for key, val in wrapper.aggregate_main(source="window").items():
                if key == "count":
                    logs["reward/health/window_size"] = float(val)
                elif key == "total":
                    logs["reward/total"] = float(val)
                else:
                    logs[f"reward/main/{key}"] = float(val)

        if self.log_sub:
            for key, val in wrapper.aggregate_sub(source="window").items():
                logs[f"reward/sub/{key}"] = float(val)

        if self.log_info:
            for key, val in wrapper.aggregate_info(source="window").items():
                # info/*     
                logs[f"reward/{key}"] = float(val)

        #    SAM        
        if self.sam_loader is not None and hasattr(self.sam_loader, "cache_stats"):
            stats = self.sam_loader.cache_stats()
            logs["reward/health/sam_cache_size"]      = float(stats.get("size", 0))
            logs["reward/health/sam_cache_hit_ratio"] = float(stats.get("hit_ratio", 0.0))

        if self.reset_after_log:
            wrapper.reset_window()

        return control
