"""
rewards/reward_engine.py

        md/reward.md       

    R_loc'  = R_loc -  _cons   L_cons
    R       =  _loc   R_loc'
            +  _coarse   R_coarse
            +  _detail   R_detail
            +  _phys     R_phys
            -   _pen     P

     
  1.               completion    geometry_l_k   R_detail = None  
        "        "                  reward 
           GRPO advantage        
  2.    reward / penalty        reward_funcs.py       
       a)      b) GT enrich   c)       d)        e)   /breakdown   
  3. RewardBreakdown                            
  4. GRPORewardWrapper   RewardEngine     trl.GRPOTrainer    
     `reward_fn(completions, **kwargs)   List[float]`    

GT         build_grpo_dataset.py  
    {
      "part_count":   int,
      "think_steps":  { ...   CompletionParser.parse_think_steps    ... },
      "overall_dict": { ...   CompletionParser.parse_overall    ... },
      "voxel_dir":    "tmp/partseg/<obj_id>/32"      dataset_root 
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from configs import RewardConfig, TaskConfig
from parsers import CompletionParser

from .reward_funcs import (
    consistency_penalty_for_loc,
    penalty_dependency,
    penalty_format,
    penalty_overflow,
    reward_bbox2d,
    reward_bbox3d,
    reward_coarse_primitive,
    reward_count,
    reward_detail,
    reward_phys_group,
    reward_phys_logic,
    reward_phys_overall_format,
)
from .voxel_utils import VoxelGTLoader

LOGGER = logging.getLogger(__name__)


#                                                                
# RewardBreakdown
#                                                                

@dataclass
class RewardBreakdown:
    """                     """
    total: float
    main: Dict[str, Optional[float]] = field(default_factory=dict)        # R_loc / R_coarse / R_detail / R_phys / P
    sub:  Dict[str, Optional[float]] = field(default_factory=dict)        #       
    info: Dict[str, Any]              = field(default_factory=dict)        #     part_count_pred   


#                                                                
# RewardEngine
#                                                                

class RewardEngine:
    """
               

       
        engine = RewardEngine(reward_cfg, parser, task_cfg, voxel_loader)
        bd     = engine.compute(completion_text, gt, meta)
        score  = bd.total
    """

    def __init__(
        self,
        reward_cfg: RewardConfig,
        parser:     CompletionParser,
        task_cfg:   TaskConfig,
        voxel_loader: Optional[VoxelGTLoader] = None,
    ):
        self.cfg          = reward_cfg
        self.parser       = parser
        self.task_cfg     = task_cfg
        self.voxel_loader = voxel_loader or VoxelGTLoader(
            dataset_root=None, cache_size=reward_cfg.voxel_cache_size,
        )

    #                                                                        

    def compute(
        self,
        completion_text: str,
        gt:              Dict[str, Any],
        meta:            Dict[str, Any],
    ) -> RewardBreakdown:
        parsed = self.parser.parse(completion_text)
        gt     = self._enrich_gt(gt)

        #          None     GT                      
        r_loc,    sub_loc    = self._compute_loc(parsed, gt)
        r_coarse, sub_coarse = self._compute_coarse(parsed, gt)
        r_detail, sub_detail = self._compute_detail(parsed, gt)
        r_phys,   sub_phys   = self._compute_phys(parsed, gt)
        penalty,  sub_pen    = self._compute_penalty(parsed)

        #                                                      
        main_weighted: List[Tuple[float, float]] = []   # (R_main,  _main)
        if r_loc    is not None: main_weighted.append((r_loc,    self.cfg.loc_weight))
        if r_coarse is not None: main_weighted.append((r_coarse, self.cfg.coarse_weight))
        if r_detail is not None: main_weighted.append((r_detail, self.cfg.detail_weight))
        if r_phys   is not None: main_weighted.append((r_phys,   self.cfg.phys_weight))

        if main_weighted:
            sum_w  = sum(w for _, w in main_weighted)
            r_main = sum(s * w for s, w in main_weighted) / sum_w if sum_w > 0 else 0.0
        else:
            r_main = 0.0

        total = r_main - self.cfg.pen_weight * penalty
        total = max(self.cfg.final_clamp_min, min(self.cfg.final_clamp_max, total))

        bd = RewardBreakdown(
            total = float(total),
            main  = {
                "R_loc":    r_loc,
                "R_coarse": r_coarse,
                "R_detail": r_detail,
                "R_phys":   r_phys,
                "P":        penalty,
                "R_main":   r_main,
            },
            sub  = {**sub_loc, **sub_coarse, **sub_detail, **sub_phys, **sub_pen},
            info = {
                "pred_part_count": parsed.get("think_steps", {}).get("part_count"),
                "gt_part_count":   gt.get("part_count"),
                "n_geometries":    len(parsed.get("geometries") or {}),
                "has_overall":     parsed.get("overall_dict") is not None,
            },
        )
        if self.cfg.enable_component_logging:
            LOGGER.debug("Reward | total=%.4f | main=%s | sub=%s",
                         bd.total, bd.main, bd.sub)
        return bd

    #                                                                
    # GT         raw text   gt 
    #                                                                

    def _enrich_gt(self, gt: Dict[str, Any]) -> Dict[str, Any]:
        if "think_steps" in gt and "overall_dict" in gt:
            return gt
        raw = gt.get("text", "")
        if not raw:
            return gt
        out = dict(gt)
        parsed = self.parser.parse(raw)
        out.setdefault("think_steps", parsed.get("think_steps", {}))
        out.setdefault("overall_dict", parsed.get("overall_dict"))
        return out

    #                                                                
    # R_loc   
    #                                                                

    def _compute_loc(
        self, parsed: Dict[str, Any], gt: Dict[str, Any],
    ) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        steps    = parsed.get("think_steps", {})
        gt_steps = gt.get("think_steps") or {}

        sub: Dict[str, Optional[float]] = {}

        sub["R_count"]  = reward_count(
            pred_n      = steps.get("part_count"),
            gt_n        = gt.get("part_count") or len(gt_steps.get("bbox_2d", {})),
            decay_gamma = self.cfg.count_decay_gamma,
        )
        sub["R_bbox2d"] = reward_bbox2d(
            pred_dict = steps.get("bbox_2d") or {},
            gt_dict   = gt_steps.get("bbox_2d") or {},
        )
        sub["R_bbox3d"] = reward_bbox3d(
            pred_dict = steps.get("bbox_3d") or {},
            gt_dict   = gt_steps.get("bbox_3d") or {},
            iou_beta  = self.cfg.bbox3d_iou_beta,
            l1_eta    = self.cfg.bbox3d_l1_eta,
            grid_size = self.cfg.voxel_grid_size,
        )

        weighted: List[Tuple[float, float]] = []
        if sub["R_count"]  is not None: weighted.append((sub["R_count"],  self.cfg.loc_count_weight))
        if sub["R_bbox2d"] is not None: weighted.append((sub["R_bbox2d"], self.cfg.loc_bbox2d_weight))
        if sub["R_bbox3d"] is not None: weighted.append((sub["R_bbox3d"], self.cfg.loc_bbox3d_weight))

        if not weighted:
            sub["L_cons"] = 0.0
            return None, sub

        sum_w   = sum(w for _, w in weighted)
        r_loc   = sum(s * w for s, w in weighted) / sum_w if sum_w > 0 else 0.0

        # 2D-3D         R_loc  
        l_cons = consistency_penalty_for_loc(parsed, self.cfg) if self.cfg.cons_enable else 0.0
        sub["L_cons"] = l_cons
        r_loc_final  = r_loc - self.cfg.cons_weight * l_cons
        return max(0.0, min(1.0, r_loc_final)), sub

    #                                                                
    # R_coarse    
    #                                                                

    def _compute_coarse(
        self, parsed: Dict[str, Any], gt: Dict[str, Any],
    ) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        steps    = parsed.get("think_steps", {})
        gt_steps = gt.get("think_steps") or {}

        score = reward_coarse_primitive(
            pred_dict = steps.get("primitive_shape") or {},
            gt_dict   = gt_steps.get("primitive_shape") or {},
            shape_w   = self.cfg.coarse_shape_weight,
            axis_w    = self.cfg.coarse_axis_weight,
            ratio_w   = self.cfg.coarse_ratio_weight,
        )
        return score, {"R_coarse_primitive": score}

    #                                                                
    # R_detail         
    #                                                                

    def _compute_detail(
        self, parsed: Dict[str, Any], gt: Dict[str, Any],
    ) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        geometries = parsed.get("geometries") or {}
        if not geometries:
            return None, {
                "R_detail_parse":  None,
                "R_detail_range":  None,
                "R_detail_local":  None,
                "R_detail_global": None,
            }

        steps    = parsed.get("think_steps", {})
        gt_steps = gt.get("think_steps") or {}
        gt_voxels = self._load_gt_voxels(gt)

        result = reward_detail(
            geometries    = geometries,
            pred_bbox_3d  = steps.get("bbox_3d") or {},
            gt_voxels     = gt_voxels,
            gt_bbox_3d    = gt_steps.get("bbox_3d") or {},
            cfg           = self.cfg,
        )
        if result is None:
            return None, {
                "R_detail_parse":  None,
                "R_detail_range":  None,
                "R_detail_local":  None,
                "R_detail_global": None,
            }
        return result["score"], {
            "R_detail_parse":  result["parse"],
            "R_detail_range":  result["range"],
            "R_detail_local":  result["local"],
            "R_detail_global": result["global"],
        }

    def _load_gt_voxels(self, gt: Dict[str, Any]) -> Dict[int, Any]:
        voxel_dir  = gt.get("voxel_dir") or ""
        part_count = gt.get("part_count") or 0
        if not voxel_dir or part_count <= 0:
            return {}
        return self.voxel_loader.load(voxel_dir, part_count)

    #                                                                
    # R_phys      overall     
    #                                                                

    def _compute_phys(
        self, parsed: Dict[str, Any], gt: Dict[str, Any],
    ) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        overall = parsed.get("overall_dict")
        if overall is None and parsed.get("final_text") is None:
            return None, {
                "R_phys_overall_format": None,
                "R_phys_group":          None,
                "R_phys_logic":          None,
            }

        ov_score    = reward_phys_overall_format(overall)
        group_score = reward_phys_group(overall)
        logic_score = reward_phys_logic(parsed.get("think_steps", {}), overall)

        sum_w = (
            self.cfg.phys_overall_weight
            + self.cfg.phys_group_weight
            + self.cfg.phys_logic_weight
        )
        score = (
            self.cfg.phys_overall_weight * ov_score
            + self.cfg.phys_group_weight  * group_score
            + self.cfg.phys_logic_weight  * logic_score
        ) / sum_w if sum_w > 0 else 0.0

        return max(0.0, min(1.0, score)), {
            "R_phys_overall_format": ov_score,
            "R_phys_group":          group_score,
            "R_phys_logic":          logic_score,
        }

    #                                                                
    # P     
    #                                                                

    def _compute_penalty(
        self, parsed: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Optional[float]]]:
        p_fmt = penalty_format(parsed, self.task_cfg)
        p_oob = penalty_overflow(parsed, self.cfg.voxel_grid_size)
        p_dep = penalty_dependency(parsed)

        sum_w = (
            self.cfg.pen_format_weight
            + self.cfg.pen_overflow_weight
            + self.cfg.pen_dependency_weight
        )
        if sum_w <= 0:
            return 0.0, {"P_format": p_fmt, "P_overflow": p_oob, "P_dependency": p_dep}

        total = (
            self.cfg.pen_format_weight    * p_fmt
            + self.cfg.pen_overflow_weight  * p_oob
            + self.cfg.pen_dependency_weight * p_dep
        ) / sum_w
        return max(0.0, min(1.0, total)), {
            "P_format":     p_fmt,
            "P_overflow":   p_oob,
            "P_dependency": p_dep,
        }


#                                                                
# trl.GRPOTrainer    
#                                                                

class GRPORewardWrapper:
    """
      RewardEngine     trl.GRPOTrainer   reward_funcs    

    GRPOTrainer     rollout          
        reward_fn(completions: List[str], **kwargs)   List[float]

       kwargs   dataset             
        gt   : List[Dict]         ground truth
        meta : List[Dict]         split / category / part_count / complexity  

         
        last_breakdowns   :      rollout   RewardBreakdown   
        window_breakdowns :     reset         RewardBreakdown 
                              callback   on_log      tensorboard / wandb
    """

    def __init__(self, reward_engine: RewardEngine):
        self.reward_engine: RewardEngine               = reward_engine
        self.last_breakdowns:   List[RewardBreakdown]  = []
        self.window_breakdowns: List[RewardBreakdown]  = []

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        gts:   List[Dict[str, Any]] = kwargs.get("gt",   [{} for _ in completions])
        metas: List[Dict[str, Any]] = kwargs.get("meta", [{} for _ in completions])

        rewards:    List[float]            = []
        breakdowns: List[RewardBreakdown]  = []

        for idx, text in enumerate(completions):
            gt   = gts[idx]   if idx < len(gts)   else {}
            meta = metas[idx] if idx < len(metas) else {}
            bd   = self.reward_engine.compute(text, gt, meta)
            rewards.append(float(bd.total))
            breakdowns.append(bd)

        self.last_breakdowns = breakdowns
        self.window_breakdowns.extend(breakdowns)
        return rewards

    #       /                                            

    def reset_window(self) -> None:
        """callback     on_log                """
        self.window_breakdowns = []

    def aggregate_main(self, source: str = "window") -> Dict[str, float]:
        """     None      

        source: 'window'       'last'       rollout 
        """
        bds = self._select_source(source)
        if not bds:
            return {}
        keys = ["R_loc", "R_coarse", "R_detail", "R_phys", "P", "R_main"]
        out: Dict[str, float] = {}
        for k in keys:
            vals = [bd.main.get(k) for bd in bds]
            vals = [v for v in vals if v is not None]
            if vals:
                out[k] = sum(vals) / len(vals)
        out["total"] = sum(bd.total for bd in bds) / len(bds)
        out["count"] = float(len(bds))
        return out

    def aggregate_sub(self, source: str = "window") -> Dict[str, float]:
        """     None      """
        bds = self._select_source(source)
        if not bds:
            return {}
        sub_keys = set()
        for bd in bds:
            sub_keys.update(bd.sub.keys())
        out: Dict[str, float] = {}
        for k in sub_keys:
            vals = [bd.sub.get(k) for bd in bds]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                out[k] = sum(vals) / len(vals)
        return out

    def aggregate_info(self, source: str = "window") -> Dict[str, float]:
        """        has_overall    /    part_count_pred """
        bds = self._select_source(source)
        if not bds:
            return {}
        out: Dict[str, float] = {}
        # has_overall   
        has = [1.0 if bd.info.get("has_overall") else 0.0 for bd in bds]
        out["info/has_overall_ratio"] = sum(has) / len(has)
        #     /   part_count
        for src_key, dst_key in [
            ("pred_part_count", "info/pred_part_count_avg"),
            ("gt_part_count",   "info/gt_part_count_avg"),
            ("n_geometries",    "info/n_geometries_avg"),
        ]:
            vals = [bd.info.get(src_key) for bd in bds]
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            if vals:
                out[dst_key] = sum(vals) / len(vals)
        return out

    def _select_source(self, source: str) -> List[RewardBreakdown]:
        if source == "last":
            return self.last_breakdowns
        if source == "window":
            return self.window_breakdowns
        raise ValueError(f"Unknown aggregate source: {source}")
