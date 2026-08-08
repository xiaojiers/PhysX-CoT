"""
rewards/reward_funcs.py

  md/reward.md        /           

     
                    +    +    GT loader 
                  None   RewardEngine          

    RewardEngine CompletionParser           
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from configs import RewardConfig
from .consistency_loss import consistency_loss_mean
from .voxel_utils import (
    VoxelGTLoader,
    bbox3d_iou,
    bbox3d_l1_normalized,
    bbox3d_valid,
    bbox3d_volume,
    global_indices_to_set,
    parse_rle,
    predicted_global_voxels,
    set_f1,
    set_iou,
)

LOGGER = logging.getLogger(__name__)


#                                                                
#     
#                                                                

def _bbox2d_iou(b1: List[float], b2: List[float]) -> float:
    """[xmin, xmax, ymin, ymax]     2D bbox IoU """
    if len(b1) != 4 or len(b2) != 4:
        return 0.0
    ix1, ix2 = max(b1[0], b2[0]), min(b1[1], b2[1])
    iy1, iy2 = max(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area1 = max(0.0, b1[1] - b1[0]) * max(0.0, b1[3] - b1[2])
    area2 = max(0.0, b2[1] - b2[0]) * max(0.0, b2[3] - b2[2])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _bbox2d_valid(b: List[float]) -> bool:
    if len(b) != 4:
        return False
    xmin, xmax, ymin, ymax = b
    return (
        0.0 <= xmin <= 1.0 and 0.0 <= xmax <= 1.0
        and 0.0 <= ymin <= 1.0 and 0.0 <= ymax <= 1.0
        and xmin <= xmax and ymin <= ymax
    )


#                                                                
# 1. R_loc:    reward part_count + bbox_2d + bbox_3d 
#                                                                

def reward_count(
    pred_n: Optional[int],
    gt_n:   Optional[int],
    decay_gamma: float,
) -> Optional[float]:
    """R_count = exp(-    |n  - n|) """
    if pred_n is None or gt_n is None or gt_n <= 0:
        return None
    return math.exp(-decay_gamma * abs(pred_n - gt_n))


def reward_bbox2d(
    pred_dict: Dict[str, List[float]],
    gt_dict:   Dict[str, List[float]],
) -> Optional[float]:
    """  part 2D IoU    GT     part      IoU=0    """
    if not gt_dict:
        return None
    ious: List[float] = []
    for pid, gt_box in gt_dict.items():
        pred_box = pred_dict.get(pid)
        ious.append(_bbox2d_iou(pred_box, gt_box) if pred_box else 0.0)
    return sum(ious) / len(ious) if ious else None


def reward_bbox3d(
    pred_dict: Dict[str, List[int]],
    gt_dict:   Dict[str, List[int]],
    iou_beta:  float,
    l1_eta:    float,
    grid_size: int,
) -> Optional[float]:
    """    IoU3d + (1- )   exp(-    || || ) """
    if not gt_dict:
        return None
    scores: List[float] = []
    for pid, gt_b in gt_dict.items():
        pred_b = pred_dict.get(pid)
        if not pred_b:
            scores.append(0.0)
            continue
        iou_v = bbox3d_iou(pred_b, gt_b)
        l1_v  = bbox3d_l1_normalized(pred_b, gt_b, grid_size=grid_size)
        s = iou_beta * iou_v + (1.0 - iou_beta) * math.exp(-l1_eta * l1_v)
        scores.append(s)
    return sum(scores) / len(scores) if scores else None


#                                                                
# 2. R_coarse:     reward primitive     
#                                                                

def reward_coarse_primitive(
    pred_dict: Dict[str, Dict[str, Optional[str]]],
    gt_dict:   Dict[str, Dict[str, Optional[str]]],
    shape_w:   float,
    axis_w:    float,
    ratio_w:   float,
) -> Optional[float]:
    """    GT part    (shape, axis, ratio)             """
    if not gt_dict:
        return None
    total_w = shape_w + axis_w + ratio_w
    if total_w <= 0:
        return None
    scores: List[float] = []
    for pid, gt_p in gt_dict.items():
        pred_p = pred_dict.get(pid, {})
        s = 0.0
        if gt_p.get("shape_label") and pred_p.get("shape_label") == gt_p["shape_label"]:
            s += shape_w
        if gt_p.get("major_axis") and pred_p.get("major_axis") == gt_p["major_axis"]:
            s += axis_w
        if gt_p.get("aspect_ratio") and pred_p.get("aspect_ratio") == gt_p["aspect_ratio"]:
            s += ratio_w
        scores.append(s / total_w)
    return sum(scores) / len(scores) if scores else None


#                                                                
# 3. R_detail:      reward    
#                                                                

def reward_detail(
    geometries:     Dict[int, str],            # {part_k: rle_str}     
    pred_bbox_3d:   Dict[str, List[int]],      #    think    bbox_3d
    gt_voxels:      Dict[int, np.ndarray],     # GT (N,3)      part    
    gt_bbox_3d:     Dict[str, List[int]],      # GT bbox_3d   part_id "l_k" 
    cfg:            RewardConfig,
) -> Optional[Dict[str, float]]:
    """
       (R_detail    ,      ) 
            [0, 1] 
        R_parse  : RLE           token   1.0
        R_range  : 1 - n_invalid / n_total
        R_local  : 0.6 IoU(local) + 0.4 F1(local)
        R_global : IoU(global) pred = pred_bbox_min + local gt = ind_{k}.npy 

      completion     geometry    None           
    """
    if not geometries:
        return None

    parse_scores: List[float] = []
    range_scores: List[float] = []
    local_scores: List[float] = []
    global_scores: List[float] = []

    for k, rle_str in geometries.items():
        pid = f"l_{k}"

        # 1) Parse
        local_ids, n_total, n_parse_inv = parse_rle(rle_str)
        parse_scores.append(0.0 if n_total == 0 or n_parse_inv > 0 else 1.0)

        # 2) Range      bbox_3d     
        pred_b = pred_bbox_3d.get(pid)
        if pred_b and bbox3d_valid(pred_b, cfg.voxel_grid_size):
            cap = bbox3d_volume(pred_b)
            n_oob = sum(1 for lid in local_ids if not (0 <= lid < cap))
        else:
            n_oob = len(local_ids)   # bbox           
            cap = 0
        denom = max(1, n_total)
        range_scores.append(1.0 - (n_parse_inv + n_oob) / denom)

        # 3) Local IoU/F1          bbox_3d      
        gt_arr = gt_voxels.get(k)
        gt_b   = gt_bbox_3d.get(pid)
        if gt_arr is not None and gt_b and bbox3d_valid(gt_b, cfg.voxel_grid_size):
            xmin, _, ymin, _, zmin, _ = gt_b
            dy = gt_b[3] - gt_b[2] + 1
            dz = gt_b[5] - gt_b[4] + 1
            gt_local: Set[int] = {
                (int(r[0]) - xmin) * dy * dz
                + (int(r[1]) - ymin) * dz
                + (int(r[2]) - zmin)
                for r in gt_arr
            }
            pred_local: Set[int] = {lid for lid in local_ids if 0 <= lid < (cap or 0)}
            iou = set_iou(pred_local, gt_local)
            f1  = set_f1(pred_local, gt_local)
            local_scores.append(cfg.detail_local_iou_w * iou + cfg.detail_local_f1_w * f1)
        else:
            local_scores.append(0.0)

        # 4) Global IoU     bbox          GT        
        if pred_b and bbox3d_valid(pred_b, cfg.voxel_grid_size) and gt_arr is not None:
            pred_global, _, _ = predicted_global_voxels(rle_str, pred_b)
            gt_global = global_indices_to_set(gt_arr)
            global_scores.append(set_iou(pred_global, gt_global))
        else:
            global_scores.append(0.0)

    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    parse_v  = _avg(parse_scores)
    range_v  = _avg(range_scores)
    local_v  = _avg(local_scores)
    global_v = _avg(global_scores)

    total_w = (
        cfg.detail_parse_weight
        + cfg.detail_range_weight
        + cfg.detail_local_weight
        + cfg.detail_global_weight
    )
    score = (
        cfg.detail_parse_weight  * parse_v
        + cfg.detail_range_weight * range_v
        + cfg.detail_local_weight * local_v
        + cfg.detail_global_weight * global_v
    ) / total_w if total_w > 0 else 0.0

    return {
        "score":        max(0.0, min(1.0, score)),
        "parse":        parse_v,
        "range":        range_v,
        "local":        local_v,
        "global":       global_v,
        "n_geometries": float(len(geometries)),
    }


#                                                                
# 4. R_phys:      reward overall     
#                                                                

_VALID_JOINT_TYPES = {"E", "R", "P", "S", "H", "F", "A", "B", "C", "D", "CB"}


def reward_phys_overall_format(overall_dict: Optional[Dict[str, Any]]) -> float:
    """<overall>       name/category/dimension/parts/group_info """
    if not overall_dict:
        return 0.0
    s = 0.0
    if overall_dict.get("name"):       s += 0.20
    if overall_dict.get("category"):   s += 0.15
    if overall_dict.get("dimension"):  s += 0.20
    if overall_dict.get("parts"):      s += 0.30
    if overall_dict.get("group_info"): s += 0.15
    return s


def reward_phys_group(overall_dict: Optional[Dict[str, Any]]) -> float:
    """
    Group_info     
        joint type   {E,R,P,S,H,F,A,B,C,D,CB}
           group     part_id    parts  
           part         group  
    """
    if not overall_dict:
        return 0.0
    groups = overall_dict.get("group_info") or []
    parts  = overall_dict.get("parts") or {}
    if not groups:
        return 0.0

    type_ok    = sum(1 for g in groups if (g.get("type") or "").split()[0] in _VALID_JOINT_TYPES)
    type_score = type_ok / len(groups)

    if parts:
        all_pids = set(parts.keys())
        ref_pids: Set[str] = set()
        for g in groups:
            for pid in g.get("parts") or []:
                ref_pids.add(pid)
        ref_score = len(ref_pids & all_pids) / max(1, len(all_pids))
        cov_score = len(ref_pids & all_pids) / max(1, len(ref_pids)) if ref_pids else 0.0
    else:
        ref_score = cov_score = 0.0

    return 0.4 * type_score + 0.3 * ref_score + 0.3 * cov_score


def reward_phys_logic(
    think_steps:  Dict[str, Any],
    overall_dict: Optional[Dict[str, Any]],
) -> float:
    """        part_count   overall.parts     part_id      """
    if not overall_dict:
        return 0.0
    parts = overall_dict.get("parts") or {}
    if not parts:
        return 0.0

    score = 0.0
    pred_n = think_steps.get("part_count")
    if pred_n is not None and pred_n == len(parts):
        score += 0.5
    elif pred_n is not None and abs(pred_n - len(parts)) == 1:
        score += 0.2

    think_ids = set(think_steps.get("bbox_2d", {}).keys())
    final_ids = set(parts.keys())
    if think_ids and final_ids:
        if think_ids == final_ids:
            score += 0.5
        else:
            overlap = len(think_ids & final_ids) / max(len(think_ids | final_ids), 1)
            score += 0.5 * overlap
    return score


#                                                                
# 5. P:      penalty
#                                                                

_VALID_PRIMITIVES = {"cuboid", "cylinder", "sphere", "complex"}
_VALID_AXIS       = {"x", "y", "z"}
_VALID_RATIO      = {"very_flat", "flat", "balanced", "tall", "elongated"}
_VALID_HARDNESS     = {"soft", "semi_rigid", "rigid"}
_VALID_ROUGHNESS    = {"smooth", "textured", "rough"}
_VALID_REFLECTIVITY = {"matte", "glossy", "highly_reflective"}
_VALID_TRANSPARENCY = {"opaque", "translucent", "transparent"}


def penalty_format(parsed: Dict[str, Any], task_cfg) -> float:
    """     think / overall        think      """
    p = 0.0
    if not parsed.get("think_text"):
        p += 0.5
    if not parsed.get("overall_dict") and not parsed.get("final_text"):
        p += 0.3
    steps_found = parsed.get("think_steps", {}).get("steps_found", set())
    if isinstance(steps_found, (list, tuple)):
        steps_found = set(steps_found)
    if len(steps_found) < 5:
        p += 0.05 * (5 - len(steps_found))
    return min(1.0, p)


def penalty_overflow(
    parsed:    Dict[str, Any],
    grid_size: int,
) -> float:
    """     bbox_2d   [0,1] bbox_3d   [0,grid) primitive/surface      """
    p = 0.0
    steps = parsed.get("think_steps", {})

    for _, b in (steps.get("bbox_2d") or {}).items():
        if not _bbox2d_valid(b):
            p += 0.10

    for _, b in (steps.get("bbox_3d") or {}).items():
        if not bbox3d_valid(b, grid_size):
            p += 0.15

    for _, prim in (steps.get("primitive_shape") or {}).items():
        if prim.get("shape_label") and prim["shape_label"] not in _VALID_PRIMITIVES:
            p += 0.05
        if prim.get("major_axis") and prim["major_axis"] not in _VALID_AXIS:
            p += 0.03
        if prim.get("aspect_ratio") and prim["aspect_ratio"] not in _VALID_RATIO:
            p += 0.03

    for _, sf in (steps.get("surface_features") or {}).items():
        if sf.get("hardness") and sf["hardness"] not in _VALID_HARDNESS:           p += 0.03
        if sf.get("roughness") and sf["roughness"] not in _VALID_ROUGHNESS:        p += 0.03
        if sf.get("reflectivity") and sf["reflectivity"] not in _VALID_REFLECTIVITY: p += 0.03
        if sf.get("transparency") and sf["transparency"] not in _VALID_TRANSPARENCY: p += 0.03

    return min(1.0, p)


def penalty_dependency(
    parsed: Dict[str, Any],
) -> float:
    """
         
        part_id     think   overall        
        inter_part_positions        part
           part_id
    """
    p = 0.0
    steps = parsed.get("think_steps", {})
    overall = parsed.get("overall_dict")

    declared_ids: Set[str] = set(steps.get("bbox_2d", {}).keys())
    declared_ids |= set(steps.get("bbox_3d", {}).keys())

    # 1) inter_part_positions        part
    for pid, neigh in (steps.get("inter_part_positions") or {}).items():
        if pid not in declared_ids:
            p += 0.05
        for nb_id in neigh.keys():
            if nb_id not in declared_ids:
                p += 0.03

    # 2) overall.parts   think      part_id      
    if overall and overall.get("parts"):
        final_ids = set(overall["parts"].keys())
        if declared_ids and final_ids != declared_ids:
            sym_diff = (declared_ids ^ final_ids)
            p += min(0.4, 0.05 * len(sym_diff))
        for pid in overall["parts"].keys():
            if not re.match(r"^l_\d+$", pid):
                p += 0.10

    return min(1.0, p)


#                                                                
# 6. 2D-3D         R_loc     
#                                                                

def consistency_penalty_for_loc(
    parsed: Dict[str, Any],
    cfg:    RewardConfig,
) -> float:
    """
         L_cons         RewardEngine   R_loc' = R_loc -  _cons   L_cons
                cons_enable=True     
    """
    if not cfg.cons_enable:
        return 0.0
    steps = parsed.get("think_steps", {})
    return consistency_loss_mean(
        bbox_2d_dict       = steps.get("bbox_2d") or {},
        bbox_3d_dict       = steps.get("bbox_3d") or {},
        grid_size          = cfg.voxel_grid_size,
        center_w           = cfg.cons_center_weight,
        proj_w             = cfg.cons_proj_weight,
        proj_ratio_w       = cfg.cons_proj_ratio_w,
        proj_shape_w       = cfg.cons_proj_shape_w,
        proj_softmin_tau   = cfg.cons_proj_softmin_tau,
    )
