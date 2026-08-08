"""
rewards/consistency_loss.py

2D-3D consistency reward. See ``md/2d-3d-consistency-loss.md``.

    L_cons =  _c   L_center  +   _p   L_proj

   
  L_center :        
        bbox_3d   (x, y)                bbox_2d      
        bbox_2d               

  L_proj   :             
      bbox_3d   xy / xz / yz              bbox_2d  
      log aspect ratio +                soft-min    
         hard-min      

   
                        Python    / list[int|float] 
  bbox_2d   : [x_min, x_max, y_min, y_max]        [0, 1]
  bbox_3d   : [x_min, x_max, y_min, y_max, z_min, z_max]       {0, ..., G-1}

            0.0         0               reward  
"""

from __future__ import annotations

import math
from typing import List, Tuple

EPS = 1e-6


#                                                                
# 1.        
#                                                                

def center_loss(
    bbox_2d:   List[float],
    bbox_3d:   List[int],
    grid_size: int = 32,
) -> float:
    """
    L_center = ((u_hat - u) / (w +  ))^2 + ((v_hat - v) / (h +  ))^2

    bbox_3d   3D    (cx_3d, cy_3d) / grid_size    image-plane      
    See ``md/2d-3d-consistency-loss.md`` for the full definition.
    """
    if len(bbox_2d) != 4 or len(bbox_3d) != 6:
        return 0.0
    if grid_size <= 0:
        return 0.0

    xmin2, xmax2, ymin2, ymax2 = bbox_2d
    u = (xmin2 + xmax2) / 2.0
    v = (ymin2 + ymax2) / 2.0
    w = max(0.0, xmax2 - xmin2)
    h = max(0.0, ymax2 - ymin2)

    xmin3, xmax3, ymin3, ymax3, _, _ = bbox_3d
    u_hat = ((xmin3 + xmax3) / 2.0) / float(grid_size)
    v_hat = ((ymin3 + ymax3) / 2.0) / float(grid_size)

    du = (u_hat - u) / (w + EPS)
    dv = (v_hat - v) / (h + EPS)
    return du * du + dv * dv


#                                                                
# 2.              soft-min 
#                                                                

def _log_aspect(width: float, height: float) -> float:
    return math.log((height + EPS) / (width + EPS))


def _normalized_wh(width: float, height: float) -> Tuple[float, float]:
    s = width + height + EPS
    return width / s, height / s


def _l1(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _single_proj_cost(
    r2d: float,
    q2d: Tuple[float, float],
    pw:  float,
    ph:  float,
    ratio_w: float,
    shape_w: float,
) -> float:
    """             _r | ratio| +  _s || shape||  """
    r_proj = _log_aspect(pw, ph)
    q_proj = _normalized_wh(pw, ph)
    return ratio_w * abs(r2d - r_proj) + shape_w * _l1(q2d, q_proj)


def proj_loss(
    bbox_2d:    List[float],
    bbox_3d:    List[int],
    ratio_w:    float = 0.5,
    shape_w:    float = 0.5,
    softmin_tau: float = 0.5,
) -> float:
    """
    L_proj = -    log( e^(-L_xy/ ) + e^(-L_xz/ ) + e^(-L_yz/ ) )

      xy / xz / yz             ratio + shape    
      soft-min      hard-min      
    """
    if len(bbox_2d) != 4 or len(bbox_3d) != 6 or softmin_tau <= 0:
        return 0.0

    xmin2, xmax2, ymin2, ymax2 = bbox_2d
    w = max(0.0, xmax2 - xmin2)
    h = max(0.0, ymax2 - ymin2)
    if w <= 0 and h <= 0:
        return 0.0

    r2d = _log_aspect(w, h)
    q2d = _normalized_wh(w, h)

    xmin3, xmax3, ymin3, ymax3, zmin3, zmax3 = bbox_3d
    dx = max(0.0, xmax3 - xmin3 + 1)
    dy = max(0.0, ymax3 - ymin3 + 1)
    dz = max(0.0, zmax3 - zmin3 + 1)

    L_xy = _single_proj_cost(r2d, q2d, dx, dy, ratio_w, shape_w)
    L_xz = _single_proj_cost(r2d, q2d, dx, dz, ratio_w, shape_w)
    L_yz = _single_proj_cost(r2d, q2d, dy, dz, ratio_w, shape_w)

    # soft-min via log-sum-exp on negatives        
    neg = [-L_xy / softmin_tau, -L_xz / softmin_tau, -L_yz / softmin_tau]
    m = max(neg)
    s = sum(math.exp(v - m) for v in neg)
    return -softmin_tau * (m + math.log(s + EPS))


#                                                                
# 3.    L_cons =  _c   L_center +  _p   L_proj
#                                                                

def consistency_loss(
    bbox_2d:        List[float],
    bbox_3d:        List[int],
    grid_size:      int   = 32,
    center_w:       float = 0.3,
    proj_w:         float = 0.7,
    proj_ratio_w:   float = 0.5,
    proj_shape_w:   float = 0.5,
    proj_softmin_tau: float = 0.5,
) -> float:
    """
      part   2D-3D       

                  2D-3D        
       reward_engine   R_loc    - _cons      L_cons      
    """
    if len(bbox_2d) != 4 or len(bbox_3d) != 6:
        return 0.0
    Lc = center_loss(bbox_2d, bbox_3d, grid_size=grid_size)
    Lp = proj_loss(
        bbox_2d, bbox_3d,
        ratio_w     = proj_ratio_w,
        shape_w     = proj_shape_w,
        softmin_tau = proj_softmin_tau,
    )
    return center_w * Lc + proj_w * Lp


def consistency_loss_mean(
    bbox_2d_dict: dict,
    bbox_3d_dict: dict,
    grid_size:    int   = 32,
    center_w:     float = 0.3,
    proj_w:       float = 0.7,
    proj_ratio_w: float = 0.5,
    proj_shape_w: float = 0.5,
    proj_softmin_tau: float = 0.5,
) -> float:
    """
      part       
    bbox_2d_dict / bbox_3d_dict : Dict[part_id, list]

       (bbox_2d   bbox_3d     )   part          0 
    """
    common = set(bbox_2d_dict.keys()) & set(bbox_3d_dict.keys())
    if not common:
        return 0.0
    losses: List[float] = []
    for pid in common:
        losses.append(consistency_loss(
            bbox_2d_dict[pid], bbox_3d_dict[pid],
            grid_size=grid_size,
            center_w=center_w, proj_w=proj_w,
            proj_ratio_w=proj_ratio_w, proj_shape_w=proj_shape_w,
            proj_softmin_tau=proj_softmin_tau,
        ))
    return sum(losses) / len(losses) if losses else 0.0
