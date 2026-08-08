"""
rewards/voxel_utils.py

              reward         

     
     RLE                                                  
     parse_rle(s)        '0 1-5 36-41'   set[int]           
     encode_rle(ids)     set[int] / list[int]   'a b-c  '   
                                                            
     bbox_3d                                             
     bbox3d_volume       (xmin,xmax,ymin,ymax,zmin,zmax)    
     bbox3d_iou                                             
     bbox3d_l1_normalized     grid_size                   
     bbox3d_dims            (dx, dy, dz)                    
     bbox3d_valid                +                 
                                                            
                                                    
     local_to_global     local_id + bbox_min   (N,3)        
     set_iou / set_f1                                       
                                                            
     GT                                                  
     VoxelGTLoader       LRU    ind_{k}.npy            
                                                            
"""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# RLE token    "12"   "12-15"    token         invalid_count 
_RLE_TOKEN = re.compile(r"^(\d+)(?:-(\d+))?$")


#                                                                
# 1. RLE    
#                                                                

def parse_rle(rle_str: str) -> Tuple[Set[int], int, int]:
    """
       RLE     

    Returns
    -------
    ids        : set[int]      local_id                 
    n_total    : int        token        
    n_invalid  : int           token   

      
        parse_rle("0 1-5 36-41")     ({0,1,2,3,4,5,36,37,...,41}, 3, 0)
        parse_rle("0 1-5 abc 7")     ({0,1,2,3,4,5,7},          4, 1)
        parse_rle("5-3")             (set(),                     1, 1)   # start>end     
    """
    if not rle_str or not rle_str.strip():
        return set(), 0, 0

    ids: Set[int] = set()
    n_total = 0
    n_invalid = 0

    for tok in rle_str.strip().split():
        n_total += 1
        m = _RLE_TOKEN.match(tok)
        if m is None:
            n_invalid += 1
            continue
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) is not None else a
        if a > b:
            n_invalid += 1
            continue
        ids.update(range(a, b + 1))
    return ids, n_total, n_invalid


def encode_rle(ids: Iterable[int]) -> str:
    """list[int] / set[int]      RLE       SFT         """
    sorted_ids = sorted(set(int(i) for i in ids))
    if not sorted_ids:
        return ""
    out: List[str] = []
    start = prev = sorted_ids[0]
    for n in sorted_ids[1:]:
        if n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return " ".join(out)


#                                                                
# 2. bbox_3d       [xmin, xmax, ymin, ymax, zmin, zmax] 
#                                                                

def bbox3d_dims(b: List[int]) -> Tuple[int, int, int]:
    """   (dx, dy, dz)    dx = xmax - xmin + 1      """
    if len(b) != 6:
        return 0, 0, 0
    return b[1] - b[0] + 1, b[3] - b[2] + 1, b[5] - b[4] + 1


def bbox3d_volume(b: List[int]) -> int:
    dx, dy, dz = bbox3d_dims(b)
    if dx <= 0 or dy <= 0 or dz <= 0:
        return 0
    return dx * dy * dz


def bbox3d_valid(b: List[int], grid_size: int = 32) -> bool:
    """     +      +          """
    if len(b) != 6:
        return False
    xmin, xmax, ymin, ymax, zmin, zmax = b
    if xmin > xmax or ymin > ymax or zmin > zmax:
        return False
    for v in b:
        if not isinstance(v, (int, float)):
            return False
        if not (0 <= v < grid_size):
            return False
    return True


def bbox3d_iou(b1: List[int], b2: List[int]) -> float:
    """   bbox_3d     IoU      """
    if len(b1) != 6 or len(b2) != 6:
        return 0.0
    ix1, ix2 = max(b1[0], b2[0]), min(b1[1], b2[1])
    iy1, iy2 = max(b1[2], b2[2]), min(b1[3], b2[3])
    iz1, iz2 = max(b1[4], b2[4]), min(b1[5], b2[5])
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1) * max(0, iz2 - iz1 + 1)
    if inter == 0:
        return 0.0
    union = bbox3d_volume(b1) + bbox3d_volume(b2) - inter
    return inter / union if union > 0 else 0.0


def bbox3d_l1_normalized(b1: List[int], b2: List[int], grid_size: int = 32) -> float:
    """6      L1     grid_size        [0, 6] """
    if len(b1) != 6 or len(b2) != 6:
        return 6.0
    return sum(abs(b1[i] - b2[i]) for i in range(6)) / float(grid_size)


#                                                                
# 3.       
#                                                                

def local_ids_to_global(
    local_ids: Iterable[int],
    bbox_3d:   List[int],
) -> Set[Tuple[int, int, int]]:
    """
          ID         (x, y, z)      

           SFT         
        local_id = (x - xmin) * dy * dz + (y - ymin) * dz + (z - zmin)
        dy = ymax - ymin + 1,  dz = zmax - zmin + 1

       local_id    dx*dy*dz   ID             
       parse_rle   n_invalid      token    
    """
    if len(bbox_3d) != 6:
        return set()
    xmin, _, ymin, _, zmin, _ = bbox_3d
    dx, dy, dz = bbox3d_dims(bbox_3d)
    if dx <= 0 or dy <= 0 or dz <= 0:
        return set()
    capacity = dx * dy * dz

    out: Set[Tuple[int, int, int]] = set()
    for lid in local_ids:
        if lid < 0 or lid >= capacity:
            continue
        zp = lid % dz
        yp = (lid // dz) % dy
        xp = lid // (dy * dz)
        out.add((xmin + xp, ymin + yp, zmin + zp))
    return out


def global_indices_to_set(arr: np.ndarray) -> Set[Tuple[int, int, int]]:
    """(N, 3) int      {(x, y, z), ...} """
    if arr is None or len(arr) == 0:
        return set()
    return {(int(r[0]), int(r[1]), int(r[2])) for r in arr}


def set_iou(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def set_f1(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tp = len(a & b)
    if tp == 0:
        return 0.0
    prec = tp / len(a)
    rec  = tp / len(b)
    return 2 * prec * rec / (prec + rec)


#                                                                
# 4.    GT    LRU    
#                                                                

class VoxelGTLoader:
    """
        + LRU    data/tmp/partseg/{obj_id}/{grid}/ind_{k}.npy 

       GRPO     GT     voxel_dir      + part_count 
         .npy     reward          LRU       N   obj_id
        part            IO 

        voxels      
        { part_k: np.ndarray(shape=(N, 3), dtype=int64) }
    """

    def __init__(self, dataset_root: Optional[str], cache_size: int = 256):
        self.dataset_root = dataset_root
        self.cache_size   = max(1, int(cache_size))
        self._cache: "OrderedDict[str, Dict[int, np.ndarray]]" = OrderedDict()

    #                                                        
    def _resolve(self, voxel_dir: str) -> Optional[str]:
        if not voxel_dir:
            return None
        if os.path.isabs(voxel_dir):
            return voxel_dir
        if self.dataset_root:
            return os.path.join(self.dataset_root, voxel_dir)
        return voxel_dir

    #          LRU                                           
    def load(self, voxel_dir: str, part_count: int) -> Dict[int, np.ndarray]:
        """
           {part_k: (N,3) int   }       part         
           key = voxel_dir       
        """
        if not voxel_dir or part_count <= 0:
            return {}

        if voxel_dir in self._cache:
            self._cache.move_to_end(voxel_dir)
            return self._cache[voxel_dir]

        abs_dir = self._resolve(voxel_dir)
        out: Dict[int, np.ndarray] = {}
        if abs_dir and os.path.isdir(abs_dir):
            for k in range(part_count):
                path = os.path.join(abs_dir, f"ind_{k}.npy")
                if not os.path.exists(path):
                    continue
                try:
                    arr = np.load(path)
                    if arr.ndim == 2 and arr.shape[1] == 3:
                        out[k] = arr.astype(np.int64, copy=False)
                except Exception as exc:
                    LOGGER.warning("VoxelGTLoader: failed %s (%s)", path, exc)
        else:
            LOGGER.debug("VoxelGTLoader: voxel_dir not found: %s", abs_dir)

        self._cache[voxel_dir] = out
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return out

    #       part                                              
    def load_one(self, voxel_dir: str, part_count: int, part_k: int) -> Optional[np.ndarray]:
        all_parts = self.load(voxel_dir, part_count)
        return all_parts.get(part_k)

    def clear(self) -> None:
        self._cache.clear()


#                                                                
# 5.        
#                                                                

def predicted_global_voxels(
    rle_str:  str,
    bbox_3d:  List[int],
) -> Tuple[Set[Tuple[int, int, int]], int, int]:
    """
                RLE     +    bbox_3d             

    Returns
    -------
    voxels    : set[(x, y, z)]             
    n_total   : int               RLE token   
    n_invalid : int                    +    local_id         
    """
    if len(bbox_3d) != 6:
        return set(), 0, 0
    local_ids, n_total, n_parse_invalid = parse_rle(rle_str)
    capacity = bbox3d_volume(bbox_3d)

    valid_local: Set[int] = set()
    n_oob = 0
    for lid in local_ids:
        if 0 <= lid < capacity:
            valid_local.add(lid)
        else:
            n_oob += 1

    voxels = local_ids_to_global(valid_local, bbox_3d)
    return voxels, n_total, n_parse_invalid + n_oob
