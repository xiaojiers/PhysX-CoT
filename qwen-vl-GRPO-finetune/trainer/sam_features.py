"""
trainer/sam_features.py

SAM               

        collator 
  trl.GRPOTrainer        prompt   token   batch      
       pixel_values / image_grid_thw       collator       

              dataset    sam_feature         
      step     [B, max_parts, 256] tensor    SAMEmbeddingInjector 

     
      1) prompt batch         sam_feature path dataset adapter     
      2) SAMFeatureLoader.load_batch(paths)    [B, max_parts, 256]
      3) trainer     injector.set_batch(...)     forward / generate

LRU       npz    < 64KB           obj        
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


class SAMFeatureLoader:
    """  npz      SAM RoI      batch         

    npz        SFT     
      keys = part_0, part_1, ..., part_{N-1}
      values = float32 array of shape [256]

                   trainer step            
          dataloader worker         per-worker 
    """

    def __init__(self, cache_size: int = 512, feat_dim: int = 256) -> None:
        self.cache_size = int(cache_size)
        self.feat_dim = int(feat_dim)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._miss = 0
        self._hit = 0

    #            LRU                                                    

    def _load_npz(self, path: str) -> Optional[np.ndarray]:
        """   npz    [N_parts, feat_dim] float32      None """
        if path in self._cache:
            self._hit += 1
            self._cache.move_to_end(path)
            return self._cache[path]

        self._miss += 1
        try:
            data = np.load(path)
            keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
            feats = [data[k].astype(np.float32, copy=False) for k in keys]
            arr = np.stack(feats, axis=0) if feats else None
        except Exception as exc:
            LOGGER.warning("Failed to load SAM npz %s: %s", path, exc)
            return None

        if arr is None or arr.ndim != 2 or arr.shape[1] != self.feat_dim:
            LOGGER.warning("Invalid SAM npz shape from %s: got %s",
                           path, None if arr is None else arr.shape)
            return None

        self._cache[path] = arr
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arr

    #                                                                    

    def load_batch(
        self,
        paths: List[Optional[str]],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """        SAM    

        Args:
          paths :    B              None 
          device:        None     CPU    GPU     
          dtype :      dtype       bfloat16       

        Returns:
          sam_feats : [B, max_parts, feat_dim]   None   batch      
          feat_mask : [B, max_parts] bool True           None
        """
        loaded: List[Optional[np.ndarray]] = [
            self._load_npz(p) if p else None for p in paths
        ]

        valid = [arr for arr in loaded if arr is not None]
        if not valid:
            return None, None

        max_parts = max(arr.shape[0] for arr in valid)
        B = len(loaded)
        sam_feats = torch.zeros(B, max_parts, self.feat_dim, dtype=dtype)
        feat_mask = torch.zeros(B, max_parts, dtype=torch.bool)

        for b, arr in enumerate(loaded):
            if arr is None:
                continue
            n = arr.shape[0]
            sam_feats[b, :n] = torch.from_numpy(arr).to(dtype=dtype)
            feat_mask[b, :n] = True

        if device is not None:
            sam_feats = sam_feats.to(device, non_blocking=True)
            feat_mask = feat_mask.to(device, non_blocking=True)

        return sam_feats, feat_mask

    #                                                                         

    def cache_stats(self) -> dict:
        total = self._hit + self._miss
        return {
            "size": len(self._cache),
            "hit": self._hit,
            "miss": self._miss,
            "hit_ratio": (self._hit / total) if total else 0.0,
        }
