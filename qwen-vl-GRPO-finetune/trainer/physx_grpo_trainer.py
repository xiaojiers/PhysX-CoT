"""
trainer/physx_grpo_trainer.py

PhysXGRPOTrainer trl.GRPOTrainer         
         step   SAM       SAMEmbeddingInjector 
   rollout / advantage / KL        

   hook     trl 0.16~0.20    API  

      1) _prepare_inputs(inputs)
          trl   rollout      inputs     list[dict]   dict[  list] 
              sam_feature           set   injector(auto_clear=False) 
               forward / generate        sam_feats 
   
      2) training_step(model, inputs)
                   set              step       
          step     clear             
   
      3) batch     
        SAMEmbeddingInjector.__call__      input_ids batch   
           repeat_interleave    sam_feats    num_generations     

            
    prompt      collation trl     
    GT / meta     GRPORewardWrapper    reward_funcs kwargs    
       rollout geometry_l_k                
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

try:
    from trl import GRPOTrainer
except Exception as exc:  # pragma: no cover
    GRPOTrainer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from .sam_features import SAMFeatureLoader
from .sam_injection import SAMEmbeddingInjector

LOGGER = logging.getLogger(__name__)


#         inputs       sam_feature                                   


def _extract_sam_paths(inputs: Any) -> List[Optional[str]]:
    """  trl      _prepare_inputs / training_step   inputs     sam_feature    

             
        list[dict]                    dict HF Dataset      
        dict[     list/Tensor] trl        batch      
        BatchEncoding          dict-like

        batch     List[Optional[str]] 
    """
    if inputs is None:
        return []
    # list[dict]
    if isinstance(inputs, (list, tuple)):
        return [
            (item.get("sam_feature") if isinstance(item, dict) else None)
            for item in inputs
        ]
    # dict[     list]
    if isinstance(inputs, dict):
        col = inputs.get("sam_feature")
        if col is None:
            return []
        if isinstance(col, (list, tuple)):
            return [str(p) if p else None for p in col]
        #     
        return [str(col)] if col else [None]
    return []


#    PhysXGRPOTrainer                                                          


if GRPOTrainer is None:
    class PhysXGRPOTrainer:  # type: ignore[no-redef]
        """trl                 """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "trl is not installed. Please run: pip install trl\n"
                f"Original import error: {_IMPORT_ERROR}"
            )
else:

    class PhysXGRPOTrainer(GRPOTrainer):  # type: ignore[misc]
        """  SAM     GRPOTrainer """

        def __init__(
            self,
            *args,
            sam_injector: Optional[SAMEmbeddingInjector] = None,
            sam_loader:   Optional[SAMFeatureLoader]     = None,
            sam_dtype:    torch.dtype = torch.bfloat16,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.sam_injector = sam_injector
            self.sam_loader   = sam_loader
            self.sam_dtype    = sam_dtype

            if sam_injector is None or sam_loader is None:
                LOGGER.warning(
                    "PhysXGRPOTrainer: sam_injector or sam_loader not provided; "
                    "SAM              <sam_feat>      /SFT      "
                )

        #          +                                                   

        def _inject_sam_for_inputs(self, inputs: Any) -> None:
            """  inputs   sam_feature           set   injector """
            if self.sam_injector is None or self.sam_loader is None:
                return
            paths = _extract_sam_paths(inputs)
            if not paths:
                self.sam_injector.clear()
                return

            try:
                device = self.accelerator.device
            except Exception:
                device = next(self.model.parameters()).device

            sam_feats, _mask = self.sam_loader.load_batch(
                paths, device=device, dtype=self.sam_dtype,
            )
            # auto_clear=False rollout / forward / KL    forward        
            self.sam_injector.set_batch(sam_feats, auto_clear=False)

        def _clear_sam(self) -> None:
            if self.sam_injector is not None:
                self.sam_injector.clear()

        #    trl rollout                                                      

        def _prepare_inputs(self, inputs):  # type: ignore[override]
            self._inject_sam_for_inputs(inputs)
            return super()._prepare_inputs(inputs)

        #       Trainer step                                             

        def training_step(self, model, inputs, *args, **kwargs):  # type: ignore[override]
            # _prepare_inputs           trl    _prepare_inputs    
            #              npz     set       tensor  
            self._inject_sam_for_inputs(inputs)
            try:
                return super().training_step(model, inputs, *args, **kwargs)
            finally:
                self._clear_sam()

        def prediction_step(self, model, inputs, *args, **kwargs):  # type: ignore[override]
            self._inject_sam_for_inputs(inputs)
            try:
                return super().prediction_step(model, inputs, *args, **kwargs)
            finally:
                self._clear_sam()
