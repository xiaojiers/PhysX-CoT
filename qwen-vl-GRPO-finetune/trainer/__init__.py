"""trainer/

GRPO          trl.GRPOTrainer            

      
    sam_injection   : SAMProjector + SAMEmbeddingInjector embed_tokens hook 
    sam_features    : SAMFeatureLoader npz     + LRU    + batch    
    model_setup     : Qwen3-VL + SFT adapter + SAM + GRPO LoRA
    reward_callback : RewardComponentLoggingCallback main / sub / info    
    physx_grpo_trainer : PhysXGRPOTrainer trl.GRPOTrainer       SAM 
"""

from .sam_injection import (
    SAMEmbeddingInjector,
    SAMProjector,
    load_sam_projector_state,
    repeat_for_generations,
)
from .sam_features import SAMFeatureLoader
from .model_setup import (
    NEW_SPECIAL_TOKENS,
    SAMSetupConfig,
    SetupResult,
    setup_model_and_processor,
)
from .reward_callback import RewardComponentLoggingCallback
from .physx_grpo_trainer import PhysXGRPOTrainer

__all__ = [
    # SAM   
    "SAMEmbeddingInjector",
    "SAMProjector",
    "load_sam_projector_state",
    "repeat_for_generations",
    "SAMFeatureLoader",
    #     
    "NEW_SPECIAL_TOKENS",
    "SAMSetupConfig",
    "SetupResult",
    "setup_model_and_processor",
    #     
    "RewardComponentLoggingCallback",
    "PhysXGRPOTrainer",
]
