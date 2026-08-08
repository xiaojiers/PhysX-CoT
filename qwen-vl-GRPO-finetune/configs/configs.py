"""
configs/configs.py

            dataclass    

     
  ModelConfig       / LoRA /     
  DataConfig        /            GT     SAM      
  ScriptConfig                   dry-run
  RewardConfig             md/reward.md   R_loc / R_coarse /
                 R_detail / R_phys / P         2D-3D       
  TaskConfig     think / overall / geometry_l_k     prompt    
                   SFT (5generate_cot_finetune2.py)      
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


#                                                                             

@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    processor_name_or_path: Optional[str] = None
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    trust_remote_code: bool = True

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    freeze_vision_tower: bool = True
    gradient_checkpointing: bool = True


#                                                                             

@dataclass
class DataConfig:
    train_file: str = ""
    eval_file: str = ""

    #       renders_cond/ 
    image_root: Optional[str] = None
    # data/        sam_feature   voxel_dir         
    dataset_root: Optional[str] = None

    dataset_format: str = "jsonl"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    prompt_style: str = "default"
    expect_ground_truth: bool = True


#        /                                                                 

@dataclass
class ScriptConfig:
    output_dir: str = "./outputs/grpo"
    seed: int = 117
    do_train: bool = True
    do_eval: bool = False
    log_level: str = "INFO"
    report_to: str = "tensorboard"
    save_code_snapshot: bool = False
    dry_run: bool = False


#                                                                             
#
#           md/reward.md  
#       R =  _loc   R_loc'  +   _coarse   R_coarse
#         +  _detail   R_detail
#         +  _phys   R_phys  -   _pen   P
#
#       R_loc' = R_loc -  _cons   L_cons      2D-3D          R_loc   
#
#        
#   1. SFT     overall_information name/category/material/group  
#         phys_weight         coarse   detail     
#                    
#   2.            reward.md              
#   3.    reward        [0, 1] penalty    [0, 1]         
#   4.       GT         completion    geometry_l_k  
#                             advantage 

@dataclass
class RewardConfig:
    #        ----------------------------------------------------------
    loc_weight:    float = 0.25   #  _loc
    coarse_weight: float = 0.20   #  _coarse
    detail_weight: float = 0.40   #  _detail           
    phys_weight:   float = 0.15   #  _phys    SFT     
    pen_weight:    float = 0.30   #  _pen

    #    R_loc         / bbox_2d / bbox_3d                  
    loc_count_weight:  float = 0.20
    loc_bbox2d_weight: float = 0.30
    loc_bbox3d_weight: float = 0.50
    count_decay_gamma: float = 0.7    # exp(- |n -n|)
    bbox3d_iou_beta:   float = 0.7    #   IoU3d + (1- ) exp(-  || || )
    bbox3d_l1_eta:     float = 0.15

    #    R_coarse    primitive                                
    coarse_shape_weight: float = 0.5
    coarse_axis_weight:  float = 0.2
    coarse_ratio_weight: float = 0.3

    #    R_detail               RL                  
    detail_parse_weight:   float = 0.10
    detail_range_weight:   float = 0.15
    detail_local_weight:   float = 0.50
    detail_global_weight:  float = 0.25
    detail_local_iou_w:    float = 0.6   # local: 0.6 IoU + 0.4 F1
    detail_local_f1_w:     float = 0.4

    #    R_phys    overall                                  
    phys_overall_weight: float = 0.2     # <overall>     
    phys_group_weight:   float = 0.5     # group_info    
    phys_logic_weight:   float = 0.3     #      

    #    2D-3D          R_loc                             
    cons_enable:           bool  = True
    cons_weight:           float = 0.30   #  _cons   R_loc       
    cons_center_weight:    float = 0.30   #  _c
    cons_proj_weight:      float = 0.70   #  _p
    cons_proj_ratio_w:     float = 0.5
    cons_proj_shape_w:     float = 0.5
    cons_proj_softmin_tau: float = 0.5

    #    Penalty                                                    
    pen_format_weight:     float = 0.30
    pen_overflow_weight:   float = 0.30
    pen_dependency_weight: float = 0.40

    #       GT                                                    
    voxel_grid_size: int  = 32
    voxel_cache_size: int = 256          # LRU         obj_id   

    #                                                               
    enable_component_logging: bool = True
    final_clamp_min: float = -2.0
    final_clamp_max: float = 2.0


#       / Prompt     SFT                                           

@dataclass
class TaskConfig:
    #    ----------------------------------------------------------------
    think_open_tag:  str = "<think>"
    think_close_tag: str = "</think>"

    overall_open_tag:  str = "<overall>"
    overall_close_tag: str = "</overall>"

    #     GRPO   
    final_open_tag:  str = "<final>"
    final_close_tag: str = "</final>"

    # geometry_l_k supports part IDs in [0, 23], matching the appendix.

    # Step 3      
    valid_directions: Tuple[str, ...] = (
        "top", "bottom", "left", "right", "front", "back", "center",
    )

    #    Prompt      data/5generate_cot_finetune2.py       
    overall_prompt: str = (
        "Analyze the 3D physical object in the image and output its complete "
        "physical asset description.\n\n"
        "First, reason step by step inside <think>:\n"
        "Step 1: Count the total number of independent structural parts (`part_count`).\n"
        "Step 2: For each part, record its 2D image bounding range "
        "`bbox_2d` = [x_min, x_max, y_min, y_max] (normalized 0~1), "
        "its 3D voxel bounding range `bbox_3d` = "
        "[x_min, x_max, y_min, y_max, z_min, z_max] in the canonical "
        "32 32 32 voxel space (both use the same min/max vertex format), "
        "and the SAM visual feature token `sam_feat` = <sam_feat_l_{part_id}> "
        "which encodes the region appearance from the SAM3 encoder.\n"
        "Step 3: For each part, describe the relative 3D position of its "
        "directly adjacent parts using discrete direction labels "
        "(top/bottom/left/right/front/back/center). Non-adjacent parts are not recorded.\n"
        "Step 4: Identify each part's dominant geometric primitive "
        "(`shape_label`: cuboid/cylinder/sphere/complex), its major axis orientation "
        "(`major_axis`: x/y/z), and its aspect ratio (`aspect_ratio`: "
        "very_flat/flat/balanced/tall/elongated).\n"
        "Step 5: Assess each part's surface perceptual properties: `hardness` "
        "(soft/semi_rigid/rigid), `roughness` (smooth/textured/rough), `reflectivity` "
        "(matte/glossy/highly_reflective), and `transparency` (opaque/translucent/transparent).\n\n"
        "Then output the structured physical description inside <overall>."
    )

    geometry_prompt: str = (
        "Based on the `bbox_3d` of `l_{part_id}` from Step 2, generate its local 3D "
        "voxel occupancy as a 1D run-length encoded sequence. "
        "Encoding: local_id = (x-x_min)*(dy*dz) + (y-y_min)*dz + (z-z_min), "
        "where dy=y_max-y_min+1, dz=z_max-z_min+1 (derived from bbox_3d). "
        "Merge maximal consecutive runs (e.g. 0 1-5 36-41 ...). "
        "Wrap the result in <geometry_l_{part_id}>.</geometry_l_{part_id}>."
    )

    #       fallback   data_loader   row   messages    
    system_prompt: str = ""
    user_prompt_template: str = ""  #   ExampleAdapter    overall_prompt


#       user_prompt_template      overall_prompt
#  ExampleAdapter fallback           
def _post_init_task(cfg: TaskConfig) -> None:
    if not cfg.user_prompt_template:
        cfg.user_prompt_template = cfg.overall_prompt
