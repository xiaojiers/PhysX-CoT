"""
parsers/arg_parser.py

          configs/    dataclass        

      
       / LoRA /   
      
       /    
    GRPO     
               +    +     + penalty 
"""

from __future__ import annotations

import argparse

from configs import DataConfig, ModelConfig, RewardConfig, ScriptConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PhysX-CoT GRPO training for Qwen3-VL")

    #                                                                         
    parser.add_argument("--model_name_or_path", type=str, default=ModelConfig.model_name_or_path)
    parser.add_argument("--processor_name_or_path", type=str, default=None)
    parser.add_argument("--torch_dtype",  type=str, default=ModelConfig.torch_dtype)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--trust_remote_code",   action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora",  action="store_false", dest="use_lora")
    parser.add_argument("--lora_r",       type=int,   default=ModelConfig.lora_r)
    parser.add_argument("--lora_alpha",   type=int,   default=ModelConfig.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=ModelConfig.lora_dropout)
    parser.add_argument("--freeze_vision_tower",  action="store_true", default=ModelConfig.freeze_vision_tower)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=ModelConfig.gradient_checkpointing)
    parser.add_argument("--sft_adapter_path", type=str, default=None,
                        help="SFT LoRA adapter            SFT      GRPO LoRA")
    parser.add_argument("--local_files_only", action="store_true", default=False)

    #    SAM    /                                                        
    parser.add_argument("--merger_weights_path", type=str, default=None,
                        help="visual.merger    .pt      sft_adapter_path/merger_weights.pt")
    parser.add_argument("--sam_projector_path", type=str, default=None,
                        help="SAMProjector    .pt      sft_adapter_path/sam_projector.pt")
    parser.add_argument("--sam_proj_hidden", type=int, default=512,
                        help="SAMProjector MLP         SFT    ")
    parser.add_argument("--sam_in_dim", type=int, default=256,
                        help="SAM3 RoI       ")
    parser.add_argument("--sam_cache_size", type=int, default=512,
                        help="SAMFeatureLoader LRU         npz     ")
    parser.add_argument("--freeze_sam_projector",  action="store_true", default=True)
    parser.add_argument("--no_freeze_sam_projector", action="store_false",
                        dest="freeze_sam_projector")
    parser.add_argument("--freeze_merger",        action="store_true", default=True)
    parser.add_argument("--no_freeze_merger",     action="store_false", dest="freeze_merger")
    parser.add_argument("--freeze_embed_tokens",  action="store_true", default=True)
    parser.add_argument("--no_freeze_embed_tokens", action="store_false",
                        dest="freeze_embed_tokens")
    parser.add_argument("--freeze_lm_head",       action="store_true", default=True)
    parser.add_argument("--no_freeze_lm_head",    action="store_false", dest="freeze_lm_head")

    #                                                                         
    parser.add_argument("--train_file",   type=str, required=True)
    parser.add_argument("--eval_file",    type=str, default="")
    parser.add_argument("--image_root",   type=str, default=None,
                        help="      renders_cond/ ")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="data/     sam_feature   voxel_dir         ")
    parser.add_argument("--dataset_format", type=str, default="jsonl", choices=["jsonl", "hf"])
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples",  type=int, default=None)
    parser.add_argument("--prompt_style",      type=str, default="default")
    parser.add_argument("--expect_ground_truth", action="store_true", default=True)

    #       /                                                              
    parser.add_argument("--output_dir", type=str, default=ScriptConfig.output_dir)
    parser.add_argument("--seed",       type=int, default=ScriptConfig.seed)
    parser.add_argument("--do_train",   action="store_true", default=True)
    parser.add_argument("--do_eval",    action="store_true", default=False)
    parser.add_argument("--report_to",  type=str, default=ScriptConfig.report_to)
    parser.add_argument("--log_level",  type=str, default=ScriptConfig.log_level)
    parser.add_argument("--save_code_snapshot", action="store_true", default=False)
    parser.add_argument("--dry_run",    action="store_true", default=False)

    #    GRPO                                                               
    parser.add_argument("--learning_rate",                 type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size",   type=int,   default=2)
    parser.add_argument("--gradient_accumulation_steps",   type=int,   default=4)
    parser.add_argument("--num_train_epochs",              type=float, default=1.0)
    parser.add_argument("--max_steps",                     type=int,   default=-1)
    parser.add_argument("--logging_steps",                 type=int,   default=10)
    parser.add_argument("--save_steps",                    type=int,   default=500)
    parser.add_argument("--eval_steps",                    type=int,   default=500)
    parser.add_argument("--num_generations",               type=int,   default=4)
    parser.add_argument("--max_prompt_length",             type=int,   default=2048)
    parser.add_argument("--max_completion_length",         type=int,   default=6528)
    parser.add_argument("--beta",                          type=float, default=0.02)
    parser.add_argument("--temperature",                   type=float, default=0.8)
    parser.add_argument("--top_p",                         type=float, default=0.9)
    parser.add_argument("--max_grad_norm",                 type=float, default=1.0)
    parser.add_argument("--num_iterations",                type=int,   default=1)
    parser.add_argument("--bf16",                          action="store_true", default=True)
    parser.add_argument("--fp16",                          action="store_true", default=False)

    #                                                                    
    parser.add_argument("--loc_weight",    type=float, default=RewardConfig.loc_weight)
    parser.add_argument("--coarse_weight", type=float, default=RewardConfig.coarse_weight)
    parser.add_argument("--detail_weight", type=float, default=RewardConfig.detail_weight)
    parser.add_argument("--phys_weight",   type=float, default=RewardConfig.phys_weight)
    parser.add_argument("--pen_weight",    type=float, default=RewardConfig.pen_weight)

    #         R_loc                                                      
    parser.add_argument("--loc_count_weight",  type=float, default=RewardConfig.loc_count_weight)
    parser.add_argument("--loc_bbox2d_weight", type=float, default=RewardConfig.loc_bbox2d_weight)
    parser.add_argument("--loc_bbox3d_weight", type=float, default=RewardConfig.loc_bbox3d_weight)
    parser.add_argument("--count_decay_gamma", type=float, default=RewardConfig.count_decay_gamma)
    parser.add_argument("--bbox3d_iou_beta",   type=float, default=RewardConfig.bbox3d_iou_beta)
    parser.add_argument("--bbox3d_l1_eta",     type=float, default=RewardConfig.bbox3d_l1_eta)

    #         R_coarse                                                   
    parser.add_argument("--coarse_shape_weight", type=float, default=RewardConfig.coarse_shape_weight)
    parser.add_argument("--coarse_axis_weight",  type=float, default=RewardConfig.coarse_axis_weight)
    parser.add_argument("--coarse_ratio_weight", type=float, default=RewardConfig.coarse_ratio_weight)

    #         R_detail                                                   
    parser.add_argument("--detail_parse_weight",  type=float, default=RewardConfig.detail_parse_weight)
    parser.add_argument("--detail_range_weight",  type=float, default=RewardConfig.detail_range_weight)
    parser.add_argument("--detail_local_weight",  type=float, default=RewardConfig.detail_local_weight)
    parser.add_argument("--detail_global_weight", type=float, default=RewardConfig.detail_global_weight)
    parser.add_argument("--detail_local_iou_w",   type=float, default=RewardConfig.detail_local_iou_w)
    parser.add_argument("--detail_local_f1_w",    type=float, default=RewardConfig.detail_local_f1_w)

    #         R_phys                                                     
    parser.add_argument("--phys_overall_weight", type=float, default=RewardConfig.phys_overall_weight)
    parser.add_argument("--phys_group_weight",   type=float, default=RewardConfig.phys_group_weight)
    parser.add_argument("--phys_logic_weight",   type=float, default=RewardConfig.phys_logic_weight)

    #    2D-3D                                                             
    parser.add_argument("--cons_enable",        action="store_true", default=RewardConfig.cons_enable)
    parser.add_argument("--no_cons_enable",     action="store_false", dest="cons_enable")
    parser.add_argument("--cons_weight",        type=float, default=RewardConfig.cons_weight)
    parser.add_argument("--cons_center_weight", type=float, default=RewardConfig.cons_center_weight)
    parser.add_argument("--cons_proj_weight",   type=float, default=RewardConfig.cons_proj_weight)
    parser.add_argument("--cons_proj_ratio_w",  type=float, default=RewardConfig.cons_proj_ratio_w)
    parser.add_argument("--cons_proj_shape_w",  type=float, default=RewardConfig.cons_proj_shape_w)
    parser.add_argument("--cons_proj_softmin_tau", type=float,
                        default=RewardConfig.cons_proj_softmin_tau)

    #    Penalty                                                              
    parser.add_argument("--pen_format_weight",     type=float, default=RewardConfig.pen_format_weight)
    parser.add_argument("--pen_overflow_weight",   type=float, default=RewardConfig.pen_overflow_weight)
    parser.add_argument("--pen_dependency_weight", type=float, default=RewardConfig.pen_dependency_weight)

    #       GT                                                              
    parser.add_argument("--voxel_grid_size",  type=int, default=RewardConfig.voxel_grid_size)
    parser.add_argument("--voxel_cache_size", type=int, default=RewardConfig.voxel_cache_size)

    #                                                                         
    parser.add_argument("--enable_component_logging", action="store_true",
                        default=RewardConfig.enable_component_logging)
    parser.add_argument("--final_clamp_min", type=float, default=RewardConfig.final_clamp_min)
    parser.add_argument("--final_clamp_max", type=float, default=RewardConfig.final_clamp_max)

    return parser
