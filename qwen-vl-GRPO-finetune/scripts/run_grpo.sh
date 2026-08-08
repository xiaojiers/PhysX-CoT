#!/bin/bash
#                                                                
#  PhysX V3 GRPO       
#                                                                
#
#         
#   1. SFT V2 adapter     physx-cot-sft/        
#        - adapter_config.json / adapter_model.safetensors   LoRA + modules_to_save 
#        - sam_projector.pt                                  SAM    MLP 
#        - merger_weights.pt                                 visual.merger     
#            "SFT    45 special token   resize      adapter  
#      merge_and_unload      merger / sam_projector      hook     GRPO LoRA" 
#                trainer/model_setup.py      
#   2.      build_grpo_dataset.py    V3 GT 
#        python datasets/build_grpo_dataset.py \
#            --sft_jsonl  ../data/cot_finetune_v3/training_set_0_cot_v3_filter.jsonl \
#            --image_root ../data/renders_cond \
#            --voxel_root ../data/tmp/partseg \
#            --out_dir    ./datasets
#          jsonl     image / sam_feature / messages / gt / meta     
#   3. dataset_root    data/ reward        voxel_dir   sam_feature      
#   4.   --dry_run                 --dry_run     
#
#    
#   bash scripts/run_grpo.sh           #     
#   bash scripts/run_grpo.sh --dry     #        

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#       dry run                                                                
DRY_RUN=false
if [[ "$1" == "--dry" ]]; then
    DRY_RUN=true
    echo "[INFO] Dry-run                 "
fi

#           SFT      2 GPU                                           
NUM_GPUS=4
MASTER_PORT=$((RANDOM % 1000 + 29000))
SEED=${PHYSX_COT_SEED:-117}

#                                                                            
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="$(cd "${GRPO_ROOT}/.." && pwd)"

# Qwen3-VL-8B base model reported in the Technical Appendix.
llm=${PHYSX_COT_BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}

# SFT V2 adapter   modules_to_save   embed_tokens/lm_head     
#    merger_weights.pt + sam_projector.pt 
sft_adapter=${PHYSX_COT_SFT_ADAPTER:-${BASE_DIR}/outputs/physx-cot-sft}

# GRPO       datasets/build_grpo_dataset.py    
train_file=${PHYSX_COT_GRPO_TRAIN:-${GRPO_ROOT}/datasets/grpo_train.jsonl}
eval_file=${PHYSX_COT_GRPO_EVAL:-${GRPO_ROOT}/datasets/grpo_val.jsonl}

#         SFT        
image_root=${PHYSX_COT_IMAGE_ROOT:-${BASE_DIR}/data/renders}
# data/     sam_feature   voxel_dir         
dataset_root=${PHYSX_COT_DATA_ROOT:-${BASE_DIR}/data}

# GRPO     
output_dir=${PHYSX_COT_GRPO_OUTPUT:-${BASE_DIR}/outputs/physx-cot-grpo}

#                                                                            
DRY_FLAG=""
if $DRY_RUN; then
    DRY_FLAG="--dry_run"
fi

torchrun --nproc_per_node=${NUM_GPUS} \
         --master_port=${MASTER_PORT} \
         "${GRPO_ROOT}/train_grpo.py" \
    \
    `#                                                         ` \
    --model_name_or_path   "${llm}" \
    --sft_adapter_path     "${sft_adapter}" \
    --torch_dtype          bfloat16 \
    --attn_implementation  flash_attention_2 \
    --use_lora \
    --lora_r               16 \
    --lora_alpha           32 \
    --lora_dropout         0.05 \
    --freeze_vision_tower \
    --gradient_checkpointing \
    --local_files_only \
    \
    `#    SAM                                           ` \
    `# merger_weights_path / sam_projector_path     sft_adapter_path    ` \
    `#           --merger_weights_path /path/to/merger_weights.pt` \
    --sam_proj_hidden        512 \
    --sam_in_dim             256 \
    --sam_cache_size         512 \
    --freeze_sam_projector \
    --freeze_merger \
    --freeze_embed_tokens \
    --freeze_lm_head \
    \
    `#                                                         ` \
    --train_file           "${train_file}" \
    --eval_file            "${eval_file}" \
    --image_root           "${image_root}" \
    --dataset_root         "${dataset_root}" \
    \
    `#                                                         ` \
    --output_dir           "${output_dir}" \
    --seed                 "${SEED}" \
    --report_to            tensorboard \
    --log_level            INFO \
    --save_code_snapshot \
    --enable_component_logging \
    \
    `#    GRPO                                              ` \
    --learning_rate              1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs           1.0 \
    --num_generations            4 \
    --max_prompt_length          2048 \
    `# Turn1+Turn2    p95 1.05   128 6528       datasets/stats_sft_completion_tokens.py ` \
    --max_completion_length      6528 \
    --beta                       0.02 \
    --temperature                0.8 \
    --top_p                      0.9 \
    --max_grad_norm              1.0 \
    --num_iterations             1 \
    --bf16 \
    \
    `#       /                                              ` \
    --logging_steps              5 \
    --save_steps                 500 \
    --eval_steps                 500 \
    \
    `#           V3                                     ` \
    --loc_weight                 0.25 \
    --coarse_weight              0.20 \
    --detail_weight              0.40 \
    --phys_weight                0.15 \
    --pen_weight                 0.30 \
    \
    `#    R_loc                                                ` \
    --loc_count_weight           0.20 \
    --loc_bbox2d_weight          0.30 \
    --loc_bbox3d_weight          0.50 \
    --bbox3d_iou_beta            0.7 \
    --bbox3d_l1_eta              0.15 \
    --count_decay_gamma          0.7 \
    \
    `#    R_coarse                                             ` \
    --coarse_shape_weight        0.5 \
    --coarse_axis_weight         0.2 \
    --coarse_ratio_weight        0.3 \
    \
    `#    R_detail                                        ` \
    --detail_parse_weight        0.10 \
    --detail_range_weight        0.15 \
    --detail_local_weight        0.50 \
    --detail_global_weight       0.25 \
    --detail_local_iou_w         0.6 \
    --detail_local_f1_w          0.4 \
    \
    `#    R_phys                                               ` \
    --phys_overall_weight        0.2 \
    --phys_group_weight          0.5 \
    --phys_logic_weight          0.3 \
    \
    `#    2D-3D       penalty      R_loc            ` \
    --cons_enable \
    --cons_weight                0.30 \
    --cons_center_weight         0.30 \
    --cons_proj_weight           0.70 \
    --cons_proj_ratio_w          0.5 \
    --cons_proj_shape_w          0.5 \
    --cons_proj_softmin_tau      0.5 \
    \
    `#    Penalty                                              ` \
    --pen_format_weight          0.30 \
    --pen_overflow_weight        0.30 \
    --pen_dependency_weight      0.40 \
    \
    `#       GT                                              ` \
    --voxel_grid_size            32 \
    --voxel_cache_size           256 \
    \
    ${DRY_FLAG}
