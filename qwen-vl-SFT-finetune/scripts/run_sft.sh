#!/bin/bash
#                                                                
#  PhysX CoT V3 LoRA         v2    embed_tokens/lm_head    
#                                                                
#
#  v2      physx-cot-sft/       
#    1. LoRA      modules_to_save=["embed_tokens","lm_head"] 
#          45   special token <think>/<overall>/<geometry_l_k>/
#       <sam_feat>     embedding   lm_head           
#       adapter        token     U+FFFD    
#    2. output_dir   physx-cot-sft/         checkpoint
#          resume   adapter           
#    3. bucket_batches   100    20         loss spike 
#    4. Standard checkpoints are saved every 500 steps.
#    5.     eval      CheckpointCompletionCallback   
#       processor/merger/sam_projector CheckpointEvalCallback   
#       fire-and-forget       
#
#       bfloat16 Flash Attention 2 gradient_checkpointing=True  
#    Appendix configuration: 8704 tokens, batch/device 2, accumulation 8.
#
#       
#      filter_dataset.py     training_set_0_cot_v3_filter.jsonl
#      callbacks.py     Completion / Snapshot / Eval 
#
#     
#    bash scripts/run_sft.sh
#                                                                

export OMP_NUM_THREADS=4
#    PyTorch                    OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=4
MASTER_PORT=$((RANDOM % 1000 + 29000))
SEED=${PHYSX_COT_SEED:-17}

#                                                           
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SFT_ROOT}/.." && pwd)"

llm=${PHYSX_COT_BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}

# V3         token>8192 outlier   126,000   
annotation=${PHYSX_COT_TRAIN_JSONL:-${PROJECT_ROOT}/data/train.jsonl}
eval_annotation=${PHYSX_COT_VAL_JSONL:-${PROJECT_ROOT}/data/val.jsonl}

#       renders_cond/ 
data_path=${PHYSX_COT_IMAGE_ROOT:-${PROJECT_ROOT}/data/renders}

# SAM       sam_feature/         
sam_feature_dir=${PHYSX_COT_SAM_FEATURE_DIR:-${PROJECT_ROOT}/data/sam_features}

#        LoRA adapter / merger / SAM projector    
#             physx-cot-sft/    adapter   modules_to_save
#   embed_tokens/lm_head         LoRA            
# PeftModel.from_pretrained   shape mismatch / key missing 
output_dir=${PHYSX_COT_SFT_OUTPUT:-${PROJECT_ROOT}/outputs/physx-cot-sft}

#    Checkpoint                                              
#            eval_image        
#   1) 2 A800      DDP 100%    eval worker     CPU
#      + FA2     CPU     eager Qwen3-VL-8B CPU     1~2h/  
#        worker      RAM 
#   2) V2             modules_to_save checkpoint     
#      periodic snapshot bucket         loss / grad_norm        
#   3)      GPU        ID + OOD      
#            eval_image          train_lora.py        
eval_image=
eval_device=cpu
eval_max_new_tokens=2048
eval_out_root=${PHYSX_COT_SFT_EVAL_OUTPUT:-${PROJECT_ROOT}/outputs/sft-evaluations}

#                    save_total_limit         
#   5000         {output_dir}/periodic_snapshots/checkpoint-XXXX/ 
#   inode        0      
periodic_save_steps=0

#    Length-bucket sampler                                      
#    100     25 optim step          loss spike 
# 20   DDP 2 + micro-batch 2     padding   loss      
bucket_batches=20

#                                                           
torchrun --nproc_per_node=${NUM_GPUS} \
         --master_port=${MASTER_PORT} \
         "${SFT_ROOT}/train_lora.py" \
    \
    --model_path            "${llm}"            \
    --annotation_path       "${annotation}"     \
    --eval_annotation_path  "${eval_annotation}" \
    --data_path             "${data_path}"      \
    --sam_feature_dir       "${sam_feature_dir}" \
    --sam_proj_hidden       512                 \
    \
    --lora_r                16                  \
    --lora_alpha            32                  \
    --lora_dropout          0.05                \
    --lora_target_modules   "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    \
    --output_dir            "${output_dir}"     \
    --num_train_epochs      2                   \
    --per_device_train_batch_size  2            \
    --gradient_accumulation_steps  8            \
    --learning_rate         2e-5                \
    --weight_decay          0.05                \
    --seed                  "${SEED}"            \
    --lr_scheduler_type     cosine              \
    --warmup_ratio          0.03                \
    \
    --bf16                                      \
    --max_length            8704                \
    --max_pixels            200704              \
    --min_pixels            200704              \
    --max_grad_norm         1.0                 \
    \
    --gradient_checkpointing        True        \
    --ddp_find_unused_parameters    True        \
    \
    --save_strategy         steps               \
    --save_steps            500                  \
    --save_total_limit      2                   \
    --logging_steps         5                   \
    --eval_strategy         steps               \
    --eval_steps            500                 \
    --report_to             none                \
    --dataloader_num_workers 8                  \
    \
    --eval_image            "${eval_image}"     \
    --eval_device           "${eval_device}"    \
    --eval_max_new_tokens   "${eval_max_new_tokens}" \
    --eval_out_root         "${eval_out_root}"  \
    \
    --periodic_save_steps   ${periodic_save_steps}  \
    --bucket_batches        ${bucket_batches}
