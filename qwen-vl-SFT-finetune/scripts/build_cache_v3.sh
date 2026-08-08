#!/bin/bash
#                                                                
#  V3            
#                                                                
#
#      ~108,000                   
#         CPU           ~17s/it    ~5s/it 
#
#              .pt     
#    input_ids       [L]          int64
#    attention_mask  [L]          int64
#    labels          [L]          int64   (<sam_feat>  =-100)
#    pixel_values    [N_patch, D] float16
#    image_grid_thw  [1, 3]       int64
#    sam_feats       [N_parts,256] float32  (None      )
#
#         ~1-2 MB/     108,000   108-216 GB
#         16      ~2s/   / 16   108,000 2/16   3.75   
#
#     
#    bash scripts/build_cache_v3.sh
#                                                                

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SFT_ROOT}/.." && pwd)"

llm=${PHYSX_COT_BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
annotation=${PHYSX_COT_TRAIN_JSONL:-${PROJECT_ROOT}/data/train.jsonl}
data_path=${PHYSX_COT_IMAGE_ROOT:-${PROJECT_ROOT}/data/renders}
sam_feature_dir=${PHYSX_COT_SAM_FEATURE_DIR:-${PROJECT_ROOT}/data/sam_features}
cache_dir=${PHYSX_COT_CACHE_DIR:-${PROJECT_ROOT}/data/cache}

python3 "${SFT_ROOT}/cache_dataset.py" \
    --model_path      "${llm}"            \
    --annotation_path "${annotation}"     \
    --data_path       "${data_path}"      \
    --sam_feature_dir "${sam_feature_dir}" \
    --cache_dir       "${cache_dir}"      \
    --max_length      6144                \
    --max_pixels      262144              \
    --min_pixels      65536               \
    --num_workers     16
