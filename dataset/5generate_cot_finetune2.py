"""
CoT            V3 (5generate_cot_finetune2.py)

   cot_tmp_v3/    V3 CoT think    txt_rep_32_finetune/        
  tmp/partseg/           2        JSON     

          = 2    
    Turn 1  Human : <image>\n{overall_cot_prompt_v3}
            GPT   : <think>{V3 5    }</think>
                    <overall>{txt_rep      }</overall>

    Turn 2  Human : {geometry_prompt}    {part_id} 
            GPT   : <geometry_l_{k}>{   1D         }</geometry_l_{k}>
                            CoT Step 2 bbox_3d   min      
                      local_id = x'*dy*dz + y'*dz + z'    [0, dx*dy*dz-1] 

   :
    cot_tmp_v3/{id}_{img_id}.txt             V3 CoT think      <sam_feat_l_k>     
    txt_rep_32_finetune/{id}.txt                   GT Overall + Parts + Group_info 
    tmp/partseg/{id}/32/ind_{k}.npy            32      
    renders_cond/{id}_/{img_id}.png                    JSON       
    sam_feature/{id}/{img_id}.npz           SAM         {part_id: [256] float32} 
           
    cot_finetune_v3/training_set_{ind}_cot_v3.json / .jsonl
    cot_finetune_v3/sample_cot_v3.json        pretty-print          

           "sam_feature"    str   null       npz        
      collator            <sam_feat_l_k>             embedding 

    :
    python3 5generate_cot_finetune2.py
    python3 5generate_cot_finetune2.py --ind 0 --range 500
    python3 5generate_cot_finetune2.py --n_samples 5   #    sample     
"""
from __future__ import annotations

import os
import json
import argparse
import logging

import numpy as np


#                                                                             

def _get_logger(filename: str, verbosity: int = 1) -> logging.Logger:
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    fmt = logging.Formatter(
        "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(filename)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


#         RLE             CoT Step 2   bbox_3d                
#
#            bbox   min        
#   local_id = x' * dy * dz  +  y' * dz  +  z'
#   x' = x - x_min,  y' = y - y_min,  z' = z - z_min
#   dy = y_max - y_min + 1,  dz = z_max - z_min + 1
#
#    [0, dx*dy*dz - 1]              [0, 32767] 
#          RLE       token            
#         =      + bbox_min CoT Step 2      

def _voxel_to_rle(indices: np.ndarray) -> str:
    """(N, 3) int32                  """
    v = np.asarray(indices, dtype=np.int64)
    x_min, y_min, z_min = int(v[:, 0].min()), int(v[:, 1].min()), int(v[:, 2].min())
    x_max, y_max, z_max = int(v[:, 0].max()), int(v[:, 1].max()), int(v[:, 2].max())
    dy = y_max - y_min + 1
    dz = z_max - z_min + 1

    local_ids = (
        (v[:, 0] - x_min) * dy * dz
        + (v[:, 1] - y_min) * dz
        + (v[:, 2] - z_min)
    )
    ids = sorted(set(local_ids.tolist()))

    result: list[str] = []
    start = prev = ids[0]
    for n in ids[1:]:
        if n == prev + 1:
            prev = n
        else:
            result.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    result.append(f"{start}-{prev}" if start != prev else str(start))
    return " ".join(result)


#       Prompt     prompt                                       

_DEFAULT_OVERALL_PROMPT = (
    "Analyze the 3D physical object in the image and output its complete physical asset description.\n\n"
    "First, reason step by step inside <think>:\n"
    "Step 1: Count the total number of independent structural parts (`part_count`).\n"
    "Step 2: For each part, record its 2D image bounding range `bbox_2d` = "
    "[x_min, x_max, y_min, y_max] (normalized 0~1), its 3D voxel bounding range "
    "`bbox_3d` = [x_min, x_max, y_min, y_max, z_min, z_max] in the canonical "
    "32 32 32 voxel space (both use the same min/max vertex format), "
    "and the SAM visual feature token `sam_feat` = <sam_feat_l_{part_id}> "
    "which encodes the region appearance from the SAM3 encoder.\n"
    "Step 3: For each part, describe the relative 3D position of its directly adjacent parts "
    "using discrete direction labels (top/bottom/left/right/front/back/center). "
    "Non-adjacent parts are not recorded.\n"
    "Step 4: Identify each part's dominant geometric primitive (`shape_label`: "
    "cuboid/cylinder/sphere/complex), its major axis orientation (`major_axis`: x/y/z), "
    "and its aspect ratio (`aspect_ratio`: very_flat/flat/balanced/tall/elongated).\n"
    "Step 5: Assess each part's surface perceptual properties: `hardness` "
    "(soft/semi_rigid/rigid), `roughness` (smooth/textured/rough), `reflectivity` "
    "(matte/glossy/highly_reflective), and `transparency` (opaque/translucent/transparent).\n\n"
    "Then output the structured physical description inside <overall>."
)

_DEFAULT_GEOMETRY_PROMPT = (
    "Based on the `bbox_3d` of `l_{part_id}` from Step 2, generate its local 3D voxel "
    "occupancy as a 1D run-length encoded sequence. "
    "Encoding: local_id = (x-x_min)*(dy*dz) + (y-y_min)*dz + (z-z_min), "
    "where dy=y_max-y_min+1, dz=z_max-z_min+1 (derived from bbox_3d). "
    "Merge maximal consecutive runs (e.g. 0 1-5 36-41 ...). "
    "Wrap the result in <geometry_l_{part_id}>.</geometry_l_{part_id}>."
)


def _load_prompt(path: str, default: str) -> str:
    """     prompt                   """
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return default


#                                                                          

def _build_sample(
    obj_id:           str,
    img_id:           str,
    part_k:           int,
    think_text:       str,
    txt_rep:          str,
    voxel_rle:        str,
    overall_prompt:   str,
    geometry_prompt:  str,
    sam_feature_path: str | None = None,
) -> dict:
    """
         2         

    Turn 1 GPT = <think>{CoT V3}</think>\n<overall>{txt_rep}</overall>
    Turn 2 Human = geometry_prompt     {part_id} 
    Turn 2 GPT   = <geometry_l_{k}>{voxel_rle}</geometry_l_{k}>

    sam_feature_path :    sam_feature/{obj_id}/{img_id}.npz          None  
           collator         SAM       <sam_feat_l_k>       
    """
    turn1_answer   = f"<think>\n{think_text}\n</think>\n<overall>\n{txt_rep}\n</overall>"
    turn2_question = geometry_prompt.replace('{part_id}', str(part_k))
    turn2_answer   = f"<geometry_l_{part_k}>\n{voxel_rle}\n</geometry_l_{part_k}>"

    return {
        'id':          f'{obj_id}_{img_id}',
        'image':       os.path.join(f'{obj_id}_', f'{img_id}.png'),
        'sam_feature': sam_feature_path,
        'conversations': [
            {'from': 'human', 'value': f'<image>\n{overall_prompt}'},
            {'from': 'gpt',   'value': turn1_answer},
            {'from': 'human', 'value': turn2_question},
            {'from': 'gpt',   'value': turn2_answer},
        ],
        'data_source': 'physx_cot_v3',
    }


#       object                                                              

def process_object(
    obj_id:          str,
    cot_dir:         str,
    txt_dir:         str,
    voxel_dir:       str,
    overall_prompt:  str,
    geometry_prompt: str,
    logger:          logging.Logger,
    sam_dir:         str = '',
    max_views:       int = 0,
) -> list[dict]:
    """
         object      (part   view)        

    Args:
        sam_dir   : sam_feature/           npz          sam_feature    
        max_views :          0          
    """
    txt_path    = os.path.join(txt_dir,   obj_id + '.txt')
    obj_vox_dir = os.path.join(voxel_dir, obj_id, '32')

    #            GT                                                       
    if not os.path.exists(txt_path):
        logger.warning(f'{obj_id}: txt_rep not found, skipping.')
        return []
    with open(txt_path, 'r', encoding='utf-8') as f:
        txt_rep = f.read().strip()

    #        object     V3 CoT                                          
    prefix    = obj_id + '_'
    cot_files = sorted(
        fn for fn in os.listdir(cot_dir)
        if fn.startswith(prefix) and fn.endswith('.txt')
    )
    if max_views > 0:
        cot_files = cot_files[:max_views]
    if not cot_files:
        logger.warning(f'{obj_id}: no cot_tmp_v3 files found, skipping.')
        return []

    #                    ind_{k}.npy                         
    n_parts = 0
    while os.path.exists(os.path.join(obj_vox_dir, f'ind_{n_parts}.npy')):
        n_parts += 1
    if n_parts == 0:
        logger.warning(f'{obj_id}: no voxel data found, skipping.')
        return []

    #             RLE                                         
    voxel_rles: list[str] = []
    for k in range(n_parts):
        npy_p = os.path.join(obj_vox_dir, f'ind_{k}.npy')
        vd    = np.load(npy_p).astype(np.int64)
        voxel_rles.append(_voxel_to_rle(vd))

    #      (view   part)                                                   
    samples: list[dict] = []

    for cot_fname in cot_files:
        stem   = cot_fname[:-4]           #   .txt
        img_id = stem[len(prefix):]       # e.g. "000"

        with open(os.path.join(cot_dir, cot_fname), 'r', encoding='utf-8') as f:
            raw = f.read()

        #    <think>...</think>          
        if '<think>' in raw and '</think>' in raw:
            start      = raw.index('<think>') + len('<think>')
            end        = raw.index('</think>')
            think_text = raw[start:end].strip()
        else:
            think_text = raw.strip()

        # SAM    npz       data/         collator    
        sam_feature_path: str | None = None
        if sam_dir:
            npz_abs = os.path.join(sam_dir, obj_id, f'{img_id}.npz')
            if os.path.exists(npz_abs):
                sam_feature_path = os.path.join('sam_feature', obj_id, f'{img_id}.npz')
            else:
                logger.warning(f'{obj_id}/{img_id}: sam_feature npz not found, sam_feature=null.')

        for part_k in range(n_parts):
            samples.append(_build_sample(
                obj_id           = obj_id,
                img_id           = img_id,
                part_k           = part_k,
                think_text       = think_text,
                txt_rep          = txt_rep,
                voxel_rle        = voxel_rles[part_k],
                overall_prompt   = overall_prompt,
                geometry_prompt  = geometry_prompt,
                sam_feature_path = sam_feature_path,
            ))

    return samples


#                                                                          

def _write_sample(
    alldata:    list[dict],
    output_dir: str,
    n_samples:  int,
    logger:     logging.Logger,
) -> None:
    """
      alldata     n_samples     pretty-print JSON       
              0          
    """
    if not alldata:
        logger.warning('alldata is empty, skipping sample output.')
        return

    samples = alldata[:n_samples]
    sample_path = os.path.join(output_dir, 'sample_cot_v3.json')
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info(f'Sample ({len(samples)} entries)   {sample_path}')

    #     0    
    s     = samples[0]
    convs = s['conversations']
    logger.info(
        '\n     0         \n'
        '  id              : %s\n'
        '  image           : %s\n'
        '  Turn1 Human     : %s  \n'
        '  Turn1 GPT       : %s  \n'
        '  Turn2 Human     : %s  \n'
        '  Turn2 GPT       : %s  ',
        s['id'],
        s['image'],
        convs[0]['value'][:80].replace('\n', ' '),
        convs[1]['value'][:150].replace('\n', ' '),
        convs[2]['value'][:80].replace('\n', ' '),
        convs[3]['value'][:80].replace('\n', ' '),
    )


#                                                                            

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate CoT V3 finetune training data for PhysX-CoT'
    )
    parser.add_argument('--ind',       type=int, default=0,
                        help='Worker index for data sharding')
    parser.add_argument('--range',     type=int, default=-1,
                        help='Number of objects per worker; -1 means all')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of entries written to sample_cot_v3.json (default 3)')
    args = parser.parse_args()

    #                                                                       
    cot_dir         = './cot_tmp_v3'
    txt_dir         = './txt_rep_32_finetune'
    voxel_dir       = './tmp/partseg'
    render_dir      = './renders_cond'
    sam_dir         = './sam_feature'
    output_dir      = './cot_finetune_v3'
    prompt_path     = './overall_cot_prompt_v3.txt'
    geo_prompt_path = './geometry_prompt.txt'

    os.makedirs(output_dir, exist_ok=True)
    logger = _get_logger(f'./tmp/cot_finetune_v3_{args.ind}.log')
    logger.info('CoT V3 finetune data generation started.')

    overall_prompt  = _load_prompt(prompt_path,     _DEFAULT_OVERALL_PROMPT)
    geometry_prompt = _load_prompt(geo_prompt_path, _DEFAULT_GEOMETRY_PROMPT)

    #       renders_cond/       object       {id}_               
    all_ids = sorted(
        d[:-1]
        for d in os.listdir(render_dir)
        if d.endswith('_') and os.path.isdir(os.path.join(render_dir, d))
    )
    if args.range != -1:
        all_ids = all_ids[args.ind * args.range: (args.ind + 1) * args.range]
    logger.info(f'Processing {len(all_ids)} objects (worker {args.ind}).')

    alldata: list[dict] = []
    for obj_id in all_ids:
        try:
            samples = process_object(
                obj_id          = obj_id,
                cot_dir         = cot_dir,
                txt_dir         = txt_dir,
                voxel_dir       = voxel_dir,
                overall_prompt  = overall_prompt,
                geometry_prompt = geometry_prompt,
                logger          = logger,
                sam_dir         = sam_dir,
            )
            alldata.extend(samples)
            if samples:
                logger.info(f'{obj_id}: {len(samples)} samples generated.')
        except Exception as e:
            logger.warning(f'{obj_id}: FAILED   {e}')

    #       JSON     + JSONL                                        
    json_path  = os.path.join(output_dir, f'training_set_{args.ind}_cot_v3.json')
    jsonl_path = os.path.join(output_dir, f'training_set_{args.ind}_cot_v3.jsonl')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(alldata, f, ensure_ascii=False, indent=2)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in alldata:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    logger.info(f'Saved {len(alldata)} samples   {json_path} / {jsonl_path}')

    #       sample                                                        
    _write_sample(alldata, output_dir, args.n_samples, logger)


if __name__ == '__main__':
    main()
