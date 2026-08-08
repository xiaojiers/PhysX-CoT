"""
CoT think        V3 (4generate_cot_data2.py)

   CoT3             <think>...</think> CoT    
    cot_tmp_v3/{obj_id}_{img_id}.txt 

   V2       
    Dim 2 :    bbox_3d   32           3D     
    Dim 3 :                26-        3D      
                      
    Dim 4 : primitive     shape_label + major_axis + aspect_ratio
            IoU < 0.3     'complex' CoT3         cuboid

   :
    tmp/finaljson/{id}.json                    
    tmp/partseg/{id}/32/ind_{k}.npy          32      
    tmp/partseg/{id}/32/mesh_new_{k}.ply          
    renders_cond/{id}_/transforms.json          
           
    cot_tmp_v3/{id}_{img_id}.txt          CoT think    V3    

    :
    python 4generate_cot_data2.py
    python 4generate_cot_data2.py --ind 0 --range 500
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import logging

import numpy as np

#       V2 cot    bbox_2d / surface_features /                        
_V2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v2')
sys.path.insert(0, _V2_DIR)

from cot.dim23_spatial import (          # noqa: E402
    load_transforms_meta,
    _build_c2w,
    compute_all_bboxes,
    compute_adjacency,
)
from cot.dim4_primitive import (         # noqa: E402
    GRID_SIZE,
    indices_to_grid,
    rasterize_cuboid,
    rasterize_cylinder,
    rasterize_sphere,
)
from cot.dim5_surface import derive_surface_features  # noqa: E402


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


#    V3 Dim 2 bbox_2d xyxy & bbox_3d                                         

def _cxcywh_to_xyxy(bbox: list[float]) -> list[float]:
    """
      compute_all_bboxes     [cx, cy, w, h]      
       [x_min, x_max, y_min, y_max]          bbox_3d    
    """
    cx, cy, w, h = bbox
    return [
        round(cx - w / 2, 3),
        round(cx + w / 2, 3),
        round(cy - h / 2, 3),
        round(cy + h / 2, 3),
    ]


def compute_bbox_3d(indices: np.ndarray) -> list[int]:
    """
      (N, 3)        3D     

    Returns:
        [x_min, x_max, y_min, y_max, z_min, z_max] 32         
    """
    mn = indices.min(axis=0).astype(int)
    mx = indices.max(axis=0).astype(int)
    return [int(mn[0]), int(mx[0]), int(mn[1]), int(mx[1]), int(mn[2]), int(mx[2])]


#    V3 Dim 3 inter-part relative position                                     

def _diff_to_directions(diff: np.ndarray, threshold: float) -> list[str]:
    """
           diff = center_j - center_i             

            Blender Z-up     
        axis 0   x left(-) / right(+)
        axis 1   y front(-) / back(+)
        axis 2   z bottom(-) / top(+)
    """
    positions: list[str] = []
    if diff[0] < -threshold:
        positions.append('left')
    elif diff[0] > threshold:
        positions.append('right')
    if diff[1] < -threshold:
        positions.append('front')
    elif diff[1] > threshold:
        positions.append('back')
    if diff[2] < -threshold:
        positions.append('bottom')
    elif diff[2] > threshold:
        positions.append('top')
    return positions if positions else ['center']


def compute_inter_part_positions(
    indices_dict: dict[str, np.ndarray],
    size: int = GRID_SIZE,
) -> dict[str, dict[str, list[str]]]:
    """
                        26-       3D    
               

    Args:
        indices_dict : {str(part_label): np.ndarray (N, 3)}       
        size         :        

    Returns:
        {part_label: {neighbor_label: [direction_labels], ...}}
           {'0': {'1': ['bottom'], '2': ['right', 'bottom']}, '1': {'0': ['top']}, ...}
    """
    adjacency = compute_adjacency(indices_dict)
    centers   = {k: v.mean(axis=0) for k, v in indices_dict.items()}
    threshold = size * 0.15

    results: dict[str, dict[str, list[str]]] = {k: {} for k in indices_dict}
    for k, neighbors in adjacency.items():
        for j in neighbors:
            diff = centers[j] - centers[k]
            results[k][j] = _diff_to_directions(diff, threshold)
    return results


#    V3 Dim 4 major_axis + aspect_ratio + fit_primitive_v3                     

def _compute_major_axis(indices: np.ndarray) -> str:
    """
           PCA                x/y/z   

    Returns:
        'x' | 'y' | 'z'     < 3     'x'       
    """
    if len(indices) < 3:
        return 'x'
    pts = indices.astype(float)
    pts -= pts.mean(axis=0)
    cov = (pts.T @ pts) / max(len(pts) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    dominant = eigenvectors[:, np.argmax(eigenvalues)]
    return ['x', 'y', 'z'][int(np.argmax(np.abs(dominant)))]


def _compute_aspect_ratio(bbox_3d: list[int]) -> str:
    """
      bbox_3d                   

         
        very_flat  : min/max < 0.15
        flat       : 0.15   min/max < 0.35
        balanced   : min/max   0.65   mid/max   0.65
        tall       : z   axis 2         max/mid   2.0
        elongated  : x/y          max/mid   2.0        
    """
    x_s = max(bbox_3d[1] - bbox_3d[0], 1)
    y_s = max(bbox_3d[3] - bbox_3d[2], 1)
    z_s = max(bbox_3d[5] - bbox_3d[4], 1)

    dims_xyz = [x_s, y_s, z_s]
    min_d, mid_d, max_d = sorted(dims_xyz)

    ratio_min_max = min_d / max_d
    ratio_mid_max = mid_d / max_d

    if ratio_min_max < 0.15:
        return 'very_flat'
    if ratio_min_max < 0.35:
        return 'flat'
    if ratio_min_max >= 0.65 and ratio_mid_max >= 0.65:
        return 'balanced'
    # z             tall
    if dims_xyz[2] == max_d and ratio_mid_max < 0.5:
        return 'tall'
    return 'elongated'


def fit_primitive_v3(indices: np.ndarray, size: int = GRID_SIZE) -> dict:
    """
    V3        IoU     +    +       

       V2     
        1. IoU < 0.3   'complex' CoT3            cuboid
        2.    major_axis PCA     
        3.    aspect_ratio        

    Returns:
        {
            'shape_label':  'cuboid' | 'cylinder' | 'sphere' | 'complex',
            'major_axis':   'x' | 'y' | 'z',
            'aspect_ratio': 'very_flat' | 'flat' | 'balanced' | 'tall' | 'elongated',
        }
    """
    V_gt = indices_to_grid(indices, size)

    xmin, ymin, zmin = indices.min(axis=0).astype(float)
    xmax, ymax, zmax = indices.max(axis=0).astype(float)

    cx = (xmin + xmax) / 2.0 + 0.5
    cy = (ymin + ymax) / 2.0 + 0.5
    cz = (zmin + zmax) / 2.0 + 0.5
    sx = xmax - xmin + 1.0
    sy = ymax - ymin + 1.0
    sz = zmax - zmin + 1.0

    _r = lambda a, b: np.sqrt(a ** 2 + b ** 2) / 2.0 * 0.85

    candidates: dict[str, np.ndarray] = {
        'cuboid':     rasterize_cuboid(cx, cy, cz, sx, sy, sz, size),
        'cylinder_X': rasterize_cylinder(cx, cy, cz, _r(sy, sz), sx, 'X', size),
        'cylinder_Y': rasterize_cylinder(cx, cy, cz, _r(sx, sz), sy, 'Y', size),
        'cylinder_Z': rasterize_cylinder(cx, cy, cz, _r(sx, sy), sz, 'Z', size),
        'sphere':     rasterize_sphere(cx, cy, cz, min(sx, sy, sz) / 2.0, size),
    }

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        u = int((a | b).sum())
        return int((a & b).sum()) / u if u > 0 else 0.0

    best_label, best_iou = max(
        ((lbl, _iou(V_gt, mask)) for lbl, mask in candidates.items()),
        key=lambda x: x[1],
    )

    shape_label = 'complex' if best_iou < 0.3 else best_label.split('_')[0]

    bbox_3d = compute_bbox_3d(indices)
    return {
        'shape_label':  shape_label,
        'major_axis':   _compute_major_axis(indices),
        'aspect_ratio': _compute_aspect_ratio(bbox_3d),
    }


#    CoT <think>      V3                                                 

def _format_think_v3(
    part_count:       int,
    jsondata:         dict,
    prim_list:        list[dict],
    surf_list:        list[dict],
    inter_pos_dict:   dict[str, dict[str, list[str]]],
    bbox_2d_list:     list[list[float]] | None,
    bbox_3d_list:     list[list[int]],
) -> str:
    """
       V3     <think>   5       CoT3    
        Step 1       part_count 
        Step 2  2D & 3D    bbox_2d + bbox_3d           
                  bbox_2d : [x_min, x_max, y_min, y_max]     [0, 1]
                  bbox_3d : [x_min, x_max, y_min, y_max, z_min, z_max]     
        Step 3        3D                    
        Step 4         shape_label + major_axis + aspect_ratio 
        Step 5       hardness / roughness / reflectivity / transparency 
    """
    lines: list[str] = []

    #    Step 1                                                                 
    lines += [
        "Step 1: Global decomposition.",
        f"The object is decomposed into `part_count` core parts: {part_count}.",
        "",
    ]

    #    Step 2                                                                 
    lines += [
        "Step 2: 2D and 3D localization.",
        "For each part, the 2D image bounding range `bbox_2d` = [x_min, x_max, y_min, y_max] "
        "(normalized 0~1), the 3D voxel bounding range "
        "`bbox_3d` = [x_min, x_max, y_min, y_max, z_min, z_max], "
        "and the SAM visual feature token `sam_feat` are recorded.",
    ]
    for k, part in enumerate(jsondata['parts']):
        label = str(part['label'])
        b2 = _cxcywh_to_xyxy(bbox_2d_list[k]) if bbox_2d_list is not None else 'N/A'
        b3 = bbox_3d_list[k]
        lines.append(
            f"Part `l_{label}`: `bbox_2d` = {b2}, `bbox_3d` = {b3}, "
            f"`sam_feat` = <sam_feat_l_{label}>."
        )
    lines.append("")

    #    Step 3                                                                 
    lines += [
        "Step 3: 3D relative position.",
        "For each part, the relative 3D position of its directly adjacent parts is recorded.",
    ]
    for part in jsondata['parts']:
        label     = str(part['label'])
        neighbors = inter_pos_dict.get(label, {})
        if not neighbors:
            lines.append(f"Part `l_{label}`: no adjacent parts.")
        else:
            neighbor_strs = [
                f"`l_{j}` is at {pos}"
                for j, pos in sorted(neighbors.items(), key=lambda x: int(x[0]))
            ]
            lines.append(f"Part `l_{label}`: {', '.join(neighbor_strs)}.")
    lines.append("")

    #    Step 4                                                                 
    lines += [
        "Step 4: Primitive abstraction.",
        "For each part, the 3D geometric template is recorded by `primitive_shape`.",
    ]
    for k, part in enumerate(jsondata['parts']):
        label = str(part['label'])
        p     = prim_list[k]
        lines.append(
            f"Part `l_{label}`: "
            f"`shape_label` = {p['shape_label']}, "
            f"`major_axis` = {p['major_axis']}, "
            f"`aspect_ratio` = {p['aspect_ratio']}."
        )
    lines.append("")

    #    Step 5                                                                 
    lines += [
        "Step 5: Surface perception.",
        "For each part, the surface perceptual state is recorded by `surface_features`.",
    ]
    for k, part in enumerate(jsondata['parts']):
        label = str(part['label'])
        s     = surf_list[k]
        lines.append(
            f"Part `l_{label}`: "
            f"`hardness` = {s['hardness']}, "
            f"`roughness` = {s['roughness']}, "
            f"`reflectivity` = {s['reflectivity']}, "
            f"`transparency` = {s['transparency']}."
        )

    return "\n".join(lines)


#       object                                                              

def process_object(
    obj_id:     str,
    json_dir:   str,
    voxel_dir:  str,
    render_dir: str,
    output_dir: str,
    logger:     logging.Logger,
) -> int:
    """
         object            cot_tmp_v3/{obj_id}_{img_id}.txt 

    Returns:
                  
    """
    json_path    = os.path.join(json_dir,   obj_id + '.json')
    obj_vox_dir  = os.path.join(voxel_dir,  obj_id, '32')
    obj_rnd_dir  = os.path.join(render_dir, obj_id + '_')
    transforms_p = os.path.join(obj_rnd_dir, 'transforms.json')

    if not os.path.exists(json_path):
        logger.warning(f'{obj_id}: finaljson not found, skipping.')
        return 0
    if not os.path.exists(transforms_p):
        logger.warning(f'{obj_id}: transforms.json not found in renders_cond, skipping.')
        return 0

    with open(json_path, 'r') as f:
        jsondata = json.load(f)

    n_parts = len(jsondata['parts'])

    #                                                                   
    indices_list: list[np.ndarray] = []
    ply_paths:    list[str]        = []
    for k in range(n_parts):
        npy_p = os.path.join(obj_vox_dir, f'ind_{k}.npy')
        ply_p = os.path.join(obj_vox_dir, f'mesh_new_{k}.ply')
        if not os.path.exists(npy_p) or not os.path.exists(ply_p):
            logger.warning(f'{obj_id}: partseg data missing for part {k}, skipping.')
            return 0
        indices_list.append(np.load(npy_p).astype(np.int64))
        ply_paths.append(ply_p)

    #             object                                          
    # Dim 2 (3D   ) bbox_3d
    bbox_3d_list: list[list[int]] = [compute_bbox_3d(idx) for idx in indices_list]

    # Dim 3 inter-part relative position         
    inter_pos_dict = compute_inter_part_positions(
        {str(k): indices_list[k] for k in range(n_parts)}
    )

    # Dim 4 fit_primitive_v3 shape_label + major_axis + aspect_ratio 
    prim_list = [fit_primitive_v3(idx) for idx in indices_list]

    # Dim 5 surface_features         V2      
    surf_list = [
        derive_surface_features(
            p['material'],
            float(p.get("Young's Modulus (GPa)", 1.0))
        )
        for p in jsondata['parts']
    ]

    #          think    Dim 2   bbox_2d                           
    tf_meta    = load_transforms_meta(transforms_p)
    frames     = tf_meta['frames']
    obj_scale  = float(tf_meta.get('scale', 1.0))
    obj_offset = tf_meta.get('offset', [0.0, 0.0, 0.0])
    n_written  = 0

    for frame in frames:
        file_path = frame.get('file_path', '')
        img_id    = os.path.splitext(os.path.basename(file_path))[0]   # "000"

        fov_x        = float(frame['camera_angle_x'])
        c2w          = _build_c2w(frame)
        bbox_2d_list = compute_all_bboxes(
            ply_paths, c2w, fov_x, resolution=1024,
            obj_scale=obj_scale,
            obj_offset=obj_offset,
        )

        think_body = _format_think_v3(
            n_parts, jsondata,
            prim_list, surf_list,
            inter_pos_dict,
            bbox_2d_list,
            bbox_3d_list,
        )
        content = f"<think>\n{think_body}\n</think>\n"

        out_path = os.path.join(output_dir, f'{obj_id}_{img_id}.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        n_written += 1

    return n_written


#                                                                            

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate CoT V3 think text for PhysX-CoT'
    )
    parser.add_argument('--ind',   type=int, default=0,
                        help='Worker index for data sharding')
    parser.add_argument('--range', type=int, default=-1,
                        help='Number of objects per worker; -1 means all')
    args = parser.parse_args()

    json_dir   = './tmp/finaljson'
    voxel_dir  = './tmp/partseg'
    render_dir = './renders_cond'
    output_dir = './cot_tmp_v3'

    os.makedirs(output_dir, exist_ok=True)
    logger = _get_logger(f'./tmp/cot_think_v3_{args.ind}.log')
    logger.info('CoT V3 think generation started.')

    all_ids = sorted(
        d[:-1]
        for d in os.listdir(render_dir)
        if d.endswith('_') and os.path.isdir(os.path.join(render_dir, d))
    )
    if args.range != -1:
        all_ids = all_ids[args.ind * args.range: (args.ind + 1) * args.range]
    logger.info(f'Processing {len(all_ids)} objects (worker {args.ind}).')

    total = 0
    for obj_id in all_ids:
        try:
            n = process_object(obj_id, json_dir, voxel_dir, render_dir, output_dir, logger)
            total += n
            if n:
                logger.info(f'{obj_id}: {n} files written.')
        except Exception as e:
            logger.warning(f'{obj_id}: FAILED   {e}')

    logger.info(f'Done. Total {total} files written to {output_dir}.')


if __name__ == '__main__':
    main()
