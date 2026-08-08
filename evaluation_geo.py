"""
Geometry evaluation for PhysX-CoT.

Metrics
-------
- PSNR    : peak signal-to-noise ratio on multi-view rendered normal maps
- CD      : Chamfer Distance = mean_{a in A} min_{b in B} ||a-b|| + symmetric
- F-score : 2*P*R/(P+R), where P/R = fraction of points within `tau` (normalized space)

Convention
----------
Meshes are normalized exactly like `evaluation_phy.py`:
  - GT mesh receives a +pi/2 rotation around X (dataset convention)
  - Both meshes are then centered to origin and scaled so that
    `max(bbox_max - bbox_min) == 1` (i.e. inscribed in a unit cube).
With this normalization, `tau=0.01` corresponds to 1% of the unit bbox.

Usage
-----
    python evaluation_geo.py

or programmatically:

    from evaluation_geo import evaluate_geometry
    metrics = evaluate_geometry(
        resultpath='./test_demo',
        datasetpath='./PhysX_mobility',
        namelist_path='./val_test_list.npy',
    )
"""

import os
import json
from typing import Optional, Dict, Any

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree

from trellis.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
    render_frames_eval,
)
from trellis.representations.mesh.cube2mesh import MeshExtractResult


# ---------------------------------------------------------------------------
# Mesh I/O & normalization (kept consistent with evaluation_phy.py)
# ---------------------------------------------------------------------------

def load_obj_geometry_fast(path: str) -> trimesh.Trimesh:
    V, F = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                V.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                F.append(idx)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    return trimesh.Trimesh(V, F, process=False)


def mov(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center to origin and scale so the longest bbox axis becomes 1 (in-place)."""
    if len(mesh.vertices) == 0:
        raise ValueError("cannot normalize an empty mesh")
    bbox_max = np.array(mesh.vertices).max(0)
    bbox_min = np.array(mesh.vertices).min(0)
    longest_axis = float(max(bbox_max - bbox_min))
    if longest_axis <= 0:
        raise ValueError("cannot normalize a zero-size mesh")
    scale = 1.0 / longest_axis
    offset = -(bbox_min + bbox_max) / 2
    mesh.apply_translation([offset[0], offset[1], offset[2]])
    mesh.apply_transform(trimesh.transformations.scale_matrix(scale))
    return mesh


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return 50.0
    return 10.0 * float(np.log10((data_range ** 2) / mse))


def _sample_surface(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
    return np.asarray(pts, dtype=np.float64)


def chamfer_and_fscore(
    mesh_gt: trimesh.Trimesh,
    mesh_eval: trimesh.Trimesh,
    n: int = 10000,
    tau: float = 0.01,
    seed: int = 0,
):
    """Returns (cd, fscore, precision, recall).

    CD convention: sum of mean nearest-neighbor distances in both directions.
    """
    a = _sample_surface(mesh_gt, n, seed=seed)
    b = _sample_surface(mesh_eval, n, seed=seed + 1)

    ta = KDTree(a)
    tb = KDTree(b)
    d_ab, _ = tb.query(a, k=1, workers=-1)
    d_ba, _ = ta.query(b, k=1, workers=-1)

    cd = float(d_ab.mean() + d_ba.mean())
    precision = float((d_ba < tau).mean())
    recall = float((d_ab < tau).mean())
    fscore = 2.0 * precision * recall / (precision + recall + 1e-12)
    return cd, fscore, precision, recall


# ---------------------------------------------------------------------------
# Multi-view normal rendering (uses the same camera trajectory as evaluation_phy.py)
# ---------------------------------------------------------------------------

def _build_render_mesh(mesh: trimesh.Trimesh) -> MeshExtractResult:
    return MeshExtractResult(
        torch.as_tensor(mesh.vertices, dtype=torch.float32).cuda(),
        torch.as_tensor(mesh.faces, dtype=torch.int64).cuda(),
        vertex_attrs=None,
        res=64,
        render_vis=torch.zeros((len(mesh.vertices), 3), dtype=torch.float32).cuda(),
    )


def _render_normals(mesh: MeshExtractResult, num_frames: int = 30, resolution: int = 512):
    yaws = torch.linspace(0, 2 * np.pi, num_frames).tolist()
    pitch = (0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * np.pi, num_frames))).tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 40)
    return render_frames_eval(
        mesh, extrinsics, intrinsics,
        {'resolution': resolution, 'bg_color': (0, 0, 0)},
        return_types=["normal", "mask"],
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Per-sample mesh assembly (matches evaluation_phy.py exactly)
# ---------------------------------------------------------------------------

def _load_part_obj(root: str, name: str, objfile: Any) -> trimesh.Trimesh:
    candidates = [
        os.path.join(root, name, 'objs', str(objfile) + '.obj'),
        os.path.join(root, name, 'objs', str(objfile), str(objfile) + '.obj'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_obj_geometry_fast(path)
    raise FileNotFoundError(f"part OBJ not found; tried: {candidates}")


def _load_eval_part_obj(
    eval_root: str,
    name: str,
    part_info: Dict[str, Any],
    index: int,
) -> trimesh.Trimesh:
    labels = [part_info.get('label', index), index]
    tried = []
    for label in dict.fromkeys(labels):
        candidates = [
            os.path.join(eval_root, name, 'objs', str(label), str(label) + '.obj'),
            os.path.join(eval_root, name, 'objs', str(label) + '.obj'),
        ]
        tried.extend(candidates)
        for path in candidates:
            if os.path.exists(path):
                return load_obj_geometry_fast(path)
    raise FileNotFoundError(f"eval part OBJ not found; tried: {tried}")


def _load_gt_mesh(gt_json_path: str, mesh_root: str, name: str) -> trimesh.Trimesh:
    with open(gt_json_path, 'r') as fp:
        jsondata = json.load(fp)
    all_mesh = trimesh.Trimesh([])
    for index, part_info in enumerate(jsondata['parts']):
        objlist = part_info.get('obj', [part_info.get('label', index)])
        for objfile in objlist:
            piece = _load_part_obj(mesh_root, name, objfile)
            all_mesh = trimesh.util.concatenate([piece, all_mesh])
    return all_mesh


def _load_eval_mesh(eval_json_path: str, eval_root: str, name: str) -> trimesh.Trimesh:
    with open(eval_json_path, 'r') as fp:
        jsondata = json.load(fp)
    all_mesh = trimesh.Trimesh([])
    for index, part_info in enumerate(jsondata['parts']):
        piece = _load_eval_part_obj(eval_root, name, part_info, index)
        all_mesh = trimesh.util.concatenate([piece, all_mesh])
    return all_mesh


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _discover_namelist(resultpath: str, gt_json_dir: str) -> list:
    """Discover sample names as the intersection of eval results and GT annotations.

    A sample is included when both `{resultpath}/{name}/basic_info.json` and
    `{gt_json_dir}/{name}.json` exist.
    """
    if not (os.path.isdir(resultpath) and os.path.isdir(gt_json_dir)):
        return []
    eval_names = {
        d for d in os.listdir(resultpath)
        if os.path.isfile(os.path.join(resultpath, d, 'basic_info.json'))
    }
    gt_names = {
        os.path.splitext(f)[0] for f in os.listdir(gt_json_dir)
        if f.endswith('.json')
    }
    return sorted(eval_names & gt_names)


def evaluate_geometry(
    resultpath: str = './test_demo',
    datasetpath: str = './PhysX_mobility',
    namelist_path: Optional[str] = './val_test_list.npy',
    n_points: int = 10000,
    fscore_tau: float = 0.01,
    num_frames: int = 30,
    seed: int = 0,
    save_json: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    jsonpath = os.path.join(datasetpath, 'finaljson')
    meshpath = os.path.join(datasetpath, 'partseg')

    if namelist_path and os.path.exists(namelist_path):
        namelist = np.load(namelist_path, allow_pickle=True)
        if verbose:
            print(f"[info] using namelist from {namelist_path} ({len(namelist)} samples)")
    else:
        namelist = _discover_namelist(resultpath, jsonpath)
        if verbose:
            print(
                f"[info] namelist not provided or not found, auto-discovered "
                f"{len(namelist)} samples from {resultpath} intersect {jsonpath}"
            )

    per_sample: Dict[str, Dict[str, Any]] = {}
    all_cd, all_fscore, all_psnr = [], [], []
    n_load_fail = 0
    n_psnr_fail = 0

    for raw_name in namelist:
        name = str(raw_name)
        gt_json = os.path.join(jsonpath, name + '.json')
        eval_json = os.path.join(resultpath, name, 'basic_info.json')

        if not (os.path.exists(gt_json) and os.path.exists(eval_json)):
            if verbose:
                print(f"[skip] {name}: missing GT or eval json")
            n_load_fail += 1
            continue

        try:
            gt_mesh = _load_gt_mesh(gt_json, meshpath, name)
            eval_mesh = _load_eval_mesh(eval_json, resultpath, name)
        except Exception as e:
            if verbose:
                print(f"[skip] {name}: load mesh failed: {e}")
            n_load_fail += 1
            continue

        # Coordinate convention: GT gets a +pi/2 X rotation, then both get unit-bbox normalized
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        gt_mesh.apply_transform(rot)
        gt_mesh = mov(gt_mesh)
        eval_mesh = mov(eval_mesh)

        try:
            cd, fscore, prec, rec = chamfer_and_fscore(
                gt_mesh, eval_mesh, n=n_points, tau=fscore_tau, seed=seed,
            )
        except Exception as e:
            if verbose:
                print(f"[skip] {name}: CD/F-score failed: {e}")
            n_load_fail += 1
            continue

        psnr_value: Optional[float] = None
        try:
            gt_render = _build_render_mesh(gt_mesh)
            eval_render = _build_render_mesh(eval_mesh)
            video_gt = _render_normals(gt_render, num_frames=num_frames)
            video_eval = _render_normals(eval_render, num_frames=num_frames)

            per_frame = []
            for i in range(len(video_gt['normal'])):
                a = video_gt['normal'][i].astype(np.float64) / 255.0
                b = video_eval['normal'][i].astype(np.float64) / 255.0
                per_frame.append(psnr(a, b, data_range=1.0))
            psnr_value = float(np.mean(per_frame))
        except Exception as e:
            if verbose:
                print(f"[warn] {name}: PSNR failed: {e}")
            n_psnr_fail += 1

        all_cd.append(cd)
        all_fscore.append(fscore)
        if psnr_value is not None:
            all_psnr.append(psnr_value)

        per_sample[name] = {
            'cd': cd,
            'fscore': fscore,
            'precision': prec,
            'recall': rec,
            'psnr': psnr_value,
        }
        if verbose:
            psnr_str = f"{psnr_value:.2f}" if psnr_value is not None else "N/A"
            print(f"{name}: CD={cd:.4f}, F={fscore:.4f}, PSNR={psnr_str}")

    summary = {
        'psnr_mean': float(np.mean(all_psnr)) if all_psnr else float('nan'),
        'cd_mean': float(np.mean(all_cd)) if all_cd else float('nan'),
        'fscore_mean': float(np.mean(all_fscore)) if all_fscore else float('nan'),
        'n_samples_geom': len(all_cd),
        'n_samples_psnr': len(all_psnr),
        'n_load_fail': n_load_fail,
        'n_psnr_fail': n_psnr_fail,
    }

    if save_json:
        out = {
            'per_sample': per_sample,
            'summary': summary,
            'config': {
                'resultpath': resultpath,
                'datasetpath': datasetpath,
                'namelist_path': namelist_path,
                'n_points': n_points,
                'fscore_tau': fscore_tau,
                'num_frames': num_frames,
                'seed': seed,
            },
        }
        os.makedirs(os.path.dirname(save_json) or '.', exist_ok=True)
        with open(save_json, 'w') as f:
            json.dump(out, f, indent=2)

    return summary


if __name__ == '__main__':
    res = evaluate_geometry(save_json='./eval_results/geometry.json')
    print('\n=== Geometry Evaluation ===')
    print(f"PSNR (normal): {res['psnr_mean']:.4f}")
    print(f"CD:            {res['cd_mean']:.6f}")
    print(f"F-score@0.01:  {res['fscore_mean']:.4f}")
    print(f"(geom samples: {res['n_samples_geom']}, psnr samples: {res['n_samples_psnr']})")
