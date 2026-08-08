"""evaluation_vlm.py   Direct evaluation of VLM text + voxel outputs.

Evaluates Stage-1 (VLM) outputs against PhysX-Net ground truth WITHOUT
running the downstream decoder / split / simready stages. This isolates the
VLM ability for diagnostic comparison across methods or checkpoints, and is
much cheaper than `evaluation_phy.py` (no rendering required).

Inputs
------
VLM predictions (one sub-dir per sample, mirroring ``1_vlm_cot.py`` layout):
    <pred_dir>/<name>/basic_info.txt   # text: Name/Category/Dim/Parts/Group_info
    <pred_dir>/<name>/ind_<i>.npy      # int voxel coords [N, 3] in voxel_res grid
    <pred_dir>/<name>/allind.npy       # optional, concatenation of ind_*.npy

Ground truth (PhysX-Net):
    <gt_root>/finaljson/<name>.json              # structured GT
    <gt_root>/partseg/<name>/objs/<label>.obj    # per-part triangle mesh

Outputs
-------
``<out>.json`` carrying per-sample and aggregate metrics in three groups:
  * Overall       : name / category / dimension / part count
  * Text-Physics  : material / density / Young's / Poisson / priority-rank / desc
  * Geometry      : voxel IoU / F1 / CD / HD95 at object & part level

Metric design (no LLM judge, fully reproducible)
------------------------------------------------
- Free-text similarity: ``sim_text = max(token_jaccard, difflib_ratio)`` on
  lowercased, alphanumeric-only inputs. No SBERT / no API call.
- Material: passed through a synonym dictionary first, then ``sim_text``.
- Part matching: Hungarian on cost = alpha * (1 - voxel_IoU) + (1 - alpha) *
  (1 - sim_text(part_name)).  alpha defaults to 0.7 (geometry-first).

CLI
---
    python evaluation_vlm.py \
        --pred_dir test_demo/<method> \
        --gt_root  ./dataset/physxnet \
        --namelist ./eval_data/sample_list.npy \
        --out      eval_results/<method>/vlm_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace; keep word boundaries for tokenization."""
    return " ".join(_TOKEN_RE.findall((s or "").lower()))


def _tokens(s: str) -> List[str]:
    return _TOKEN_RE.findall((s or "").lower())


def text_similarity(a: str, b: str) -> float:
    """Symmetric similarity in [0, 1].

    Combines token-level Jaccard with character-level difflib ratio and takes
    the max, so synonyms with disjoint tokens but high char overlap still score
    well (and vice versa).
    """
    na, nb = _normalize(a), _normalize(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    ta, tb = set(_tokens(a)), set(_tokens(b))
    jaccard = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    char_ratio = SequenceMatcher(None, na, nb).ratio()
    return max(jaccard, char_ratio)


# Material synonyms: rough mapping so e.g. "PE" / "polyethylene" / "poly ethylene"
# all collapse to the same canonical form. Conservative and extendable.
_MATERIAL_ALIASES: Dict[str, str] = {
    "pe": "polyethylene",
    "pp": "polypropylene",
    "pvc": "polyvinyl chloride",
    "abs": "acrylonitrile butadiene styrene",
    "pet": "polyethylene terephthalate",
    "ps": "polystyrene",
    "pmma": "poly methyl methacrylate",
    "ptfe": "polytetrafluoroethylene",
    "mdf": "medium density fiberboard",
    "ss": "stainless steel",
    "al": "aluminum",
    "alu": "aluminum",
    "aluminium": "aluminum",
    "cu": "copper",
    "fe": "iron",
}


def canonical_material(s: str) -> str:
    toks = _tokens(s)
    expanded = [_MATERIAL_ALIASES.get(t, t) for t in toks]
    return " ".join(expanded)


def material_similarity(a: str, b: str) -> float:
    return text_similarity(canonical_material(a), canonical_material(b))


# ---------------------------------------------------------------------------
# Numeric utilities
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def parse_number(s) -> Optional[float]:
    """Best-effort float extraction. Returns None if no number found."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = _NUM_RE.search(str(s))
    return float(m.group()) if m else None


def parse_dim_triplet(s: str) -> Optional[Tuple[float, float, float]]:
    """Parse "18*15*7" / "18 * 15 * 7 cm" -> (18.0, 15.0, 7.0). None on failure."""
    if s is None:
        return None
    nums = _NUM_RE.findall(str(s))
    if len(nums) < 3:
        return None
    return tuple(float(x) for x in nums[:3])


def mape(pred: float, gt: float, eps: float = 1e-6) -> float:
    return abs(pred - gt) / (abs(gt) + eps)


# ---------------------------------------------------------------------------
# basic_info.txt parser
# ---------------------------------------------------------------------------

@dataclass
class PartPred:
    label: int
    name: str = ""
    priority_rank: Optional[float] = None
    material: str = ""
    density: Optional[float] = None
    young: Optional[float] = None
    poisson: Optional[float] = None
    description: str = ""
    voxels: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))


@dataclass
class SamplePred:
    name: str
    object_name: str = ""
    category: str = ""
    dimension: Optional[Tuple[float, float, float]] = None
    parts: List[PartPred] = field(default_factory=list)


def parse_basic_info(text: str) -> Tuple[str, str, Optional[Tuple[float, float, float]], List[PartPred]]:
    """Parse VLM-emitted basic_info.txt; tolerant to minor format drift."""
    object_name = ""
    category = ""
    dimension: Optional[Tuple[float, float, float]] = None
    parts: List[PartPred] = []

    lines = [ln.rstrip() for ln in text.splitlines()]
    in_parts = False
    in_groups = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # section toggles
        if line.lower().startswith("parts:"):
            in_parts, in_groups = True, False
            continue
        if line.lower().startswith("group_info"):
            in_parts, in_groups = False, True
            continue

        if not in_parts and not in_groups:
            # object-level headers
            low = line.lower()
            if low.startswith("name:"):
                object_name = line.split(":", 1)[1].strip()
            elif low.startswith("category:"):
                category = line.split(":", 1)[1].strip()
            elif low.startswith("dimension:"):
                dimension = parse_dim_triplet(line.split(":", 1)[1])
            continue

        if in_parts:
            m = re.match(r"^l_(\d+)\s*:\s*(.+)$", line)
            if not m:
                continue
            label = int(m.group(1))
            payload = m.group(2)
            # name, rank, material, density, young, poisson, description (desc may contain commas)
            chunks = [c.strip() for c in payload.split(",", 6)]
            while len(chunks) < 7:
                chunks.append("")
            parts.append(PartPred(
                label=label,
                name=chunks[0],
                priority_rank=parse_number(chunks[1]),
                material=chunks[2],
                density=parse_number(chunks[3]),
                young=parse_number(chunks[4]),
                poisson=parse_number(chunks[5]),
                description=chunks[6],
            ))
            continue

        # group_info parsing intentionally omitted: kinematics is evaluated
        # separately by evaluation_kine.py per user spec.

    return object_name, category, dimension, parts


def load_pred(pred_dir: str, name: str, voxel_res: int = 32) -> SamplePred:
    sample_dir = os.path.join(pred_dir, name)
    txt_path = os.path.join(sample_dir, "basic_info.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"missing prediction text: {txt_path}")

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    object_name, category, dimension, parts = parse_basic_info(text)

    for p in parts:
        npy_path = os.path.join(sample_dir, f"ind_{p.label}.npy")
        if os.path.exists(npy_path):
            v = np.load(npy_path).astype(np.int64).reshape(-1, 3)
            np.clip(v, 0, voxel_res - 1, out=v)
            p.voxels = np.unique(v, axis=0)

    return SamplePred(
        name=name,
        object_name=object_name,
        category=category,
        dimension=dimension,
        parts=parts,
    )


# ---------------------------------------------------------------------------
# GT loader + voxelizer (mirrors dataset/1voxel.py protocol)
# ---------------------------------------------------------------------------

@dataclass
class PartGT:
    label: int
    name: str
    material: str
    density: Optional[float]
    young: Optional[float]
    poisson: Optional[float]
    priority_rank: Optional[float]
    description: str
    voxels: np.ndarray


@dataclass
class SampleGT:
    name: str
    object_name: str
    category: str
    dimension: Optional[Tuple[float, float, float]]
    parts: List[PartGT]


def _sample_surface_seeded(mesh: trimesh.Trimesh, n_pts: int, seed: int) -> np.ndarray:
    """Surface sampling with a deterministic seed; tolerant to trimesh version skew."""
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n_pts, seed=seed)
        return pts
    except TypeError:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            pts, _ = trimesh.sample.sample_surface(mesh, n_pts)
            return pts
        finally:
            np.random.set_state(state)


def _voxelize_normalized_mesh(mesh: trimesh.Trimesh, res: int, n_pts: int, rng: np.random.Generator) -> np.ndarray:
    """Sample mesh surface and quantize to a res^3 grid in [-0.5, 0.5]^3.

    Approximates the open3d Poisson-disk routine used in dataset/1voxel.py with
    uniform surface sampling; surface coverage is what matters for voxel sets.
    """
    if mesh.is_empty or len(mesh.faces) == 0:
        return np.empty((0, 3), dtype=np.int64)
    seed = int(rng.integers(0, 2**31 - 1))
    pts = _sample_surface_seeded(mesh, n_pts, seed)
    pts = np.clip(pts, -0.5 + 1e-6, 0.5 - 1e-6)
    idx = np.floor((pts + 0.5) * res).astype(np.int64)
    idx = np.clip(idx, 0, res - 1)
    return np.unique(idx, axis=0)


def load_gt(gt_root: str, name: str, voxel_res: int = 32, n_surface_pts: int = 81920,
            seed: int = 0) -> SampleGT:
    json_path = os.path.join(gt_root, "finaljson", f"{name}.json")
    objs_dir = os.path.join(gt_root, "partseg", name, "objs")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"GT json missing: {json_path}")
    if not os.path.isdir(objs_dir):
        raise FileNotFoundError(f"GT objs missing: {objs_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_parts = data.get("parts", [])
    # Resolve per-part mesh files (PhysX-Net uses label -> <label>.obj 1:1).
    mesh_paths: List[str] = []
    for p in raw_parts:
        obj_path = os.path.join(objs_dir, f"{p['label']}.obj")
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"part obj missing: {obj_path}")
        mesh_paths.append(obj_path)

    # Concatenate to compute the global normalization (parity with 1voxel.py).
    merged = trimesh.Trimesh()
    for op in mesh_paths:
        merged = trimesh.util.concatenate([merged, trimesh.load(op, process=False)])
    merged.merge_vertices()
    if merged.is_empty:
        raise ValueError(f"empty merged mesh for {name}")

    bbox_max = np.asarray(merged.vertices).max(0)
    bbox_min = np.asarray(merged.vertices).min(0)
    scale = 1.0 / max((bbox_max - bbox_min).max(), 1e-9)
    offset = -(bbox_min + bbox_max) / 2.0
    M_rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])

    rng = np.random.default_rng(seed)
    parts: List[PartGT] = []
    for p, op in zip(raw_parts, mesh_paths):
        m = trimesh.load(op, process=False)
        m.apply_transform(trimesh.transformations.scale_matrix(scale))
        m.apply_translation(list(offset))
        m.apply_transform(M_rot)
        vox = _voxelize_normalized_mesh(m, voxel_res, n_surface_pts, rng)
        parts.append(PartGT(
            label=p["label"],
            name=p.get("name", ""),
            material=p.get("material", ""),
            density=parse_number(p.get("density")),
            young=parse_number(p.get("Young's Modulus (GPa)")),
            poisson=parse_number(p.get("Poisson's Ratio")),
            priority_rank=parse_number(p.get("priority_rank")),
            description=p.get("Basic_description", ""),
            voxels=vox,
        ))

    return SampleGT(
        name=name,
        object_name=data.get("object_name", ""),
        category=data.get("category", ""),
        dimension=parse_dim_triplet(data.get("dimension")),
        parts=parts,
    )


# ---------------------------------------------------------------------------
# Voxel-set metrics
# ---------------------------------------------------------------------------

def voxel_set_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Precision / Recall / F1 / IoU on (N, 3) voxel coordinate sets."""
    if len(pred) == 0 and len(gt) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}
    set_p = {tuple(v) for v in pred}
    set_g = {tuple(v) for v in gt}
    tp = len(set_p & set_g)
    fp = len(set_p - set_g)
    fn = len(set_g - set_p)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}


def voxel_distance_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Chamfer distance and Hausdorff-95 (voxel units, symmetric)."""
    if len(pred) == 0 or len(gt) == 0:
        return {"cd": float("nan"), "hd95": float("nan")}
    tree_p = cKDTree(pred)
    tree_g = cKDTree(gt)
    d_pg, _ = tree_g.query(pred, k=1)
    d_gp, _ = tree_p.query(gt, k=1)
    cd = 0.5 * (float(d_pg.mean()) + float(d_gp.mean()))
    hd95 = max(float(np.quantile(d_pg, 0.95)), float(np.quantile(d_gp, 0.95)))
    return {"cd": cd, "hd95": hd95}


def voxel_centroid_err(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(pred) == 0 or len(gt) == 0:
        return float("nan")
    return float(np.linalg.norm(pred.mean(0) - gt.mean(0)))


# ---------------------------------------------------------------------------
# Part matching
# ---------------------------------------------------------------------------

def match_parts(pred_parts: List[PartPred], gt_parts: List[PartGT],
                alpha: float = 0.7) -> List[Tuple[int, int]]:
    """Hungarian matching by cost = alpha*(1-IoU) + (1-alpha)*(1-name_sim).

    Returns a list of (pred_idx, gt_idx) for matched pairs only (rectangular
    matrices are padded internally so the assignment is over min(N,M) pairs).
    """
    n_p, n_g = len(pred_parts), len(gt_parts)
    if n_p == 0 or n_g == 0:
        return []

    cost = np.ones((n_p, n_g), dtype=np.float64)
    for i, p in enumerate(pred_parts):
        for j, g in enumerate(gt_parts):
            iou = voxel_set_metrics(p.voxels, g.voxels)["iou"]
            ns = text_similarity(p.name, g.name)
            cost[i, j] = alpha * (1.0 - iou) + (1.0 - alpha) * (1.0 - ns)

    row, col = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(row, col)]


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def _safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _dim_iou(pred: Tuple[float, float, float], gt: Tuple[float, float, float]) -> float:
    """Axis-aligned 3D bbox IoU after sorting axes (orientation-agnostic)."""
    p = sorted(pred, reverse=True)
    g = sorted(gt, reverse=True)
    inter = float(np.prod([min(a, b) for a, b in zip(p, g)]))
    vol_p = float(np.prod(p))
    vol_g = float(np.prod(g))
    union = vol_p + vol_g - inter
    return inter / union if union > 0 else 0.0


def _dim_mape(pred: Tuple[float, float, float], gt: Tuple[float, float, float]) -> float:
    p = sorted(pred, reverse=True)
    g = sorted(gt, reverse=True)
    return float(np.mean([mape(a, b) for a, b in zip(p, g)]))


def _spearman(x: List[float], y: List[float]) -> float:
    """Spearman rho with average-rank ties; returns nan when length < 2 or no variance."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return float("nan")
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    # average ranks
    def _rank(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a))
        # tie-averaging
        uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        means = np.zeros(len(uniq))
        for i, c in enumerate(counts):
            mask = inv == i
            means[i] = ranks[mask].mean()
        return means[inv]
    rx, ry = _rank(xa), _rank(ya)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_sample(pred: SamplePred, gt: SampleGT, alpha: float = 0.7) -> Dict:
    """Compute all metrics for one sample."""
    metrics: Dict = {}

    # ----- Overall -----
    name_sim = text_similarity(pred.object_name, gt.object_name)
    cat_sim = text_similarity(pred.category, gt.category)
    cat_strict = float(_normalize(pred.category) == _normalize(gt.category))
    dim_mape = _dim_mape(pred.dimension, gt.dimension) if (pred.dimension and gt.dimension) else float("nan")
    dim_iou = _dim_iou(pred.dimension, gt.dimension) if (pred.dimension and gt.dimension) else float("nan")
    n_p, n_g = len(pred.parts), len(gt.parts)
    part_count_acc = 1.0 - abs(n_p - n_g) / max(n_p, n_g, 1)
    metrics["overall"] = {
        "name_sim": name_sim,
        "category_sim": cat_sim,
        "category_strict_acc": cat_strict,
        "dim_mape": dim_mape,
        "dim_iou": dim_iou,
        "part_count": {"pred": n_p, "gt": n_g, "score": part_count_acc},
    }

    # ----- Part matching -----
    pairs = match_parts(pred.parts, gt.parts, alpha=alpha)
    matched_iou_threshold = 0.10

    # ----- Geometry (object-level) -----
    pred_all = np.concatenate([p.voxels for p in pred.parts] or [np.empty((0, 3), int)], axis=0)
    pred_all = np.unique(pred_all, axis=0) if len(pred_all) else pred_all
    gt_all = np.concatenate([g.voxels for g in gt.parts] or [np.empty((0, 3), int)], axis=0)
    gt_all = np.unique(gt_all, axis=0) if len(gt_all) else gt_all

    obj_set = voxel_set_metrics(pred_all, gt_all)
    obj_dist = voxel_distance_metrics(pred_all, gt_all)
    obj_centroid = voxel_centroid_err(pred_all, gt_all)

    # ----- Geometry (part-level, on matched pairs) -----
    part_ious, part_cds, part_centroids = [], [], []
    matched_ok = 0
    for i, j in pairs:
        iou = voxel_set_metrics(pred.parts[i].voxels, gt.parts[j].voxels)["iou"]
        part_ious.append(iou)
        dist = voxel_distance_metrics(pred.parts[i].voxels, gt.parts[j].voxels)
        part_cds.append(dist["cd"])
        part_centroids.append(voxel_centroid_err(pred.parts[i].voxels, gt.parts[j].voxels))
        if iou >= matched_iou_threshold:
            matched_ok += 1
    match_ratio = matched_ok / max(n_g, 1)

    metrics["geometry"] = {
        "object": {
            "iou": obj_set["iou"],
            "precision": obj_set["precision"],
            "recall": obj_set["recall"],
            "f1": obj_set["f1"],
            "cd": obj_dist["cd"],
            "hd95": obj_dist["hd95"],
            "centroid_err": obj_centroid,
        },
        "part": {
            "mean_iou": _safe_mean(part_ious),
            "mean_cd": _safe_mean(part_cds),
            "mean_centroid_err": _safe_mean(part_centroids),
            "matched_ratio": match_ratio,
            "matched_count": matched_ok,
            "num_pairs": len(pairs),
        },
    }

    # ----- Text-physics (on matched pairs) -----
    mat_sims, dens_mapes, young_mapes, poisson_maes = [], [], [], []
    desc_sims = []
    pred_ranks, gt_ranks = [], []
    for i, j in pairs:
        p = pred.parts[i]
        g = gt.parts[j]
        mat_sims.append(material_similarity(p.material, g.material))
        if p.density is not None and g.density is not None:
            dens_mapes.append(mape(p.density, g.density))
        if p.young is not None and g.young is not None:
            young_mapes.append(mape(p.young, g.young))
        if p.poisson is not None and g.poisson is not None:
            poisson_maes.append(abs(p.poisson - g.poisson))
        if p.priority_rank is not None and g.priority_rank is not None:
            pred_ranks.append(p.priority_rank)
            gt_ranks.append(g.priority_rank)
        desc_sims.append(text_similarity(p.description, g.description))

    metrics["text_physics"] = {
        "material_sim": _safe_mean(mat_sims),
        "density_mape": _safe_mean(dens_mapes),
        "young_mape": _safe_mean(young_mapes),
        "poisson_mae": _safe_mean(poisson_maes),
        "priority_rank_spearman": _spearman(pred_ranks, gt_ranks),
        "description_sim": _safe_mean(desc_sims),
    }

    metrics["meta"] = {
        "matched_pairs": [{"pred_idx": int(i), "gt_idx": int(j)} for i, j in pairs],
    }
    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _collect(per_sample: List[Dict], path: List[str]) -> List[float]:
    out = []
    for s in per_sample:
        cur = s.get("metrics", {})
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if isinstance(cur, (int, float)) and not (isinstance(cur, float) and np.isnan(cur)):
            out.append(float(cur))
    return out


_AGG_SPEC: List[Tuple[str, List[str]]] = [
    ("overall.name_sim", ["overall", "name_sim"]),
    ("overall.category_sim", ["overall", "category_sim"]),
    ("overall.category_strict_acc", ["overall", "category_strict_acc"]),
    ("overall.dim_mape", ["overall", "dim_mape"]),
    ("overall.dim_iou", ["overall", "dim_iou"]),
    ("overall.part_count_score", ["overall", "part_count", "score"]),
    ("geometry.object.iou", ["geometry", "object", "iou"]),
    ("geometry.object.f1", ["geometry", "object", "f1"]),
    ("geometry.object.precision", ["geometry", "object", "precision"]),
    ("geometry.object.recall", ["geometry", "object", "recall"]),
    ("geometry.object.cd", ["geometry", "object", "cd"]),
    ("geometry.object.hd95", ["geometry", "object", "hd95"]),
    ("geometry.object.centroid_err", ["geometry", "object", "centroid_err"]),
    ("geometry.part.mean_iou", ["geometry", "part", "mean_iou"]),
    ("geometry.part.mean_cd", ["geometry", "part", "mean_cd"]),
    ("geometry.part.mean_centroid_err", ["geometry", "part", "mean_centroid_err"]),
    ("geometry.part.matched_ratio", ["geometry", "part", "matched_ratio"]),
    ("text_physics.material_sim", ["text_physics", "material_sim"]),
    ("text_physics.density_mape", ["text_physics", "density_mape"]),
    ("text_physics.young_mape", ["text_physics", "young_mape"]),
    ("text_physics.poisson_mae", ["text_physics", "poisson_mae"]),
    ("text_physics.priority_rank_spearman", ["text_physics", "priority_rank_spearman"]),
    ("text_physics.description_sim", ["text_physics", "description_sim"]),
]


def _aggregate(per_sample: List[Dict]) -> Tuple[Dict, Dict]:
    """Return (nested, flat) aggregate dictionaries; nan-skipping mean."""
    flat: Dict[str, float] = {}
    nested: Dict[str, Dict] = {}
    for dotted, path in _AGG_SPEC:
        val = _safe_mean(_collect(per_sample, path))
        flat[dotted] = val
        cur = nested
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = val
    return nested, flat


# ---------------------------------------------------------------------------
# Difficulty classification
# ---------------------------------------------------------------------------

# Composite difficulty score: weighted mean of higher-is-better metrics in [0,1].
DEFAULT_COMPOSITE_WEIGHTS: Dict[str, float] = {
    "geometry.object.iou": 0.4,
    "geometry.part.mean_iou": 0.2,
    "overall.dim_iou": 0.2,
    "overall.category_strict_acc": 0.1,
    "text_physics.material_sim": 0.1,
}

# Key scores surfaced in lean per-sample records (for quick triage).
LEAN_KEY_SCORE_PATHS: List[str] = [
    "geometry.object.iou",
    "geometry.part.mean_iou",
    "overall.dim_iou",
    "overall.category_strict_acc",
    "text_physics.material_sim",
    "overall.name_sim",
]


def _get_metric_by_path(metrics: Dict, dotted: str) -> Optional[float]:
    """Look up a metric by 'a.b.c' path; return None when missing / NaN / non-numeric."""
    cur = metrics
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if not isinstance(cur, (int, float)):
        return None
    f = float(cur)
    return None if np.isnan(f) else f


def composite_score(metrics: Dict,
                    weights: Dict[str, float] = DEFAULT_COMPOSITE_WEIGHTS) -> float:
    """Weighted mean (in [0,1], higher = better) over available metrics.

    Re-normalizes weights over the subset that is actually present so missing
    fields do not silently zero-out the score.
    """
    num, den = 0.0, 0.0
    for path, w in weights.items():
        v = _get_metric_by_path(metrics, path)
        if v is None:
            continue
        v = float(np.clip(v, 0.0, 1.0))
        num += w * v
        den += w
    return num / den if den > 0 else float("nan")


def difficulty_score(metrics: Dict, metric_key: str) -> float:
    """Score used for difficulty bucketing. 'composite' or any dotted-key."""
    if metric_key == "composite":
        return composite_score(metrics)
    v = _get_metric_by_path(metrics, metric_key)
    return v if v is not None else float("nan")


def classify_difficulty(score: float, easy_min: float, hard_max: float) -> str:
    if score is None or np.isnan(score):
        return "unknown"
    if score >= easy_min:
        return "easy"
    if score <= hard_max:
        return "hard"
    return "medium"


def annotate_difficulty(per_sample: List[Dict], metric_key: str,
                        easy_min: float, hard_max: float) -> None:
    """In-place: attach 'difficulty' and 'difficulty_score' to each sample."""
    for s in per_sample:
        score = difficulty_score(s.get("metrics", {}), metric_key)
        s["difficulty_score"] = score
        s["difficulty"] = classify_difficulty(score, easy_min, hard_max)


def difficulty_summary(per_sample: List[Dict], metric_key: str,
                       easy_min: float, hard_max: float) -> Dict:
    counts = {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}
    bins: Dict[str, List[str]] = {k: [] for k in counts}
    for s in per_sample:
        d = s.get("difficulty", "unknown")
        counts[d] = counts.get(d, 0) + 1
        bins[d].append(s.get("name", ""))
    return {
        "metric": metric_key,
        "thresholds": {"easy_min": easy_min, "hard_max": hard_max},
        "counts": counts,
        "bins": bins,
    }


def lean_per_sample(per_sample: List[Dict],
                    key_paths: List[str] = LEAN_KEY_SCORE_PATHS) -> List[Dict]:
    """Strip per-sample records down to triage essentials."""
    out: List[Dict] = []
    for s in per_sample:
        m = s.get("metrics", {})
        out.append({
            "name": s.get("name", ""),
            "difficulty": s.get("difficulty", "unknown"),
            "difficulty_score": s.get("difficulty_score"),
            "key_scores": {p: _get_metric_by_path(m, p) for p in key_paths},
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_namelist(namelist_arg: str, pred_dir: str, gt_root: str) -> List[str]:
    if namelist_arg:
        if namelist_arg.endswith(".npy"):
            arr = np.load(namelist_arg, allow_pickle=True)
            return [str(x) for x in np.asarray(arr).ravel().tolist()]
        if namelist_arg.endswith(".txt"):
            with open(namelist_arg, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        # treat as comma-separated
        return [s.strip() for s in namelist_arg.split(",") if s.strip()]

    # auto-discovery: intersect predictions with available GT
    pred_names = set()
    if os.path.isdir(pred_dir):
        for d in os.listdir(pred_dir):
            full = os.path.join(pred_dir, d)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "basic_info.txt")):
                pred_names.add(d)
    gt_names = set()
    json_root = os.path.join(gt_root, "finaljson")
    if os.path.isdir(json_root):
        for f in os.listdir(json_root):
            if f.endswith(".json"):
                gt_names.add(f[:-5])
    return sorted(pred_names & gt_names)


@lru_cache(maxsize=None)
def _load_gt_cached(gt_root: str, name: str, voxel_res: int,
                    n_surface_pts: int, seed: int) -> SampleGT:
    """Process-wide GT cache; safe because evaluate_sample never mutates GT."""
    return load_gt(gt_root, name, voxel_res=voxel_res,
                   n_surface_pts=n_surface_pts, seed=seed)


def evaluate_method(pred_dir: str, gt_root: str, namelist_arg: str, out_path: str,
                    voxel_res: int, n_surface_pts: int, match_alpha: float,
                    seed: int, difficulty_metric: str, easy_min: float,
                    hard_max: float, full: bool, verbose: bool) -> Optional[Dict]:
    """Evaluate one method directory; write JSON to ``out_path``; return summary."""
    names = _resolve_namelist(namelist_arg, pred_dir, gt_root)
    if not names:
        print(f"[evaluation_vlm] no samples under {pred_dir}", file=sys.stderr)
        return None

    print(f"[evaluation_vlm] {pred_dir}: evaluating {len(names)} samples", flush=True)

    per_sample: List[Dict] = []
    failed: List[Dict] = []
    for k, name in enumerate(names):
        try:
            pred = load_pred(pred_dir, name, voxel_res=voxel_res)
            gt = _load_gt_cached(gt_root, name, voxel_res, n_surface_pts, seed)
            metrics = evaluate_sample(pred, gt, alpha=match_alpha)
            per_sample.append({"name": name, "metrics": metrics})
            if verbose:
                obj_iou = metrics["geometry"]["object"]["iou"]
                part_iou = metrics["geometry"]["part"]["mean_iou"]
                print(f"  [{k+1}/{len(names)}] {name}: obj_IoU={obj_iou:.3f}  "
                      f"part_meanIoU={part_iou:.3f}", flush=True)
        except Exception as e:
            failed.append({"name": name, "error": f"{type(e).__name__}: {e}"})
            print(f"  [WARN] skip {name}: {e}", file=sys.stderr)

    annotate_difficulty(per_sample, difficulty_metric, easy_min, hard_max)
    diff_summary = difficulty_summary(per_sample, difficulty_metric, easy_min, hard_max)
    nested_agg, flat_agg = _aggregate(per_sample)
    per_sample_out = per_sample if full else lean_per_sample(per_sample)

    summary = {
        "config": {
            "pred_dir": os.path.abspath(pred_dir),
            "gt_root": os.path.abspath(gt_root),
            "voxel_res": voxel_res,
            "n_surface_pts": n_surface_pts,
            "match_alpha": match_alpha,
            "seed": seed,
            "difficulty_metric": difficulty_metric,
            "difficulty_thresholds": {"easy_min": easy_min, "hard_max": hard_max},
            "per_sample_mode": "full" if full else "lean",
        },
        "num_samples": len(per_sample),
        "num_failed": len(failed),
        "aggregate": nested_agg,
        "aggregate_flat": flat_agg,
        "difficulty": diff_summary,
        "per_sample": per_sample_out,
        "failed": failed,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[evaluation_vlm] saved -> {out_path}")
    counts = diff_summary["counts"]
    print(f"[evaluation_vlm] difficulty ({difficulty_metric}, "
          f"easy>={easy_min}, hard<={hard_max}): "
          f"easy={counts['easy']} medium={counts['medium']} "
          f"hard={counts['hard']} unknown={counts['unknown']}")
    print("[evaluation_vlm] aggregate:")
    for k, v in flat_agg.items():
        print(f"  {k:40s} {v:.4f}" if not np.isnan(v) else f"  {k:40s} nan")
    return summary


def _discover_methods(pred_root: str) -> List[str]:
    """Return method names = sub-dirs containing at least one <sample>/basic_info.txt."""
    if not os.path.isdir(pred_root):
        return []
    out = []
    for d in sorted(os.listdir(pred_root)):
        full = os.path.join(pred_root, d)
        if not os.path.isdir(full):
            continue
        for s in os.listdir(full):
            if os.path.exists(os.path.join(full, s, "basic_info.txt")):
                out.append(d)
                break
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Direct evaluation of VLM-stage outputs against PhysX-Net GT."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred_dir", default=None,
                     help="single-method dir containing one <name>/ per sample")
    src.add_argument("--pred_root", default=None,
                     help="batch mode: dir whose sub-dirs are method names "
                          "(e.g. ./outputs)")
    parser.add_argument("--methods", default=None,
                        help="(batch mode) comma-separated method whitelist; "
                             "default = auto-discover all under --pred_root")
    parser.add_argument("--gt_root", default="./dataset/physxnet",
                        help="PhysX-Net root holding finaljson/ and partseg/")
    parser.add_argument("--namelist", default="",
                        help="optional sample list: .npy / .txt / comma-separated. "
                             "Empty => auto-discover intersection of pred & GT.")
    parser.add_argument("--out", default=None,
                        help="single-method output path; default: "
                             "<pred_dir>/vlm_eval.json. Ignored in batch mode.")
    parser.add_argument("--out_root", default="./eval_results",
                        help="(batch mode) results land at "
                             "<out_root>/<method>/vlm_eval.json")
    parser.add_argument("--voxel_res", type=int, default=32)
    parser.add_argument("--n_surface_pts", type=int, default=81920,
                        help="surface samples per part for GT voxelization")
    parser.add_argument("--match_alpha", type=float, default=0.7,
                        help="weight on geometry term in Hungarian cost")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty_metric", default="composite",
                        help="'composite' or any dotted metric key "
                             "(higher-is-better; e.g. geometry.object.iou)")
    parser.add_argument("--difficulty_thresholds", default="0.6,0.3",
                        help="comma-separated 'easy_min,hard_max' "
                             "(score>=easy_min -> easy; score<=hard_max -> hard)")
    parser.add_argument("--full", action="store_true",
                        help="keep full per-sample metrics in JSON (default: lean)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        easy_min_str, hard_max_str = args.difficulty_thresholds.split(",")
        easy_min, hard_max = float(easy_min_str), float(hard_max_str)
    except ValueError:
        parser.error("--difficulty_thresholds must be 'easy_min,hard_max', "
                     f"got {args.difficulty_thresholds!r}")
    if not (hard_max <= easy_min):
        parser.error(f"thresholds must satisfy hard_max <= easy_min, "
                     f"got easy_min={easy_min}, hard_max={hard_max}")

    common = dict(
        gt_root=args.gt_root,
        namelist_arg=args.namelist,
        voxel_res=args.voxel_res,
        n_surface_pts=args.n_surface_pts,
        match_alpha=args.match_alpha,
        seed=args.seed,
        difficulty_metric=args.difficulty_metric,
        easy_min=easy_min,
        hard_max=hard_max,
        full=args.full,
        verbose=args.verbose,
    )

    # --- single-method mode ---
    if args.pred_dir is not None:
        out_path = args.out or os.path.join(args.pred_dir, "vlm_eval.json")
        summary = evaluate_method(pred_dir=args.pred_dir, out_path=out_path, **common)
        if summary is None:
            sys.exit(1)
        return

    # --- batch mode ---
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = _discover_methods(args.pred_root)
    if not methods:
        print(f"[evaluation_vlm] no methods discovered under {args.pred_root}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[evaluation_vlm] batch mode: {len(methods)} method(s) "
          f"-> {methods}", flush=True)

    overall: Dict[str, Dict] = {}
    for m in methods:
        pred_dir = os.path.join(args.pred_root, m)
        out_path = os.path.join(args.out_root, m, "vlm_eval.json")
        print(f"\n[evaluation_vlm] === method: {m} ===", flush=True)
        summary = evaluate_method(pred_dir=pred_dir, out_path=out_path, **common)
        if summary is None:
            print(f"[evaluation_vlm] skipped {m}: no samples", file=sys.stderr)
            continue
        overall[m] = {
            "aggregate_flat": summary["aggregate_flat"],
            "difficulty_counts": summary["difficulty"]["counts"],
            "num_samples": summary["num_samples"],
            "num_failed": summary["num_failed"],
        }

    if overall:
        idx_path = os.path.join(args.out_root, "vlm_eval_index.json")
        os.makedirs(args.out_root, exist_ok=True)
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2, ensure_ascii=False)
        print(f"\n[evaluation_vlm] batch index -> {idx_path}")


if __name__ == "__main__":
    main()
