"""Physical-field evaluation for PhysX-CoT compatible outputs.

Metrics:
  - absolute scale error
  - affordance PSNR
  - material PSNR
  - description-region PSNR

The loader accepts both `objs/<label>.obj` and
`objs/<label>/<label>.obj`, and skips incomplete samples instead of
terminating the whole evaluation batch.
"""

import argparse
import json
import os
from typing import Any, Dict, Optional

import clip
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from trellis.representations.mesh.cube2mesh import MeshExtractResult
from trellis.utils import render_utils


def load_obj_geometry_fast(path: str) -> trimesh.Trimesh:
    vertices, faces = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return trimesh.Trimesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _load_part_obj(root: str, name: str, objfile: Any) -> trimesh.Trimesh:
    candidates = [
        os.path.join(root, name, "objs", str(objfile) + ".obj"),
        os.path.join(root, name, "objs", str(objfile), str(objfile) + ".obj"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_obj_geometry_fast(path)
    raise FileNotFoundError(f"part OBJ not found; tried: {candidates}")


def _load_eval_part_obj(
    root: str,
    name: str,
    part_info: Dict[str, Any],
    index: int,
) -> trimesh.Trimesh:
    labels = [part_info.get("label", index), index]
    tried = []
    for label in dict.fromkeys(labels):
        candidates = [
            os.path.join(root, name, "objs", str(label), str(label) + ".obj"),
            os.path.join(root, name, "objs", str(label) + ".obj"),
        ]
        tried.extend(candidates)
        for path in candidates:
            if os.path.exists(path):
                return load_obj_geometry_fast(path)
    raise FileNotFoundError(f"eval part OBJ not found; tried: {tried}")


def mov(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center to origin and scale so the longest bbox axis becomes 1."""
    if len(mesh.vertices) == 0:
        raise ValueError("cannot normalize an empty mesh")
    bbox_max = np.asarray(mesh.vertices).max(0)
    bbox_min = np.asarray(mesh.vertices).min(0)
    longest_axis = float(max(bbox_max - bbox_min))
    if longest_axis <= 0:
        raise ValueError("cannot normalize a zero-size mesh")
    offset = -(bbox_min + bbox_max) / 2
    mesh.apply_translation(offset)
    mesh.apply_transform(trimesh.transformations.scale_matrix(1.0 / longest_axis))
    return mesh


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return 50.0
    return 10.0 * float(np.log10((data_range ** 2) / mse))


def _number_from_field(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).split(" ")[0])
    except Exception:
        return float(default)


def _safe_norm_image(img: np.ndarray) -> np.ndarray:
    max_value = float(np.max(img)) if img.size else 0.0
    return img if max_value <= 1e-8 else img / max_value


def _longest_dimension(info: Dict[str, Any]) -> float:
    raw = str(info["dimension"]).split(" ")[0]
    return max(float(value) for value in raw.split("*"))


def _description_part_index(name: str, part_count: int) -> int:
    if part_count <= 0:
        raise ValueError("sample has no GT parts")
    return sum(name.encode("utf-8")) % part_count


def _clip_weights_path() -> Optional[str]:
    candidates = [
        os.environ.get("CLIP_VIT_L14_PATH", ""),
        os.path.join(os.path.dirname(__file__), "pretrain", "clip", "ViT-L-14.pt"),
    ]
    return next((path for path in candidates if path and os.path.exists(path)), None)


def _mean_or_nan(values) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate_physics(
    resultpath: str = "./test_demo",
    datasetpath: str = "./PhysX_mobility",
    namelist_path: str = "./val_test_list.npy",
    num_frames: int = 30,
    save_json: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    jsonpath = os.path.join(datasetpath, "finaljson")
    meshpath = os.path.join(datasetpath, "partseg")
    namelist = np.load(namelist_path, allow_pickle=True)

    clip_weight = _clip_weights_path()
    clipmodel, _ = clip.load(clip_weight or "ViT-L/14", jit=False)
    clipmodel = clipmodel.eval().cuda()

    allscale, allaffordance, allmaterial, alldescription = [], [], [], []
    per_sample: Dict[str, Dict[str, Any]] = {}
    n_skip = 0

    for raw_name in namelist:
        name = str(raw_name)
        try:
            gt_json_path = os.path.join(jsonpath, name + ".json")
            eval_json_path = os.path.join(resultpath, name, "basic_info.json")
            if not (os.path.exists(gt_json_path) and os.path.exists(eval_json_path)):
                raise FileNotFoundError("missing GT or eval basic_info.json")

            with open(gt_json_path, "r") as fp:
                jsongtdata = json.load(fp)
            with open(eval_json_path, "r") as fp:
                jsonevaldata = json.load(fp)

            scale_error = abs(_longest_dimension(jsongtdata) - _longest_dimension(jsonevaldata))

            description_ind = _description_part_index(name, len(jsongtdata["parts"]))
            text_description = jsongtdata["parts"][description_ind].get("Basic_description", "")
            info_emb_gt = clipmodel.encode_text(clip.tokenize(text_description).cuda()).float()

            allrenobj_gt = trimesh.Trimesh([])
            allsourcedata_gt = np.zeros((0, 3))
            for index, part_info in enumerate(jsongtdata["parts"]):
                eachobj_gt = trimesh.Trimesh([])
                for objfile in part_info.get("obj", [part_info.get("label", index)]):
                    eachobj_gt = trimesh.util.concatenate(
                        [_load_part_obj(meshpath, name, objfile), eachobj_gt]
                    )
                allrenobj_gt = trimesh.util.concatenate([eachobj_gt, allrenobj_gt])
                sourcedata = np.zeros((len(eachobj_gt.vertices), 3))
                sourcedata[:, 0] = _number_from_field(part_info.get("priority_rank", 0))
                sourcedata[:, 1] = _number_from_field(part_info.get("density", 0))
                sourcedata[:, 2] = 1 if index == description_ind else 0
                allsourcedata_gt = np.concatenate([sourcedata, allsourcedata_gt])

            description_scores = []
            for part_info in jsonevaldata["parts"]:
                description = part_info.get("Basic_description", part_info.get("name", ""))
                info_emb_eval = clipmodel.encode_text(clip.tokenize(description).cuda()).float()
                description_scores.append(F.cosine_similarity(info_emb_eval, info_emb_gt, dim=1))
            if not description_scores:
                raise ValueError("sample has no evaluated parts")
            description_ind_eval = int(torch.cat(description_scores).cpu().argmax())

            allrenobj_eval = trimesh.Trimesh([])
            allsourcedata_eval = np.zeros((0, 3))
            for index, part_info in enumerate(jsonevaldata["parts"]):
                eachpart = _load_eval_part_obj(resultpath, name, part_info, index)
                allrenobj_eval = trimesh.util.concatenate([eachpart, allrenobj_eval])
                sourcedata = np.zeros((len(eachpart.vertices), 3))
                sourcedata[:, 0] = _number_from_field(part_info.get("priority_rank", 0))
                sourcedata[:, 1] = _number_from_field(part_info.get("density", 0))
                sourcedata[:, 2] = 1 if index == description_ind_eval else 0
                allsourcedata_eval = np.concatenate([sourcedata, allsourcedata_eval])

            rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            allrenobj_gt.apply_transform(rotation_matrix)
            allrenobj_gt = mov(allrenobj_gt)
            allrenobj_eval = mov(allrenobj_eval)

            gtmesh = MeshExtractResult(
                torch.as_tensor(allrenobj_gt.vertices, dtype=torch.float32).cuda(),
                torch.as_tensor(allrenobj_gt.faces, dtype=torch.int64).cuda(),
                vertex_attrs=None,
                res=64,
                render_vis=torch.as_tensor(allsourcedata_gt, dtype=torch.float32).cuda(),
            )
            evalmesh = MeshExtractResult(
                torch.as_tensor(allrenobj_eval.vertices, dtype=torch.float32).cuda(),
                torch.as_tensor(allrenobj_eval.faces, dtype=torch.int64).cuda(),
                vertex_attrs=None,
                res=64,
                render_vis=torch.as_tensor(allsourcedata_eval, dtype=torch.float32).cuda(),
            )
            video_gt = render_utils.render_video_gt(gtmesh, num_frames=num_frames)
            video_eval = render_utils.render_video_gt(evalmesh, num_frames=num_frames)

            sample_affordance, sample_material, sample_description = [], [], []
            for frame_index in range(len(video_gt["rendervis"])):
                vis_gt = video_gt["rendervis"][frame_index] * video_gt["mask"][frame_index]
                vis_eval = video_eval["rendervis"][frame_index] * video_eval["mask"][frame_index]
                fields_gt = vis_gt.detach().cpu().numpy()
                fields_eval = vis_eval.detach().cpu().numpy()
                sample_affordance.append(psnr(_safe_norm_image(fields_gt[0]), _safe_norm_image(fields_eval[0])))
                sample_material.append(psnr(_safe_norm_image(fields_gt[1]), _safe_norm_image(fields_eval[1])))
                sample_description.append(psnr(fields_gt[2], fields_eval[2]))

            sample_metrics = {
                "scale": float(scale_error),
                "affordance": _mean_or_nan(sample_affordance),
                "material": _mean_or_nan(sample_material),
                "description": _mean_or_nan(sample_description),
            }
            per_sample[name] = sample_metrics
            allscale.append(sample_metrics["scale"])
            allaffordance.extend(sample_affordance)
            allmaterial.extend(sample_material)
            alldescription.extend(sample_description)
            if verbose:
                print(
                    f"{name}: scale={sample_metrics['scale']:.4f}, "
                    f"affordance={sample_metrics['affordance']:.4f}, "
                    f"material={sample_metrics['material']:.4f}, "
                    f"description={sample_metrics['description']:.4f}"
                )
        except Exception as error:
            n_skip += 1
            if verbose:
                print(f"[skip] {name}: {error}")
        finally:
            torch.cuda.empty_cache()

    summary = {
        "scale": _mean_or_nan(allscale),
        "affordance": _mean_or_nan(allaffordance),
        "material": _mean_or_nan(allmaterial),
        "description": _mean_or_nan(alldescription),
        "n_samples_phy": len(per_sample),
        "n_skip": n_skip,
    }
    if save_json:
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w") as fp:
            json.dump(
                {
                    "per_sample": per_sample,
                    "summary": summary,
                    "config": {
                        "resultpath": resultpath,
                        "datasetpath": datasetpath,
                        "namelist_path": namelist_path,
                        "num_frames": num_frames,
                    },
                },
                fp,
                indent=2,
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PhysX physical fields")
    parser.add_argument("--resultpath", default="./test_demo")
    parser.add_argument("--datasetpath", default="./PhysX_mobility")
    parser.add_argument("--namelist", default="./val_test_list.npy")
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--out", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = evaluate_physics(
        resultpath=args.resultpath,
        datasetpath=args.datasetpath,
        namelist_path=args.namelist,
        num_frames=args.num_frames,
        save_json=args.out or None,
        verbose=not args.quiet,
    )
    print("scale: ", result["scale"])
    print("affordance: ", result["affordance"])
    print("material: ", result["material"])
    print("description: ", result["description"])
    print("n_samples_phy: ", result["n_samples_phy"])
    print("n_skip: ", result["n_skip"])


if __name__ == "__main__":
    main()
