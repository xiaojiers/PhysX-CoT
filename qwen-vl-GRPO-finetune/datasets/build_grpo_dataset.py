"""
datasets/build_grpo_dataset.py

  V3 SFT      cot_finetune_v3/training_set_0_cot_v3*.jsonl    
GRPO           SFT JSONL   GRPO JSONL          
              

     
  1.   (obj_id, img_id)       SFT          part   GRPO   
           GRPO    turn-2 rollout       part   geometry 
  2.    parsers.completion_parser.CompletionParser   V3     
     Turn-1 GPT response   think_steps + overall_dict   GT  
  3.      GT    voxel_dir    dataset_root  
     RewardEngine       part_k     ind_{k}.npy 
  4. Object         category    SFT          
  5.      + sample             

   
    grpo_train.jsonl
    grpo_val.jsonl
    grpo_sample.json    pretty-printed slice
    quality_report.json
    split_stats.json

       ExampleAdapter / RewardEngine       
{
  "id":           "<obj_id>_<img_id>",
  "image":        "<obj_id>_/<img_id>.png",               image_root 
  "sam_feature":  "sam_feature/<obj_id>/<img_id>.npz" | null,
  "messages":     [{role: user, content:[{type:image},{type:text,text:overall_prompt}]}],
  "gt": {
      "part_count":   int,
      "think_steps":  {...},                      CompletionParser.parse_think_steps   
      "overall_dict": {...},                      CompletionParser.parse_overall      
      "voxel_dir":    "tmp/partseg/<obj_id>/32"    dataset_root
  },
  "meta": {
      "split":      "train" | "val",
      "category":   <Category>,
      "part_count": int,
      "complexity": "easy" | "medium" | "hard",
      "quality_ok": bool
  }
}

   
    cd qwen-vl-GRPO-finetune
    python datasets/build_grpo_dataset.py \
        --sft_jsonl   ../data/cot_finetune_v3/training_set_0_cot_v3_filter.jsonl \
        --image_root  ../data/renders_cond \
        --voxel_root  ../data/tmp/partseg \
        --voxel_grid  32 \
        --out_dir     ./datasets \
        --val_ratio   0.10 \
        --seed        42
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

#      cd   qwen-vl-GRPO-finetune/    import    configs/ + parsers/
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from configs import TaskConfig
from parsers.completion_parser import CompletionParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


#                                                                          

COMPLEXITY_EASY   = "easy"
COMPLEXITY_MEDIUM = "medium"
COMPLEXITY_HARD   = "hard"


def _complexity(part_count: int) -> str:
    if part_count <= 2:
        return COMPLEXITY_EASY
    if part_count <= 5:
        return COMPLEXITY_MEDIUM
    return COMPLEXITY_HARD


#    SFT item   think_text + overall_text                                      

def _split_sft_response(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    """
    SFT       conversations [Human, GPT, Human, GPT]
      Turn-1   GPT response    <think>...</think>   <overall>...</overall> 

    Returns:
        think_text, overall_text, raw_gpt_response
    """
    convs = item.get("conversations", [])
    gpt_resp = ""
    for c in convs:
        if c.get("from") == "gpt":
            gpt_resp = c.get("value", "")
            break

    think_text   : Optional[str] = None
    overall_text : Optional[str] = None

    if "<think>" in gpt_resp and "</think>" in gpt_resp:
        think_text = gpt_resp.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    after = gpt_resp.split("</think>", 1)[1] if "</think>" in gpt_resp else gpt_resp
    if "<overall>" in after and "</overall>" in after:
        overall_text = after.split("<overall>", 1)[1].split("</overall>", 1)[0].strip()
    elif after.strip():
        #     SFT      <overall>    
        overall_text = after.strip()

    return think_text, overall_text, gpt_resp


#                                                                           

def _check_quality(
    sample_id:    str,
    think_steps:  Dict[str, Any],
    overall_dict: Optional[Dict[str, Any]],
    voxel_dir_abs: Optional[str],
    part_count:    int,
) -> List[str]:
    issues: List[str] = []

    if overall_dict is None:
        issues.append("overall_parse_failed: missing <overall> or unparsable")
        return issues

    steps_found = set(think_steps.get("steps_found") or [])
    if isinstance(steps_found, (list, tuple)):
        steps_found = set(steps_found)
    if len(steps_found) < 5:
        missing = {1, 2, 3, 4, 5} - steps_found
        issues.append(f"think_steps_incomplete: missing {sorted(missing)}")

    think_n  = think_steps.get("part_count")
    struct_n = len(overall_dict.get("parts") or {})
    if think_n is not None and think_n != struct_n:
        issues.append(f"part_count_mismatch: think={think_n} overall={struct_n}")

    # bbox_2d   
    for pid, b in (think_steps.get("bbox_2d") or {}).items():
        if len(b) != 4:
            issues.append(f"bbox2d_format: {pid}={b}")
            continue
        xmin, xmax, ymin, ymax = b
        if not (0.0 <= xmin <= xmax <= 1.0 and 0.0 <= ymin <= ymax <= 1.0):
            issues.append(f"bbox2d_oob: {pid}={b}")

    # bbox_3d   
    for pid, b in (think_steps.get("bbox_3d") or {}).items():
        if len(b) != 6:
            issues.append(f"bbox3d_format: {pid}={b}")
            continue
        if not all(0 <= v < 32 for v in b):
            issues.append(f"bbox3d_oob: {pid}={b}")

    #    GT       
    if voxel_dir_abs and not os.path.isdir(voxel_dir_abs):
        issues.append(f"voxel_dir_missing: {voxel_dir_abs}")
    elif voxel_dir_abs and part_count > 0:
        n_avail = sum(
            1 for k in range(part_count)
            if os.path.exists(os.path.join(voxel_dir_abs, f"ind_{k}.npy"))
        )
        if n_avail < part_count:
            issues.append(f"voxel_partial: {n_avail}/{part_count} parts found")

    return issues


#                                                                         

def _build_grpo_sample(
    item:           Dict[str, Any],
    parser:         CompletionParser,
    overall_prompt: str,
    voxel_root_rel: str,
    voxel_grid:     int,
    voxel_root_abs: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Returns:
        sample (dict)  GRPO JSONL           None
        issues (list)                    sample            
    """
    obj_id, img_id = _split_sample_id(item.get("id", ""))
    if not obj_id:
        return None, ["sample_id_invalid"]

    think_text, overall_text, gpt_resp = _split_sft_response(item)
    think_steps  = parser.parse_think_steps(think_text)
    overall_dict = parser.parse_overall(overall_text)

    # steps_found   set      JSON    
    if isinstance(think_steps.get("steps_found"), set):
        think_steps["steps_found"] = sorted(think_steps["steps_found"])

    voxel_dir_rel = os.path.join(voxel_root_rel, obj_id, str(voxel_grid))
    voxel_dir_abs = (
        os.path.join(voxel_root_abs, obj_id, str(voxel_grid))
        if voxel_root_abs else None
    )

    part_count = think_steps.get("part_count") or len(
        (overall_dict or {}).get("parts") or {}
    )
    category = (overall_dict or {}).get("category") or "Unknown"

    issues = _check_quality(item.get("id", ""), think_steps, overall_dict,
                            voxel_dir_abs, part_count)
    if overall_dict is None:
        return None, issues

    sample = {
        "id":          item.get("id", f"{obj_id}_{img_id}"),
        "image":       item.get("image", os.path.join(f"{obj_id}_", f"{img_id}.png")),
        "sam_feature": item.get("sam_feature"),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": overall_prompt},
                ],
            }
        ],
        "gt": {
            "part_count":   part_count,
            "think_steps":  think_steps,
            "overall_dict": overall_dict,
            "voxel_dir":    voxel_dir_rel,
            #          RewardEngine fallback enrich       jsonl    
            #               "text": gpt_resp 
        },
        "meta": {
            "split":      "",
            "category":   category,
            "part_count": part_count,
            "complexity": _complexity(part_count),
            "quality_ok": len(issues) == 0,
        },
    }
    return sample, issues


def _split_sample_id(sid: str) -> Tuple[str, str]:
    """'10049_000'   ('10049', '000') """
    if not sid:
        return "", ""
    parts = sid.rsplit("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (sid, "")


#    Train/Val    object     category                                 

def _split_by_object(
    samples:   List[Dict[str, Any]],
    val_ratio: float,
    seed:      int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)

    cat2objs: Dict[str, Set[str]] = defaultdict(set)
    for s in samples:
        obj_id, _ = _split_sample_id(s["id"])
        cat2objs[s["meta"]["category"]].add(obj_id)

    val_objs: Set[str] = set()
    for cat, objs in cat2objs.items():
        obj_list = sorted(objs)
        rng.shuffle(obj_list)
        n_val = max(1, round(len(obj_list) * val_ratio)) if len(obj_list) >= 2 else 0
        val_objs.update(obj_list[:n_val])

    train, val = [], []
    for s in samples:
        obj_id, _ = _split_sample_id(s["id"])
        if obj_id in val_objs:
            s["meta"]["split"] = "val"
            val.append(s)
        else:
            s["meta"]["split"] = "train"
            train.append(s)
    return train, val


#                                                                             

def _compute_stats(
    train: List[Dict[str, Any]],
    val:   List[Dict[str, Any]],
) -> Dict[str, Any]:

    def _stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        cat_cnt:  Dict[str, int] = defaultdict(int)
        comp_cnt: Dict[str, int] = defaultdict(int)
        quality_ok = 0
        for s in samples:
            cat_cnt[s["meta"]["category"]] += 1
            comp_cnt[s["meta"]["complexity"]] += 1
            quality_ok += int(s["meta"]["quality_ok"])
        obj_ids = {_split_sample_id(s["id"])[0] for s in samples}
        return {
            "total_samples":  len(samples),
            "unique_objects": len(obj_ids),
            "quality_ok":     quality_ok,
            "quality_issue_ratio": round(1 - quality_ok / max(len(samples), 1), 4),
            "complexity":     dict(comp_cnt),
            "top_categories": dict(sorted(cat_cnt.items(), key=lambda x: -x[1])[:20]),
        }

    train_obj = {_split_sample_id(s["id"])[0] for s in train}
    val_obj   = {_split_sample_id(s["id"])[0] for s in val}
    return {
        "train":          _stats(train),
        "val":            _stats(val),
        "object_overlap": len(train_obj & val_obj),  #     0
    }


#                                                                          

def _write_sample_slice(
    train:       List[Dict[str, Any]],
    val:         List[Dict[str, Any]],
    out_path:    str,
    n_per_complexity: int,
    seed:        int,
) -> None:
    rng = random.Random(seed)

    def _pick(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_comp: Dict[str, List] = defaultdict(list)
        for s in samples:
            by_comp[s["meta"]["complexity"]].append(s)
        picked: List[Dict[str, Any]] = []
        for comp in (COMPLEXITY_EASY, COMPLEXITY_MEDIUM, COMPLEXITY_HARD):
            pool = list(by_comp.get(comp, []))
            rng.shuffle(pool)
            picked.extend(pool[:n_per_complexity])
        return picked

    payload = {
        "_note": (
            f"Sample slice for quick inspection. "
            f"{n_per_complexity} samples per complexity tier from train and val."
        ),
        "train": [copy.deepcopy(s) for s in _pick(train)],
        "val":   [copy.deepcopy(s) for s in _pick(val)],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("       %s", out_path)


#                                                                            

def main() -> None:
    parser = argparse.ArgumentParser(description="V3 SFT JSONL   GRPO JSONL")
    parser.add_argument("--sft_jsonl",   required=True,
                        help="SFT V3     JSONL   sam_feature    ")
    parser.add_argument("--image_root",  default="",
                        help="      renders_cond/          sample.image      ")
    parser.add_argument("--voxel_root",  default="",
                        help="   GT          /data/tmp/partseg         ")
    parser.add_argument("--voxel_root_rel", default="tmp/partseg",
                        help="   sample.gt.voxel_dir            dataset_root ")
    parser.add_argument("--voxel_grid",  type=int, default=32,
                        help="         ind_{k}.npy        ")
    parser.add_argument("--out_dir",     default="./datasets")
    parser.add_argument("--val_ratio",   type=float, default=0.10)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--sample_n",    type=int,   default=3,
                        help="        complexity tier     ")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    task_cfg = TaskConfig()
    cparser  = CompletionParser(task_cfg)

    #    Step 1:                                                         
    logger.info("   SFT JSONL: %s", args.sft_jsonl)
    seen_ids: Set[str] = set()
    unique_items: List[Dict[str, Any]] = []
    total_lines = 0
    with open(args.sft_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            item = json.loads(line)
            sid = item.get("id", "")
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            unique_items.append(item)
    logger.info("SFT    =%d    =%d     %.1fx ",
                total_lines, len(unique_items),
                total_lines / max(len(unique_items), 1))

    #    Step 2:    GRPO                                                  
    grpo_samples:   List[Dict[str, Any]] = []
    quality_report: List[Dict[str, Any]] = []
    skipped = 0
    for item in unique_items:
        sample, issues = _build_grpo_sample(
            item            = item,
            parser          = cparser,
            overall_prompt  = task_cfg.overall_prompt,
            voxel_root_rel  = args.voxel_root_rel,
            voxel_grid      = args.voxel_grid,
            voxel_root_abs  = args.voxel_root or None,
        )
        if issues:
            quality_report.append({"id": item.get("id", ""), "issues": issues})
        if sample is None:
            skipped += 1
            continue
        grpo_samples.append(sample)
    logger.info("         =%d   =%d     =%d",
                len(grpo_samples), skipped, len(quality_report))

    #    Step 3:                                                            
    train_samples, val_samples = _split_by_object(
        grpo_samples, val_ratio=args.val_ratio, seed=args.seed,
    )
    logger.info("  : train=%d val=%d", len(train_samples), len(val_samples))

    #    Step 4:                                                           
    def _write_jsonl(path: str, samples: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info("    %d     %s", len(samples), path)

    _write_jsonl(os.path.join(args.out_dir, "grpo_train.jsonl"), train_samples)
    _write_jsonl(os.path.join(args.out_dir, "grpo_val.jsonl"),   val_samples)

    qr_path = os.path.join(args.out_dir, "quality_report.json")
    with open(qr_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)
    logger.info("     %d     %s", len(quality_report), qr_path)

    stats = _compute_stats(train_samples, val_samples)
    stats_path = os.path.join(args.out_dir, "split_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("     %s", stats_path)

    _write_sample_slice(
        train_samples, val_samples,
        out_path           = os.path.join(args.out_dir, "grpo_sample.json"),
        n_per_complexity   = args.sample_n,
        seed               = args.seed,
    )

    #    Summary                                                             
    logger.info("=" * 60)
    logger.info("TRAIN: %d    / %d    / quality_ok=%d",
                stats["train"]["total_samples"],
                stats["train"]["unique_objects"],
                stats["train"]["quality_ok"])
    logger.info("VAL:   %d    / %d    / quality_ok=%d",
                stats["val"]["total_samples"],
                stats["val"]["unique_objects"],
                stats["val"]["quality_ok"])
    logger.info("Object        0 : %d", stats["object_overlap"])
    logger.info("Complexity    (train): %s", stats["train"]["complexity"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
