"""
        grpo_train.jsonl / grpo_val.jsonl         

   
    python datasets/make_sample.py               #      3  
    python datasets/make_sample.py --n 5         #    5  
    python datasets/make_sample.py --out_dir ./datasets --n 3
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

COMPLEXITY_EASY   = "easy"
COMPLEXITY_MEDIUM = "medium"
COMPLEXITY_HARD   = "hard"


def _load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _trim(sample: dict) -> dict:
    """                      """
    s = copy.deepcopy(sample)
    gt = s.get("gt", {})
    # gt.text Turn1               300   
    if isinstance(gt.get("text"), str) and len(gt["text"]) > 300:
        gt["text"] = gt["text"][:300] + "   [truncated]"
    # think_steps.steps_found     list    JSON     
    ts = gt.get("think_steps", {})
    if isinstance(ts.get("steps_found"), (set, list)):
        ts["steps_found"] = sorted(ts["steps_found"])
    return s


def _pick_by_complexity(samples: list, n: int, rng: random.Random) -> list:
    by_comp: dict = defaultdict(list)
    for s in samples:
        by_comp[s["meta"]["complexity"]].append(s)
    picked = []
    for comp in (COMPLEXITY_EASY, COMPLEXITY_MEDIUM, COMPLEXITY_HARD):
        pool = by_comp.get(comp, [])
        rng.shuffle(pool)
        picked.extend(pool[:n])
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./datasets")
    parser.add_argument("--n",       type=int, default=3,
                        help="   complexity   easy/medium/hard    N      3")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    train = _load_jsonl(str(out_dir / "grpo_train.jsonl"))
    val   = _load_jsonl(str(out_dir / "grpo_val.jsonl"))

    train_picks = _pick_by_complexity(train, args.n, rng)
    val_picks   = _pick_by_complexity(val,   args.n, rng)

    payload = {
        "_note": (
            f"Sample slice for quick inspection. "
            f"train: {len(train_picks)} samples ({args.n} per complexity tier: "
            f"easy/medium/hard), val: {len(val_picks)} samples. "
            f"gt.text is truncated to 300 chars."
        ),
        "train": [_trim(s) for s in train_picks],
        "val":   [_trim(s) for s in val_picks],
    }

    out_path = out_dir / "grpo_sample.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"    train={len(train_picks)} + val={len(val_picks)}       {out_path}")


if __name__ == "__main__":
    main()
