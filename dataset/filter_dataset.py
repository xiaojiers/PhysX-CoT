"""
filter_dataset.py       outlier ID      CoT V3 JSONL

   
  # Step 1           outlier ID       20    
  python3 length_statistics.py \\
      --tokenizer Qwen/Qwen3-VL-8B-Instruct \\
      --max_filter 8192

  # Step 2    JSONL
  python3 filter_dataset.py \\
      --jsonl   cot_finetune_v3/training_set_0_cot_v3.jsonl \\
      --ids     cot_finetune_v3/outliers_above_8192_ids.txt \\
      --output  cot_finetune_v3/training_set_0_cot_v3_filtered.jsonl
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",  required=True, help="   JSONL     ")
    parser.add_argument("--ids",    required=True, help="outlier ID         ID ")
    parser.add_argument("--output", required=True, help="       JSONL   ")
    args = parser.parse_args()

    jsonl_path  = Path(args.jsonl)
    ids_path    = Path(args.ids)
    output_path = Path(args.output)

    if not jsonl_path.exists():
        print(f"[ERROR] JSONL    : {jsonl_path}")
        return
    if not ids_path.exists():
        print(f"[ERROR] ID      : {ids_path}")
        return

    #    outlier ID   
    outlier_ids: set[str] = set()
    with open(ids_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                outlier_ids.add(s)
    print(f"[INFO]    {len(outlier_ids):,}   outlier ID")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = skipped = parse_err = 0

    #              
    total_lines = sum(1 for ln in open(jsonl_path, encoding="utf-8") if ln.strip())

    def _iter():
        with open(jsonl_path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    yield ln

    line_iter = tqdm(_iter(), total=total_lines, unit=" ", dynamic_ncols=True) \
                if _HAS_TQDM else _iter()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in line_iter:
            total += 1
            try:
                sample = json.loads(line)
                sid = sample.get("id", "")
                if sid in outlier_ids:
                    skipped += 1
                else:
                    out_f.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
            except json.JSONDecodeError:
                parse_err += 1

    print(f"\n[  ]")
    print(f"        : {total:,}")
    print(f"     outlier: {skipped:,} {skipped/total*100:.2f}% ")
    print(f"        : {kept:,} {kept/total*100:.2f}% ")
    if parse_err:
        print(f"        : {parse_err:,}")
    print(f"        : {output_path}")


if __name__ == "__main__":
    main()
