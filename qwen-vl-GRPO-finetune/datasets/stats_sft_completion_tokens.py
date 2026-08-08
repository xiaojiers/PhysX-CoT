#!/usr/bin/env python3
"""
   SFT V3 JSONL   assistant    token           GRPO   --max_completion_length 

            
         2   gpt turn Turn1 = think + <overall> Turn2 = <geometry_l_k> RLE
        GRPO      completion             tokenize(turn1 + "\\n" + turn2)

   
  python datasets/stats_sft_completion_tokens.py \\
      --jsonl ../data/cot_finetune_v3/training_set_0_cot_v3_filter.jsonl \\
      --tokenizer Qwen/Qwen3-VL-8B-Instruct \\
      --max_samples 50000

    transformers     --approx_chars_per_token        3.5       
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def round_up_mult(x: float, m: int) -> int:
    return int(math.ceil(x / m) * m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default=None, help="HF        tokenizer ")
    ap.add_argument("--max_samples", type=int, default=0, help="0     ")
    ap.add_argument("--approx_chars_per_token", type=float, default=0.0,
                    help=">0     tokenizer   len(text)/ratio    token  ")
    ap.add_argument("--margin", type=float, default=1.05, help="    =       margin")
    ap.add_argument("--align", type=int, default=128, help="        ")
    args = ap.parse_args()

    if args.approx_chars_per_token > 0:
        def cnt(text: str) -> int:
            if not text:
                return 0
            return max(1, int(len(text) / args.approx_chars_per_token))
        tok_mode = f"approx len/{args.approx_chars_per_token}"
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            sys.stderr.write(
                "    transformers   pip install transformers     --tokenizer "
                "    --approx_chars_per_token 3.5     \n"
            )
            raise SystemExit(1) from exc
        if not args.tokenizer:
            sys.stderr.write("    --tokenizer   --approx_chars_per_token\n")
            raise SystemExit(1)
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True, local_files_only=True
        )

        def cnt(text: str) -> int:
            if not text:
                return 0
            return len(tok.encode(text, add_special_tokens=False))

        tok_mode = str(args.tokenizer)

    t1s, t2s, sums = [], [], []
    path = Path(args.jsonl)
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            gpt = [c["value"] for c in o.get("conversations", []) if c.get("from") == "gpt"]
            if len(gpt) != 2:
                continue
            a, b = gpt[0], gpt[1]
            c1, c2 = cnt(a), cnt(b)
            t1s.append(c1)
            t2s.append(c2)
            sums.append(cnt(a + "\n" + b))
            n += 1
            if args.max_samples and n >= args.max_samples:
                break

    def report(name: str, vals: list[int]) -> None:
        vals = sorted(vals)
        print(f"\n=== {name} ===")
        print(f"  n={len(vals)}  mean={sum(vals)/len(vals):.1f}")
        print(
            f"  min={vals[0]}  p50={percentile(vals,50):.0f}  p90={percentile(vals,90):.0f}  "
            f"p95={percentile(vals,95):.0f}  p99={percentile(vals,99):.0f}  max={vals[-1]}"
        )

    print(f"tokenizer /     : {tok_mode}")
    print(f"samples used: {n}")
    report("Turn1 think + overall ", t1s)
    report("Turn2 geometry RLE ", t2s)
    report("Turn1+Turn2       GRPO      ", sums)

    sv = sorted(sums)
    p95, p99 = percentile(sv, 95), percentile(sv, 99)
    rec95 = round_up_mult(p95 * args.margin, args.align)
    rec99 = round_up_mult(p99 * args.margin, args.align)

    over_4096 = sum(1 for v in sums if v > 4096)
    print(f"\n===    --max_completion_length ===")
    print(f"     p95 {args.margin}    {args.align} : {rec95}")
    print(f"     p99 {args.margin}    {args.align} : {rec99}")
    print(f"\n  max_completion_length=4096       {over_4096}/{n} "
          f"({100.0 * over_4096 / max(n,1):.2f}%)")


if __name__ == "__main__":
    main()
