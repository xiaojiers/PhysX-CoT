"""
     Token        CoT V3        

   
  1.    CoT V3 JSONL          token     
  2.      <think>    bbox_3d     RLE       
            (x<<10)|(y<<5)|z        token 
               Qwen3-VL tokenizer      token   
  3.         max_length     
  4.     --max_filter N       token    N   outlier    
                        

   
  #      +    8192     outlier
  python data/length_statistics.py \\
      --tokenizer Qwen/Qwen3-VL-8B-Instruct \\
      --max_filter 8192

  #     5%     
  python data/length_statistics.py \\
      --tokenizer Qwen/Qwen3-VL-8B-Instruct \\
      --sample_rate 0.05
"""

from __future__ import annotations

import json
import math
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

SCRIPT_DIR    = Path(__file__).parent
DEFAULT_JSONL = SCRIPT_DIR / "cot_finetune_v3" / "training_set_0_cot_v3.jsonl"

IMAGE_TOKEN_COUNT = 196   # Qwen3-VL, 448 x 448, after 2 x 2 spatial merge

SAM_FEAT_PAT  = re.compile(r"<sam_feat_l_\d+>")
BBOX3D_PAT    = re.compile(
    r"Part `l_(\d+)`.*?`bbox_3d`\s*=\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
)
GEO_BLOCK_PAT = re.compile(
    r"(<geometry_l_(\d+)>)\s*([\s\S]*?)\s*(</geometry_l_\2>)"
)


#    Tokenizer                                                               

def build_tokenizer(tokenizer_path: str = ""):
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True, use_fast=True
            )
            print(f"[INFO]    Qwen tokenizer (fast): {tokenizer_path}")
            return lambda text: len(tok.encode(text, add_special_tokens=False))
        except Exception as e:
            print(f"[WARN]    Qwen tokenizer    {e}      tiktoken ")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        print("[INFO]    tiktoken cl100k_base   Qwen tokenizer       ")
        return lambda text: len(enc.encode(text))
    except ImportError:
        pass

    print("[INFO] tiktoken        len(text)/3.5      ")
    return lambda text: max(1, math.ceil(len(text) / 3.5))


#              numpy                                              

def _local_rle_to_voxels(rle: str, bbox: list) -> "np.ndarray":
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    dy = y_max - y_min + 1
    dz = z_max - z_min + 1
    ids: list[int] = []
    for tok in rle.split():
        if '-' in tok:
            a, b = map(int, tok.split('-', 1))
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(tok))
    a = np.asarray(ids, dtype=np.int64)
    x = a // (dy * dz) + x_min
    y = a % (dy * dz) // dz + y_min
    z = a % dz + z_min
    return np.stack([x, y, z], axis=1)


def _voxels_to_global_rle(voxels: "np.ndarray") -> str:
    v = np.asarray(voxels, dtype=np.int64)
    ids = sorted(set(((v[:, 0] << 10) | (v[:, 1] << 5) | v[:, 2]).tolist()))
    result: list[str] = []
    s = p = ids[0]
    for n in ids[1:]:
        if n == p + 1:
            p = n
        else:
            result.append(f"{s}-{p}" if s != p else str(s))
            s = p = n
    result.append(f"{s}-{p}" if s != p else str(s))
    return " ".join(result)


def _make_global_geo_block(k: int, local_rle: str, bbox: list) -> str:
    voxels  = _local_rle_to_voxels(local_rle, bbox)
    g_rle   = _voxels_to_global_rle(voxels)
    return f"<geometry_l_{k}>\n{g_rle}\n</geometry_l_{k}>"


#    bbox                                                                

def _parse_bboxes(think_text: str) -> dict:
    bboxes = {}
    for m in BBOX3D_PAT.finditer(think_text):
        k = int(m.group(1))
        bboxes[k] = [int(m.group(i)) for i in range(2, 8)]
    return bboxes


#                                                                    

def estimate_length_dual(sample: dict, count_fn) -> dict:
    """
                        token   

    Returns dict:
      total_local  / total_global :        token       
      text_local   / text_global  :    token  
      geo_local    / geo_global   :      token       gpt turn 
      image        :    token  
      sam_feat     : sam_feat   
      has_compare  :          bbox      
    """
    convs = sample.get("conversations", [])

    #        <think>   gpt turn    bbox
    bboxes: dict = {}
    if _HAS_NUMPY:
        for turn in convs:
            if turn.get("from") == "gpt" and "<think>" in turn.get("value", ""):
                bboxes = _parse_bboxes(turn["value"])
                break

    text_local = text_global = 0
    geo_local  = geo_global  = 0
    n_images   = 0
    n_sam      = 0
    compare_ok = False

    for turn in convs:
        val = turn.get("value", "")
        n_images += val.count("<image>")
        n_sam    += len(SAM_FEAT_PAT.findall(val))

        #       <image>    sam_feat     
        clean = val.replace("<image>", "")
        clean = SAM_FEAT_PAT.sub("X", clean)

        local_toks  = count_fn(clean)
        global_toks = local_toks  #     local   

        #    gpt turn          
        if turn.get("from") == "gpt" and bboxes:
            geo_m = GEO_BLOCK_PAT.search(val)
            if geo_m:
                geo_k     = int(geo_m.group(2))
                local_rle = geo_m.group(3).strip()
                if geo_k in bboxes:
                    try:
                        g_block  = _make_global_geo_block(geo_k, local_rle, bboxes[geo_k])
                        g_clean  = GEO_BLOCK_PAT.sub(g_block, clean)
                        global_toks = count_fn(g_clean)
                        #         
                        geo_clean_local  = f"<geometry_l_{geo_k}>\n{local_rle}\n</geometry_l_{geo_k}>"
                        g_rle_str        = _voxels_to_global_rle(
                            _local_rle_to_voxels(local_rle, bboxes[geo_k])
                        )
                        geo_clean_global = f"<geometry_l_{geo_k}>\n{g_rle_str}\n</geometry_l_{geo_k}>"
                        geo_local  += count_fn(geo_clean_local)
                        geo_global += count_fn(geo_clean_global)
                        compare_ok  = True
                    except Exception:
                        geo_local  += local_toks
                        geo_global += global_toks
                else:
                    geo_local  += local_toks
                    geo_global += global_toks

        text_local  += local_toks
        text_global += global_toks

    img_tokens = n_images * IMAGE_TOKEN_COUNT
    return {
        "total_local":   text_local  + img_tokens,
        "total_global":  text_global + img_tokens,
        "text_local":    text_local,
        "text_global":   text_global,
        "geo_local":     geo_local,
        "geo_global":    geo_global,
        "image":         img_tokens,
        "sam_feat":      n_sam,
        "n_images":      n_images,
        "has_compare":   compare_ok,
    }


#                                                                       

def pct(data: list, p: float) -> float:
    if not data:
        return 0.0
    idx = (len(data) - 1) * p / 100
    lo  = int(idx)
    hi  = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


#    ASCII                                                               

def ascii_histogram(lengths_a: list, lengths_b: list,
                    label_a: str, label_b: str,
                    bins: list, width: int = 36) -> None:
    def _bin(data):
        c = defaultdict(int)
        for v in data:
            for i, b in enumerate(bins):
                if v <= b:
                    c[i] += 1
                    break
            else:
                c[len(bins)] += 1
        return c

    ca = _bin(lengths_a)
    cb = _bin(lengths_b)
    labels = [f" {b//1024}K" for b in bins] + [f">{bins[-1]//1024}K"]
    na, nb = len(lengths_a), len(lengths_b)
    max_c = max(max(ca.values(), default=1), max(cb.values(), default=1))

    print(f"\n  {'  ':>6}  {label_a[:18]:>18}  {'':2}  {label_b[:18]:<18}")
    print(f"  {' '*6}  {' '*18}  {'':2}  {' '*18}")
    for i, lbl in enumerate(labels):
        va, vb   = ca.get(i, 0), cb.get(i, 0)
        pct_a    = va / na * 100 if na else 0
        pct_b    = vb / nb * 100 if nb else 0
        bar_a    = " " * round(va / max_c * width)
        bar_b    = " " * round(vb / max_c * width)
        print(f"  {lbl:>6}  {bar_a:<{width}} {pct_a:5.1f}%    {bar_b:<{width}} {pct_b:5.1f}%")


#    max_length                                                           

def _recommend(lengths: list, label: str, thresholds: list, W: int) -> None:
    n = len(lengths)
    print(f"\n  [{label}] max_length      ")
    print(f"  {'max_length':>10}  {'    ':>10}  {'   ':>8}  {'   ':>9}")
    print(f"  {' '*10}  {' '*10}  {' '*8}  {' '*9}")
    for t in thresholds:
        over  = sum(1 for v in lengths if v > t)
        print(f"  {t:>10,}  {over:>10,}  {over/n*100:>7.2f}%  {(n-over)/n*100:>8.2f}%")
    #   
    for t in thresholds:
        if sum(1 for v in lengths if v <= t) / n >= 0.95:
            print(f"    P95    max_length = {t:,}")
            break
    for t in thresholds:
        if sum(1 for v in lengths if v <= t) / n >= 0.99:
            print(f"    P99    max_length = {t:,}")
            break


#                                                    

def _print_stats(
    records: list[dict],
    title: str,
    tokenizer_label: str,
    stride: int,
    total_lines: int,
    W: int = 70,
    thresholds: list | None = None,
    hist_bins: list | None = None,
) -> None:
    thresholds = thresholds or [2048, 3072, 4096, 5120, 6144, 7168, 8192]
    hist_bins  = hist_bins  or [2048, 3072, 4096, 5120, 6144, 8192]

    loc_total = sorted(rec["total_local"]  for rec in records)
    glo_total = sorted(rec["total_global"] for rec in records)
    loc_geo   = [rec["geo_local"]  for rec in records]
    glo_geo   = [rec["geo_global"] for rec in records]
    img_lens  = [rec["image"]      for rec in records]
    sam_cnts  = [rec["sam_feat"]   for rec in records]
    n_compare = sum(1 for rec in records if rec["has_compare"])
    n = len(records)

    print("\n" + " " * W)
    print(f"  {title}")
    print(f"  Tokenizer: {tokenizer_label}")
    print(f"Image estimate: 448 x 448, {IMAGE_TOKEN_COUNT} visual tokens/image")
    if stride > 1:
        print(f"        :   {stride}    1   {n:,}      {total_lines:,} ")
    else:
        print(f"       : {n:,}    ")
    print(f"      : {n_compare:,} / {n:,} {n_compare/n*100:.1f}% ")
    print(" " * W)

    col = 24
    def row(label, fn_a, fn_b=""):
        a = f"{fn_a:,.0f}" if isinstance(fn_a, float) else str(fn_a)
        b = f"{fn_b:,.0f}" if isinstance(fn_b, float) else str(fn_b)
        diff = ""
        if fn_b != "" and isinstance(fn_a, (int, float)) and isinstance(fn_b, (int, float)):
            d = fn_b - fn_a
            diff = f"   ={d:+,.0f} ({d/fn_a*100:+.1f}%)" if fn_a else ""
        print(f"  {label:<16} {a:>{col}}  {b:>{col}}{diff}")

    print(f"  {'   ':<16} {'    ':>{col}}  {'    ':>{col}}")
    print(f"  {' '*16} {' '*col}  {' '*col}")
    for label, pa, pb in [
        ("   ",     n,                         n),
        ("   token", sum(loc_total)/n,           sum(glo_total)/n),
        ("  ",       loc_total[0],               glo_total[0]),
        ("P25",        pct(loc_total, 25),         pct(glo_total, 25)),
        ("P50    ", pct(loc_total, 50),         pct(glo_total, 50)),
        ("P75",        pct(loc_total, 75),         pct(glo_total, 75)),
        ("P90",        pct(loc_total, 90),         pct(glo_total, 90)),
        ("P95",        pct(loc_total, 95),         pct(glo_total, 95)),
        ("P99",        pct(loc_total, 99),         pct(glo_total, 99)),
        ("  ",       loc_total[-1],              glo_total[-1]),
    ]:
        row(label, pa, pb)

    print(f"  {' '*16} {' '*col}  {' '*col}")
    avg_img   = sum(img_lens) / n
    avg_sam   = sum(sam_cnts) / n
    avg_geo_l = sum(loc_geo)  / n if loc_geo else 0
    avg_geo_g = sum(glo_geo)  / n if glo_geo else 0
    avg_ctx_l = sum(loc_total)/n - avg_img - avg_geo_l
    avg_ctx_g = sum(glo_total)/n - avg_img - avg_geo_g
    print(f"  {'           ':<55}")
    for label, va, vb in [
        ("   tokens",   avg_img,   avg_img),
        ("   tokens",   avg_geo_l, avg_geo_g),
        ("    tokens", avg_ctx_l, avg_ctx_g),
        ("sam_feat  ",   avg_sam,   avg_sam),
    ]:
        row(label, va, vb)
    print(" " * W)

    ascii_histogram(loc_total, glo_total, "    ", "    ", hist_bins)
    _recommend(loc_total, "    ", thresholds, W)
    _recommend(glo_total, "    ", thresholds, W)

    loc_g_valid = [x for x in loc_geo if x > 0]
    glo_g_valid = [x for x in glo_geo if x > 0]
    if loc_g_valid and glo_g_valid:
        avg_l  = sum(loc_g_valid) / len(loc_g_valid)
        avg_g  = sum(glo_g_valid) / len(glo_g_valid)
        ratio  = avg_l / avg_g * 100
        saving = (avg_g - avg_l) / avg_g * 100
        print(f"\n          token      {len(loc_g_valid):,}       ")
        print(f"         token {avg_l:,.1f}")
        print(f"         token {avg_g:,.1f}")
        print(f"     /          {ratio:.2f}%    {saving:.2f}% ")


#                                                                        

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",       default=str(DEFAULT_JSONL))
    parser.add_argument("--tokenizer",   default="",
                        help="Qwen tokenizer         ")
    parser.add_argument("--sample_rate", type=float, default=1.0,
                        help="    0<x 1.0     ")
    parser.add_argument("--max_filter",  type=int, default=0,
                        help="        token                "
                             "     --sample_rate 1.0         outlier    ")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[ERROR]      : {jsonl_path}")
        sys.exit(1)

    if not _HAS_NUMPY:
        print("[WARN] numpy                        ")

    if args.max_filter > 0 and args.sample_rate < 1.0:
        print(f"[WARN] --max_filter       sample_rate={args.sample_rate} "
              f"outlier                       --sample_rate 1.0 ")

    count_fn    = build_tokenizer(args.tokenizer)
    tok_label   = "Qwen3-VL (fast)" if args.tokenizer else "tiktoken cl100k_base"
    sample_rate = max(1e-6, min(1.0, args.sample_rate))
    stride      = max(1, round(1.0 / sample_rate))

    total_lines = sum(1 for ln in open(jsonl_path, encoding="utf-8") if ln.strip())
    est_samples = max(1, total_lines // stride)
    print(f"[INFO]     {total_lines:,}     {sample_rate:.4f}   {stride}    1  "
          f"     ~{est_samples:,}  ", flush=True)

    def _iter():
        with open(jsonl_path, encoding="utf-8") as f:
            for line_no, ln in enumerate(f):
                if line_no % stride == 0 and ln.strip():
                    yield line_no, ln.strip()

    raw_iter = _iter()
    if _HAS_TQDM:
        raw_iter = _tqdm(raw_iter, total=est_samples, unit=" ", dynamic_ncols=True)

    #           id        outlier    
    records: list[dict] = []
    parse_err = 0

    for line_no, line in raw_iter:
        try:
            sample = json.loads(line)
            r = estimate_length_dual(sample, count_fn)
            records.append({
                "id":           sample.get("id", f"line_{line_no}"),
                "line":         line_no,
                "total_local":  r["total_local"],
                "total_global": r["total_global"],
                "geo_local":    r["geo_local"],
                "geo_global":   r["geo_global"],
                "image":        r["image"],
                "sam_feat":     r["sam_feat"],
                "has_compare":  r["has_compare"],
            })
        except (json.JSONDecodeError, Exception):
            parse_err += 1

    if not records:
        print("[ERROR]         ")
        sys.exit(1)

    if parse_err:
        print(f"[WARN]      {parse_err:,}  ")

    thresholds = [2048, 3072, 4096, 5120, 6144, 7168, 8192]
    hist_bins  = [2048, 3072, 4096, 5120, 6144, 8192]
    W = 70

    #                                                       
    _print_stats(records,
                 title="CoT V3      Token            [  ]",
                 tokenizer_label=tok_label,
                 stride=stride, total_lines=total_lines,
                 W=W, thresholds=thresholds, hist_bins=hist_bins)

    #    max_filter outlier    +                        
    if args.max_filter > 0:
        threshold = args.max_filter
        outliers  = [rec for rec in records if rec["total_local"] > threshold]
        normal    = [rec for rec in records if rec["total_local"] <= threshold]

        print(f"\n{' '*W}")
        print(f"  Outlier         token > {threshold:,} ")
        print(f"{' '*W}")
        print(f"  Outlier    : {len(outliers):,} / {len(records):,} "
              f" {len(outliers)/len(records)*100:.2f}% ")
        print(f"          : {len(normal):,} "
              f" {len(normal)/len(records)*100:.2f}% ")

        if outliers:
            ol_lens = sorted(rec["total_local"] for rec in outliers)
            print(f"  Outlier    : {ol_lens[0]:,}")
            print(f"  Outlier    : {pct(ol_lens, 50):,.0f}")
            print(f"  Outlier    : {ol_lens[-1]:,}")
            print(f"{' '*W}")

            #   token         20  
            top20 = sorted(outliers, key=lambda x: x["total_local"], reverse=True)[:20]
            print(f"  {'  ':<5} {'   ID':<20} {'  ':>8} {'   token':>12} {'   token':>12}")
            print(f"  {' '*5} {' '*20} {' '*8} {' '*12} {' '*12}")
            for rank, rec in enumerate(top20, 1):
                print(f"  {rank:<5} {str(rec['id']):<20} {rec['line']:>8,} "
                      f"{rec['total_local']:>12,} {rec['total_global']:>12,}")
            if len(outliers) > 20:
                print(f"  ...   {len(outliers):,}              ")

            #    outlier   
            out_dir  = jsonl_path.parent
            out_json = out_dir / f"outliers_above_{threshold}.json"
            out_txt  = out_dir / f"outliers_above_{threshold}_ids.txt"

            save_data = [
                {
                    "id":          rec["id"],
                    "line":        rec["line"],
                    "total_local": rec["total_local"],
                    "total_global":rec["total_global"],
                }
                for rec in sorted(outliers, key=lambda x: x["total_local"], reverse=True)
            ]
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            with open(out_txt, "w", encoding="utf-8") as f:
                for rec in save_data:
                    f.write(f"{rec['id']}\n")

            print(f"\n  [  ]    outlier      {out_json}")
            print(f"  [  ]   ID            {out_txt}")
        print(f"{' '*W}")

        #             
        if normal:
            _print_stats(normal,
                         title=f"         token   {threshold:,}   {len(normal):,}   ",
                         tokenizer_label=tok_label,
                         stride=stride, total_lines=total_lines,
                         W=W, thresholds=thresholds, hist_bins=hist_bins)

    print()


if __name__ == "__main__":
    main()
