"""
    tokenizer          <think>/<overall>/<sam_feat>/<geometry_l_k> decode      

   
  cd qwen-vl-SFT-finetune
  python3 debug_tokenizer.py

                tokenizer              
"""

from __future__ import annotations

import os
import sys
from transformers import AutoProcessor, AutoTokenizer
import transformers
import tokenizers as _tokenizers_lib

ADAPTER_DIR = os.environ.get(
    "ADAPTER_DIR",
    "./outputs/physx-cot-sft",
)

_CRITICAL = [
    "<sam_feat>",
    "<think>", "</think>",
    "<overall>", "</overall>",
    *[f"<geometry_l_{k}>"  for k in range(3)],
    *[f"</geometry_l_{k}>" for k in range(3)],
]

print(" " * 60)
print(f"transformers    : {transformers.__version__}")
print(f"tokenizers      : {_tokenizers_lib.__version__}")
print(f"python          : {sys.version.split()[0]}")
print(f"ADAPTER_DIR     : {ADAPTER_DIR}")
print(" " * 60)

#        Processor   Tokenizer               
processor = AutoProcessor.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
tok_proc  = processor.tokenizer
tok_auto  = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)

for name, tok in [("processor.tokenizer", tok_proc), ("AutoTokenizer", tok_auto)]:
    print(f"\n    {name}  ({type(tok).__name__})    ")
    print(f"  len(tok)        = {len(tok)}")
    print(f"  unk_token_id    = {tok.unk_token_id}")
    print(f"  vocab_size      = {tok.vocab_size}")
    print(f"  is_fast         = {tok.is_fast}")

    # 1)    id
    print("\n  convert_tokens_to_ids:")
    for t in _CRITICAL:
        tid = tok.convert_tokens_to_ids(t)
        flag = " <UNK!>" if tid == tok.unk_token_id else ""
        print(f"    {t:>22s} -> {tid}{flag}")

    # 2)    added_tokens_decoder
    atd = tok.added_tokens_decoder
    print(f"\n  added_tokens_decoder size = {len(atd)}")
    for key_id in [151665, 151666, 151667, 151668, 151669, 151670, 151690]:
        if key_id in atd:
            info = atd[key_id]
            content  = getattr(info, "content",   None)
            special  = getattr(info, "special",   None)
            normalized = getattr(info, "normalized", None)
            print(f"    {key_id}: content={content!r}  special={special}  normalized={normalized}")
        else:
            print(f"    {key_id}: MISSING")

    # 3)   id decode           
    print("\n  decode tests:")
    for tid_int in [151666, 151667, 151668, 151669, 151665, 151670]:
        try:
            raw_false = tok.decode([tid_int], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            raw_true  = tok.decode([tid_int], skip_special_tokens=True,  clean_up_tokenization_spaces=False)
            bytes_false = raw_false.encode("utf-8")
            print(f"    id={tid_int}: skip=False -> {raw_false!r} bytes={bytes_false.hex()}  |  skip=True -> {raw_true!r}")
        except Exception as exc:
            print(f"    id={tid_int}: DECODE EXCEPTION: {exc}")

    # 4)      decode
    mixed = [151666, 198, 8979, 220, 16, 151667]  # <think> \n Step  1 </think>  (      id        )
    out_f = tok.decode(mixed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    out_t = tok.decode(mixed, skip_special_tokens=True,  clean_up_tokenization_spaces=False)
    print(f"\n  mixed decode skip=False: {out_f!r}")
    print(f"  mixed decode skip=True : {out_t!r}")
    print(f"  mixed decode skip=False bytes[:30]: {out_f.encode('utf-8')[:30].hex()}")

    # 5) encode   decode             roundtrip
    roundtrip_src = "<think>\nStep 1: test.\n</think>"
    ids = tok.encode(roundtrip_src, add_special_tokens=False)
    print(f"\n  roundtrip source: {roundtrip_src!r}")
    print(f"  roundtrip ids   : {ids}")
    print(f"  roundtrip decode skip=False: {tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)!r}")

print("\n  END  ")
