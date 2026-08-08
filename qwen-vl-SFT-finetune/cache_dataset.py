"""
         V3    SAM      +     Token 

                           torch.load 
        CPU    tokenize / image_preprocess / npz     

V3    V1     
  -       token <sam_feat> <overall> <geometry_l_k>   
  - <sam_feat_l_k>   <sam_feat>     
  - <sam_feat>    labels = -100
  -       SAM      sam_feats [N_parts, 256]      .pt   

   V3  
  python cache_dataset.py \\
    --model_path Qwen/Qwen3-VL-8B-Instruct \\
    --annotation_path ./data/train.jsonl \\
    --data_path ./data/renders \\
    --sam_feature_dir ./data/sam_features \\
    --cache_dir ./data/cache \\
    --max_length 6144 \\
    --max_pixels 262144 \\
    --min_pixels 65536 \\
    --num_workers 16

   V1       sam_feature_dir     SAM     
  python cache_dataset.py \\
    --model_path ... --annotation_path ...cot.jsonl \\
    --data_path ... --cache_dir ... \\
    --max_length 3072 --max_pixels 262144
"""

import os
import re
import json
import argparse
import logging
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#   train_lora.py       
_NEW_SPECIAL_TOKENS: List[str] = [
    "<sam_feat>",
    "<overall>",  "</overall>",
    *[f"<geometry_l_{k}>"  for k in range(10)],
    *[f"</geometry_l_{k}>" for k in range(10)],
]
_SAM_FEAT_PAT = re.compile(r"<sam_feat_l_\d+>")


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_labels(
    input_ids: torch.Tensor,
    im_start_id: int,
    assistant_id: int,
    sam_token_id: int,
) -> torch.Tensor:
    """   labels human turn   -100 <sam_feat>      -100 """
    labels = input_ids.clone()
    in_assistant = False
    i = 0
    while i < len(input_ids):
        if (input_ids[i] == im_start_id
                and i + 1 < len(input_ids)
                and input_ids[i + 1] == assistant_id):
            in_assistant = True
            labels[i] = labels[i + 1] = -100
            i += 2
            if i < len(input_ids):
                labels[i] = -100
                i += 1
        elif input_ids[i] == im_start_id and in_assistant:
            in_assistant = False
            labels[i] = -100
            i += 1
        elif not in_assistant:
            labels[i] = -100
            i += 1
        else:
            i += 1

    # <sam_feat>      loss                 
    if sam_token_id >= 0:
        sam_pos = (input_ids == sam_token_id).nonzero(as_tuple=True)[0]
        labels[sam_pos] = -100

    return labels


def process_one(args_tuple):
    """
                     (idx, save_path)   (idx, None) 

              processor           
    """
    (
        idx,
        sample,
        data_root,
        sam_feature_dir,
        model_path,
        max_length,
        max_pixels,
        min_pixels,
        cache_dir,
    ) = args_tuple

    save_path = os.path.join(cache_dir, f"{idx:07d}.pt")
    if os.path.exists(save_path):
        return idx, save_path

    try:
        #    Processor +    Token                           
        #        transformers       use_fast / slow processor    
        #    16               
        warnings.filterwarnings("ignore")
        import transformers
        transformers.logging.set_verbosity_error()

        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        new_tokens = [t for t in _NEW_SPECIAL_TOKENS if t not in processor.tokenizer.get_vocab()]
        if new_tokens:
            processor.tokenizer.add_tokens(new_tokens, special_tokens=True)

        sam_token_id = processor.tokenizer.convert_tokens_to_ids("<sam_feat>")
        im_start_id  = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")

        #       messages                                     
        convs     = sample["conversations"]
        messages  = []
        image_obj = None

        for turn in convs:
            role    = "user" if turn["from"] == "human" else "assistant"
            # <sam_feat_l_0>   <sam_feat>        {part_id}    
            content = _SAM_FEAT_PAT.sub("<sam_feat>", turn["value"])

            if "<image>" in content and image_obj is None:
                img_rel  = sample.get("image", "")
                img_path = os.path.join(data_root, img_rel)
                try:
                    image_obj = Image.open(img_path).convert("RGB")
                except Exception:
                    image_obj = Image.new("RGB", (224, 224), color=0)

                text_part = content.replace("<image>", "").strip()
                messages.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": image_obj},
                        {"type": "text",  "text": text_part},
                    ],
                })
            else:
                messages.append({"role": role, "content": content})

        #       +                                       
        text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        images = [image_obj] if image_obj is not None else []
        inputs = processor(
            text=text,
            images=images or None,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        NO_SQUEEZE = {"image_grid_thw"}
        inputs = {k: v if k in NO_SQUEEZE else v.squeeze(0) for k, v in inputs.items()}

        #    Labels                                            
        inputs["labels"] = _build_labels(
            inputs["input_ids"], im_start_id, assistant_id, sam_token_id
        )

        #    SAM    V3                                  
        sam_feat_tensor: Optional[torch.Tensor] = None
        if sam_feature_dir and sample.get("sam_feature"):
            npz_path = os.path.join(sam_feature_dir, sample["sam_feature"])
            try:
                data     = np.load(npz_path)
                part_ids = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
                feats    = [torch.tensor(data[k], dtype=torch.float32) for k in part_ids]
                sam_feat_tensor = torch.stack(feats, dim=0)  # [N_parts, 256]
            except Exception:
                pass  # SAM        None        

        inputs["sam_feats"] = sam_feat_tensor  # None   [N_parts, 256]

        torch.save(inputs, save_path)
        return idx, save_path

    except Exception as e:
        logger.warning(f"   {idx}      {e}")
        return idx, None


def main():
    parser = argparse.ArgumentParser(description="PhysX CoT        V3 ")
    parser.add_argument("--model_path",       required=True)
    parser.add_argument("--annotation_path",  required=True)
    parser.add_argument("--data_path",        required=True)
    parser.add_argument("--sam_feature_dir",  default="",
                        help="data/           SAM      V1      ")
    parser.add_argument("--cache_dir",        required=True)
    parser.add_argument("--max_length",       type=int, default=6144)
    parser.add_argument("--max_pixels",       type=int, default=262144)
    parser.add_argument("--min_pixels",       type=int, default=65536)
    parser.add_argument("--num_workers",      type=int, default=16,
                        help="         = CPU    / 2      OOM ")
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    samples  = load_jsonl(args.annotation_path)
    existing = len(list(Path(args.cache_dir).glob("*.pt")))
    logger.info(f"    : {len(samples)}    : {existing}")

    task_args = [
        (
            i, s,
            args.data_path,
            args.sam_feature_dir,
            args.model_path,
            args.max_length,
            args.max_pixels,
            args.min_pixels,
            args.cache_dir,
        )
        for i, s in enumerate(samples)
    ]

    failed = 0
    done   = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, t): t[0] for t in task_args}
        with tqdm(
            total=len(task_args),
            initial=existing,
            desc="    ",
            unit="  ",
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(futures):
                idx, path = future.result()
                if path is None:
                    failed += 1
                    pbar.set_postfix(failed=failed, refresh=False)
                else:
                    done += 1
                pbar.update(1)

    #                      
    index = [
        i for i in range(len(samples))
        if os.path.exists(os.path.join(args.cache_dir, f"{i:07d}.pt"))
    ]
    index_path = os.path.join(args.cache_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f)

    logger.info(f"        {len(index)}    {len(samples) - len(index)}")
    logger.info(f"     {args.cache_dir}     {index_path}")


if __name__ == "__main__":
    main()
