"""
      LoRA adapter     base model              

   
    python tools/merge_lora.py \
        --base_model  Qwen/Qwen3-VL-8B-Instruct \
        --lora_ckpt   ./outputs/physx-cot-sft/checkpoint-xxx \
        --output_dir  ./merged_model

       
    -            safetensors    
    - tokenizer / processor     
"""

import os
import argparse
import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base_model", required=True,
                        help="Base model path (e.g. Qwen3-VL-8B-Instruct)")
    parser.add_argument("--lora_ckpt", required=True,
                        help="LoRA checkpoint directory (contains adapter_model.bin)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for merged model")
    return parser.parse_args()


def merge(args):
    print(f"[1/5] Loading base model from: {args.base_model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map="cpu",
    )

    print(f"[2/5] Loading LoRA adapters from: {args.lora_ckpt}")
    model = PeftModel.from_pretrained(model, args.lora_ckpt, is_trainable=False)

    print("[3/5] Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    #            Merger   
    merger_path = os.path.join(args.lora_ckpt, "merger_weights.pt")
    if os.path.exists(merger_path):
        print(f"[4/5] Loading Merger weights from: {merger_path}")
        merger_state = torch.load(merger_path, map_location="cpu")
        missing, unexpected = [], []
        visual = getattr(model, "visual", None) or getattr(model.model, "visual", None)
        if visual is None or not hasattr(visual, "merger"):
            raise AttributeError("Qwen3-VL visual merger was not found.")
        model_merger_state = dict(visual.merger.named_parameters())
        for name, param in merger_state.items():
            if name in model_merger_state:
                model_merger_state[name].data.copy_(param)
            else:
                unexpected.append(name)
        missing = [k for k in model_merger_state if k not in merger_state]
        if missing:
            print(f"  [WARN] Missing Merger keys: {missing}")
        if unexpected:
            print(f"  [WARN] Unexpected Merger keys: {unexpected}")
        print(f"  Loaded {len(merger_state)} Merger tensors.")
    else:
        print("[4/5] No merger_weights.pt found, skipping Merger injection.")

    print(f"[5/5] Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    processor.save_pretrained(args.output_dir)

    print("Done. Merged model saved.")


if __name__ == "__main__":
    merge(parse_args())
