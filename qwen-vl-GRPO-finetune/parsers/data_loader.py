"""
parsers/data_loader.py

           

ExampleAdapter       V3 GRPO JSONL      trl.GRPOTrainer       
    messages   :           image    path     
    gt         : Dict   think_steps / overall_dict / part_count / voxel_dir 
    meta       : Dict split / category / part_count / complexity / quality_ok 
    sam_feature: Optional[str]   sam_feature/<obj>/<view>.npz    dataset_root 

     
  1.    adapter      sam_feature .npz         Arrow    ndarray 
              collator   SFT       
  2.    adapter      voxel reward      VoxelGTLoader     
  3.          fallback    row   messages    task_cfg       
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

from configs import DataConfig, TaskConfig

LOGGER = logging.getLogger(__name__)


#    JSONL / HF                                                              

def load_jsonl_dataset(path: str) -> Dataset:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx} in {path}: {exc}") from exc
    return Dataset.from_list(records)


def load_datasets(data_cfg: DataConfig) -> DatasetDict:
    if data_cfg.dataset_format == "jsonl":
        train_ds = load_jsonl_dataset(data_cfg.train_file)
        eval_ds  = load_jsonl_dataset(data_cfg.eval_file) if data_cfg.eval_file else None
    elif data_cfg.dataset_format == "hf":
        train_ds = load_dataset(data_cfg.train_file, split="train")
        eval_ds  = load_dataset(data_cfg.eval_file,  split="train") if data_cfg.eval_file else None
    else:
        raise ValueError(f"Unsupported dataset_format: {data_cfg.dataset_format}")

    if data_cfg.max_train_samples is not None:
        train_ds = train_ds.select(range(min(len(train_ds), data_cfg.max_train_samples)))
    if eval_ds is not None and data_cfg.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(len(eval_ds), data_cfg.max_eval_samples)))

    dsd = DatasetDict({"train": train_ds})
    if eval_ds is not None:
        dsd["eval"] = eval_ds
    return dsd


#                                                                          

class ExampleAdapter:
    """
        V3 GRPO JSONL      GRPOTrainer           

            build_grpo_dataset.py     
      {
        "id":          "...",
        "image":       "<obj>_/<view>.png",
        "sam_feature": "sam_feature/<obj>/<view>.npz" | null,
        "messages":    [{"role":"user","content":[{"type":"image"},{"type":"text",...}]}],
        "gt":          { part_count, think_steps, overall_dict, voxel_dir },
        "meta":        { split, category, part_count, complexity, quality_ok }
      }

          
      {
        "messages":         messages   image         
        "gt":            gt       
        "meta":          meta       
        "sam_feature": Optional[str]            collator    npz
      }
    """

    def __init__(
        self,
        task_cfg:     TaskConfig,
        image_root:   Optional[str] = None,
        dataset_root: Optional[str] = None,
    ):
        self.task_cfg     = task_cfg
        self.image_root   = Path(image_root)   if image_root   else None
        self.dataset_root = Path(dataset_root) if dataset_root else None

    #                                                                       

    def resolve_image_path(self, row: Dict[str, Any]) -> str:
        rel = row.get("image") or row.get("image_path") or row.get("img_path")
        if not rel:
            raise KeyError("Could not find image path field in dataset row.")
        if self.image_root is not None:
            return str((self.image_root / rel).resolve())
        p = Path(rel)
        return str(p.resolve() if not p.is_absolute() else p)

    def resolve_sam_feature_path(self, row: Dict[str, Any]) -> Optional[str]:
        rel = row.get("sam_feature")
        if not rel:
            return None
        if self.dataset_root is not None:
            return str((self.dataset_root / rel).resolve())
        p = Path(rel)
        return str(p.resolve() if not p.is_absolute() else p)

    #    messages                                                             

    def build_messages(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """          messages    image           """
        image_path = self.resolve_image_path(row)

        if "messages" in row and row["messages"]:
            messages = copy.deepcopy(row["messages"])
            for turn in messages:
                content = turn.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "image":
                            block["image"] = image_path
            return messages

        # fallback      
        text = row.get("prompt") or self.task_cfg.overall_prompt
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": text},
            ],
        }]

    #    GT / meta                                                            

    @staticmethod
    def build_ground_truth(row: Dict[str, Any]) -> Dict[str, Any]:
        if "gt" in row:
            return row["gt"]
        if "label" in row:
            return row["label"]
        return {}

    @staticmethod
    def build_meta(row: Dict[str, Any]) -> Dict[str, Any]:
        return row.get("meta") or {}

    #                                                                       

    def adapt_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "messages":    self.build_messages(row),
            "gt":          self.build_ground_truth(row),
            "meta":        self.build_meta(row),
            "sam_feature": self.resolve_sam_feature_path(row),
        }


def adapt_dataset(ds: Dataset, adapter: ExampleAdapter) -> Dataset:
    rows = [adapter.adapt_row(row) for row in ds]
    return Dataset.from_list(rows)
