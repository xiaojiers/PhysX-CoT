#!/usr/bin/env python3
"""
SAM3         
=====================
   renders_cond/           cot_tmp_v3/    bbox_2d 
   encoder_hidden_states   RoI      feat_B  
    sam_feature/{object_id}/{image_id}.npz 

           npz  
  {part_id: np.ndarray [256] float32}
  e.g. {"l_0": ..., "l_1": ..., "l_2": ...}

     
           forward_image backbone  
            backbone_out     encoder        

     
  python 6catch_sam_feature.py
  python 6catch_sam_feature.py --workers 1 --skip_existing
"""

import os
import re
import sys
import random
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.ops as tvops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm

#    SAM3                                                        
_SAM3_ROOT = Path(__file__).parent.parent.parent / "sam3-main"
if str(_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_ROOT))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


#                                                                
#                 data/ 
#                                                                

_DATASET_DIR   = Path(__file__).parent
_RENDERS_DIR   = _DATASET_DIR / "renders_cond"
_COT_DIR       = _DATASET_DIR / "cot_tmp_v3"
_OUTPUT_DIR    = _DATASET_DIR / "sam_feature"
_CHECKPOINT    = _SAM3_ROOT / "checkpoints" / "sam3" / "sam3.pt"


#                                                                
#  COT   
#                                                                

def parse_bbox_2d(cot_path: Path) -> Dict[str, List[float]]:
    """
       COT          bbox_2d = [x_min, x_max, y_min, y_max]     0~1  
        [cx, cy, w, h]        
    """
    bboxes: Dict[str, List[float]] = {}
    content = cot_path.read_text()
    for m in re.finditer(r"Part `(l_\d+)`: `bbox_2d` = \[([^\]]+)\]", content):
        part_id = m.group(1)
        x_min, x_max, y_min, y_max = [float(v.strip()) for v in m.group(2).split(",")]
        bboxes[part_id] = [
            (x_min + x_max) / 2,  # cx
            (y_min + y_max) / 2,  # cy
            x_max - x_min,        # w
            y_max - y_min,        # h
        ]
    return bboxes


#                                                                
#  SAM3             
#                                                                

class SAM3BatchExtractor:
    """
         feat_B encoder_hidden_states RoI          

         
      1. set_image()      forward_image       backbone_out 
      2. per-part loop    add_geometric_prompt   encoder N      

    Hook     TransformerEncoderFusion        SAM3    
    """

    def __init__(
        self,
        model,
        processor: Sam3Processor,
        roi_output_size: int = 3,
    ):
        self.model = model
        self.processor = processor
        self.roi_output_size = roi_output_size
        self._enc_memory: Optional[torch.Tensor] = None
        self._enc_spatial_shapes: Optional[torch.Tensor] = None
        self._hook_handle = None

    #    Hook                                                    

    def _register_hook(self) -> None:
        def _fn(module, input, output):
            self._enc_memory = output["memory"].detach()
            self._enc_spatial_shapes = output["spatial_shapes"].detach()
        self._hook_handle = self.model.transformer.encoder.register_forward_hook(_fn)

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    #    RoI                                                   

    def _roi_pool_enc(
        self,
        memory: torch.Tensor,           # [ HW, N, d]
        spatial_shapes: torch.Tensor,   # [num_levels, 2]
        box_cxcywh: List[float],
    ) -> np.ndarray:
        """
          encoder memory    box   RoI Align    [d] float32 numpy    
        """
        enc_h, enc_w = spatial_shapes[0].tolist()
        N, d = memory.shape[1], memory.shape[2]
        feat_map = (
            memory.permute(1, 2, 0)        # [N, d,  HW]
            .reshape(N, d, enc_h, enc_w)
            .float()
        )  # [1, d, enc_h, enc_w]

        cx, cy, w, h = box_cxcywh
        x1 = max((cx - w / 2) * enc_w, 0.0)
        y1 = max((cy - h / 2) * enc_h, 0.0)
        x2 = min((cx + w / 2) * enc_w, enc_w - 1e-3)
        y2 = min((cy + h / 2) * enc_h, enc_h - 1e-3)

        box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=feat_map.device)
        roi = tvops.roi_align(feat_map, [box], output_size=self.roi_output_size, aligned=True)
        return roi.mean(dim=[-2, -1]).squeeze(0).cpu().float().numpy()  # [d]

    #                                                

    def extract_image(
        self,
        image: Image.Image,
        bboxes: Dict[str, List[float]],
    ) -> Dict[str, np.ndarray]:
        """
                     feat_B 

        Parameters
        ----------
        image  : PIL.Image (RGB)
        bboxes : {part_id: [cx, cy, w, h]}     

        Returns
        -------
        {part_id: np.ndarray [256] float32}
        """
        device_type = str(self.model.device.type)

        # Step 1 backbone_fpn         
        with torch.autocast(device_type, dtype=torch.bfloat16):
            base_state = self.processor.set_image(image)
        backbone_out = base_state["backbone_out"]

        feats: Dict[str, np.ndarray] = {}

        # Step 2     encoder       backbone_out 
        for part_id, box_cxcywh in bboxes.items():
            #    backbone_out      _forward_grounding          
            part_state = {
                "backbone_out":    backbone_out,
                "original_height": base_state["original_height"],
                "original_width":  base_state["original_width"],
            }

            self._register_hook()
            try:
                with torch.autocast(device_type, dtype=torch.bfloat16):
                    self.processor.add_geometric_prompt(
                        box=box_cxcywh, label=True, state=part_state
                    )
            finally:
                self._remove_hook()

            if self._enc_memory is not None:
                feats[part_id] = self._roi_pool_enc(
                    self._enc_memory, self._enc_spatial_shapes, box_cxcywh
                )
                self._enc_memory = None
                self._enc_spatial_shapes = None

        return feats


#                                                                
#         
#                                                                

#      
_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
]


def _feat_to_heatmap(feat: np.ndarray) -> np.ndarray:
    """  [256]      reshape   [16, 16]      [0,1] """
    hmap = feat.reshape(16, 16).astype(np.float32)
    lo, hi = hmap.min(), hmap.max()
    return (hmap - lo) / (hi - lo + 1e-6)


def visualize_random_samples(
    output_dir: Path,
    renders_dir: Path,
    cot_dir: Path,
    n_samples: int = 6,
    save_path: Optional[Path] = None,
) -> None:
    """
         n_samples       npz            
         :      +    bbox   
         :       feat_B    16 16 

    Parameters
    ----------
    output_dir : sam_feature/   
    renders_dir: renders_cond/   
    cot_dir    : cot_tmp_v3/   
    n_samples  :       
    save_path  :      None     output_dir/sample_visualization.png 
    """
    #             npz                                  
    all_npz: List[Tuple[str, str, Path]] = []  # (obj_id, img_id, npz_path)
    for obj_dir in sorted(output_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        for npz_path in sorted(obj_dir.glob("*.npz")):
            all_npz.append((obj_dir.name, npz_path.stem, npz_path))

    if not all_npz:
        print("            npz      ")
        return

    #                                                       
    selected = random.sample(all_npz, min(n_samples, len(all_npz)))

    #                                             
    max_parts = 0
    sample_data = []
    for obj_id, img_id, npz_path in selected:
        data = dict(np.load(str(npz_path)))
        part_ids = sorted(data.keys())
        max_parts = max(max_parts, len(part_ids))

        img_path = renders_dir / f"{obj_id}_" / f"{img_id}.png"
        cot_path = cot_dir / f"{obj_id}_{img_id}.txt"
        bboxes   = parse_bbox_2d(cot_path) if cot_path.exists() else {}

        sample_data.append({
            "obj_id":   obj_id,
            "img_id":   img_id,
            "img_path": img_path,
            "part_ids": part_ids,
            "feats":    data,
            "bboxes":   bboxes,
        })

    #                                                         
    #             = 1    + max_parts    
    n_cols  = 1 + max_parts
    n_rows  = len(sample_data)
    fig_w   = 2.8 * n_cols
    fig_h   = 3.2 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    #    axes        
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, sd in enumerate(sample_data):
        img_path = sd["img_path"]
        obj_id   = sd["obj_id"]
        img_id   = sd["img_id"]
        part_ids = sd["part_ids"]
        feats    = sd["feats"]
        bboxes   = sd["bboxes"]

        #      0    + bbox                                 
        ax = axes[row, 0]
        if img_path.exists():
            img = np.array(Image.open(img_path).convert("RGB"))
            ax.imshow(img)
            h, w = img.shape[:2]
            legend_handles = []
            for pi, pid in enumerate(part_ids):
                color = _COLORS[pi % len(_COLORS)]
                if pid in bboxes:
                    cx, cy, bw, bh = bboxes[pid]
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    rect = mpatches.Rectangle(
                        (x1, y1), bw * w, bh * h,
                        linewidth=1.8, edgecolor=color, facecolor="none",
                    )
                    ax.add_patch(rect)
                legend_handles.append(
                    mpatches.Patch(color=color, label=pid)
                )
            ax.legend(
                handles=legend_handles, fontsize=6,
                loc="lower right", framealpha=0.7,
            )
        else:
            ax.text(0.5, 0.5, "image\nnot found",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)

        ax.set_title(f"{obj_id}/{img_id}", fontsize=8, fontweight="bold")
        ax.axis("off")

        #      1..N       feat_B                     
        for pi, pid in enumerate(part_ids):
            ax = axes[row, 1 + pi]
            hmap = _feat_to_heatmap(feats[pid])
            im = ax.imshow(hmap, cmap="plasma", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            color = _COLORS[pi % len(_COLORS)]
            ax.set_title(pid, fontsize=8, color=color, fontweight="bold")
            ax.set_xlabel("feat_B [16 16]", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

        #                                               
        for pi in range(len(part_ids), max_parts):
            axes[row, 1 + pi].axis("off")

    fig.suptitle(
        f"SAM3 feat_B         {len(sample_data)}     ",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save_path is None:
        save_path = output_dir / "sample_visualization.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"           {save_path}")


#                                                                
#        
#                                                                

def build_task_list(
    renders_dir: Path,
    cot_dir: Path,
    output_dir: Path,
    skip_existing: bool,
) -> List[tuple]:
    """
              (object_id, image_id, image_path, cot_path, out_path)
    """
    tasks = []
    for obj_dir in sorted(renders_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        #       {object_id}_
        obj_id = obj_dir.name.rstrip("_")
        obj_out = output_dir / obj_id

        for img_path in sorted(obj_dir.glob("*.png")):
            img_id = img_path.stem           # e.g. "000"
            cot_path = cot_dir / f"{obj_id}_{img_id}.txt"
            out_path = obj_out / f"{img_id}.npz"

            if not cot_path.exists():
                continue
            if skip_existing and out_path.exists():
                continue

            tasks.append((obj_id, img_id, img_path, cot_path, out_path))

    return tasks


def run_batch(args) -> None:
    #                                                        
    log_path = _OUTPUT_DIR / "extract.log"
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)

    #                                                      
    device = args.device
    ckpt   = Path(args.checkpoint)
    log.info(f"   SAM3     checkpoint={ckpt}  device={device}")
    model = build_sam3_image_model(
        checkpoint_path=str(ckpt),
        load_from_HF=False,
        device=device,
    )
    processor = Sam3Processor(model, device=device)
    extractor = SAM3BatchExtractor(model, processor, roi_output_size=args.roi_size)
    log.info("      ")

    #                                                      
    log.info("      ...")
    tasks = build_task_list(
        renders_dir=Path(args.renders_dir),
        cot_dir=Path(args.cot_dir),
        output_dir=Path(args.output_dir),
        skip_existing=args.skip_existing,
    )
    log.info(f"  {len(tasks)}   (object, image)     ")

    if not tasks:
        log.info("         ")
        return

    #                                                      
    n_ok = n_skip = n_err = 0
    err_list = []

    for obj_id, img_id, img_path, cot_path, out_path in tqdm(tasks, desc="    "):
        try:
            #         
            out_path.parent.mkdir(parents=True, exist_ok=True)

            #    bbox
            bboxes = parse_bbox_2d(cot_path)
            if not bboxes:
                n_skip += 1
                continue

            #     
            image = Image.open(img_path).convert("RGB")

            #     
            feats = extractor.extract_image(image, bboxes)
            if not feats:
                n_skip += 1
                continue

            #    npz   = part_id   = float32 [256]
            np.savez_compressed(str(out_path), **feats)
            n_ok += 1

        except Exception:
            n_err += 1
            err_msg = f"{obj_id}/{img_id}: {traceback.format_exc()}"
            err_list.append(err_msg)
            log.error(err_msg)

    #                                                        
    log.info(
        f"     ={n_ok}    ={n_skip}    ={n_err}  "
        f"    ={args.output_dir}"
    )
    if err_list:
        err_log = _OUTPUT_DIR / "errors.log"
        err_log.write_text("\n\n".join(err_list))
        log.info(f"      {err_log}")

    #                                                    
    if n_ok > 0:
        log.info(f"          n={args.vis_samples} ...")
        try:
            visualize_random_samples(
                output_dir=Path(args.output_dir),
                renders_dir=Path(args.renders_dir),
                cot_dir=Path(args.cot_dir),
                n_samples=args.vis_samples,
            )
        except Exception:
            log.warning(f"                 \n{traceback.format_exc()}")


#                                                                
#  CLI
#                                                                

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM3    feat_B      encoder RoI    "
    )
    parser.add_argument(
        "--renders_dir", default=str(_RENDERS_DIR),
        help="renders_cond     ",
    )
    parser.add_argument(
        "--cot_dir", default=str(_COT_DIR),
        help="cot_tmp_v3     ",
    )
    parser.add_argument(
        "--output_dir", default=str(_OUTPUT_DIR),
        help="     sam_feature/ ",
    )
    parser.add_argument(
        "--checkpoint", default=str(_CHECKPOINT),
        help="SAM3       ",
    )
    parser.add_argument(
        "--roi_size", type=int, default=3,
        help="RoI Align      roi_size roi_size     3",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="     npz                ",
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="              --skip_existing ",
    )
    parser.add_argument(
        "--vis_samples", type=int, default=6,
        help="                   6 ",
    )
    args = parser.parse_args()

    if args.no_skip:
        args.skip_existing = False

    run_batch(args)


if __name__ == "__main__":
    main()
