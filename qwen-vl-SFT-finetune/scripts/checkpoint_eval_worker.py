"""
checkpoint_eval_worker.py

      checkpoint       CheckpointEvalCallback             

     
  Step 1.       python {physx_root}/1_vlm_cot.py
  Step 2.         python {physx_root}/visualize_voxels.py
  Step 3.        python {physx_root}/visualize4SFT/vis_loss.py

     CPU    CUDA_VISIBLE_DEVICES=""      DDP        
    --device cuda:X        GPU 

       
  {output_dir}/
    knife_002/
      knife_002_cot.txt         Turn1 CoT   
      ind_0.npy, ind_1.npy...   Turn2     
      voxel_vis.obj                  OBJ
    loss_curve.png
    lr_schedule.png
    loss_lr_combined.png
    grad_norm.png
    eval.log                               callback    
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run(cmd: list[str], label: str) -> bool:
    """                    """
    logger.info("[%s]   : %s", label, " ".join(cmd))
    ret = subprocess.run(cmd, capture_output=False)
    if ret.returncode != 0:
        logger.error("[%s]       =%d", label, ret.returncode)
        return False
    logger.info("[%s]   ", label)
    return True


def step1_inference(args: argparse.Namespace) -> str:
    """
       1_vlm_cot.py             

       
      1. CPU      CUDA_VISIBLE_DEVICES=""          DDP       
            cuda:X       CUDA_VISIBLE_DEVICES         
      2.           --no_auto_extract_sam 
         SAM3        /      CPU        
         <sam_feat>       embedding                 
                
    """
    physx_root = args.physx_root
    infer_script = os.path.join(physx_root, "1_vlm_cot.py")

    device = args.device
    max_new_tokens = str(args.max_new_tokens)

    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda:"):
        gpu_id = device.split(":")[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = [
        sys.executable, infer_script,
        "--adapter_path",   args.checkpoint,
        "--base_model",     args.base_model,
        "--image",          args.image,
        "--output_dir",     args.output_dir,
        "--device",         device,
        "--max_new_tokens", max_new_tokens,
    ]
    if args.skip_auto_sam:
        cmd.append("--no_auto_extract_sam")

    logger.info(
        "[Step1-Infer]     =%s max_new_tokens=%s skip_auto_sam=%s",
        device, max_new_tokens, args.skip_auto_sam,
    )
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        logger.error("[Step1-Infer]         =%d", ret.returncode)
        return ""

    stem = Path(args.image).stem
    infer_dir = os.path.join(args.output_dir, stem)
    logger.info("[Step1-Infer]        : %s", infer_dir)
    return infer_dir


def step2_voxel_vis(args: argparse.Namespace, infer_dir: str) -> None:
    """
       visualize_voxels.py        OBJ 
            ind_*.npy                   
    """
    if not infer_dir or not os.path.isdir(infer_dir):
        logger.warning("[Step2-Voxel]                ")
        return

    npy_files = list(Path(infer_dir).glob("ind_*.npy"))
    if not npy_files:
        logger.warning("[Step2-Voxel]     ind_*.npy         Turn1        ")
        return

    vis_script = os.path.join(args.physx_root, "visualize_voxels.py")
    out_obj    = os.path.join(args.output_dir, "voxel_vis.obj")

    run(
        [sys.executable, vis_script, "--dir", infer_dir, "--out", out_obj],
        label="Step2-Voxel",
    )


def step3_loss_curve(args: argparse.Namespace) -> None:
    """
       visualize4SFT/vis_loss.py           

         --state_path          fallback   checkpoint    
      trainer_state.json Trainer   checkpoint     state       
    """
    vis_script = os.path.join(args.physx_root, "visualize4SFT", "vis_loss.py")
    if not os.path.isfile(vis_script):
        logger.warning("[Step3-Loss] vis_loss.py    : %s", vis_script)
        return

    state_path = args.state_path
    if state_path and not os.path.isfile(state_path):
        fallback = os.path.join(args.checkpoint, "trainer_state.json")
        if os.path.isfile(fallback):
            logger.info(
                "[Step3-Loss] --state_path     (%s)    checkpoint  : %s",
                state_path, fallback,
            )
            state_path = fallback
        else:
            logger.warning(
                "[Step3-Loss] trainer_state.json     (    %s   %s)",
                state_path, fallback,
            )
            return

    run(
        [
            sys.executable, vis_script,
            "--state_path", state_path,
            "--out_dir",    args.output_dir,
        ],
        label="Step3-Loss",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint      worker")
    parser.add_argument("--checkpoint",      required=True, help="checkpoint      adapter weights ")
    parser.add_argument("--image",           required=True, help="       ")
    parser.add_argument("--base_model",      required=True, help="      ")
    parser.add_argument("--physx_root",      required=True, help="PhysX-CoT repository root.")
    parser.add_argument("--output_dir",      required=True, help="        ")
    parser.add_argument("--device",          default="cpu", help="     cpu / cuda:X ")
    parser.add_argument("--max_new_tokens",  type=int, default=2048, help="     token            ")
    parser.add_argument("--state_path",      default="", help="trainer_state.json           checkpoint  ")
    parser.add_argument("--skip_auto_sam",   action="store_true", default=False,
                        help="     --no_auto_extract_sam    SAM3     ")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Checkpoint eval worker   ")
    logger.info("  checkpoint : %s", args.checkpoint)
    logger.info("  image      : %s", args.image)
    logger.info("  device     : %s", args.device)
    logger.info("  output_dir : %s", args.output_dir)
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1:   
    infer_dir = step1_inference(args)

    # Step 2:      
    step2_voxel_vis(args, infer_dir)

    # Step 3:     
    step3_loss_curve(args)

    logger.info("=" * 60)
    logger.info("Checkpoint eval worker        : %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
