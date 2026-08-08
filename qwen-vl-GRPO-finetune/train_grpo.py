#!/usr/bin/env python3
"""
train_grpo.py

Qwen3-VL GRPO for PhysX-CoT

     
  1.    CLI      ModelConfig / DataConfig / RewardConfig / SAMSetupConfig
  2.    V3 GRPO JSONL   ExampleAdapter      train/eval Dataset
  3. setup_model_and_processor 
        special token    + base load + SFT adapter merge
      + merger / sam_projector    + embed_tokens hook    + GRPO LoRA
  4. SAMFeatureLoader npz     + RewardEngine + GRPORewardWrapper
  5. PhysXGRPOTrainer trl.GRPOTrainer         SAM      
     rollout / forward   
  6. RewardComponentLoggingCallback   main / sub / info      logs
  7. trainer.train()      LoRA + sam_projector.pt + merger_weights.pt

        /    /                "  " 
  configs/   - dataclass
  parsers/   -    /    / completion   
  rewards/   - reward    + engine + wrapper
  trainer/   - SAM    /      / GRPOTrainer    / callback
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from transformers import set_seed

try:
    from trl import GRPOConfig
except Exception:  # pragma: no cover
    GRPOConfig = None

from configs import DataConfig, ModelConfig, RewardConfig, ScriptConfig, TaskConfig
from parsers import (
    CompletionParser,
    ExampleAdapter,
    adapt_dataset,
    build_arg_parser,
    load_datasets,
)
from rewards import GRPORewardWrapper, RewardEngine, VoxelGTLoader
from trainer import (
    PhysXGRPOTrainer,
    RewardComponentLoggingCallback,
    SAMFeatureLoader,
    SAMSetupConfig,
    setup_model_and_processor,
)

LOGGER = logging.getLogger("train_grpo")


#                                                                              
#    /   
#                                                                              


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def maybe_save_code_snapshot(output_dir: str) -> None:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / "train_grpo_snapshot.py"
    try:
        source = Path(__file__).read_text(encoding="utf-8")
        snapshot_path.write_text(source, encoding="utf-8")
        LOGGER.info("Saved code snapshot to %s", snapshot_path)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to save code snapshot: %s", exc)


def inspect_random_examples(ds: Dataset, n: int = 2) -> None:
    if len(ds) == 0:
        LOGGER.warning("Dataset is empty; nothing to inspect.")
        return
    n = min(n, len(ds))
    for idx in random.sample(range(len(ds)), n):
        row = ds[idx]
        LOGGER.info("Sample[%d] keys: %s", idx, list(row.keys()))
        LOGGER.info("Sample[%d] sam_feature: %s", idx, row.get("sam_feature"))
        LOGGER.info("Sample[%d] meta: %s", idx, row.get("meta"))


#                                                                              
#       
#                                                                              


def build_training_args(args) -> "GRPOConfig":
    if GRPOConfig is None:
        raise ImportError("trl is not installed. Please run: pip install trl")
    return GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        max_grad_norm=args.max_grad_norm,
        num_iterations=args.num_iterations,
        scale_rewards=False,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
    )


#                                                                              
#     
#                                                                              


def build_model_cfg(args) -> ModelConfig:
    return ModelConfig(
        model_name_or_path     = args.model_name_or_path,
        processor_name_or_path = args.processor_name_or_path,
        torch_dtype            = args.torch_dtype,
        attn_implementation    = args.attn_implementation,
        trust_remote_code      = args.trust_remote_code,
        use_lora               = args.use_lora,
        lora_r                 = args.lora_r,
        lora_alpha             = args.lora_alpha,
        lora_dropout           = args.lora_dropout,
        freeze_vision_tower    = args.freeze_vision_tower,
        gradient_checkpointing = args.gradient_checkpointing,
    )


def build_sam_cfg(args) -> SAMSetupConfig:
    return SAMSetupConfig(
        sft_adapter_path     = args.sft_adapter_path,
        merger_weights_path  = args.merger_weights_path,
        sam_projector_path   = args.sam_projector_path,
        sam_proj_hidden      = args.sam_proj_hidden,
        sam_in_dim           = args.sam_in_dim,
        freeze_sam_projector = args.freeze_sam_projector,
        freeze_merger        = args.freeze_merger,
        freeze_embed_tokens  = args.freeze_embed_tokens,
        freeze_lm_head       = args.freeze_lm_head,
        local_files_only     = args.local_files_only,
    )


def build_data_cfg(args) -> DataConfig:
    return DataConfig(
        train_file          = args.train_file,
        eval_file           = args.eval_file,
        image_root          = args.image_root,
        dataset_root        = args.dataset_root,
        dataset_format      = args.dataset_format,
        max_train_samples   = args.max_train_samples,
        max_eval_samples    = args.max_eval_samples,
        prompt_style        = args.prompt_style,
        expect_ground_truth = args.expect_ground_truth,
    )


def build_reward_cfg(args) -> RewardConfig:
    return RewardConfig(
        loc_weight    = args.loc_weight,
        coarse_weight = args.coarse_weight,
        detail_weight = args.detail_weight,
        phys_weight   = args.phys_weight,
        pen_weight    = args.pen_weight,

        loc_count_weight  = args.loc_count_weight,
        loc_bbox2d_weight = args.loc_bbox2d_weight,
        loc_bbox3d_weight = args.loc_bbox3d_weight,
        count_decay_gamma = args.count_decay_gamma,
        bbox3d_iou_beta   = args.bbox3d_iou_beta,
        bbox3d_l1_eta     = args.bbox3d_l1_eta,

        coarse_shape_weight = args.coarse_shape_weight,
        coarse_axis_weight  = args.coarse_axis_weight,
        coarse_ratio_weight = args.coarse_ratio_weight,

        detail_parse_weight  = args.detail_parse_weight,
        detail_range_weight  = args.detail_range_weight,
        detail_local_weight  = args.detail_local_weight,
        detail_global_weight = args.detail_global_weight,
        detail_local_iou_w   = args.detail_local_iou_w,
        detail_local_f1_w    = args.detail_local_f1_w,

        phys_overall_weight = args.phys_overall_weight,
        phys_group_weight   = args.phys_group_weight,
        phys_logic_weight   = args.phys_logic_weight,

        cons_enable           = args.cons_enable,
        cons_weight           = args.cons_weight,
        cons_center_weight    = args.cons_center_weight,
        cons_proj_weight      = args.cons_proj_weight,
        cons_proj_ratio_w     = args.cons_proj_ratio_w,
        cons_proj_shape_w     = args.cons_proj_shape_w,
        cons_proj_softmin_tau = args.cons_proj_softmin_tau,

        pen_format_weight     = args.pen_format_weight,
        pen_overflow_weight   = args.pen_overflow_weight,
        pen_dependency_weight = args.pen_dependency_weight,

        voxel_grid_size  = args.voxel_grid_size,
        voxel_cache_size = args.voxel_cache_size,

        enable_component_logging = args.enable_component_logging,
        final_clamp_min          = args.final_clamp_min,
        final_clamp_max          = args.final_clamp_max,
    )


#                                                                              
#      +   
#                                                                              


def build_datasets(data_cfg: DataConfig, task_cfg: TaskConfig):
    LOGGER.info("Loading datasets from %s", data_cfg.train_file)
    raw = load_datasets(data_cfg)
    adapter = ExampleAdapter(
        task_cfg     = task_cfg,
        image_root   = data_cfg.image_root,
        dataset_root = data_cfg.dataset_root,
    )
    train_ds = adapt_dataset(raw["train"], adapter)
    eval_ds  = adapt_dataset(raw["eval"], adapter) if "eval" in raw else None
    LOGGER.info("Train size: %d", len(train_ds))
    if eval_ds is not None:
        LOGGER.info("Eval size: %d", len(eval_ds))
    inspect_random_examples(train_ds, n=2)
    return train_ds, eval_ds


#                                                                              
#      SAM Projector / Merger    
#                                                                              


def save_artifacts(
    trainer: PhysXGRPOTrainer,
    processor,
    output_dir: str,
) -> None:
    """   LoRA adapter + processor + sam_projector.pt + merger_weights.pt """
    if not trainer.args.should_save:
        return

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Unwrap PEFT for the SFT-specific modules.
    inner = trainer.model
    for _ in range(4):
        if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
            inner = inner.base_model.model
        else:
            break

    visual_owner = inner
    visual = None
    for _ in range(4):
        visual = getattr(visual_owner, "visual", None)
        if visual is not None:
            break
        nested = getattr(visual_owner, "model", None)
        if nested is None or nested is visual_owner:
            break
        visual_owner = nested

    if visual is not None and hasattr(visual, "merger"):
        merger_state = {
            n: p.detach().cpu()
            for n, p in visual.merger.named_parameters()
        }
        torch.save(merger_state, os.path.join(output_dir, "merger_weights.pt"))
        LOGGER.info("Saved visual.merger   merger_weights.pt")

    if hasattr(inner, "sam_projector"):
        torch.save(
            inner.sam_projector.state_dict(),
            os.path.join(output_dir, "sam_projector.pt"),
        )
        LOGGER.info("Saved sam_projector   sam_projector.pt")


#                                                                              
#    
#                                                                              


def main() -> None:
    args = build_arg_parser().parse_args()

    setup_logging(args.log_level)
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    if args.save_code_snapshot:
        maybe_save_code_snapshot(args.output_dir)
    LOGGER.info("Parsed args: %s", vars(args))

    #                                                                         
    model_cfg  = build_model_cfg(args)
    sam_cfg    = build_sam_cfg(args)
    data_cfg   = build_data_cfg(args)
    reward_cfg = build_reward_cfg(args)
    task_cfg   = TaskConfig()

    #                                                                         
    train_ds, eval_ds = build_datasets(data_cfg, task_cfg)

    if args.dry_run:
        LOGGER.info("Dry run: dataset OK, exiting before model load.")
        return

    #         11                                                        
    setup = setup_model_and_processor(model_cfg, sam_cfg)

    #    SAM     npz   tensor                                             
    sam_loader = SAMFeatureLoader(
        cache_size=args.sam_cache_size,
        feat_dim=args.sam_in_dim,
    )

    #    Reward Engine + Wrapper                                               
    voxel_loader = VoxelGTLoader(
        dataset_root=data_cfg.dataset_root,
        cache_size=reward_cfg.voxel_cache_size,
    )
    reward_wrapper = GRPORewardWrapper(
        RewardEngine(
            reward_cfg   = reward_cfg,
            parser       = CompletionParser(task_cfg),
            task_cfg     = task_cfg,
            voxel_loader = voxel_loader,
        )
    )

    #    Trainer                                                             
    training_args = build_training_args(args)

    trainer = PhysXGRPOTrainer(
        model            = setup.model,
        processing_class = setup.processor,
        reward_funcs     = reward_wrapper,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        sam_injector     = setup.sam_injector,
        sam_loader       = sam_loader,
        sam_dtype        = setup.model.dtype if hasattr(setup.model, "dtype") else torch.bfloat16,
    )

    #    Callbacks                                                             
    trainer.add_callback(RewardComponentLoggingCallback(
        reward_wrapper = reward_wrapper,
        sam_loader     = sam_loader,
    ))

    #                                                                         
    if args.do_train:
        LOGGER.info("Starting GRPO training...")
        train_result = trainer.train()
        LOGGER.info("Training finished. Result: %s", train_result)
        save_artifacts(trainer, setup.processor, args.output_dir)

    if args.do_eval and eval_ds is not None:
        LOGGER.info("Starting evaluation...")
        LOGGER.info("Eval metrics: %s", trainer.evaluate())

    #    hook        /           
    if setup.hook_handle is not None:
        try:
            setup.hook_handle.remove()
        except Exception:
            pass


if __name__ == "__main__":
    main()
