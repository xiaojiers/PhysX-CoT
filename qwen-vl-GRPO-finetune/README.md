# CoT-aligned GRPO

This directory contains the post-SFT optimization stage used by PhysX-CoT.
`train_grpo.py` loads the Qwen3-VL base model, attaches the PhysX-CoT SFT
adapter, injects cached SAM features, and optimizes structured completions with
group-relative policy updates.

## Reward contract

The reward engine follows the Technical Appendix:

| Component | Target | Weight |
| --- | --- | ---: |
| `R_loc` | part count, 2D/3D grounding, projection consistency | 0.25 |
| `R_coarse` | primitive, major axis, aspect ratio | 0.20 |
| `R_detail` | local and global voxel geometry | 0.40 |
| `R_phys` | physical fields and joint consistency | 0.15 |

Format, range, dependency, and missing-field errors are handled by the penalty
terms in `rewards/reward_engine.py`. The exact component options are exposed by
`parsers/arg_parser.py` and the launch script.

## Data

Build the GRPO JSONL files from the released SFT annotation format with
`datasets/build_grpo_dataset.py`. Large generated JSONL files, voxel targets,
images, and SAM features are intentionally excluded from Git.

## Launch

Set the external paths and start distributed training:

```bash
export PHYSX_COT_BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
export PHYSX_COT_SFT_ADAPTER=/path/to/physx-cot-sft
export PHYSX_COT_GRPO_TRAIN=/path/to/grpo_train.jsonl
export PHYSX_COT_GRPO_EVAL=/path/to/grpo_val.jsonl
export PHYSX_COT_IMAGE_ROOT=/path/to/renders
export PHYSX_COT_DATA_ROOT=/path/to/data
bash scripts/run_grpo.sh
```

The launch defaults match Table 5: four A800-class workers, batch size 2 per
device, gradient accumulation 4, four candidates, `1e-5` learning rate,
`0.02` KL coefficient, and no reward standard-deviation scaling.
Use `PHYSX_COT_SEED=117`, `129`, and `143` for the three reported runs.
