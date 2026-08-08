# State-level supervised fine-tuning

This directory contains the PhysX-CoT state-level SFT pipeline. The target
sequence supervises the ordered reasoning states, object-level physical fields,
and per-part local geometry tokens used by the inference frontend.

## Expected inputs

- the `Qwen/Qwen3-VL-8B-Instruct` base model reported in the appendix;
- JSONL annotations in the PhysX-CoT conversation format;
- rendered RGB images;
- optional cached SAM features for each declared part.

Large datasets, cached tensors, model checkpoints, and evaluation images are
excluded from this repository.

## Launch

```bash
export PHYSX_COT_BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
export PHYSX_COT_TRAIN_JSONL=/path/to/train.jsonl
export PHYSX_COT_VAL_JSONL=/path/to/val.jsonl
export PHYSX_COT_IMAGE_ROOT=/path/to/renders
export PHYSX_COT_SAM_FEATURE_DIR=/path/to/sam_features
export PHYSX_COT_SFT_OUTPUT=/path/to/physx-cot-sft
bash scripts/run_sft.sh
```

The release defaults match the appendix: four A800-class workers, two epochs,
batch size 2 per device, gradient accumulation 8, 448 x 448 aspect-padded
images, an 8704-token sequence cap, and checkpoint validation every 500 steps.
Run seeds `17`, `29`, and `43` with `PHYSX_COT_SEED` for the three reported
SFT runs. The appendix's composite checkpoint ranking is applied after the
geometry and state evaluators finish; it is not approximated with validation
loss in the training script.

Use `scripts/build_cache_v3.sh` to prepare cached samples. `run_sft.sh`
automatically resumes from the latest checkpoint in the configured output
directory.
