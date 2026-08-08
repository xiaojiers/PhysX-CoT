# PhysX-CoT

**Structured Physical Reasoning from a Single Image to Simulation-Ready 3D Assets**

PhysX-CoT converts a single RGB image into a structured physical reasoning
trajectory and a simulation-ready asset. This repository contains the current
inference pipeline, frozen-decoder interface, part splitting, SimReady asset
generation, evaluation, SFT, and CoT-aligned GRPO code.

The manuscript is currently under anonymous review. The public paper link,
checkpoint URLs, dataset manifest, citation, and license will be added only
after release clearance.

Run `python tools/verify_release.py` before creating a public repository. It
checks Python syntax, non-ASCII text and filenames, and accidental private
paths.

See `PAPER_ALIGNMENT.md` for the implementation settings checked against the
manuscript and Technical Appendix, including the remaining checkpoint-selection
boundary that depends on unreleased validation assets.

## Method contract

Round 1 emits an ordered, machine-parseable state trajectory rather than a
single opaque geometry string:

1. **Part decomposition** declares the part inventory and stable semantic IDs.
2. **Visual grounding** predicts 2D boxes for the declared parts.
3. **Spatial grounding** predicts canonical-frame 3D boxes.
4. **Part relations** records adjacent-part support, connection, and motion links.
5. **Coarse geometry** selects primitive/aspect priors.
6. **Surface cues** records hardness, roughness, reflectivity, and transparency.
7. **Object-level physical attributes** records absolute scale, material and
   density, affordance, kinematics, and a physical description.

Rounds 2 through K emit one local-frame run-length encoded voxel sequence per
part. The 3D box carries placement while the local code carries shape. A frozen
decoder reconstructs the mesh; the SimReady stage adds physical metadata and
exports engine-ready files.

Training first applies state-level supervised fine-tuning. CoT-aligned GRPO then
optimizes process rewards for localization, coarse geometry, detailed local and
global geometry, physical consistency, and format/dependency penalties.

## Four-stage pipeline

```text
1_vlm_cot.py       image -> structured CoT + physical attributes + local RLE
2_decoder.py       image + local RLE -> sample.glb
3_split.py         sample.glb + boxes -> objs/<part>/<part>.obj
4_simready_gen.py  basic_info.txt + objs/ -> basic_info.json + basic.urdf
```

`run_inference.py` orchestrates the stages, normalizes the VLM output, and
supports resumable per-sample execution. Every stage also exposes a standalone
command-line interface with explicit input and output paths.

## Repository layout

- Root scripts: VLM inference, decoder, part splitting, SimReady generation,
  evaluation, and rendering helpers.
- `configs/`: VAE and image-conditioned generation configurations.
- `dataset/`: annotation, prompt, statistics, and feature-preparation tools.
- `qwen-vl-SFT-finetune/`: state-level SFT components and launch scripts.
- `qwen-vl-GRPO-finetune/`: CoT-aligned rewards, parsers, trainer, and launch
  scripts.
- `pretrain/`: intentionally empty in this release; checkpoints are not
  redistributed.

Large generated datasets (`grpo_train.jsonl`, `grpo_val.jsonl`), model weights,
cache directories, and Python bytecode are excluded. They should be supplied
through a separately documented data or model release.

## Environment

The appendix reports Ubuntu 22.04.5, Python 3.11.9, PyTorch 2.5.1, CUDA 12.4,
cuDNN 9.1, Transformers 4.57, PEFT 0.15.2, and TRL 0.17. Install the PyTorch
wheel that matches the local CUDA runtime, then install the remaining packages:

```bash
pip install -r requirements.txt
```

Install `requirements-training.txt` for SFT/GRPO or
`requirements-evaluation.txt` for the complete evaluation suite. Kinematic
evaluation uses an external vision-language API and requires credentials; the
geometry and physical-attribute evaluators run locally.

The following external components are not vendored here:

- Qwen3-VL base model and the PhysX-CoT LoRA adapter;
- SAM3 source and checkpoint for on-the-fly visual features;
- TRELLIS decoder and its DINOv2 dependency;
- the renderer and simulator dependencies required by the evaluation scripts.

The repository accepts local paths through CLI arguments or these optional
environment variables: `PHYSX_COT_BASE_MODEL`, `PHYSX_COT_ADAPTER`,
`PHYSX_COT_SAM3_ROOT`, `PHYSX_COT_SAM3_CHECKPOINT`, and
`PHYSX_COT_SAM_FEATURE_DIR`.

## Example invocation

```bash
python run_inference.py \
  --images_dir ./eval_data/images \
  --method_name demo \
  --output_root ./outputs \
  --ckpt_vlm ./pretrain/vlm \
  --vlm_base_model Qwen/Qwen3-VL-8B-Instruct \
  --stages 1,2,3,4
```

Start with `--stages 1` or `--stages 3,4` when validating an individual stage.
The generated asset package follows the contract shown above and can be
inspected with the evaluation and rendering helpers.

For state-level SFT, use `qwen-vl-SFT-finetune/scripts/run_sft.sh`. For the
post-SFT policy optimization described in the appendix, use
`qwen-vl-GRPO-finetune/scripts/run_grpo.sh`. Both scripts read paths from the
`PHYSX_COT_*` environment variables documented in the scripts.

## Release checklist

- [ ] Add the final arXiv identifier and repository URL.
- [ ] Add a license chosen by the authors.
- [ ] Publish checkpoint and dataset provenance.
- [ ] Run a clean end-to-end sample on the target CUDA image.
- [ ] Confirm that the public package does not include private paths or
      restricted assets.
