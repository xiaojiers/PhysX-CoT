# Third-Party Notices

## PhysX-Anything

PhysX-CoT builds on and contains code derived from
[PhysX-Anything](https://github.com/ziangcao0312/PhysX-Anything):

- Source repository: https://github.com/ziangcao0312/PhysX-Anything
- Audited revision: `e221826e6176d940905126d1894f9c1c933b70a8`
- Copyright: Copyright 2023 S-Lab
- License: S-Lab License 1.0

The following files are reproduced from that revision without substantive
changes:

- `configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json`
- `configs/generation/ss_flow_img_dit_L_16l8_fp16.json`
- `configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json`
- `configs/vae/slat_vae_dec_rf_swin8_B_64l8_fp16.json`
- `configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json`
- `configs/vae/ss_vae_conv3d_16l8_fp16.json`
- `qwen-vl-SFT-finetune/scripts/zero2.json`
- `qwen-vl-SFT-finetune/scripts/zero3.json`
- `qwen-vl-SFT-finetune/scripts/zero3_offload.json`
- `qwen-vl-SFT-finetune/tools/check_image.py`
- `render_urdf.py`

Modified or extended portions of the PhysX-Anything pipeline include the
part-splitting, SimReady generation, kinematic evaluation, decoder, physical
evaluation, and vision-language inference stages. PhysX-CoT adds structured
physical chain-of-thought reasoning, updated model integration, SFT and
CoT-aligned GRPO training, reward modeling, evaluation, and orchestration.

The S-Lab License 1.0 terms and disclaimer are reproduced in the repository's
`LICENSE` file. No endorsement by S-Lab or the PhysX-Anything contributors is
implied.

## Other Components

PhysX-CoT integrates with external models and tools including Qwen3-VL, SAM3,
TRELLIS, and NVIDIA PhysX-related simulation tooling. Those components are not
redistributed as model weights or external source packages in this repository
and remain subject to their respective licenses and terms.
