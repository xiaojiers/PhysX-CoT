# Paper and Technical Appendix Alignment

This release treats the manuscript and Technical Appendix as the source of
truth. The following implementation details have been checked explicitly.

## Aligned model and representation

- Base model: `Qwen/Qwen3-VL-8B-Instruct` via
  `Qwen3VLForConditionalGeneration` and `AutoProcessor`.
- Structured state order: part inventory, 2D and 3D boxes, adjacent-part
  relations, coarse primitive cues, surface cues, and object-level physical
  information.
- Geometry: integer closed-interval boxes in a 32 cubed object grid plus
  per-part local RLE occupancy.
- Part limit: 1 to 24 parts, with geometry tags registered for IDs 0 to 23.
- Decoder: geometry reconstruction remains a separate frozen-decoder stage.

## Aligned SFT configuration

- LoRA rank 16, alpha 32, dropout 0.05 on `q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Vision tower frozen; visual merger, SAM projector, resized embeddings, and
  LM head trained jointly with the SFT LoRA adapter.
- Four workers, batch size 2 per device, gradient accumulation 8, two epochs,
  learning rate `2e-5`, weight decay 0.05, cosine schedule, and 3% warmup.
- Images aspect-padded to 448 x 448 and sequences capped at 8704 tokens.
- Gradient clipping 1.0, bfloat16, activation checkpointing, and checkpoint
  evaluation every 500 steps.
- Reported SFT seeds: 17, 29, and 43.

## Aligned GRPO configuration

- Initialized from the selected SFT checkpoint; only the seven LoRA projection
  families remain trainable.
- Four workers, batch size 2 per device, gradient accumulation 4, group size 4,
  one policy iteration, learning rate `1e-5`, KL beta 0.02, temperature 0.8,
  top-p 0.9, maximum prompt 2048, and maximum completion 6528.
- Group-mean-centered advantages are used without reward standard-deviation
  scaling; groups with negligible reward variance are handled by TRL.
- Reported GRPO seeds: 117, 129, and 143.

## Aligned reward contract

- Main weights: localization 0.25, coarse geometry 0.20, voxel detail 0.40,
  and physical consistency 0.15.
- Localization subweights: count 0.20, 2D box 0.30, and 3D box 0.50.
- Coarse subweights: primitive 0.50, axis 0.20, and aspect ratio 0.30.
- Detail subweights: parse 0.10, range 0.15, local 0.50, and global 0.25.
- Physical subweights: format 0.20, group 0.50, and logic 0.30.
- Penalty subweights: format 0.30, overflow 0.30, and dependency 0.40.

## Remaining reproduction boundary

The appendix selects checkpoints with a validation composite built from state,
mesh F-score, Chamfer distance, and parse validity. The checked-in training
loop saves and evaluates every 500 steps, but the final composite ranking still
requires the released validation manifest, frozen decoder checkpoint, and
evaluation assets. The code does not substitute validation loss for this
paper-specific selection rule.
