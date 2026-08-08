"""Decode PhysX-CoT voxel predictions into textured GLB assets."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")

LOGGER = logging.getLogger("physx_cot.decoder")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def decode_directory(
    images_dir: Path,
    output_dir: Path,
    decoder_path: str,
    *,
    seed: int = 1,
    voxel_size: int = 32,
    resolution: int = 64,
    simplify: float = 0.5,
    texture_size: int = 1024,
) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipeline = TrellisImageTo3DPipeline.from_pretrained(decoder_path)
    if not torch.cuda.is_available():
        raise RuntimeError("The TRELLIS decoder requires a CUDA device.")
    pipeline.cuda()

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        sample_dir = output_dir / image_path.stem
        coordinates_path = sample_dir / "allind.npy"
        if not coordinates_path.is_file():
            LOGGER.warning("Skipping %s: %s is missing", image_path.name, coordinates_path)
            continue

        coordinates = np.load(coordinates_path)
        offset = resolution // 2 - voxel_size // 2
        coordinates = coordinates + offset
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(f"Invalid voxel coordinate shape for {image_path.name}: {coordinates.shape}")
        if np.any(coordinates < 0) or np.any(coordinates >= resolution):
            raise ValueError(f"Voxel coordinates are outside the {resolution}^3 decoder grid: {image_path.name}")

        occupancy = torch.zeros(
            1, resolution, resolution, resolution, dtype=torch.float32, device="cuda"
        )
        occupancy[:, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1
        occupancy = occupancy.unsqueeze(0)

        with Image.open(image_path) as image:
            outputs = pipeline.run_control(occupancy, image.convert("RGB"), seed=seed)

        asset = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=simplify,
            texture_size=texture_size,
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        asset.export(sample_dir / "sample.glb")
        LOGGER.info("Saved %s", sample_dir / "sample.glb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--decoder_path", default="./pretrain/decoder")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--voxel_size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--simplify", type=float, default=0.5)
    parser.add_argument("--texture_size", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not args.images_dir.is_dir():
        parser.error(f"images directory not found: {args.images_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decode_directory(
        args.images_dir,
        args.output_dir,
        args.decoder_path,
        seed=args.seed,
        voxel_size=args.voxel_size,
        resolution=args.resolution,
        simplify=args.simplify,
        texture_size=args.texture_size,
    )


if __name__ == "__main__":
    main()
