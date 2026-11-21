"""Fuse saved PNG pairs with the frozen FFHQ Gaussian RGB arithmetic."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from gaussian_profile import read_profile


def fuse_rgb(ddpm, ddim, weight):
    if ddpm.shape != ddim.shape or any(
        image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
        for image in (ddpm, ddim)
    ):
        raise ValueError("Expected equal-shaped uint8 RGB arrays")
    first = ddim.astype(np.float32)
    second = ddpm.astype(np.float32)
    blended = (1.0 - weight) * first + weight * second
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def pairs(first, second):
    names = sorted(path.name for path in first.glob("*.png") if path.is_file())
    if not names or names != sorted(path.name for path in second.glob("*.png") if path.is_file()):
        raise ValueError("Input directories must contain the same nonempty list of PNG names")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddpm", type=Path, required=True)
    parser.add_argument("--ddim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weight = read_profile()["reconstruction"]["ddpm_output_weight"]
    names = pairs(args.ddpm, args.ddim)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in names:
        fused = fuse_rgb(load_rgb(args.ddpm / name), load_rgb(args.ddim / name), weight)
        Image.fromarray(fused).save(args.output / name)
    print(f"Fused {len(names)} saved RGB PNGs")


if __name__ == "__main__":
    main()
