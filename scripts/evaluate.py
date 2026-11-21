"""Evaluate saved RGB PNGs for new runs; not a frozen benchmark replay tool."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fuse import load_rgb, pairs


def fidelity(ref, pred):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    return (
        peak_signal_noise_ratio(ref, pred, data_range=1.0),
        structural_similarity(ref, pred, channel_axis=2, data_range=1.0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--recon", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    names = pairs(args.reference, args.recon)
    rows = []
    for name in names:
        ref = load_rgb(args.reference / name).astype(np.float32) / 255.0
        pred = load_rgb(args.recon / name).astype(np.float32) / 255.0
        if ref.shape != pred.shape:
            raise ValueError(f"Image shape mismatch: {name}")
        psnr, ssim = fidelity(ref, pred)
        rows.append(dict(name=name, psnr=psnr, ssim=ssim))
    means = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "name"}
    with args.csv.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"count": len(rows), **means}, indent=2))


if __name__ == "__main__":
    main()
