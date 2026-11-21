# Frequency-Decoupled Diffusion Guidance

The v0.1.0 code release for the FFHQ Gaussian single-image demo of
[Frequency-Decoupled Diffusion Guidance for Non-Blind Image Deblurring](https://arxiv.org/abs/2610.06221).
It includes the training-free guidance, the zero-boundary Gaussian operator wrapper,
and one-image DDPM/DDIM entrypoints. Both 1,000-step branches, saved-RGB fusion,
and the optional PSNR/SSIM evaluator were validated end to end on an RTX 4060 Laptop GPU.

Motion, ImageNet, Uniform, and reproduction of the Full1000 paper benchmarks are
outside the scope of this release. No weights, datasets, images, third-party
source code, or historical benchmark evaluator are included.

## Requirements

The validation environment uses Python 3.12, PyTorch 2.6.0 and torchvision 0.21.0 on CUDA 11.8.
Use the official PyTorch build for your driver and operating system. For the tested CUDA build:

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Obtain the following dependencies independently, subject to their applicable terms, and
place their unmodified Git checkouts at `runtime/fgps` and `runtime/motionblur`:

| Dependency | Required commit |
| --- | --- |
| [FGPS](https://github.com/darshanthaker/FGPS) | `499e21d4488d2c15b0177546907db288dbb9c470` |
| [motionblur](https://github.com/LeviBorodenko/motionblur) | `578137887b2067c532bd4e422ff5afdcabfccebf` |

Use `git clone -c core.autocrlf=false --no-checkout URL DESTINATION` and
`git -C DESTINATION checkout --detach COMMIT` with the repository URL and commit above.
Use fresh checkouts: runtime verification checks the pinned trees and tracked file bytes.
No FGPS patch, overlay copy or upstream source edit is needed. FGPS imports motionblur at
module load even for this Gaussian demo.

Obtain [the FFHQ checkpoint linked by DPS](https://drive.google.com/file/d/1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh/view)
independently, and check that `ffhq_10m.pt` has SHA256
`81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`.
Use your own RGB image that you are authorized to process.

## Reconstruct and Fuse

```sh
python -B scripts/fgps_runtime.py --runtime runtime
python -B scripts/reconstruct.py --branch ddpm --runtime runtime --image input.png --checkpoint models/ffhq_10m.pt --output outputs/ddpm
python -B scripts/reconstruct.py --branch ddim --runtime runtime --image input.png --checkpoint models/ffhq_10m.pt --output outputs/ddim
python -B scripts/fuse.py --ddpm outputs/ddpm/recon --ddim outputs/ddim/recon --output outputs/fused
```

Each branch uses 1,000 reverse steps and 1,000 logical UNet calls; DDIM uses eta=0.
The profile defines RGB preprocessing, the Gaussian observation, both seeds, the
executed guidance step schedule (0.60 early to 2.5 late), and float32 0.5/0.5 fusion
of the saved PNGs with nearest-even rounding before conversion to uint8.
Both branches use the same observation and initial-noise seeds by default.

```sh
python -m pip install -r requirements-evaluation.txt
python -B scripts/evaluate.py --reference outputs/ddpm/label --recon outputs/fused --csv outputs/fused.csv
python -B -m unittest discover -s tests -v
```

For the dependency-backed CPU test, set `FDG_TEST_RUNTIME` to the path of the two
external checkouts' parent directory. The evaluator reports scores for this demo
input, not the paper benchmark.

## License

The [MIT License](LICENSE) applies to the project-owned code. It does not cover
external implementations, model weights, datasets, or images. Obtain external
dependencies independently and follow their applicable terms. See
[Third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{wang2026frequency,
  title={Frequency-Decoupled Diffusion Guidance for Non-Blind Image Deblurring},
  author={Wang, Sihan and Huang, Jinshu and Su, Haibin and Xue, Yunhua},
  journal={arXiv preprint arXiv:2610.06221},
  year={2026}
}
```
