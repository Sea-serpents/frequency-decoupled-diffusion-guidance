"""Reconstruct one FFHQ-style RGB input with the frozen Gaussian protocol."""

import argparse
from pathlib import Path

from fgps_runtime import activate, frequency_filter
from gaussian_profile import guidance_args, model_args, read_profile, sampler_args

KERNEL_SHA256 = "54adab092d941df97cbcae8bcc8106bdd18da60f8b6268efec4e56dd1a6fad36"


def build_operator(profile, device):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import torch
    from protocol_utils import SpatialConvOperator, tensor_sha256

    o = profile["observation"]
    if o["boundary"] != "zero" or o["resize"] != "rgb-lanczos-if-needed":
        raise ValueError("Unsupported observation protocol")
    width = o["gaussian_width"]
    impulse = np.zeros((width, width), dtype=np.float32)
    impulse[width // 2, width // 2] = 1.0
    kernel = torch.from_numpy(gaussian_filter(impulse, sigma=o["gaussian_std"]))
    if tensor_sha256(kernel) != KERNEL_SHA256:
        raise RuntimeError("Kernel differs from the frozen Gaussian protocol")
    return SpatialConvOperator(kernel, device)


def read_demo(args, profile, operator):
    import torch
    from protocol_utils import load_image, make_x_t

    gt = load_image(args.image)
    with torch.no_grad():
        clean = operator.forward(gt.to(args.device)).cpu()
        rng = torch.Generator(device="cpu").manual_seed(args.noise_seed)
        measurement = clean + profile["observation"]["noise_std"] * torch.randn(clean.shape, generator=rng)
    return gt, measurement, make_x_t(args.seed)


def prepare(profile, branch, device):
    import final_guidance  # Register the project method in the external registry.
    from guided_diffusion.condition_methods import get_conditioning_method
    from guided_diffusion.gaussian_diffusion import create_sampler
    from guided_diffusion.measurements import get_noise

    operator = build_operator(profile, device)
    factory = sampler_args(profile, branch)
    condition = get_conditioning_method("fgps_dps_map", operator,
                                        get_noise(name="gaussian", sigma=profile["observation"]["noise_std"]),
                                        cond_model=frequency_filter(device),
                                        T=factory["steps"], **guidance_args(profile))
    return operator, condition, create_sampler(**factory)


def load_model(profile, checkpoint, device):
    import torch
    from guided_diffusion.unet import create_model
    from protocol_utils import file_sha256

    if file_sha256(checkpoint) != profile["model"]["checkpoint_sha256"]:
        raise ValueError("Wrong FFHQ checkpoint SHA256")
    model = create_model(**model_args(profile, checkpoint))
    # The pinned factory may catch load errors; require a strict successful reload.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    del state
    return model.to(device).eval()


def run_sampler(model, sampler, condition, noisy, initial, seed, device, output):
    from protocol_utils import seed_everything

    seed_everything(seed)
    return sampler.p_sample_loop(model=model, x_start=initial.to(device).requires_grad_(),
                                 measurement=noisy.to(device), measurement_cond_fn=condition.conditioning,
                                 record=False, save_root=str(output), label=None)


def valid_tensor(tensor):
    return tuple(tensor.shape) == (1, 3, 256, 256) and tensor.isfinite().all().item()


def main():
    profile = read_profile()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=profile["reconstruction"]["branches"], required=True)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=profile["reconstruction"]["sampling_seed"])
    parser.add_argument("--noise-seed", type=int, default=profile["observation"]["noise_seed"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not args.preflight and (args.checkpoint is None or args.output is None):
        parser.error("--checkpoint and --output are required for sampling")
    activate(args.runtime)
    import torch
    from protocol_utils import file_sha256, save_image_tensor

    device = torch.device(args.device)
    operator, condition, sampler = prepare(profile, args.branch, device)
    gt, measurement, initial = read_demo(args, profile, operator)
    if not all(valid_tensor(t) for t in (gt, measurement, initial)):
        raise ValueError("Invalid demo tensor")
    if args.preflight:
        if args.checkpoint and file_sha256(args.checkpoint) != profile["model"]["checkpoint_sha256"]:
            raise ValueError("Wrong FFHQ checkpoint SHA256")
        print(f"Preflight: {type(operator).__name__}, {type(condition).__name__}, {type(sampler).__name__}")
        return

    model = load_model(profile, args.checkpoint, device)
    args.output.mkdir(parents=True, exist_ok=False)
    rec = run_sampler(model, sampler, condition, measurement, initial, args.seed, device, args.output)
    if not valid_tensor(rec):
        raise RuntimeError("Nonfinite or wrong-shaped reconstruction")
    for kind, tensor in (("recon", rec), ("label", gt), ("input", measurement)):
        save_image_tensor(tensor[0], args.output / kind / "00000.png")
    print("Saved index 0")


if __name__ == "__main__":
    main()
