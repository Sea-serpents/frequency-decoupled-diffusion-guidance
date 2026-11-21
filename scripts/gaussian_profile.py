"""Map the FFHQ Gaussian profile to the pinned runtime's factory arguments."""

import json
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "configs" / "ffhq_gaussian.json"


def read_profile():
    return json.loads(FILE.read_text(encoding="utf-8"))


def model_args(profile, checkpoint):
    m = profile["model"]
    return dict(image_size=m["resolution"], num_channels=m["base_width"],
                num_res_blocks=m["blocks_per_level"], attention_resolutions=m["attention_resolution"],
                num_heads=m["attention_heads"], num_head_channels=m["channels_per_head"],
                num_heads_upsample=-1, channel_mult="", class_cond=False, dropout=0.0,
                learn_sigma=True, resblock_updown=True, use_checkpoint=False, use_fp16=False,
                use_new_attention_order=False, use_scale_shift_norm=True, model_path=str(checkpoint))


def sampler_args(profile, branch):
    r = profile["reconstruction"]
    if branch not in r["branches"] or r["ddim_eta"] != 0.0:
        raise ValueError("Expected a frozen DDPM or eta=0 DDIM branch")
    n = r["iterations"]
    return dict(sampler=branch, steps=n, timestep_respacing=n, noise_schedule="linear",
                model_mean_type="epsilon", model_var_type="learned_range",
                dynamic_threshold=False, clip_denoised=True, rescale_timesteps=False)


def guidance_args(profile):
    g = profile["guidance"]
    lo, hi = g["cutoff_bins"]
    step = g["executed_step"]
    measure, prior, tv = (g[k] for k in ("measurement", "spectral_prior", "trajectory_tv"))
    # The historical factory indexes its cosine table from T-1 down to zero.
    return dict(start_bound=lo, end_bound=hi, schedule="exponential", step_schedule="cosine",
                step_start=step["late"], step_end=step["early"], eps=g["normalization_epsilon"],
                lambda_lp=measure["strength"], lp_t0=measure["gate"][0], lp_k=measure["gate"][1],
                lambda_prior=prior["strength"], prior_t0=prior["gate"][0], prior_k=prior["gate"][1],
                reliability_lambda=prior["kernel_ridge"],
                lambda_reg=tv["strength"], reg_t0=tv["gate"][0], reg_k=tv["gate"][1],
                step_size=0.35, lambda_hp=0.0, hp_t0=0.35, hp_k=12.0, normalize_grad=True,
                grad_clip_norm=1.0, grad_clip_value=0.0, update_interval=1, reg_type="tv",
                reg_tv_weight=1.0, reg_f1_weight=1.0, reliability_mode="auto", reliability_floor=0.0,
                reliability_mask_shifted=False, use_closed_form_map=False, lambda_map=0.0,
                lambda_cf_aux=0.0, prior_mode="loss", prior_freq_mode="x0_unreliable_high_l1",
                prior_freq_alpha=0.0)
