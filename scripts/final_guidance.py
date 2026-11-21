"""Frequency-decoupled guidance for the locked Gaussian and Motion configurations."""

import numpy as np
import torch
from torch.fft import fftn, ifftshift
from torch.nn import functional as F
from guided_diffusion.condition_methods import ConditioningMethod, register_conditioning_method


@register_conditioning_method(name="fgps_dps_map")
class FGPSDPSMAP(ConditioningMethod):
    def __init__(self, operator, noiser, cond_model, schedule, start_bound, end_bound, T, **kwargs):
        super().__init__(operator, noiser)
        locked = {
            "step_schedule": "cosine", "update_interval": 1, "reg_type": "tv",
            "reliability_mode": "auto", "reliability_mask_shifted": False,
            "normalize_grad": True, "grad_clip_value": 0.0, "lambda_hp": 0.0,
            "use_closed_form_map": False, "lambda_cf_aux": 0.0, "lambda_map": 0.0,
            "lambda_meas_space": 0.0, "prior_mode": "loss",
            "prior_freq_mode": "x0_unreliable_high_l1",
        }
        for key, value in locked.items():
            if kwargs.get(key, value) != value:
                raise ValueError(f"{key}={value!r} is required by this release")
        if cond_model is None or schedule != "exponential" or noiser.__name__ != "gaussian":
            raise ValueError("Expected a frequency filter, exponential cutoff and Gaussian noise")
        self.cond_model = cond_model
        self.T = int(T)
        self.eps = float(kwargs.get("eps", 1e-8))
        step_start = float(kwargs["step_start"])
        step_end = float(kwargs["step_end"])
        t_lin = np.linspace(0, self.T, self.T, dtype=np.float32)
        values = 0.5 * (step_start + step_end) + 0.5 * (step_start - step_end) * np.cos(np.pi * t_lin / max(self.T, 1))
        self.step_size = torch.tensor(values, dtype=torch.float32)
        self.cutoff_schedule = (
            end_bound - (end_bound - start_bound) * torch.exp(-torch.linspace(0, 5, self.T))
        ).flip(0)

        self.lambda_lp = float(kwargs.get("lambda_lp", 1.0))
        self.lambda_hp = 0.0
        self.lambda_prior = float(kwargs.get("lambda_prior", 0.2))
        self.lambda_reg = float(kwargs.get("lambda_reg", 0.05))
        self.lp_t0 = float(kwargs.get("lp_t0", -0.2))
        self.lp_k = float(kwargs.get("lp_k", 10.0))
        self.hp_t0 = float(kwargs.get("hp_t0", 0.35))
        self.hp_k = float(kwargs.get("hp_k", 12.0))
        self.prior_t0 = float(kwargs.get("prior_t0", 0.2))
        self.prior_k = float(kwargs.get("prior_k", 10.0))
        self.reg_t0 = float(kwargs.get("reg_t0", 0.55))
        self.reg_k = float(kwargs.get("reg_k", 12.0))
        self.reg_tv_weight = float(kwargs.get("reg_tv_weight", 1.0))
        self.reliability_lambda = float(kwargs.get("reliability_lambda", 1e-3))
        self.reliability_floor = float(kwargs.get("reliability_floor", 0.0))
        self._kernel_fft_cache = {}

    @staticmethod
    def _tv(x):
        dx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        return dx + dy

    def _step_index(self, t):
        if torch.is_tensor(t):
            return int(t.flatten()[0].detach().item())
        return int(t)

    def _progress(self, idx):
        denom = max(self.T - 1, 1)
        return 1.0 - float(idx) / float(denom)

    def _gate(self, progress, t0, k, device, dtype):
        p = torch.tensor(progress, device=device, dtype=dtype)
        return torch.sigmoid(k * (p - t0))

    def _step_size_at(self, idx, device, dtype):
        idx = int(max(0, min(idx, self.T - 1)))
        return self.step_size[idx].to(device=device, dtype=dtype)

    def _cutoff_at(self, idx, device):
        idx = int(max(0, min(idx, self.T - 1)))
        return self.cutoff_schedule[idx].to(device=device)

    def _build_lowpass_mask(self, x, cutoff):
        mask_shifted = self.cond_model.get_mask(x.shape, cutoff).to(device=x.device, dtype=x.real.dtype)
        return ifftshift(mask_shifted, dim=(-2, -1))

    def _kernel_fft_from_operator(self, x):
        key = (x.shape[-2], x.shape[-1], x.device.type, x.device.index, x.dtype)
        if key in self._kernel_fft_cache:
            return self._kernel_fft_cache[key]
        kernel = self.operator.get_kernel()
        if kernel.ndim != 4 or kernel.shape[:2] != (1, 1):
            raise ValueError("Expected the spatial operator's (1, 1, H, W) kernel")
        kernel2d = kernel[0, 0]
        kernel2d = kernel2d.to(device=x.device, dtype=x.real.dtype)
        h, w = x.shape[-2], x.shape[-1]
        kh, kw = kernel2d.shape[-2], kernel2d.shape[-1]
        if kh > h or kw > w:
            raise ValueError("Image is smaller than the blur kernel")
        psf = torch.zeros((h, w), device=x.device, dtype=x.real.dtype)
        top = (h - kh) // 2
        left = (w - kw) // 2
        psf[top : top + kh, left : left + kw] = kernel2d
        # Centered PSF -> OTF, without orthonormal scaling.
        kernel_fft = fftn(ifftshift(psf, dim=(-2, -1)), dim=(-2, -1)).unsqueeze(0).unsqueeze(0)
        self._kernel_fft_cache[key] = kernel_fft
        return kernel_fft

    def _reliability_mask(self, x):
        kernel_fft = self._kernel_fft_from_operator(x)
        power = torch.abs(kernel_fft) ** 2
        rel = power / (power + self.reliability_lambda)
        if rel.shape[1] == 1 and x.shape[1] > 1:
            rel = rel.repeat(1, x.shape[1], 1, 1)
        return rel.clamp(min=self.reliability_floor, max=1.0).to(dtype=x.real.dtype)

    def _regularization(self, x_prev, x0_center):
        if self.lambda_reg <= 0.0:
            return torch.zeros((), device=x_prev.device, dtype=x_prev.dtype)
        correction = x_prev - x0_center
        return self.reg_tv_weight * self._tv(correction)

    def _prior_term(self, x_prev, cutoff, reliability_mask, target):
        target_fft = fftn(target, dim=(-2, -1), norm="ortho")
        lowpass = self._build_lowpass_mask(target, cutoff).to(
            device=x_prev.device,
            dtype=target_fft.real.dtype,
        )
        unreliable = (1.0 - reliability_mask).to(device=x_prev.device, dtype=target_fft.real.dtype)
        if unreliable.shape[-2:] != target_fft.shape[-2:]:
            unreliable = F.interpolate(
                unreliable,
                size=target_fft.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if unreliable.shape[1] == 1 and target_fft.shape[1] > 1:
            unreliable = unreliable.repeat(1, target_fft.shape[1], 1, 1)
        prior_mask = (1.0 - lowpass) * unreliable
        return torch.abs(prior_mask * target_fft).mean()

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, t, **kwargs):
        if kwargs.get("reliability_mask") is not None:
            raise ValueError("External reliability masks are not supported")
        idx = self._step_index(t)
        cutoff = self._cutoff_at(idx, x_prev.device)
        progress = self._progress(idx)
        x0_center = x_0_hat.detach()
        Ax = self.operator.forward(x_0_hat, **kwargs)
        residual = Ax - measurement
        R = fftn(residual, dim=(-2, -1), norm="ortho")
        R_lp = self.cond_model(residual, cutoff)
        reliability = self._reliability_mask(R)
        R_hp = (R - R_lp) * reliability
        lp_term = (torch.abs(R_lp) ** 2).mean()
        hp_term = (torch.abs(R_hp) ** 2).mean()
        meas_space_term = torch.zeros((), device=x_prev.device, dtype=x_prev.dtype)
        prior_term = self._prior_term(x_prev, cutoff, reliability, x_0_hat)
        reg_term = self._regularization(x_prev, x0_center)
        gate_lp = self._gate(progress, self.lp_t0, self.lp_k, x_prev.device, x_prev.dtype)
        gate_hp = self._gate(progress, self.hp_t0, self.hp_k, x_prev.device, x_prev.dtype)
        gate_prior = self._gate(progress, self.prior_t0, self.prior_k, x_prev.device, x_prev.dtype)
        gate_reg = self._gate(progress, self.reg_t0, self.reg_k, x_prev.device, x_prev.dtype)
        gate_meas = torch.zeros((), device=x_prev.device, dtype=x_prev.dtype)
        # Retain the evaluated zero-weight terms and the original addition order.
        loss = (
            self.lambda_lp * gate_lp * lp_term
            + self.lambda_hp * gate_hp * hp_term
            + 0.0 * gate_meas * meas_space_term
            + self.lambda_prior * gate_prior * prior_term
            + self.lambda_reg * gate_reg * reg_term
        )
        grad = torch.autograd.grad(outputs=loss, inputs=x_prev, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(x_prev)
        grad_flat = grad.view(grad.shape[0], -1)
        grad_norm = torch.linalg.norm(grad_flat, dim=1, keepdim=True).view(-1, 1, 1, 1)
        grad = grad / (grad_norm + self.eps)
        step = self._step_size_at(idx, x_prev.device, x_prev.dtype)
        x_next = x_t - step * grad
        return x_next.to(dtype=x_prev.dtype), loss
