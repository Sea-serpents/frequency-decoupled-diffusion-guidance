"""Image, seed and spatial-operator helpers for single-image reconstruction."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils as tvu
from PIL import Image


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
        tensor = TF.to_tensor(image).unsqueeze(0)
    return tensor.mul(2.0).sub(1.0)


def save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tvu.save_image(((tensor.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0), path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_x_t(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((1, 3, 256, 256), generator=generator)


class SpatialConvOperator:
    """Zero-boundary convolution with an exact transpose."""

    def __init__(self, kernel: torch.Tensor, device: torch.device):
        kernel = kernel.to(device=device, dtype=torch.float32)
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError(f"Expected a square 2D kernel, got {tuple(kernel.shape)}")
        self.kernel = kernel
        self.device = device
        self.padding = kernel.shape[0] // 2
        self.weight = kernel.view(1, 1, *kernel.shape).repeat(3, 1, 1, 1)

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.conv2d(data, self.weight, padding=self.padding, groups=3)

    def transpose(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.conv_transpose2d(data, self.weight, padding=self.padding, groups=3)

    def get_kernel(self) -> torch.Tensor:
        return self.kernel.view(1, 1, *self.kernel.shape)
