"""Use unmodified, user-supplied checkouts and adapt only the cutoff argument."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

PINS = {
    "fgps": "499e21d4488d2c15b0177546907db288dbb9c470",
    "motionblur": "578137887b2067c532bd4e422ff5afdcabfccebf",
}


def verify(root):
    for name, pin in PINS.items():
        path = root / name
        def git(*args):
            return subprocess.check_output(["git", "--no-optional-locks", "-C", str(path), *args],
                                           text=True, encoding="utf-8").strip()

        if Path(git("rev-parse", "--show-toplevel")).resolve() != path.resolve():
            raise ValueError(f"{name}: expected an independent checkout")
        if git("rev-parse", "HEAD") != pin or git("status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"{name}: expected clean checkout at {pin}")
        # Check actual working files, including ignored modules that might shadow imports.
        entries = [row.split("\t", 1) for row in git("ls-tree", "-r", pin).splitlines()]
        tracked = {rel: spec.split()[2] for spec, rel in entries if spec.split()[1] == "blob"}
        if len(tracked) != len(entries):
            raise ValueError(f"{name}: unsupported external tree")
        actual = {p.relative_to(path).as_posix() for p in path.rglob("*")
                  if p.is_file() and ".git" not in p.relative_to(path).parts}
        if actual != set(tracked):
            raise ValueError(f"{name}: unexpected runtime files (use a fresh checkout)")
        for rel, blob in tracked.items():
            file = path / rel
            raw = file.read_bytes()
            if file.is_symlink() or hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != blob:
                raise ValueError(f"{name}: file bytes differ from pinned Git blob: {rel}")


def activate(root):
    root = root.resolve()
    verify(root)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root / "fgps"), str(root)]


def frequency_filter(device):
    from guided_diffusion.fgps_model import FGPSModel
    import torch

    class ScalarCutoffFilter(FGPSModel):
        def get_mask(self, x_shape, cutoff):
            if torch.is_tensor(cutoff):
                cutoff = float(cutoff.detach().to(dtype=torch.float32).item())
            return super().get_mask(x_shape, cutoff)

        def forward(self, x, cutoff):
            if x.dtype != torch.float32:
                raise TypeError("Frozen FGPS filter integration requires float32")
            return super().forward(x, cutoff)

    return ScalarCutoffFilter(device).eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    verify(parser.parse_args().runtime.resolve())
    print("Clean pinned external runtime verified.")
