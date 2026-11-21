import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fuse
import reconstruct


class GaussianReleaseTests(unittest.TestCase):
    def test_profile(self):
        p = reconstruct.read_profile()
        self.assertEqual(p["reconstruction"]["branches"], ["ddpm", "ddim"])
        for branch in ("ddpm", "ddim"):
            s = reconstruct.sampler_args(p, branch)
            self.assertEqual((s["sampler"], s["steps"], s["timestep_respacing"]),
                             (branch, 1000, 1000))
        self.assertEqual(p["reconstruction"]["ddim_eta"], 0.0)
        g = reconstruct.guidance_args(p)
        self.assertEqual((g["step_end"], g["step_start"]), (0.60, 2.5))
        self.assertEqual((g["lambda_lp"], g["lambda_prior"], g["lambda_reg"]), (1.30, .003, .005))
        with self.assertRaises(ValueError):
            reconstruct.sampler_args(p, "cgj")

    def test_fusion(self):
        ddpm = np.array([[[0, 1, 255], [2, 3, 255]]], dtype=np.uint8)
        ddim = np.array([[[1, 2, 255], [3, 4, 0]]], dtype=np.uint8)
        expected = np.array([[[0, 2, 255], [2, 4, 128]]], dtype=np.uint8)
        weight = reconstruct.read_profile()["reconstruction"]["ddpm_output_weight"]
        np.testing.assert_array_equal(fuse.fuse_rgb(ddpm, ddim, weight), expected)
        with self.assertRaises(ValueError):
            fuse.fuse_rgb(ddpm.astype(float), ddim, weight)

    @unittest.skipUnless(os.environ.get("FDG_TEST_RUNTIME"), "Requires verified external checkouts")
    def test_external_cpu_guidance(self):
        import torch

        torch.set_num_threads(2)
        dev = torch.device("cpu")
        reconstruct.activate(Path(os.environ["FDG_TEST_RUNTIME"]))
        p = reconstruct.read_profile()
        for branch in ("ddpm", "ddim"):
            op, guidance, sampler = reconstruct.prepare(p, branch, dev)
            self.assertEqual(sampler.num_timesteps, 1000)
            self.assertEqual(type(sampler).__name__.lower(), branch)
            z = torch.randn(1, 3, 80, 80, generator=torch.Generator().manual_seed(81)).requires_grad_()
            prediction = z * 0.2
            proposal = torch.zeros_like(z)
            goal = torch.full_like(z, .1)
            grads = []
            z.register_hook(lambda g: grads.append(g))
            corrected, loss = guidance.conditioning(z, proposal, prediction, goal, torch.tensor([999]))
            step = guidance._step_size_at(999, dev, z.dtype)
            self.assertAlmostEqual(float(step), .60, places=6)
            self.assertEqual(len(grads), 1)
            self.assertTrue(torch.isfinite(loss))
            torch.testing.assert_close(corrected, -step * grads[0] / (grads[0].norm() + guidance.eps))
            self.assertEqual(tuple(op.get_kernel().shape), (1, 1, 61, 61))


if __name__ == "__main__":
    unittest.main()
