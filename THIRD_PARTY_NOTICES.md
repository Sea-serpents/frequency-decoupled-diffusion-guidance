# Third-Party Notices

This release ships the project guidance and wrappers, not an upstream tree,
patch, model, dataset or image. The project's [MIT License](LICENSE) covers
project-owned code only. External downloads retain their own applicable terms
and notices.

| Component | Attribution and boundary |
| --- | --- |
| FGPS | [Thaker et al., Frequency Guided Posterior Sampling, ICCV 2025](https://github.com/darshanthaker/FGPS). The externally obtained FGPS provides the conditioning interface, Fourier filtering, UNet integration and DDPM/DDIM. The in-process adapter changes only the cutoff representation before calling its interface. No general redistribution permission for the pinned tree was verified. |
| DPS | [Chung et al., Diffusion Posterior Sampling, ICLR 2023](https://github.com/DPS2022/diffusion-posterior-sampling). FGPS builds on DPS. The FFHQ checkpoint is linked through DPS, not mirrored here. No general redistribution permission for DPS source or the FFHQ checkpoint is asserted. |
| Guided Diffusion | [Dhariwal and Nichol, Diffusion Models Beat GANs on Image Synthesis](https://github.com/openai/guided-diffusion). Its original [source license](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/LICENSE) is MIT; it does not blanket-license later FGPS/DPS changes or weights. |
| motionblur and inherited utilities | [motionblur](https://github.com/LeviBorodenko/motionblur) is an external pinned import of FGPS and is not used to generate the Gaussian kernel. Old [Resizer](https://github.com/assafshocher/Resizer) and [fastMRI](https://github.com/facebookresearch/fastMRI)-derived FFT utilities remain in the external tree. No additional redistribution rights are asserted. |
| Installed Python libraries | PyTorch, torchvision, NumPy, SciPy, Pillow, PyYAML, Matplotlib, tqdm, packaging, and optional [scikit-image](https://scikit-image.org/) are separately installed and retain their own terms. |
