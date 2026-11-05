"""Device and reference-image defaults for LatentFlowNet."""

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Reference affine for encoded latent NIfTI files.
# predict.py uses its baseline input image's affine directly.
path = "./data/sample.nii.gz"
