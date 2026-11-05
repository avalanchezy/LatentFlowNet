# LatentFlowNet implementation

The public model and all user-facing entry points use the name **LatentFlowNet**. The release follows the journal method and its final overview figure.

## Figure-to-code correspondence

| Figure component | Implementation |
| --- | --- |
| Endpoint-trained 3D KL-VAE | `train_vae.py`, months 0 and 24 for optimization and validation |
| Gaussian posterior during VAE training | `KLVAE.reparameterize` |
| Frozen encoder and decoder during trajectory learning | Latents encoded once with `KLVAE.get_latent`; decoder loaded in evaluation mode with gradients disabled |
| Baseline latent state | Posterior mean of baseline PET |
| Conv3D → four residual blocks → Conv3D | `LatentFlowNet.drift` |
| Clinical-time input | Follow-up month divided by target month |
| Optional baseline condition | Numeric subject covariates plus availability mask, embedded through an MLP |
| Residual update | `z_next = z + (1 / N) * drift(tau, z, condition)` |
| Fixed rollout | `N = 20`; M6/M12/M18 correspond to steps 5/10/15 for a 24-month horizon |
| Available intermediate visit | Predicted state receives masked anchor supervision |
| Missing visit | Flow state still exists; no observed latent or anchor loss is created |
| Baseline-only inference | `predict.py`: input PET → posterior mean → flow → decoder |

Off-grid anchor times use interpolation within the corresponding Euler step. No intermediate image is inserted into the model state at inference.

## Network and training settings

The VAE uses 4 latent channels, base width 32, channel multipliers `(1, 2, 4, 8)` and two residual blocks per level. Only the first two encoder levels downsample, so the spatial reduction is fourfold. The VAE training objective in the released implementation is voxelwise **L1 reconstruction + 1e-6 × KL**; posterior sampling is used during training and posterior means during encoding.

The flow has 64 hidden channels, a 128-dimensional time embedding, four residual blocks and dropout 0.1. When covariates are supplied, their embedding dimension is 32. The field predicts the derivative of the current latent state. Training differentiates through the complete Euler rollout and applies endpoint and intermediate-trajectory supervision.

| Setting | VAE | LatentFlowNet |
| --- | --- | --- |
| Optimizer | Adam, betas `(0.5, 0.9)` | AdamW, betas `(0.9, 0.999)`, weight decay `1e-5` |
| Initial learning rate | `1e-4` | `1e-4` |
| Schedule | Cosine to `1e-6` | Cosine to `1e-6` |
| Epoch limit | 1000 | 500 |
| Batch size | 1 | 2 |
| Early stopping patience | 100 | 100 |
| Seed | 0 | 0 |

## Loss implementation

The released code preserves the original training computation and checkpoint parameter names. The scalar loss is:

```text
1.0 * endpoint_residual_MSE
+ 0.5 * endpoint_state_MSE
+ 2.0 * intermediate_anchor_loss
+ 0.05 * endpoint_contrastive_loss
+ 0.01 * velocity_smoothness
+ 0.2 * radial_monotonicity
```

The endpoint residual error and endpoint state error are algebraically equal. The implementation retains both weighted terms, giving an effective endpoint MSE coefficient of 1.5. This states the executable loss explicitly alongside the journal's compact residual-loss notation.

Each intermediate anchor combines state MSE with 0.5 times residual MSE. Anchors are averaged using normalized visit time and their observed fraction in the batch. Missing observations are masked before reduction. This is the exact time-weighted reduction used by the released implementation.

The endpoint contrastive loss compares flattened cosine similarities against the true target (positive) and baseline (negative), with temperature 0.1. Velocity smoothness averages squared differences between successive drift predictions over the `N - 1` transitions. Monotonicity penalizes decreases in the latent distance from baseline over successive states. These explicit reductions preserve the training code; they also resolve shorthand normalization in the manuscript equations.

## Evaluation

The evaluator distinguishes metrics on latents, images decoded from target latents, and supplied raw target PET. Raw image PSNR and SSIM use the target image's intensity range; MAE is the mean absolute voxel error. The loader and predictor preserve input intensities. Spatial preprocessing and any return to a different original acquisition grid belong to the study preprocessing pipeline; the evaluator aligns compatible supplied volume shapes by central cropping rather than performing anatomical registration.

The repository contains source code and the method figure. Patient images, subject tables, cohort splits, trained weights and historical result directories are distributed separately. Verification of this release concerns code execution and the figure-to-code mapping; study scores require the corresponding data, preprocessing, split and trained weights.
