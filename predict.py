"""Predict a future PET volume from one preprocessed baseline NIfTI."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from latentflownet import LatentFlowNet
from model_vae import KLVAE


def _state_dict(checkpoint):
    for key in ('model_state_dict', 'state_dict', 'state'):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_models(vae_checkpoint, checkpoint, device='cpu', n_steps=20,
                target_time=24):
    """Restore a KL-VAE and LatentFlowNet, inferring dimensions from weights."""
    vae_state = _state_dict(torch.load(vae_checkpoint, map_location='cpu',
                                      weights_only=False))
    flow_state = _state_dict(torch.load(checkpoint, map_location='cpu',
                                       weights_only=False))
    base_ch = vae_state['encoder.input_conv.weight'].shape[0]
    levels = sorted({int(key.split('.')[2]) for key in vae_state
                     if key.startswith('encoder.down_blocks.')})
    ch_mult = tuple(vae_state[f'encoder.down_blocks.{i}.0.block.2.weight'].shape[0]
                    // base_ch for i in levels)
    num_res_blocks = len({int(key.split('.')[3]) for key in vae_state
                          if key.startswith('encoder.down_blocks.0.')})
    latent_ch = vae_state['post_quant_conv.weight'].shape[0]
    vae = KLVAE(latent_ch=latent_ch, base_ch=base_ch, ch_mult=ch_mult,
                num_res_blocks=num_res_blocks)
    vae.load_state_dict(vae_state)

    obs_dim, cond_dim = 0, 0
    if 'metadata_conditioner.net.0.weight' in flow_state:
        obs_dim = flow_state['metadata_conditioner.net.0.weight'].shape[1] - 1
        cond_dim = flow_state['metadata_conditioner.net.3.weight'].shape[0]
    num_classes = (flow_state['drift.label_embed.weight'].shape[0]
                   if 'drift.label_embed.weight' in flow_state else 0)
    model = LatentFlowNet(
        latent_ch=flow_state['drift.input_conv.weight'].shape[1],
        hidden_ch=flow_state['drift.input_conv.weight'].shape[0],
        time_dim=flow_state['drift.time_mlp.0.weight'].shape[1],
        num_blocks=len({int(key.split('.')[2]) for key in flow_state
                        if key.startswith('drift.blocks.')}),
        num_classes=num_classes, obs_dim=obs_dim, cond_dim=cond_dim,
        n_steps=n_steps, target_time=target_time,
    )
    model.load_state_dict(flow_state)
    vae.to(device).eval().requires_grad_(False)
    model.to(device).eval().requires_grad_(False)
    return vae, model


def load_metadata(obs_file, subject_id, obs_dim, device='cpu'):
    """Read baseline covariates in training CSV column order and row mask."""
    if obs_dim == 0:
        return None, None
    if obs_file is None:
        raise ValueError('This checkpoint requires baseline metadata: provide --obs_file.')
    if subject_id is None:
        raise ValueError('Provide --subject_id to select baseline metadata.')
    table = pd.read_csv(obs_file, dtype={'subject_id': str})
    if 'subject_id' not in table.columns:
        raise ValueError('obs_file must contain a subject_id column.')
    columns = [column for column in table.columns if column != 'subject_id']
    if len(columns) != obs_dim:
        raise ValueError(f'Checkpoint expects {obs_dim} metadata columns; found {len(columns)}.')
    rows = table.loc[table['subject_id'].str.strip() == str(subject_id).strip(), columns]
    if len(rows) == 0:
        # Match the training dataset's whole-row availability convention.
        return (torch.zeros(1, obs_dim, device=device),
                torch.zeros(1, 1, device=device))
    if len(rows) != 1:
        raise ValueError(f'Metadata contains multiple rows for subject_id {subject_id}.')
    values = rows.to_numpy(dtype=np.float32)
    return (torch.from_numpy(values).to(device),
            torch.ones(1, 1, device=device))


@torch.inference_mode()
def predict_volume(input_path, output_path, vae, model, device='cpu',
                   obs=None, obs_mask=None):
    """Encode M0 posterior mean, integrate the latent flow, and decode once."""
    if model.obs_dim > 0 and obs is None:
        raise ValueError('This checkpoint requires baseline metadata from load_metadata.')
    source = nib.load(input_path)
    data = source.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f'Expected one 3D baseline volume, got shape {data.shape}.')
    vae.to(device).eval().requires_grad_(False)
    model.to(device).eval().requires_grad_(False)
    baseline = torch.from_numpy(data.copy())[None, None].to(device)
    z_0 = vae.get_latent(baseline)
    z_target = model.predict(z_0, obs=obs, obs_mask=obs_mask)
    prediction = vae.decode(z_target)[0, 0].cpu().numpy().astype(np.float32)
    if prediction.shape != data.shape:
        raise ValueError(f'Decoded shape {prediction.shape} differs from input {data.shape}.')
    header = source.header.copy()
    header.set_data_dtype(np.float32)
    result = source.__class__(prediction, source.affine, header=header)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Preprocessed baseline M0 NIfTI.')
    parser.add_argument('--output', required=True, help='Predicted future NIfTI.')
    parser.add_argument('--vae_checkpoint', required=True, help='Trained KL-VAE checkpoint.')
    parser.add_argument('--checkpoint', required=True, help='Trained LatentFlowNet checkpoint.')
    parser.add_argument('--obs_file', help='Baseline metadata CSV in training column order.')
    parser.add_argument('--subject_id', help='Subject row to select from --obs_file.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_steps', type=int, default=20, help='Fixed Euler integration steps.')
    parser.add_argument('--target_time', type=int, default=24, help='Trained endpoint month.')
    args = parser.parse_args()
    if args.n_steps <= 0:
        parser.error('--n_steps must be positive.')
    vae, model = load_models(args.vae_checkpoint, args.checkpoint, args.device,
                             args.n_steps, args.target_time)
    obs, obs_mask = load_metadata(args.obs_file, args.subject_id, model.obs_dim,
                                  args.device)
    output = predict_volume(args.input, args.output, vae, model, args.device,
                             obs=obs, obs_mask=obs_mask)
    print(f'Saved prediction to {output}')


if __name__ == '__main__':
    main()
