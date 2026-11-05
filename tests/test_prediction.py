"""Baseline-only NIfTI prediction and checkpoint loading checks."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch


torch.set_num_threads(1)
PREDICT_AVAILABLE = importlib.util.find_spec('predict') is not None


class PredictionTests(unittest.TestCase):
    def test_prediction_entry_is_available(self):
        self.assertIsNotNone(
            importlib.util.find_spec('predict'),
            'A baseline-only prediction entry must be available.',
        )

    @unittest.skipUnless(PREDICT_AVAILABLE, 'Prediction entry pending')
    def test_real_checkpoint_prediction_uses_posterior_mean_and_input_geometry(self):
        from latentflownet import LatentFlowNet
        from model_vae import KLVAE
        from predict import load_models, predict_volume

        vae = KLVAE(latent_ch=1, base_ch=4, ch_mult=(1, 1, 2, 2),
                    num_res_blocks=1)
        model = LatentFlowNet(latent_ch=1, hidden_ch=4, time_dim=8,
                              num_blocks=1, dropout=0.0)
        # An analytic constant flow: integration advances each latent by 0.25.
        with torch.no_grad():
            model.drift.output_conv.weight.zero_()
            model.drift.output_conv.bias.fill_(0.25)

        data = np.linspace(0.0, 1.0, 8 ** 3, dtype=np.float32).reshape(8, 8, 8)
        affine = np.array([[2.0, 0.0, 0.0, -12.0],
                           [0.0, 3.0, 0.0, 7.0],
                           [0.0, 0.0, 4.0, 5.0],
                           [0.0, 0.0, 0.0, 1.0]])
        image = nib.Nifti1Image(data, affine)
        image.header['descrip'] = b'Preprocessed baseline M0'
        image.header.set_xyzt_units('mm', 'sec')
        image.set_qform(affine, code=1)
        image.set_sform(affine, code=2)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            baseline = directory / 'baseline.nii.gz'
            output = directory / 'prediction.nii.gz'
            vae_checkpoint = directory / 'vae.pth'
            checkpoint = directory / 'flow.pth'
            nib.save(image, baseline)
            torch.save({'state_dict': vae.state_dict()}, vae_checkpoint)
            torch.save({'model_state_dict': model.state_dict()}, checkpoint)

            loaded_vae, loaded_model = load_models(
                vae_checkpoint, checkpoint, device='cpu', n_steps=3,
            )
            with torch.no_grad():
                x = torch.from_numpy(data)[None, None]
                mu, _ = vae.eval().encode(x)
                expected = vae.decode(mu + 0.25)[0, 0].numpy()

            predict_volume(baseline, output, loaded_vae, loaded_model, device='cpu')

            result = nib.load(output)
            np.testing.assert_allclose(result.get_fdata(), expected, atol=1e-5)
            np.testing.assert_array_equal(result.affine, nib.load(baseline).affine)
            self.assertEqual(result.header['descrip'], image.header['descrip'])
            self.assertEqual(result.header.get_xyzt_units(), ('mm', 'sec'))
            self.assertEqual(int(result.header['qform_code']), 1)
            self.assertEqual(int(result.header['sform_code']), 2)
            self.assertEqual(result.get_data_dtype(), np.dtype('float32'))
            self.assertFalse(loaded_vae.training)
            self.assertFalse(loaded_model.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None
                                for p in loaded_vae.parameters()))
            self.assertEqual({p.name for p in directory.iterdir()},
                             {'baseline.nii.gz', 'prediction.nii.gz', 'vae.pth', 'flow.pth'})

    @unittest.skipUnless(PREDICT_AVAILABLE, 'Prediction entry pending')
    def test_missing_vae_checkpoint_fails_instead_of_using_random_weights(self):
        from predict import load_models

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_models(Path(directory) / 'missing_vae.pth',
                            Path(directory) / 'flow.pth', device='cpu')

    @unittest.skipUnless(PREDICT_AVAILABLE, 'Prediction entry pending')
    def test_cli_reads_only_baseline_and_105_baseline_metadata_features(self):
        from latentflownet import LatentFlowNet
        from model_vae import KLVAE
        from predict import load_models

        vae = KLVAE(latent_ch=1, base_ch=4, ch_mult=(1, 2, 2, 2),
                    num_res_blocks=1).eval()
        model = LatentFlowNet(latent_ch=1, hidden_ch=4, time_dim=8,
                              num_blocks=1, obs_dim=105, cond_dim=32)
        data = np.ones((8, 8, 8), dtype=np.float32)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        with torch.no_grad():
            # Zero-initialized flow output leaves the posterior mean unchanged.
            expected = vae.decode(vae.get_latent(torch.from_numpy(data)[None, None]))
            expected = expected[0, 0].numpy()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            baseline = directory / 'm0.nii.gz'
            output = directory / 'm24_prediction.nii.gz'
            vae_checkpoint = directory / 'vae.pth'
            checkpoint = directory / 'flow.pth'
            table = directory / 'baseline_metadata.csv'
            nib.save(nib.Nifti1Image(data, affine), baseline)
            torch.save({'state_dict': vae.state_dict()}, vae_checkpoint)
            torch.save({'model_state_dict': model.state_dict()}, checkpoint)
            columns = ','.join(f'feature_{index:03d}' for index in range(105))
            table.write_text(f'subject_id,{columns}\n001,' + ','.join(['0.5'] * 105) + '\n',
                             encoding='utf-8')
            _, restored_model = load_models(vae_checkpoint, checkpoint, device='cpu')
            self.assertEqual(restored_model.obs_dim, 105)
            self.assertEqual(restored_model.cond_dim, 32)
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'predict.py'),
                       '--input', str(baseline), '--output', str(output),
                       '--vae_checkpoint', str(vae_checkpoint), '--checkpoint', str(checkpoint),
                       '--obs_file', str(table), '--subject_id', '001', '--device', 'cpu',
                       '--n_steps', '2', '--target_time', '24']
            result = subprocess.run(command, capture_output=True, text=True,
                                    env={**os.environ, 'OMP_NUM_THREADS': '1',
                                         'MKL_NUM_THREADS': '1'})
            self.assertEqual(result.returncode, 0, result.stderr)
            prediction = nib.load(output)
            np.testing.assert_allclose(prediction.get_fdata(), expected, atol=1e-5)
            np.testing.assert_array_equal(prediction.affine, affine)
            self.assertEqual(len(list(directory.iterdir())), 5)

    @unittest.skipUnless(PREDICT_AVAILABLE, 'Prediction entry pending')
    def test_conditional_metadata_uses_csv_column_order_and_whole_row_mask(self):
        from predict import load_metadata

        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / 'baseline_metadata.csv'
            table.write_text('subject_id,feature_b,feature_a\n001,2.5,7.0\n',
                             encoding='utf-8')
            obs, mask = load_metadata(table, '001', obs_dim=2, device='cpu')
            torch.testing.assert_close(obs, torch.tensor([[2.5, 7.0]]))
            torch.testing.assert_close(mask, torch.ones(1, 1))
            missing_obs, missing_mask = load_metadata(table, 'absent', obs_dim=2,
                                                       device='cpu')
            torch.testing.assert_close(missing_obs, torch.zeros(1, 2))
            torch.testing.assert_close(missing_mask, torch.zeros(1, 1))
            with self.assertRaisesRegex(ValueError, 'metadata|obs_file'):
                load_metadata(None, '001', obs_dim=2, device='cpu')
            with self.assertRaisesRegex(ValueError, 'subject_id'):
                load_metadata(table, None, obs_dim=2, device='cpu')
            with self.assertRaisesRegex(ValueError, '2|3'):
                load_metadata(table, '001', obs_dim=3, device='cpu')


if __name__ == '__main__':
    unittest.main()
