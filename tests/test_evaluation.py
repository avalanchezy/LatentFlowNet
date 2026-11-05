"""Check that reported comparisons use the same target representation."""

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import train_latentflownet as training
from dataset_longitudinal import LongitudinalPETDataset
from predict import load_metadata
import nibabel as nib
import numpy as np


class EvaluationSummaryTests(unittest.TestCase):
    def test_zero_padded_metadata_ids_match_training_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / '001'
            subject.mkdir()
            for month in (0, 24):
                nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 4)), np.eye(4)),
                         subject / f'001_ses-M{month:02d}.nii.gz')
            table = root / 'baseline.csv'
            table.write_text('subject_id,age\n001,70\n', encoding='utf-8')
            dataset = LongitudinalPETDataset(directory, obs_file=str(table))
            self.assertIn('001', dataset.obs_map)
            obs_map, dimension = training.load_obs_table(str(table))
            self.assertEqual(dimension, 1)
            obs, mask = load_metadata(str(table), '001', dimension)
            np.testing.assert_array_equal(dataset.obs_map['001'], obs_map['001'])
            np.testing.assert_array_equal(obs_map['001'], obs.numpy()[0])
            self.assertEqual(mask.item(), 1.0)

    def test_summary_compares_prediction_and_identity_against_the_same_target(self):
        self.assertTrue(hasattr(training, 'write_test_results'))
        row = {
            'subject': 'example', 'psnr': 30.0, 'ssim': 0.95, 'mae': 0.02,
            'psnr_raw': 27.0, 'ssim_raw': 0.91, 'mae_raw': 0.03,
            'psnr_identity': 24.0, 'ssim_identity': 0.88,
            'psnr_identity_raw': 24.0, 'ssim_identity_raw': 0.88,
            'mae_identity_raw': 0.04,
            'psnr_identity_decoded': 25.0, 'ssim_identity_decoded': 0.90,
            'delta_pred': 1.0, 'delta_gt': 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                training.write_test_results([row], directory, 'klvae')
            text = output.getvalue()
            self.assertIn('Raw improvement over identity: PSNR=+3.00 dB SSIM=+0.0300', text)
            self.assertIn('Decoded target: PSNR=30.00', text)
            with (Path(directory) / 'test_results.csv').open(newline='') as stream:
                saved = next(csv.DictReader(stream))
            self.assertEqual(float(saved['psnr_raw']), 27.0)

            for key in ('psnr_raw', 'ssim_raw', 'mae_raw', 'psnr_identity_raw',
                        'ssim_identity_raw', 'mae_identity_raw'):
                row[key] = None
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                training.write_test_results([row], directory, 'klvae')
            self.assertIn('Decoded improvement over identity: PSNR=+5.00 dB SSIM=+0.0500',
                          output.getvalue())
            self.assertNotIn('Raw improvement', output.getvalue())


if __name__ == '__main__':
    unittest.main()
