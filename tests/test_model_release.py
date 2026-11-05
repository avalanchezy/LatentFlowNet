"""Numerical checks for LatentFlowNet prediction and sparse supervision."""

import importlib.util
import unittest

import torch
from torch import nn


torch.set_num_threads(1)


class ConstantDrift(nn.Module):
    """An analytic flow field for checking the numerical trajectory."""

    def __init__(self, velocity):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(float(velocity)))

    def forward(self, t, z, cond_emb=None, label=None):
        return torch.ones_like(z) * self.velocity


def make_model():
    from latentflownet import LatentFlowNet

    return LatentFlowNet(
        latent_ch=1, hidden_ch=4, time_dim=8, num_blocks=1,
        n_steps=4, dropout=0.0, lambda_c=0.0, lambda_s=0.0,
        lambda_m=0.0, inter_weight=2.0, delta_mid_weight=0.5,
    ).eval()


class ModelReleaseTests(unittest.TestCase):
    def test_public_model_is_available(self):
        self.assertIsNotNone(
            importlib.util.find_spec('latentflownet'),
            'The LatentFlowNet public model module must be available.',
        )

    @unittest.skipUnless(importlib.util.find_spec('latentflownet'), 'Public model pending')
    def test_prediction_integrates_constant_velocity(self):
        model = make_model()
        model.drift = ConstantDrift(3.0)
        baseline = torch.full((2, 1, 2, 2, 2), 2.0)
        prediction = model.predict(baseline)
        torch.testing.assert_close(prediction, torch.full_like(baseline, 5.0))

    @unittest.skipUnless(importlib.util.find_spec('latentflownet'), 'Public model pending')
    def test_sparse_off_grid_anchors_use_only_observed_subjects(self):
        model = make_model()
        model.drift = ConstantDrift(24.0)
        baseline = torch.zeros(2, 1, 2, 2, 2)
        endpoint = torch.full_like(baseline, 24.0)
        month7 = torch.stack([torch.full_like(baseline[0], 9.0),
                              torch.full_like(baseline[0], 1000.0)])
        month12 = torch.stack([torch.full_like(baseline[0], -1000.0),
                               torch.full_like(baseline[0], 15.0)])
        anchors = {
            7: {'z': month7, 'mask': torch.tensor([1.0, 0.0])},
            12: {'z': month12, 'mask': torch.tensor([0.0, 1.0])},
        }

        loss, metrics = model.compute_loss(baseline, endpoint, observations=anchors)

        # z(7)=7 and z(12)=12. Squared errors are 4 and 9;
        # the weighted state+delta penalty is 1.5*(7*4+12*9)/(7+12).
        self.assertAlmostEqual(metrics['inter'], 204.0 / 19.0, places=5)
        self.assertAlmostEqual(loss.item(), 408.0 / 19.0, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(model.drift.velocity.grad))
        self.assertLess(model.drift.velocity.grad.item(), 0.0)

        month7[1].fill_(-2000.0)
        month12[0].fill_(2000.0)
        masked_loss, _ = model.compute_loss(baseline, endpoint, observations=anchors)
        torch.testing.assert_close(masked_loss, loss.detach())


if __name__ == '__main__':
    unittest.main()
