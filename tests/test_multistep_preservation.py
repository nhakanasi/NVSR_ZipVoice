import unittest

import torch
from torch import nn

from zipvoice.models.zipvoice_adapter import ZipVoiceAdapter


class Backbone(nn.Module):
    def __init__(self, coupling):
        super().__init__()
        self.coupling = nn.Parameter(torch.tensor(float(coupling)), requires_grad=False)

    def forward(self, x, t, padding_mask=None):
        # The stock low-band velocity responds to the preceding high-band state.
        return torch.stack((self.coupling * x[..., 1], x[..., 1] * 0), dim=-1)


class Adapter(nn.Module):
    multistep_preservation = ZipVoiceAdapter.multistep_preservation

    def __init__(self, coupling=1.0, amplitude=0.4):
        super().__init__()
        self.fm_decoder = Backbone(coupling)
        self.amplitude = nn.Parameter(torch.tensor(amplitude))

    def forward_fm_decoder(self, t, xt, text_condition, speech_condition,
                           padding_mask, band_mask, **kwargs):
        base = self.fm_decoder(torch.cat((xt, text_condition, speech_condition), -1), t)
        return base + band_mask * self.amplitude


def evaluate(model):
    x = torch.zeros(1, 4, 2)
    return model.multistep_preservation(
        x, torch.full((1, 1, 1), 0.25), x, x,
        torch.zeros(1, 4, dtype=torch.bool), torch.tensor([[[0., 1.]]]),
        torch.tensor([4000.]), torch.zeros(1, 4, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool), steps=3,
    )


class PreservationTests(unittest.TestCase):
    def test_indirect_gradient_through_frozen_backbone(self):
        model = Adapter().train()
        loss, drift = evaluate(model)
        loss.backward()
        self.assertGreater(loss.item(), 0)
        self.assertGreater(drift.item(), 0)
        self.assertGreater(model.amplitude.grad.item(), 0)
        self.assertIsNone(model.fm_decoder.coupling.grad)
        self.assertTrue(model.training)
        self.assertTrue(model.fm_decoder.training)

    def test_zero_correction_preserves_stock(self):
        loss, drift = evaluate(Adapter(amplitude=0.0))
        self.assertEqual(loss.item(), 0)
        self.assertEqual(drift.item(), 0)

    def test_no_cross_band_response_means_no_penalty(self):
        loss, _ = evaluate(Adapter(coupling=0.0))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(loss.grad_fn is None, False)


if __name__ == '__main__':
    unittest.main()
