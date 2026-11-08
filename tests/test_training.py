import unittest

import torch
import torch.nn.functional as F

import run_experiments_new as runexp


class TestFocalLoss(unittest.TestCase):
    def test_gamma_zero_matches_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 8, 8)
        target = torch.randint(0, 3, (2, 8, 8))

        focal = runexp.FocalLoss(num_classes=3, gamma=0.0, weight=None)
        got = focal(logits, target)
        want = F.cross_entropy(logits, target, reduction="mean")

        self.assertTrue(torch.allclose(got, want, atol=1e-6))

    def test_combined_loss_returns_scalar(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 8, 8)
        target = torch.randint(0, 3, (2, 8, 8))

        loss = runexp.CombinedLoss(3)(logits, target)

        self.assertEqual(loss.dim(), 0)


if __name__ == "__main__":
    unittest.main()
