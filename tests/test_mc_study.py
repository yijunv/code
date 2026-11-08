import math
import unittest

import torch

import mc_sampling_study as study


class TestMCAggregate(unittest.TestCase):
    def test_mean_pred_and_entropy(self):
        # 两个样本，3 类，1x1 像素
        probs = torch.tensor(
            [
                [[[0.7]], [[0.2]], [[0.1]]],
                [[[0.1]], [[0.0]], [[0.9]]],
            ],
            dtype=torch.float32,
        )  # (2, 3, 1, 1)

        mean_probs, mean_pred, entropy = study.mc_aggregate(probs)

        # 平均概率 = [0.4, 0.1, 0.5]，argmax = 类别 2
        self.assertEqual(int(mean_pred.item()), 2)
        expected = -sum(p * math.log(p + 1e-8) for p in [0.4, 0.1, 0.5])
        self.assertAlmostEqual(float(entropy.item()), expected, places=4)


if __name__ == "__main__":
    unittest.main()
