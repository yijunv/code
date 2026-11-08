import unittest

import numpy as np

import stats_validation as sv


class TestEntropyBins(unittest.TestCase):
    def test_error_rate_by_entropy_bin(self):
        # 5 个低熵切片 Dice=0.9（错误率应为 0），5 个高熵切片 Dice=0.0（错误率应为 1）
        ent = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        dice = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])

        centers, rates, counts = sv.error_rate_by_entropy_bin(ent, dice, n_bins=2)

        self.assertEqual(len(centers), 2)
        self.assertAlmostEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[1], 1.0)
        self.assertEqual(counts[0], 5)
        self.assertEqual(counts[1], 5)


if __name__ == "__main__":
    unittest.main()
