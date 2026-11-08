import unittest

import numpy as np

import run_experiments_new as runexp


class TestEvalMetricsSelectiveAccuracy(unittest.TestCase):
    def test_selective_accuracy_and_auroc(self):
        # 4 张肿瘤阳性切片：低熵两张 Dice=1.0，高熵两张 Dice=0.0。
        pred = np.array([[[2]], [[2]], [[1]], [[1]]], dtype=np.int64)
        target = np.array([[[2]], [[2]], [[2]], [[2]]], dtype=np.int64)
        entropy = np.array([[[0.0]], [[0.1]], [[0.9]], [[1.0]]], dtype=np.float32)

        metrics, _, _, _ = runexp.eval_metrics(pred, target, entropy)

        self.assertAlmostEqual(metrics["selective_accuracy_all"], 0.5, places=6)
        self.assertAlmostEqual(metrics["selective_accuracy_50"], 1.0, places=6)
        self.assertAlmostEqual(metrics["selective_accuracy_70"], 0.6667, places=4)
        self.assertAlmostEqual(metrics["auroc_entropy_wrong"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
