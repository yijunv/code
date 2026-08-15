# -*- coding: utf-8 -*-
"""降采样工具单元测试（stdlib unittest，任何 Python 环境都能跑）。

运行方式（在仓库根目录）:
    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import downsample as ds


def fake_cases(n, prefix="case_"):
    return [prefix + "%05d" % i for i in range(n)]


class SplitCasesTest(unittest.TestCase):
    def test_split_counts_4_cases(self):
        cases = fake_cases(4)
        train, val, test = ds.split_cases(cases, 0.5, 0.25, 42)
        self.assertEqual((len(train), len(val), len(test)), (2, 1, 1))

    def test_split_partitions_all_cases(self):
        cases = fake_cases(30)
        train, val, test = ds.split_cases(cases, 0.7, 0.15, 7)
        self.assertEqual(len(train) + len(val) + len(test), len(cases))
        self.assertEqual(set(train) | set(val) | set(test), set(cases))

    def test_split_no_overlap(self):
        cases = fake_cases(30)
        train, val, test = ds.split_cases(cases, 0.7, 0.15, 7)
        self.assertTrue(set(train).isdisjoint(val))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(val).isdisjoint(test))

    def test_split_reproducible(self):
        cases = fake_cases(20)
        self.assertEqual(
            ds.split_cases(cases, 0.7, 0.15, 42),
            ds.split_cases(cases, 0.7, 0.15, 42),
        )


class DownsampleTrainTest(unittest.TestCase):
    def test_fraction_10pct(self):
        train = fake_cases(100)
        sub = ds.downsample_train_cases(train, 0.10, 42)
        self.assertEqual(len(sub), 10)
        self.assertLessEqual(set(sub), set(train))

    def test_fraction_20pct(self):
        train = fake_cases(100)
        sub = ds.downsample_train_cases(train, 0.20, 42)
        self.assertEqual(len(sub), 20)

    def test_reproducible(self):
        train = fake_cases(50)
        self.assertEqual(
            ds.downsample_train_cases(train, 0.20, 42),
            ds.downsample_train_cases(train, 0.20, 42),
        )

    def test_min_one_case_when_tiny(self):
        train = fake_cases(2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sub = ds.downsample_train_cases(train, 0.10, 42)
        self.assertEqual(len(sub), 1)
        self.assertIn("警告", buf.getvalue())

    def test_invalid_fraction_raises(self):
        with self.assertRaises(ValueError):
            ds.downsample_train_cases(fake_cases(10), 0.0, 42)
        with self.assertRaises(ValueError):
            ds.downsample_train_cases(fake_cases(10), 1.5, 42)


class BuildExperimentsTest(unittest.TestCase):
    """用临时目录造 4 个假病例，验证端到端生成 manifest。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, "kits19_small")
        os.makedirs(self.root)
        for i in range(4):
            d = os.path.join(self.root, "case_%05d" % i)
            os.makedirs(d)
            with open(os.path.join(d, "imaging.nii.gz"), "wb") as f:
                f.write(b"x")
            with open(os.path.join(d, "segmentation.nii.gz"), "wb") as f:
                f.write(b"x")
        self.out = os.path.join(self.tmp, "experiments")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_manifests_contract(self):
        manifests = ds.build_experiments(
            self.root, self.out, fractions=(0.10, 0.20),
            seed=42, train_ratio=0.5, val_ratio=0.25,
        )
        self.assertEqual(len(manifests), 2)
        for p in manifests:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            # 契约 C 必备字段（含配置：划分比例与 max_cases）
            for key in ("seed", "data_fraction", "train_ratio", "val_ratio", "max_cases",
                        "train_cases", "val_cases", "test_cases",
                        "n_train_cases", "n_val_cases", "n_test_cases"):
                self.assertIn(key, m)
            # 4 病例 -> 2 训练 / 1 验证 / 1 测试；训练 2 例降采样后至少 1 例
            self.assertEqual(m["n_val_cases"], 1)
            self.assertEqual(m["n_test_cases"], 1)
            self.assertGreaterEqual(m["n_train_cases"], 1)
            self.assertEqual(m["train_ratio"], 0.5)
            self.assertEqual(m["val_ratio"], 0.25)
            self.assertEqual(m["max_cases"], None)
            self.assertEqual(len(m["train_cases"]), m["n_train_cases"])
            self.assertEqual(len(m["val_cases"]), m["n_val_cases"])
            self.assertEqual(len(m["test_cases"]), m["n_test_cases"])
        # 测试集冻结：不同 fraction 的 test 必须完全一致
        t0 = self._load(manifests[0])["test_cases"]
        t1 = self._load(manifests[1])["test_cases"]
        self.assertEqual(t0, t1)

    def _load(self, path):
        return ds.load_manifest(path)

    def test_scan_valid_cases(self):
        names = ds.scan_valid_cases(self.root)
        self.assertEqual(names, ["case_%05d" % i for i in range(4)])
        self.assertEqual(len(ds.scan_valid_cases(self.root, max_cases=2)), 2)


if __name__ == "__main__":
    unittest.main()