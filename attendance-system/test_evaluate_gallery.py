"""Unit tests for the empirical gallery calibration module."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from evaluate_gallery import (
    load_gallery_matrix,
    compute_stats,
    evaluate_pairwise_distributions,
    run_loocv,
    run_threshold_sweep,
    run_margin_sweep,
    prune_gallery_outliers,
)


class TestEvaluateGallery(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.gallery_path = Path(self.temp_dir.name) / "test_gallery.json"

        # Create two clearly separable synthetic classes (128-d unit vectors)
        v_alice = np.zeros(128, dtype=np.float32)
        v_alice[0] = 1.0  # Basis e_0

        v_bob = np.zeros(128, dtype=np.float32)
        v_bob[1] = 1.0    # Basis e_1

        # Add slight perturbations for realistic intra-class variation
        alice_refs = []
        for i in range(5):
            v = v_alice.copy()
            v[2] = 0.05 * i
            v /= np.linalg.norm(v)
            alice_refs.append(v.tolist())

        bob_refs = []
        for i in range(5):
            v = v_bob.copy()
            v[3] = 0.05 * i
            v /= np.linalg.norm(v)
            bob_refs.append(v.tolist())

        self.sample_data = {"Alice": alice_refs, "Bob": bob_refs}
        with open(self.gallery_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_data, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_gallery_matrix(self):
        names, X, y = load_gallery_matrix(self.gallery_path)
        self.assertEqual(names, ["Alice", "Bob"])
        self.assertEqual(X.shape, (10, 128))
        self.assertEqual(len(y), 10)
        self.assertEqual(list(np.unique(y)), [0, 1])

        # Test L2-norm is ~1.0
        norms = np.linalg.norm(X, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_compute_stats(self):
        vals = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        stats = compute_stats(vals)
        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["min"], 0.2)
        self.assertAlmostEqual(stats["max"], 1.0)
        self.assertAlmostEqual(stats["mean"], 0.6)
        self.assertAlmostEqual(stats["median"], 0.6)

    def test_evaluate_pairwise_distributions(self):
        names, X, y = load_gallery_matrix(self.gallery_path)
        dist = evaluate_pairwise_distributions(names, X, y)

        self.assertIn("per_person_intra", dist)
        self.assertIn("Alice", dist["per_person_intra"])
        self.assertIn("Bob", dist["per_person_intra"])

        # Intra-class pairs: Alice has 5 samples -> 5*4/2 = 10 pairs
        self.assertEqual(dist["per_person_intra"]["Alice"]["count"], 10)
        self.assertGreater(dist["per_person_intra"]["Alice"]["min"], 0.90)

        # Inter-class cross pairs: 5 * 5 = 25 pairs
        self.assertEqual(dist["inter_class"]["count"], 25)
        # Alice (e0) vs Bob (e1) orthogonal -> similarity ~ 0.0
        self.assertLess(dist["inter_class"]["max"], 0.10)

    def test_loocv_separable_classes(self):
        names, X, y = load_gallery_matrix(self.gallery_path)
        res = run_loocv(names, X, y, threshold=0.50, margin=0.08)

        self.assertEqual(res["total"], 10)
        self.assertEqual(res["correct"], 10)
        self.assertEqual(res["unknown"], 0)
        self.assertEqual(res["mismatch"], 0)
        self.assertAlmostEqual(res["tar"], 1.0)
        self.assertAlmostEqual(res["far"], 0.0)

    def test_loocv_threshold_and_margin_sweeps(self):
        names, X, y = load_gallery_matrix(self.gallery_path)
        th_res = run_threshold_sweep(names, X, y, margin=0.08, thresholds=[0.40, 0.60, 0.999])
        self.assertEqual(len(th_res), 3)
        # At 0.40 and 0.60, TAR should be 100%
        self.assertEqual(th_res[0]["correct"], 10)
        self.assertEqual(th_res[1]["correct"], 10)
        # At 0.999, high threshold may reject
        self.assertGreaterEqual(th_res[2]["unknown"], 0)

        mg_res = run_margin_sweep(names, X, y, threshold=0.50, margins=[0.0, 0.08, 0.5])
        self.assertEqual(len(mg_res), 3)
        self.assertEqual(mg_res[0]["correct"], 10)

    def test_invalid_gallery_validation(self):
        # 1. Non-existent file
        bogus_file = Path(self.temp_dir.name) / "non_existent.json"
        with self.assertRaises(FileNotFoundError):
            load_gallery_matrix(bogus_file)

        # 2. Invalid embedding dimension
        bad_file = Path(self.temp_dir.name) / "bad_dim.json"
        with open(bad_file, "w", encoding="utf-8") as f:
            json.dump({"Alice": [[1.0, 0.0]]}, f)
        with self.assertRaises(ValueError):
            load_gallery_matrix(bad_file)

    def test_prune_gallery_outliers(self):
        # Inject a noisy outlier embedding for Alice (orthogonal vector e_10)
        with open(self.gallery_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        outlier = [0.0] * 128
        outlier[10] = 1.0  # orthogonal to Alice (e_0) -> cosine similarity ~ 0.0
        data["Alice"].append(outlier)
        with open(self.gallery_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Before pruning: Alice has 6 samples
        self.assertEqual(len(data["Alice"]), 6)

        ok, msg, stats = prune_gallery_outliers(self.gallery_path, min_similarity=0.45)
        self.assertTrue(ok)
        self.assertEqual(stats["Alice"]["original"], 6)
        self.assertEqual(stats["Alice"]["kept"], 5)
        self.assertEqual(stats["Alice"]["removed"], 1)
        self.assertEqual(stats["Bob"]["removed"], 0)

        # Confirm gallery file updated
        with open(self.gallery_path, "r", encoding="utf-8") as f:
            pruned = json.load(f)
        self.assertEqual(len(pruned["Alice"]), 5)

        # Confirm backup was saved
        backup_dir = self.gallery_path.parent / "gallery_backups"
        self.assertTrue(backup_dir.exists())
        backups = list(backup_dir.glob("gallery_pre_prune_*.json"))
        self.assertGreaterEqual(len(backups), 1)


if __name__ == "__main__":
    unittest.main()
