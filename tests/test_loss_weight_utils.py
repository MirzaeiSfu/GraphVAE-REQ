import unittest

from loss_weight_utils import apply_kia_bce_kl_weights


class KiaLossWeightTests(unittest.TestCase):
    def test_aids_plain_graphvae_replaces_only_bce_and_kl(self):
        self.assertEqual(
            apply_kia_bce_kl_weights([1, 1], "AIDS", True),
            [50.0, 2000.0],
        )

    def test_lobster_replaces_only_bce_and_kl(self):
        self.assertEqual(
            apply_kia_bce_kl_weights([1, 1], "LOBSTER", True),
            [40.0, 2000.0],
        )

    def test_graphvae_mm_statistics_are_preserved(self):
        alpha = [1] * 8 + [1, 1]
        self.assertEqual(
            apply_kia_bce_kl_weights(alpha, "LOBSTER", True),
            [1] * 8 + [40.0, 2000.0],
        )

    def test_disabled_is_a_no_op(self):
        self.assertEqual(apply_kia_bce_kl_weights([1, 1], "LOBSTER", False), [1, 1])

    def test_unsupported_dataset_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "only defined"):
            apply_kia_bce_kl_weights([1, 1], "QM9", True)


if __name__ == "__main__":
    unittest.main()
