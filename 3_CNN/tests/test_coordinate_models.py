import sys
import unittest
from pathlib import Path

import torch


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.models import CoordinateSurrogate


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class CoordinateSurrogateTests(unittest.TestCase):
    def test_forward_returns_stiffness_plus_one_local_target_per_hole(self):
        model = CoordinateSurrogate(dropout=0.2)
        coordinates = torch.zeros((4, 24, 6), dtype=torch.float32)

        outputs = model(coordinates)

        self.assertEqual(tuple(outputs.shape), (4, 25))

    def test_global_branch_is_permutation_invariant_and_local_branch_is_equivariant(self):
        torch.manual_seed(1234)
        model = CoordinateSurrogate(
            point_feature_dim=6,
            point_hidden_dim=16,
            context_hidden_dim=32,
            dropout=0.0,
        )
        model.eval()
        coordinates = torch.randn((2, 24, 6), dtype=torch.float32)
        permutation = torch.tensor(
            [5, 2, 9, 1, 7, 0, 12, 3, 8, 4, 10, 6, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            dtype=torch.long,
        )

        outputs = model(coordinates)
        permuted_outputs = model(coordinates[:, permutation, :])

        self.assertTrue(torch.allclose(outputs[:, :1], permuted_outputs[:, :1], atol=1.0e-6))
        self.assertTrue(torch.allclose(outputs[:, 1:][:, permutation], permuted_outputs[:, 1:], atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
