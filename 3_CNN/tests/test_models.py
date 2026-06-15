import sys
import unittest
from pathlib import Path

import torch


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate import models


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class CnnSurrogateTests(unittest.TestCase):
    def test_forward_returns_stiffness_and_local_map(self):
        model = models.CnnSurrogate(
            dropout=0.2,
            embedding_dim=128,
            pooled_height=10,
            pooled_width=5,
        )
        inputs = torch.zeros((4, 1, 80, 40), dtype=torch.float32)
        stiffness, local_map = model(inputs)
        self.assertEqual(tuple(stiffness.shape), (4, 1))
        self.assertEqual(tuple(local_map.shape), (4, 1, 80, 40))

    def test_spatial_image_encoder_returns_spatial_features_and_embedding(self):
        encoder = models.SpatialImageEncoder(
            embedding_dim=32,
            pooled_height=10,
            pooled_width=5,
            dropout=0.0,
        )
        inputs = torch.zeros((2, 1, 80, 40), dtype=torch.float32)
        spatial_features, embedding = encoder(inputs)
        self.assertEqual(tuple(spatial_features.shape), (2, 128, 20, 10))
        self.assertEqual(tuple(embedding.shape), (2, 32))


class DistilledModelCompatibilityTests(unittest.TestCase):
    def test_teacher_and_student_models_keep_legacy_two_target_outputs(self):
        images = torch.zeros((3, 1, 80, 40), dtype=torch.float32)
        local_features = torch.zeros((3, 24), dtype=torch.float32)

        teacher = models.TeacherSurrogate(dropout=0.2)
        student = models.StudentSurrogate(dropout=0.2)

        self.assertEqual(tuple(teacher(images, local_features).shape), (3, 2))
        self.assertEqual(tuple(student(images).shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
