import sys
import unittest
from pathlib import Path

import torch


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.models import TeacherSurrogate, StudentSurrogate


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class DistillationModelTests(unittest.TestCase):
    def test_teacher_uses_image_and_local_features(self):
        model = TeacherSurrogate(dropout=0.2)
        images = torch.zeros((4, 1, 80, 40), dtype=torch.float32)
        local_features = torch.zeros((4, 24), dtype=torch.float32)
        outputs = model(images, local_features)
        self.assertEqual(tuple(outputs.shape), (4, 2))

    def test_student_uses_only_image(self):
        model = StudentSurrogate(dropout=0.2)
        images = torch.zeros((4, 1, 80, 40), dtype=torch.float32)
        outputs = model(images)
        self.assertEqual(tuple(outputs.shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
