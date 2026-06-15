import sys
import unittest
from pathlib import Path

import torch


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.devices import resolve_device, should_pin_memory


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class DeviceResolutionTests(unittest.TestCase):
    def test_resolve_device_uses_cpu_when_cpu_is_requested(self):
        device = resolve_device("cpu")

        self.assertEqual(device.type, "cpu")
        self.assertFalse(should_pin_memory(device))

    def test_resolve_device_auto_falls_back_to_cpu_without_cuda(self):
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            device = resolve_device("auto")
        finally:
            torch.cuda.is_available = original_is_available

        self.assertEqual(device.type, "cpu")

    def test_resolve_device_auto_selects_cuda_when_available(self):
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: True
        try:
            device = resolve_device("auto")
        finally:
            torch.cuda.is_available = original_is_available

        self.assertEqual(device.type, "cuda")
        self.assertTrue(should_pin_memory(device))

    def test_resolve_device_rejects_cuda_when_cuda_is_unavailable(self):
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                resolve_device("cuda")
        finally:
            torch.cuda.is_available = original_is_available


if __name__ == "__main__":
    unittest.main()
