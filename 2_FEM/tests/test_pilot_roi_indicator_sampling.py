import os
import shutil
import sys
import unittest
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = FEM_ROOT / "scripts"
ABAQUS_WORKDIR = FEM_ROOT / "temp"
TEST_WORKDIR = ABAQUS_WORKDIR / "tests"

TEST_WORKDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR))

import pilot_roi_indicator_sampling as pilot


def tearDownModule():
    try:
        sys.path.remove(str(SCRIPT_DIR))
    except ValueError:
        pass


class PilotRoiIndicatorSamplingTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        os.chdir(ABAQUS_WORKDIR)

    def tearDown(self):
        os.chdir(self.old_cwd)

    def test_test_module_uses_two_fem_temp_as_workdir(self):
        self.assertEqual(Path.cwd(), ABAQUS_WORKDIR)

    def test_main_rejects_candidate_count_below_balanced_minimum(self):
        output_dir = TEST_WORKDIR / "pilot_underfilled_precheck"
        if output_dir.exists():
            shutil.rmtree(str(output_dir))

        with self.assertRaisesRegex(ValueError, "candidate-count"):
            pilot.main([
                "--candidate-count",
                "9",
                "--target-per-bin",
                "2",
                "--seed",
                "2",
                "--output-dir",
                str(output_dir),
            ])

        self.assertFalse(output_dir.exists())

    def test_select_balanced_records_rejects_underfilled_bins(self):
        records = []
        for index in range(3):
            records.append({
                "layout_id": "layout_%d" % index,
                "cluster_label": "low",
                "orientation_label": "x",
                "bin": "low_x",
                "holes": [],
                "metrics": {},
            })

        with self.assertRaisesRegex(ValueError, "underfilled sampling bins"):
            pilot.select_balanced_records(records, 1)


if __name__ == "__main__":
    unittest.main()
