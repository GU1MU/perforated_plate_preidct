import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = FEM_ROOT / "scripts"
ORIGINAL_CWD = Path.cwd()

sys.path.insert(0, str(SCRIPT_DIR))

import plot_stiffness_ratio_histogram as plotter


def tearDownModule():
    try:
        sys.path.remove(str(SCRIPT_DIR))
    except ValueError:
        pass
    os.chdir(ORIGINAL_CWD)


class PlotStiffnessRatioHistogramTests(unittest.TestCase):
    def write_result(self, root, stem, rows):
        path = root / f"{stem}_stiffness.csv"
        fieldnames = [
            "odb_path",
            "status",
            "frame_index",
            "equivalent_stiffness",
        ]
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def test_compute_ratio_records_uses_solid_baseline_and_last_ok_frame(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_result(root, "solid_1_plate", [
                {"odb_path": "solid_1_plate.odb", "status": "zero_displacement", "frame_index": 0, "equivalent_stiffness": ""},
                {"odb_path": "solid_1_plate.odb", "status": "ok", "frame_index": 1, "equivalent_stiffness": 100.0},
            ])
            self.write_result(root, "1_1_plate", [
                {"odb_path": "1_1_plate.odb", "status": "ok", "frame_index": 1, "equivalent_stiffness": 40.0},
                {"odb_path": "1_1_plate.odb", "status": "ok", "frame_index": 2, "equivalent_stiffness": 50.0},
            ])
            self.write_result(root, "5_1_plate", [
                {"odb_path": "5_1_plate.odb", "status": "ok", "frame_index": 1, "equivalent_stiffness": 75.0},
            ])
            self.write_result(root, "longitudinal_1_plate", [
                {"odb_path": "longitudinal_1_plate.odb", "status": "ok", "frame_index": 1, "equivalent_stiffness": 95.0},
            ])

            rows = plotter.load_stiffness_rows(root)
            ratios = plotter.compute_ratio_records(rows, solid_stem="solid_1_plate")

            self.assertEqual([row["plate"] for row in ratios], ["1_1_plate", "5_1_plate"])
            self.assertEqual([row["label"] for row in ratios], ["Group1+1", "Group5+1"])
            self.assertEqual([row["group_id"] for row in ratios], [1, 5])
            self.assertEqual([row["instance_index"] for row in ratios], [1, 1])
            self.assertEqual([row["frame_index"] for row in ratios], [2, 1])
            self.assertAlmostEqual(ratios[0]["stiffness_ratio"], 0.5)
            self.assertAlmostEqual(ratios[1]["stiffness_ratio"], 0.75)

    def test_render_bar_chart_writes_figure_and_ratio_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ratios = [
                {"plate": "1_1_plate", "label": "Group1+1", "group_id": 1, "instance_index": 1, "equivalent_stiffness": 50.0, "solid_stiffness": 100.0, "stiffness_ratio": 0.5, "frame_index": 1},
                {"plate": "5_1_plate", "label": "Group5+1", "group_id": 5, "instance_index": 1, "equivalent_stiffness": 75.0, "solid_stiffness": 100.0, "stiffness_ratio": 0.75, "frame_index": 1},
            ]

            outputs = plotter.render_outputs(ratios, root, formats=("png",))

            self.assertTrue((root / "stiffness_ratio_by_specimen.png").is_file())
            self.assertTrue((root / "stiffness_ratio_table.csv").is_file())
            self.assertIn(root / "stiffness_ratio_by_specimen.png", outputs)
            self.assertIn(root / "stiffness_ratio_table.csv", outputs)


if __name__ == "__main__":
    unittest.main()
