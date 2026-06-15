import math
import os
import sys
import tempfile
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(THIS_DIR / ".ezdxf_cache"))

import ezdxf
import generate_tensile_sample_cad as cad


class GeometryTests(unittest.TestCase):
    def test_schemes_have_expected_dimensions(self):
        schemes = {scheme.name: scheme for scheme in cad.build_schemes()}

        self.assertEqual(set(schemes), {"scheme1", "scheme2"})
        self.assertAlmostEqual(schemes["scheme1"].research_width, 80.0)
        self.assertAlmostEqual(schemes["scheme1"].research_length, 160.0)
        self.assertAlmostEqual(schemes["scheme1"].grip_width, 40.0)
        self.assertAlmostEqual(schemes["scheme1"].grip_length, 60.0)
        self.assertAlmostEqual(schemes["scheme1"].transition_length, 30.0)
        self.assertAlmostEqual(schemes["scheme1"].total_length, 340.0)
        self.assertAlmostEqual(schemes["scheme1"].research_y_min, 90.0)
        self.assertAlmostEqual(schemes["scheme1"].research_y_max, 250.0)

        self.assertAlmostEqual(schemes["scheme2"].research_width, 80.0)
        self.assertAlmostEqual(schemes["scheme2"].research_length, 160.0)
        self.assertAlmostEqual(schemes["scheme2"].grip_width, 35.0)
        self.assertAlmostEqual(schemes["scheme2"].grip_length, 50.0)
        self.assertAlmostEqual(schemes["scheme2"].transition_length, 20.0)
        self.assertAlmostEqual(schemes["scheme2"].total_length, 300.0)
        self.assertAlmostEqual(schemes["scheme2"].research_y_min, 70.0)
        self.assertAlmostEqual(schemes["scheme2"].research_y_max, 230.0)

    def test_specimens_have_expected_names_and_hole_counts(self):
        specimens = cad.build_specimens()

        self.assertEqual(
            [specimen.name for specimen in specimens],
            [
                "scheme1_solid_plate",
                "scheme1_single_hole_plate",
                "scheme1_uniform_perforated_plate",
                "scheme2_solid_plate",
                "scheme2_single_hole_plate",
                "scheme2_uniform_perforated_plate",
            ],
        )
        self.assertEqual(
            {specimen.name: len(specimen.holes) for specimen in specimens},
            {
                "scheme1_solid_plate": 0,
                "scheme1_single_hole_plate": 1,
                "scheme1_uniform_perforated_plate": 24,
                "scheme2_solid_plate": 0,
                "scheme2_single_hole_plate": 1,
                "scheme2_uniform_perforated_plate": 24,
            },
        )

    def test_single_hole_area_matches_twenty_four_small_holes(self):
        for specimen in cad.build_specimens():
            if "single_hole" not in specimen.name:
                continue
            with self.subTest(specimen=specimen.name):
                hole = specimen.holes[0]
                self.assertAlmostEqual(hole.diameter, cad.HOLE_DIAMETER * math.sqrt(24))
                self.assertAlmostEqual(
                    math.pi * hole.radius ** 2,
                    24 * math.pi * (cad.HOLE_DIAMETER / 2) ** 2,
                )

    def test_all_specimens_pass_geometry_validation(self):
        for specimen in cad.build_specimens():
            with self.subTest(specimen=specimen.name):
                cad.validate_specimen(specimen)

    def test_outer_outline_is_centered_and_uses_expected_bounds(self):
        for scheme in cad.build_schemes():
            with self.subTest(scheme=scheme.name):
                outline = cad.build_outline_points(scheme)
                xs = [point[0] for point in outline]
                ys = [point[1] for point in outline]
                self.assertAlmostEqual(min(xs), 0.0)
                self.assertAlmostEqual(max(xs), 80.0)
                self.assertAlmostEqual(min(ys), 0.0)
                self.assertAlmostEqual(max(ys), scheme.total_length)


class OutputTests(unittest.TestCase):
    def test_write_outputs_creates_six_dxf_files_and_supporting_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            written_paths = cad.write_outputs(output_dir)

            expected_dxf_names = {
                "scheme1_solid_plate.dxf",
                "scheme1_single_hole_plate.dxf",
                "scheme1_uniform_perforated_plate.dxf",
                "scheme2_solid_plate.dxf",
                "scheme2_single_hole_plate.dxf",
                "scheme2_uniform_perforated_plate.dxf",
            }
            self.assertEqual({path.name for path in written_paths if path.suffix.lower() == ".dxf"}, expected_dxf_names)
            self.assertTrue((output_dir / "tensile_sample_hole_coordinates.csv").is_file())
            self.assertTrue((output_dir / "README.md").is_file())

            expected_circle_counts = {
                "scheme1_solid_plate.dxf": 0,
                "scheme1_single_hole_plate.dxf": 1,
                "scheme1_uniform_perforated_plate.dxf": 24,
                "scheme2_solid_plate.dxf": 0,
                "scheme2_single_hole_plate.dxf": 1,
                "scheme2_uniform_perforated_plate.dxf": 24,
            }
            for filename, expected_count in expected_circle_counts.items():
                with self.subTest(filename=filename):
                    doc = ezdxf.readfile(output_dir / filename)
                    circles = sum(1 for entity in doc.modelspace() if entity.dxftype() == "CIRCLE")
                    outlines = sum(1 for entity in doc.modelspace() if entity.dxftype() == "LWPOLYLINE")
                    self.assertEqual(circles, expected_count)
                    self.assertEqual(outlines, 1)

    def test_coordinate_table_contains_all_non_solid_holes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            cad.write_outputs(output_dir)

            rows = (output_dir / "tensile_sample_hole_coordinates.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1 + 1 + 24 + 1 + 24)
            self.assertEqual(rows[0], "specimen,hole_index,x_mm,y_mm,radius_mm,diameter_mm,note")


if __name__ == "__main__":
    unittest.main()
