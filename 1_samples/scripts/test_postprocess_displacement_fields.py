import json
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import postprocess_displacement_fields as post


def grid_points(size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rows = [(float(x), float(y)) for x in range(size) for y in range(size)]
    points = np.array(rows, dtype=float)
    return points[:, 0], points[:, 1]


class DisplacementPostprocessingTests(unittest.TestCase):
    def test_affine_fit_recovers_translation_and_gradient(self):
        x, y = grid_points()
        center = (float(x.mean()), float(y.mean()))
        H = np.array([[0.12, -0.03], [0.08, 0.22]], dtype=float)
        translation = np.array([1.5, -0.7], dtype=float)
        xy = np.column_stack([x - center[0], y - center[1]])
        uv = translation + xy @ H.T

        result = post.fit_affine_displacement(x, y, uv[:, 0], uv[:, 1])

        np.testing.assert_allclose(result.translation, translation, atol=1e-12)
        np.testing.assert_allclose(result.H, H, atol=1e-12)
        np.testing.assert_allclose(result.center, center, atol=1e-12)

    def test_rigid_removal_keeps_symmetric_strain_only(self):
        x, y = grid_points()
        center = (float(x.mean()), float(y.mean()))
        strain = np.array([[-0.04, 0.03], [0.03, 0.16]], dtype=float)
        rotation = np.array([[0.0, -0.11], [0.11, 0.0]], dtype=float)
        H = strain + rotation
        xy = np.column_stack([x - center[0], y - center[1]])
        uv = np.array([0.4, -0.2]) + xy @ H.T
        fit = post.fit_affine_displacement(x, y, uv[:, 0], uv[:, 1])

        ux1, uy1 = post.remove_rigid_motion(x, y, uv[:, 0], uv[:, 1], fit)
        corrected = post.fit_affine_displacement(x, y, ux1, uy1, center=fit.center)

        np.testing.assert_allclose(corrected.translation, [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(corrected.H, strain, atol=1e-12)

    def test_average_shear_removal_zeroes_off_diagonal_strain(self):
        x, y = grid_points()
        center = (float(x.mean()), float(y.mean()))
        strain = np.array([[-0.05, 0.06], [0.06, 0.18]], dtype=float)
        xy = np.column_stack([x - center[0], y - center[1]])
        uv = xy @ strain.T
        fit = post.fit_affine_displacement(x, y, uv[:, 0], uv[:, 1])
        metrics = post.compute_average_strain(fit.H)

        ux2, uy2 = post.remove_average_shear(
            x,
            y,
            uv[:, 0],
            uv[:, 1],
            metrics["gamma_xy"],
            center=fit.center,
        )
        corrected = post.fit_affine_displacement(x, y, ux2, uy2, center=fit.center)

        np.testing.assert_allclose(np.diag(corrected.H), np.diag(strain), atol=1e-12)
        self.assertAlmostEqual(0.0, corrected.H[0, 1], places=12)
        self.assertAlmostEqual(0.0, corrected.H[1, 0], places=12)

    def test_compute_local_small_strain_fields_recovers_linear_field(self):
        x, y = grid_points()
        data = pd.DataFrame(
            {
                "index": range(len(x)),
                "x": x,
                "y": y,
                "z": np.zeros(len(x)),
                "u": 0.20 * x + 0.40 * y + 1.0,
                "v": -0.10 * x + 0.30 * y - 0.5,
            }
        )

        strain = post.compute_local_small_strain_fields(
            data,
            "u",
            "v",
            prefix="raw",
            neighbor_count=12,
            min_neighbors=6,
        )

        self.assertEqual(len(data), len(strain))
        self.assertAlmostEqual(0.20, float(strain["raw_exx"].median()), places=12)
        self.assertAlmostEqual(0.30, float(strain["raw_eyy"].median()), places=12)
        self.assertAlmostEqual(0.15, float(strain["raw_exy"].median()), places=12)
        self.assertAlmostEqual(0.30, float(strain["raw_gamma_xy"].median()), places=12)
        self.assertIn("raw_max_principal_strain", strain.columns)
        self.assertIn("raw_principal_angle_to_y_deg", strain.columns)

    def test_render_strain_maps_creates_key_component_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y = grid_points(size=3)
            strain = pd.DataFrame({"index": range(len(x)), "x": x, "y": y})
            for prefix in ("raw", "rigid_removed", "shear_removed"):
                strain[f"{prefix}_exx"] = 0.01 + 0.001 * x
                strain[f"{prefix}_eyy"] = 0.02 + 0.001 * y
                strain[f"{prefix}_exy"] = 0.003
                strain[f"{prefix}_gamma_xy"] = 0.006
                strain[f"{prefix}_max_principal_strain"] = 0.025

            rendered = post.render_strain_maps(strain, "sample_1", root)

            self.assertEqual(30, len(rendered))
            self.assertTrue((root / "raw_exx_strain.png").is_file())
            self.assertTrue((root / "shear_removed_max_principal_strain.svg").is_file())

    def test_process_sample_writes_postprocessed_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "longitudinal"
            cloud_dir = sample / "Analysis" / "PtCloudAnaly_1"
            output_root = root / "processed"
            cloud_dir.mkdir(parents=True)

            dx_lines = ["index,x,y,z,DX"]
            dy_lines = ["index,x,y,z,DY"]
            for x in range(4):
                for y in range(4):
                    index = x * 4 + y
                    ux = 0.25 + 0.02 * x - 0.04 * y
                    uy = -0.10 + 0.05 * x + 0.15 * y
                    dx_lines.append(f"{index},{x},{y},0,{ux}")
                    dy_lines.append(f"{index},{x},{y},0,{uy}")
            (cloud_dir / "field_DX_3D.csv").write_text("\n".join(dx_lines), encoding="utf-8")
            (cloud_dir / "field_DY_3D.csv").write_text("\n".join(dy_lines), encoding="utf-8")

            outputs = post.process_sample(sample, output_root, write_plots=False)

            csv_path = output_root / "longitudinal" / "longitudinal_displacement_postprocessed.csv"
            summary_path = output_root / "longitudinal" / "longitudinal_affine_summary.json"
            strain_path = output_root / "longitudinal" / "longitudinal_strain_fields.csv"
            self.assertIn(csv_path, outputs)
            self.assertIn(summary_path, outputs)
            self.assertIn(strain_path, outputs)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(strain_path.is_file())

            cleaned = pd.read_csv(csv_path)
            self.assertEqual(16, len(cleaned))
            self.assertIn("ux_rigid_removed", cleaned.columns)
            self.assertIn("uy_shear_removed", cleaned.columns)
            self.assertIn("fit_inlier", cleaned.columns)
            strain = pd.read_csv(strain_path)
            self.assertEqual(16, len(strain))
            self.assertIn("raw_exx", strain.columns)
            self.assertIn("rigid_removed_exy", strain.columns)
            self.assertIn("shear_removed_gamma_xy", strain.columns)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual("longitudinal", summary["sample"])
            self.assertEqual(16, summary["point_count"])
            self.assertIn("H", summary["raw_fit"])
            self.assertIn("principal_angle_to_y_deg", summary["raw_fit"])
            self.assertIn("strain_fields", summary)


if __name__ == "__main__":
    unittest.main()
