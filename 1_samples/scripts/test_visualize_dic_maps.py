import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import visualize_dic_maps as viz


class DicMapVisualizationTests(unittest.TestCase):
    def test_discovers_only_analysis_cloud_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample_1"
            cloud_dir = sample / "Analysis" / "PtCloudAnaly_1"
            cloud_dir.mkdir(parents=True)
            (sample / "machine.csv").write_text("time,force\n0,0\n", encoding="utf-8")
            cloud_csv = cloud_dir / "field_DX_3D.csv"
            cloud_csv.write_text(
                "index,x,y,z,DX\n1,0,0,0,0.1\n2,1,0,0,0.2\n",
                encoding="utf-8",
            )

            cloud_files = viz.find_dic_cloud_csvs(sample)

            self.assertEqual([cloud_csv], cloud_files)

    def test_render_sample_creates_png_and_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample_1"
            cloud_dir = sample / "Analysis" / "PtCloudAnaly_1"
            output_root = root / "figures"
            cloud_dir.mkdir(parents=True)
            cloud_csv = cloud_dir / "field_Exx_3D.csv"
            cloud_csv.write_text(
                "\n".join(
                    [
                        "index,x,y,z,Exx",
                        "1,0,0,0,-0.10",
                        "2,1,0,0,0.05",
                        "3,0,1,0,0.10",
                        "4,1,1,0,-0.02",
                    ]
                ),
                encoding="utf-8",
            )

            rendered = viz.render_sample(sample, output_root)

            pngs = [path for path in rendered if path.suffix == ".png"]
            svgs = [path for path in rendered if path.suffix == ".svg"]
            self.assertEqual(1, len(pngs))
            self.assertEqual(1, len(svgs))
            self.assertTrue(pngs[0].is_file())
            self.assertTrue(svgs[0].is_file())
            self.assertEqual(output_root / "sample_1", pngs[0].parent)

    def test_abaqus_colormap_is_continuous_and_saturated(self):
        cmap, norm = viz.build_colormap(viz.pd.Series([-1.0, 0.0, 1.0]), "blue")

        self.assertNotIsInstance(norm, viz.mcolors.BoundaryNorm)
        self.assertGreaterEqual(cmap.N, 256)
        self.assertLess(cmap(0)[0], 0.25)
        self.assertGreater(cmap(0)[2], 0.45)
        self.assertGreater(cmap(1.0)[0], 0.75)
        self.assertLess(cmap(1.0)[2], 0.25)

    def test_plot_axes_are_bold_without_grid(self):
        fig, ax = viz.plt.subplots()
        try:
            viz.format_axes(ax)

            self.assertFalse(any(line.get_visible() for line in ax.get_xgridlines()))
            self.assertFalse(any(line.get_visible() for line in ax.get_ygridlines()))
            self.assertGreaterEqual(ax.xaxis.label.get_size(), 14)
            self.assertEqual("bold", ax.xaxis.label.get_weight())
            self.assertGreaterEqual(ax.spines["left"].get_linewidth(), 2.0)
            self.assertGreater(ax.xaxis.majorTicks[0].tick1line.get_markersize(), 0)
            self.assertTrue(ax.xaxis.majorTicks[0].tick1line.get_visible())
            self.assertGreater(ax.xaxis.minorTicks[0].tick1line.get_markersize(), 0)
            self.assertTrue(ax.xaxis.minorTicks[0].tick1line.get_visible())
        finally:
            viz.plt.close(fig)

    def test_cloud_field_uses_continuous_mesh_not_point_scatter(self):
        field_data = viz.pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 1.0, 0.5],
                "y": [0.0, 0.0, 1.0, 1.0, 0.5],
                "value": [0.0, 0.2, 0.4, 0.6, 0.3],
            }
        )
        fig, ax = viz.plt.subplots()
        try:
            cmap, norm = viz.build_colormap(field_data["value"], "blue")

            artist = viz.draw_continuous_cloud_field(ax, field_data, cmap, norm)

            self.assertNotEqual("PathCollection", type(artist).__name__)
            self.assertEqual((0.0,), artist.get_linewidths())
            self.assertTrue(artist.get_rasterized())
        finally:
            viz.plt.close(fig)

    def test_clean_boundary_outliers_removes_only_boundary_extremes(self):
        data = viz.pd.DataFrame(
            {
                "index": range(9),
                "x": [0, 5, 10, 0, 5, 10, 0, 5, 10],
                "y": [0, 0, 0, 5, 5, 5, 10, 10, 10],
                "value": [0.0, 0.01, -0.8, 0.02, 1.5, 0.01, 0.0, 0.01, 0.02],
            }
        )

        cleaned, removed = viz.clean_boundary_outliers(
            data,
            boundary_fraction=0.2,
            iqr_multiplier=1.5,
        )

        self.assertEqual({2}, set(removed["index"]))
        self.assertIn(4, set(cleaned["index"]))

    def test_impute_boundary_outliers_keeps_points_and_replaces_values(self):
        data = viz.pd.DataFrame(
            {
                "index": range(9),
                "x": [0, 5, 10, 0, 5, 10, 0, 5, 10],
                "y": [0, 0, 0, 5, 5, 5, 10, 10, 10],
                "value": [0.0, 0.01, -0.8, 0.02, 1.5, 0.01, 0.0, 0.01, 0.02],
            }
        )

        imputed, replacements = viz.impute_boundary_outliers(
            data,
            boundary_fraction=0.2,
            iqr_multiplier=1.5,
            neighbor_count=2,
        )

        self.assertEqual(len(data), len(imputed))
        self.assertEqual({2}, set(replacements["index"]))
        self.assertAlmostEqual(-0.8, float(replacements["original_value"].iloc[0]))
        self.assertAlmostEqual(0.01, float(replacements["imputed_value"].iloc[0]))
        self.assertAlmostEqual(0.01, float(imputed.loc[imputed["index"] == 2, "value"].iloc[0]))

    def test_impute_strain_component_boundary_outliers_keeps_coordinates(self):
        rows = []
        for x in range(5):
            for y in range(5):
                rows.append(
                    {
                        "index": x * 5 + y,
                        "x": float(x),
                        "y": float(y),
                        "eyy": 1.0 if (x, y) == (4, 2) else 0.02,
                        "exy": 0.5,
                    }
                )
        strain = viz.pd.DataFrame(rows)

        imputed, replacements = viz.impute_strain_component_boundary_outliers(
            strain,
            "eyy",
            boundary_fraction=0.2,
            iqr_multiplier=1.5,
            neighbor_count=3,
        )

        self.assertEqual(len(strain), len(imputed))
        self.assertEqual({22}, set(replacements["index"]))
        self.assertAlmostEqual(1.0, float(replacements["original_value"].iloc[0]))
        self.assertAlmostEqual(0.02, float(imputed.loc[imputed["index"] == 22, "eyy"].iloc[0]))
        self.assertAlmostEqual(0.5, float(imputed.loc[imputed["index"] == 22, "exy"].iloc[0]))

    def test_render_longitudinal_keeps_displacement_points_after_exx_imputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "longitudinal"
            cloud_dir = sample / "Analysis" / "PtCloudAnaly_1"
            output_root = root / "figures"
            processed_root = root / "processed"
            cloud_dir.mkdir(parents=True)

            dx_lines = ["index,x,y,z,DX"]
            dy_lines = ["index,x,y,z,DY"]
            exx_lines = ["index,x,y,z,Exx"]
            for x in range(5):
                for y in range(5):
                    index = x * 5 + y
                    dx_lines.append(f"{index},{x},{y},0,{0.2 * x + 0.4 * y}")
                    dy_lines.append(f"{index},{x},{y},0,{-0.1 * x + 0.3 * y}")
                    exx_value = -0.8 if (x, y) == (4, 0) else 0.01
                    exx_lines.append(f"{index},{x},{y},0,{exx_value}")
            (cloud_dir / "field_DX_3D.csv").write_text("\n".join(dx_lines), encoding="utf-8")
            (cloud_dir / "field_DY_3D.csv").write_text("\n".join(dy_lines), encoding="utf-8")
            (cloud_dir / "field_Exx_3D.csv").write_text("\n".join(exx_lines), encoding="utf-8")

            viz.render_sample(sample, output_root, processed_root)

            cleaned_exx = viz.pd.read_csv(processed_root / "longitudinal_cleaned_exx.csv")
            derived = viz.pd.read_csv(processed_root / "longitudinal_derived_strains.csv")
            replacements = viz.pd.read_csv(processed_root / "longitudinal_replaced_exx_outliers.csv")
            eyy_replacements = viz.pd.read_csv(processed_root / "longitudinal_replaced_eyy_outliers.csv")

            self.assertEqual(25, len(cleaned_exx))
            self.assertEqual(25, len(derived))
            self.assertEqual({20}, set(replacements["index"]))
            self.assertEqual(0, len(eyy_replacements))

    def test_compute_small_strains_from_linear_displacement_field(self):
        rows = []
        for x in range(5):
            for y in range(5):
                rows.append(
                    {
                        "index": len(rows),
                        "x": float(x),
                        "y": float(y),
                        "u": 0.2 * x + 0.4 * y + 1.0,
                        "v": -0.1 * x + 0.3 * y - 0.5,
                    }
                )
        displacement = viz.pd.DataFrame(rows)

        strain = viz.compute_small_strain_fields(
            displacement,
            neighbor_count=12,
            min_neighbors=6,
        )

        self.assertAlmostEqual(0.3, float(strain["eyy"].median()), places=10)
        self.assertAlmostEqual(0.15, float(strain["exy"].median()), places=10)

    def test_compute_max_principal_strain_from_components(self):
        strain = viz.pd.DataFrame(
            {
                "exx": [0.30, -0.10],
                "eyy": [0.10, 0.20],
                "exy": [0.20, 0.00],
            }
        )

        result = viz.compute_max_principal_strain(strain)

        self.assertAlmostEqual(0.423606797749979, float(result.iloc[0]), places=12)
        self.assertAlmostEqual(0.20, float(result.iloc[1]), places=12)

    def test_render_sample_creates_max_principal_strain_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "transverse"
            cloud_dir = sample / "Analysis" / "PtCloudAnaly_1"
            output_root = root / "figures"
            processed_root = root / "processed"
            cloud_dir.mkdir(parents=True)

            dx_lines = ["index,x,y,z,DX"]
            dy_lines = ["index,x,y,z,DY"]
            exx_lines = ["index,x,y,z,Exx"]
            for x in range(5):
                for y in range(5):
                    index = x * 5 + y
                    dx_lines.append(f"{index},{x},{y},0,{0.2 * x + 0.4 * y}")
                    dy_lines.append(f"{index},{x},{y},0,{-0.1 * x + 0.3 * y}")
                    exx_lines.append(f"{index},{x},{y},0,0.1")
            (cloud_dir / "field_DX_3D.csv").write_text("\n".join(dx_lines), encoding="utf-8")
            (cloud_dir / "field_DY_3D.csv").write_text("\n".join(dy_lines), encoding="utf-8")
            (cloud_dir / "field_Exx_3D.csv").write_text("\n".join(exx_lines), encoding="utf-8")

            rendered = viz.render_sample(sample, output_root, processed_root)
            derived = viz.pd.read_csv(processed_root / "transverse_derived_strains.csv")

            self.assertIn("max_principal_strain", derived.columns)
            self.assertTrue((output_root / "transverse" / "max_principal_strain.png").is_file())
            self.assertTrue((output_root / "transverse" / "max_principal_strain.svg").is_file())
            self.assertIn(output_root / "transverse" / "max_principal_strain.png", rendered)

    def test_render_sample_creates_local_kf_curve_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "longitudinal"
            (sample / "Analysis").mkdir(parents=True)
            output_root = root / "figures"
            displacement = viz.np.linspace(0.0, 10.0, 120)
            force = 3.0 * displacement
            raw = viz.pd.DataFrame(
                {
                    "序号()": range(len(displacement)),
                    "时间(s)": viz.np.linspace(0.0, 119.0, len(displacement)),
                    "垂向Y1力(kN)": force,
                    "垂向Y2力(kN)": force,
                    "垂向Y2位移(mm)": displacement,
                }
            )
            raw.to_csv(sample / "KF_curve.csv", index=False, encoding="utf-8")

            rendered = viz.render_sample(sample, output_root)

            png = output_root / "longitudinal" / "KF_curve_linear_fit.png"
            svg = output_root / "longitudinal" / "KF_curve_linear_fit.svg"
            self.assertTrue(png.is_file())
            self.assertTrue(svg.is_file())
            self.assertIn(png, rendered)

    def test_elastic_modulus_uses_linear_segment_stiffness(self):
        displacement = viz.np.linspace(0.0, 10.0, 120)
        force = viz.np.piecewise(
            displacement,
            [displacement < 2.0, (displacement >= 2.0) & (displacement <= 8.0), displacement > 8.0],
            [
                lambda x: 0.8 * x,
                lambda x: 3.0 * x - 4.0,
                lambda x: 20.0 + 0.8 * (x - 8.0),
            ],
        )
        curve = viz.pd.DataFrame({"displacement_mm": displacement, "force_kN": force})

        fit = viz.fit_linear_elastic_segment(
            curve,
            area_mm2=300.0,
            gauge_length_mm=200.0,
            min_points=30,
        )

        self.assertAlmostEqual(3.0, fit["stiffness_kN_per_mm"], places=8)
        self.assertAlmostEqual(2000.0, fit["elastic_modulus_MPa"], places=6)


if __name__ == "__main__":
    unittest.main()
