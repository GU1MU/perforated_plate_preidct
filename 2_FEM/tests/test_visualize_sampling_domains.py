import json
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = FEM_ROOT / "scripts"
ORIGINAL_CWD = Path.cwd()

sys.path.insert(0, str(SCRIPT_DIR))

import visualize_sampling_domains as viz


def tearDownModule():
    try:
        sys.path.remove(str(SCRIPT_DIR))
    except ValueError:
        pass
    os.chdir(ORIGINAL_CWD)


class VisualizeSamplingDomainsTests(unittest.TestCase):
    def write_manifest(self, root):
        manifest = {
            "plate": {
                "x": 150.0,
                "y": 200.0,
                "thickness": 2.0,
            },
            "constraints": {
                "hole_count": 40,
                "hole_radius": 5.0,
                "min_center_distance": 14.0,
                "min_center_to_edge": 10.0,
            },
            "sampling_domains": [
                {
                    "group_id": 1,
                    "cluster": "low",
                    "direction": "x",
                    "x_range": [10.0, 140.0],
                    "y_range": [40.0, 160.0],
                },
                {
                    "group_id": 5,
                    "cluster": "medium",
                    "direction": "none",
                    "x_range": [17.5, 132.5],
                    "y_range": [42.5, 157.5],
                },
                {
                    "group_id": 9,
                    "cluster": "high",
                    "direction": "y",
                    "x_range": [42.5, 107.5],
                    "y_range": [10.0, 190.0],
                },
            ],
        }
        manifest_path = root / "group_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_load_sampling_domains_keeps_requested_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = self.write_manifest(Path(temp_dir))

            manifest = viz.load_manifest(manifest_path)
            domains = viz.load_sampling_domains(manifest, [9, 1, 5])

            self.assertEqual([domain.group_id for domain in domains], [9, 1, 5])
            self.assertEqual(domains[0].x_range, (42.5, 107.5))
            self.assertEqual(domains[1].y_range, (40.0, 160.0))

    def test_load_generator_manifest_uses_current_group_definitions(self):
        spec = importlib.util.spec_from_file_location(
            "expected_generate_perforated_plate_inp",
            viz.DEFAULT_GENERATOR_PATH,
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        expected = [
            group for group in generator.GROUP_DEFINITIONS if group["id"] == 5
        ][0]

        manifest = viz.load_generator_manifest(viz.DEFAULT_GENERATOR_PATH)
        domains = viz.load_sampling_domains(manifest, [5])

        self.assertEqual(domains[0].x_range, expected["x_range"])
        self.assertEqual(domains[0].y_range, expected["y_range"])

    def test_render_sampling_domain_figures_writes_combined_and_individual_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            manifest = viz.load_manifest(manifest_path)
            domains = viz.load_sampling_domains(manifest, [1, 5, 9])
            output_dir = root / "figures"

            outputs = viz.render_sampling_domain_figures(
                manifest,
                domains,
                output_dir,
                formats=("png",),
            )

            output_names = sorted(path.name for path in outputs)
            self.assertEqual(
                output_names,
                [
                    "sampling_domain_group_1.png",
                    "sampling_domain_group_5.png",
                    "sampling_domain_group_9.png",
                    "sampling_domains_groups_1_5_9.png",
                ],
            )
            for output_path in outputs:
                self.assertTrue(output_path.exists(), output_path)
                self.assertGreater(output_path.stat().st_size, 0, output_path)

    def test_render_sampling_domain_figures_do_not_draw_legends(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            manifest = viz.load_manifest(manifest_path)
            domains = viz.load_sampling_domains(manifest, [1, 5, 9])
            saw_legend = []
            original_save_figure = viz._save_figure

            def capture_figure(fig, output_dir, stem, formats):
                saw_legend.append(fig.axes[0].get_legend() is not None)
                viz.plt.close(fig)
                return [Path(output_dir) / f"{stem}.png"]

            try:
                viz._save_figure = capture_figure
                viz.render_sampling_domain_figures(
                    manifest,
                    domains,
                    root / "figures",
                    formats=("png",),
                )
            finally:
                viz._save_figure = original_save_figure

            self.assertEqual(saw_legend, [False, False, False, False])


if __name__ == "__main__":
    unittest.main()
