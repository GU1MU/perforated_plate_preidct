import os
import sys
import tempfile
import unittest
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = FEM_ROOT / "scripts"
ORIGINAL_CWD = Path.cwd()

sys.path.insert(0, str(SCRIPT_DIR))

import render_abaqus_strain_colorbar as colorbar


def tearDownModule():
    try:
        sys.path.remove(str(SCRIPT_DIR))
    except ValueError:
        pass
    os.chdir(ORIGINAL_CWD)


class RenderAbaqusStrainColorbarTests(unittest.TestCase):
    def test_render_colorbar_uses_requested_range_and_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs = colorbar.render_colorbar(root, formats=("png",))

            self.assertEqual(outputs, [root / "max_principal_strain_colorbar.png"])
            self.assertTrue(outputs[0].is_file())
            self.assertGreater(outputs[0].stat().st_size, 0)

    def test_colorbar_ticks_and_label_are_fixed(self):
        fig, cbar = colorbar.build_colorbar_figure()
        try:
            tick_labels = [label.get_text() for label in cbar.ax.get_yticklabels()]

            self.assertEqual(tick_labels, ["0.00", "0.01", "0.02", "0.03", "0.04"])
            self.assertEqual(cbar.ax.get_ylabel(), "Max principal strain")
            self.assertEqual(float(cbar.norm.vmin), 0.0)
            self.assertEqual(float(cbar.norm.vmax), 0.04)
        finally:
            colorbar.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
