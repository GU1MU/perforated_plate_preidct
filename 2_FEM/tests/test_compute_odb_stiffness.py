import io
import os
import sys
import tempfile
import unittest
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = FEM_ROOT / "scripts"
ORIGINAL_CWD = Path.cwd()

sys.path.insert(0, str(SCRIPT_DIR))

import compute_odb_stiffness as stiffness


def tearDownModule():
    try:
        sys.path.remove(str(SCRIPT_DIR))
    except ValueError:
        pass
    os.chdir(ORIGINAL_CWD)


class FakeFieldValue(object):
    def __init__(self, node_label, data):
        self.nodeLabel = node_label
        self.data = data


class FakeSubset(object):
    def __init__(self, values):
        self.values = values


class FakeFieldOutput(object):
    def __init__(self, values_by_region):
        self.values_by_region = values_by_region

    def getSubset(self, region):
        return FakeSubset(self.values_by_region[region.name])


class FakeRegion(object):
    def __init__(self, name):
        self.name = name


class FakeInstance(object):
    def __init__(self):
        self.nodeSets = {
            "TOPEDGE": FakeRegion("TOPEDGE"),
            "BOTTOMEDGE": FakeRegion("BOTTOMEDGE"),
        }


class FakeAssembly(object):
    def __init__(self):
        self.nodeSets = {}
        self.instances = {
            "PLATEINSTANCE": FakeInstance(),
        }


class FakeFrame(object):
    def __init__(self, factor=1.0):
        self.fieldOutputs = {
            "RF": FakeFieldOutput({
                "TOPEDGE": [
                    FakeFieldValue(1, (0.0, 12.0 * factor)),
                    FakeFieldValue(2, (0.0, 18.0 * factor)),
                ],
                "BOTTOMEDGE": [
                    FakeFieldValue(3, (0.0, -14.0 * factor)),
                    FakeFieldValue(4, (0.0, -16.0 * factor)),
                ],
            }),
            "U": FakeFieldOutput({
                "TOPEDGE": [
                    FakeFieldValue(1, (0.0, 0.42 * factor)),
                    FakeFieldValue(2, (0.0, 0.38 * factor)),
                ],
                "BOTTOMEDGE": [
                    FakeFieldValue(3, (0.0, -0.41 * factor)),
                    FakeFieldValue(4, (0.0, -0.39 * factor)),
                ],
            }),
        }


class FakeStep(object):
    def __init__(self):
        self.frames = [FakeFrame(0.5), FakeFrame(1.0), FakeFrame(2.0)]


class FakeStepWithZeroFrame(object):
    def __init__(self):
        self.frames = [FakeFrame(0.0), FakeFrame(1.0)]


class FakeOdb(object):
    def __init__(self):
        self.rootAssembly = FakeAssembly()
        self.steps = {"Load": FakeStep()}


class FakeOdbWithZeroFrame(object):
    def __init__(self):
        self.rootAssembly = FakeAssembly()
        self.steps = {"Load": FakeStepWithZeroFrame()}


class ComputeOdbStiffnessTests(unittest.TestCase):
    def test_resolve_node_set_finds_instance_set_case_insensitively(self):
        odb = FakeOdb()

        region = stiffness.resolve_node_set(odb, "TopEdge", instance_name="PlateInstance")

        self.assertEqual(region.name, "TOPEDGE")

    def test_compute_equivalent_stiffness_from_boundary_mean_reaction_and_displacement(self):
        odb = FakeOdb()

        result = stiffness.compute_equivalent_stiffness(
            odb,
            step_name="Load",
            frame_index=1,
            top_set="TopEdge",
            bottom_set="BottomEdge",
            instance_name="PlateInstance",
            component=2,
        )

        self.assertEqual(result["top_node_count"], 2)
        self.assertEqual(result["bottom_node_count"], 2)
        self.assertAlmostEqual(result["top_mean_reaction"], 15.0)
        self.assertAlmostEqual(result["bottom_mean_reaction"], -15.0)
        self.assertAlmostEqual(result["top_total_reaction"], 30.0)
        self.assertAlmostEqual(result["bottom_total_reaction"], -30.0)
        self.assertAlmostEqual(result["total_reaction_magnitude"], 30.0)
        self.assertAlmostEqual(result["top_mean_displacement"], 0.4)
        self.assertAlmostEqual(result["bottom_mean_displacement"], -0.4)
        self.assertAlmostEqual(result["mean_reaction_magnitude"], 15.0)
        self.assertAlmostEqual(result["mean_displacement_magnitude"], 0.4)
        self.assertAlmostEqual(result["equivalent_stiffness"], 37.5)

    def test_print_result_reports_total_support_reaction(self):
        odb = FakeOdb()
        result = stiffness.compute_equivalent_stiffness(
            odb,
            step_name="Load",
            frame_index=1,
            top_set="TopEdge",
            bottom_set="BottomEdge",
            instance_name="PlateInstance",
            component=2,
        )
        old_stdout = sys.stdout
        stdout = io.StringIO()
        try:
            sys.stdout = stdout
            stiffness.print_result(result)
        finally:
            sys.stdout = old_stdout

        output = stdout.getvalue()
        self.assertIn("top_total_reaction: 30", output)
        self.assertIn("bottom_total_reaction: -30", output)
        self.assertIn("total_reaction_magnitude: 30", output)

    def test_normalize_frame_indices_supports_multiple_and_all_frames(self):
        odb = FakeOdb()

        self.assertEqual(stiffness.normalize_frame_indices(odb, "Load", [0, -1]), [0, 2])
        self.assertEqual(stiffness.normalize_frame_indices(odb, "Load", "all"), [0, 1, 2])

    def test_compute_stiffness_for_frames_returns_one_result_per_frame(self):
        odb = FakeOdb()

        results = stiffness.compute_stiffness_for_frames(
            odb,
            step_name="Load",
            frame_indices=[0, -1],
        )

        self.assertEqual([result["frame_index"] for result in results], [0, 2])
        self.assertEqual([result["requested_frame_index"] for result in results], [0, -1])
        self.assertAlmostEqual(results[0]["equivalent_stiffness"], 37.5)
        self.assertAlmostEqual(results[1]["equivalent_stiffness"], 37.5)

    def test_zero_displacement_frame_is_reported_without_stopping_batch(self):
        odb = FakeOdbWithZeroFrame()

        results = stiffness.compute_stiffness_for_frames(
            odb,
            step_name="Load",
            frame_indices="all",
        )

        self.assertEqual([result["frame_index"] for result in results], [0, 1])
        self.assertEqual(results[0]["status"], "zero_displacement")
        self.assertIsNone(results[0]["equivalent_stiffness"])
        self.assertEqual(results[1]["status"], "ok")
        self.assertAlmostEqual(results[1]["equivalent_stiffness"], 37.5)

    def test_configured_run_uses_temp_relative_odb_and_output_paths(self):
        calls = []

        class FakeOpenOdb(object):
            def __call__(self, path):
                calls.append(path)
                return FakeOdb()

        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                results = stiffness.run_configured_analysis(
                    odb_paths=["odb/sample.odb"],
                    frames=[1],
                    output_dir="stiffness_results",
                    open_odb_func=FakeOpenOdb(),
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(calls, [str(Path(temp_dir) / "odb" / "sample.odb")])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["odb_path"], str(Path(temp_dir) / "odb" / "sample.odb"))
            self.assertTrue((Path(temp_dir) / "stiffness_results" / "sample_stiffness.json").is_file())
            self.assertTrue((Path(temp_dir) / "stiffness_results" / "sample_stiffness.csv").is_file())

    def test_missing_rf_output_has_clear_error(self):
        odb = FakeOdb()
        frame = odb.steps["Load"].frames[-1]
        del frame.fieldOutputs["RF"]

        with self.assertRaisesRegex(ValueError, "field output RF"):
            stiffness.compute_equivalent_stiffness(odb)

    def test_write_result_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "stiffness.json"
            result = {"equivalent_stiffness": 37.5, "component": 2}

            stiffness.write_result_json(result, output_path)

            self.assertIn('"equivalent_stiffness": 37.5', output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
