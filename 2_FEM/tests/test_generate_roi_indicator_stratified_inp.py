import json
import os
import sys
import unittest
import codecs
import types
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = FEM_ROOT / "scripts"
ABAQUS_WORKDIR = FEM_ROOT / "temp"
TEST_WORKDIR = ABAQUS_WORKDIR / "tests"

TEST_WORKDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR))

import generate_roi_indicator_stratified_inp as roi


def tearDownModule():
    try:
        sys.path.remove(str(SCRIPT_DIR))
    except ValueError:
        pass


class GenerateRoiIndicatorStratifiedInpTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        os.chdir(ABAQUS_WORKDIR)

    def tearDown(self):
        os.chdir(self.old_cwd)

    def assertSamePath(self, actual, expected):
        actual_path = os.path.normcase(os.path.normpath(str(actual)))
        expected_path = os.path.normcase(os.path.normpath(str(expected)))
        self.assertEqual(actual_path, expected_path)

    def valid_holes(self):
        holes = []
        x_values = [10.0, 30.0, 50.0, 70.0]
        y_values = [10.0, 38.0, 66.0, 94.0, 122.0, 150.0]
        for y in y_values:
            for x in x_values:
                holes.append({"x": x, "y": y, "r": roi.HOLE_RADIUS})
        self.assertEqual(len(holes), roi.HOLE_COUNT)
        return holes

    def payload(self):
        holes = self.valid_holes()
        return {
            "schema": "roi_indicator_stratified_layouts_v1",
            "seed": 1,
            "candidate_count": 9,
            "target_per_bin": 1,
            "geometry": {
                "plate_x": roi.PLATE_X,
                "plate_y": roi.PLATE_Y,
                "plate_thickness": roi.PLATE_THICKNESS,
                "hole_radius": roi.HOLE_RADIUS,
                "hole_count": roi.HOLE_COUNT,
                "min_center_distance": roi.MIN_CENTER_DISTANCE,
                "min_center_to_edge": roi.MIN_CENTER_TO_EDGE,
            },
            "thresholds": {},
            "bin_counts": {"low_x": 1},
            "layouts": [{
                "layout_id": "roi_test_00001",
                "cluster_label": "low",
                "orientation_label": "x",
                "bin": "low_x",
                "holes": holes,
                "metrics": roi.calculate_metrics(holes),
            }],
        }

    def layout_for_group(self, layout_id, cluster_label, orientation_label):
        holes = self.valid_holes()
        return {
            "layout_id": layout_id,
            "cluster_label": cluster_label,
            "orientation_label": orientation_label,
            "bin": "%s_%s" % (cluster_label, orientation_label),
            "holes": holes,
            "metrics": roi.calculate_metrics(holes),
        }

    def install_fake_abaqus(self):
        class FakeCollection(list):
            def getByBoundingBox(self, **_kwargs):
                return self

        class FakeSketch(object):
            def __init__(self):
                self.rectangles = []
                self.circles = []

            def rectangle(self, point1, point2):
                self.rectangles.append((point1, point2))

            def CircleByCenterPerimeter(self, center, point1):
                self.circles.append((center, point1))

        class FakePart(object):
            def __init__(self):
                self.faces = FakeCollection(["face"])
                self.edges = FakeCollection(["edge"])
                self.vertices = FakeCollection(["vertex"])
                self.sets = {}

            def BaseShell(self, sketch):
                self.sketch = sketch

            def Set(self, **kwargs):
                self.sets[kwargs["name"]] = kwargs
                return kwargs

            def SectionAssignment(self, **_kwargs):
                return None

            def seedPart(self, **_kwargs):
                return None

            def setMeshControls(self, **_kwargs):
                return None

            def setElementType(self, **_kwargs):
                return None

            def generateMesh(self):
                return None

        class FakeInstance(object):
            def __init__(self, part):
                self.sets = part.sets

        class FakeAssembly(object):
            def Instance(self, name, part, dependent):
                return FakeInstance(part)

        class FakeOutputRequest(object):
            def setValues(self, variables):
                self.variables = variables

        class FakeModel(object):
            def __init__(self, name):
                self.name = name
                self.rootAssembly = FakeAssembly()
                self.fieldOutputRequests = {"F-Output-1": FakeOutputRequest()}

            def ConstrainedSketch(self, name, sheetSize):
                sketch = FakeSketch()
                sketch.name = name
                sketch.sheetSize = sheetSize
                return sketch

            def Part(self, **_kwargs):
                return FakePart()

            def Material(self, name):
                class FakeMaterial(object):
                    def Elastic(self, table):
                        self.table = table
                return FakeMaterial()

            def HomogeneousSolidSection(self, **_kwargs):
                return None

            def StaticStep(self, **_kwargs):
                return None

            def DisplacementBC(self, **_kwargs):
                return None

        class FakeJob(object):
            def __init__(self, name, model):
                self.name = name
                self.model = model

            def writeInput(self, consistencyChecking):
                with open(self.name + ".inp", "w", encoding="utf-8") as inp_file:
                    inp_file.write("fake input for %s" % self.name)

        class FakeMdb(object):
            def __init__(self):
                self.models = {}
                self.jobs = {}
                self.last_model_name = None
                self.last_job_name = None

            def Model(self, name):
                self.last_model_name = name
                model = FakeModel(name)
                self.models[name] = model
                return model

            def Job(self, name, model):
                self.last_job_name = name
                job = FakeJob(name, model)
                self.jobs[name] = job
                return job

        class FakeElemType(object):
            def __init__(self, elemCode, elemLibrary):
                self.elemCode = elemCode
                self.elemLibrary = elemLibrary

        fake_mdb = FakeMdb()
        fake_abaqus = types.ModuleType("abaqus")
        fake_abaqus.mdb = fake_mdb
        fake_constants = types.ModuleType("abaqusConstants")
        for name in (
            "TWO_D_PLANAR",
            "DEFORMABLE_BODY",
            "ON",
            "OFF",
            "UNSET",
            "STANDARD",
            "TRI",
            "CPS6",
        ):
            setattr(fake_constants, name, name)
        fake_mesh = types.ModuleType("mesh")
        fake_mesh.ElemType = FakeElemType

        old_modules = {}
        for name, module in (
            ("abaqus", fake_abaqus),
            ("abaqusConstants", fake_constants),
            ("mesh", fake_mesh),
        ):
            old_modules[name] = sys.modules.get(name)
            sys.modules[name] = module
        return fake_mdb, old_modules

    def restore_fake_abaqus(self, old_modules):
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def write_payload(self, payload, name):
        path = TEST_WORKDIR / name
        with open(str(path), "w", encoding="utf-8") as json_file:
            json.dump(payload, json_file)
        return path

    def test_test_module_uses_two_fem_temp_as_workdir(self):
        self.assertEqual(Path.cwd(), ABAQUS_WORKDIR)

    def test_temp_work_dir_is_explicit_two_fem_temp(self):
        old_cwd = Path.cwd()
        try:
            entered = roi.enter_temp_work_dir()
            self.assertSamePath(entered, ABAQUS_WORKDIR)
            self.assertSamePath(Path.cwd(), ABAQUS_WORKDIR)
        finally:
            os.chdir(old_cwd)

    def test_directory_helpers_align_with_legacy_solve_inp_output_dir(self):
        self.assertSamePath(roi.fem_root(), FEM_ROOT)
        self.assertSamePath(roi.temp_dir(), FEM_ROOT / "temp")
        self.assertSamePath(roi.inp_dir(), FEM_ROOT / "temp" / "solve_inp")
        self.assertSamePath(roi.test_work_dir(), TEST_WORKDIR)
        self.assertSamePath(roi.DEFAULT_OUTPUT_DIR, roi.inp_dir())

    def test_editable_parameters_align_with_legacy_script_style(self):
        self.assertTrue(hasattr(roi, "SELECTED_LAYOUTS_JSON"))
        self.assertAlmostEqual(roi.E, roi.DEFAULT_E)
        self.assertAlmostEqual(roi.U, roi.DEFAULT_U)
        self.assertAlmostEqual(roi.MESH_SIZE, roi.DEFAULT_MESH_SIZE)
        self.assertGreaterEqual(roi.INSTANCE, 0)
        self.assertIn(roi.REF, roi.VALID_REFS)
        self.assertEqual(roi.VALID_REFS, (None, "solid", "uniform"))
        self.assertTrue(hasattr(roi, "GROUP_INSTANCE_COUNTS"))
        self.assertIsInstance(roi.WARM_START, bool)

    def test_group_definitions_follow_legacy_nine_group_order(self):
        groups = roi.GROUP_DEFINITIONS
        self.assertEqual([group["id"] for group in groups], list(range(1, 10)))
        self.assertEqual(
            [(group["cluster"], group["direction"], group["bin"]) for group in groups],
            [
                ("low", "x", "low_x"),
                ("low", "none", "low_none"),
                ("low", "y", "low_y"),
                ("medium", "x", "medium_x"),
                ("medium", "none", "medium_none"),
                ("medium", "y", "medium_y"),
                ("high", "x", "high_x"),
                ("high", "none", "high_none"),
                ("high", "y", "high_y"),
            ],
        )

    def test_build_run_plan_warm_start_derives_missing_group_bin(self):
        payload = self.payload()
        old_groups = roi.GROUP_DEFINITIONS
        old_inp_dir = roi.inp_dir
        groups = [dict(group) for group in old_groups]
        groups[0].pop("bin", None)
        groups[0]["binTrue"] = "low_x"
        try:
            roi.GROUP_DEFINITIONS = groups
            roi.inp_dir = lambda: str(TEST_WORKDIR / "warm_start_missing_group_bin")
            plan = roi.build_run_plan(
                payload,
                instance_count={1: 1},
                warm_start=True,
            )
        finally:
            roi.GROUP_DEFINITIONS = old_groups
            roi.inp_dir = old_inp_dir

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["group_id"], 1)
        self.assertEqual(plan[0]["bin"], "low_x")

    def test_build_run_plan_names_outputs_by_indicator_group_and_instance(self):
        payload = self.payload()
        payload["layouts"] = [
            self.layout_for_group("roi_low_x_1", "low", "x"),
            self.layout_for_group("roi_medium_none_1", "medium", "none"),
            self.layout_for_group("roi_low_x_2", "low", "x"),
        ]
        old_inp_dir = roi.inp_dir
        try:
            roi.inp_dir = lambda: str(TEST_WORKDIR / "aligned_names")
            plan = roi.build_run_plan(
                payload,
                instance_count={1: 2, 5: 1},
                warm_start=False,
            )
        finally:
            roi.inp_dir = old_inp_dir

        self.assertEqual([item["inp_name"] for item in plan], [
            "1_1_plate.inp",
            "1_2_plate.inp",
            "5_1_plate.inp",
        ])
        self.assertEqual([item["group_id"] for item in plan], [1, 1, 5])
        self.assertEqual([item["instance_index"] for item in plan], [1, 2, 1])
        for item in plan:
            self.assertNotIn("job_name", item)

    def test_build_run_plan_rejects_missing_group_instance_count(self):
        payload = self.payload()

        with self.assertRaisesRegex(ValueError, "not enough layouts.*group 2"):
            roi.build_run_plan(payload, instance_count={1: 1, 2: 1}, warm_start=False)

    def test_uniform_reference_holes_are_four_by_six_grid(self):
        holes = roi.load_reference_holes("uniform")

        self.assertEqual(len(holes), roi.HOLE_COUNT)
        roi.validate_holes(holes)
        x_values = sorted(set(round(hole["x"], 6) for hole in holes))
        y_values = sorted(set(round(hole["y"], 6) for hole in holes))
        expected_x = [
            round(value, 6)
            for value in roi.linspace(
                roi.MIN_CENTER_TO_EDGE,
                roi.PLATE_X - roi.MIN_CENTER_TO_EDGE,
                4,
            )
        ]
        expected_y = [
            round(value, 6)
            for value in roi.linspace(
                roi.MIN_CENTER_TO_EDGE,
                roi.PLATE_Y - roi.MIN_CENTER_TO_EDGE,
                6,
            )
        ]
        self.assertEqual(x_values, expected_x)
        self.assertEqual(y_values, expected_y)

    def test_reference_run_plan_names_outputs_like_legacy_script(self):
        old_inp_dir = roi.inp_dir
        try:
            roi.inp_dir = lambda: str(TEST_WORKDIR / "reference_names")
            solid_plan = roi.build_run_plan(ref="solid", instance_count=2)
            uniform_plan = roi.build_run_plan(ref="uniform", instance_count=2)
        finally:
            roi.inp_dir = old_inp_dir

        self.assertEqual(
            [item["inp_name"] for item in solid_plan],
            ["solid_1_plate.inp", "solid_2_plate.inp"],
        )
        self.assertEqual([item["holes"] for item in solid_plan], [[], []])
        self.assertEqual(
            [item["inp_name"] for item in uniform_plan],
            ["uniform_1_plate.inp", "uniform_2_plate.inp"],
        )
        for item in uniform_plan:
            self.assertEqual(item["ref"], "uniform")
            self.assertEqual(len(item["holes"]), roi.HOLE_COUNT)
            roi.validate_holes(item["holes"])

    def test_build_run_plan_rejects_unknown_reference(self):
        with self.assertRaisesRegex(ValueError, "unknown run plan reference"):
            roi.build_run_plan(ref="bad")

    def test_manifest_contains_legacy_group_and_runtime_sections(self):
        payload = self.payload()
        old_inp_dir = roi.inp_dir
        try:
            roi.inp_dir = lambda: str(TEST_WORKDIR / "manifest_sections")
            plan = roi.build_run_plan(payload, instance_count={1: 1}, warm_start=False)
        finally:
            roi.inp_dir = old_inp_dir

        manifest = roi.build_manifest(payload, plan, 1234.0, 0.8, 4.5)

        self.assertEqual(
            manifest["plate"],
            {"x": roi.PLATE_X, "y": roi.PLATE_Y, "thickness": roi.PLATE_THICKNESS},
        )
        self.assertEqual(manifest["material"], {"E": 1234.0, "nu": roi.DEFAULT_NU})
        self.assertEqual(manifest["load"], {"u": 0.8})
        self.assertEqual(manifest["mesh_size"], 4.5)
        self.assertEqual(manifest["mesh_size_default"], roi.DEFAULT_MESH_SIZE)
        self.assertEqual(manifest["constraints"]["hole_count"], roi.HOLE_COUNT)
        self.assertEqual(len(manifest["groups"]), len(roi.GROUP_DEFINITIONS))
        self.assertEqual(len(manifest["sampling_domains"]), len(roi.GROUP_DEFINITIONS))
        self.assertEqual(manifest["runs"][0]["group_id"], 1)
        self.assertEqual(manifest["runs"][0]["direction"], "x")

    def test_main_uses_editable_parameters_without_cli_args(self):
        payload_path = self.write_payload(self.payload(), "roi_main_direct.json")
        target_dir = TEST_WORKDIR / ("roi_main_direct_%d" % id(self))
        calls = []
        old_values = {
            "SELECTED_LAYOUTS_JSON": roi.SELECTED_LAYOUTS_JSON,
            "E": roi.E,
            "U": roi.U,
            "MESH_SIZE": roi.MESH_SIZE,
            "GROUP_INSTANCE_COUNTS": roi.GROUP_INSTANCE_COUNTS,
            "INSTANCE": roi.INSTANCE,
            "REF": roi.REF,
            "WARM_START": roi.WARM_START,
            "write_inp_with_abaqus": roi.write_inp_with_abaqus,
            "inp_dir": roi.inp_dir,
        }

        def fake_write_inp(run_item, material_e, displacement_u, mesh_size):
            calls.append((run_item["inp_name"], material_e, displacement_u, mesh_size))
            return run_item["inp_path"]

        try:
            roi.SELECTED_LAYOUTS_JSON = str(payload_path)
            roi.E = 3456.0
            roi.U = 1.2
            roi.MESH_SIZE = 3.75
            roi.GROUP_INSTANCE_COUNTS = {1: 1}
            roi.REF = None
            roi.WARM_START = False
            roi.write_inp_with_abaqus = fake_write_inp
            roi.inp_dir = lambda: str(target_dir)

            manifest_path = roi.main()
        finally:
            for name, value in old_values.items():
                setattr(roi, name, value)

        self.assertSamePath(manifest_path, target_dir / "group_manifest.json")
        self.assertEqual(calls, [("1_1_plate.inp", 3456.0, 1.2, 3.75)])
        with open(str(manifest_path), "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        self.assertEqual([run["inp_name"] for run in manifest["runs"]], ["1_1_plate.inp"])

    def test_main_writes_reference_inputs_without_layout_payload_or_manifest(self):
        target_dir = TEST_WORKDIR / ("roi_main_ref_%d" % id(self))
        calls = []
        old_values = {
            "SELECTED_LAYOUTS_JSON": roi.SELECTED_LAYOUTS_JSON,
            "E": roi.E,
            "U": roi.U,
            "MESH_SIZE": roi.MESH_SIZE,
            "INSTANCE": roi.INSTANCE,
            "REF": roi.REF,
            "WARM_START": roi.WARM_START,
            "write_inp_with_abaqus": roi.write_inp_with_abaqus,
            "inp_dir": roi.inp_dir,
        }

        def fake_write_inp(run_item, material_e, displacement_u, mesh_size):
            calls.append((run_item["inp_name"], len(run_item["holes"]), material_e, displacement_u, mesh_size))
            return run_item["inp_path"]

        try:
            roi.SELECTED_LAYOUTS_JSON = str(TEST_WORKDIR / "missing_selected_layouts.json")
            roi.E = 3456.0
            roi.U = 1.2
            roi.MESH_SIZE = 3.75
            roi.INSTANCE = 2
            roi.REF = "uniform"
            roi.WARM_START = True
            roi.write_inp_with_abaqus = fake_write_inp
            roi.inp_dir = lambda: str(target_dir)

            result = roi.main()
        finally:
            for name, value in old_values.items():
                setattr(roi, name, value)

        self.assertIsNone(result)
        self.assertEqual(calls, [
            ("uniform_1_plate.inp", roi.HOLE_COUNT, 3456.0, 1.2, 3.75),
            ("uniform_2_plate.inp", roi.HOLE_COUNT, 3456.0, 1.2, 3.75),
        ])
        self.assertFalse((target_dir / "group_manifest.json").exists())

    def test_abaqus_job_name_uses_final_inp_path_like_legacy_generator(self):
        self.assertEqual(
            roi.abaqus_job_name_from_inp_path(os.path.join("x", "1_1_plate.inp")),
            "job_1_1_plate",
        )
        self.assertEqual(
            roi.abaqus_job_name_from_inp_path(os.path.join("x", "roi_1_plate.inp")),
            "roi_1_plate",
        )

    def test_write_inp_with_fake_abaqus_derives_model_name_from_inp_path(self):
        fake_mdb, old_modules = self.install_fake_abaqus()
        old_cwd = Path.cwd()
        target_dir = TEST_WORKDIR / ("roi_fake_writer_%d" % id(self))
        target_path = target_dir / "1_1_plate.inp"
        run_item = {
            "inp_name": "1_1_plate.inp",
            "inp_path": str(target_path),
            "holes": [{"x": 20.0, "y": 30.0, "r": roi.HOLE_RADIUS}],
        }

        try:
            roi.write_inp_with_abaqus(run_item, 1234.0, 0.8, 4.5)
        finally:
            self.restore_fake_abaqus(old_modules)
            os.chdir(str(old_cwd))

        self.assertTrue(target_path.is_file())
        self.assertEqual(target_path.read_text(encoding="utf-8"), "fake input for job_1_1_plate")
        self.assertEqual(fake_mdb.last_job_name, "job_1_1_plate")
        self.assertEqual(fake_mdb.last_model_name, "Model_job_1_1_plate")

    def test_load_layout_payload_rejects_geometry_mismatch(self):
        payload = self.payload()
        payload["geometry"]["plate_x"] = roi.PLATE_X + 1.0
        path = self.write_payload(payload, "roi_geometry_mismatch.json")

        with self.assertRaisesRegex(ValueError, "geometry mismatch.*plate_x"):
            roi.load_layout_payload(str(path))

    def test_load_layout_payload_accepts_matching_geometry(self):
        path = self.write_payload(self.payload(), "roi_geometry_valid.json")

        loaded = roi.load_layout_payload(str(path))

        self.assertEqual(loaded["schema"], "roi_indicator_stratified_layouts_v1")
        self.assertEqual(len(loaded["layouts"]), 1)

    def test_load_layout_payload_accepts_utf8_bom_json(self):
        path = TEST_WORKDIR / "roi_geometry_valid_bom.json"
        content = json.dumps(self.payload()).encode("utf-8")
        with open(str(path), "wb") as json_file:
            json_file.write(codecs.BOM_UTF8)
            json_file.write(content)

        loaded = roi.load_layout_payload(str(path))

        self.assertEqual(loaded["schema"], "roi_indicator_stratified_layouts_v1")
        self.assertEqual(len(loaded["layouts"]), 1)


if __name__ == "__main__":
    unittest.main()
