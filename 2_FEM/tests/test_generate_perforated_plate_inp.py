import os
import sys
import io
import json
import types
import unittest
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = FEM_ROOT / "scripts"
TEST_WORKDIR = FEM_ROOT / "temp" / "tests"
ORIGINAL_CWD = Path.cwd()

TEST_WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(TEST_WORKDIR)
sys.path.insert(0, str(SCRIPT_DIR))

import generate_perforated_plate_inp as fem


def tearDownModule():
    os.chdir(ORIGINAL_CWD)
    try:
        sys.path.remove(str(SCRIPT_DIR))
    except ValueError:
        pass


class GeneratePerforatedPlateInpTests(unittest.TestCase):
    def assertSamePath(self, actual, expected):
        actual_path = os.path.normcase(os.path.normpath(actual))
        expected_path = os.path.normcase(os.path.normpath(str(expected)))
        self.assertEqual(actual_path, expected_path)

    def round_holes(self, holes):
        rounded = []
        for hole in holes:
            rounded.append((round(hole["x"], 6), round(hole["y"], 6)))
        return rounded

    def valid_holes(self):
        holes = []
        for row in range(8):
            for col in range(5):
                holes.append({
                    "x": 15.0 + 30.0 * col,
                    "y": 15.0 + 24.0 * row,
                    "r": fem.HOLE_RADIUS,
                })
        self.assertEqual(len(holes), fem.HOLE_COUNT)
        return holes

    def script_path_resolver(self):
        resolver = getattr(fem, "_resolve_script_path", None)
        self.assertIsNotNone(resolver, "_resolve_script_path helper missing")
        return resolver

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

        class FakeMaterial(object):
            def __init__(self, name):
                self.name = name
                self.elastic_table = None

            def Elastic(self, table):
                self.elastic_table = table

        class FakePart(object):
            def __init__(self, name, dimensionality, type):
                self.name = name
                self.dimensionality = dimensionality
                self.type = type
                self.faces = FakeCollection(["face"])
                self.edges = FakeCollection(["edge"])
                self.vertices = FakeCollection(["vertex"])
                self.sets = {}
                self.base_shell_sketch = None
                self.section_assignments = []
                self.seed_size = None
                self.mesh_controls = []
                self.element_types = []
                self.mesh_generated = False

            def BaseShell(self, sketch):
                self.base_shell_sketch = sketch

            def Set(self, **kwargs):
                name = kwargs["name"]
                self.sets[name] = kwargs
                return kwargs

            def SectionAssignment(self, region, sectionName):
                self.section_assignments.append((region, sectionName))

            def seedPart(self, size, **_kwargs):
                self.seed_size = size

            def setMeshControls(self, regions, elemShape):
                self.mesh_controls.append((regions, elemShape))

            def setElementType(self, regions, elemTypes):
                self.element_types.append((regions, elemTypes))

            def generateMesh(self):
                self.mesh_generated = True

        class FakeInstance(object):
            def __init__(self, name, part, dependent):
                self.name = name
                self.part = part
                self.dependent = dependent
                self.sets = part.sets

        class FakeAssembly(object):
            def __init__(self):
                self.instances = {}

            def Instance(self, name, part, dependent):
                instance = FakeInstance(name, part, dependent)
                self.instances[name] = instance
                return instance

        class FakeModel(object):
            def __init__(self, name):
                self.name = name
                self.rootAssembly = FakeAssembly()
                self.sketches = {}
                self.parts = {}
                self.materials = {}
                self.sections = {}
                self.steps = []
                self.boundary_conditions = []
                self.fieldOutputRequests = {"F-Output-1": FakeOutputRequest("F-Output-1")}

            def ConstrainedSketch(self, name, sheetSize):
                sketch = FakeSketch()
                sketch.name = name
                sketch.sheetSize = sheetSize
                self.sketches[name] = sketch
                return sketch

            def Part(self, name, dimensionality, type):
                part = FakePart(name, dimensionality, type)
                self.parts[name] = part
                return part

            def Material(self, name):
                material = FakeMaterial(name)
                self.materials[name] = material
                return material

            def HomogeneousSolidSection(self, name, material, thickness):
                self.sections[name] = {
                    "material": material,
                    "thickness": thickness,
                }

            def StaticStep(self, name, previous):
                self.steps.append((name, previous))

            def DisplacementBC(self, **kwargs):
                self.boundary_conditions.append(kwargs)

        class FakeOutputRequest(object):
            def __init__(self, name):
                self.name = name
                self.variables = None

            def setValues(self, variables):
                self.variables = variables

        class FakeJob(object):
            def __init__(self, name, model):
                self.name = name
                self.model = model
                self.write_consistency = None

            def writeInput(self, consistencyChecking):
                self.write_consistency = consistencyChecking
                with open(self.name + ".inp", "w", encoding="utf-8") as inp_file:
                    inp_file.write("fake input for %s" % self.name)

        class FakeMdb(object):
            def __init__(self):
                self.models = {}
                self.jobs = {}
                self.last_model = None
                self.last_job = None

            def Model(self, name):
                model = FakeModel(name)
                self.models[name] = model
                self.last_model = model
                return model

            def Job(self, name, model):
                job = FakeJob(name, model)
                self.jobs[name] = job
                self.last_job = job
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

    def test_plate_material_load_and_mesh_constants_match_defaults(self):
        expected_constants = {
            "PLATE_X": 150.0,
            "PLATE_Y": 200.0,
            "PLATE_THICKNESS": 2.0,
            "HOLE_RADIUS": 5.0,
            "HOLE_COUNT": 40,
            "MIN_CENTER_DISTANCE": 14.0,
            "MIN_CENTER_TO_EDGE": 10.0,
            "DEFAULT_E": 2100.0,
            "DEFAULT_NU": 0.33,
            "DEFAULT_MESH_SIZE": 2.5,
        }
        for name, expected in expected_constants.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(getattr(fem, name), expected)
        self.assertGreater(fem.DEFAULT_U, 0.0)

    def test_editable_runtime_parameters_are_valid_and_self_consistent(self):
        self.assertAlmostEqual(fem.E, fem.DEFAULT_E)
        self.assertAlmostEqual(fem.U, fem.DEFAULT_U)
        self.assertAlmostEqual(fem.MESH_SIZE, fem.DEFAULT_MESH_SIZE)
        self.assertGreaterEqual(fem.INSTANCE, 0)
        self.assertIn(fem.REF, fem.VALID_REFS)
        normalized_counts = fem._normalize_group_instance_counts(fem.GROUP_INSTANCE_COUNTS)
        self.assertEqual(
            sorted(normalized_counts.keys()),
            [group["id"] for group in fem.GROUP_DEFINITIONS],
        )
        for count in normalized_counts.values():
            self.assertGreaterEqual(count, 0)
        self.assertEqual(fem.VALID_REFS, (None, "solid", "transverse", "longitudinal"))

    def test_hole_constraints_keep_centers_inside_plate_and_apart(self):
        self.assertEqual(fem.HOLE_COUNT, 40)
        self.assertGreater(fem.MIN_CENTER_DISTANCE, 2.0 * fem.HOLE_RADIUS)
        self.assertGreaterEqual(fem.MIN_CENTER_TO_EDGE, fem.HOLE_RADIUS)
        self.assertLessEqual(2.0 * fem.MIN_CENTER_TO_EDGE, fem.PLATE_X)
        self.assertLessEqual(2.0 * fem.MIN_CENTER_TO_EDGE, fem.PLATE_Y)

    def test_group_definitions_are_ordered_and_bounded(self):
        self.assertAlmostEqual(fem.DEFAULT_MESH_SIZE, 2.5)
        groups = fem.GROUP_DEFINITIONS
        self.assertEqual([group["id"] for group in groups], list(range(1, 10)))
        self.assertEqual(groups[0]["cluster"], "low")
        self.assertEqual(groups[0]["direction"], "x")
        self.assertEqual(groups[8]["cluster"], "high")
        self.assertEqual(groups[8]["direction"], "y")
        for group in groups:
            xmin, xmax = group["x_range"]
            ymin, ymax = group["y_range"]
            self.assertGreaterEqual(xmin, fem.MIN_CENTER_TO_EDGE)
            self.assertLessEqual(xmax, fem.PLATE_X - fem.MIN_CENTER_TO_EDGE)
            self.assertGreaterEqual(ymin, fem.MIN_CENTER_TO_EDGE)
            self.assertLessEqual(ymax, fem.PLATE_Y - fem.MIN_CENTER_TO_EDGE)

    def test_uniform_groups_keep_plate_aspect_ratio(self):
        expected_ranges = {
            2: ((17.5, 132.5), (23.33333333333333, 176.66666666666669)),
            5: ((25.0, 125.0), (33.33333333333333, 166.66666666666669)),
            8: ((31.25, 118.75), (41.66666666666667, 158.33333333333331)),
        }
        group_areas = {}
        for group in fem.GROUP_DEFINITIONS:
            x_span = group["x_range"][1] - group["x_range"][0]
            y_span = group["y_range"][1] - group["y_range"][0]
            group_areas[group["id"]] = x_span * y_span
            if group["direction"] != "none":
                continue
            expected_x, expected_y = expected_ranges[group["id"]]
            self.assertEqual(group["x_range"], expected_x)
            self.assertAlmostEqual(group["y_range"][0], expected_y[0])
            self.assertAlmostEqual(group["y_range"][1], expected_y[1])
            self.assertAlmostEqual(x_span / y_span, fem.PLATE_X / fem.PLATE_Y)
        self.assertGreater(group_areas[2], group_areas[1])
        self.assertGreater(group_areas[1], group_areas[5])
        self.assertGreater(group_areas[5], group_areas[4])
        self.assertGreater(group_areas[4], group_areas[8])
        self.assertGreater(group_areas[8], group_areas[7])

    def test_directory_helpers_return_expected_paths(self):
        self.assertSamePath(fem.project_root(), FEM_ROOT.parent)
        self.assertSamePath(fem.fem_root(), FEM_ROOT)
        self.assertSamePath(fem.temp_dir(), FEM_ROOT / "temp")
        self.assertSamePath(fem.inp_dir(), FEM_ROOT / "temp" / "solve_inp")
        self.assertSamePath(fem.test_work_dir(), TEST_WORKDIR)

        cwd = os.getcwd()
        try:
            self.assertSamePath(fem.enter_temp_work_dir(), FEM_ROOT / "temp")
            self.assertSamePath(os.getcwd(), FEM_ROOT / "temp")
        finally:
            os.chdir(cwd)

    def test_resolve_script_path_prefers_explicit_file(self):
        resolver = self.script_path_resolver()
        script_path = SCRIPT_DIR / "generate_perforated_plate_inp.py"

        resolved = resolver(str(script_path), ["ignored"], str(FEM_ROOT.parent))

        self.assertSamePath(resolved, script_path)

    def test_resolve_script_path_uses_abaqus_nogui_argv_without_file(self):
        resolver = self.script_path_resolver()
        script_path = SCRIPT_DIR / "generate_perforated_plate_inp.py"
        argv = [
            "abaqus",
            "cae",
            "noGUI=2_FEM\\scripts\\generate_perforated_plate_inp.py",
        ]

        resolved = resolver(None, argv, str(FEM_ROOT.parent))

        self.assertSamePath(resolved, script_path)

    def test_resolve_script_path_supports_temp_as_abaqus_working_directory(self):
        resolver = self.script_path_resolver()
        script_path = SCRIPT_DIR / "generate_perforated_plate_inp.py"

        resolved = resolver(None, [], str(FEM_ROOT / "temp"))

        self.assertSamePath(resolved, script_path)

    def test_resolve_relative_nogui_path_from_temp_working_directory(self):
        resolver = self.script_path_resolver()
        script_path = SCRIPT_DIR / "generate_perforated_plate_inp.py"
        argv = [
            "abaqus",
            "cae",
            "noGUI=2_FEM\\scripts\\generate_perforated_plate_inp.py",
        ]

        resolved = resolver(None, argv, str(FEM_ROOT / "temp"))

        self.assertSamePath(resolved, script_path)

    def test_directory_helpers_ignore_later_cwd_and_file_path_changes(self):
        old_file = fem.__file__
        old_cwd = os.getcwd()
        try:
            fem.__file__ = os.path.join(
                "2_FEM", "scripts", "generate_perforated_plate_inp.py"
            )
            os.chdir(str(FEM_ROOT / "temp"))
            self.assertSamePath(fem.project_root(), FEM_ROOT.parent)
            self.assertSamePath(fem.fem_root(), FEM_ROOT)
            self.assertSamePath(fem.temp_dir(), FEM_ROOT / "temp")
            self.assertSamePath(fem.inp_dir(), FEM_ROOT / "temp" / "solve_inp")
        finally:
            fem.__file__ = old_file
            os.chdir(old_cwd)

    def test_validate_holes_accepts_valid_forty_hole_layout(self):
        self.assertIsNone(fem.validate_holes(self.valid_holes()))

    def test_validate_holes_rejects_wrong_hole_count(self):
        with self.assertRaises(ValueError):
            fem.validate_holes(self.valid_holes()[:-1])

    def test_validate_holes_rejects_center_outside_global_bounds(self):
        holes = self.valid_holes()
        holes[0]["x"] = 9.9
        with self.assertRaises(ValueError):
            fem.validate_holes(holes)

    def test_validate_holes_rejects_centers_closer_than_minimum_distance(self):
        holes = self.valid_holes()
        holes[1]["x"] = holes[0]["x"] + fem.MIN_CENTER_DISTANCE - 0.1
        holes[1]["y"] = holes[0]["y"]
        with self.assertRaises(ValueError):
            fem.validate_holes(holes)

    def test_reference_coordinates_swap_x_and_y_without_scaling(self):
        holes = fem.load_reference_holes("transverse")
        self.assertEqual(len(holes), fem.HOLE_COUNT)
        self.assertAlmostEqual(holes[0]["x"], 31.0)
        self.assertAlmostEqual(holes[0]["y"], 23.0)
        self.assertAlmostEqual(holes[0]["r"], fem.HOLE_RADIUS)
        fem.validate_holes(holes)

    def test_longitudinal_reference_coordinates_swap_x_and_y_without_scaling(self):
        holes = fem.load_reference_holes("longitudinal")
        self.assertEqual(len(holes), fem.HOLE_COUNT)
        self.assertAlmostEqual(holes[0]["x"], 31.0)
        self.assertAlmostEqual(holes[0]["y"], 34.0)
        self.assertAlmostEqual(holes[0]["r"], fem.HOLE_RADIUS)
        fem.validate_holes(holes)

    def test_random_generation_is_deterministic_per_group_and_instance(self):
        first = fem.generate_holes_for_group(fem.GROUP_DEFINITIONS[0], 1)
        second = fem.generate_holes_for_group(fem.GROUP_DEFINITIONS[0], 1)
        self.assertEqual(self.round_holes(first), self.round_holes(second))
        self.assertEqual(len(first), fem.HOLE_COUNT)
        fem.validate_holes(first)

    def test_random_generation_creates_valid_layout_for_each_group(self):
        for group in fem.GROUP_DEFINITIONS:
            with self.subTest(group_id=group["id"]):
                holes = fem.generate_holes_for_group(group, 1)
                self.assertEqual(len(holes), fem.HOLE_COUNT)
                fem.validate_holes(holes)

    def test_random_generation_uses_instance_in_seed(self):
        first = fem.generate_holes_for_group(fem.GROUP_DEFINITIONS[0], 1)
        second = fem.generate_holes_for_group(fem.GROUP_DEFINITIONS[0], 2)
        self.assertNotEqual(self.round_holes(first), self.round_holes(second))

    def test_random_run_plan_names_outputs_by_group_and_instance(self):
        plan = fem.build_run_plan(ref=None, instance_count=2)
        names = [item["inp_name"] for item in plan]
        self.assertIn("1_1_plate.inp", names)
        self.assertIn("9_2_plate.inp", names)
        self.assertEqual(len(names), 18)

    def test_random_run_plan_accepts_per_group_instance_counts(self):
        plan = fem.build_run_plan(ref=None, instance_count={1: 2, 4: 1})
        names = [item["inp_name"] for item in plan]
        self.assertEqual(names, ["1_1_plate.inp", "1_2_plate.inp", "4_1_plate.inp"])
        self.assertEqual([item["group_id"] for item in plan], [1, 1, 4])
        self.assertEqual([item["instance_index"] for item in plan], [1, 2, 1])
        for item in plan:
            fem.validate_holes(item["holes"])

    def test_random_run_plan_warm_start_continues_after_existing_group_indices(self):
        old_inp_dir = fem.inp_dir
        target_dir = TEST_WORKDIR / ("warm_start_%d" % id(self))
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "1_1_plate.inp",
            "1_3_plate.inp",
            "4_7_plate.inp",
            "solid_99_plate.inp",
            "1_bad_plate.inp",
        ]:
            (target_dir / name).write_text("", encoding="utf-8")

        try:
            fem.inp_dir = lambda: str(target_dir)
            plan = fem.build_run_plan(
                ref=None,
                instance_count={1: 2, 2: 0, 4: 1},
                warm_start=True,
            )
        finally:
            fem.inp_dir = old_inp_dir

        names = [item["inp_name"] for item in plan]
        self.assertEqual(names, ["1_4_plate.inp", "1_5_plate.inp", "4_8_plate.inp"])
        self.assertEqual([item["instance_index"] for item in plan], [4, 5, 8])
        self.assertTrue(plan[0]["seed_text"].startswith("1:4:"))
        self.assertTrue(plan[2]["seed_text"].startswith("4:8:"))

    def test_random_run_plan_items_use_absolute_inp_paths_and_valid_holes(self):
        plan = fem.build_run_plan(ref=None, instance_count=1)
        inp_root = os.path.abspath(fem.inp_dir())
        self.assertEqual(len(plan), len(fem.GROUP_DEFINITIONS))
        for item in plan:
            with self.subTest(inp_name=item["inp_name"]):
                self.assertTrue(os.path.isabs(item["inp_path"]))
                self.assertSamePath(
                    item["inp_path"], os.path.join(inp_root, item["inp_name"])
                )
                self.assertEqual(len(item["holes"]), fem.HOLE_COUNT)
                fem.validate_holes(item["holes"])
                self.assertIn("seed", item)
                self.assertIn("restart_index", item)

    def test_solid_run_plan_names_outputs_and_uses_empty_holes(self):
        plan = fem.build_run_plan(ref="solid", instance_count=2)
        names = [item["inp_name"] for item in plan]
        self.assertEqual(names, ["solid_1_plate.inp", "solid_2_plate.inp"])
        for item in plan:
            self.assertEqual(item["holes"], [])
            self.assertTrue(os.path.isabs(item["inp_path"]))
            self.assertSamePath(
                item["inp_path"], os.path.join(fem.inp_dir(), item["inp_name"])
            )

    def test_transverse_run_plan_loads_swapped_reference_holes(self):
        plan = fem.build_run_plan(ref="transverse", instance_count=1)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["inp_name"], "transverse_1_plate.inp")
        self.assertEqual(len(plan[0]["holes"]), fem.HOLE_COUNT)
        self.assertAlmostEqual(plan[0]["holes"][0]["x"], 31.0)
        self.assertAlmostEqual(plan[0]["holes"][0]["y"], 23.0)
        fem.validate_holes(plan[0]["holes"])

    def test_longitudinal_run_plan_loads_swapped_reference_holes(self):
        plan = fem.build_run_plan(ref="longitudinal", instance_count=1)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["inp_name"], "longitudinal_1_plate.inp")
        self.assertEqual(len(plan[0]["holes"]), fem.HOLE_COUNT)
        self.assertAlmostEqual(plan[0]["holes"][0]["x"], 31.0)
        self.assertAlmostEqual(plan[0]["holes"][0]["y"], 34.0)
        fem.validate_holes(plan[0]["holes"])

    def test_manifest_contains_defaults_mappings_runs_and_random_seeds(self):
        plan = fem.build_run_plan(ref=None, instance_count=1)
        manifest = fem.build_manifest(plan)
        json.dumps(manifest, sort_keys=True)

        self.assertEqual(manifest["mesh_size_default"], 2.5)
        self.assertEqual(manifest["element_type"], "CPS6")
        self.assertEqual(
            manifest["plate"],
            {"x": fem.PLATE_X, "y": fem.PLATE_Y, "thickness": fem.PLATE_THICKNESS},
        )
        self.assertEqual(
            manifest["material"],
            {"E": fem.DEFAULT_E, "nu": fem.DEFAULT_NU},
        )
        self.assertEqual(manifest["load"], {"u": fem.DEFAULT_U})
        self.assertEqual(manifest["constraints"]["hole_count"], fem.HOLE_COUNT)
        self.assertEqual(
            manifest["constraints"]["min_center_distance"],
            fem.MIN_CENTER_DISTANCE,
        )
        self.assertEqual(len(manifest["groups"]), len(fem.GROUP_DEFINITIONS))
        self.assertEqual(len(manifest["sampling_domains"]), len(fem.GROUP_DEFINITIONS))
        self.assertIn("transverse", manifest["references"])
        self.assertIn("longitudinal", manifest["references"])
        self.assertEqual(len(manifest["runs"]), len(plan))
        self.assertEqual(len(manifest["seeds"]), len(plan))
        for run, seed in zip(manifest["runs"], manifest["seeds"]):
            self.assertIn("inp_name", run)
            self.assertIn("seed", seed)
            self.assertEqual(run["inp_name"], seed["inp_name"])

    def test_manifest_records_runtime_material_load_and_mesh_values(self):
        plan = [{
            "ref": "solid",
            "instance_index": 1,
            "holes": [],
            "inp_name": "solid_1_plate.inp",
            "inp_path": os.path.join(fem.inp_dir(), "solid_1_plate.inp"),
        }]

        manifest = fem.build_manifest(
            plan,
            material_e=1234.0,
            displacement_u=0.42,
            mesh_size=6.5,
        )

        self.assertEqual(manifest["material"], {"E": 1234.0, "nu": fem.DEFAULT_NU})
        self.assertEqual(manifest["load"], {"u": 0.42})
        self.assertEqual(manifest["mesh_size"], 6.5)
        self.assertEqual(
            manifest["defaults"],
            {
                "material_e": fem.DEFAULT_E,
                "displacement_u": fem.DEFAULT_U,
                "mesh_size": fem.DEFAULT_MESH_SIZE,
            },
        )

    def test_ensure_directory_and_write_json_file_create_utf8_json(self):
        target_dir = TEST_WORKDIR / ("json_helper_%d" % id(self))
        target_path = target_dir / "manifest.json"

        fem.ensure_directory(str(target_dir))
        fem.write_json_file(str(target_path), {"b": 2, "a": "孔"})

        self.assertTrue(target_dir.is_dir())
        with open(str(target_path), "r", encoding="utf-8") as manifest_file:
            text = manifest_file.read()
        self.assertLess(text.find('"a"'), text.find('"b"'))
        self.assertEqual(json.loads(text), {"a": "孔", "b": 2})

    def test_write_json_file_decodes_python2_json_bytes_for_utf8_writer(self):
        old_dump = fem.json.dump
        old_dumps = fem.json.dumps
        old_version_info = fem.sys.version_info
        target_dir = TEST_WORKDIR / ("json_py2_helper_%d" % id(self))
        target_path = target_dir / "manifest.json"

        def fake_dump(_data, json_file, indent, sort_keys):
            json_file.write(b'{"a": 1}')

        def fake_dumps(_data, indent, sort_keys):
            return b'{"a": 1}'

        try:
            fem.json.dump = fake_dump
            fem.json.dumps = fake_dumps
            fem.sys.version_info = (2, 7, 18)
            fem.write_json_file(str(target_path), {"a": 1})
        finally:
            fem.json.dump = old_dump
            fem.json.dumps = old_dumps
            fem.sys.version_info = old_version_info

        with open(str(target_path), "r", encoding="utf-8") as manifest_file:
            self.assertEqual(manifest_file.read(), '{"a": 1}\n')

    def test_abaqus_job_name_uses_stem_and_prefixes_numeric_stems(self):
        self.assertEqual(
            fem.abaqus_job_name_from_inp_path(os.path.join("x", "1_1_plate.inp")),
            "job_1_1_plate",
        )
        self.assertEqual(
            fem.abaqus_job_name_from_inp_path(os.path.join("x", "solid_1_plate.inp")),
            "solid_1_plate",
        )

    def test_write_inp_with_fake_abaqus_builds_plate_and_moves_inp(self):
        fake_mdb, old_modules = self.install_fake_abaqus()
        old_cwd = os.getcwd()
        target_dir = TEST_WORKDIR / ("fake_writer_%d" % id(self))
        target_path = target_dir / "1_1_plate.inp"
        run_item = {
            "inp_name": "1_1_plate.inp",
            "inp_path": str(target_path),
            "holes": [{"x": 25.0, "y": 35.0, "r": fem.HOLE_RADIUS}],
        }

        try:
            fem.write_inp_with_abaqus(run_item, 1234.0, 0.8, 4.5)
        finally:
            self.restore_fake_abaqus(old_modules)
            os.chdir(old_cwd)

        self.assertTrue(target_path.is_file())
        self.assertEqual(target_path.read_text(encoding="utf-8"), "fake input for job_1_1_plate")

        model = fake_mdb.last_model
        part = model.parts["Plate"]
        sketch = part.base_shell_sketch
        self.assertEqual(sketch.rectangles, [((0.0, 0.0), (fem.PLATE_X, fem.PLATE_Y))])
        self.assertEqual(
            sketch.circles,
            [((25.0, 35.0), (25.0 + fem.HOLE_RADIUS, 35.0))],
        )
        self.assertEqual(model.materials["PlateMaterial"].elastic_table, ((1234.0, fem.DEFAULT_NU),))
        self.assertEqual(model.sections["PlateSection"]["thickness"], fem.PLATE_THICKNESS)
        self.assertEqual(model.steps, [("Load", "Initial")])
        self.assertIn("E", model.fieldOutputRequests["F-Output-1"].variables)
        self.assertIn("RF", model.fieldOutputRequests["F-Output-1"].variables)
        self.assertEqual(part.seed_size, 4.5)
        self.assertTrue(part.mesh_generated)
        self.assertEqual(fake_mdb.last_job.write_consistency, "OFF")

        bc_names = [bc["name"] for bc in model.boundary_conditions]
        self.assertEqual(bc_names, ["BottomUY", "TopUY", "FixLowerLeftUX"])
        self.assertEqual(model.boundary_conditions[0]["u2"], -0.4)
        self.assertEqual(model.boundary_conditions[1]["u2"], 0.4)
        self.assertEqual(model.boundary_conditions[2]["createStepName"], "Load")
        self.assertEqual(model.boundary_conditions[2]["u1"], 0.0)

    def test_main_writes_solid_inputs_without_updating_manifest(self):
        old_values = {
            "E": fem.E,
            "U": fem.U,
            "MESH_SIZE": fem.MESH_SIZE,
            "INSTANCE": fem.INSTANCE,
            "REF": fem.REF,
            "build_run_plan": fem.build_run_plan,
            "write_inp_with_abaqus": fem.write_inp_with_abaqus,
            "inp_dir": fem.inp_dir,
        }
        target_dir = TEST_WORKDIR / ("main_inp_%d" % id(self))
        plan = [
            {
                "ref": "solid",
                "instance_index": 1,
                "holes": [],
                "inp_name": "solid_1_plate.inp",
                "inp_path": str(target_dir / "solid_1_plate.inp"),
            },
            {
                "ref": "solid",
                "instance_index": 2,
                "holes": [],
                "inp_name": "solid_2_plate.inp",
                "inp_path": str(target_dir / "solid_2_plate.inp"),
            },
        ]
        calls = []
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "group_manifest.json"
        manifest_path.write_text('{"existing": true}\n', encoding="utf-8")

        def fake_build_run_plan(ref=None, instance_count=1, warm_start=False):
            self.assertEqual(ref, "solid")
            self.assertEqual(instance_count, 2)
            self.assertFalse(warm_start)
            return plan

        def fake_write_inp(run_item, material_e, displacement_u, mesh_size):
            calls.append((run_item["inp_name"], material_e, displacement_u, mesh_size))
            return run_item["inp_path"]

        old_stdout = sys.stdout
        stdout = io.StringIO()
        try:
            fem.E = 3456.0
            fem.U = 1.2
            fem.MESH_SIZE = 3.75
            fem.INSTANCE = 2
            fem.REF = "solid"
            fem.build_run_plan = fake_build_run_plan
            fem.write_inp_with_abaqus = fake_write_inp
            fem.inp_dir = lambda: str(target_dir)

            sys.stdout = stdout
            result = fem.main()
        finally:
            sys.stdout = old_stdout
            for name, value in old_values.items():
                setattr(fem, name, value)

        self.assertIsNone(result)
        self.assertEqual(
            calls,
            [
                ("solid_1_plate.inp", 3456.0, 1.2, 3.75),
                ("solid_2_plate.inp", 3456.0, 1.2, 3.75),
            ],
        )
        self.assertIn("Saved inp:", stdout.getvalue())
        self.assertIn(str(target_dir / "solid_1_plate.inp"), stdout.getvalue())
        self.assertIn(str(target_dir / "solid_2_plate.inp"), stdout.getvalue())
        self.assertEqual(manifest_path.read_text(encoding="utf-8"), '{"existing": true}\n')

    def test_main_writes_reference_inputs_without_updating_manifest(self):
        old_values = {
            "E": fem.E,
            "U": fem.U,
            "MESH_SIZE": fem.MESH_SIZE,
            "INSTANCE": fem.INSTANCE,
            "REF": fem.REF,
            "build_run_plan": fem.build_run_plan,
            "write_inp_with_abaqus": fem.write_inp_with_abaqus,
            "inp_dir": fem.inp_dir,
        }
        target_dir = TEST_WORKDIR / ("main_ref_inp_%d" % id(self))
        plan = [
            {
                "ref": "transverse",
                "instance_index": 1,
                "holes": self.valid_holes(),
                "inp_name": "transverse_1_plate.inp",
                "inp_path": str(target_dir / "transverse_1_plate.inp"),
            },
        ]
        calls = []
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "group_manifest.json"
        manifest_path.write_text('{"existing": true}\n', encoding="utf-8")

        def fake_build_run_plan(ref=None, instance_count=1, warm_start=False):
            self.assertEqual(ref, "transverse")
            self.assertEqual(instance_count, 1)
            self.assertFalse(warm_start)
            return plan

        def fake_write_inp(run_item, material_e, displacement_u, mesh_size):
            calls.append((run_item["inp_name"], material_e, displacement_u, mesh_size))

        try:
            fem.E = 3456.0
            fem.U = 1.2
            fem.MESH_SIZE = 3.75
            fem.INSTANCE = 1
            fem.REF = "transverse"
            fem.build_run_plan = fake_build_run_plan
            fem.write_inp_with_abaqus = fake_write_inp
            fem.inp_dir = lambda: str(target_dir)

            result = fem.main()
        finally:
            for name, value in old_values.items():
                setattr(fem, name, value)

        self.assertIsNone(result)
        self.assertEqual(calls, [("transverse_1_plate.inp", 3456.0, 1.2, 3.75)])
        self.assertEqual(manifest_path.read_text(encoding="utf-8"), '{"existing": true}\n')

    def test_main_writes_random_inputs_and_manifest_with_editable_parameters(self):
        old_values = {
            "E": fem.E,
            "U": fem.U,
            "MESH_SIZE": fem.MESH_SIZE,
            "REF": fem.REF,
            "WARM_START": fem.WARM_START,
            "GROUP_INSTANCE_COUNTS": fem.GROUP_INSTANCE_COUNTS,
            "build_run_plan": fem.build_run_plan,
            "write_inp_with_abaqus": fem.write_inp_with_abaqus,
            "inp_dir": fem.inp_dir,
        }
        target_dir = TEST_WORKDIR / ("main_random_inp_%d" % id(self))
        plan = [
            {
                "ref": None,
                "group_id": 1,
                "cluster": "low",
                "direction": "x",
                "instance_index": 1,
                "holes": [],
                "inp_name": "1_1_plate.inp",
                "inp_path": str(target_dir / "1_1_plate.inp"),
                "seed": 123,
                "seed_text": "seed-text",
                "restart_index": 0,
            },
        ]
        calls = []

        def fake_build_run_plan(ref=None, instance_count=1, warm_start=False):
            self.assertIsNone(ref)
            self.assertEqual(instance_count, {1: 1})
            self.assertFalse(warm_start)
            return plan

        def fake_write_inp(run_item, material_e, displacement_u, mesh_size):
            calls.append((run_item["inp_name"], material_e, displacement_u, mesh_size))

        try:
            fem.E = 3456.0
            fem.U = 1.2
            fem.MESH_SIZE = 3.75
            fem.REF = None
            fem.WARM_START = False
            fem.GROUP_INSTANCE_COUNTS = {1: 1}
            fem.build_run_plan = fake_build_run_plan
            fem.write_inp_with_abaqus = fake_write_inp
            fem.inp_dir = lambda: str(target_dir)

            manifest_path = fem.main()
        finally:
            for name, value in old_values.items():
                setattr(fem, name, value)

        with open(str(manifest_path), "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        self.assertEqual(manifest["material"]["E"], 3456.0)
        self.assertEqual(manifest["load"]["u"], 1.2)
        self.assertEqual(manifest["mesh_size"], 3.75)
        self.assertEqual(calls, [("1_1_plate.inp", 3456.0, 1.2, 3.75)])
        self.assertEqual([run["inp_name"] for run in manifest["runs"]], ["1_1_plate.inp"])

    def test_main_uses_group_instance_counts_for_random_generation(self):
        old_values = {
            "INSTANCE": fem.INSTANCE,
            "REF": fem.REF,
            "WARM_START": fem.WARM_START,
            "GROUP_INSTANCE_COUNTS": fem.GROUP_INSTANCE_COUNTS,
            "build_run_plan": fem.build_run_plan,
            "write_inp_with_abaqus": fem.write_inp_with_abaqus,
            "inp_dir": fem.inp_dir,
        }
        target_dir = TEST_WORKDIR / ("main_group_counts_%d" % id(self))
        calls = []

        def fake_build_run_plan(ref=None, instance_count=1, warm_start=False):
            calls.append((ref, instance_count, warm_start))
            return []

        try:
            fem.INSTANCE = 999
            fem.REF = None
            fem.WARM_START = True
            fem.GROUP_INSTANCE_COUNTS = {1: 2, 4: 1}
            fem.build_run_plan = fake_build_run_plan
            fem.write_inp_with_abaqus = lambda *_args: None
            fem.inp_dir = lambda: str(target_dir)

            manifest_path = fem.main()
        finally:
            for name, value in old_values.items():
                setattr(fem, name, value)

        self.assertEqual(calls, [(None, {1: 2, 4: 1}, True)])
        self.assertSamePath(manifest_path, target_dir / "group_manifest.json")

    def test_main_warm_start_merges_existing_manifest_runs_and_seeds(self):
        old_values = {
            "REF": fem.REF,
            "WARM_START": fem.WARM_START,
            "GROUP_INSTANCE_COUNTS": fem.GROUP_INSTANCE_COUNTS,
            "build_run_plan": fem.build_run_plan,
            "write_inp_with_abaqus": fem.write_inp_with_abaqus,
            "inp_dir": fem.inp_dir,
        }
        target_dir = TEST_WORKDIR / ("main_warm_manifest_%d" % id(self))
        target_dir.mkdir(parents=True, exist_ok=True)
        existing_manifest = {
            "runs": [{
                "inp_name": "1_3_plate.inp",
                "group_id": 1,
                "instance_index": 3,
                "seed": 111,
            }],
            "seeds": [{
                "inp_name": "1_3_plate.inp",
                "group_id": 1,
                "instance_index": 3,
                "seed": 111,
                "seed_text": "1:3:0",
                "restart_index": 0,
            }],
        }
        with open(str(target_dir / "group_manifest.json"), "w", encoding="utf-8") as manifest_file:
            json.dump(existing_manifest, manifest_file)
        plan = [{
            "ref": None,
            "group_id": 1,
            "cluster": "low",
            "direction": "x",
            "instance_index": 4,
            "holes": self.valid_holes(),
            "inp_name": "1_4_plate.inp",
            "inp_path": str(target_dir / "1_4_plate.inp"),
            "seed": 222,
            "seed_text": "1:4:0",
            "restart_index": 0,
        }]

        def fake_build_run_plan(ref=None, instance_count=1, warm_start=False):
            self.assertIsNone(ref)
            self.assertEqual(instance_count, {1: 1})
            self.assertTrue(warm_start)
            return plan

        try:
            fem.REF = None
            fem.WARM_START = True
            fem.GROUP_INSTANCE_COUNTS = {1: 1}
            fem.build_run_plan = fake_build_run_plan
            fem.write_inp_with_abaqus = lambda *_args: None
            fem.inp_dir = lambda: str(target_dir)

            manifest_path = fem.main()
        finally:
            for name, value in old_values.items():
                setattr(fem, name, value)

        with open(str(manifest_path), "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        self.assertEqual(
            [run["inp_name"] for run in manifest["runs"]],
            ["1_3_plate.inp", "1_4_plate.inp"],
        )
        self.assertEqual(
            [seed["seed_text"] for seed in manifest["seeds"]],
            ["1:3:0", "1:4:0"],
        )
