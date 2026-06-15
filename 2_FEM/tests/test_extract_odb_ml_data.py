import io
import os
import sys
import tempfile
import unittest
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = FEM_ROOT / "scripts"
TEMP_DIR = FEM_ROOT / "temp"

sys.path.insert(0, str(SCRIPT_DIR))

import extract_odb_ml_data as extractor


def tearDownModule():
    try:
        sys.path.remove(str(SCRIPT_DIR))
    except ValueError:
        pass


class FakeFieldValue(object):
    def __init__(self, data, node_label=None, element_label=None, coordinates=None, max_principal=None):
        self.data = data
        if node_label is not None:
            self.nodeLabel = node_label
        if element_label is not None:
            self.elementLabel = element_label
        if coordinates is not None:
            self.coordinates = coordinates
        if max_principal is not None:
            self.maxPrincipal = max_principal


class FakeSubset(object):
    def __init__(self, values):
        self.values = values


class FakeFieldOutput(object):
    def __init__(self, values_by_region=None, values=None):
        self.values_by_region = values_by_region or {}
        self.values = values or []

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
        self.nodes = []
        self.elements = []


class FakeAssembly(object):
    def __init__(self):
        self.nodeSets = {}
        self.instances = {
            "PLATEINSTANCE": FakeInstance(),
        }


class FakeFrame(object):
    def __init__(self, strain_values=None):
        self.fieldOutputs = {
            "RF": FakeFieldOutput({
                "TOPEDGE": [
                    FakeFieldValue((0.0, 12.0), node_label=1),
                    FakeFieldValue((0.0, 18.0), node_label=2),
                ],
                "BOTTOMEDGE": [
                    FakeFieldValue((0.0, -14.0), node_label=3),
                    FakeFieldValue((0.0, -16.0), node_label=4),
                ],
            }),
            "U": FakeFieldOutput({
                "TOPEDGE": [
                    FakeFieldValue((0.0, 0.42), node_label=1),
                    FakeFieldValue((0.0, 0.38), node_label=2),
                ],
                "BOTTOMEDGE": [
                    FakeFieldValue((0.0, -0.41), node_label=3),
                    FakeFieldValue((0.0, -0.39), node_label=4),
                ],
            }),
            "E": FakeFieldOutput(values=strain_values or []),
        }


class FakeStep(object):
    def __init__(self, strain_values=None):
        self.frames = [FakeFrame(strain_values=strain_values)]


class FakeOdb(object):
    def __init__(self, strain_values=None):
        self.rootAssembly = FakeAssembly()
        self.steps = {"Load": FakeStep(strain_values=strain_values)}
        self.closed = False

    def close(self):
        self.closed = True


class FakeNode(object):
    def __init__(self, label, coordinates):
        self.label = label
        self.coordinates = coordinates


class FakeElement(object):
    def __init__(self, label, connectivity):
        self.label = label
        self.connectivity = connectivity


class ExtractOdbMlDataPathTests(unittest.TestCase):
    def test_default_constants(self):
        self.assertEqual(extractor.DEFAULT_STEP, "Load")
        self.assertEqual(extractor.DEFAULT_INSTANCE, "PlateInstance")
        self.assertEqual(extractor.DEFAULT_TOP_SET, "TopEdge")
        self.assertEqual(extractor.DEFAULT_BOTTOM_SET, "BottomEdge")
        self.assertEqual(extractor.DEFAULT_COMPONENT, 2)
        self.assertEqual(extractor.NEAR_HOLE_BAND_MM, 1.0)
        self.assertIsNone(extractor.REF)
        if extractor.GROUP_COUNT is not None:
            self.assertGreater(int(extractor.GROUP_COUNT), 0)
        if extractor.MAX_ODB_PER_RUN is not None:
            self.assertGreater(int(extractor.MAX_ODB_PER_RUN), 0)
        self.assertTrue(extractor.WARM_START)

    def test_default_paths_are_temp_relative(self):
        self.assertEqual(extractor.ODB_DIR, os.path.join("odb"))
        self.assertEqual(
            extractor.LAYOUT_FILE,
            os.path.join(
                "..",
                "results",
                "pilot_indicator_sampling",
                "seed_20260609_candidates_100000_target_200",
                "selected_layouts.json",
            ),
        )
        self.assertEqual(extractor.OUTPUT_DIR, os.path.join("..", "results", "odb_ml_data"))

    def test_format_progress_message_reports_count_percent_status_and_name(self):
        message = extractor.format_progress_message(2, 10, "odb/1_2_plate.odb", "processing")

        self.assertIn("[2/10", message)
        self.assertIn("20.0%", message)
        self.assertIn("processing", message)
        self.assertIn("1_2_plate.odb", message)

    def test_enter_temp_work_dir_can_print_resolved_working_directory(self):
        old_cwd = os.getcwd()
        old_temp_dir = extractor.TEMP_DIR
        old_stdout = sys.stdout
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    extractor.TEMP_DIR = str(Path(temp_dir) / "2_FEM" / "temp")
                    sys.stdout = io.StringIO()

                    entered = extractor.enter_temp_work_dir(verbose=True)

                    output = sys.stdout.getvalue()
                    self.assertEqual(os.path.abspath(entered), os.path.abspath(extractor.TEMP_DIR))
                    self.assertEqual(os.path.abspath(os.getcwd()), os.path.abspath(extractor.TEMP_DIR))
                    self.assertIn("Working directory:", output)
                    self.assertIn(os.path.abspath(extractor.TEMP_DIR), output)
                finally:
                    os.chdir(old_cwd)
        finally:
            sys.stdout = old_stdout
            extractor.TEMP_DIR = old_temp_dir
            os.chdir(old_cwd)

    def test_resolve_script_path_uses_abaqus_nogui_argument_when_file_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            script_dir = project_root / "2_FEM" / "scripts"
            script_dir.mkdir(parents=True)
            script_path = script_dir / "extract_odb_ml_data.py"
            script_path.write_text("# placeholder", encoding="utf-8")

            resolved = extractor._resolve_script_path(
                None,
                ["abaqus", "cae", "noGUI=2_FEM/scripts/extract_odb_ml_data.py"],
                str(project_root),
            )

            self.assertEqual(
                os.path.normcase(os.path.normpath(resolved)),
                os.path.normcase(os.path.normpath(str(script_path))),
            )


class ExtractOdbMlDataDiscoveryTests(unittest.TestCase):
    def test_parse_grouped_odb_name(self):
        self.assertEqual(
            extractor.parse_odb_name("9_150_plate.odb"),
            {
                "odb_name": "9_150_plate.odb",
                "ref": None,
                "group_index": 9,
                "instance_index": 150,
            },
        )

    def test_parse_reference_odb_name(self):
        self.assertEqual(
            extractor.parse_odb_name("solid_1_plate.odb"),
            {
                "odb_name": "solid_1_plate.odb",
                "ref": "solid",
                "group_index": None,
                "instance_index": 1,
            },
        )

    def test_filter_odb_names_selects_grouped_by_default(self):
        self.assertEqual(
            extractor.filter_odb_names(
                ["1_1_plate.odb", "2_3_plate.odb", "solid_1_plate.odb", "uniform_1_plate.odb"],
                ref=None,
            ),
            ["1_1_plate.odb", "2_3_plate.odb"],
        )

    def test_filter_odb_names_selects_requested_reference(self):
        self.assertEqual(
            extractor.filter_odb_names(
                ["1_1_plate.odb", "solid_2_plate.odb", "uniform_1_plate.odb", "solid_1_plate.odb"],
                ref="solid",
            ),
            ["solid_1_plate.odb", "solid_2_plate.odb"],
        )

    def test_limit_group_paths_per_group_applies_group_quota(self):
        self.assertEqual(
            extractor.limit_group_paths_per_group(
                [
                    "odb/1_1_plate.odb",
                    "odb/1_2_plate.odb",
                    "odb/1_3_plate.odb",
                    "odb/2_1_plate.odb",
                    "odb/2_2_plate.odb",
                ],
                group_count=2,
                ref=None,
            ),
            [
                "odb/1_1_plate.odb",
                "odb/1_2_plate.odb",
                "odb/2_1_plate.odb",
                "odb/2_2_plate.odb",
            ],
        )

    def test_discover_odb_paths_filters_and_sorts_temp_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            odb_dir = Path(temp_dir) / "odb"
            odb_dir.mkdir()
            for name in ["solid_1_plate.odb", "2_1_plate.odb", "1_2_plate.odb", "notes.txt"]:
                (odb_dir / name).write_text("", encoding="utf-8")

            self.assertEqual(
                extractor.discover_odb_paths(str(odb_dir), ref=None),
                [
                    os.path.join(str(odb_dir), "1_2_plate.odb"),
                    os.path.join(str(odb_dir), "2_1_plate.odb"),
                ],
            )


class ExtractOdbMlDataHoleTests(unittest.TestCase):
    def test_build_uniform_holes_returns_expected_grid(self):
        holes = extractor.build_uniform_holes()

        self.assertEqual(len(holes), 24)
        self.assertAlmostEqual(holes[0]["x"], 7.0)
        self.assertAlmostEqual(holes[0]["y"], 7.0)
        self.assertAlmostEqual(holes[0]["r"], 4.0)
        self.assertAlmostEqual(holes[-1]["x"], 73.0)
        self.assertAlmostEqual(holes[-1]["y"], 153.0)

    def test_holes_for_grouped_model_uses_layout_file_group_and_instance(self):
        parsed = {
            "odb_name": "3_2_plate.odb",
            "ref": None,
            "group_index": 3,
            "instance_index": 2,
        }
        layout_payload = {
            "layouts": [
                {
                    "layout_id": "other_group",
                    "bin": "low_x",
                    "holes": [{"x": 99.0, "y": 99.0, "r": 4.0}],
                },
                {
                    "layout_id": "first_group_3",
                    "bin": "low_y",
                    "holes": [{"x": 1.0, "y": 2.0, "r": 4.0}],
                },
                {
                    "layout_id": "second_group_3",
                    "bin": "low_y",
                    "holes": [{"x": 3.0, "y": 4.0, "r": 4.0}],
                },
            ]
        }

        self.assertEqual(extractor.holes_for_model(parsed, layout_payload), [{"x": 3.0, "y": 4.0, "r": 4.0}])

    def test_holes_for_grouped_model_requires_layout_entry(self):
        parsed = {
            "odb_name": "3_4_plate.odb",
            "ref": None,
            "group_index": 3,
            "instance_index": 4,
        }
        layout_payload = {"layouts": [{"layout_id": "only_one", "bin": "low_y", "holes": []}]}

        with self.assertRaisesRegex(ValueError, "layout entry not found"):
            extractor.holes_for_model(parsed, layout_payload)

    def test_holes_for_reference_models(self):
        self.assertEqual(
            extractor.holes_for_model({"odb_name": "solid_1_plate.odb", "ref": "solid"}, {}),
            [],
        )
        self.assertEqual(
            len(extractor.holes_for_model({"odb_name": "uniform_1_plate.odb", "ref": "uniform"}, {})),
            24,
        )

    def test_load_layout_payload_accepts_selected_layouts_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            selected = Path(temp_dir) / "selected_layouts.json"
            selected.write_text(
                json_text({
                    "schema": "roi_indicator_stratified_layouts_v1",
                    "layouts": [{"layout_id": "layout_1", "bin": "low_x", "holes": []}],
                }),
                encoding="utf-8",
            )

            payload = extractor.load_layout_payload(str(selected))

            self.assertEqual(payload["layouts"][0]["layout_id"], "layout_1")


class ExtractOdbMlDataStiffnessTests(unittest.TestCase):
    def test_resolve_node_set_finds_instance_set_case_insensitively(self):
        region = extractor.resolve_node_set(FakeOdb(), "TopEdge", instance_name="PlateInstance")

        self.assertEqual(region.name, "TOPEDGE")

    def test_compute_equivalent_stiffness_from_boundary_means(self):
        result = extractor.compute_equivalent_stiffness(
            FakeOdb(),
            step_name="Load",
            frame_index=-1,
            top_set="TopEdge",
            bottom_set="BottomEdge",
            instance_name="PlateInstance",
            component=2,
        )

        self.assertEqual(result["top_node_count"], 2)
        self.assertEqual(result["bottom_node_count"], 2)
        self.assertAlmostEqual(result["top_mean_reaction"], 15.0)
        self.assertAlmostEqual(result["bottom_mean_reaction"], -15.0)
        self.assertAlmostEqual(result["top_mean_displacement"], 0.4)
        self.assertAlmostEqual(result["bottom_mean_displacement"], -0.4)
        self.assertAlmostEqual(result["equivalent_stiffness"], 37.5)


class ExtractOdbMlDataStrainTests(unittest.TestCase):
    def test_max_principal_strain_prefers_value_attribute(self):
        value = FakeFieldValue((0.0, 0.0, 0.0), max_principal=0.123)

        self.assertAlmostEqual(extractor.max_principal_strain(value), 0.123)

    def test_max_principal_strain_computes_planar_tensor_value(self):
        value = FakeFieldValue((0.02, 0.01, 0.004))
        expected = 0.015 + (((0.005) ** 2 + 0.004 ** 2) ** 0.5)

        self.assertAlmostEqual(extractor.max_principal_strain(value), expected)

    def test_local_hole_max_principal_selects_annular_band(self):
        values = [
            FakeFieldValue((0.01, 0.0, 0.0), coordinates=(0.0, 0.0)),
            FakeFieldValue((0.02, 0.0, 0.0), coordinates=(4.5, 0.0)),
            FakeFieldValue((0.03, 0.0, 0.0), coordinates=(5.5, 0.0)),
        ]
        hole = {"x": 0.0, "y": 0.0, "r": 4.0}

        self.assertAlmostEqual(extractor.local_hole_max_principal(values, hole, band_mm=1.0), 0.02)

    def test_strain_field_summary_returns_mean_and_max(self):
        values = [
            FakeFieldValue((0.01, 0.0, 0.0), coordinates=(0.0, 0.0)),
            FakeFieldValue((0.03, 0.0, 0.0), coordinates=(1.0, 0.0)),
        ]

        result = extractor.strain_field_summary(values)

        self.assertAlmostEqual(result["mean_max_principal_strain"], 0.02)
        self.assertAlmostEqual(result["max_principal_strain"], 0.03)

    def test_element_centroid_lookup_uses_instance_nodes_and_elements(self):
        instance = FakeInstance()
        instance.nodes = [
            FakeNode(1, (0.0, 0.0, 0.0)),
            FakeNode(2, (2.0, 0.0, 0.0)),
            FakeNode(3, (2.0, 2.0, 0.0)),
            FakeNode(4, (0.0, 2.0, 0.0)),
        ]
        instance.elements = [FakeElement(10, (1, 2, 3, 4))]

        self.assertEqual(extractor.element_centroid_lookup(instance), {10: (1.0, 1.0)})

    def test_element_centroid_lookup_does_not_pass_generators_to_sum(self):
        instance = FakeInstance()
        instance.nodes = [
            FakeNode(1, (0.0, 0.0, 0.0)),
            FakeNode(2, (2.0, 0.0, 0.0)),
            FakeNode(3, (2.0, 2.0, 0.0)),
            FakeNode(4, (0.0, 2.0, 0.0)),
        ]
        instance.elements = [FakeElement(10, (1, 2, 3, 4))]

        old_sum = getattr(extractor, "sum", None)

        def strict_sum(values):
            if hasattr(values, "gi_code"):
                raise TypeError("generator rejected")
            return __builtins__["sum"](values)

        try:
            extractor.sum = strict_sum
            self.assertEqual(extractor.element_centroid_lookup(instance), {10: (1.0, 1.0)})
        finally:
            if old_sum is None:
                del extractor.sum
            else:
                extractor.sum = old_sum

    def test_strain_values_from_frame_attaches_centroid_coordinates(self):
        frame = FakeFrame(strain_values=[FakeFieldValue((0.02, 0.0, 0.0), element_label=10)])
        instance = FakeInstance()
        instance.nodes = [
            FakeNode(1, (0.0, 0.0, 0.0)),
            FakeNode(2, (2.0, 0.0, 0.0)),
            FakeNode(3, (2.0, 2.0, 0.0)),
            FakeNode(4, (0.0, 2.0, 0.0)),
        ]
        instance.elements = [FakeElement(10, (1, 2, 3, 4))]

        values = extractor.strain_values_from_frame(frame, instance=instance)

        self.assertEqual(extractor.value_xy(values[0]), (1.0, 1.0))


class ExtractOdbMlDataRowTests(unittest.TestCase):
    def test_build_model_result_computes_normalized_metrics(self):
        parsed = {
            "odb_name": "1_2_plate.odb",
            "ref": None,
            "group_index": 1,
            "instance_index": 2,
        }
        holes = [{"x": 0.0, "y": 0.0, "r": 4.0}]
        strain_values = [
            FakeFieldValue((0.03, 0.0, 0.0), coordinates=(20.0, 0.0)),
            FakeFieldValue((0.02, 0.0, 0.0), coordinates=(4.5, 0.0)),
        ]
        stiffness = {
            "step": "Load",
            "frame_index": -1,
            "equivalent_stiffness": 40.0,
            "top_node_count": 2,
            "bottom_node_count": 2,
            "top_mean_reaction": 15.0,
            "bottom_mean_reaction": -15.0,
            "top_mean_displacement": 0.4,
            "bottom_mean_displacement": -0.4,
            "warning": "",
        }
        reference = {
            "solid_mean_max_principal_strain": 0.01,
            "solid_equivalent_stiffness": 100.0,
        }

        row = extractor.build_model_result(parsed, "odb/1_2_plate.odb", holes, strain_values, stiffness, reference)

        self.assertEqual(row["status"], "ok")
        self.assertAlmostEqual(row["model_max_principal_strain"], 0.03)
        self.assertAlmostEqual(row["max_strain_concentration_factor"], 3.0)
        self.assertAlmostEqual(row["holes"][0]["local_max_principal_strain"], 0.02)
        self.assertAlmostEqual(row["holes"][0]["strain_concentration_factor"], 2.0)
        self.assertAlmostEqual(row["relative_equivalent_stiffness"], 0.4)


class ExtractOdbMlDataOutputTests(unittest.TestCase):
    def test_completed_odb_names_returns_only_ok_rows(self):
        rows = [
            {"odb_name": "done.odb", "status": "ok"},
            {"odb_name": "failed.odb", "status": "failed"},
            {"odb_name": "blank.odb", "status": ""},
        ]

        self.assertEqual(extractor.completed_odb_names(rows), set(["done.odb"]))

    def test_flatten_row_writes_wide_hole_columns(self):
        row = {
            "odb_name": "1_1_plate.odb",
            "odb_path": "odb/1_1_plate.odb",
            "status": "ok",
            "holes": [
                {
                    "x": 1.0,
                    "y": 2.0,
                    "local_max_principal_strain": 0.03,
                    "strain_concentration_factor": 3.0,
                }
            ],
        }

        flat = extractor.flatten_row(row)

        self.assertEqual(flat["hole_01_x"], 1.0)
        self.assertEqual(flat["hole_01_y"], 2.0)
        self.assertEqual(flat["hole_01_local_max_principal_strain"], 0.03)
        self.assertEqual(flat["hole_01_strain_concentration_factor"], 3.0)
        self.assertIn("hole_24_strain_concentration_factor", flat)

    def test_paths_after_warm_start_removes_completed_paths_only(self):
        paths = ["odb/1_1_plate.odb", "odb/1_2_plate.odb", "odb/1_3_plate.odb"]
        existing_rows = [
            {"odb_name": "1_1_plate.odb", "status": "ok"},
            {"odb_name": "1_2_plate.odb", "status": "failed"},
        ]

        self.assertEqual(
            extractor.paths_after_warm_start(paths, existing_rows, warm_start=True),
            ["odb/1_2_plate.odb", "odb/1_3_plate.odb"],
        )

    def test_group_count_can_be_applied_after_warm_start_filtering(self):
        paths = ["odb/1_1_plate.odb", "odb/1_2_plate.odb", "odb/1_3_plate.odb"]
        unfinished = extractor.paths_after_warm_start(
            paths,
            [{"odb_name": "1_1_plate.odb", "status": "ok"}],
            warm_start=True,
        )

        self.assertEqual(
            extractor.limit_group_paths_per_group(unfinished, group_count=1, ref=None),
            ["odb/1_2_plate.odb"],
        )

    def test_limit_paths_per_run_caps_current_abaqus_process_batch(self):
        paths = ["odb/1_1_plate.odb", "odb/1_2_plate.odb", "odb/1_3_plate.odb"]

        self.assertEqual(
            extractor.limit_paths_per_run(paths, max_odb_per_run=2),
            ["odb/1_1_plate.odb", "odb/1_2_plate.odb"],
        )
        self.assertEqual(
            extractor.limit_paths_per_run(paths, max_odb_per_run=None),
            paths,
        )

    def test_write_and_read_summary_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.csv"
            row = {
                "odb_name": "1_1_plate.odb",
                "odb_path": "odb/1_1_plate.odb",
                "status": "ok",
                "holes": [],
            }

            extractor.write_summary_csv([row], str(output_path))
            rows = extractor.read_existing_summary(str(output_path))

            self.assertEqual(rows[0]["odb_name"], "1_1_plate.odb")
            self.assertIn("hole_24_x", extractor.flatten_row(rows[0]))

    def test_read_existing_summary_rebuilds_nested_holes_from_wide_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.csv"
            row = {
                "odb_name": "1_1_plate.odb",
                "odb_path": "odb/1_1_plate.odb",
                "status": "ok",
                "hole_count": 1,
                "max_strain_concentration_factor": 3.0,
                "holes": [
                    {
                        "x": 1.0,
                        "y": 2.0,
                        "r": 4.0,
                        "local_max_principal_strain": 0.03,
                        "strain_concentration_factor": 3.0,
                    }
                ],
            }

            extractor.write_summary_csv([row], str(output_path))
            rows = extractor.read_existing_summary(str(output_path))

            self.assertEqual(rows[0]["holes"][0]["index"], 1)
            self.assertEqual(rows[0]["hole_count"], 1)
            self.assertAlmostEqual(rows[0]["max_strain_concentration_factor"], 3.0)
            self.assertAlmostEqual(rows[0]["holes"][0]["x"], 1.0)
            self.assertAlmostEqual(rows[0]["holes"][0]["y"], 2.0)
            self.assertAlmostEqual(rows[0]["holes"][0]["r"], 4.0)
            self.assertAlmostEqual(rows[0]["holes"][0]["local_max_principal_strain"], 0.03)
            self.assertAlmostEqual(rows[0]["holes"][0]["strain_concentration_factor"], 3.0)


class ExtractOdbMlDataRunnerTests(unittest.TestCase):
    def test_script_does_not_import_compute_odb_stiffness(self):
        script_text = (SCRIPT_DIR / "extract_odb_ml_data.py").read_text(encoding="utf-8")

        self.assertNotIn("import compute_odb_stiffness", script_text)
        self.assertNotIn("from compute_odb_stiffness", script_text)

    def test_warm_start_and_group_limit_compose(self):
        paths = [
            "odb/1_1_plate.odb",
            "odb/1_2_plate.odb",
            "odb/1_3_plate.odb",
            "odb/2_1_plate.odb",
        ]
        unfinished = extractor.paths_after_warm_start(
            paths,
            [{"odb_name": "1_1_plate.odb", "status": "ok"}],
            warm_start=True,
        )

        self.assertEqual(
            extractor.limit_group_paths_per_group(unfinished, group_count=1, ref=None),
            ["odb/1_2_plate.odb", "odb/2_1_plate.odb"],
        )

    def test_process_one_odb_uses_injected_open_function(self):
        calls = []

        def open_fake(path):
            calls.append(path)
            return FakeOdb(strain_values=[
                FakeFieldValue((0.02, 0.0, 0.0), coordinates=(4.5, 0.0)),
                FakeFieldValue((0.03, 0.0, 0.0), coordinates=(20.0, 0.0)),
            ])

        layout_payload = {
            "layouts": [
                {
                    "layout_id": "first",
                    "bin": "low_x",
                    "holes": [{"x": 99.0, "y": 99.0, "r": 4.0}],
                },
                {
                    "layout_id": "second",
                    "bin": "low_x",
                    "holes": [{"x": 0.0, "y": 0.0, "r": 4.0}],
                }
            ]
        }
        reference = {
            "solid_mean_max_principal_strain": 0.01,
            "solid_equivalent_stiffness": 100.0,
        }

        row = extractor.process_one_odb("odb/1_2_plate.odb", layout_payload, reference, open_odb_func=open_fake)

        self.assertEqual(calls, ["odb/1_2_plate.odb"])
        self.assertEqual(row["odb_name"], "1_2_plate.odb")
        self.assertAlmostEqual(row["max_strain_concentration_factor"], 3.0)

    def test_run_configured_analysis_writes_successes_and_failures(self):
        old_cwd = os.getcwd()
        old_ref = extractor.REF
        old_warm_start = extractor.WARM_START
        old_group_count = extractor.GROUP_COUNT
        old_layout_file = extractor.LAYOUT_FILE
        old_print_progress = extractor.PRINT_PROGRESS
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                fem_root = Path(temp_dir) / "2_FEM"
                temp_root = fem_root / "temp"
                odb_dir = temp_root / "odb"
                odb_dir.mkdir(parents=True)
                for name in ["solid_1_plate.odb", "1_1_plate.odb", "1_2_plate.odb"]:
                    (odb_dir / name).write_text("", encoding="utf-8")
                layout_file = temp_root / "selected_layouts.json"
                layout_file.write_text(
                    json_text({
                        "schema": "roi_indicator_stratified_layouts_v1",
                        "layouts": [
                            {
                                "layout_id": "first",
                                "bin": "low_x",
                                "holes": [{"x": 0.0, "y": 0.0, "r": 4.0}],
                            },
                            {"layout_id": "second", "bin": "low_x"},
                        ]
                    }),
                    encoding="utf-8",
                )

                extractor.TEMP_DIR = str(temp_root)
                extractor.REF = None
                extractor.WARM_START = True
                extractor.GROUP_COUNT = None
                extractor.LAYOUT_FILE = str(layout_file)
                extractor.PRINT_PROGRESS = False

                def open_fake(path):
                    return FakeOdb(strain_values=[
                        FakeFieldValue((0.02, 0.0, 0.0), coordinates=(4.5, 0.0)),
                        FakeFieldValue((0.03, 0.0, 0.0), coordinates=(20.0, 0.0)),
                    ])

                result = extractor.run_configured_analysis(open_odb_func=open_fake)
                os.chdir(old_cwd)

                summary_path = fem_root / "results" / "odb_ml_data" / "odb_ml_summary.csv"
                failures_path = fem_root / "results" / "odb_ml_data" / "odb_ml_failures.csv"
                self.assertTrue(summary_path.is_file())
                self.assertTrue(failures_path.is_file())
                self.assertEqual([row["odb_name"] for row in result["rows"]], ["1_1_plate.odb"])
                self.assertEqual(result["failures"][0]["odb_name"], "1_2_plate.odb")
                self.assertIn("has no holes array", result["failures"][0]["message"])
        finally:
            extractor.TEMP_DIR = str(TEMP_DIR)
            extractor.REF = old_ref
            extractor.WARM_START = old_warm_start
            extractor.GROUP_COUNT = old_group_count
            extractor.LAYOUT_FILE = old_layout_file
            extractor.PRINT_PROGRESS = old_print_progress
            os.chdir(old_cwd)

    def test_run_configured_analysis_persists_each_success_before_next_odb(self):
        old_cwd = os.getcwd()
        old_ref = extractor.REF
        old_warm_start = extractor.WARM_START
        old_group_count = extractor.GROUP_COUNT
        old_layout_file = extractor.LAYOUT_FILE
        old_print_progress = extractor.PRINT_PROGRESS
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                fem_root = Path(temp_dir) / "2_FEM"
                temp_root = fem_root / "temp"
                odb_dir = temp_root / "odb"
                odb_dir.mkdir(parents=True)
                for name in ["solid_1_plate.odb", "1_1_plate.odb", "1_2_plate.odb"]:
                    (odb_dir / name).write_text("", encoding="utf-8")
                layout_file = temp_root / "selected_layouts.json"
                layout_file.write_text(
                    json_text({
                        "schema": "roi_indicator_stratified_layouts_v1",
                        "layouts": [
                            {
                                "layout_id": "first",
                                "bin": "low_x",
                                "holes": [{"x": 0.0, "y": 0.0, "r": 4.0}],
                            },
                            {
                                "layout_id": "second",
                                "bin": "low_x",
                                "holes": [{"x": 0.0, "y": 0.0, "r": 4.0}],
                            },
                        ]
                    }),
                    encoding="utf-8",
                )

                summary_path = fem_root / "results" / "odb_ml_data" / "odb_ml_summary.csv"
                extractor.TEMP_DIR = str(temp_root)
                extractor.REF = None
                extractor.WARM_START = True
                extractor.GROUP_COUNT = None
                extractor.LAYOUT_FILE = str(layout_file)
                extractor.PRINT_PROGRESS = False

                def open_fake(path):
                    if os.path.basename(str(path)) == "1_2_plate.odb":
                        if not summary_path.is_file():
                            raise RuntimeError("first result was not persisted before next ODB")
                        rows = extractor.read_existing_summary(str(summary_path))
                        if [row["odb_name"] for row in rows] != ["1_1_plate.odb"]:
                            raise RuntimeError("unexpected persisted rows before next ODB")
                    return FakeOdb(strain_values=[
                        FakeFieldValue((0.02, 0.0, 0.0), coordinates=(4.5, 0.0)),
                        FakeFieldValue((0.03, 0.0, 0.0), coordinates=(20.0, 0.0)),
                    ])

                result = extractor.run_configured_analysis(open_odb_func=open_fake)
                os.chdir(old_cwd)

                self.assertEqual(result["failures"], [])
                self.assertEqual([row["odb_name"] for row in result["rows"]], ["1_1_plate.odb", "1_2_plate.odb"])
        finally:
            extractor.TEMP_DIR = str(TEMP_DIR)
            extractor.REF = old_ref
            extractor.WARM_START = old_warm_start
            extractor.GROUP_COUNT = old_group_count
            extractor.LAYOUT_FILE = old_layout_file
            extractor.PRINT_PROGRESS = old_print_progress
            os.chdir(old_cwd)

    def test_run_configured_analysis_limits_current_process_batch_size(self):
        old_cwd = os.getcwd()
        old_ref = extractor.REF
        old_warm_start = extractor.WARM_START
        old_group_count = extractor.GROUP_COUNT
        old_layout_file = extractor.LAYOUT_FILE
        old_print_progress = extractor.PRINT_PROGRESS
        old_max_odb_per_run = extractor.MAX_ODB_PER_RUN
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                fem_root = Path(temp_dir) / "2_FEM"
                temp_root = fem_root / "temp"
                odb_dir = temp_root / "odb"
                odb_dir.mkdir(parents=True)
                for name in ["solid_1_plate.odb", "1_1_plate.odb", "1_2_plate.odb", "1_3_plate.odb"]:
                    (odb_dir / name).write_text("", encoding="utf-8")
                layout_file = temp_root / "selected_layouts.json"
                layout_file.write_text(
                    json_text({
                        "schema": "roi_indicator_stratified_layouts_v1",
                        "layouts": [
                            {"layout_id": "first", "bin": "low_x", "holes": [{"x": 0.0, "y": 0.0, "r": 4.0}]},
                            {"layout_id": "second", "bin": "low_x", "holes": [{"x": 0.0, "y": 0.0, "r": 4.0}]},
                            {"layout_id": "third", "bin": "low_x", "holes": [{"x": 0.0, "y": 0.0, "r": 4.0}]},
                        ]
                    }),
                    encoding="utf-8",
                )

                extractor.TEMP_DIR = str(temp_root)
                extractor.REF = None
                extractor.WARM_START = True
                extractor.GROUP_COUNT = None
                extractor.MAX_ODB_PER_RUN = 1
                extractor.LAYOUT_FILE = str(layout_file)
                extractor.PRINT_PROGRESS = False

                def open_fake(path):
                    return FakeOdb(strain_values=[
                        FakeFieldValue((0.02, 0.0, 0.0), coordinates=(4.5, 0.0)),
                        FakeFieldValue((0.03, 0.0, 0.0), coordinates=(20.0, 0.0)),
                    ])

                result = extractor.run_configured_analysis(open_odb_func=open_fake)
                os.chdir(old_cwd)

                self.assertEqual([row["odb_name"] for row in result["rows"]], ["1_1_plate.odb"])
        finally:
            extractor.TEMP_DIR = str(TEMP_DIR)
            extractor.REF = old_ref
            extractor.WARM_START = old_warm_start
            extractor.GROUP_COUNT = old_group_count
            extractor.MAX_ODB_PER_RUN = old_max_odb_per_run
            extractor.LAYOUT_FILE = old_layout_file
            extractor.PRINT_PROGRESS = old_print_progress
            os.chdir(old_cwd)

    def test_run_configured_analysis_records_reference_failure(self):
        old_cwd = os.getcwd()
        old_ref = extractor.REF
        old_warm_start = extractor.WARM_START
        old_group_count = extractor.GROUP_COUNT
        old_layout_file = extractor.LAYOUT_FILE
        old_print_progress = extractor.PRINT_PROGRESS
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                fem_root = Path(temp_dir) / "2_FEM"
                temp_root = fem_root / "temp"
                odb_dir = temp_root / "odb"
                odb_dir.mkdir(parents=True)
                (odb_dir / "1_1_plate.odb").write_text("", encoding="utf-8")
                layout_file = temp_root / "selected_layouts.json"
                layout_file.write_text(
                    json_text({
                        "schema": "roi_indicator_stratified_layouts_v1",
                        "layouts": [
                            {
                                "layout_id": "first",
                                "bin": "low_x",
                                "holes": [{"x": 0.0, "y": 0.0, "r": 4.0}],
                            }
                        ]
                    }),
                    encoding="utf-8",
                )

                extractor.TEMP_DIR = str(temp_root)
                extractor.REF = None
                extractor.WARM_START = True
                extractor.GROUP_COUNT = None
                extractor.LAYOUT_FILE = str(layout_file)
                extractor.PRINT_PROGRESS = False

                def open_missing_reference(path):
                    raise RuntimeError("missing solid reference")

                try:
                    result = extractor.run_configured_analysis(open_odb_func=open_missing_reference)
                finally:
                    os.chdir(old_cwd)

                summary_path = fem_root / "results" / "odb_ml_data" / "odb_ml_summary.csv"
                failures_path = fem_root / "results" / "odb_ml_data" / "odb_ml_failures.csv"
                self.assertTrue(summary_path.is_file())
                self.assertTrue(failures_path.is_file())
                self.assertEqual(result["rows"], [])
                self.assertEqual(result["failures"][0]["odb_name"], "solid_1_plate.odb")
                self.assertIn("reference normalization failed", result["failures"][0]["message"])
        finally:
            extractor.TEMP_DIR = str(TEMP_DIR)
            extractor.REF = old_ref
            extractor.WARM_START = old_warm_start
            extractor.GROUP_COUNT = old_group_count
            extractor.LAYOUT_FILE = old_layout_file
            extractor.PRINT_PROGRESS = old_print_progress
            os.chdir(old_cwd)


def json_text(payload):
    import json

    return json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
