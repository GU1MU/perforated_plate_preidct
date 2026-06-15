import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import BaselineTrainingConfig, CoordinateTrainingConfig
from cnn_surrogate import grid_search as grid_search_module
from cnn_surrogate.grid_search import (
    average_rmse,
    build_trial_id,
    extract_scores,
    grid_parameter_combinations,
    parameter_hash,
    run_grid_search,
    select_best_records,
)


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class SearchConfig(object):
    SEARCH_ID = "unit_v1"
    RESULT_PREFIX = "unit_fine_tuning"
    PARAM_GRID = {
        "batch_size": [16, 32],
        "learning_rate": [1.0e-3],
    }


def _base_config():
    return BaselineTrainingConfig(
        data_csv="input.csv",
        output_dir="output",
        figure_dir="figures",
        temp_dir="temp",
        train_test_split=1,
        split_shuffle=False,
        random_seed=123,
        pixel_size=2.0,
        image_height=80,
        image_width=40,
        batch_size=2,
        epochs=3,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        dropout=0.2,
        loss_weight_stiffness=1.0,
        loss_weight_strain=1.0,
        early_stopping_patience=5,
        device="cpu",
        show_progress=False,
        progress_description="base",
        save_model=False,
    )


def _coordinate_base_config():
    return CoordinateTrainingConfig(
        data_csv="input.csv",
        output_dir="output",
        figure_dir="figures",
        temp_dir="temp",
        train_test_split=1,
        split_shuffle=False,
        random_seed=123,
        coordinate_domain_width=80.0,
        coordinate_domain_height=160.0,
        coordinate_feature_dim=6,
        batch_size=2,
        epochs=3,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        dropout=0.2,
        point_hidden_dim=32,
        context_hidden_dim=64,
        loss_weight_stiffness=1.0,
        loss_weight_local_strain=1.0,
        early_stopping_patience=5,
        device="cpu",
        show_progress=False,
        progress_description="base",
        save_model=False,
        warm_start=False,
        checkpoint_path="checkpoint.pt",
    )


def _cnn_metric_result(stiffness_rmse, local_strain_rmse):
    return {
        "metrics": {
            "val": {
                "targets": {
                    "relative_equivalent_stiffness": {"rmse": stiffness_rmse},
                },
                "local_strain_summary": {"rmse": local_strain_rmse},
            }
        }
    }


def _distilled_metric_result(stiffness_rmse, strain_rmse):
    return {
        "student_metrics": {
            "val": {
                "targets": {
                    "relative_equivalent_stiffness": {"rmse": stiffness_rmse},
                    "max_strain_concentration_factor": {"rmse": strain_rmse},
                },
            }
        }
    }


class GridSearchUtilityTests(unittest.TestCase):
    def test_grid_parameter_combinations_returns_cartesian_product(self):
        combinations = grid_parameter_combinations({"a": [1, 2], "b": ["x", "y"]})

        self.assertEqual(combinations, [
            {"a": 1, "b": "x"},
            {"a": 1, "b": "y"},
            {"a": 2, "b": "x"},
            {"a": 2, "b": "y"},
        ])

    def test_select_best_records_tracks_each_target_and_average(self):
        records = [
            {
                "trial_id": "trial_1",
                "status": "completed",
                "scores": {
                    "stiffness_rmse": 0.2,
                    "local_strain_rmse": 0.4,
                    "strain_rmse": 0.4,
                    "average_rmse": 0.3,
                },
            },
            {
                "trial_id": "trial_2",
                "status": "completed",
                "scores": {
                    "stiffness_rmse": 0.1,
                    "local_strain_rmse": 0.6,
                    "strain_rmse": 0.6,
                    "average_rmse": 0.35,
                },
            },
            {
                "trial_id": "trial_3",
                "status": "completed",
                "scores": {
                    "stiffness_rmse": 0.3,
                    "local_strain_rmse": 0.2,
                    "strain_rmse": 0.2,
                    "average_rmse": 0.25,
                },
            },
        ]

        best = select_best_records(records)

        self.assertEqual(best["best_stiffness"]["trial_id"], "trial_2")
        self.assertEqual(best["best_local_strain"]["trial_id"], "trial_3")
        self.assertEqual(best["best_strain"]["trial_id"], "trial_3")
        self.assertEqual(best["best_average"]["trial_id"], "trial_3")

    def test_select_best_records_preserves_distilled_legacy_strain_score(self):
        records = [
            {
                "trial_id": "trial_1",
                "status": "completed",
                "scores": {"stiffness_rmse": 0.2, "strain_rmse": 0.4, "average_rmse": 0.3},
            },
            {
                "trial_id": "trial_2",
                "status": "completed",
                "scores": {"stiffness_rmse": 0.1, "strain_rmse": 0.3, "average_rmse": 0.2},
            },
        ]

        best = select_best_records(records)

        self.assertIsNone(best["best_local_strain"])
        self.assertEqual(best["best_strain"]["trial_id"], "trial_2")

    def test_average_rmse_requires_both_targets(self):
        self.assertIsNone(average_rmse(None, 0.2))
        self.assertIsNone(average_rmse(0.2, None))
        self.assertEqual(average_rmse(0.2, 0.4), 0.30000000000000004)

    def test_extract_scores_reads_corrected_cnn_metrics_without_legacy_strain_target(self):
        scores = extract_scores("cnn", _cnn_metric_result(0.2, 0.6))

        self.assertEqual(scores["stiffness_rmse"], 0.2)
        self.assertEqual(scores["local_strain_rmse"], 0.6)
        self.assertEqual(scores["strain_rmse"], 0.6)
        self.assertEqual(scores["average_rmse"], 0.4)

    def test_extract_scores_preserves_distilled_legacy_strain_target(self):
        scores = extract_scores("distilled", _distilled_metric_result(0.2, 0.6))

        self.assertEqual(scores, {
            "stiffness_rmse": 0.2,
            "strain_rmse": 0.6,
            "average_rmse": 0.4,
        })


class GridSearchRunnerTests(unittest.TestCase):
    def test_run_grid_search_places_trial_artifacts_under_search_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            captured_configs = []

            def fake_runner(config):
                captured_configs.append(config)
                os.makedirs(config.output_dir, exist_ok=True)
                with open(os.path.join(config.output_dir, "marker.txt"), "w", encoding="utf-8") as handle:
                    handle.write("ran")
                return _cnn_metric_result(0.1, 0.5)

            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_grid_search(
                    model_name="cnn",
                    search_config=SearchConfig,
                    base_config_builder=_base_config,
                    training_runner=fake_runner,
                    results_root=os.path.join(temp_dir, "results"),
                    temp_root=os.path.join(temp_dir, "temp"),
                    device="cuda",
                    train_test_split=180,
                    early_stopping_patience=50,
                    save_figures=False,
                    save_all=True,
                    save_best_model=False,
                )

            search_dir = os.path.join(temp_dir, "results", "unit_fine_tuning_unit_v1")
            self.assertEqual(summary["search_dir"], search_dir)
            self.assertTrue(captured_configs[0].output_dir.startswith(search_dir))
            self.assertIn(os.path.join("trials", "unit_v1_0001", "results"), captured_configs[0].output_dir)
            self.assertIsNone(captured_configs[0].figure_dir)
            self.assertFalse(captured_configs[0].save_figures)
            self.assertTrue(os.path.isfile(os.path.join(search_dir, "search_results.jsonl")))

    def test_run_grid_search_save_all_retains_each_trial_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            def fake_runner(config):
                os.makedirs(config.output_dir, exist_ok=True)
                with open(os.path.join(config.output_dir, "marker.txt"), "w", encoding="utf-8") as handle:
                    handle.write("ran")
                return _cnn_metric_result(0.2, 0.2)

            with contextlib.redirect_stdout(io.StringIO()):
                run_grid_search(
                    model_name="cnn",
                    search_config=SearchConfig,
                    base_config_builder=_base_config,
                    training_runner=fake_runner,
                    results_root=os.path.join(temp_dir, "results"),
                    temp_root=os.path.join(temp_dir, "temp"),
                    device="cuda",
                    train_test_split=180,
                    early_stopping_patience=50,
                    save_figures=False,
                    save_all=True,
                    save_best_model=False,
                )

            search_dir = os.path.join(temp_dir, "results", "unit_fine_tuning_unit_v1")
            self.assertTrue(os.path.isfile(os.path.join(
                search_dir, "trials", "unit_v1_0001", "results", "marker.txt"
            )))
            self.assertTrue(os.path.isfile(os.path.join(
                search_dir, "trials", "unit_v1_0002", "results", "marker.txt"
            )))

    def test_run_grid_search_prunes_non_best_trial_directories_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []

            def fake_runner(config):
                calls.append(config)
                os.makedirs(config.output_dir, exist_ok=True)
                with open(os.path.join(config.output_dir, "marker.txt"), "w", encoding="utf-8") as handle:
                    handle.write("ran")
                if len(calls) == 1:
                    return _cnn_metric_result(0.4, 0.4)
                return _cnn_metric_result(0.1, 0.1)

            with contextlib.redirect_stdout(io.StringIO()):
                run_grid_search(
                    model_name="cnn",
                    search_config=SearchConfig,
                    base_config_builder=_base_config,
                    training_runner=fake_runner,
                    results_root=os.path.join(temp_dir, "results"),
                    temp_root=os.path.join(temp_dir, "temp"),
                    device="cuda",
                    train_test_split=180,
                    early_stopping_patience=50,
                    save_figures=False,
                    save_all=False,
                    save_best_model=False,
                )

            search_dir = os.path.join(temp_dir, "results", "unit_fine_tuning_unit_v1")
            self.assertFalse(os.path.exists(os.path.join(search_dir, "trials", "unit_v1_0001")))
            self.assertTrue(os.path.exists(os.path.join(search_dir, "trials", "unit_v1_0002")))
            self.assertTrue(os.path.isfile(os.path.join(search_dir, "search_results.jsonl")))
            self.assertTrue(os.path.isfile(os.path.join(search_dir, "best_results.json")))

    def test_run_grid_search_retained_failed_trial_writes_attempted_trial_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            def failing_runner(config):
                raise RuntimeError("boom")

            with contextlib.redirect_stdout(io.StringIO()):
                run_grid_search(
                    model_name="cnn",
                    search_config=SearchConfig,
                    base_config_builder=_base_config,
                    training_runner=failing_runner,
                    results_root=os.path.join(temp_dir, "results"),
                    temp_root=os.path.join(temp_dir, "temp"),
                    device="cuda",
                    train_test_split=180,
                    early_stopping_patience=50,
                    save_figures=False,
                    save_all=True,
                    save_best_model=False,
                )

            search_dir = os.path.join(temp_dir, "results", "unit_fine_tuning_unit_v1")
            trial_config_path = os.path.join(
                search_dir,
                "trials",
                "unit_v1_0001",
                "results",
                "trial_config.json",
            )

            with open(trial_config_path, "r", encoding="utf-8") as handle:
                trial_config = json.load(handle)

            self.assertEqual(trial_config["batch_size"], 16)
            self.assertIn(os.path.join("trials", "unit_v1_0001", "results"), trial_config["output_dir"])
            self.assertIsNone(trial_config["figure_dir"])
            self.assertFalse(trial_config["save_figures"])
            self.assertEqual(
                trial_config["checkpoint_path"],
                os.path.join(trial_config["output_dir"], "checkpoint.pt"),
            )

    def test_is_within_directory_rejects_sibling_paths(self):
        parent = os.path.join("tmp", "search")
        child = os.path.join("tmp", "search-trial")

        self.assertTrue(grid_search_module._is_within_directory(parent, os.path.join(parent, "trial")))
        self.assertFalse(grid_search_module._is_within_directory(parent, child))

    def test_prune_trial_artifacts_ignores_malformed_search_parent_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            search_dir = os.path.join(temp_dir, "results", "unit_fine_tuning_unit_v1")
            os.makedirs(search_dir)
            marker_path = os.path.join(search_dir, "search_results.jsonl")
            with open(marker_path, "w", encoding="utf-8") as handle:
                handle.write("")

            records = [{"trial_id": "unit_v1_0001", "trial_dir": search_dir}]

            grid_search_module._prune_trial_artifacts(search_dir, records, {}, save_all=False)

            self.assertTrue(os.path.isfile(marker_path))

    def test_run_grid_search_skips_completed_trial_and_records_internal_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            search_dir = os.path.join(temp_dir, "results", "unit_fine_tuning_unit_v1")
            os.makedirs(search_dir)
            first_params = {"batch_size": 16, "learning_rate": 1.0e-3}
            first_record = {
                "trial_id": build_trial_id("unit_v1", 1),
                "param_hash": parameter_hash(first_params),
                "status": "completed",
                "params": first_params,
                "scores": {
                    "stiffness_rmse": 0.2,
                    "local_strain_rmse": 0.4,
                    "strain_rmse": 0.4,
                    "average_rmse": 0.3,
                },
                "trial_dir": os.path.join(search_dir, "trials", "unit_v1_0001"),
            }
            with open(os.path.join(search_dir, "search_results.jsonl"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(first_record, sort_keys=True) + "\n")

            captured_configs = []

            def fake_runner(config):
                captured_configs.append(config)
                return _cnn_metric_result(0.1, 0.5)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summary = run_grid_search(
                    model_name="cnn",
                    search_config=SearchConfig,
                    base_config_builder=_base_config,
                    training_runner=fake_runner,
                    results_root=os.path.join(temp_dir, "results"),
                    temp_root=os.path.join(temp_dir, "temp"),
                    device="cuda",
                    train_test_split=180,
                    early_stopping_patience=50,
                    save_figures=False,
                    save_all=False,
                    save_best_model=False,
                )

            self.assertEqual(len(captured_configs), 1)
            self.assertEqual(captured_configs[0].train_test_split, 180)
            self.assertEqual(captured_configs[0].early_stopping_patience, 50)
            self.assertEqual(captured_configs[0].device, "cuda")
            self.assertTrue(captured_configs[0].show_progress)
            self.assertFalse(captured_configs[0].save_model)
            self.assertFalse(os.path.exists(first_record["trial_dir"]))
            self.assertEqual(summary["records"][1]["scores"]["local_strain_rmse"], 0.5)
            self.assertEqual(summary["records"][1]["scores"]["strain_rmse"], 0.5)
            self.assertEqual(summary["best_results"]["best_stiffness"]["trial_id"], "unit_v1_0002")
            self.assertEqual(summary["best_results"]["best_local_strain"]["trial_id"], "unit_v1_0001")
            self.assertIn("[1/2] skipped unit_v1_0001", output.getvalue())
            self.assertIn("[2/2] running unit_v1_0002", output.getvalue())
            self.assertNotIn("val_rmse", output.getvalue())
            self.assertNotIn("current_best", output.getvalue())

            with open(os.path.join(search_dir, "best_results.json"), "r", encoding="utf-8") as handle:
                best_results = json.load(handle)
            self.assertEqual(best_results["best_stiffness"]["trial_id"], "unit_v1_0002")
            self.assertTrue(os.path.isfile(os.path.join(search_dir, "search_results.csv")))
            with open(os.path.join(search_dir, "search_results.csv"), "r", encoding="utf-8") as handle:
                header = handle.readline()
            self.assertIn("local_strain_rmse", header)
            self.assertIn("trial_dir", header)

    def test_run_grid_search_save_model_retrains_best_average_to_final_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []

            def fake_runner(config):
                calls.append(config)
                if len(calls) == 1:
                    return _cnn_metric_result(0.4, 0.4)
                if len(calls) == 2:
                    return _cnn_metric_result(0.2, 0.2)
                return _cnn_metric_result(0.2, 0.2)

            with contextlib.redirect_stdout(io.StringIO()):
                run_grid_search(
                    model_name="cnn",
                    search_config=SearchConfig,
                    base_config_builder=_base_config,
                    training_runner=fake_runner,
                    results_root=os.path.join(temp_dir, "results"),
                    temp_root=os.path.join(temp_dir, "temp"),
                    device="cuda",
                    train_test_split=180,
                    early_stopping_patience=50,
                    save_figures=False,
                    save_all=False,
                    save_best_model=True,
                )

            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[-1].output_dir, os.path.join("3_CNN", "results", "cnn_surrogate"))
            self.assertEqual(calls[-1].figure_dir, os.path.join("3_CNN", "figures", "cnn_surrogate"))
            self.assertTrue(calls[-1].save_figures)
            self.assertTrue(calls[-1].save_model)
            self.assertEqual(calls[-1].batch_size, 32)

    def test_run_grid_search_save_model_supports_coordinate_final_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []

            def fake_runner(config):
                calls.append(config)
                return _cnn_metric_result(0.2, 0.2)

            with contextlib.redirect_stdout(io.StringIO()):
                run_grid_search(
                    model_name="coordinate",
                    search_config=SearchConfig,
                    base_config_builder=_coordinate_base_config,
                    training_runner=fake_runner,
                    results_root=os.path.join(temp_dir, "results"),
                    temp_root=os.path.join(temp_dir, "temp"),
                    device="cuda",
                    train_test_split=180,
                    early_stopping_patience=50,
                    save_figures=False,
                    save_all=False,
                    save_best_model=True,
                )

            self.assertEqual(calls[-1].output_dir, os.path.join("3_CNN", "results", "coordinate_surrogate"))
            self.assertEqual(calls[-1].figure_dir, os.path.join("3_CNN", "figures", "coordinate_surrogate"))
            self.assertEqual(calls[-1].checkpoint_path, os.path.join("3_CNN", "results", "coordinate_surrogate", "checkpoint.pt"))
            self.assertTrue(calls[-1].save_figures)
            self.assertTrue(calls[-1].save_model)
            self.assertEqual(calls[0].checkpoint_path, os.path.join(calls[0].output_dir, "checkpoint.pt"))


if __name__ == "__main__":
    unittest.main()
