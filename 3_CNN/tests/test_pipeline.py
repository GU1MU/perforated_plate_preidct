import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate import plotting
from cnn_surrogate.config import BaselineTrainingConfig
from cnn_surrogate.data import cnn_target_columns
from cnn_surrogate.pipeline import run_baseline_training


def _expected_prediction_columns():
    columns = ["odb_name", "group_index", "instance_index", "split"]
    for target_column in cnn_target_columns():
        columns.append(target_column + "_true")
        columns.append(target_column + "_pred")
    return columns


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class BaselineConfigTests(unittest.TestCase):
    def test_config_carries_user_facing_paths_and_training_parameters(self):
        config = BaselineTrainingConfig(
            data_csv=os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv"),
            output_dir=os.path.join("3_CNN", "results", "cnn_surrogate"),
            figure_dir=os.path.join("3_CNN", "figures", "cnn_surrogate"),
            temp_dir=os.path.join("3_CNN", "temp"),
            train_test_split=25,
            split_shuffle=True,
            random_seed=20260611,
            pixel_size=2.0,
            image_height=80,
            image_width=40,
            batch_size=32,
            epochs=300,
            learning_rate=1.0e-3,
            weight_decay=1.0e-4,
            dropout=0.2,
            loss_weight_stiffness=1.0,
            loss_weight_strain=1.0,
            early_stopping_patience=30,
            device="auto",
            show_progress=True,
            progress_description="Training CNN surrogate",
            save_model=True,
        )
        self.assertEqual(config.image_height, 80)
        self.assertEqual(config.image_width, 40)
        self.assertTrue(config.show_progress)


class CnnPlottingTests(unittest.TestCase):
    def test_local_strain_pred_vs_true_uses_sample_maximum(self):
        rows = []
        for row_offset in [0.0, 100.0]:
            row = {}
            for index, target_column in enumerate(cnn_target_columns()[1:], start=1):
                row[target_column + "_true"] = row_offset + float(index)
                row[target_column + "_pred"] = row_offset + float(index * 2)
            rows.append(row)
        predictions = pd.DataFrame(rows)

        captured = {}
        original_scatter = plotting._plot_identity_scatter

        def capture_scatter(true_values, pred_values, xlabel, ylabel):
            captured["true_values"] = list(true_values)
            captured["pred_values"] = list(pred_values)
            captured["xlabel"] = xlabel
            captured["ylabel"] = ylabel

        plotting._plot_identity_scatter = capture_scatter
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                plotting.plot_cnn_local_strain_pred_vs_true(predictions, temp_dir)
        finally:
            plotting._plot_identity_scatter = original_scatter

        self.assertEqual(captured["true_values"], [24.0, 124.0])
        self.assertEqual(captured["pred_values"], [48.0, 148.0])
        self.assertEqual(captured["xlabel"], "FEM max local strain concentration")
        self.assertEqual(captured["ylabel"], "CNN max local strain concentration")


def _sample_row(group=1, instance=1, status="ok"):
    row = {
        "odb_name": "%d_%d_plate.odb" % (group, instance),
        "status": status,
        "group_index": group,
        "instance_index": instance,
        "relative_equivalent_stiffness": 0.5 + 0.01 * instance + 0.02 * group,
        "max_strain_concentration_factor": 1.5 + 0.03 * instance + 0.04 * group,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float((index + instance) % 40)
        row["hole_%02d_y" % index] = float((2 * index + group) % 80)
        row["hole_%02d_strain_concentration_factor" % index] = (
            1.0 + 0.01 * index + 0.02 * instance + 0.03 * group
        )
    return row


def _write_sample_csv(path):
    rows = []
    for group in [1, 2]:
        for instance in range(1, 5):
            rows.append(_sample_row(group=group, instance=instance))
    pd.DataFrame(rows).to_csv(path, index=False)


def _tiny_config(data_csv, output_dir, figure_dir, work_dir, save_model):
    return BaselineTrainingConfig(
        data_csv=data_csv,
        output_dir=output_dir,
        figure_dir=figure_dir,
        temp_dir=work_dir,
        train_test_split=2,
        split_shuffle=False,
        random_seed=20260611,
        pixel_size=2.0,
        image_height=80,
        image_width=40,
        batch_size=2,
        epochs=1,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        dropout=0.2,
        loss_weight_stiffness=1.0,
        loss_weight_strain=1.0,
        loss_weight_local_strain=1.0,
        early_stopping_patience=None,
        device="cpu",
        show_progress=False,
        progress_description="Training CNN surrogate",
        save_model=save_model,
        spatial_pool_height=10,
        spatial_pool_width=5,
        embedding_dim=16,
        warm_start=True,
        checkpoint_path=os.path.join(output_dir, "checkpoint.pt"),
    )


class BaselinePipelineTests(unittest.TestCase):
    def test_run_baseline_training_writes_expected_artifacts_without_model_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_csv = os.path.join(temp_dir, "summary.csv")
            output_dir = os.path.join(temp_dir, "results")
            figure_dir = os.path.join(temp_dir, "figures")
            work_dir = os.path.join(temp_dir, "temp")
            _write_sample_csv(data_csv)
            os.makedirs(output_dir)
            for filename in [
                "model.pt",
                "stiffness_scaler.pkl",
                "local_strain_scaler.pkl",
                "target_scaler.pkl",
            ]:
                with open(os.path.join(output_dir, filename), "wb") as artifact_file:
                    artifact_file.write(b"stale")

            config = _tiny_config(data_csv, output_dir, figure_dir, work_dir, save_model=False)

            result = run_baseline_training(config)

            self.assertIn("history", result)
            self.assertIn("metrics", result)
            self.assertIn("predictions", result)
            for filename in [
                "split_manifest.csv",
                "train_history.csv",
                "metrics.json",
                "predictions.csv",
                "checkpoint.pt",
            ]:
                self.assertTrue(os.path.exists(os.path.join(output_dir, filename)))
            for filename in [
                "loss_curve.png",
                "stiffness_pred_vs_true.png",
                "local_strain_pred_vs_true.png",
                "local_strain_error_distribution.png",
            ]:
                self.assertTrue(os.path.exists(os.path.join(figure_dir, filename)))
            with open(os.path.join(output_dir, "metrics.json"), "r", encoding="utf-8") as metrics_file:
                metrics_payload = metrics_file.read()
                self.assertIn("local_strain_summary", metrics_payload)
                self.assertIn("local_strain_error_quantiles", metrics_payload)
            self.assertIn("local_strain_summary", result["metrics"]["train"])
            self.assertIn("local_strain_error_quantiles", result["metrics"]["train"])
            self.assertNotIn("local_strain_by_hole", result["metrics"]["train"])
            for filename in [
                "model.pt",
                "stiffness_scaler.pkl",
                "local_strain_scaler.pkl",
                "target_scaler.pkl",
            ]:
                self.assertFalse(os.path.exists(os.path.join(output_dir, filename)))

    def test_run_baseline_training_writes_required_predictions_and_model_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_csv = os.path.join(temp_dir, "summary.csv")
            output_dir = os.path.join(temp_dir, "results")
            figure_dir = os.path.join(temp_dir, "figures")
            work_dir = os.path.join(temp_dir, "temp")
            _write_sample_csv(data_csv)
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, "target_scaler.pkl"), "wb") as artifact_file:
                artifact_file.write(b"stale")

            config = _tiny_config(data_csv, output_dir, figure_dir, work_dir, save_model=True)

            run_baseline_training(config)

            predictions = pd.read_csv(os.path.join(output_dir, "predictions.csv"))
            self.assertEqual(list(predictions.columns), _expected_prediction_columns())

            model_path = os.path.join(output_dir, "model.pt")
            stiffness_scaler_path = os.path.join(output_dir, "stiffness_scaler.pkl")
            local_strain_scaler_path = os.path.join(output_dir, "local_strain_scaler.pkl")
            self.assertTrue(os.path.isfile(model_path))
            self.assertTrue(os.path.isfile(stiffness_scaler_path))
            self.assertTrue(os.path.isfile(local_strain_scaler_path))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "target_scaler.pkl")))

            model_package = torch.load(model_path, map_location="cpu")
            self.assertIn("model_state_dict", model_package)
            self.assertEqual(model_package["image_height"], config.image_height)
            self.assertEqual(model_package["image_width"], config.image_width)
            self.assertEqual(model_package["pixel_size"], config.pixel_size)
            self.assertEqual(model_package["target_columns"], cnn_target_columns())
            self.assertEqual(model_package["embedding_dim"], config.embedding_dim)
            self.assertEqual(model_package["spatial_pool_height"], config.spatial_pool_height)
            self.assertEqual(model_package["spatial_pool_width"], config.spatial_pool_width)


if __name__ == "__main__":
    unittest.main()
