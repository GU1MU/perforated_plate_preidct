import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import CoordinateTrainingConfig
from cnn_surrogate.data import CoordinateLayoutDataset, coordinate_target_columns
from cnn_surrogate.evaluation import compute_coordinate_metrics, predict_coordinate_frame, predict_frame
from cnn_surrogate.io import save_coordinate_model_package
from cnn_surrogate.models import CoordinateSurrogate
from cnn_surrogate import plotting


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _coordinate_config(data_csv, output_dir, figure_dir, temp_dir, save_model=False):
    return CoordinateTrainingConfig(
        data_csv=data_csv,
        output_dir=output_dir,
        figure_dir=figure_dir,
        temp_dir=temp_dir,
        train_test_split=2,
        split_shuffle=False,
        random_seed=20260611,
        coordinate_domain_width=80.0,
        coordinate_domain_height=160.0,
        coordinate_feature_dim=6,
        batch_size=2,
        epochs=1,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        dropout=0.0,
        point_hidden_dim=8,
        context_hidden_dim=16,
        loss_weight_stiffness=1.0,
        loss_weight_local_strain=1.0,
        early_stopping_patience=None,
        device="cpu",
        show_progress=False,
        progress_description="Training coordinate surrogate",
        save_model=save_model,
        warm_start=False,
        checkpoint_path=os.path.join(output_dir, "checkpoint.pt"),
    )


def _coordinate_row(group=1, instance=1):
    row = {
        "odb_name": "%d_%d_plate.odb" % (group, instance),
        "status": "ok",
        "group_index": group,
        "instance_index": instance,
        "relative_equivalent_stiffness": 0.75 + 0.001 * instance + 0.002 * group,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float(8.0 + index + instance)
        row["hole_%02d_y" % index] = float(12.0 + index + group)
        row["hole_%02d_strain_concentration_factor" % index] = 2.0 + 0.01 * index + 0.001 * instance
    return row


def _write_coordinate_csv(path):
    rows = []
    for group in [1, 2]:
        for instance in range(1, 5):
            rows.append(_coordinate_row(group=group, instance=instance))
    pd.DataFrame(rows).to_csv(path, index=False)


class CoordinateEvaluationTests(unittest.TestCase):
    def test_predict_coordinate_frame_writes_true_and_pred_columns(self):
        model = CoordinateSurrogate(point_hidden_dim=8, context_hidden_dim=16, dropout=0.0)
        coordinates = torch.zeros((2, 24, 6), dtype=torch.float32)
        stiffness_targets = torch.zeros((2, 1), dtype=torch.float32)
        local_targets = torch.zeros((2, 24), dtype=torch.float32)
        dataset = TensorDataset(coordinates, stiffness_targets, local_targets)
        dataset.frame = pd.DataFrame([
            {"odb_name": "a.odb", "group_index": 1, "instance_index": 1},
            {"odb_name": "b.odb", "group_index": 1, "instance_index": 2},
        ])
        dataset.stiffness_scaler = StandardScaler().fit(np.zeros((2, 1), dtype=np.float32))
        dataset.local_strain_scaler = StandardScaler().fit(np.zeros((48, 1), dtype=np.float32))

        predictions = predict_coordinate_frame(model, dataset, batch_size=2, split_name="val", device="cpu")

        self.assertEqual(list(predictions[["odb_name", "group_index", "instance_index", "split"]].columns), [
            "odb_name",
            "group_index",
            "instance_index",
            "split",
        ])
        self.assertIn("relative_equivalent_stiffness_true", predictions.columns)
        self.assertIn("relative_equivalent_stiffness_pred", predictions.columns)
        self.assertIn("hole_24_strain_concentration_factor_true", predictions.columns)
        self.assertIn("hole_24_strain_concentration_factor_pred", predictions.columns)
        self.assertNotIn("max_strain_concentration_factor_true", predictions.columns)

    def test_predict_frame_dispatches_coordinate_dataset(self):
        model = CoordinateSurrogate(point_hidden_dim=8, context_hidden_dim=16, dropout=0.0)
        frame = pd.DataFrame([_coordinate_row(instance=1), _coordinate_row(instance=2)])
        dataset = CoordinateLayoutDataset(
            frame,
            domain_width=80.0,
            domain_height=160.0,
            stiffness_scaler=StandardScaler(),
            local_strain_scaler=StandardScaler(),
            fit_scalers=True,
        )

        predictions = predict_frame(model, dataset, batch_size=2, split_name="val", device="cpu")

        self.assertIn("relative_equivalent_stiffness_pred", predictions.columns)
        self.assertIn("hole_24_strain_concentration_factor_pred", predictions.columns)

    def test_predict_coordinate_frame_returns_expected_empty_columns(self):
        model = CoordinateSurrogate(point_hidden_dim=8, context_hidden_dim=16, dropout=0.0)
        coordinates = torch.zeros((0, 24, 6), dtype=torch.float32)
        stiffness_targets = torch.zeros((0, 1), dtype=torch.float32)
        local_targets = torch.zeros((0, 24), dtype=torch.float32)
        dataset = TensorDataset(coordinates, stiffness_targets, local_targets)
        dataset.frame = pd.DataFrame(columns=["odb_name", "group_index", "instance_index"])
        dataset.stiffness_scaler = StandardScaler().fit(np.zeros((2, 1), dtype=np.float32))
        dataset.local_strain_scaler = StandardScaler().fit(np.zeros((48, 1), dtype=np.float32))

        predictions = predict_coordinate_frame(model, dataset, batch_size=2, split_name="test", device="cpu")

        self.assertEqual(len(predictions), 0)
        self.assertIn("relative_equivalent_stiffness_true", predictions.columns)
        self.assertIn("hole_24_strain_concentration_factor_pred", predictions.columns)

    def test_compute_coordinate_metrics_has_stiffness_and_local_strain_summary(self):
        frame = pd.DataFrame({
            "split": ["val", "val"],
            "relative_equivalent_stiffness_true": [1.0, 2.0],
            "relative_equivalent_stiffness_pred": [1.0, 2.0],
        })
        for column in coordinate_target_columns()[1:]:
            frame[column + "_true"] = [1.0, 2.0]
            frame[column + "_pred"] = [1.0, 2.0]

        metrics = compute_coordinate_metrics(frame)

        self.assertEqual(metrics["val"]["count"], 2)
        self.assertEqual(metrics["val"]["targets"]["relative_equivalent_stiffness"]["rmse"], 0.0)
        self.assertEqual(metrics["val"]["local_strain_summary"]["rmse"], 0.0)
        self.assertEqual(metrics["val"]["local_strain_by_hole"]["hole_24_strain_concentration_factor"]["rmse"], 0.0)
        self.assertIsNone(metrics["train"]["targets"]["relative_equivalent_stiffness"]["r2"])


class CoordinateIoTests(unittest.TestCase):
    def test_save_coordinate_model_package_respects_save_model_flag_and_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = CoordinateSurrogate(point_hidden_dim=8, context_hidden_dim=16, dropout=0.0)
            stiffness_scaler = StandardScaler().fit(np.zeros((2, 1), dtype=np.float32))
            local_strain_scaler = StandardScaler().fit(np.zeros((48, 1), dtype=np.float32))
            config = _coordinate_config(
                data_csv="input.csv",
                output_dir=temp_dir,
                figure_dir=os.path.join(temp_dir, "figures"),
                temp_dir=os.path.join(temp_dir, "temp"),
                save_model=False,
            )
            for filename in ["model.pt", "stiffness_scaler.pkl", "local_strain_scaler.pkl", "target_scaler.pkl"]:
                with open(os.path.join(temp_dir, filename), "wb") as stale_file:
                    stale_file.write(b"stale")

            result = save_coordinate_model_package(model, stiffness_scaler, local_strain_scaler, temp_dir, config)

            self.assertIsNone(result)
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "model.pt")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "stiffness_scaler.pkl")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "local_strain_scaler.pkl")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "target_scaler.pkl")))

    def test_save_coordinate_model_package_writes_metadata_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = CoordinateSurrogate(point_hidden_dim=8, context_hidden_dim=16, dropout=0.0)
            stiffness_scaler = StandardScaler().fit(np.zeros((2, 1), dtype=np.float32))
            local_strain_scaler = StandardScaler().fit(np.zeros((48, 1), dtype=np.float32))
            config = _coordinate_config(
                data_csv="input.csv",
                output_dir=temp_dir,
                figure_dir=os.path.join(temp_dir, "figures"),
                temp_dir=os.path.join(temp_dir, "temp"),
                save_model=True,
            )

            model_path = save_coordinate_model_package(model, stiffness_scaler, local_strain_scaler, temp_dir, config)
            package = torch.load(model_path, map_location="cpu")

            self.assertEqual(package["target_columns"], coordinate_target_columns())
            self.assertEqual(package["coordinate_feature_dim"], 6)
            self.assertEqual(package["point_hidden_dim"], 8)
            self.assertEqual(package["context_hidden_dim"], 16)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "stiffness_scaler.pkl")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "local_strain_scaler.pkl")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "target_scaler.pkl")))


class CoordinatePlottingTests(unittest.TestCase):
    def test_local_strain_pred_vs_true_uses_per_sample_maximum(self):
        frame = pd.DataFrame({
            "odb_name": ["a.odb", "b.odb"],
            "group_index": [1, 1],
            "instance_index": [1, 2],
            "split": ["val", "val"],
        })
        for offset, column in enumerate(coordinate_target_columns()[1:], start=1):
            frame[column + "_true"] = [float(offset), float(100 + offset)]
            frame[column + "_pred"] = [float(offset + 1), float(101 + offset)]
        captured = {}
        original_plot = plotting._plot_identity_scatter

        def capture_identity_scatter(true_values, pred_values, xlabel, ylabel):
            captured["true_values"] = np.asarray(true_values)
            captured["pred_values"] = np.asarray(pred_values)
            captured["xlabel"] = xlabel
            captured["ylabel"] = ylabel

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                plotting._plot_identity_scatter = capture_identity_scatter
                path = plotting.plot_coordinate_local_strain_pred_vs_true(frame, temp_dir)
            finally:
                plotting._plot_identity_scatter = original_plot

            self.assertTrue(os.path.exists(path))

        np.testing.assert_allclose(captured["true_values"], np.asarray([24.0, 124.0]))
        np.testing.assert_allclose(captured["pred_values"], np.asarray([25.0, 125.0]))
        self.assertEqual(captured["xlabel"], "FEM max local strain concentration")
        self.assertEqual(captured["ylabel"], "CoordinateSurrogate max local strain concentration")


class CoordinatePipelineTests(unittest.TestCase):
    def test_run_coordinate_training_saves_best_validation_model_package(self):
        from cnn_surrogate import training
        from cnn_surrogate.pipeline import run_coordinate_training

        with tempfile.TemporaryDirectory() as temp_dir:
            data_csv = os.path.join(temp_dir, "summary.csv")
            output_dir = os.path.join(temp_dir, "results")
            figure_dir = os.path.join(temp_dir, "figures")
            work_dir = os.path.join(temp_dir, "temp")
            _write_coordinate_csv(data_csv)
            config = _coordinate_config(
                data_csv=data_csv,
                output_dir=output_dir,
                figure_dir=figure_dir,
                temp_dir=work_dir,
                save_model=True,
            )
            config.epochs = 2
            original_run_coordinate_epoch = training.run_coordinate_epoch
            train_call_count = {"value": 0}
            val_losses = [0.1, 0.2]

            def controlled_coordinate_epoch(model, loader, config, optimizer=None, device=None):
                if optimizer is not None:
                    train_call_count["value"] += 1
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.fill_(float(train_call_count["value"]))
                    return float(train_call_count["value"])
                return val_losses.pop(0)

            try:
                training.run_coordinate_epoch = controlled_coordinate_epoch
                run_coordinate_training(config)
            finally:
                training.run_coordinate_epoch = original_run_coordinate_epoch

            package = torch.load(os.path.join(output_dir, "model.pt"), map_location=torch.device("cpu"))

        first_tensor = next(iter(package["model_state_dict"].values()))
        self.assertTrue(torch.allclose(first_tensor, torch.ones_like(first_tensor)))

    def test_run_coordinate_training_writes_expected_artifacts_without_model_package(self):
        from cnn_surrogate.pipeline import run_coordinate_training

        with tempfile.TemporaryDirectory() as temp_dir:
            data_csv = os.path.join(temp_dir, "summary.csv")
            output_dir = os.path.join(temp_dir, "results")
            figure_dir = os.path.join(temp_dir, "figures")
            work_dir = os.path.join(temp_dir, "temp")
            _write_coordinate_csv(data_csv)
            config = _coordinate_config(
                data_csv=data_csv,
                output_dir=output_dir,
                figure_dir=figure_dir,
                temp_dir=work_dir,
                save_model=False,
            )

            result = run_coordinate_training(config)
            predictions = pd.read_csv(os.path.join(output_dir, "predictions.csv"))

            self.assertIn("history", result)
            self.assertIn("metrics", result)
            self.assertIn("predictions", result)
            for filename in ["split_manifest.csv", "train_history.csv", "metrics.json", "predictions.csv"]:
                self.assertTrue(os.path.exists(os.path.join(output_dir, filename)))
            for filename in [
                "loss_curve.png",
                "stiffness_pred_vs_true.png",
                "local_strain_pred_vs_true.png",
                "local_strain_rmse_by_hole.png",
            ]:
                self.assertTrue(os.path.exists(os.path.join(figure_dir, filename)))
            self.assertIn("hole_24_strain_concentration_factor_true", predictions.columns)
            self.assertIn("hole_24_strain_concentration_factor_pred", predictions.columns)
            self.assertNotIn("max_strain_concentration_factor_true", predictions.columns)
            self.assertFalse(os.path.exists(os.path.join(output_dir, "model.pt")))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "stiffness_scaler.pkl")))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "local_strain_scaler.pkl")))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "target_scaler.pkl")))

    def test_run_coordinate_training_skips_figures_when_disabled(self):
        from cnn_surrogate.pipeline import run_coordinate_training

        with tempfile.TemporaryDirectory() as temp_dir:
            data_csv = os.path.join(temp_dir, "summary.csv")
            output_dir = os.path.join(temp_dir, "results")
            figure_dir = os.path.join(temp_dir, "figures_disabled")
            work_dir = os.path.join(temp_dir, "temp")
            _write_coordinate_csv(data_csv)
            config = _coordinate_config(
                data_csv=data_csv,
                output_dir=output_dir,
                figure_dir=figure_dir,
                temp_dir=work_dir,
                save_model=False,
            )
            config.save_figures = False

            run_coordinate_training(config)

            self.assertFalse(os.path.exists(figure_dir))
            for filename in ["split_manifest.csv", "train_history.csv", "metrics.json", "predictions.csv"]:
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)))


if __name__ == "__main__":
    unittest.main()
