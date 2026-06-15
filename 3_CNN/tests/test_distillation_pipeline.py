import os
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import DistillationTrainingConfig
from cnn_surrogate.data import DistillationLayoutDataset, HoleLayoutDataset, local_feature_columns
from cnn_surrogate.evaluation import predict_student_frame, predict_teacher_frame
from cnn_surrogate.io import save_distillation_package
from cnn_surrogate.pipeline import run_distillation_training
from cnn_surrogate.plotting import plot_teacher_vs_student


EXPECTED_TEACHER_COLUMNS = [
    "odb_name",
    "group_index",
    "instance_index",
    "split",
    "relative_equivalent_stiffness_true",
    "max_strain_concentration_factor_true",
    "relative_equivalent_stiffness_teacher_pred",
    "max_strain_concentration_factor_teacher_pred",
]

EXPECTED_STUDENT_COLUMNS = [
    "odb_name",
    "group_index",
    "instance_index",
    "split",
    "relative_equivalent_stiffness_true",
    "max_strain_concentration_factor_true",
    "relative_equivalent_stiffness_student_pred",
    "max_strain_concentration_factor_student_pred",
]


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _sample_row(group=1, instance=1, split="train"):
    row = {
        "odb_name": "%d_%d_plate.odb" % (group, instance),
        "status": "ok",
        "group_index": group,
        "instance_index": instance,
        "split": split,
        "relative_equivalent_stiffness": 0.6 + 0.01 * instance,
        "max_strain_concentration_factor": 1.8 + 0.02 * instance,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float((index + instance) % 40)
        row["hole_%02d_y" % index] = float((2 * index + group) % 80)
        row["hole_%02d_strain_concentration_factor" % index] = 1.0 + 0.01 * index
    return row


def _smoke_frame():
    rows = []
    for group in range(1, 3):
        for instance in range(1, 5):
            rows.append(_sample_row(group=group, instance=instance))
    return pd.DataFrame(rows)


def _dataset():
    frame = pd.DataFrame([
        _sample_row(instance=1, split="train"),
        _sample_row(instance=2, split="test"),
    ])
    return DistillationLayoutDataset(
        frame,
        image_height=80,
        image_width=40,
        pixel_size=2.0,
        target_scaler=StandardScaler(),
        local_feature_scaler=StandardScaler(),
        fit_scalers=True,
    )


def _config(save_model):
    return DistillationTrainingConfig(
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
        epochs=1,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        dropout=0.2,
        loss_weight_stiffness=1.0,
        loss_weight_strain=1.0,
        early_stopping_patience=None,
        device="cpu",
        show_progress=False,
        progress_description="Training distillation test",
        save_model=save_model,
        teacher_epochs=1,
        student_epochs=1,
        teacher_learning_rate=1.0e-3,
        student_learning_rate=1.0e-3,
        distill_weight=0.3,
    )


class ConstantTeacher(nn.Module):
    def forward(self, images, local_features):
        return torch.zeros((images.shape[0], 2), dtype=images.dtype)


class ImageOnlyStudent(nn.Module):
    def forward(self, images):
        return torch.zeros((images.shape[0], 2), dtype=images.dtype)


class DistillationPipelineTask4Tests(unittest.TestCase):
    def test_predict_teacher_frame_creates_teacher_true_and_prediction_columns(self):
        predictions = predict_teacher_frame(ConstantTeacher(), _dataset(), batch_size=2)

        self.assertEqual(list(predictions.columns), EXPECTED_TEACHER_COLUMNS)
        self.assertEqual(len(predictions), 2)

    def test_predict_student_frame_creates_student_true_and_prediction_columns(self):
        predictions = predict_student_frame(ImageOnlyStudent(), _dataset(), batch_size=2)

        self.assertEqual(list(predictions.columns), EXPECTED_STUDENT_COLUMNS)
        self.assertEqual(len(predictions), 2)

    def test_predict_student_frame_accepts_image_only_dataset_without_local_features(self):
        frame = pd.DataFrame([
            _sample_row(instance=1, split="train"),
            _sample_row(instance=2, split="test"),
        ]).drop(columns=local_feature_columns())
        dataset = HoleLayoutDataset(
            frame,
            image_height=80,
            image_width=40,
            pixel_size=2.0,
            target_scaler=StandardScaler(),
            fit_scaler=True,
        )

        predictions = predict_student_frame(ImageOnlyStudent(), dataset, batch_size=2)

        self.assertEqual(list(predictions.columns), EXPECTED_STUDENT_COLUMNS)
        self.assertEqual(len(predictions), 2)

    def test_plot_teacher_vs_student_writes_figure(self):
        teacher_predictions = predict_teacher_frame(ConstantTeacher(), _dataset(), batch_size=2)
        student_predictions = predict_student_frame(ImageOnlyStudent(), _dataset(), batch_size=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = plot_teacher_vs_student(teacher_predictions, student_predictions, temp_dir)

            self.assertEqual(os.path.basename(path), "teacher_vs_student.png")
            self.assertTrue(os.path.isfile(path))

    def test_save_distillation_package_writes_and_removes_package_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_distillation_package(
                ConstantTeacher(),
                ImageOnlyStudent(),
                StandardScaler(),
                StandardScaler(),
                temp_dir,
                _config(save_model=True),
            )
            for filename in [
                "teacher_model.pt",
                "student_model.pt",
                "target_scaler.pkl",
                "local_feature_scaler.pkl",
            ]:
                self.assertTrue(os.path.isfile(os.path.join(temp_dir, filename)))
            teacher_package = torch.load(os.path.join(temp_dir, "teacher_model.pt"), map_location="cpu")
            student_package = torch.load(os.path.join(temp_dir, "student_model.pt"), map_location="cpu")
            self.assertIn("local_feature_columns", teacher_package)
            self.assertNotIn("local_feature_columns", student_package)

            save_distillation_package(
                ConstantTeacher(),
                ImageOnlyStudent(),
                StandardScaler(),
                StandardScaler(),
                temp_dir,
                _config(save_model=False),
            )
            for filename in [
                "teacher_model.pt",
                "student_model.pt",
                "target_scaler.pkl",
                "local_feature_scaler.pkl",
            ]:
                self.assertFalse(os.path.exists(os.path.join(temp_dir, filename)))


class DistillationPipelineTask5Tests(unittest.TestCase):
    def test_run_distillation_training_writes_expected_outputs_without_model_packages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_csv = os.path.join(temp_dir, "distillation_input.csv")
            output_dir = os.path.join(temp_dir, "output")
            figure_dir = os.path.join(temp_dir, "figures")
            temp_work_dir = os.path.join(temp_dir, "temp")
            _smoke_frame().to_csv(data_csv, index=False)

            config = DistillationTrainingConfig(
                data_csv=data_csv,
                output_dir=output_dir,
                figure_dir=figure_dir,
                temp_dir=temp_work_dir,
                train_test_split=2,
                split_shuffle=False,
                random_seed=123,
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
                early_stopping_patience=None,
                device="cpu",
                show_progress=False,
                progress_description="Training distillation smoke test",
                save_model=False,
                teacher_epochs=1,
                student_epochs=1,
                teacher_learning_rate=1.0e-3,
                student_learning_rate=1.0e-3,
                distill_weight=0.3,
            )

            result = run_distillation_training(config)

            for filename in [
                "split_manifest.csv",
                "teacher_train_history.csv",
                "student_train_history.csv",
                "teacher_predictions.csv",
                "student_predictions.csv",
                "teacher_metrics.json",
                "student_metrics.json",
            ]:
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)))
            for filename in [
                "teacher_true_vs_predict.png",
                "student_true_vs_predict.png",
                "teacher_vs_student.png",
            ]:
                self.assertTrue(os.path.isfile(os.path.join(figure_dir, filename)))
            for filename in [
                "teacher_model.pt",
                "student_model.pt",
                "target_scaler.pkl",
                "local_feature_scaler.pkl",
            ]:
                self.assertFalse(os.path.exists(os.path.join(output_dir, filename)))

            self.assertIn("teacher_model", result)
            self.assertIn("student_model", result)
            self.assertEqual(len(result["teacher_predictions"]), 8)
            self.assertEqual(len(result["student_predictions"]), 8)

    def test_run_distillation_training_skips_figures_when_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_csv = os.path.join(temp_dir, "distillation_input.csv")
            output_dir = os.path.join(temp_dir, "output")
            figure_dir = os.path.join(temp_dir, "figures_disabled")
            temp_work_dir = os.path.join(temp_dir, "temp")
            _smoke_frame().to_csv(data_csv, index=False)

            config = _config(save_model=False)
            config.data_csv = data_csv
            config.output_dir = output_dir
            config.figure_dir = figure_dir
            config.temp_dir = temp_work_dir
            config.train_test_split = 2
            config.save_figures = False

            run_distillation_training(config)

            self.assertFalse(os.path.exists(figure_dir))
            for filename in [
                "split_manifest.csv",
                "teacher_train_history.csv",
                "student_train_history.csv",
                "teacher_predictions.csv",
                "student_predictions.csv",
                "teacher_metrics.json",
                "student_metrics.json",
            ]:
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)))

    def test_train_distilled_surrogate_script_exports_expected_config(self):
        script_path = CNN_ROOT / "scripts" / "train_distilled_surrogate.py"
        spec = importlib.util.spec_from_file_location("train_distilled_surrogate", str(script_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(
            module.DATA_CSV,
            os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv"),
        )
        self.assertEqual(module.OUTPUT_DIR, os.path.join("3_CNN", "results", "distilled_surrogate"))
        self.assertEqual(module.FIGURE_DIR, os.path.join("3_CNN", "figures", "distilled_surrogate"))
        self.assertEqual(module.DISTILL_WEIGHT, 0.3)
        self.assertEqual(module.EARLY_STOPPING_PATIENCE, 30)
        self.assertEqual(module.DEVICE, "auto")
        self.assertEqual(module.SAVE_FIGURES, True)
        self.assertEqual(module.build_config().early_stopping_patience, module.EARLY_STOPPING_PATIENCE)
        self.assertEqual(module.build_config().device, module.DEVICE)
        self.assertTrue(module.build_config().save_figures)
        self.assertIsInstance(module.build_config(), DistillationTrainingConfig)


if __name__ == "__main__":
    unittest.main()
