import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import BaselineTrainingConfig, DistillationTrainingConfig
from cnn_surrogate import data


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _sample_row(instance=1):
    row = {
        "odb_name": "%d_plate.odb" % instance,
        "status": "ok",
        "group_index": 1,
        "instance_index": instance,
        "relative_equivalent_stiffness": 0.7 + 0.01 * instance,
        "max_strain_concentration_factor": 2.0 + 0.02 * instance,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float(index)
        row["hole_%02d_y" % index] = float(index * 2)
        row["hole_%02d_strain_concentration_factor" % index] = 1.0 + 0.01 * index
    return row


class DistillationConfigTests(unittest.TestCase):
    def test_distillation_config_carries_baseline_and_distillation_fields(self):
        config = DistillationTrainingConfig(
            data_csv="data.csv",
            output_dir="out",
            figure_dir="fig",
            temp_dir="temp",
            train_test_split=3,
            split_shuffle=False,
            random_seed=1,
            pixel_size=2.0,
            image_height=80,
            image_width=40,
            batch_size=2,
            epochs=5,
            learning_rate=1.0e-3,
            weight_decay=1.0e-4,
            dropout=0.2,
            loss_weight_stiffness=1.0,
            loss_weight_strain=1.0,
            early_stopping_patience=30,
            device="auto",
            show_progress=False,
            progress_description="test",
            save_model=False,
            teacher_epochs=7,
            student_epochs=9,
            teacher_learning_rate=2.0e-3,
            student_learning_rate=3.0e-3,
            distill_weight=0.3,
        )

        self.assertIsInstance(config, BaselineTrainingConfig)
        self.assertEqual(config.teacher_epochs, 7)
        self.assertEqual(config.student_epochs, 9)
        self.assertEqual(config.teacher_learning_rate, 2.0e-3)
        self.assertEqual(config.student_learning_rate, 3.0e-3)
        self.assertEqual(config.distill_weight, 0.3)


class DistillationColumnTests(unittest.TestCase):
    def test_local_feature_columns_are_the_24_hole_strain_concentration_columns(self):
        columns = data.local_feature_columns()
        self.assertEqual(len(columns), 24)
        self.assertEqual(columns[0], "hole_01_strain_concentration_factor")
        self.assertEqual(columns[-1], "hole_24_strain_concentration_factor")

    def test_required_distillation_columns_include_baseline_and_local_features(self):
        columns = data.required_distillation_columns()
        for column in data.required_columns():
            self.assertIn(column, columns)
        for column in data.local_feature_columns():
            self.assertIn(column, columns)

    def test_validate_distillation_columns_reports_missing_local_feature_columns(self):
        frame = pd.DataFrame([_sample_row(1)]).drop(columns=["hole_07_strain_concentration_factor"])

        with self.assertRaises(ValueError) as raised:
            data.validate_distillation_columns(frame)

        self.assertIn("hole_07_strain_concentration_factor", str(raised.exception))


class DistillationLayoutDatasetTests(unittest.TestCase):
    def test_distillation_dataset_returns_image_local_features_and_target(self):
        frame = pd.DataFrame([_sample_row(1), _sample_row(2)])
        dataset = data.DistillationLayoutDataset(
            frame,
            image_height=80,
            image_width=40,
            pixel_size=2.0,
            target_scaler=StandardScaler(),
            local_feature_scaler=StandardScaler(),
            fit_scaler=True,
            fit_local_feature_scaler=True,
        )

        image, local_features, target = dataset[0]

        self.assertEqual(tuple(image.shape), (1, 80, 40))
        self.assertEqual(tuple(local_features.shape), (24,))
        self.assertEqual(tuple(target.shape), (2,))
        self.assertIs(dataset.target_scaler.__class__, StandardScaler)
        self.assertIs(dataset.local_feature_scaler.__class__, StandardScaler)

    def test_empty_distillation_dataset_uses_configured_shapes(self):
        frame = pd.DataFrame(columns=data.required_distillation_columns())
        target_scaler = StandardScaler()
        target_scaler.fit(np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32))
        local_feature_scaler = StandardScaler()
        local_feature_scaler.fit(np.ones((2, 24), dtype=np.float32))

        dataset = data.DistillationLayoutDataset(
            frame,
            image_height=12,
            image_width=7,
            pixel_size=2.0,
            target_scaler=target_scaler,
            local_feature_scaler=local_feature_scaler,
            fit_scaler=False,
            fit_local_feature_scaler=False,
        )

        self.assertEqual(dataset.images.shape, (0, 1, 12, 7))
        self.assertEqual(dataset.local_features.shape, (0, 24))
        self.assertEqual(dataset.targets.shape, (0, 2))
        self.assertIs(dataset.target_scaler, target_scaler)
        self.assertIs(dataset.local_feature_scaler, local_feature_scaler)


if __name__ == "__main__":
    unittest.main()
