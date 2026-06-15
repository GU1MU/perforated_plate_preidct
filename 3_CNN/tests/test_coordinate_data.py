import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import CoordinateTrainingConfig
from cnn_surrogate import data


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _coordinate_row(group=1, instance=1):
    row = {
        "odb_name": "%d_%d_plate.odb" % (group, instance),
        "status": "ok",
        "group_index": group,
        "instance_index": instance,
        "relative_equivalent_stiffness": 0.75 + 0.001 * instance,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float(8.0 + index)
        row["hole_%02d_y" % index] = float(12.0 + index)
        row["hole_%02d_strain_concentration_factor" % index] = 2.0 + 0.01 * index
    return row


class CoordinateConfigTests(unittest.TestCase):
    def test_coordinate_config_carries_model_and_feature_parameters(self):
        config = CoordinateTrainingConfig(
            data_csv="input.csv",
            output_dir="output",
            figure_dir="figures",
            temp_dir="temp",
            train_test_split=180,
            split_shuffle=True,
            random_seed=20260611,
            coordinate_domain_width=80.0,
            coordinate_domain_height=160.0,
            coordinate_feature_dim=6,
            batch_size=32,
            epochs=500,
            learning_rate=1.0e-3,
            weight_decay=1.0e-4,
            dropout=0.2,
            point_hidden_dim=128,
            context_hidden_dim=256,
            loss_weight_stiffness=1.0,
            loss_weight_local_strain=1.0,
            early_stopping_patience=50,
            device="auto",
            show_progress=True,
            progress_description="Training coordinate surrogate",
            save_model=True,
            warm_start=True,
            checkpoint_path="checkpoint.pt",
        )
        self.assertEqual(config.coordinate_feature_dim, 6)
        self.assertEqual(config.point_hidden_dim, 128)
        self.assertEqual(config.context_hidden_dim, 256)
        self.assertTrue(config.warm_start)
        self.assertEqual(config.checkpoint_path, "checkpoint.pt")


class CoordinateColumnTests(unittest.TestCase):
    def test_coordinate_target_columns_are_stiffness_plus_24_local_strain_targets(self):
        columns = data.coordinate_target_columns()
        self.assertEqual(len(columns), 25)
        self.assertEqual(columns[0], "relative_equivalent_stiffness")
        self.assertEqual(columns[1], "hole_01_strain_concentration_factor")
        self.assertEqual(columns[-1], "hole_24_strain_concentration_factor")
        self.assertNotIn("max_strain_concentration_factor", columns)

    def test_coordinate_required_columns_include_inputs_and_targets(self):
        columns = data.required_coordinate_columns()
        self.assertIn("hole_01_x", columns)
        self.assertIn("hole_24_y", columns)
        self.assertIn("relative_equivalent_stiffness", columns)
        self.assertIn("hole_24_strain_concentration_factor", columns)

    def test_coordinate_feature_names_are_six_inspectable_features(self):
        self.assertEqual(data.coordinate_feature_names(), [
            "x_norm",
            "y_norm",
            "left_distance_norm",
            "right_distance_norm",
            "bottom_distance_norm",
            "top_distance_norm",
        ])


class CoordinateTableTests(unittest.TestCase):
    def test_load_coordinate_dataset_table_keeps_ok_complete_rows_and_casts_indices(self):
        rows = [
            _coordinate_row(group=1, instance=1),
            _coordinate_row(group=1, instance=2),
            _coordinate_row(group=1, instance=3),
        ]
        rows[1]["status"] = "failed"
        rows[2]["hole_24_strain_concentration_factor"] = np.nan
        rows[0]["group_index"] = "1"
        rows[0]["instance_index"] = "1"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "summary.csv")
            pd.DataFrame(rows).to_csv(path, index=False)

            frame = data.load_coordinate_dataset_table(path)

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "odb_name"], "1_1_plate.odb")
        self.assertEqual(frame.index.tolist(), [0])
        self.assertTrue(np.issubdtype(frame["group_index"].dtype, np.integer))
        self.assertTrue(np.issubdtype(frame["instance_index"].dtype, np.integer))

    def test_validate_coordinate_columns_reports_missing_columns(self):
        frame = pd.DataFrame([_coordinate_row()]).drop(columns=["hole_12_y"])

        with self.assertRaises(ValueError) as raised:
            data.validate_coordinate_columns(frame)

        self.assertIn("hole_12_y", str(raised.exception))


class CoordinateEncodingTests(unittest.TestCase):
    def test_encode_coordinate_features_returns_24_by_6_features(self):
        row = _coordinate_row()
        row["hole_01_x"] = 20.0
        row["hole_01_y"] = 40.0

        features = data.encode_coordinate_features(row, domain_width=80.0, domain_height=160.0)

        self.assertEqual(features.shape, (24, 6))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_allclose(
            features[0],
            np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.75], dtype=np.float32),
        )


class CoordinateLayoutDatasetTests(unittest.TestCase):
    def test_coordinate_dataset_returns_coordinate_tensor_and_scaled_target(self):
        frame = pd.DataFrame([_coordinate_row(instance=1), _coordinate_row(instance=2)])

        dataset = data.CoordinateLayoutDataset(
            frame,
            domain_width=80.0,
            domain_height=160.0,
            stiffness_scaler=StandardScaler(),
            local_strain_scaler=StandardScaler(),
            fit_scalers=True,
        )

        coordinates, stiffness_target, local_targets = dataset[0]

        self.assertEqual(tuple(coordinates.shape), (24, 6))
        self.assertEqual(tuple(stiffness_target.shape), (1,))
        self.assertEqual(tuple(local_targets.shape), (24,))
        self.assertEqual(dataset.coordinates.shape, (2, 24, 6))
        self.assertEqual(dataset.stiffness_targets.shape, (2, 1))
        self.assertEqual(dataset.local_targets.shape, (2, 24))
        self.assertIs(dataset.stiffness_scaler.__class__, StandardScaler)
        self.assertIs(dataset.local_strain_scaler.__class__, StandardScaler)

    def test_coordinate_dataset_uses_existing_scalers_without_refitting(self):
        frame = pd.DataFrame([_coordinate_row(instance=1), _coordinate_row(instance=2)])
        stiffness_scaler = StandardScaler()
        stiffness_scaler.fit(np.ones((2, 1), dtype=np.float32))
        local_strain_scaler = StandardScaler()
        local_strain_scaler.fit(np.ones((48, 1), dtype=np.float32))

        dataset = data.CoordinateLayoutDataset(
            frame,
            domain_width=80.0,
            domain_height=160.0,
            stiffness_scaler=stiffness_scaler,
            local_strain_scaler=local_strain_scaler,
            fit_scalers=False,
        )

        self.assertIs(dataset.stiffness_scaler, stiffness_scaler)
        self.assertIs(dataset.local_strain_scaler, local_strain_scaler)
        self.assertEqual(dataset.stiffness_targets.shape, (2, 1))
        self.assertEqual(dataset.local_targets.shape, (2, 24))

    def test_empty_coordinate_dataset_uses_configured_shapes(self):
        frame = pd.DataFrame(columns=data.required_coordinate_columns())
        stiffness_scaler = StandardScaler()
        stiffness_scaler.fit(np.ones((2, 1), dtype=np.float32))
        local_strain_scaler = StandardScaler()
        local_strain_scaler.fit(np.ones((48, 1), dtype=np.float32))

        dataset = data.CoordinateLayoutDataset(
            frame,
            domain_width=80.0,
            domain_height=160.0,
            stiffness_scaler=stiffness_scaler,
            local_strain_scaler=local_strain_scaler,
            fit_scalers=False,
        )

        self.assertEqual(dataset.coordinates.shape, (0, 24, 6))
        self.assertEqual(dataset.stiffness_targets.shape, (0, 1))
        self.assertEqual(dataset.local_targets.shape, (0, 24))
        self.assertIs(dataset.stiffness_scaler, stiffness_scaler)
        self.assertIs(dataset.local_strain_scaler, local_strain_scaler)


if __name__ == "__main__":
    unittest.main()
