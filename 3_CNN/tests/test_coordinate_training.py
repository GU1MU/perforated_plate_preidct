import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import CoordinateTrainingConfig
from cnn_surrogate.data import CoordinateLayoutDataset, coordinate_target_columns
from cnn_surrogate.losses import coordinate_weighted_mse_loss
from cnn_surrogate import training


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _coordinate_config(
    output_dir="output",
    epochs=1,
    warm_start=False,
    checkpoint_path=None,
    show_progress=False,
    early_stopping_patience=None,
):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    return CoordinateTrainingConfig(
        data_csv="input.csv",
        output_dir=output_dir,
        figure_dir="figures",
        temp_dir="temp",
        train_test_split=1,
        split_shuffle=False,
        random_seed=123,
        coordinate_domain_width=80.0,
        coordinate_domain_height=160.0,
        coordinate_feature_dim=6,
        batch_size=2,
        epochs=epochs,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        dropout=0.0,
        point_hidden_dim=16,
        context_hidden_dim=32,
        loss_weight_stiffness=3.0,
        loss_weight_local_strain=0.5,
        early_stopping_patience=early_stopping_patience,
        device="cpu",
        show_progress=show_progress,
        progress_description="Training coordinate test",
        save_model=False,
        warm_start=warm_start,
        checkpoint_path=checkpoint_path,
    )


def _coordinate_row(instance=1):
    row = {
        "odb_name": "%d_plate.odb" % instance,
        "status": "ok",
        "group_index": 1,
        "instance_index": instance,
        "relative_equivalent_stiffness": 0.75 + 0.001 * instance,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float(8.0 + index + instance)
        row["hole_%02d_y" % index] = float(12.0 + index)
        row["hole_%02d_strain_concentration_factor" % index] = 2.0 + 0.01 * index + 0.001 * instance
    return row


def _coordinate_dataset(count=4):
    frame = pd.DataFrame([_coordinate_row(instance=index) for index in range(1, count + 1)])
    return CoordinateLayoutDataset(
        frame,
        domain_width=80.0,
        domain_height=160.0,
        stiffness_scaler=StandardScaler(),
        local_strain_scaler=StandardScaler(),
        fit_scalers=True,
    )


class CoordinateWeightedLossTests(unittest.TestCase):
    def test_coordinate_weighted_mse_loss_weights_stiffness_and_local_targets(self):
        prediction = torch.zeros((1, 25), dtype=torch.float32)
        stiffness_target = torch.ones((1, 1), dtype=torch.float32)
        local_targets = torch.ones((1, 24), dtype=torch.float32)

        loss = coordinate_weighted_mse_loss(
            prediction,
            stiffness_target,
            local_targets,
            stiffness_weight=3.0,
            local_strain_weight=0.5,
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(tuple(loss.shape), ())
        self.assertAlmostEqual(float(loss.item()), 3.0 + 0.5)


class CoordinateCheckpointSignatureTests(unittest.TestCase):
    def test_coordinate_checkpoint_signature_includes_target_columns(self):
        signature = training.coordinate_checkpoint_signature(_coordinate_config())

        self.assertEqual(signature["coordinate_domain_width"], 80.0)
        self.assertEqual(signature["loss_weight_local_strain"], 0.5)
        self.assertEqual(signature["target_columns"], coordinate_target_columns())


class CoordinateTrainingTests(unittest.TestCase):
    def test_train_coordinate_model_resumes_checkpoint_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _coordinate_config(
                output_dir=temp_dir,
                epochs=1,
                warm_start=True,
                checkpoint_path=os.path.join(temp_dir, "checkpoint.pt"),
            )
            train_dataset = _coordinate_dataset()
            model, history = training.train_coordinate_model(train_dataset, train_dataset, config)

            self.assertEqual([record["epoch"] for record in history], [1])
            self.assertTrue(os.path.exists(config.checkpoint_path))

            resumed_config = _coordinate_config(
                output_dir=temp_dir,
                epochs=2,
                warm_start=True,
                checkpoint_path=config.checkpoint_path,
            )
            resumed_model, resumed_history = training.train_coordinate_model(
                train_dataset,
                train_dataset,
                resumed_config,
            )

            self.assertIsNot(resumed_model, model)
            self.assertEqual([record["epoch"] for record in resumed_history], [1, 2])

    def test_train_coordinate_model_rejects_incompatible_checkpoint_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            config = _coordinate_config(
                output_dir=temp_dir,
                epochs=1,
                warm_start=True,
                checkpoint_path=checkpoint_path,
            )
            train_dataset = _coordinate_dataset()
            training.train_coordinate_model(train_dataset, train_dataset, config)

            incompatible_config = _coordinate_config(
                output_dir=temp_dir,
                epochs=2,
                warm_start=True,
                checkpoint_path=checkpoint_path,
            )
            incompatible_config.point_hidden_dim = 24

            with self.assertRaisesRegex(ValueError, "remove checkpoint.pt or set WARM_START=False"):
                training.train_coordinate_model(train_dataset, train_dataset, incompatible_config)

    def test_train_coordinate_model_rejects_incompatible_checkpoint_scaler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            config = _coordinate_config(
                output_dir=temp_dir,
                epochs=1,
                warm_start=True,
                checkpoint_path=checkpoint_path,
            )
            train_dataset = _coordinate_dataset()
            training.train_coordinate_model(train_dataset, train_dataset, config)

            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            checkpoint["local_strain_scaler_mean"] = (np.asarray(checkpoint["local_strain_scaler_mean"]) + 1.0).tolist()
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "remove checkpoint.pt or set WARM_START=False"):
                training.train_coordinate_model(train_dataset, train_dataset, _coordinate_config(
                    output_dir=temp_dir,
                    epochs=2,
                    warm_start=True,
                    checkpoint_path=checkpoint_path,
                ))

    def test_checkpoint_is_controlled_by_warm_start_not_save_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _coordinate_config(
                output_dir=temp_dir,
                epochs=1,
                warm_start=True,
                checkpoint_path=os.path.join(temp_dir, "checkpoint.pt"),
            )
            self.assertFalse(config.save_model)

            training.train_coordinate_model(_coordinate_dataset(), None, config)

            self.assertTrue(os.path.exists(config.checkpoint_path))

    def test_checkpoint_is_not_written_when_warm_start_is_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _coordinate_config(
                output_dir=temp_dir,
                epochs=1,
                warm_start=False,
                checkpoint_path=os.path.join(temp_dir, "checkpoint.pt"),
            )

            training.train_coordinate_model(_coordinate_dataset(), None, config)

            self.assertFalse(os.path.exists(config.checkpoint_path))


if __name__ == "__main__":
    unittest.main()
