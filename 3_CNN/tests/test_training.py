import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import BaselineTrainingConfig
from cnn_surrogate import losses
from cnn_surrogate.losses import weighted_mse_loss
from cnn_surrogate import training


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _config(epochs=1, batch_size=2, show_progress=False, early_stopping_patience=None, device="cpu", **overrides):
    values = {
        "data_csv": "input.csv",
        "output_dir": "output",
        "figure_dir": "figures",
        "temp_dir": "temp",
        "train_test_split": 1,
        "split_shuffle": False,
        "random_seed": 123,
        "pixel_size": 2.0,
        "image_height": 80,
        "image_width": 40,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "dropout": 0.2,
        "loss_weight_stiffness": 1.5,
        "loss_weight_strain": 0.5,
        "early_stopping_patience": early_stopping_patience,
        "device": device,
        "show_progress": show_progress,
        "progress_description": "Training test",
        "save_model": False,
    }
    values.update(overrides)
    return BaselineTrainingConfig(**values)


def _spatial_dataset(sample_count=2, with_scalers=False, scaler_offset=0.0):
    images = torch.zeros((sample_count, 1, 80, 40), dtype=torch.float32)
    stiffness_targets = torch.zeros((sample_count, 1), dtype=torch.float32)
    local_maps = torch.zeros((sample_count, 1, 80, 40), dtype=torch.float32)
    local_masks = torch.zeros((sample_count, 1, 80, 40), dtype=torch.float32)
    if sample_count > 0:
        local_masks[:, :, 1, 1] = 1.0
    dataset = TensorDataset(images, stiffness_targets, local_maps, local_masks)
    if with_scalers:
        dataset.stiffness_scaler = StandardScaler().fit([
            [0.0 + scaler_offset],
            [1.0 + scaler_offset],
        ])
        dataset.local_strain_scaler = StandardScaler().fit([
            [0.0 + scaler_offset],
            [1.0 + scaler_offset],
            [2.0 + scaler_offset],
        ])
    return dataset


class WeightedLossTests(unittest.TestCase):
    def test_weighted_mse_loss_returns_scalar_tensor(self):
        prediction = torch.zeros((2, 2), dtype=torch.float32)
        target = torch.ones((2, 2), dtype=torch.float32)
        loss = weighted_mse_loss(
            prediction,
            target,
            stiffness_weight=1.0,
            strain_weight=2.0,
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(tuple(loss.shape), ())

    def test_cnn_spatial_supervision_loss_uses_masked_local_loss(self):
        stiffness_prediction = torch.zeros((2, 1), dtype=torch.float32)
        stiffness_target = torch.ones((2, 1), dtype=torch.float32)
        local_map_prediction = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
        local_map_target = torch.ones((2, 1, 4, 4), dtype=torch.float32)
        local_map_mask = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
        local_map_mask[:, :, 1, 1] = 1.0

        loss = losses.cnn_spatial_supervision_loss(
            stiffness_prediction,
            stiffness_target,
            local_map_prediction,
            local_map_target,
            local_map_mask,
            stiffness_weight=1.0,
            local_strain_weight=1.0,
        )

        self.assertEqual(tuple(loss.shape), ())
        self.assertAlmostEqual(float(loss.item()), 2.0)


class BaselineConfigTests(unittest.TestCase):
    def test_baseline_config_provides_spatial_training_defaults(self):
        config = _config()
        self.assertEqual(config.spatial_pool_height, 10)
        self.assertEqual(config.spatial_pool_width, 5)
        self.assertEqual(config.embedding_dim, 256)
        self.assertEqual(config.loss_weight_local_strain, 1.0)
        self.assertFalse(config.warm_start)
        self.assertIsNone(config.checkpoint_path)


class RunEpochTests(unittest.TestCase):
    def test_run_epoch_consumes_spatial_batches_and_masked_loss(self):
        class ConstantSpatialModel(torch.nn.Module):
            def __init__(self):
                super(ConstantSpatialModel, self).__init__()
                self.value = torch.nn.Parameter(torch.zeros(()))

            def forward(self, images):
                batch_size = images.shape[0]
                stiffness = self.value.expand(batch_size, 1)
                local_map = self.value.expand(batch_size, 1, images.shape[-2], images.shape[-1])
                return stiffness, local_map

        images = torch.zeros((2, 1, 80, 40), dtype=torch.float32)
        stiffness_targets = torch.ones((2, 1), dtype=torch.float32)
        local_maps = torch.ones((2, 1, 80, 40), dtype=torch.float32)
        local_masks = torch.zeros((2, 1, 80, 40), dtype=torch.float32)
        local_masks[:, :, 2, 3] = 1.0
        dataset = TensorDataset(images, stiffness_targets, local_maps, local_masks)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

        loss = training.run_epoch(
            ConstantSpatialModel(),
            loader,
            _config(loss_weight_stiffness=1.5, loss_weight_local_strain=0.5),
            optimizer=None,
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(loss, 2.0)


class ProgressTests(unittest.TestCase):
    def test_iter_progress_writes_tqdm_to_stdout(self):
        calls = []
        original_tqdm = training.tqdm

        def fake_tqdm(iterable, desc, unit, file, dynamic_ncols, leave):
            values = list(iterable)
            calls.append({
                "values": values,
                "desc": desc,
                "unit": unit,
                "file": file,
                "dynamic_ncols": dynamic_ncols,
                "leave": leave,
            })
            return values

        training.tqdm = fake_tqdm
        try:
            result = list(training.iter_progress(range(1, 3), "Visible progress", enabled=True))
        finally:
            training.tqdm = original_tqdm

        self.assertEqual(result, [1, 2])
        self.assertEqual(calls[0]["desc"], "Visible progress")
        self.assertEqual(calls[0]["unit"], "epoch")
        self.assertIs(calls[0]["file"], sys.stdout)
        self.assertTrue(calls[0]["dynamic_ncols"])
        self.assertTrue(calls[0]["leave"])


class TrainModelTests(unittest.TestCase):
    def test_train_model_wraps_epochs_with_iter_progress(self):
        calls = []
        original_iter_progress = training.iter_progress

        def fake_iter_progress(iterable, description, enabled=True):
            values = list(iterable)
            calls.append((values, description, enabled))
            return values

        training.iter_progress = fake_iter_progress
        try:
            _, history = training.train_model(
                _spatial_dataset(),
                None,
                _config(epochs=2, batch_size=2, show_progress=True, embedding_dim=16),
            )
        finally:
            training.iter_progress = original_iter_progress

        self.assertEqual(calls, [([1, 2], "Training test", True)])
        self.assertEqual([record["epoch"] for record in history], [1, 2])

    def test_train_model_rejects_empty_training_dataset(self):
        with self.assertRaisesRegex(ValueError, "training split is empty"):
            training.train_model(_spatial_dataset(sample_count=0), None, _config())

    def test_train_model_stops_after_patience_when_validation_loss_does_not_improve(self):
        calls = []
        original_run_epoch = training.run_epoch

        def fake_run_epoch(model, loader, config, optimizer=None, device=None):
            calls.append(optimizer is not None)
            if optimizer is not None:
                return 0.5
            return 1.0

        training.run_epoch = fake_run_epoch
        try:
            _, history = training.train_model(
                _spatial_dataset(sample_count=4),
                _spatial_dataset(sample_count=4),
                _config(epochs=5, batch_size=2, early_stopping_patience=1, embedding_dim=16),
            )
        finally:
            training.run_epoch = original_run_epoch

        self.assertEqual([record["epoch"] for record in history], [1, 2])
        self.assertEqual(calls, [True, False, True, False])

    def test_train_model_passes_resolved_device_to_epoch_runner(self):
        calls = []
        original_run_epoch = training.run_epoch

        def fake_run_epoch(model, loader, config, optimizer=None, device=None):
            calls.append((torch.device(device).type, config.loss_weight_local_strain))
            return 0.5

        training.run_epoch = fake_run_epoch
        try:
            training.train_model(
                _spatial_dataset(),
                None,
                _config(epochs=1, batch_size=2, device="cpu", loss_weight_local_strain=2.5, embedding_dim=16),
            )
        finally:
            training.run_epoch = original_run_epoch

        self.assertEqual(calls, [("cpu", 2.5), ("cpu", 2.5)])

    def test_cnn_checkpoint_signature_includes_spatial_target_contract(self):
        signature = training.cnn_checkpoint_signature(_config(
            embedding_dim=128,
            spatial_pool_height=8,
            spatial_pool_width=4,
            loss_weight_local_strain=2.5,
        ))
        self.assertEqual(signature["embedding_dim"], 128)
        self.assertEqual(signature["spatial_pool_height"], 8)
        self.assertEqual(signature["spatial_pool_width"], 4)
        self.assertEqual(signature["loss_weight_local_strain"], 2.5)
        self.assertEqual(len(signature["target_columns"]), 25)
        self.assertEqual(signature["target_columns"][0], "relative_equivalent_stiffness")
        self.assertEqual(signature["target_columns"][-1], "hole_24_strain_concentration_factor")

    def test_train_model_writes_checkpoint_when_warm_start_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            config = _config(
                epochs=1,
                batch_size=2,
                embedding_dim=16,
                warm_start=True,
                checkpoint_path=checkpoint_path,
            )

            _, history = training.train_model(_spatial_dataset(), None, config)

            self.assertTrue(os.path.exists(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["history"], history)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("best_val_loss", checkpoint)
            self.assertIn("stale_epoch_count", checkpoint)
            self.assertEqual(checkpoint["config_signature"], training.cnn_checkpoint_signature(config))

    def test_train_model_checkpoint_records_cnn_scaler_states(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            config = _config(
                epochs=1,
                batch_size=2,
                embedding_dim=16,
                warm_start=True,
                checkpoint_path=checkpoint_path,
            )

            training.train_model(_spatial_dataset(with_scalers=True), None, config)

            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.assertIn("scaler_states", checkpoint)
            self.assertIn("stiffness_scaler", checkpoint["scaler_states"])
            self.assertIn("local_strain_scaler", checkpoint["scaler_states"])

    def test_train_model_resumes_checkpoint_history_when_signature_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            first_config = _config(
                epochs=1,
                batch_size=2,
                embedding_dim=16,
                warm_start=True,
                checkpoint_path=checkpoint_path,
            )
            training.train_model(_spatial_dataset(), None, first_config)

            second_config = _config(
                epochs=2,
                batch_size=2,
                embedding_dim=16,
                warm_start=True,
                checkpoint_path=checkpoint_path,
            )
            _, history = training.train_model(_spatial_dataset(), None, second_config)

            self.assertEqual([record["epoch"] for record in history], [1, 2])

    def test_train_model_rejects_checkpoint_when_cnn_scalers_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            first_config = _config(
                epochs=1,
                batch_size=2,
                embedding_dim=16,
                warm_start=True,
                checkpoint_path=checkpoint_path,
            )
            training.train_model(_spatial_dataset(with_scalers=True), None, first_config)

            second_config = _config(
                epochs=2,
                batch_size=2,
                embedding_dim=16,
                warm_start=True,
                checkpoint_path=checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "remove checkpoint.pt or set WARM_START=False"):
                training.train_model(
                    _spatial_dataset(with_scalers=True, scaler_offset=10.0),
                    None,
                    second_config,
                )


if __name__ == "__main__":
    unittest.main()
