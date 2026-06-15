import os
import sys
import unittest
from pathlib import Path


CNN_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = CNN_ROOT / "scripts"
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

import train_coordinate_surrogate as coordinate
from cnn_surrogate.config import CoordinateTrainingConfig


def tearDownModule():
    for path in [str(SCRIPT_DIR), str(SRC_DIR)]:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


class TrainCoordinateSurrogateScriptTests(unittest.TestCase):
    def test_default_constants_match_coordinate_training_defaults(self):
        self.assertEqual(coordinate.DATA_CSV, os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv"))
        self.assertEqual(coordinate.OUTPUT_DIR, os.path.join("3_CNN", "results", "coordinate_surrogate"))
        self.assertEqual(coordinate.FIGURE_DIR, os.path.join("3_CNN", "figures", "coordinate_surrogate"))
        self.assertEqual(coordinate.TEMP_DIR, os.path.join("3_CNN", "temp"))
        self.assertEqual(coordinate.TRAIN_TEST_SPLIT, 180)
        self.assertEqual(coordinate.SPLIT_SHUFFLE, True)
        self.assertEqual(coordinate.RANDOM_SEED, 20260611)
        self.assertEqual(coordinate.COORDINATE_DOMAIN_WIDTH, 80.0)
        self.assertEqual(coordinate.COORDINATE_DOMAIN_HEIGHT, 160.0)
        self.assertEqual(coordinate.COORDINATE_FEATURE_DIM, 6)
        self.assertEqual(coordinate.BATCH_SIZE, 32)
        self.assertEqual(coordinate.EPOCHS, 500)
        self.assertEqual(coordinate.LEARNING_RATE, 1.0e-3)
        self.assertEqual(coordinate.WEIGHT_DECAY, 1.0e-4)
        self.assertEqual(coordinate.DROPOUT, 0.2)
        self.assertEqual(coordinate.POINT_HIDDEN_DIM, 128)
        self.assertEqual(coordinate.CONTEXT_HIDDEN_DIM, 256)
        self.assertEqual(coordinate.LOSS_WEIGHT_STIFFNESS, 1.0)
        self.assertEqual(coordinate.LOSS_WEIGHT_LOCAL_STRAIN, 1.0)
        self.assertEqual(coordinate.EARLY_STOPPING_PATIENCE, 50)
        self.assertEqual(coordinate.DEVICE, "auto")
        self.assertEqual(coordinate.SHOW_PROGRESS, True)
        self.assertEqual(coordinate.PROGRESS_DESCRIPTION, "Training coordinate surrogate")
        self.assertEqual(coordinate.SAVE_MODEL, True)
        self.assertEqual(coordinate.WARM_START, True)
        self.assertEqual(coordinate.CHECKPOINT_PATH, os.path.join(coordinate.OUTPUT_DIR, "checkpoint.pt"))

    def test_build_config_maps_constants_to_coordinate_training_config(self):
        config = coordinate.build_config()

        self.assertIsInstance(config, CoordinateTrainingConfig)
        self.assertEqual(config.data_csv, coordinate.DATA_CSV)
        self.assertEqual(config.output_dir, coordinate.OUTPUT_DIR)
        self.assertEqual(config.figure_dir, coordinate.FIGURE_DIR)
        self.assertEqual(config.temp_dir, coordinate.TEMP_DIR)
        self.assertEqual(config.train_test_split, coordinate.TRAIN_TEST_SPLIT)
        self.assertEqual(config.split_shuffle, coordinate.SPLIT_SHUFFLE)
        self.assertEqual(config.random_seed, coordinate.RANDOM_SEED)
        self.assertEqual(config.coordinate_domain_width, coordinate.COORDINATE_DOMAIN_WIDTH)
        self.assertEqual(config.coordinate_domain_height, coordinate.COORDINATE_DOMAIN_HEIGHT)
        self.assertEqual(config.coordinate_feature_dim, coordinate.COORDINATE_FEATURE_DIM)
        self.assertEqual(config.batch_size, coordinate.BATCH_SIZE)
        self.assertEqual(config.epochs, coordinate.EPOCHS)
        self.assertEqual(config.learning_rate, coordinate.LEARNING_RATE)
        self.assertEqual(config.weight_decay, coordinate.WEIGHT_DECAY)
        self.assertEqual(config.dropout, coordinate.DROPOUT)
        self.assertEqual(config.point_hidden_dim, coordinate.POINT_HIDDEN_DIM)
        self.assertEqual(config.context_hidden_dim, coordinate.CONTEXT_HIDDEN_DIM)
        self.assertEqual(config.loss_weight_stiffness, coordinate.LOSS_WEIGHT_STIFFNESS)
        self.assertEqual(config.loss_weight_local_strain, coordinate.LOSS_WEIGHT_LOCAL_STRAIN)
        self.assertEqual(config.early_stopping_patience, coordinate.EARLY_STOPPING_PATIENCE)
        self.assertEqual(config.device, coordinate.DEVICE)
        self.assertEqual(config.show_progress, coordinate.SHOW_PROGRESS)
        self.assertEqual(config.progress_description, coordinate.PROGRESS_DESCRIPTION)
        self.assertEqual(config.save_model, coordinate.SAVE_MODEL)
        self.assertEqual(config.warm_start, coordinate.WARM_START)
        self.assertEqual(config.checkpoint_path, coordinate.CHECKPOINT_PATH)

    def test_main_runs_coordinate_training_with_built_config(self):
        calls = []
        original_run = coordinate.run_coordinate_training

        def fake_run(config):
            calls.append(config)
            return {"history": [], "metrics": {}}

        try:
            coordinate.run_coordinate_training = fake_run

            self.assertEqual(coordinate.main(), 0)
        finally:
            coordinate.run_coordinate_training = original_run

        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], CoordinateTrainingConfig)
        self.assertEqual(calls[0], coordinate.build_config())


if __name__ == "__main__":
    unittest.main()
