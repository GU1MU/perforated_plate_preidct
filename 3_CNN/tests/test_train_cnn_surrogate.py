import inspect
import os
import sys
import unittest
from pathlib import Path


CNN_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = CNN_ROOT / "scripts"
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

import train_cnn_surrogate as cnn
from cnn_surrogate.config import BaselineTrainingConfig


def tearDownModule():
    for path in [str(SCRIPT_DIR), str(SRC_DIR)]:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


class TrainCnnSurrogateScriptTests(unittest.TestCase):
    def test_default_constants_match_baseline_training_defaults(self):
        self.assertEqual(cnn.DATA_CSV, os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv"))
        self.assertEqual(cnn.OUTPUT_DIR, os.path.join("3_CNN", "results", "cnn_surrogate"))
        self.assertEqual(cnn.FIGURE_DIR, os.path.join("3_CNN", "figures", "cnn_surrogate"))
        self.assertEqual(cnn.TEMP_DIR, os.path.join("3_CNN", "temp"))
        self.assertEqual(cnn.TRAIN_TEST_SPLIT, 180)
        self.assertEqual(cnn.SPLIT_SHUFFLE, True)
        self.assertEqual(cnn.RANDOM_SEED, 20260611)
        self.assertEqual(cnn.PIXEL_SIZE, 2.0)
        self.assertEqual(cnn.IMAGE_HEIGHT, 80)
        self.assertEqual(cnn.IMAGE_WIDTH, 40)
        self.assertEqual(cnn.BATCH_SIZE, 32)
        self.assertEqual(cnn.EPOCHS, 300)
        self.assertEqual(cnn.LEARNING_RATE, 1.0e-3)
        self.assertEqual(cnn.WEIGHT_DECAY, 1.0e-4)
        self.assertEqual(cnn.DROPOUT, 0.2)
        self.assertEqual(cnn.LOSS_WEIGHT_STIFFNESS, 1.0)
        self.assertEqual(cnn.SPATIAL_POOL_HEIGHT, 10)
        self.assertEqual(cnn.SPATIAL_POOL_WIDTH, 5)
        self.assertEqual(cnn.EMBEDDING_DIM, 256)
        self.assertEqual(cnn.LOSS_WEIGHT_LOCAL_STRAIN, 1.0)
        self.assertTrue(cnn.WARM_START)
        self.assertEqual(cnn.CHECKPOINT_PATH, os.path.join(cnn.OUTPUT_DIR, "checkpoint.pt"))
        self.assertFalse(hasattr(cnn, "LOSS_WEIGHT_STRAIN"))
        self.assertEqual(cnn.EARLY_STOPPING_PATIENCE, 30)
        self.assertEqual(cnn.DEVICE, "auto")
        self.assertEqual(cnn.SHOW_PROGRESS, True)
        self.assertEqual(cnn.PROGRESS_DESCRIPTION, "Training CNN surrogate")
        self.assertEqual(cnn.SAVE_MODEL, True)
        self.assertEqual(cnn.SAVE_FIGURES, True)

    def test_build_config_maps_constants_to_training_config(self):
        config = cnn.build_config()

        self.assertIsInstance(config, BaselineTrainingConfig)
        self.assertEqual(config.data_csv, cnn.DATA_CSV)
        self.assertEqual(config.output_dir, cnn.OUTPUT_DIR)
        self.assertEqual(config.figure_dir, cnn.FIGURE_DIR)
        self.assertEqual(config.temp_dir, cnn.TEMP_DIR)
        self.assertEqual(config.train_test_split, cnn.TRAIN_TEST_SPLIT)
        self.assertEqual(config.train_test_split, 180)
        self.assertEqual(config.split_shuffle, cnn.SPLIT_SHUFFLE)
        self.assertEqual(config.random_seed, cnn.RANDOM_SEED)
        self.assertEqual(config.pixel_size, cnn.PIXEL_SIZE)
        self.assertEqual(config.image_height, cnn.IMAGE_HEIGHT)
        self.assertEqual(config.image_width, cnn.IMAGE_WIDTH)
        self.assertEqual(config.batch_size, cnn.BATCH_SIZE)
        self.assertEqual(config.epochs, cnn.EPOCHS)
        self.assertEqual(config.learning_rate, cnn.LEARNING_RATE)
        self.assertEqual(config.weight_decay, cnn.WEIGHT_DECAY)
        self.assertEqual(config.dropout, cnn.DROPOUT)
        self.assertEqual(config.loss_weight_stiffness, cnn.LOSS_WEIGHT_STIFFNESS)
        self.assertEqual(config.spatial_pool_height, cnn.SPATIAL_POOL_HEIGHT)
        self.assertEqual(config.spatial_pool_width, cnn.SPATIAL_POOL_WIDTH)
        self.assertEqual(config.embedding_dim, cnn.EMBEDDING_DIM)
        self.assertEqual(config.loss_weight_local_strain, cnn.LOSS_WEIGHT_LOCAL_STRAIN)
        self.assertEqual(config.warm_start, cnn.WARM_START)
        self.assertEqual(config.checkpoint_path, cnn.CHECKPOINT_PATH)
        self.assertEqual(config.early_stopping_patience, cnn.EARLY_STOPPING_PATIENCE)
        self.assertEqual(config.device, cnn.DEVICE)
        self.assertEqual(config.show_progress, cnn.SHOW_PROGRESS)
        self.assertEqual(config.progress_description, cnn.PROGRESS_DESCRIPTION)
        self.assertEqual(config.save_model, cnn.SAVE_MODEL)
        self.assertEqual(config.save_figures, cnn.SAVE_FIGURES)

    def test_build_config_adds_run_id_to_result_paths(self):
        self.assertIn("run_id", inspect.signature(cnn.build_config).parameters)

        config = cnn.build_config(run_id="abc")

        expected_output_dir = os.path.join("3_CNN", "results", "cnn_surrogate_abc")
        expected_figure_dir = os.path.join("3_CNN", "figures", "cnn_surrogate_abc")
        self.assertEqual(config.output_dir, expected_output_dir)
        self.assertEqual(config.figure_dir, expected_figure_dir)
        self.assertEqual(config.temp_dir, cnn.TEMP_DIR)
        self.assertEqual(config.checkpoint_path, os.path.join(expected_output_dir, "checkpoint.pt"))
        self.assertTrue(config.save_figures)

    def test_main_runs_baseline_training_with_built_config(self):
        calls = []
        original_run = cnn.run_baseline_training

        def fake_run(config):
            calls.append(config)
            return {"history": [], "metrics": {}}

        try:
            cnn.run_baseline_training = fake_run

            self.assertEqual(cnn.main(), 0)
        finally:
            cnn.run_baseline_training = original_run

        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], BaselineTrainingConfig)
        self.assertEqual(calls[0], cnn.build_config())

    def test_main_parses_run_id_for_training_config(self):
        self.assertIn("argv", inspect.signature(cnn.main).parameters)

        calls = []
        original_run = cnn.run_baseline_training

        def fake_run(config):
            calls.append(config)
            return {"history": [], "metrics": {}}

        try:
            cnn.run_baseline_training = fake_run

            self.assertEqual(cnn.main(["--id", "abc"]), 0)
        finally:
            cnn.run_baseline_training = original_run

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].output_dir, os.path.join("3_CNN", "results", "cnn_surrogate_abc"))
        self.assertEqual(calls[0].figure_dir, os.path.join("3_CNN", "figures", "cnn_surrogate_abc"))
        self.assertEqual(
            calls[0].checkpoint_path,
            os.path.join("3_CNN", "results", "cnn_surrogate_abc", "checkpoint.pt"),
        )


if __name__ == "__main__":
    unittest.main()
