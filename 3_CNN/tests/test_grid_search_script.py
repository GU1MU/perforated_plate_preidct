import importlib.util
import sys
import unittest
from pathlib import Path


CNN_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = CNN_ROOT / "scripts"
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))


def _load_script():
    script_path = SCRIPT_DIR / "grid_search_surrogate.py"
    spec = importlib.util.spec_from_file_location("grid_search_surrogate", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tearDownModule():
    for path in [str(SCRIPT_DIR), str(SRC_DIR)]:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


class GridSearchScriptTests(unittest.TestCase):
    def test_script_defaults_to_cuda_and_new_artifact_flags(self):
        module = _load_script()
        args = module.parse_args(["--model", "cnn"])

        self.assertEqual(args.model, "cnn")
        self.assertEqual(args.device, "cuda")
        self.assertFalse(args.figure)
        self.assertFalse(args.save_all)
        self.assertFalse(args.save_model)
        self.assertEqual(module.EARLY_STOPPING_PATIENCE, 50)
        self.assertEqual(module.TRAIN_TEST_SPLIT, 180)

    def test_script_accepts_figure_save_all_and_save_model(self):
        module = _load_script()
        args = module.parse_args([
            "--model", "cnn",
            "--figure",
            "--save-all",
            "--save-model",
        ])

        self.assertTrue(args.figure)
        self.assertTrue(args.save_all)
        self.assertTrue(args.save_model)

    def test_script_rejects_removed_save_argument(self):
        module = _load_script()

        with self.assertRaises(SystemExit):
            module.parse_args(["--model", "cnn", "--save"])

    def test_main_dispatches_distilled_search(self):
        module = _load_script()
        calls = []

        def fake_run_grid_search(**kwargs):
            calls.append(kwargs)
            return {"best_results": {}}

        original_run_grid_search = module.run_grid_search
        try:
            module.run_grid_search = fake_run_grid_search
            self.assertEqual(module.main(["--model", "distilled", "--device", "cpu", "--save-model"]), 0)
        finally:
            module.run_grid_search = original_run_grid_search

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model_name"], "distilled")
        self.assertEqual(calls[0]["device"], "cpu")
        self.assertTrue(calls[0]["save_best_model"])
        self.assertFalse(calls[0]["save_all"])
        self.assertFalse(calls[0]["save_figures"])
        self.assertEqual(calls[0]["train_test_split"], 180)
        self.assertEqual(calls[0]["early_stopping_patience"], 50)

    def test_main_dispatches_comma_separated_models_in_order(self):
        module = _load_script()
        calls = []

        def fake_run_grid_search(**kwargs):
            calls.append(kwargs)
            return {"best_results": {}}

        original_run_grid_search = module.run_grid_search
        try:
            module.run_grid_search = fake_run_grid_search
            self.assertEqual(module.main(["--model", "cnn,distilled", "--device", "cpu"]), 0)
        finally:
            module.run_grid_search = original_run_grid_search

        self.assertEqual([call["model_name"] for call in calls], ["cnn", "distilled"])
        for call in calls:
            self.assertEqual(call["device"], "cpu")
            self.assertFalse(call["save_best_model"])
            self.assertFalse(call["save_all"])
            self.assertFalse(call["save_figures"])
            self.assertEqual(call["train_test_split"], 180)
            self.assertEqual(call["early_stopping_patience"], 50)

    def test_main_dispatches_coordinate_search(self):
        module = _load_script()
        calls = []

        def fake_run_grid_search(**kwargs):
            calls.append(kwargs)
            return {"best_results": {}}

        original_run_grid_search = module.run_grid_search
        try:
            module.run_grid_search = fake_run_grid_search
            self.assertEqual(module.main(["--model", "coordinate", "--device", "cpu"]), 0)
        finally:
            module.run_grid_search = original_run_grid_search

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model_name"], "coordinate")
        self.assertIs(calls[0]["search_config"], module.grid_search_config_coordinate)
        self.assertIs(calls[0]["base_config_builder"], module.build_coordinate_config)
        self.assertIs(calls[0]["training_runner"], module.run_coordinate_training)
        self.assertEqual(calls[0]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
