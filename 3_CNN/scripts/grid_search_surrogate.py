import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(CNN_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cnn_surrogate import grid_search_config_cnn, grid_search_config_coordinate, grid_search_config_distilled
from cnn_surrogate.grid_search import run_grid_search
from cnn_surrogate.pipeline import run_baseline_training, run_coordinate_training, run_distillation_training

from train_cnn_surrogate import build_config as build_cnn_config
from train_coordinate_surrogate import build_config as build_coordinate_config
from train_distilled_surrogate import build_config as build_distilled_config


TRAIN_TEST_SPLIT = 180
EARLY_STOPPING_PATIENCE = 50
VALID_MODELS = ["cnn", "coordinate", "distilled"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run CNN surrogate hyperparameter grid search.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model family to search: cnn, coordinate, distilled, or a comma-separated list.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device. Defaults to cuda for GPU search; use cpu for local CPU debugging.",
    )
    parser.add_argument(
        "--figure",
        action="store_true",
        help="Save per-trial training figures. Disabled by default to reduce search artifacts.",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Keep every trial output directory instead of pruning non-best trial artifacts.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Retrain the best_average parameter set into the official model result directory.",
    )
    return parser.parse_args(argv)


def parse_model_names(model_value):
    models = [item.strip() for item in model_value.split(",") if item.strip()]
    if not models:
        raise ValueError("--model must include at least one model name")
    invalid = [model for model in models if model not in VALID_MODELS]
    if invalid:
        raise ValueError("unsupported model name(s): %s" % ", ".join(invalid))
    return models


def _search_components(model_name):
    if model_name == "distilled":
        return {
            "search_config": grid_search_config_distilled,
            "base_config_builder": build_distilled_config,
            "training_runner": run_distillation_training,
        }
    if model_name == "coordinate":
        return {
            "search_config": grid_search_config_coordinate,
            "base_config_builder": build_coordinate_config,
            "training_runner": run_coordinate_training,
        }
    return {
        "search_config": grid_search_config_cnn,
        "base_config_builder": build_cnn_config,
        "training_runner": run_baseline_training,
    }


def main(argv=None):
    args = parse_args(argv)
    for model_name in parse_model_names(args.model):
        components = _search_components(model_name)
        run_grid_search(
            model_name=model_name,
            search_config=components["search_config"],
            base_config_builder=components["base_config_builder"],
            training_runner=components["training_runner"],
            results_root=os.path.join("3_CNN", "results"),
            temp_root=os.path.join("3_CNN", "temp"),
            device=args.device,
            train_test_split=TRAIN_TEST_SPLIT,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            save_figures=args.figure,
            save_all=args.save_all,
            save_best_model=args.save_model,
        )
    return 0


if __name__ == "__main__":
    main()
