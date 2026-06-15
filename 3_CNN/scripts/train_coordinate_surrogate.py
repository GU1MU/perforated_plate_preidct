import os
import sys

DATA_CSV = os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv")
OUTPUT_DIR = os.path.join("3_CNN", "results", "coordinate_surrogate")
FIGURE_DIR = os.path.join("3_CNN", "figures", "coordinate_surrogate")
TEMP_DIR = os.path.join("3_CNN", "temp")

TRAIN_TEST_SPLIT = 180
SPLIT_SHUFFLE = True
RANDOM_SEED = 20260611

COORDINATE_DOMAIN_WIDTH = 80.0
COORDINATE_DOMAIN_HEIGHT = 160.0
COORDINATE_FEATURE_DIM = 6

BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
DROPOUT = 0.2
POINT_HIDDEN_DIM = 128
CONTEXT_HIDDEN_DIM = 256

LOSS_WEIGHT_STIFFNESS = 1.0
LOSS_WEIGHT_LOCAL_STRAIN = 1.0
EARLY_STOPPING_PATIENCE = 50
DEVICE = "auto"

SHOW_PROGRESS = True
PROGRESS_DESCRIPTION = "Training coordinate surrogate"

SAVE_MODEL = True
SAVE_FIGURES = True
WARM_START = True
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(CNN_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cnn_surrogate.config import CoordinateTrainingConfig
from cnn_surrogate.pipeline import run_coordinate_training


def build_config():
    return CoordinateTrainingConfig(
        data_csv=DATA_CSV,
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        temp_dir=TEMP_DIR,
        train_test_split=TRAIN_TEST_SPLIT,
        split_shuffle=SPLIT_SHUFFLE,
        random_seed=RANDOM_SEED,
        coordinate_domain_width=COORDINATE_DOMAIN_WIDTH,
        coordinate_domain_height=COORDINATE_DOMAIN_HEIGHT,
        coordinate_feature_dim=COORDINATE_FEATURE_DIM,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        point_hidden_dim=POINT_HIDDEN_DIM,
        context_hidden_dim=CONTEXT_HIDDEN_DIM,
        loss_weight_stiffness=LOSS_WEIGHT_STIFFNESS,
        loss_weight_local_strain=LOSS_WEIGHT_LOCAL_STRAIN,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        device=DEVICE,
        show_progress=SHOW_PROGRESS,
        progress_description=PROGRESS_DESCRIPTION,
        save_model=SAVE_MODEL,
        save_figures=SAVE_FIGURES,
        warm_start=WARM_START,
        checkpoint_path=CHECKPOINT_PATH,
    )


def main():
    run_coordinate_training(build_config())
    return 0


if __name__ == "__main__":
    main()
