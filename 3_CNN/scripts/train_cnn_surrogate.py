import argparse
import os
import sys

DATA_CSV = os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv")
OUTPUT_DIR = os.path.join("3_CNN", "results", "cnn_surrogate")
FIGURE_DIR = os.path.join("3_CNN", "figures", "cnn_surrogate")
TEMP_DIR = os.path.join("3_CNN", "temp")

TRAIN_TEST_SPLIT = 180
SPLIT_SHUFFLE = True
RANDOM_SEED = 20260611

PIXEL_SIZE = 2.0
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 40

BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
DROPOUT = 0.2

LOSS_WEIGHT_STIFFNESS = 1.0
SPATIAL_POOL_HEIGHT = 10
SPATIAL_POOL_WIDTH = 5
EMBEDDING_DIM = 256
LOSS_WEIGHT_LOCAL_STRAIN = 1.0
EARLY_STOPPING_PATIENCE = 30
DEVICE = "auto"

SHOW_PROGRESS = True
PROGRESS_DESCRIPTION = "Training CNN surrogate"

SAVE_MODEL = True
SAVE_FIGURES = True
WARM_START = True
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(CNN_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cnn_surrogate.config import BaselineTrainingConfig
from cnn_surrogate.pipeline import run_baseline_training


def build_config(run_id=None):
    output_dir = OUTPUT_DIR
    figure_dir = FIGURE_DIR
    if run_id:
        output_dir = OUTPUT_DIR + "_" + run_id
        figure_dir = FIGURE_DIR + "_" + run_id
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")

    return BaselineTrainingConfig(
        data_csv=DATA_CSV,
        output_dir=output_dir,
        figure_dir=figure_dir,
        temp_dir=TEMP_DIR,
        train_test_split=TRAIN_TEST_SPLIT,
        split_shuffle=SPLIT_SHUFFLE,
        random_seed=RANDOM_SEED,
        pixel_size=PIXEL_SIZE,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        loss_weight_stiffness=LOSS_WEIGHT_STIFFNESS,
        loss_weight_strain=LOSS_WEIGHT_LOCAL_STRAIN,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        device=DEVICE,
        show_progress=SHOW_PROGRESS,
        progress_description=PROGRESS_DESCRIPTION,
        save_model=SAVE_MODEL,
        save_figures=SAVE_FIGURES,
        spatial_pool_height=SPATIAL_POOL_HEIGHT,
        spatial_pool_width=SPATIAL_POOL_WIDTH,
        embedding_dim=EMBEDDING_DIM,
        loss_weight_local_strain=LOSS_WEIGHT_LOCAL_STRAIN,
        warm_start=WARM_START,
        checkpoint_path=checkpoint_path,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the CNN surrogate model.")
    parser.add_argument("--id", dest="run_id", default=None, help="Optional run id suffix for output directories.")
    args = parser.parse_args([] if argv is None else argv)

    run_baseline_training(build_config(run_id=args.run_id))
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
