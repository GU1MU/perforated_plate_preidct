from __future__ import print_function

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(CNN_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cnn_surrogate.config import DistillationTrainingConfig
from cnn_surrogate.pipeline import run_distillation_training


DATA_CSV = os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv")
OUTPUT_DIR = os.path.join("3_CNN", "results", "distilled_surrogate")
FIGURE_DIR = os.path.join("3_CNN", "figures", "distilled_surrogate")
TEMP_DIR = os.path.join("3_CNN", "temp")

TRAIN_TEST_SPLIT = 25
SPLIT_SHUFFLE = True
RANDOM_SEED = 20260611

PIXEL_SIZE = 2.0
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 40

BATCH_SIZE = 32
TEACHER_EPOCHS = 300
STUDENT_EPOCHS = 300
TEACHER_LEARNING_RATE = 1.0e-3
STUDENT_LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
DROPOUT = 0.2

LOSS_WEIGHT_STIFFNESS = 1.0
LOSS_WEIGHT_STRAIN = 1.0
DISTILL_WEIGHT = 0.3
EARLY_STOPPING_PATIENCE = 30
DEVICE = "auto"

SHOW_PROGRESS = True
PROGRESS_DESCRIPTION = "Training distilled CNN surrogate"

SAVE_MODEL = True


def build_config():
    return DistillationTrainingConfig(
        data_csv=DATA_CSV,
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        temp_dir=TEMP_DIR,
        train_test_split=TRAIN_TEST_SPLIT,
        split_shuffle=SPLIT_SHUFFLE,
        random_seed=RANDOM_SEED,
        pixel_size=PIXEL_SIZE,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        batch_size=BATCH_SIZE,
        epochs=STUDENT_EPOCHS,
        learning_rate=STUDENT_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        loss_weight_stiffness=LOSS_WEIGHT_STIFFNESS,
        loss_weight_strain=LOSS_WEIGHT_STRAIN,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        device=DEVICE,
        show_progress=SHOW_PROGRESS,
        progress_description=PROGRESS_DESCRIPTION,
        save_model=SAVE_MODEL,
        teacher_epochs=TEACHER_EPOCHS,
        student_epochs=STUDENT_EPOCHS,
        teacher_learning_rate=TEACHER_LEARNING_RATE,
        student_learning_rate=STUDENT_LEARNING_RATE,
        distill_weight=DISTILL_WEIGHT,
    )


def main():
    run_distillation_training(build_config())
    return 0


if __name__ == "__main__":
    main()
