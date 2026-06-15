# CNN Teacher-Student Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a teacher-student surrogate workflow where the teacher can use training-only local FEM strain-concentration features, while the final student predictor uses only the hole distribution image to predict \(K^*\) and \(K_{\varepsilon,\max}\).

**Architecture:** Build this after `2026-06-11-cnn-baseline-src-refactor.md` is complete. Reuse the refactored `3_CNN/src/cnn_surrogate/` package. Add local-FEM feature extraction, teacher and student model classes, distillation losses, a distillation training pipeline, and a thin `3_CNN/scripts/train_distilled_surrogate.py` entrypoint. The final deployed model is the student; it must not require any FEM-derived input at prediction time.

**Tech Stack:** Python 3, pandas, numpy, PyTorch, scikit-learn, matplotlib, tqdm, pickle from the standard library, unittest.

---

## 1. Dependency On Baseline Refactor

Do not start this plan until the baseline refactor plan has passed its acceptance checklist.

Required existing files from the baseline refactor:

- `3_CNN/src/cnn_surrogate/config.py`
- `3_CNN/src/cnn_surrogate/data.py`
- `3_CNN/src/cnn_surrogate/models.py`
- `3_CNN/src/cnn_surrogate/losses.py`
- `3_CNN/src/cnn_surrogate/training.py`
- `3_CNN/src/cnn_surrogate/evaluation.py`
- `3_CNN/src/cnn_surrogate/plotting.py`
- `3_CNN/src/cnn_surrogate/io.py`
- `3_CNN/src/cnn_surrogate/pipeline.py`
- `3_CNN/scripts/train_cnn_surrogate.py`

If these do not exist, stop and implement the baseline refactor first.

## 2. Scope And Files

Create:

- `3_CNN/scripts/train_distilled_surrogate.py`
- `3_CNN/tests/test_distillation_data.py`
- `3_CNN/tests/test_distillation_models.py`
- `3_CNN/tests/test_distillation_training.py`
- `3_CNN/tests/test_distillation_pipeline.py`

Modify:

- `3_CNN/src/cnn_surrogate/config.py`
- `3_CNN/src/cnn_surrogate/data.py`
- `3_CNN/src/cnn_surrogate/models.py`
- `3_CNN/src/cnn_surrogate/losses.py`
- `3_CNN/src/cnn_surrogate/training.py`
- `3_CNN/src/cnn_surrogate/evaluation.py`
- `3_CNN/src/cnn_surrogate/plotting.py`
- `3_CNN/src/cnn_surrogate/io.py`
- `3_CNN/src/cnn_surrogate/pipeline.py`

Do not modify:

- `2_FEM/scripts/extract_odb_ml_data.py`
- `3_CNN/scripts/train_cnn_surrogate.py`, unless a shared API change from the baseline refactor requires a small compatibility update
- Any README file

Write distilled workflow outputs under:

- `3_CNN/results/distilled_surrogate/`
- `3_CNN/figures/distilled_surrogate/`

## 3. Modeling Design

The workflow trains three comparable predictors:

1. Baseline CNN from the refactor plan:

$$
\text{hole image}\rightarrow(K^*,K_{\varepsilon,\max})
$$

2. Teacher:

$$
(\text{hole image},\mathbf{s}_{local})\rightarrow(K^*,K_{\varepsilon,\max})
$$

where:

$$
\mathbf{s}_{local}=\left[s_1,s_2,\ldots,s_{24}\right]
$$

and \(s_i\) is `hole_XX_strain_concentration_factor`.

3. Student:

$$
\text{hole image}\rightarrow(K^*,K_{\varepsilon,\max})
$$

The student is the final deployable model.

The global target columns remain:

- `relative_equivalent_stiffness`
- `max_strain_concentration_factor`

These global target columns must not be teacher or student input features. They are labels only.

## 4. Model Internals

Teacher:

- Image encoder:
  - `Conv2d(1,16,kernel_size=3,padding=1)`
  - `BatchNorm2d(16)`
  - `ReLU`
  - `Conv2d(16,32,kernel_size=3,padding=1)`
  - `BatchNorm2d(32)`
  - `ReLU`
  - `MaxPool2d(2)`
  - `Conv2d(32,64,kernel_size=3,padding=1)`
  - `BatchNorm2d(64)`
  - `ReLU`
  - `MaxPool2d(2)`
  - `AdaptiveAvgPool2d((1,1))`
  - flatten to \(z_{\text{img}}\in\mathbb{R}^{64}\)

- Local FEM encoder:
  - input \(24\)
  - `Linear(24,64)`
  - `ReLU`
  - `Dropout`
  - `Linear(64,32)`
  - `ReLU`
  - output \(z_{\text{local}}\in\mathbb{R}^{32}\)

- Fusion head:
  - concatenate to \(96\)
  - `Linear(96,64)`
  - `ReLU`
  - `Dropout`
  - `Linear(64,2)`

Student:

- Same image encoder as teacher.
- Prediction head:
  - `Linear(64,64)`
  - `ReLU`
  - `Dropout`
  - `Linear(64,2)`

## 5. Distillation Loss

Teacher training:

$$
L_{teacher}=L_{\text{sup}}(\hat{y}_{teacher},y_{true})
$$

Student training:

$$
L_{student}=L_{\text{sup}}(\hat{y}_{student},y_{true})+\lambda L_{\text{distill}}(\hat{y}_{student},\hat{y}_{teacher})
$$

Use weighted MSE for both terms. Initial default:

$$
\lambda=0.3
$$

Add user-facing constant:

```python
DISTILL_WEIGHT = 0.3
```

Do not implement temperature-scaled classification distillation. This is regression, so use direct scaled-target MSE.

## 6. Required Columns

The distilled workflow requires all baseline columns plus:

```text
hole_01_strain_concentration_factor
hole_02_strain_concentration_factor
...
hole_24_strain_concentration_factor
```

These columns are training-only teacher inputs. They must not be required by the final student prediction API.

## 7. Implementation Tasks

### Task 1: Add Distillation Config And Local FEM Feature Extraction

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/config.py`
- Modify: `3_CNN/src/cnn_surrogate/data.py`
- Create: `3_CNN/tests/test_distillation_data.py`

- [ ] **Step 1: Write failing tests for local FEM feature columns**

Create `3_CNN/tests/test_distillation_data.py`:

```python
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import DistillationTrainingConfig
from cnn_surrogate.data import local_feature_columns, DistillationLayoutDataset


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _sample_row(instance=1):
    row = {
        "odb_name": "%d_plate.odb" % instance,
        "status": "ok",
        "group_index": 1,
        "instance_index": instance,
        "relative_equivalent_stiffness": 0.7 + 0.01 * instance,
        "max_strain_concentration_factor": 2.0 + 0.02 * instance,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float(index)
        row["hole_%02d_y" % index] = float(index * 2)
        row["hole_%02d_strain_concentration_factor" % index] = 1.0 + 0.01 * index
    return row


class DistillationDataTests(unittest.TestCase):
    def test_local_feature_columns_are_the_24_hole_strain_concentration_columns(self):
        columns = local_feature_columns()
        self.assertEqual(len(columns), 24)
        self.assertEqual(columns[0], "hole_01_strain_concentration_factor")
        self.assertEqual(columns[-1], "hole_24_strain_concentration_factor")

    def test_distillation_dataset_returns_image_local_features_and_target(self):
        frame = pd.DataFrame([_sample_row(1), _sample_row(2)])
        dataset = DistillationLayoutDataset(
            frame,
            image_height=80,
            image_width=40,
            pixel_size=2.0,
            target_scaler=StandardScaler(),
            local_feature_scaler=StandardScaler(),
            fit_scalers=True,
        )
        image, local_features, target = dataset[0]
        self.assertEqual(tuple(image.shape), (1, 80, 40))
        self.assertEqual(tuple(local_features.shape), (24,))
        self.assertEqual(tuple(target.shape), (2,))
```

- [ ] **Step 2: Run tests and verify missing symbol failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_distillation_data -v
```

Expected:

```text
ImportError
```

- [ ] **Step 3: Implement config and data additions**

Add to `config.py`:

```python
@dataclass
class DistillationTrainingConfig(BaselineTrainingConfig):
    teacher_epochs: int
    student_epochs: int
    teacher_learning_rate: float
    student_learning_rate: float
    distill_weight: float
```

Add to `data.py`:

```python
def local_feature_columns():
    return ["hole_%02d_strain_concentration_factor" % index for index in range(1, HOLE_COUNT + 1)]


def distillation_required_columns():
    return required_columns() + local_feature_columns()
```

Add `DistillationLayoutDataset`:

```python
class DistillationLayoutDataset(Dataset):
    def __init__(self, frame, image_height, image_width, pixel_size, target_scaler=None,
                 local_feature_scaler=None, fit_scalers=False):
        ...

    def __getitem__(self, index):
        return image_tensor, local_feature_tensor, target_tensor
```

For empty frames, return arrays with shapes:

- images: `(0, 1, image_height, image_width)`
- local features: `(0, 24)`
- targets: `(0, 2)`

- [ ] **Step 4: Run distillation data tests and verify they pass**

### Task 2: Add Teacher And Student Model Classes

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/models.py`
- Create: `3_CNN/tests/test_distillation_models.py`

- [ ] **Step 1: Write failing model shape tests**

Create `3_CNN/tests/test_distillation_models.py`:

```python
import sys
import unittest
from pathlib import Path

import torch


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.models import TeacherSurrogate, StudentSurrogate


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class DistillationModelTests(unittest.TestCase):
    def test_teacher_uses_image_and_local_features(self):
        model = TeacherSurrogate(dropout=0.2)
        images = torch.zeros((4, 1, 80, 40), dtype=torch.float32)
        local_features = torch.zeros((4, 24), dtype=torch.float32)
        outputs = model(images, local_features)
        self.assertEqual(tuple(outputs.shape), (4, 2))

    def test_student_uses_only_image(self):
        model = StudentSurrogate(dropout=0.2)
        images = torch.zeros((4, 1, 80, 40), dtype=torch.float32)
        outputs = model(images)
        self.assertEqual(tuple(outputs.shape), (4, 2))
```

- [ ] **Step 2: Run tests and verify missing class failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_distillation_models -v
```

- [ ] **Step 3: Implement model classes**

Refactor shared image encoder inside `models.py`:

```python
class ImageEncoder(nn.Module):
    def __init__(self):
        ...

    def forward(self, inputs):
        ...
```

Update `CnnSurrogate` to use `ImageEncoder`.

Add:

```python
class TeacherSurrogate(nn.Module):
    def __init__(self, dropout=0.2):
        ...

    def forward(self, images, local_features):
        ...


class StudentSurrogate(nn.Module):
    def __init__(self, dropout=0.2):
        ...

    def forward(self, images):
        ...
```

- [ ] **Step 4: Run model tests and verify they pass**

### Task 3: Add Distillation Losses And Training Loops

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/losses.py`
- Modify: `3_CNN/src/cnn_surrogate/training.py`
- Create: `3_CNN/tests/test_distillation_training.py`

- [ ] **Step 1: Write failing training tests**

Create `3_CNN/tests/test_distillation_training.py` with tests that assert:

- `distillation_loss(student_prediction, true_target, teacher_prediction, config)` equals supervised loss plus `config.distill_weight` times teacher loss.
- `train_teacher_model()` returns a `TeacherSurrogate` and non-empty history.
- `train_student_model()` returns a `StudentSurrogate` and non-empty history.
- `train_student_model()` calls `teacher.eval()` and does not require teacher gradients.

Use `epochs=1`, `batch_size=2`, and `show_progress=False`.

- [ ] **Step 2: Run tests and verify missing function failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_distillation_training -v
```

- [ ] **Step 3: Implement losses and training**

Add to `losses.py`:

```python
def distillation_loss(student_prediction, true_target, teacher_prediction, config):
    supervised = weighted_mse_loss(
        student_prediction,
        true_target,
        config.loss_weight_stiffness,
        config.loss_weight_strain,
    )
    distilled = weighted_mse_loss(
        student_prediction,
        teacher_prediction,
        config.loss_weight_stiffness,
        config.loss_weight_strain,
    )
    return supervised + config.distill_weight * distilled
```

Add to `training.py`:

```python
def train_teacher_model(train_dataset, val_dataset, config):
    ...

def train_student_model(train_dataset, val_dataset, teacher_model, config):
    ...
```

Teacher loop consumes `(image, local_features, target)`.

Student loop consumes `(image, local_features, target)` but forwards only `image` through student. It forwards `(image, local_features)` through frozen teacher under `torch.no_grad()`.

- [ ] **Step 4: Run distillation training tests and verify they pass**

### Task 4: Add Distillation Evaluation, Plots, And IO

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/evaluation.py`
- Modify: `3_CNN/src/cnn_surrogate/plotting.py`
- Modify: `3_CNN/src/cnn_surrogate/io.py`
- Create: `3_CNN/tests/test_distillation_pipeline.py`

- [ ] **Step 1: Write failing evaluation and plot tests inside pipeline test file**

Add tests for:

- `predict_teacher_frame()` creates teacher true/pred columns.
- `predict_student_frame()` creates student true/pred columns.
- `plot_teacher_vs_student()` writes `teacher_vs_student.png`.
- `save_distillation_package()` writes:
  - `teacher_model.pt`
  - `student_model.pt`
  - `target_scaler.pkl`
  - `local_feature_scaler.pkl`

- [ ] **Step 2: Run tests and verify missing function failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_distillation_pipeline -v
```

- [ ] **Step 3: Implement evaluation, plotting, and IO additions**

Add to `evaluation.py`:

```python
def predict_teacher_frame(model, dataset, batch_size, split_name=None):
    ...

def predict_student_frame(model, dataset, batch_size, split_name=None):
    ...
```

Prediction columns must include:

- `odb_name`
- `group_index`
- `instance_index`
- `split`
- `relative_equivalent_stiffness_true`
- `max_strain_concentration_factor_true`
- `relative_equivalent_stiffness_teacher_pred` or `_student_pred`
- `max_strain_concentration_factor_teacher_pred` or `_student_pred`

Add to `plotting.py`:

```python
def plot_teacher_vs_student(teacher_predictions, student_predictions, figure_dir):
    ...
```

Add to `io.py`:

```python
def save_distillation_package(teacher_model, student_model, target_scaler,
                              local_feature_scaler, output_dir, config):
    ...
```

Only save model packages when `config.save_model=True`.

- [ ] **Step 4: Run distillation pipeline tests and verify they pass**

### Task 5: Add End-To-End Distillation Pipeline

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/pipeline.py`
- Modify: `3_CNN/tests/test_distillation_pipeline.py`

- [ ] **Step 1: Write failing end-to-end distillation smoke test**

Create a temporary CSV with:

- 2 groups
- 4 instances per group
- 24 hole coordinates
- 24 local strain-concentration factors
- the 2 global targets

Use `DistillationTrainingConfig` with:

- `teacher_epochs=1`
- `student_epochs=1`
- `batch_size=2`
- `train_test_split=2`
- `split_shuffle=False`
- `save_model=False`
- `show_progress=False`
- `distill_weight=0.3`

Call:

```python
from cnn_surrogate.pipeline import run_distillation_training

result = run_distillation_training(config)
```

Assert outputs:

- `split_manifest.csv`
- `teacher_train_history.csv`
- `student_train_history.csv`
- `teacher_predictions.csv`
- `student_predictions.csv`
- `teacher_metrics.json`
- `student_metrics.json`
- `teacher_true_vs_predict.png`
- `student_true_vs_predict.png`
- `teacher_vs_student.png`

Assert no model packages exist when `save_model=False`.

- [ ] **Step 2: Run smoke test and verify missing pipeline failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_distillation_pipeline -v
```

- [ ] **Step 3: Implement `run_distillation_training(config)`**

Pipeline order:

1. Create `config.output_dir`, `config.figure_dir`, and `config.temp_dir`.
2. Load and validate required distillation columns.
3. Assign splits with `config.train_test_split`.
4. Write `split_manifest.csv`.
5. Fit target scaler on training targets only.
6. Fit local-feature scaler on training local features only.
7. Build distillation datasets for train, validation, and test.
8. Train teacher.
9. Train student with frozen teacher.
10. Predict teacher and student on all non-empty splits.
11. Write teacher/student histories, predictions, and metrics.
12. Write teacher/student true-vs-predict plots and teacher-vs-student plot.
13. Save teacher/student packages only when `config.save_model=True`.

- [ ] **Step 4: Run distillation pipeline tests and verify they pass**

### Task 6: Add Thin Distillation Script

**Files:**

- Create: `3_CNN/scripts/train_distilled_surrogate.py`
- Modify: `3_CNN/tests/test_distillation_pipeline.py`

- [ ] **Step 1: Write failing script import test**

Add a test that imports `3_CNN/scripts/train_distilled_surrogate.py` by path and asserts:

- `DATA_CSV == os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv")`
- `OUTPUT_DIR == os.path.join("3_CNN", "results", "distilled_surrogate")`
- `FIGURE_DIR == os.path.join("3_CNN", "figures", "distilled_surrogate")`
- `DISTILL_WEIGHT == 0.3`
- `build_config()` returns `DistillationTrainingConfig`

- [ ] **Step 2: Run the test and verify missing script failure**

- [ ] **Step 3: Create the thin script**

Create `3_CNN/scripts/train_distilled_surrogate.py`:

```python
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
```

- [ ] **Step 4: Run script tests and verify they pass**

### Task 7: Full Distillation Verification

**Files:**

- Verify: `3_CNN/src/cnn_surrogate/*.py`
- Verify: `3_CNN/scripts/train_distilled_surrogate.py`
- Verify: `3_CNN/tests/test_distillation_*.py`

- [ ] **Step 1: Run all CNN tests**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run syntax verification**

```powershell
.\.venv\Scripts\python.exe -m py_compile 3_CNN\scripts\train_distilled_surrogate.py 3_CNN\src\cnn_surrogate\*.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run the distilled training script on current data**

```powershell
.\.venv\Scripts\python.exe 3_CNN\scripts\train_distilled_surrogate.py
```

Expected outputs:

```text
3_CNN/results/distilled_surrogate/split_manifest.csv
3_CNN/results/distilled_surrogate/teacher_train_history.csv
3_CNN/results/distilled_surrogate/student_train_history.csv
3_CNN/results/distilled_surrogate/teacher_predictions.csv
3_CNN/results/distilled_surrogate/student_predictions.csv
3_CNN/results/distilled_surrogate/teacher_metrics.json
3_CNN/results/distilled_surrogate/student_metrics.json
3_CNN/figures/distilled_surrogate/teacher_true_vs_predict.png
3_CNN/figures/distilled_surrogate/student_true_vs_predict.png
3_CNN/figures/distilled_surrogate/teacher_vs_student.png
```

If `SAVE_MODEL=True`, also expect:

```text
3_CNN/results/distilled_surrogate/teacher_model.pt
3_CNN/results/distilled_surrogate/student_model.pt
3_CNN/results/distilled_surrogate/target_scaler.pkl
3_CNN/results/distilled_surrogate/local_feature_scaler.pkl
```

- [ ] **Step 4: Verify prediction-time student input contract**

Add or run a test proving `StudentSurrogate.forward()` accepts only image tensors:

```python
student = StudentSurrogate(dropout=0.2)
images = torch.zeros((2, 1, 80, 40), dtype=torch.float32)
outputs = student(images)
assert tuple(outputs.shape) == (2, 2)
```

There must be no final student prediction API requiring `hole_XX_strain_concentration_factor` columns.

- [ ] **Step 5: Confirm no CNN-owned files were written under `2_FEM/`**

```powershell
rg -n "distilled_surrogate|teacher_model\.pt|student_model\.pt|local_feature_scaler\.pkl" 2_FEM
```

Expected: no matches.

## 8. Acceptance Checklist

- [ ] Teacher model uses hole image plus 24 local strain-concentration features.
- [ ] Student model uses hole image only.
- [ ] Global target columns are labels only, never input features.
- [ ] Distillation loss combines true-label loss and teacher-output loss.
- [ ] `DISTILL_WEIGHT` is a user-facing script constant.
- [ ] Teacher, student, and teacher-vs-student diagnostics are written under `3_CNN/figures/distilled_surrogate`.
- [ ] Distilled tabular outputs and model packages are written under `3_CNN/results/distilled_surrogate`.
- [ ] Baseline script remains available for comparison.
- [ ] Full `3_CNN` tests pass.
- [ ] No README file is modified.
- [ ] No CNN-owned files are written under `2_FEM/`.
