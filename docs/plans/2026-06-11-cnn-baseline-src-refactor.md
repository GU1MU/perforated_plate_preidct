# CNN Baseline Source Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the existing baseline CNN surrogate workflow so reusable source code lives under `3_CNN/src/cnn_surrogate/`, while `3_CNN/scripts/train_cnn_surrogate.py` remains a thin parameter and orchestration entrypoint.

**Architecture:** Preserve the current baseline behavior: read `2_FEM/results/odb_ml_data/odb_ml_summary.csv`, encode 24 hole centers into a \(1\times80\times40\) image, train a two-output CNN for \(K^*\) and \(K_{\varepsilon,\max}\), and write the same result and figure artifacts. Move data loading, encoding, model, training, metrics, plotting, and IO into focused source modules. Keep user-editable constants in the script by constructing a config object and passing it into a package-level pipeline.

**Tech Stack:** Python 3, pandas, numpy, PyTorch, scikit-learn, matplotlib, tqdm, pickle from the standard library, unittest.

---

## 1. Scope And Files

Create:

- `3_CNN/src/cnn_surrogate/__init__.py`
- `3_CNN/src/cnn_surrogate/config.py`
- `3_CNN/src/cnn_surrogate/data.py`
- `3_CNN/src/cnn_surrogate/models.py`
- `3_CNN/src/cnn_surrogate/losses.py`
- `3_CNN/src/cnn_surrogate/training.py`
- `3_CNN/src/cnn_surrogate/evaluation.py`
- `3_CNN/src/cnn_surrogate/plotting.py`
- `3_CNN/src/cnn_surrogate/io.py`
- `3_CNN/src/cnn_surrogate/pipeline.py`
- `3_CNN/tests/test_data.py`
- `3_CNN/tests/test_models.py`
- `3_CNN/tests/test_training.py`
- `3_CNN/tests/test_pipeline.py`

Modify:

- `3_CNN/scripts/train_cnn_surrogate.py`
- `3_CNN/tests/test_train_cnn_surrogate.py`

Do not modify:

- `2_FEM/scripts/extract_odb_ml_data.py`
- Any CNN-owned script, test, temporary file, result file, or figure under `2_FEM/`
- Any README file

Expected structure after refactor:

```text
3_CNN/
  src/
    cnn_surrogate/
      __init__.py
      config.py
      data.py
      models.py
      losses.py
      training.py
      evaluation.py
      plotting.py
      io.py
      pipeline.py
  scripts/
    train_cnn_surrogate.py
  tests/
    test_data.py
    test_models.py
    test_training.py
    test_pipeline.py
    test_train_cnn_surrogate.py
```

## 2. Module Responsibilities

`config.py`:

- Define `BaselineTrainingConfig`.
- Keep all user-facing parameters explicit and typed.
- Provide no hidden command-line parsing.

Required config fields:

```python
@dataclass
class BaselineTrainingConfig(object):
    data_csv: str
    output_dir: str
    figure_dir: str
    temp_dir: str
    train_test_split: int
    split_shuffle: bool
    random_seed: int
    pixel_size: float
    image_height: int
    image_width: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    loss_weight_stiffness: float
    loss_weight_strain: float
    show_progress: bool
    progress_description: str
    save_model: bool
```

`data.py`:

- Define `TARGET_COLUMNS = ["relative_equivalent_stiffness", "max_strain_concentration_factor"]`.
- Define `HOLE_COUNT = 24`.
- Implement required-column construction, CSV loading, `status == "ok"` filtering, per-group split assignment, image encoding, and `HoleLayoutDataset`.
- `HoleLayoutDataset` must accept config-derived image shape and pixel size instead of reading global script constants.

`models.py`:

- Define `CnnSurrogate`.
- Preserve the current architecture unless a test requires a change:
  - image input \(1\times80\times40\)
  - conv channels 16, 32, 64
  - adaptive average pooling
  - `Linear(64,64)`
  - dropout
  - `Linear(64,2)`

`losses.py`:

- Define `weighted_mse_loss(prediction, target, stiffness_weight, strain_weight)`.

`training.py`:

- Define `iter_progress()`, `run_epoch()`, and `train_model()`.
- Use `tqdm.auto.tqdm` when available and `config.show_progress=True`.
- Keep a plain text fallback if `tqdm` import fails.
- Return `(model, history)` where `history` contains `epoch`, `train_loss`, and `val_loss`.

`evaluation.py`:

- Define `predict_frame()` and `compute_metrics()`.
- Preserve prediction CSV columns:
  - `odb_name`
  - `group_index`
  - `instance_index`
  - `split`
  - `relative_equivalent_stiffness_true`
  - `max_strain_concentration_factor_true`
  - `relative_equivalent_stiffness_pred`
  - `max_strain_concentration_factor_pred`

`plotting.py`:

- Define `plot_loss_curve()` and `plot_pred_vs_true()`.
- Write plots only under `config.figure_dir`.

`io.py`:

- Define `ensure_directory()`, `write_split_manifest()`, `write_train_history()`, `write_metrics()`, `write_predictions()`, and `save_model_package()`.
- `save_model_package()` remains the only model-saving switch.

`pipeline.py`:

- Define `run_baseline_training(config)`.
- Own the top-level orchestration currently in `main()`.

`3_CNN/scripts/train_cnn_surrogate.py`:

- Keep only top-level constants, config construction, import-path setup, and `main()`.
- Script must add `3_CNN/src` to `sys.path` so it can be run directly from the project root:

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(CNN_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
```

## 3. Behavioral Invariants

- Running `.\.venv\Scripts\python.exe 3_CNN\scripts\train_cnn_surrogate.py` still writes:
  - `3_CNN/results/cnn_surrogate/split_manifest.csv`
  - `3_CNN/results/cnn_surrogate/train_history.csv`
  - `3_CNN/results/cnn_surrogate/metrics.json`
  - `3_CNN/results/cnn_surrogate/predictions.csv`
  - `3_CNN/figures/cnn_surrogate/loss_curve.png`
  - `3_CNN/figures/cnn_surrogate/pred_vs_true.png`
- If `SAVE_MODEL=True`, it still writes:
  - `3_CNN/results/cnn_surrogate/model.pt`
  - `3_CNN/results/cnn_surrogate/target_scaler.pkl`
- If `SAVE_MODEL=False`, it skips `model.pt` and `target_scaler.pkl`.
- `TRAIN_TEST_SPLIT` remains a per-group training instance count.
- Validation and test splits can be empty without crashing.
- No dataset size, group count, or sample count is hard-coded.
- Final prediction-time inputs remain only the hole distribution image.

## 4. Implementation Tasks

### Task 1: Add Source Package Skeleton And Config

**Files:**

- Create: `3_CNN/src/cnn_surrogate/__init__.py`
- Create: `3_CNN/src/cnn_surrogate/config.py`
- Modify: `3_CNN/tests/test_train_cnn_surrogate.py`
- Create: `3_CNN/tests/test_pipeline.py`

- [ ] **Step 1: Write failing config import test**

Create `3_CNN/tests/test_pipeline.py`:

```python
import os
import sys
import unittest
from pathlib import Path


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import BaselineTrainingConfig


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class BaselineConfigTests(unittest.TestCase):
    def test_config_carries_user_facing_paths_and_training_parameters(self):
        config = BaselineTrainingConfig(
            data_csv=os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv"),
            output_dir=os.path.join("3_CNN", "results", "cnn_surrogate"),
            figure_dir=os.path.join("3_CNN", "figures", "cnn_surrogate"),
            temp_dir=os.path.join("3_CNN", "temp"),
            train_test_split=25,
            split_shuffle=True,
            random_seed=20260611,
            pixel_size=2.0,
            image_height=80,
            image_width=40,
            batch_size=32,
            epochs=300,
            learning_rate=1.0e-3,
            weight_decay=1.0e-4,
            dropout=0.2,
            loss_weight_stiffness=1.0,
            loss_weight_strain=1.0,
            show_progress=True,
            progress_description="Training CNN surrogate",
            save_model=True,
        )
        self.assertEqual(config.image_height, 80)
        self.assertEqual(config.image_width, 40)
        self.assertTrue(config.show_progress)
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_pipeline.BaselineConfigTests -v
```

Expected:

```text
ModuleNotFoundError: No module named 'cnn_surrogate'
```

- [ ] **Step 3: Create package skeleton and config dataclass**

Create `3_CNN/src/cnn_surrogate/__init__.py`:

```python
"""Reusable CNN surrogate training package."""
```

Create `3_CNN/src/cnn_surrogate/config.py`:

```python
from dataclasses import dataclass


@dataclass
class BaselineTrainingConfig(object):
    data_csv: str
    output_dir: str
    figure_dir: str
    temp_dir: str
    train_test_split: int
    split_shuffle: bool
    random_seed: int
    pixel_size: float
    image_height: int
    image_width: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    loss_weight_stiffness: float
    loss_weight_strain: float
    show_progress: bool
    progress_description: str
    save_model: bool
```

- [ ] **Step 4: Run the config test and verify it passes**

Expected:

```text
OK
```

### Task 2: Move Data Loading, Splitting, Encoding, And Dataset

**Files:**

- Create: `3_CNN/src/cnn_surrogate/data.py`
- Create: `3_CNN/tests/test_data.py`
- Modify: `3_CNN/scripts/train_cnn_surrogate.py`

- [ ] **Step 1: Write failing data module tests**

Create `3_CNN/tests/test_data.py` with tests copied from the current `test_train_cnn_surrogate.py` for:

- `required_columns()`
- `filter_valid_rows()`
- `assign_splits()`
- `encode_hole_image()`
- `HoleLayoutDataset`

Use this import pattern:

```python
CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate import data
```

Add a dataset shape assertion:

```python
def test_dataset_uses_configured_image_shape(self):
    frame = pd.DataFrame([_sample_row(instance=1), _sample_row(instance=2)])
    scaler = StandardScaler()
    dataset = data.HoleLayoutDataset(
        frame,
        image_height=80,
        image_width=40,
        pixel_size=2.0,
        target_scaler=scaler,
        fit_scaler=True,
    )
    image, target = dataset[0]
    self.assertEqual(tuple(image.shape), (1, 80, 40))
    self.assertEqual(tuple(target.shape), (2,))
```

- [ ] **Step 2: Run tests and verify missing module failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_data -v
```

Expected:

```text
ImportError
```

- [ ] **Step 3: Implement `data.py`**

Move the equivalent code from `3_CNN/scripts/train_cnn_surrogate.py`, but remove dependence on script globals. Function signatures must be:

```python
TARGET_COLUMNS = ["relative_equivalent_stiffness", "max_strain_concentration_factor"]
HOLE_COUNT = 24

def required_columns():
    ...

def filter_valid_rows(frame):
    ...

def validate_columns(frame):
    ...

def load_dataset_table(path):
    ...

def assign_splits(frame, train_count, shuffle, random_seed):
    ...

def encode_hole_image(row, pixel_size, height, width):
    ...

class HoleLayoutDataset(Dataset):
    def __init__(self, frame, image_height, image_width, pixel_size, target_scaler=None, fit_scaler=False):
        ...
```

- [ ] **Step 4: Run data tests and verify they pass**

### Task 3: Move Model, Loss, And Training Loop

**Files:**

- Create: `3_CNN/src/cnn_surrogate/models.py`
- Create: `3_CNN/src/cnn_surrogate/losses.py`
- Create: `3_CNN/src/cnn_surrogate/training.py`
- Create: `3_CNN/tests/test_models.py`
- Create: `3_CNN/tests/test_training.py`

- [ ] **Step 1: Write failing model and training tests**

`3_CNN/tests/test_models.py` must assert:

```python
model = CnnSurrogate(dropout=0.2)
inputs = torch.zeros((4, 1, 80, 40), dtype=torch.float32)
outputs = model(inputs)
self.assertEqual(tuple(outputs.shape), (4, 2))
```

`3_CNN/tests/test_training.py` must assert:

- `weighted_mse_loss()` returns a scalar tensor.
- `train_model()` wraps epochs with `iter_progress()`.
- `train_model()` raises `ValueError("training split is empty")` for an empty training dataset.

- [ ] **Step 2: Run tests and verify missing module failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_models 3_CNN.tests.test_training -v
```

- [ ] **Step 3: Implement modules**

Move `CnnSurrogate` to `models.py`.

Move `weighted_mse_loss()` to `losses.py` with signature:

```python
def weighted_mse_loss(prediction, target, stiffness_weight, strain_weight):
    ...
```

Move progress and training helpers to `training.py` with signatures:

```python
def iter_progress(iterable, description, enabled=True):
    ...

def run_epoch(model, loader, optimizer=None, stiffness_weight=1.0, strain_weight=1.0):
    ...

def train_model(train_dataset, val_dataset, config):
    ...
```

`train_model()` must use `config.batch_size`, `config.epochs`, `config.learning_rate`, `config.weight_decay`, `config.dropout`, `config.show_progress`, and `config.progress_description`.

- [ ] **Step 4: Run model and training tests and verify they pass**

### Task 4: Move Evaluation, Plotting, IO, And Pipeline

**Files:**

- Create: `3_CNN/src/cnn_surrogate/evaluation.py`
- Create: `3_CNN/src/cnn_surrogate/plotting.py`
- Create: `3_CNN/src/cnn_surrogate/io.py`
- Create: `3_CNN/src/cnn_surrogate/pipeline.py`
- Modify: `3_CNN/tests/test_pipeline.py`

- [ ] **Step 1: Add failing pipeline smoke test**

Extend `3_CNN/tests/test_pipeline.py` with a temporary CSV test that calls:

```python
from cnn_surrogate.pipeline import run_baseline_training
```

Use `BaselineTrainingConfig` with:

- `epochs=1`
- `batch_size=2`
- `train_test_split=2`
- `split_shuffle=False`
- `save_model=False`
- `show_progress=False`

Assert these exist:

- `split_manifest.csv`
- `train_history.csv`
- `metrics.json`
- `predictions.csv`
- `loss_curve.png`
- `pred_vs_true.png`

Assert these do not exist:

- `model.pt`
- `target_scaler.pkl`

- [ ] **Step 2: Run the pipeline test and verify it fails**

Expected:

```text
ModuleNotFoundError: No module named 'cnn_surrogate.pipeline'
```

- [ ] **Step 3: Implement evaluation, plotting, IO, and pipeline**

Move current script helpers into the modules with these signatures:

```python
# evaluation.py
def predict_frame(model, dataset, batch_size, split_name=None):
    ...

def compute_metrics(predictions_frame):
    ...

# plotting.py
def plot_loss_curve(history, figure_dir):
    ...

def plot_pred_vs_true(predictions, figure_dir):
    ...

# io.py
def ensure_directory(path):
    ...

def write_split_manifest(frame, output_dir):
    ...

def write_train_history(history, output_dir):
    ...

def write_metrics(metrics, output_dir):
    ...

def write_predictions(predictions, output_dir):
    ...

def save_model_package(model, target_scaler, output_dir, config):
    ...

# pipeline.py
def run_baseline_training(config):
    ...
```

`save_model_package()` must store:

```python
{
    "model_state_dict": model.state_dict(),
    "image_height": config.image_height,
    "image_width": config.image_width,
    "pixel_size": config.pixel_size,
    "target_columns": TARGET_COLUMNS,
}
```

- [ ] **Step 4: Run pipeline tests and verify they pass**

### Task 5: Thin The Baseline Script

**Files:**

- Modify: `3_CNN/scripts/train_cnn_surrogate.py`
- Modify: `3_CNN/tests/test_train_cnn_surrogate.py`

- [ ] **Step 1: Add script-level tests for thin orchestration**

Update `3_CNN/tests/test_train_cnn_surrogate.py` so it verifies:

- Script constants still match current defaults.
- `build_config()` returns `BaselineTrainingConfig`.
- `main()` calls `run_baseline_training(build_config())`.

Use a fake `run_baseline_training` in the imported script module:

```python
def fake_run(config):
    calls.append(config)
    return {"history": [], "metrics": {}}
```

- [ ] **Step 2: Run script tests and verify they fail because `build_config()` is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_train_cnn_surrogate -v
```

- [ ] **Step 3: Replace script internals with thin wrapper**

`3_CNN/scripts/train_cnn_surrogate.py` must keep the user-facing constants at top level:

```python
DATA_CSV = os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv")
OUTPUT_DIR = os.path.join("3_CNN", "results", "cnn_surrogate")
FIGURE_DIR = os.path.join("3_CNN", "figures", "cnn_surrogate")
TEMP_DIR = os.path.join("3_CNN", "temp")

TRAIN_TEST_SPLIT = 25
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
LOSS_WEIGHT_STRAIN = 1.0

SHOW_PROGRESS = True
PROGRESS_DESCRIPTION = "Training CNN surrogate"

SAVE_MODEL = True
```

Add:

```python
def build_config():
    return BaselineTrainingConfig(...)


def main():
    run_baseline_training(build_config())
    return 0
```

- [ ] **Step 4: Run script tests and verify they pass**

### Task 6: Full Refactor Verification

**Files:**

- Verify all new `3_CNN/src/cnn_surrogate/*.py`
- Verify all new and modified `3_CNN/tests/*.py`
- Verify `3_CNN/scripts/train_cnn_surrogate.py`

- [ ] **Step 1: Run full CNN test suite**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run syntax verification**

```powershell
.\.venv\Scripts\python.exe -m py_compile 3_CNN\scripts\train_cnn_surrogate.py 3_CNN\src\cnn_surrogate\*.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run the baseline training script on current data**

```powershell
.\.venv\Scripts\python.exe 3_CNN\scripts\train_cnn_surrogate.py
```

Expected outputs:

```text
3_CNN/results/cnn_surrogate/split_manifest.csv
3_CNN/results/cnn_surrogate/train_history.csv
3_CNN/results/cnn_surrogate/metrics.json
3_CNN/results/cnn_surrogate/predictions.csv
3_CNN/figures/cnn_surrogate/loss_curve.png
3_CNN/figures/cnn_surrogate/pred_vs_true.png
```

If `SAVE_MODEL=True`, also expect:

```text
3_CNN/results/cnn_surrogate/model.pt
3_CNN/results/cnn_surrogate/target_scaler.pkl
```

- [ ] **Step 4: Confirm no CNN-owned files were written under `2_FEM/`**

```powershell
rg -n "cnn_surrogate|train_cnn_surrogate|target_scaler\.pkl|model\.pt" 2_FEM
```

Expected: no matches.

## 5. Acceptance Checklist

- [ ] Reusable baseline code lives under `3_CNN/src/cnn_surrogate/`.
- [ ] `3_CNN/scripts/train_cnn_surrogate.py` contains only constants, import-path setup, config construction, and orchestration.
- [ ] Existing baseline outputs and figure names are preserved.
- [ ] Final prediction-time inputs remain only the hole distribution image.
- [ ] `tqdm` progress still works through the refactored training loop.
- [ ] `SAVE_MODEL` remains the only switch controlling model/scaler package writes.
- [ ] Validation/test empty split handling remains covered.
- [ ] Focused and full `3_CNN` tests pass.
- [ ] No README file is modified.
- [ ] No CNN-owned files are written under `2_FEM/`.
