# Coordinate Surrogate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent coordinate-based surrogate that maps the 24 hole centers to one global relative equivalent stiffness target and 24 per-hole strain-concentration targets.

**Architecture:** Reuse the existing `3_CNN/src/cnn_surrogate/` package style, but add coordinate-specific config, data, model, training, evaluation, plotting, IO, and pipeline entry points. The model consumes a `(24, 6)` coordinate feature tensor, builds shared per-hole embeddings, predicts \(K^*\) from pooled layout context, and predicts each `hole_XX_strain_concentration_factor` from that hole embedding plus the global context. The script remains a thin `3_CNN/scripts/train_coordinate_surrogate.py` entry point with editable top-level constants, matching the current CNN scripts.

**Tech Stack:** Python 3, pandas, numpy, PyTorch, scikit-learn, matplotlib, tqdm, pickle from the standard library, unittest.

---

## 1. Scope And Files

Create:

- `3_CNN/scripts/train_coordinate_surrogate.py`
- `3_CNN/tests/test_coordinate_data.py`
- `3_CNN/tests/test_coordinate_models.py`
- `3_CNN/tests/test_coordinate_training.py`
- `3_CNN/tests/test_coordinate_pipeline.py`
- `3_CNN/tests/test_train_coordinate_surrogate.py`

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

- Any README file
- Existing result files under `3_CNN/results/`
- Existing figure files under `3_CNN/figures/`
- `2_FEM/scripts/extract_odb_ml_data.py`
- Existing CNN or distilled training behavior except for shared imports needed by the new coordinate workflow
- `3_CNN/scripts/grid_search_surrogate.py` in this first coordinate implementation

Write coordinate workflow outputs under:

- `3_CNN/results/coordinate_surrogate/`
- `3_CNN/figures/coordinate_surrogate/`

## 2. Data Contract

The coordinate workflow reads the same upstream FEM table:

```python
DATA_CSV = os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv")
```

Use only rows where:

```text
status == "ok"
```

Required coordinate input columns:

```text
odb_name
status
group_index
instance_index
hole_01_x
hole_01_y
hole_02_x through hole_23_y
hole_24_x
hole_24_y
```

Required target columns:

```text
relative_equivalent_stiffness
hole_01_strain_concentration_factor
hole_02_strain_concentration_factor
hole_03_strain_concentration_factor through hole_23_strain_concentration_factor
hole_24_strain_concentration_factor
```

The target vector is:

$$
y=\left[K^*,s_1,s_2,\ldots,s_{24}\right]\in\mathbb{R}^{25}
$$

where \(K^*\) is `relative_equivalent_stiffness` and \(s_i\) is `hole_XX_strain_concentration_factor`.

These 25 values are supervision targets only. They must never be concatenated into the coordinate input features.

The `hole_XX` label is only a CSV pairing key: `hole_XX_x`, `hole_XX_y`, and `hole_XX_strain_concentration_factor` describe the same physical hole in one row. The model must not learn `XX` as a physical identity. Local strain concentration is supervised as a function of the hole position and the surrounding hole distribution.

Do not include `max_strain_concentration_factor` as a coordinate target in this workflow. It can still remain in the CSV for other workflows, but this coordinate surrogate supervises the 24 local strain-concentration outputs directly.

## 3. Coordinate Features

Each hole receives 6 normalized coordinate features:

$$
\phi_i=\left[
\frac{x_i}{W},
\frac{y_i}{H},
\frac{x_i}{W},
\frac{W-x_i}{W},
\frac{y_i}{H},
\frac{H-y_i}{H}
\right]
$$

Use:

```python
COORDINATE_DOMAIN_WIDTH = 80.0
COORDINATE_DOMAIN_HEIGHT = 160.0
COORDINATE_FEATURE_DIM = 6
```

Feature names, in order:

```python
[
    "x_norm",
    "y_norm",
    "left_distance_norm",
    "right_distance_norm",
    "bottom_distance_norm",
    "top_distance_norm",
]
```

The first and third entries both equal \(x_i/W\), and the second and fifth entries both equal \(y_i/H\). Keep this explicit duplication because it gives the model both absolute position names and boundary-distance names while keeping the feature contract simple and inspectable.

The encoded coordinate tensor shape is:

```text
(24, 6)
```

The dataset batch shape is:

```text
(batch_size, 24, 6)
```

## 4. Model Design

Add `CoordinateSurrogate` to `3_CNN/src/cnn_surrogate/models.py`.

Use a PointNet-style multi-task structure:

1. Shared point encoder maps every hole feature vector from 6 dimensions to `point_hidden_dim`.
2. Mean and max pooling over the 24 holes build a global layout context.
3. Global stiffness head predicts one scalar \(K^*\) from the layout context.
4. Local strain head predicts one scalar \(s_i\) for each hole from that hole embedding plus the repeated global context.
5. Concatenate stiffness and 24 local predictions into a `(batch_size, 25)` output.

Required behavior:

- The global stiffness branch is permutation-invariant over holes.
- The local strain branch is permutation-equivariant: if input holes are reordered and local targets are reordered the same way, local predictions reorder with them.
- The output column order is exactly `coordinate_target_columns()`.
- Do not add a hole-index embedding, one-hot hole ID, or 24 independent local output heads. The same local head must be shared by every hole.

Model skeleton:

```python
class CoordinateSurrogate(nn.Module):
    def __init__(self, point_feature_dim=6, point_hidden_dim=128, context_hidden_dim=256, dropout=0.2):
        super(CoordinateSurrogate, self).__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(point_feature_dim, point_hidden_dim),
            nn.ReLU(),
            nn.Linear(point_hidden_dim, point_hidden_dim),
            nn.ReLU(),
        )
        self.context_head = nn.Sequential(
            nn.Linear(point_hidden_dim * 2, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.stiffness_head = nn.Linear(context_hidden_dim, 1)
        self.local_head = nn.Sequential(
            nn.Linear(point_hidden_dim + context_hidden_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_hidden_dim, 1),
        )

    def forward(self, coordinates):
        point_features = self.point_encoder(coordinates)
        mean_features = point_features.mean(dim=1)
        max_features = point_features.max(dim=1).values
        context = self.context_head(torch.cat([mean_features, max_features], dim=1))
        stiffness = self.stiffness_head(context)
        repeated_context = context.unsqueeze(1).expand(-1, point_features.shape[1], -1)
        local_inputs = torch.cat([point_features, repeated_context], dim=2)
        local_strain = self.local_head(local_inputs).squeeze(-1)
        return torch.cat([stiffness, local_strain], dim=1)
```

## 5. Training Targets And Loss

Use two scalers, both fitted on the training split only:

- `stiffness_scaler`: fit on `relative_equivalent_stiffness`.
- `local_strain_scaler`: fit on all 24 local strain-concentration values flattened into one column.

Do not fit one 25-column scaler. Per-column scaling of the 24 local strain values would encode hole numbering into preprocessing, which is not physically meaningful.

Use a weighted MSE with two user-facing weights:

```python
LOSS_WEIGHT_STIFFNESS = 1.0
LOSS_WEIGHT_LOCAL_STRAIN = 1.0
```

Split the model output into one stiffness value and 24 local strain values:

```python
stiffness_prediction = prediction[:, :1]
local_prediction = prediction[:, 1:]
stiffness_loss = ((stiffness_prediction - stiffness_target) ** 2).mean()
local_loss = ((local_prediction - local_targets) ** 2).mean()
loss = stiffness_weight * stiffness_loss + local_strain_weight * local_loss
```

Keep all loss calculations in scaled target space, matching the existing CNN workflows.

## 6. Script Defaults

Create `3_CNN/scripts/train_coordinate_surrogate.py` with this top-level style:

```python
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
WARM_START = True
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")
```

Then follow the current script pattern:

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(CNN_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cnn_surrogate.config import CoordinateTrainingConfig
from cnn_surrogate.pipeline import run_coordinate_training
```

The script must expose:

```python
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
        warm_start=WARM_START,
        checkpoint_path=CHECKPOINT_PATH,
    )
```

and:

```python
def main():
    run_coordinate_training(build_config())
    return 0
```

## 7. Output Files

Always write:

- `3_CNN/results/coordinate_surrogate/split_manifest.csv`
- `3_CNN/results/coordinate_surrogate/train_history.csv`
- `3_CNN/results/coordinate_surrogate/metrics.json`
- `3_CNN/results/coordinate_surrogate/predictions.csv`
- `3_CNN/figures/coordinate_surrogate/loss_curve.png`
- `3_CNN/figures/coordinate_surrogate/stiffness_pred_vs_true.png`
- `3_CNN/figures/coordinate_surrogate/local_strain_pred_vs_true.png`
- `3_CNN/figures/coordinate_surrogate/local_strain_error_distribution.png`

Write during training when `WARM_START=True`:

- `3_CNN/results/coordinate_surrogate/checkpoint.pt`

`checkpoint.pt` is independent of `SAVE_MODEL`. It is the resumable training state and must be updated after every completed epoch using an atomic write to a temporary path followed by `os.replace`.

Write only when `SAVE_MODEL=True`:

- `3_CNN/results/coordinate_surrogate/model.pt`
- `3_CNN/results/coordinate_surrogate/stiffness_scaler.pkl`
- `3_CNN/results/coordinate_surrogate/local_strain_scaler.pkl`

`model.pt` must store:

```python
{
    "model_state_dict": model.state_dict(),
    "target_columns": coordinate_target_columns(),
    "coordinate_feature_names": coordinate_feature_names(),
    "coordinate_domain_width": config.coordinate_domain_width,
    "coordinate_domain_height": config.coordinate_domain_height,
    "coordinate_feature_dim": config.coordinate_feature_dim,
    "point_hidden_dim": config.point_hidden_dim,
    "context_hidden_dim": config.context_hidden_dim,
}
```

`checkpoint.pt` must store:

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
    "best_val_loss": best_val_loss,
    "stale_epoch_count": stale_epoch_count,
    "stiffness_scaler_mean": train_dataset.stiffness_scaler.mean_.tolist(),
    "stiffness_scaler_scale": train_dataset.stiffness_scaler.scale_.tolist(),
    "local_strain_scaler_mean": train_dataset.local_strain_scaler.mean_.tolist(),
    "local_strain_scaler_scale": train_dataset.local_strain_scaler.scale_.tolist(),
    "config_signature": coordinate_checkpoint_signature(config),
}
```

Warm start means resumable training from this checkpoint. It does not mean loading an arbitrary old model for fine-tuning. If `WARM_START=True` and `CHECKPOINT_PATH` exists, the trainer must load model state, optimizer state, epoch, history, best validation loss, and stale-epoch count before continuing. If the checkpoint configuration signature or target-scaler values do not match the current run, raise `ValueError` with a message telling the user to remove the checkpoint or set `WARM_START=False`.

## 8. Implementation Tasks

### Task 1: Add Coordinate Config And Data Contracts

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/config.py`
- Modify: `3_CNN/src/cnn_surrogate/data.py`
- Create: `3_CNN/tests/test_coordinate_data.py`

- [ ] **Step 1: Write failing tests for coordinate config, columns, and feature names**

Create `3_CNN/tests/test_coordinate_data.py`:

```python
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import CoordinateTrainingConfig
from cnn_surrogate import data


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _coordinate_row(group=1, instance=1):
    row = {
        "odb_name": "%d_%d_plate.odb" % (group, instance),
        "status": "ok",
        "group_index": group,
        "instance_index": instance,
        "relative_equivalent_stiffness": 0.75 + 0.001 * instance,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float(8.0 + index)
        row["hole_%02d_y" % index] = float(12.0 + index)
        row["hole_%02d_strain_concentration_factor" % index] = 2.0 + 0.01 * index
    return row


class CoordinateConfigTests(unittest.TestCase):
    def test_coordinate_config_carries_model_and_feature_parameters(self):
        config = CoordinateTrainingConfig(
            data_csv="input.csv",
            output_dir="output",
            figure_dir="figures",
            temp_dir="temp",
            train_test_split=180,
            split_shuffle=True,
            random_seed=20260611,
            coordinate_domain_width=80.0,
            coordinate_domain_height=160.0,
            coordinate_feature_dim=6,
            batch_size=32,
            epochs=500,
            learning_rate=1.0e-3,
            weight_decay=1.0e-4,
            dropout=0.2,
            point_hidden_dim=128,
            context_hidden_dim=256,
            loss_weight_stiffness=1.0,
            loss_weight_local_strain=1.0,
            early_stopping_patience=50,
            device="auto",
            show_progress=True,
            progress_description="Training coordinate surrogate",
            save_model=True,
            warm_start=True,
            checkpoint_path="checkpoint.pt",
        )
        self.assertEqual(config.coordinate_feature_dim, 6)
        self.assertEqual(config.point_hidden_dim, 128)
        self.assertEqual(config.context_hidden_dim, 256)
        self.assertTrue(config.warm_start)
        self.assertEqual(config.checkpoint_path, "checkpoint.pt")


class CoordinateColumnTests(unittest.TestCase):
    def test_coordinate_target_columns_are_stiffness_plus_24_local_strain_targets(self):
        columns = data.coordinate_target_columns()
        self.assertEqual(len(columns), 25)
        self.assertEqual(columns[0], "relative_equivalent_stiffness")
        self.assertEqual(columns[1], "hole_01_strain_concentration_factor")
        self.assertEqual(columns[-1], "hole_24_strain_concentration_factor")

    def test_coordinate_required_columns_include_inputs_and_targets(self):
        columns = data.required_coordinate_columns()
        self.assertIn("hole_01_x", columns)
        self.assertIn("hole_24_y", columns)
        self.assertIn("relative_equivalent_stiffness", columns)
        self.assertIn("hole_24_strain_concentration_factor", columns)

    def test_coordinate_feature_names_are_six_inspectable_features(self):
        self.assertEqual(data.coordinate_feature_names(), [
            "x_norm",
            "y_norm",
            "left_distance_norm",
            "right_distance_norm",
            "bottom_distance_norm",
            "top_distance_norm",
        ])
```

- [ ] **Step 2: Run the focused test and verify missing-symbol failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests -p test_coordinate_data.py
```

Expected:

```text
ImportError: cannot import name 'CoordinateTrainingConfig'
```

- [ ] **Step 3: Add coordinate config and column helpers**

Add to `3_CNN/src/cnn_surrogate/config.py`:

```python
@dataclass
class CoordinateTrainingConfig(object):
    data_csv: str
    output_dir: str
    figure_dir: str
    temp_dir: str
    train_test_split: int
    split_shuffle: bool
    random_seed: int
    coordinate_domain_width: float
    coordinate_domain_height: float
    coordinate_feature_dim: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    point_hidden_dim: int
    context_hidden_dim: int
    loss_weight_stiffness: float
    loss_weight_local_strain: float
    early_stopping_patience: Optional[int]
    device: str
    show_progress: bool
    progress_description: str
    save_model: bool
    warm_start: bool
    checkpoint_path: str
```

Add to `3_CNN/src/cnn_surrogate/data.py`:

```python
def coordinate_feature_names():
    return [
        "x_norm",
        "y_norm",
        "left_distance_norm",
        "right_distance_norm",
        "bottom_distance_norm",
        "top_distance_norm",
    ]


def coordinate_target_columns():
    return ["relative_equivalent_stiffness"] + local_feature_columns()


def required_coordinate_columns():
    columns = ["odb_name", "status", "group_index", "instance_index"]
    for index in range(1, HOLE_COUNT + 1):
        columns.append("hole_%02d_x" % index)
        columns.append("hole_%02d_y" % index)
    columns.extend(coordinate_target_columns())
    return columns


def validate_coordinate_columns(frame):
    missing = [column for column in required_coordinate_columns() if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(missing))
    return None


def load_coordinate_dataset_table(path):
    frame = pd.read_csv(path)
    validate_coordinate_columns(frame)
    frame = filter_valid_rows(frame)
    frame = frame.dropna(subset=required_coordinate_columns())
    frame["group_index"] = frame["group_index"].astype(int)
    frame["instance_index"] = frame["instance_index"].astype(int)
    return frame.reset_index(drop=True)
```

- [ ] **Step 4: Run the focused test and verify the config and column tests pass**

Expected:

```text
OK
```

### Task 2: Add Coordinate Encoding And Dataset

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/data.py`
- Modify: `3_CNN/tests/test_coordinate_data.py`

- [ ] **Step 1: Add failing tests for coordinate feature encoding and dataset shapes**

Append to `3_CNN/tests/test_coordinate_data.py`:

```python
class CoordinateEncodingTests(unittest.TestCase):
    def test_encode_coordinate_features_returns_24_by_6_features(self):
        row = _coordinate_row()
        row["hole_01_x"] = 20.0
        row["hole_01_y"] = 40.0

        features = data.encode_coordinate_features(
            row,
            domain_width=80.0,
            domain_height=160.0,
        )

        self.assertEqual(features.shape, (24, 6))
        np.testing.assert_allclose(
            features[0],
            np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.75], dtype=np.float32),
        )

    def test_coordinate_dataset_returns_coordinate_tensor_and_25_targets(self):
        frame = pd.DataFrame([_coordinate_row(instance=1), _coordinate_row(instance=2)])
        dataset = data.CoordinateLayoutDataset(
            frame,
            domain_width=80.0,
            domain_height=160.0,
            stiffness_scaler=StandardScaler(),
            local_strain_scaler=StandardScaler(),
            fit_scalers=True,
        )
        coordinates, stiffness_target, local_targets = dataset[0]
        self.assertEqual(tuple(coordinates.shape), (24, 6))
        self.assertEqual(tuple(stiffness_target.shape), (1,))
        self.assertEqual(tuple(local_targets.shape), (24,))

    def test_empty_coordinate_dataset_has_stable_shapes(self):
        scaler = StandardScaler()
        scaler.fit(np.zeros((2, 25), dtype=np.float32))
        dataset = data.CoordinateLayoutDataset(
            pd.DataFrame(columns=data.required_coordinate_columns()),
            domain_width=80.0,
            domain_height=160.0,
            stiffness_scaler=StandardScaler().fit(np.zeros((2, 1), dtype=np.float32)),
            local_strain_scaler=scaler,
            fit_scalers=False,
        )
        self.assertEqual(dataset.coordinates.shape, (0, 24, 6))
        self.assertEqual(dataset.stiffness_targets.shape, (0, 1))
        self.assertEqual(dataset.local_targets.shape, (0, 24))
```

- [ ] **Step 2: Run the focused test and verify missing-symbol failures**

Expected failure includes:

```text
AttributeError: module 'cnn_surrogate.data' has no attribute 'encode_coordinate_features'
```

- [ ] **Step 3: Implement coordinate encoding**

Add to `3_CNN/src/cnn_surrogate/data.py`:

```python
def encode_coordinate_features(row, domain_width, domain_height):
    features = np.zeros((HOLE_COUNT, len(coordinate_feature_names())), dtype=np.float32)
    for offset, index in enumerate(range(1, HOLE_COUNT + 1)):
        x = float(row["hole_%02d_x" % index])
        y = float(row["hole_%02d_y" % index])
        x_norm = x / float(domain_width)
        y_norm = y / float(domain_height)
        features[offset, 0] = x_norm
        features[offset, 1] = y_norm
        features[offset, 2] = x_norm
        features[offset, 3] = (float(domain_width) - x) / float(domain_width)
        features[offset, 4] = y_norm
        features[offset, 5] = (float(domain_height) - y) / float(domain_height)
    return features
```

Add `CoordinateLayoutDataset`:

```python
class CoordinateLayoutDataset(Dataset):
    def __init__(
        self,
        frame,
        domain_width,
        domain_height,
        stiffness_scaler=None,
        local_strain_scaler=None,
        fit_scalers=False,
    ):
        self.frame = frame.reset_index(drop=True)
        if stiffness_scaler is None:
            stiffness_scaler = StandardScaler()
        if local_strain_scaler is None:
            local_strain_scaler = StandardScaler()
        self.stiffness_scaler = stiffness_scaler
        self.local_strain_scaler = local_strain_scaler

        if len(self.frame) == 0:
            self.coordinates = np.zeros((0, HOLE_COUNT, len(coordinate_feature_names())), dtype=np.float32)
            self.stiffness_targets = np.zeros((0, 1), dtype=np.float32)
            self.local_targets = np.zeros((0, HOLE_COUNT), dtype=np.float32)
            return

        self.coordinates = np.stack([
            encode_coordinate_features(row, domain_width=domain_width, domain_height=domain_height)
            for _, row in self.frame.iterrows()
        ])
        stiffness_targets = self.frame[["relative_equivalent_stiffness"]].values.astype(np.float32)
        local_targets = self.frame[local_feature_columns()].values.astype(np.float32)
        if fit_scalers:
            stiffness_targets = stiffness_scaler.fit_transform(stiffness_targets)
            local_targets = local_strain_scaler.fit_transform(local_targets.reshape(-1, 1)).reshape(local_targets.shape)
        else:
            stiffness_targets = stiffness_scaler.transform(stiffness_targets)
            local_targets = local_strain_scaler.transform(local_targets.reshape(-1, 1)).reshape(local_targets.shape)
        self.stiffness_targets = stiffness_targets.astype(np.float32)
        self.local_targets = local_targets.astype(np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.coordinates[index]),
            torch.from_numpy(self.stiffness_targets[index]),
            torch.from_numpy(self.local_targets[index]),
        )
```

- [ ] **Step 4: Run the focused test and verify it passes**

### Task 3: Add CoordinateSurrogate Model

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/models.py`
- Create: `3_CNN/tests/test_coordinate_models.py`

- [ ] **Step 1: Write failing tests for coordinate model output shape**

Create `3_CNN/tests/test_coordinate_models.py`:

```python
import sys
import unittest
from pathlib import Path

import torch


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.models import CoordinateSurrogate


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class CoordinateSurrogateTests(unittest.TestCase):
    def test_forward_returns_stiffness_plus_24_local_strain_targets(self):
        model = CoordinateSurrogate(
            point_feature_dim=6,
            point_hidden_dim=128,
            context_hidden_dim=256,
            dropout=0.2,
        )
        coordinates = torch.zeros((4, 24, 6), dtype=torch.float32)
        outputs = model(coordinates)
        self.assertEqual(tuple(outputs.shape), (4, 25))

    def test_forward_rejects_wrong_feature_dimension(self):
        model = CoordinateSurrogate(point_feature_dim=6)
        coordinates = torch.zeros((4, 24, 5), dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            model(coordinates)

    def test_local_outputs_are_permutation_equivariant(self):
        model = CoordinateSurrogate(
            point_feature_dim=6,
            point_hidden_dim=64,
            context_hidden_dim=128,
            dropout=0.0,
        )
        model.eval()
        coordinates = torch.randn((2, 24, 6), dtype=torch.float32)
        permutation = torch.randperm(24)

        original = model(coordinates)
        permuted = model(coordinates[:, permutation, :])

        self.assertTrue(torch.allclose(original[:, 0], permuted[:, 0], atol=1.0e-6))
        self.assertTrue(torch.allclose(original[:, 1:][:, permutation], permuted[:, 1:], atol=1.0e-6))
```

- [ ] **Step 2: Run the focused test and verify missing-class failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests -p test_coordinate_models.py
```

Expected:

```text
ImportError: cannot import name 'CoordinateSurrogate'
```

- [ ] **Step 3: Implement `CoordinateSurrogate`**

Add the model skeleton from Section 4 to `3_CNN/src/cnn_surrogate/models.py`.

Required constructor defaults:

```python
def __init__(self, point_feature_dim=6, point_hidden_dim=128, context_hidden_dim=256, dropout=0.2):
```

Required output:

```text
(batch_size, 25)
```

- [ ] **Step 4: Run the focused test and verify it passes**

### Task 4: Add Coordinate Loss And Training Loop

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/losses.py`
- Modify: `3_CNN/src/cnn_surrogate/training.py`
- Create: `3_CNN/tests/test_coordinate_training.py`

- [ ] **Step 1: Write failing tests for weighted coordinate loss and training**

Create `3_CNN/tests/test_coordinate_training.py`:

```python
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import CoordinateTrainingConfig
from cnn_surrogate.losses import coordinate_weighted_mse_loss
from cnn_surrogate import training


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _coordinate_config(epochs=1, early_stopping_patience=None, device="cpu", warm_start=False, checkpoint_path="checkpoint.pt"):
    return CoordinateTrainingConfig(
        data_csv="input.csv",
        output_dir="output",
        figure_dir="figures",
        temp_dir="temp",
        train_test_split=2,
        split_shuffle=False,
        random_seed=123,
        coordinate_domain_width=80.0,
        coordinate_domain_height=160.0,
        coordinate_feature_dim=6,
        batch_size=2,
        epochs=epochs,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        dropout=0.2,
        point_hidden_dim=32,
        context_hidden_dim=64,
        loss_weight_stiffness=2.0,
        loss_weight_local_strain=0.5,
        early_stopping_patience=early_stopping_patience,
        device=device,
        show_progress=False,
        progress_description="Training coordinate surrogate",
        save_model=False,
        warm_start=warm_start,
        checkpoint_path=checkpoint_path,
    )


class CoordinateLossTests(unittest.TestCase):
    def test_coordinate_weighted_mse_loss_returns_scalar(self):
        prediction = torch.zeros((2, 25), dtype=torch.float32)
        stiffness_target = torch.ones((2, 1), dtype=torch.float32)
        local_targets = torch.ones((2, 24), dtype=torch.float32)
        loss = coordinate_weighted_mse_loss(
            prediction,
            stiffness_target,
            local_targets,
            stiffness_weight=2.0,
            local_strain_weight=0.5,
        )
        self.assertEqual(tuple(loss.shape), ())
        self.assertGreater(float(loss.item()), 0.0)


class CoordinateTrainingTests(unittest.TestCase):
    def test_train_coordinate_model_returns_history(self):
        coordinates = torch.zeros((4, 24, 6), dtype=torch.float32)
        stiffness_targets = torch.zeros((4, 1), dtype=torch.float32)
        local_targets = torch.zeros((4, 24), dtype=torch.float32)
        dataset = TensorDataset(coordinates, stiffness_targets, local_targets)

        model, history = training.train_coordinate_model(
            dataset,
            None,
            _coordinate_config(epochs=2),
        )

        self.assertEqual([record["epoch"] for record in history], [1, 2])
        self.assertEqual(tuple(model(torch.zeros((1, 24, 6), dtype=torch.float32)).shape), (1, 25))

    def test_train_coordinate_model_warm_starts_from_checkpoint(self):
        coordinates = torch.zeros((4, 24, 6), dtype=torch.float32)
        stiffness_targets = torch.zeros((4, 1), dtype=torch.float32)
        local_targets = torch.zeros((4, 24), dtype=torch.float32)
        dataset = TensorDataset(coordinates, stiffness_targets, local_targets)
        dataset.stiffness_scaler = type("ScalerState", (object,), {
            "mean_": torch.zeros(1).numpy(),
            "scale_": torch.ones(1).numpy(),
        })()
        dataset.local_strain_scaler = type("ScalerState", (object,), {
            "mean_": torch.zeros(1).numpy(),
            "scale_": torch.ones(1).numpy(),
        })()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            training.train_coordinate_model(
                dataset,
                None,
                _coordinate_config(epochs=1, warm_start=True, checkpoint_path=checkpoint_path),
            )
            self.assertTrue(Path(checkpoint_path).is_file())

            _, history = training.train_coordinate_model(
                dataset,
                None,
                _coordinate_config(epochs=2, warm_start=True, checkpoint_path=checkpoint_path),
            )

        self.assertEqual([record["epoch"] for record in history], [1, 2])
```

- [ ] **Step 2: Run the focused test and verify missing-function failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests -p test_coordinate_training.py
```

Expected failure includes:

```text
ImportError: cannot import name 'coordinate_weighted_mse_loss'
```

- [ ] **Step 3: Implement coordinate loss**

Add to `3_CNN/src/cnn_surrogate/losses.py`:

```python
def coordinate_weighted_mse_loss(prediction, stiffness_target, local_targets, stiffness_weight, local_strain_weight):
    stiffness_prediction = prediction[:, :1]
    local_prediction = prediction[:, 1:]
    stiffness_loss = ((stiffness_prediction - stiffness_target) ** 2).mean()
    local_loss = ((local_prediction - local_targets) ** 2).mean()
    return stiffness_weight * stiffness_loss + local_strain_weight * local_loss
```

- [ ] **Step 4: Implement coordinate epoch runner and trainer**

Update imports in `3_CNN/src/cnn_surrogate/training.py`:

```python
import os

from cnn_surrogate.data import coordinate_target_columns
from cnn_surrogate.models import CoordinateSurrogate
```

Add to `3_CNN/src/cnn_surrogate/training.py`:

```python
def coordinate_checkpoint_signature(config):
    return {
        "data_csv": config.data_csv,
        "train_test_split": config.train_test_split,
        "split_shuffle": config.split_shuffle,
        "random_seed": config.random_seed,
        "coordinate_domain_width": config.coordinate_domain_width,
        "coordinate_domain_height": config.coordinate_domain_height,
        "coordinate_feature_dim": config.coordinate_feature_dim,
        "point_hidden_dim": config.point_hidden_dim,
        "context_hidden_dim": config.context_hidden_dim,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "loss_weight_stiffness": config.loss_weight_stiffness,
        "loss_weight_local_strain": config.loss_weight_local_strain,
        "early_stopping_patience": config.early_stopping_patience,
        "target_columns": coordinate_target_columns(),
    }


def _coordinate_scaler_state(stiffness_scaler, local_strain_scaler):
    return {
        "stiffness_mean": [float(value) for value in stiffness_scaler.mean_],
        "stiffness_scale": [float(value) for value in stiffness_scaler.scale_],
        "local_mean": [float(value) for value in local_strain_scaler.mean_],
        "local_scale": [float(value) for value in local_strain_scaler.scale_],
    }


def _coordinate_scaler_matches(train_dataset, checkpoint):
    state = _coordinate_scaler_state(train_dataset.stiffness_scaler, train_dataset.local_strain_scaler)
    return all([
        np.allclose(state["stiffness_mean"], checkpoint.get("stiffness_scaler_mean", [])),
        np.allclose(state["stiffness_scale"], checkpoint.get("stiffness_scaler_scale", [])),
        np.allclose(state["local_mean"], checkpoint.get("local_strain_scaler_mean", [])),
        np.allclose(state["local_scale"], checkpoint.get("local_strain_scaler_scale", [])),
    ])


def _load_coordinate_checkpoint_if_available(model, optimizer, train_dataset, config, device):
    if not config.warm_start or not config.checkpoint_path or not os.path.isfile(config.checkpoint_path):
        return 0, [], None, 0
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    expected_signature = coordinate_checkpoint_signature(config)
    if checkpoint.get("config_signature") != expected_signature:
        raise ValueError("coordinate checkpoint is incompatible; remove checkpoint.pt or set WARM_START=False")
    if not _coordinate_scaler_matches(train_dataset, checkpoint):
        raise ValueError("coordinate checkpoint scalers are incompatible; remove checkpoint.pt or set WARM_START=False")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint.get("epoch", 0)),
        list(checkpoint.get("history", [])),
        checkpoint.get("best_val_loss"),
        int(checkpoint.get("stale_epoch_count", 0)),
    )


def _save_coordinate_checkpoint(model, optimizer, train_dataset, config, epoch, history, best_val_loss, stale_epoch_count):
    if not config.warm_start or not config.checkpoint_path:
        return None
    directory = os.path.dirname(config.checkpoint_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    scaler_state = _coordinate_scaler_state(train_dataset.stiffness_scaler, train_dataset.local_strain_scaler)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "stale_epoch_count": stale_epoch_count,
        "stiffness_scaler_mean": scaler_state["stiffness_mean"],
        "stiffness_scaler_scale": scaler_state["stiffness_scale"],
        "local_strain_scaler_mean": scaler_state["local_mean"],
        "local_strain_scaler_scale": scaler_state["local_scale"],
        "config_signature": coordinate_checkpoint_signature(config),
    }
    temp_path = config.checkpoint_path + ".tmp"
    torch.save(payload, temp_path)
    os.replace(temp_path, config.checkpoint_path)
    return config.checkpoint_path


def run_coordinate_epoch(model, loader, config, optimizer=None, device=None):
    if loader is None:
        return None
    if device is None:
        device = torch.device("cpu")
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    for coordinates, stiffness_targets, local_targets in loader:
        coordinates = move_tensor_to_device(coordinates, device)
        stiffness_targets = move_tensor_to_device(stiffness_targets, device)
        local_targets = move_tensor_to_device(local_targets, device)
        if training:
            optimizer.zero_grad()
        predictions = model(coordinates)
        loss = coordinate_weighted_mse_loss(
            predictions,
            stiffness_targets,
            local_targets,
            stiffness_weight=config.loss_weight_stiffness,
            local_strain_weight=config.loss_weight_local_strain,
        )
        if training:
            loss.backward()
            optimizer.step()
        batch_count = int(coordinates.shape[0])
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
    if total_count == 0:
        return None
    return total_loss / float(total_count)
```

Add `train_coordinate_model(train_dataset, val_dataset, config)` following the same seed setup, `resolve_device`, `DataLoader`, `Adam`, progress iterator, warm-start loading, checkpoint saving, and early-stopping logic used by `train_model()`. Instantiate:

```python
model = CoordinateSurrogate(
    point_feature_dim=config.coordinate_feature_dim,
    point_hidden_dim=config.point_hidden_dim,
    context_hidden_dim=config.context_hidden_dim,
    dropout=config.dropout,
).to(device)
```

After creating the optimizer, call:

```python
start_epoch, history, best_val_loss, stale_epoch_count = _load_coordinate_checkpoint_if_available(
    model,
    optimizer,
    train_dataset,
    config,
    device,
)
```

Iterate from `start_epoch + 1` through `config.epochs`. After appending each history record, call `_save_coordinate_checkpoint(...)`. If `start_epoch >= config.epochs`, return the loaded model and history without running more epochs.

- [ ] **Step 5: Run the focused test and verify it passes**

### Task 5: Add Coordinate Evaluation, Plotting, And IO

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/evaluation.py`
- Modify: `3_CNN/src/cnn_surrogate/plotting.py`
- Modify: `3_CNN/src/cnn_surrogate/io.py`
- Modify: `3_CNN/tests/test_coordinate_pipeline.py`

- [ ] **Step 1: Write failing tests for prediction columns and model package**

Create `3_CNN/tests/test_coordinate_pipeline.py` with these imports and helpers:

```python
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import CoordinateTrainingConfig
from cnn_surrogate.data import coordinate_target_columns
from cnn_surrogate.evaluation import compute_coordinate_metrics, predict_coordinate_frame
from cnn_surrogate.io import save_coordinate_model_package
from cnn_surrogate.models import CoordinateSurrogate


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _coordinate_config(data_csv="input.csv", output_dir="output", figure_dir="figures", temp_dir="temp", save_model=False):
    return CoordinateTrainingConfig(
        data_csv=data_csv,
        output_dir=output_dir,
        figure_dir=figure_dir,
        temp_dir=temp_dir,
        train_test_split=2,
        split_shuffle=False,
        random_seed=123,
        coordinate_domain_width=80.0,
        coordinate_domain_height=160.0,
        coordinate_feature_dim=6,
        batch_size=2,
        epochs=1,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        dropout=0.2,
        point_hidden_dim=32,
        context_hidden_dim=64,
        loss_weight_stiffness=1.0,
        loss_weight_local_strain=1.0,
        early_stopping_patience=None,
        device="cpu",
        show_progress=False,
        progress_description="Training coordinate surrogate",
        save_model=save_model,
        warm_start=False,
        checkpoint_path=os.path.join(output_dir, "checkpoint.pt"),
    )
```

Add tests:

```python
class CoordinateEvaluationTests(unittest.TestCase):
    def test_predict_coordinate_frame_writes_true_and_pred_columns(self):
        model = CoordinateSurrogate(point_hidden_dim=32, context_hidden_dim=64)
        coordinates = torch.zeros((2, 24, 6), dtype=torch.float32)
        stiffness_targets = torch.zeros((2, 1), dtype=torch.float32)
        local_targets = torch.zeros((2, 24), dtype=torch.float32)
        dataset = TensorDataset(coordinates, stiffness_targets, local_targets)
        dataset.frame = pd.DataFrame([
            {"odb_name": "a.odb", "group_index": 1, "instance_index": 1},
            {"odb_name": "b.odb", "group_index": 1, "instance_index": 2},
        ])
        stiffness_scaler = StandardScaler()
        stiffness_scaler.fit(np.zeros((2, 1), dtype=np.float32))
        local_strain_scaler = StandardScaler()
        local_strain_scaler.fit(np.zeros((48, 1), dtype=np.float32))
        dataset.stiffness_scaler = stiffness_scaler
        dataset.local_strain_scaler = local_strain_scaler

        predictions = predict_coordinate_frame(model, dataset, batch_size=2, split_name="val", device="cpu")

        self.assertIn("relative_equivalent_stiffness_true", predictions.columns)
        self.assertIn("relative_equivalent_stiffness_pred", predictions.columns)
        self.assertIn("hole_24_strain_concentration_factor_true", predictions.columns)
        self.assertIn("hole_24_strain_concentration_factor_pred", predictions.columns)

    def test_compute_coordinate_metrics_has_stiffness_and_local_strain_summary(self):
        frame = pd.DataFrame({
            "split": ["val", "val"],
            "relative_equivalent_stiffness_true": [1.0, 2.0],
            "relative_equivalent_stiffness_pred": [1.0, 2.0],
        })
        for column in coordinate_target_columns()[1:]:
            frame[column + "_true"] = [1.0, 2.0]
            frame[column + "_pred"] = [1.0, 2.0]

        metrics = compute_coordinate_metrics(frame)

        self.assertEqual(metrics["val"]["count"], 2)
        self.assertEqual(metrics["val"]["targets"]["relative_equivalent_stiffness"]["rmse"], 0.0)
        self.assertEqual(metrics["val"]["local_strain_summary"]["rmse"], 0.0)
```

Add an IO test:

```python
class CoordinateIoTests(unittest.TestCase):
    def test_save_coordinate_model_package_respects_save_model_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = CoordinateSurrogate(point_hidden_dim=32, context_hidden_dim=64)
            stiffness_scaler = StandardScaler()
            stiffness_scaler.fit(np.zeros((2, 1), dtype=np.float32))
            local_strain_scaler = StandardScaler()
            local_strain_scaler.fit(np.zeros((48, 1), dtype=np.float32))
            config = _coordinate_config(
                data_csv="input.csv",
                output_dir=temp_dir,
                figure_dir=os.path.join(temp_dir, "figures"),
                temp_dir=os.path.join(temp_dir, "temp"),
                save_model=False,
            )

            save_coordinate_model_package(model, stiffness_scaler, local_strain_scaler, temp_dir, config)

            self.assertFalse(os.path.exists(os.path.join(temp_dir, "model.pt")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "stiffness_scaler.pkl")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "local_strain_scaler.pkl")))
```

- [ ] **Step 2: Run the focused test and verify missing-function failures**

- [ ] **Step 3: Implement coordinate prediction and metrics**

Add to `3_CNN/src/cnn_surrogate/evaluation.py`:

```python
def predict_coordinate_frame(model, dataset, batch_size, split_name=None, device="cpu"):
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=should_pin_memory(device))
    predictions = []
    stiffness_targets = []
    local_targets = []
    with torch.no_grad():
        for coordinates, batch_stiffness_targets, batch_local_targets in loader:
            coordinates = move_tensor_to_device(coordinates, device)
            outputs = model(coordinates).detach().cpu().numpy()
            predictions.append(outputs)
            stiffness_targets.append(batch_stiffness_targets.detach().cpu().numpy())
            local_targets.append(batch_local_targets.detach().cpu().numpy())
    if len(predictions) == 0:
        return _empty_coordinate_prediction_frame(split_name)
    prediction_values = np.vstack(predictions)
    stiffness_pred = dataset.stiffness_scaler.inverse_transform(prediction_values[:, :1])
    local_pred = dataset.local_strain_scaler.inverse_transform(prediction_values[:, 1:].reshape(-1, 1)).reshape(-1, 24)
    stiffness_true = dataset.stiffness_scaler.inverse_transform(np.vstack(stiffness_targets))
    local_true = dataset.local_strain_scaler.inverse_transform(np.vstack(local_targets).reshape(-1, 1)).reshape(-1, 24)
    frame = dataset.frame[["odb_name", "group_index", "instance_index"]].copy()
    frame["split"] = split_name
    frame["relative_equivalent_stiffness_true"] = stiffness_true[:, 0]
    frame["relative_equivalent_stiffness_pred"] = stiffness_pred[:, 0]
    for index, target_column in enumerate(coordinate_target_columns()[1:]):
        frame[target_column + "_true"] = local_true[:, index]
        frame[target_column + "_pred"] = local_pred[:, index]
    return frame
```

Add `compute_coordinate_metrics(predictions)` that returns `train`, `val`, and `test` keys. For each split:

- `count`
- `targets["relative_equivalent_stiffness"]`
- `local_strain_summary`
- `local_strain_error_quantiles`

Use RMSE, MAE, and \(R^2\). Compute `local_strain_summary` by flattening all 24 true columns and all 24 predicted columns for that split.

- [ ] **Step 4: Implement coordinate plotting**

Add to `3_CNN/src/cnn_surrogate/plotting.py`:

```python
def plot_coordinate_stiffness_pred_vs_true(predictions, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "stiffness_pred_vs_true.png")
    true_column = "relative_equivalent_stiffness_true"
    pred_column = "relative_equivalent_stiffness_pred"
    plt.figure(figsize=(5, 4))
    if len(predictions) > 0:
        plt.scatter(predictions[true_column], predictions[pred_column], s=18, alpha=0.75)
        values = np.concatenate([predictions[true_column].values.astype(float), predictions[pred_column].values.astype(float)])
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        plt.plot([lower, upper], [lower, upper], color="black", linewidth=1.0)
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
    plt.xlabel("FEM")
    plt.ylabel("CoordinateSurrogate")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
```

Add `plot_coordinate_local_strain_pred_vs_true(predictions, figure_dir)` by taking the per-sample maximum across the 24 local strain true/pred columns, plotting those global maximum values in one scatter plot, and writing `local_strain_pred_vs_true.png`.

Add `plot_coordinate_local_strain_error_distribution(predictions, figure_dir)` by flattening all 24 local true/pred errors and plotting the error distribution. Write `local_strain_error_distribution.png`.

- [ ] **Step 5: Implement coordinate model package saving**

Add to `3_CNN/src/cnn_surrogate/io.py`:

```python
def save_coordinate_model_package(model, stiffness_scaler, local_strain_scaler, output_dir, config):
    remove_if_exists(os.path.join(output_dir, "model.pt"))
    remove_if_exists(os.path.join(output_dir, "stiffness_scaler.pkl"))
    remove_if_exists(os.path.join(output_dir, "local_strain_scaler.pkl"))
    if not config.save_model:
        return None
    torch.save({
        "model_state_dict": model.state_dict(),
        "target_columns": coordinate_target_columns(),
        "coordinate_feature_names": coordinate_feature_names(),
        "coordinate_domain_width": config.coordinate_domain_width,
        "coordinate_domain_height": config.coordinate_domain_height,
        "coordinate_feature_dim": config.coordinate_feature_dim,
        "point_hidden_dim": config.point_hidden_dim,
        "context_hidden_dim": config.context_hidden_dim,
    }, os.path.join(output_dir, "model.pt"))
    with open(os.path.join(output_dir, "stiffness_scaler.pkl"), "wb") as scaler_file:
        pickle.dump(stiffness_scaler, scaler_file)
    with open(os.path.join(output_dir, "local_strain_scaler.pkl"), "wb") as scaler_file:
        pickle.dump(local_strain_scaler, scaler_file)
    return os.path.join(output_dir, "model.pt")
```

If `remove_if_exists()` is not available in `io.py`, add it using the same behavior already used by existing model-package helpers: remove stale files before respecting `save_model=False`.

- [ ] **Step 6: Run the focused test and verify it passes**

### Task 6: Add Coordinate Pipeline

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/pipeline.py`
- Modify: `3_CNN/tests/test_coordinate_pipeline.py`

- [ ] **Step 1: Add an end-to-end pipeline smoke test**

Append to `3_CNN/tests/test_coordinate_pipeline.py`:

```python
def _coordinate_row(group=1, instance=1):
    row = {
        "odb_name": "%d_%d_plate.odb" % (group, instance),
        "status": "ok",
        "group_index": group,
        "instance_index": instance,
        "relative_equivalent_stiffness": 0.75 + 0.001 * instance + 0.002 * group,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float(8.0 + index + instance)
        row["hole_%02d_y" % index] = float(12.0 + index + group)
        row["hole_%02d_strain_concentration_factor" % index] = 2.0 + 0.01 * index + 0.001 * instance
    return row


def _write_coordinate_csv(path):
    rows = []
    for group in [1, 2]:
        for instance in range(1, 5):
            rows.append(_coordinate_row(group=group, instance=instance))
    pd.DataFrame(rows).to_csv(path, index=False)
```

Add:

```python
class CoordinatePipelineTests(unittest.TestCase):
    def test_run_coordinate_training_writes_expected_artifacts_without_model_package(self):
        from cnn_surrogate.pipeline import run_coordinate_training

        with tempfile.TemporaryDirectory() as temp_dir:
            data_csv = os.path.join(temp_dir, "summary.csv")
            output_dir = os.path.join(temp_dir, "results")
            figure_dir = os.path.join(temp_dir, "figures")
            work_dir = os.path.join(temp_dir, "temp")
            _write_coordinate_csv(data_csv)
            config = _coordinate_config(
                data_csv=data_csv,
                output_dir=output_dir,
                figure_dir=figure_dir,
                temp_dir=work_dir,
                save_model=False,
            )

            result = run_coordinate_training(config)

            self.assertIn("history", result)
            self.assertIn("metrics", result)
            self.assertIn("predictions", result)
            for filename in ["split_manifest.csv", "train_history.csv", "metrics.json", "predictions.csv"]:
                self.assertTrue(os.path.exists(os.path.join(output_dir, filename)))
            for filename in [
                "loss_curve.png",
                "stiffness_pred_vs_true.png",
                "local_strain_pred_vs_true.png",
                "local_strain_error_distribution.png",
            ]:
                self.assertTrue(os.path.exists(os.path.join(figure_dir, filename)))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "model.pt")))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "stiffness_scaler.pkl")))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "local_strain_scaler.pkl")))
```

- [ ] **Step 2: Run the focused test and verify missing-pipeline failure**

Expected failure includes:

```text
ImportError: cannot import name 'run_coordinate_training'
```

- [ ] **Step 3: Implement `run_coordinate_training(config)`**

Add to `3_CNN/src/cnn_surrogate/pipeline.py`:

```python
def run_coordinate_training(config):
    ensure_directory(config.output_dir)
    ensure_directory(config.figure_dir)
    ensure_directory(config.temp_dir)

    frame = load_coordinate_dataset_table(config.data_csv)
    frame = assign_splits(
        frame,
        train_count=config.train_test_split,
        shuffle=config.split_shuffle,
        random_seed=config.random_seed,
    )
    write_split_manifest(frame, config.output_dir)

    train_frame = frame[frame["split"] == "train"].copy()
    stiffness_scaler = StandardScaler()
    local_strain_scaler = StandardScaler()
    train_dataset = CoordinateLayoutDataset(
        train_frame,
        domain_width=config.coordinate_domain_width,
        domain_height=config.coordinate_domain_height,
        stiffness_scaler=stiffness_scaler,
        local_strain_scaler=local_strain_scaler,
        fit_scalers=True,
    )
    val_dataset = _build_coordinate_dataset_for_split(
        frame,
        "val",
        train_dataset.stiffness_scaler,
        train_dataset.local_strain_scaler,
        config,
    )
    test_dataset = _build_coordinate_dataset_for_split(
        frame,
        "test",
        train_dataset.stiffness_scaler,
        train_dataset.local_strain_scaler,
        config,
    )

    model, history = train_coordinate_model(train_dataset, val_dataset, config)
    write_train_history(history, config.output_dir)

    prediction_frames = [
        predict_coordinate_frame(model, train_dataset, config.batch_size, split_name="train", device=config.device),
        predict_coordinate_frame(model, val_dataset, config.batch_size, split_name="val", device=config.device),
        predict_coordinate_frame(model, test_dataset, config.batch_size, split_name="test", device=config.device),
    ]
    predictions = _combine_prediction_frames(prediction_frames)
    metrics = compute_coordinate_metrics(predictions)

    write_predictions(predictions, config.output_dir)
    write_metrics(metrics, config.output_dir)
    plot_loss_curve(history, config.figure_dir)
    plot_coordinate_stiffness_pred_vs_true(predictions, config.figure_dir)
    plot_coordinate_local_strain_pred_vs_true(predictions, config.figure_dir)
    plot_coordinate_local_strain_error_distribution(predictions, config.figure_dir)
    save_coordinate_model_package(
        model,
        train_dataset.stiffness_scaler,
        train_dataset.local_strain_scaler,
        config.output_dir,
        config,
    )

    return {"history": history, "metrics": metrics, "predictions": predictions}
```

Add `_build_coordinate_dataset_for_split(frame, split_name, stiffness_scaler, local_strain_scaler, config)` near the existing split-specific dataset builders.

- [ ] **Step 4: Run the focused test and verify it passes**

### Task 7: Add Coordinate Training Script

**Files:**

- Create: `3_CNN/scripts/train_coordinate_surrogate.py`
- Create: `3_CNN/tests/test_train_coordinate_surrogate.py`

- [ ] **Step 1: Write failing script import tests**

Create `3_CNN/tests/test_train_coordinate_surrogate.py`:

```python
import os
import sys
import unittest
from pathlib import Path


CNN_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = CNN_ROOT / "scripts"
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

import train_coordinate_surrogate as script
from cnn_surrogate.config import CoordinateTrainingConfig


def tearDownModule():
    for path in [str(SCRIPT_DIR), str(SRC_DIR)]:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


class TrainCoordinateSurrogateScriptTests(unittest.TestCase):
    def test_default_paths_and_parameters_match_coordinate_workflow(self):
        self.assertEqual(script.DATA_CSV, os.path.join("2_FEM", "results", "odb_ml_data", "odb_ml_summary.csv"))
        self.assertEqual(script.OUTPUT_DIR, os.path.join("3_CNN", "results", "coordinate_surrogate"))
        self.assertEqual(script.FIGURE_DIR, os.path.join("3_CNN", "figures", "coordinate_surrogate"))
        self.assertEqual(script.COORDINATE_DOMAIN_WIDTH, 80.0)
        self.assertEqual(script.COORDINATE_DOMAIN_HEIGHT, 160.0)
        self.assertEqual(script.COORDINATE_FEATURE_DIM, 6)
        self.assertEqual(script.DEVICE, "auto")
        self.assertTrue(script.WARM_START)
        self.assertEqual(script.CHECKPOINT_PATH, os.path.join(script.OUTPUT_DIR, "checkpoint.pt"))

    def test_build_config_returns_coordinate_training_config(self):
        config = script.build_config()
        self.assertIsInstance(config, CoordinateTrainingConfig)
        self.assertEqual(config.output_dir, os.path.join("3_CNN", "results", "coordinate_surrogate"))
        self.assertEqual(config.loss_weight_local_strain, script.LOSS_WEIGHT_LOCAL_STRAIN)
        self.assertTrue(config.warm_start)
        self.assertEqual(config.checkpoint_path, script.CHECKPOINT_PATH)
```

- [ ] **Step 2: Run the focused test and verify missing-script failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests -p test_train_coordinate_surrogate.py
```

- [ ] **Step 3: Create `train_coordinate_surrogate.py`**

Use the script constants and `build_config()` from Section 6.

Use this main body:

```python
def main():
    run_coordinate_training(build_config())
    return 0


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the focused script test and verify it passes**

### Task 8: Full Verification

**Files:**

- Verify: `3_CNN/src/cnn_surrogate/*.py`
- Verify: `3_CNN/scripts/train_coordinate_surrogate.py`
- Verify: `3_CNN/tests/test_coordinate_*.py`
- Verify: `3_CNN/tests/test_train_coordinate_surrogate.py`

- [ ] **Step 1: Run all coordinate-focused tests**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests -p test_coordinate_*.py
```

Expected:

```text
OK
```

- [ ] **Step 2: Run the coordinate script test**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests -p test_train_coordinate_surrogate.py
```

Expected:

```text
OK
```

- [ ] **Step 3: Run all existing `3_CNN` tests**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests
```

Expected: no regressions in baseline CNN, distilled CNN, grid-search, GPU-device, or coordinate tests.

- [ ] **Step 4: Run syntax verification**

```powershell
.\.venv\Scripts\python.exe -m py_compile 3_CNN\scripts\train_coordinate_surrogate.py
.\.venv\Scripts\python.exe -m py_compile 3_CNN\src\cnn_surrogate\config.py
.\.venv\Scripts\python.exe -m py_compile 3_CNN\src\cnn_surrogate\data.py
.\.venv\Scripts\python.exe -m py_compile 3_CNN\src\cnn_surrogate\models.py
.\.venv\Scripts\python.exe -m py_compile 3_CNN\src\cnn_surrogate\losses.py
.\.venv\Scripts\python.exe -m py_compile 3_CNN\src\cnn_surrogate\training.py
.\.venv\Scripts\python.exe -m py_compile 3_CNN\src\cnn_surrogate\evaluation.py
.\.venv\Scripts\python.exe -m py_compile 3_CNN\src\cnn_surrogate\plotting.py
.\.venv\Scripts\python.exe -m py_compile 3_CNN\src\cnn_surrogate\io.py
.\.venv\Scripts\python.exe -m py_compile 3_CNN\src\cnn_surrogate\pipeline.py
```

Expected: no output and exit code 0 for each command.

- [ ] **Step 5: Run the coordinate training script on current extracted data if dependencies and runtime permit**

```powershell
.\.venv\Scripts\python.exe 3_CNN\scripts\train_coordinate_surrogate.py
```

Expected outputs:

```text
3_CNN/results/coordinate_surrogate/split_manifest.csv
3_CNN/results/coordinate_surrogate/train_history.csv
3_CNN/results/coordinate_surrogate/metrics.json
3_CNN/results/coordinate_surrogate/predictions.csv
3_CNN/results/coordinate_surrogate/checkpoint.pt
3_CNN/figures/coordinate_surrogate/loss_curve.png
3_CNN/figures/coordinate_surrogate/stiffness_pred_vs_true.png
3_CNN/figures/coordinate_surrogate/local_strain_pred_vs_true.png
3_CNN/figures/coordinate_surrogate/local_strain_error_distribution.png
```

If `SAVE_MODEL=True`, also expect:

```text
3_CNN/results/coordinate_surrogate/model.pt
3_CNN/results/coordinate_surrogate/stiffness_scaler.pkl
3_CNN/results/coordinate_surrogate/local_strain_scaler.pkl
```

## 9. Acceptance Checklist

- [ ] `CoordinateSurrogate` exists and returns `(batch_size, 25)`.
- [ ] Coordinate inputs use exactly 24 holes and 6 features per hole.
- [ ] Target order is `relative_equivalent_stiffness` followed by `hole_01_strain_concentration_factor` through `hole_24_strain_concentration_factor`.
- [ ] The `hole_XX` number is used only to pair each coordinate with its local target in the CSV.
- [ ] The model has no hole-index embedding, one-hot hole ID, or 24 separate local heads.
- [ ] Local outputs are permutation-equivariant with respect to the 24 input holes.
- [ ] `max_strain_concentration_factor` is not used as a coordinate target.
- [ ] Target scaler is fit on training data only.
- [ ] Loss supports separate stiffness and local-strain weights.
- [ ] Early stopping, device handling, and progress bars match the existing training style.
- [ ] `WARM_START=True` writes `checkpoint.pt` after each completed epoch and resumes model, optimizer, history, and early-stopping state on the next run.
- [ ] Incompatible warm-start checkpoints raise `ValueError` instead of loading silently.
- [ ] The script exposes top-level constants and `build_config()`.
- [ ] Coordinate outputs are isolated under `3_CNN/results/coordinate_surrogate/` and `3_CNN/figures/coordinate_surrogate/`.
- [ ] Existing CNN and distilled workflows continue to pass tests.
- [ ] No README file is modified.
