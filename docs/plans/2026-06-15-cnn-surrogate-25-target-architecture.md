# CNN Surrogate 25-Target Architecture Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the baseline `CnnSurrogate` so it preserves spatial layout information and uses 25 supervision values: global relative equivalent stiffness plus 24 per-hole strain-concentration factors.

**Architecture:** Keep the CNN input as a hole-layout image, but replace the current global-average-only encoder with a spatial encoder that retains layout information. Predict \(K^*\) with a global head and predict a local strain-concentration map with a spatial head; supervise that map only at the 24 hole pixels. This avoids forcing a one-channel occupancy image to infer arbitrary hole IDs while still producing the requested 25 supervised values in outputs and metrics.

**Tech Stack:** Python 3, pandas, numpy, PyTorch, scikit-learn, matplotlib, tqdm, pickle from the standard library, unittest.

---

## 1. Scope And Files

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
- `3_CNN/scripts/train_cnn_surrogate.py`
- `3_CNN/src/cnn_surrogate/grid_search.py`
- `3_CNN/src/cnn_surrogate/grid_search_config_cnn.py`

Modify tests:

- `3_CNN/tests/test_data.py`
- `3_CNN/tests/test_models.py`
- `3_CNN/tests/test_training.py`
- `3_CNN/tests/test_pipeline.py`
- `3_CNN/tests/test_train_cnn_surrogate.py`
- `3_CNN/tests/test_grid_search.py`
- `3_CNN/tests/test_grid_search_config.py`

Do not modify:

- Any README file
- Existing result files under `3_CNN/results/`
- Existing figure files under `3_CNN/figures/`
- `2_FEM/scripts/extract_odb_ml_data.py`
- Distilled teacher-student target behavior in this first change

## 2. Root Cause Being Fixed

The current CNN encoder ends with:

```python
nn.AdaptiveAvgPool2d((1, 1))
```

For a sparse binary hole image with a fixed count of 24 active pixels, this strongly compresses away absolute position and layout details. The observed search figures show near-horizontal prediction bands, which is consistent with the model outputting a narrow range around the mean.

The fix must:

- Preserve spatial layout beyond the final convolution stage.
- Add a local spatial supervision path, so the model must learn where high local strain concentration occurs.
- Keep the baseline CNN independent from the coordinate model and distilled model.

## 3. Supervision Contract

The corrected CNN uses these 25 supervision values:

```text
relative_equivalent_stiffness
hole_01_strain_concentration_factor
hole_02_strain_concentration_factor
hole_03_strain_concentration_factor through hole_23_strain_concentration_factor
hole_24_strain_concentration_factor
```

These are targets only. They must not be concatenated into the CNN input.

The `hole_XX` label is only a CSV pairing key for one row's coordinate and local strain value. It is not a physical identity and must not be modeled as one. The corrected CNN must supervise local strain at the hole pixel location and must not use 24 separate numbered output heads.

Do not use `max_strain_concentration_factor` as a corrected CNN target.

Keep the existing legacy two-target list for distilled workflows. Do not change shared behavior in a way that silently turns the distilled teacher/student task into a 25-target task.

Add CNN-specific target helpers in `data.py`:

```python
LEGACY_GLOBAL_TARGET_COLUMNS = ["relative_equivalent_stiffness", "max_strain_concentration_factor"]
TARGET_COLUMNS = LEGACY_GLOBAL_TARGET_COLUMNS


def cnn_target_columns():
    return ["relative_equivalent_stiffness"] + local_feature_columns()
```

`TARGET_COLUMNS` remains the legacy two-target value until the distilled workflow is explicitly redesigned.

## 4. Data Representation

Input image:

```text
image shape = (1, 80, 40)
```

Use the existing binary hole-center image encoding:

```python
r = floor(y / pixel_size)
c = floor(x / pixel_size)
image[0, r, c] = 1.0
```

Add local target map and mask:

```text
local_target_map shape = (1, 80, 40)
local_target_mask shape = (1, 80, 40)
```

For each hole \(i\):

```python
local_target_map[0, r_i, c_i] = scaled_hole_i_strain_concentration_factor
local_target_mask[0, r_i, c_i] = 1.0
```

All non-hole pixels have mask 0 and do not contribute to the local strain loss.

Use two scalers:

- `stiffness_scaler`: fit on `relative_equivalent_stiffness` from training rows.
- `local_strain_scaler`: fit on all 24 local strain values from training rows flattened into one column.

Do not fit one 25-column `StandardScaler` for the local map head. A spatial map value does not know a hole column index, so all local strain targets must share one local-strain scaler.

## 5. Model Architecture

Replace the current `ImageEncoder` with a spatial-preserving encoder.

Recommended first implementation:

```python
class SpatialImageEncoder(nn.Module):
    def __init__(self, embedding_dim=256, pooled_height=10, pooled_width=5, dropout=0.2):
        super(SpatialImageEncoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((pooled_height, pooled_width))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pooled_height * pooled_width, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, images):
        spatial_features = self.backbone(images)
        pooled_features = self.spatial_pool(spatial_features)
        embedding = self.embedding(pooled_features)
        return spatial_features, embedding
```

Update `CnnSurrogate`:

```python
class CnnSurrogate(nn.Module):
    def __init__(self, dropout=0.2, embedding_dim=256, pooled_height=10, pooled_width=5):
        super(CnnSurrogate, self).__init__()
        self.encoder = SpatialImageEncoder(
            embedding_dim=embedding_dim,
            pooled_height=pooled_height,
            pooled_width=pooled_width,
            dropout=dropout,
        )
        self.stiffness_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.local_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, images):
        spatial_features, embedding = self.encoder(images)
        stiffness = self.stiffness_head(embedding)
        local_low_res = self.local_head(spatial_features)
        local_map = torch.nn.functional.interpolate(
            local_low_res,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return stiffness, local_map
```

This return contract replaces the old `(batch_size, 2)` tensor for the baseline CNN only.

## 6. Open Parameters

Expose these in `3_CNN/scripts/train_cnn_surrogate.py`:

```python
SPATIAL_POOL_HEIGHT = 10
SPATIAL_POOL_WIDTH = 5
EMBEDDING_DIM = 256

LOSS_WEIGHT_STIFFNESS = 1.0
LOSS_WEIGHT_LOCAL_STRAIN = 1.0

WARM_START = True
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")
```

Keep existing training controls:

```python
BATCH_SIZE
EPOCHS
LEARNING_RATE
WEIGHT_DECAY
DROPOUT
EARLY_STOPPING_PATIENCE
DEVICE
SHOW_PROGRESS
SAVE_MODEL
```

Remove or stop using `LOSS_WEIGHT_STRAIN` in the corrected CNN path. The local strain loss weight is now `LOSS_WEIGHT_LOCAL_STRAIN`.

## 7. Loss

Add a CNN-specific loss:

```python
def cnn_spatial_supervision_loss(
    stiffness_prediction,
    stiffness_target,
    local_map_prediction,
    local_map_target,
    local_map_mask,
    stiffness_weight,
    local_strain_weight,
):
    stiffness_loss = ((stiffness_prediction - stiffness_target) ** 2).mean()
    local_squared_error = (local_map_prediction - local_map_target) ** 2 * local_map_mask
    local_denominator = local_map_mask.sum().clamp_min(1.0)
    local_loss = local_squared_error.sum() / local_denominator
    return stiffness_weight * stiffness_loss + local_strain_weight * local_loss
```

Both targets are scaled:

- `stiffness_target` uses `stiffness_scaler`.
- `local_map_target` uses `local_strain_scaler`.

## 8. Outputs

Always write:

- `3_CNN/results/cnn_surrogate/split_manifest.csv`
- `3_CNN/results/cnn_surrogate/train_history.csv`
- `3_CNN/results/cnn_surrogate/metrics.json`
- `3_CNN/results/cnn_surrogate/predictions.csv`
- `3_CNN/results/cnn_surrogate/checkpoint.pt` when `WARM_START=True`
- `3_CNN/figures/cnn_surrogate/loss_curve.png`
- `3_CNN/figures/cnn_surrogate/stiffness_pred_vs_true.png`
- `3_CNN/figures/cnn_surrogate/local_strain_pred_vs_true.png`
- `3_CNN/figures/cnn_surrogate/local_strain_error_distribution.png`

`predictions.csv` must include:

```text
odb_name
group_index
instance_index
split
relative_equivalent_stiffness_true
relative_equivalent_stiffness_pred
hole_01_strain_concentration_factor_true
hole_01_strain_concentration_factor_pred
hole_02_strain_concentration_factor_true
hole_02_strain_concentration_factor_pred
hole_03_strain_concentration_factor_true through hole_23_strain_concentration_factor_pred
hole_24_strain_concentration_factor_true
hole_24_strain_concentration_factor_pred
```

Write only when `SAVE_MODEL=True`:

- `3_CNN/results/cnn_surrogate/model.pt`
- `3_CNN/results/cnn_surrogate/stiffness_scaler.pkl`
- `3_CNN/results/cnn_surrogate/local_strain_scaler.pkl`

## 9. Implementation Tasks

### Task 1: Add CNN-Specific 25-Target Data Helpers

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/data.py`
- Modify: `3_CNN/tests/test_data.py`

- [ ] **Step 1: Write failing tests for CNN target columns**

Add tests:

```python
def test_cnn_target_columns_are_stiffness_plus_24_local_strain_targets(self):
    columns = data.cnn_target_columns()
    self.assertEqual(len(columns), 25)
    self.assertEqual(columns[0], "relative_equivalent_stiffness")
    self.assertEqual(columns[1], "hole_01_strain_concentration_factor")
    self.assertEqual(columns[-1], "hole_24_strain_concentration_factor")


def test_legacy_target_columns_remain_two_targets_for_distillation(self):
    self.assertEqual(data.TARGET_COLUMNS, [
        "relative_equivalent_stiffness",
        "max_strain_concentration_factor",
    ])
```

- [ ] **Step 2: Run focused data tests and verify missing-function failure**

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_data -v
```

- [ ] **Step 3: Implement target helpers without changing `TARGET_COLUMNS`**

Add:

```python
LEGACY_GLOBAL_TARGET_COLUMNS = ["relative_equivalent_stiffness", "max_strain_concentration_factor"]
TARGET_COLUMNS = LEGACY_GLOBAL_TARGET_COLUMNS


def cnn_target_columns():
    return ["relative_equivalent_stiffness"] + local_feature_columns()
```

Update `required_columns()` only if it is still intended to represent legacy baseline input. Add a new helper:

```python
def required_cnn_columns():
    columns = ["odb_name", "status", "group_index", "instance_index"]
    for index in range(1, HOLE_COUNT + 1):
        columns.append("hole_%02d_x" % index)
        columns.append("hole_%02d_y" % index)
    columns.extend(cnn_target_columns())
    return columns
```

- [ ] **Step 4: Run focused data tests and verify they pass**

### Task 2: Add Local Target Map Encoding And Dataset

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/data.py`
- Modify: `3_CNN/tests/test_data.py`

- [ ] **Step 1: Write failing tests for local map and mask shapes**

Add tests:

```python
def test_encode_local_strain_map_sets_only_hole_pixels(self):
    row = _sample_row(instance=1)
    for index in range(1, 25):
        row["hole_%02d_strain_concentration_factor" % index] = float(index)
    row["hole_01_x"] = 4.0
    row["hole_01_y"] = 6.0

    target_map, mask = data.encode_local_strain_map(
        row,
        pixel_size=2.0,
        height=80,
        width=40,
        local_strain_scaler=None,
    )

    self.assertEqual(target_map.shape, (1, 80, 40))
    self.assertEqual(mask.shape, (1, 80, 40))
    self.assertEqual(mask.sum(), 24.0)
    self.assertEqual(target_map[0, 3, 2], 1.0)


def test_cnn_spatial_dataset_returns_image_stiffness_target_local_map_and_mask(self):
    frame = pd.DataFrame([_sample_row(instance=1), _sample_row(instance=2)])
    for index in range(1, 25):
        frame["hole_%02d_strain_concentration_factor" % index] = 2.0 + index * 0.01

    dataset = data.CnnSpatialLayoutDataset(
        frame,
        image_height=80,
        image_width=40,
        pixel_size=2.0,
        stiffness_scaler=StandardScaler(),
        local_strain_scaler=StandardScaler(),
        fit_scalers=True,
    )

    image, stiffness_target, local_map, local_mask = dataset[0]
    self.assertEqual(tuple(image.shape), (1, 80, 40))
    self.assertEqual(tuple(stiffness_target.shape), (1,))
    self.assertEqual(tuple(local_map.shape), (1, 80, 40))
    self.assertEqual(tuple(local_mask.shape), (1, 80, 40))
```

- [ ] **Step 2: Implement `encode_local_strain_map()`**

```python
def encode_local_strain_map(row, pixel_size, height, width, local_strain_scaler=None):
    target_map = np.zeros((1, height, width), dtype=np.float32)
    mask = np.zeros((1, height, width), dtype=np.float32)
    values = []
    positions = []
    for index in range(1, HOLE_COUNT + 1):
        x = float(row["hole_%02d_x" % index])
        y = float(row["hole_%02d_y" % index])
        r = _clamp(int(math.floor(y / pixel_size)), 0, height - 1)
        c = _clamp(int(math.floor(x / pixel_size)), 0, width - 1)
        value = float(row["hole_%02d_strain_concentration_factor" % index])
        values.append([value])
        positions.append((r, c))
    values = np.asarray(values, dtype=np.float32)
    if local_strain_scaler is not None:
        values = local_strain_scaler.transform(values)
    for offset, (r, c) in enumerate(positions):
        target_map[0, r, c] = float(values[offset, 0])
        mask[0, r, c] = 1.0
    return target_map, mask
```

- [ ] **Step 3: Implement `CnnSpatialLayoutDataset`**

The dataset must:

1. Store `self.frame`.
2. Encode images with `encode_hole_image()`.
3. Fit `stiffness_scaler` on training stiffness values only when `fit_scalers=True`.
4. Fit `local_strain_scaler` on flattened training local strain values only when `fit_scalers=True`.
5. Return `(image, stiffness_target, local_map, local_mask)`.
6. Support empty frames with stable shapes.

- [ ] **Step 4: Run focused data tests and verify they pass**

### Task 3: Replace The Baseline CNN Architecture

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/models.py`
- Modify: `3_CNN/tests/test_models.py`

- [ ] **Step 1: Write failing model tests**

Update the baseline CNN test:

```python
def test_forward_returns_stiffness_and_local_map(self):
    model = CnnSurrogate(
        dropout=0.2,
        embedding_dim=128,
        pooled_height=10,
        pooled_width=5,
    )
    inputs = torch.zeros((4, 1, 80, 40), dtype=torch.float32)
    stiffness, local_map = model(inputs)
    self.assertEqual(tuple(stiffness.shape), (4, 1))
    self.assertEqual(tuple(local_map.shape), (4, 1, 80, 40))
```

- [ ] **Step 2: Run focused model test and verify old output-contract failure**

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_models -v
```

- [ ] **Step 3: Implement `SpatialImageEncoder` and update `CnnSurrogate`**

Use the architecture from Section 5. Leave `TeacherSurrogate` and `StudentSurrogate` unchanged for this plan unless their imports require a compatibility shim.

If distilled models depend on `ImageEncoder`, keep the old `ImageEncoder` class available. Add `SpatialImageEncoder` separately instead of deleting `ImageEncoder`.

- [ ] **Step 4: Run focused model tests and verify they pass**

### Task 4: Add Spatial CNN Loss And Training Loop

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/losses.py`
- Modify: `3_CNN/src/cnn_surrogate/training.py`
- Modify: `3_CNN/tests/test_training.py`

- [ ] **Step 1: Write failing loss test**

Add:

```python
def test_cnn_spatial_supervision_loss_uses_masked_local_loss(self):
    stiffness_prediction = torch.zeros((2, 1), dtype=torch.float32)
    stiffness_target = torch.ones((2, 1), dtype=torch.float32)
    local_map_prediction = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    local_map_target = torch.ones((2, 1, 4, 4), dtype=torch.float32)
    local_map_mask = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    local_map_mask[:, :, 1, 1] = 1.0

    loss = cnn_spatial_supervision_loss(
        stiffness_prediction,
        stiffness_target,
        local_map_prediction,
        local_map_target,
        local_map_mask,
        stiffness_weight=1.0,
        local_strain_weight=1.0,
    )

    self.assertEqual(tuple(loss.shape), ())
    self.assertGreater(float(loss.item()), 0.0)
```

- [ ] **Step 2: Implement `cnn_spatial_supervision_loss()`**

Use the exact loss structure from Section 7.

- [ ] **Step 3: Update `run_epoch()` for the baseline CNN**

The baseline CNN training loop must consume:

```python
for images, stiffness_targets, local_maps, local_masks in loader:
```

Forward:

```python
stiffness_predictions, local_map_predictions = model(images)
loss = cnn_spatial_supervision_loss(
    stiffness_predictions,
    stiffness_targets,
    local_map_predictions,
    local_maps,
    local_masks,
    stiffness_weight=config.loss_weight_stiffness,
    local_strain_weight=config.loss_weight_local_strain,
)
```

Keep device handling, early stopping, progress bars, and Adam optimizer style consistent with the existing `train_model()`.

- [ ] **Step 4: Add warm-start checkpoint support for the baseline CNN**

Add:

```python
def cnn_checkpoint_signature(config):
    return {
        "data_csv": config.data_csv,
        "train_test_split": config.train_test_split,
        "split_shuffle": config.split_shuffle,
        "random_seed": config.random_seed,
        "pixel_size": config.pixel_size,
        "image_height": config.image_height,
        "image_width": config.image_width,
        "embedding_dim": config.embedding_dim,
        "spatial_pool_height": config.spatial_pool_height,
        "spatial_pool_width": config.spatial_pool_width,
        "dropout": config.dropout,
        "loss_weight_stiffness": config.loss_weight_stiffness,
        "loss_weight_local_strain": config.loss_weight_local_strain,
        "target_columns": cnn_target_columns(),
    }
```

Save `checkpoint.pt` after every completed epoch when `config.warm_start=True`. Restore model, optimizer, history, `best_val_loss`, and `stale_epoch_count` when the checkpoint exists and the signature matches.

- [ ] **Step 5: Run focused training tests and verify they pass**

### Task 5: Update Evaluation, Metrics, And Plots

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/evaluation.py`
- Modify: `3_CNN/src/cnn_surrogate/plotting.py`
- Modify: `3_CNN/tests/test_pipeline.py`

- [ ] **Step 1: Add expected prediction columns for 25 supervision values**

The corrected CNN prediction output must include stiffness true/pred plus 24 local strain true/pred column pairs.

- [ ] **Step 2: Implement `predict_cnn_spatial_frame()`**

For each sample:

1. Run `model(image)`.
2. Inverse-transform stiffness with `stiffness_scaler`.
3. Sample the predicted local map at each `hole_XX` pixel.
4. Inverse-transform sampled local values with `local_strain_scaler`.
5. Write true and predicted columns for all 25 supervision values.

- [ ] **Step 3: Implement corrected CNN metrics**

Metrics must include:

```text
train/val/test.count
train/val/test.targets.relative_equivalent_stiffness
train/val/test.local_strain_summary
train/val/test.local_strain_error_quantiles
```

`local_strain_summary` is computed by flattening all 24 local true/pred columns for the split.

- [ ] **Step 4: Replace `pred_vs_true.png` for the corrected CNN**

Write:

- `stiffness_pred_vs_true.png`
- `local_strain_pred_vs_true.png`
- `local_strain_error_distribution.png`

Do not keep using the old two-panel `pred_vs_true.png` for the corrected CNN path.

- [ ] **Step 5: Run focused pipeline tests and verify they pass**

### Task 6: Update Pipeline And Model Saving

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/pipeline.py`
- Modify: `3_CNN/src/cnn_surrogate/io.py`
- Modify: `3_CNN/tests/test_pipeline.py`

- [ ] **Step 1: Update `run_baseline_training(config)`**

Use:

```python
CnnSpatialLayoutDataset
train_model
predict_cnn_spatial_frame
compute_cnn_spatial_metrics
plot_cnn_stiffness_pred_vs_true
plot_cnn_local_strain_pred_vs_true
plot_cnn_local_strain_error_distribution
save_cnn_spatial_model_package
```

Do not use `HoleLayoutDataset` for corrected baseline CNN training.

- [ ] **Step 2: Save two scalers**

Update model package behavior:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "image_height": config.image_height,
    "image_width": config.image_width,
    "pixel_size": config.pixel_size,
    "target_columns": cnn_target_columns(),
    "embedding_dim": config.embedding_dim,
    "spatial_pool_height": config.spatial_pool_height,
    "spatial_pool_width": config.spatial_pool_width,
}, os.path.join(output_dir, "model.pt"))
```

Save:

```text
stiffness_scaler.pkl
local_strain_scaler.pkl
```

Remove stale `target_scaler.pkl` from the corrected CNN output directory when saving or when `SAVE_MODEL=False`, because the corrected path no longer uses one 2-column target scaler.

- [ ] **Step 3: Update smoke test**

The pipeline smoke test must assert:

- `predictions.csv` has the 25 target true/pred columns.
- `metrics.json` has `local_strain_summary`.
- `stiffness_pred_vs_true.png` exists.
- `local_strain_pred_vs_true.png` exists.
- `local_strain_error_distribution.png` exists.
- `checkpoint.pt` exists when `warm_start=True`.
- `model.pt`, `stiffness_scaler.pkl`, and `local_strain_scaler.pkl` exist only when `save_model=True`.

- [ ] **Step 4: Run focused pipeline tests and verify they pass**

### Task 7: Update Script Parameters

**Files:**

- Modify: `3_CNN/scripts/train_cnn_surrogate.py`
- Modify: `3_CNN/tests/test_train_cnn_surrogate.py`

- [ ] **Step 1: Update script constants**

Add:

```python
SPATIAL_POOL_HEIGHT = 10
SPATIAL_POOL_WIDTH = 5
EMBEDDING_DIM = 256
LOSS_WEIGHT_LOCAL_STRAIN = 1.0
WARM_START = True
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")
```

Remove or stop passing:

```python
LOSS_WEIGHT_STRAIN
```

- [ ] **Step 2: Update `build_config()`**

Pass the new fields into `BaselineTrainingConfig` or a renamed `CnnTrainingConfig`.

If keeping `BaselineTrainingConfig`, add these fields:

```python
spatial_pool_height: int
spatial_pool_width: int
embedding_dim: int
loss_weight_local_strain: float
warm_start: bool
checkpoint_path: str
```

- [ ] **Step 3: Update script import tests**

Assert:

```python
self.assertEqual(script.SPATIAL_POOL_HEIGHT, 10)
self.assertEqual(script.SPATIAL_POOL_WIDTH, 5)
self.assertEqual(script.EMBEDDING_DIM, 256)
self.assertEqual(script.LOSS_WEIGHT_LOCAL_STRAIN, 1.0)
self.assertTrue(script.WARM_START)
self.assertEqual(script.CHECKPOINT_PATH, os.path.join(script.OUTPUT_DIR, "checkpoint.pt"))
```

- [ ] **Step 4: Run focused script tests and verify they pass**

### Task 8: Update CNN Grid Search Compatibility

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/grid_search.py`
- Modify: `3_CNN/src/cnn_surrogate/grid_search_config_cnn.py`
- Modify: `3_CNN/tests/test_grid_search.py`
- Modify: `3_CNN/tests/test_grid_search_config.py`

- [ ] **Step 1: Update CNN score extraction**

For `model_name == "cnn"`, extract:

```text
stiffness_rmse = metrics["val"]["targets"]["relative_equivalent_stiffness"]["rmse"]
local_strain_rmse = metrics["val"]["local_strain_summary"]["rmse"]
average_rmse = (stiffness_rmse + local_strain_rmse) / 2
```

Do not look for `max_strain_concentration_factor` in corrected CNN metrics.

- [ ] **Step 2: Update result records**

Use score keys:

```text
stiffness_rmse
local_strain_rmse
average_rmse
```

If preserving CSV column names matters for old readers, keep `strain_rmse` as an alias of `local_strain_rmse` for CNN records only, but document it as local-strain summary RMSE.

- [ ] **Step 3: Add architecture parameters to CNN search config**

Add searchable values:

```python
"spatial_pool_height": [8, 10, 12],
"spatial_pool_width": [4, 5, 6],
"embedding_dim": [128, 256, 384],
"loss_weight_local_strain": [0.5, 1.0, 2.0],
```

Keep search size reasonable before rerunning on the remote GPU. Do not reuse old `cnn_v1` results after the target contract changes; set:

```python
SEARCH_ID = "cnn_spatial25_v1"
```

- [ ] **Step 4: Run grid-search tests and verify they pass**

### Task 9: Verification

**Files:**

- Verify all modified `3_CNN` files.

- [ ] **Step 1: Run focused tests**

```powershell
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_data -v
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_models -v
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_training -v
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_pipeline -v
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_train_cnn_surrogate -v
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_grid_search -v
.\.venv\Scripts\python.exe -m unittest 3_CNN.tests.test_grid_search_config -v
```

- [ ] **Step 2: Run all CNN tests**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 3_CNN\tests
```

Expected: no regressions in corrected CNN, distilled, device, and grid-search tests.

- [ ] **Step 3: Run syntax verification**

```powershell
.\.venv\Scripts\python.exe -m py_compile 3_CNN\scripts\train_cnn_surrogate.py
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

- [ ] **Step 4: Run one short local smoke training if runtime permits**

Temporarily set `EPOCHS=1`, `DEVICE="cpu"`, and `SAVE_MODEL=False` in a test-controlled config, not by editing the production script manually.

Expected outputs:

```text
3_CNN/results/cnn_surrogate/split_manifest.csv
3_CNN/results/cnn_surrogate/train_history.csv
3_CNN/results/cnn_surrogate/metrics.json
3_CNN/results/cnn_surrogate/predictions.csv
3_CNN/results/cnn_surrogate/checkpoint.pt
3_CNN/figures/cnn_surrogate/loss_curve.png
3_CNN/figures/cnn_surrogate/stiffness_pred_vs_true.png
3_CNN/figures/cnn_surrogate/local_strain_pred_vs_true.png
3_CNN/figures/cnn_surrogate/local_strain_error_distribution.png
```

## 10. Acceptance Checklist

- [ ] Corrected `CnnSurrogate` no longer uses `AdaptiveAvgPool2d((1, 1))` as the only final spatial summary.
- [ ] Corrected CNN predicts global stiffness and a local strain map.
- [ ] The 25 supervision values are \(K^*\) plus 24 hole strain-concentration factors.
- [ ] The 25 supervision values are never used as CNN input features.
- [ ] The `hole_XX` number is used only to pair each coordinate with its local target in the CSV.
- [ ] Corrected CNN local strain supervision is applied at hole pixel locations, not through 24 numbered output heads.
- [ ] `max_strain_concentration_factor` is not a corrected CNN target.
- [ ] Local strain loss is masked to the 24 hole pixels.
- [ ] Two scalers are used: stiffness scaler and flattened local-strain scaler.
- [ ] `predictions.csv` exposes all 25 true/pred supervised values.
- [ ] Metrics include stiffness and local strain summary/error-distribution blocks.
- [ ] Warm start writes and resumes `checkpoint.pt`.
- [ ] CNN grid search uses the corrected local-strain validation score.
- [ ] Distilled workflow tests still pass.
- [ ] No README file is modified.
