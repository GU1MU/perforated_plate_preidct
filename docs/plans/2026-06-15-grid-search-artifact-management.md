# Grid Search Artifact Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `3_CNN/scripts/grid_search_surrogate.py` and its grid-search backend so one search ID owns all trial artifacts, figures are optional, non-best trial artifacts are pruned by default, and the old `--save` flag is replaced by `--save-model`.

**Architecture:** Keep `grid_search_surrogate.py` as the command-line entry point and keep `3_CNN/src/cnn_surrogate/grid_search.py` as the orchestration layer. Move every per-trial output directory under one search parent directory, add explicit search-level retention flags, and make figure generation controlled through a configuration flag that the existing training pipelines honor. Preserve warm-start behavior through `search_results.jsonl` and per-trial checkpoints for trials that are still present.

**Tech Stack:** Python 3, argparse, dataclasses, shutil, json/csv from the standard library, unittest, existing PyTorch training pipelines.

---

## 1. Scope And Files

Modify:

- `3_CNN/scripts/grid_search_surrogate.py`
- `3_CNN/src/cnn_surrogate/grid_search.py`
- `3_CNN/src/cnn_surrogate/config.py`
- `3_CNN/src/cnn_surrogate/pipeline.py`

Modify tests:

- `3_CNN/tests/test_grid_search.py`
- `3_CNN/tests/test_grid_search_script.py`
- `3_CNN/tests/test_pipeline.py`
- `3_CNN/tests/test_coordinate_pipeline.py`
- `3_CNN/tests/test_distillation_pipeline.py`

Do not modify:

- Any README file
- Existing result files under `3_CNN/results/`
- Existing figure files under `3_CNN/figures/`
- Grid-search parameter config files unless a test requires a tiny fixture-only local class

## 2. Current Behavior To Replace

Current search parent:

```text
3_CNN/results/<result_prefix>_<search_id>/
```

Current trial output directories are siblings of the search parent:

```text
3_CNN/results/<result_prefix>_<trial_id>/
3_CNN/figures/<result_prefix>_<trial_id>/
3_CNN/temp/<result_prefix>_<trial_id>/
```

This makes one search ID hard to move, archive, or delete. The revised structure must make the search parent the only durable container for that search.

## 3. Target Directory Contract

For a CNN search with:

```python
RESULT_PREFIX = "cnn_surrogate_fine_tuning"
SEARCH_ID = "cnn_spatial25_anti_overfit_v1"
```

the parent directory must be:

```text
3_CNN/results/cnn_surrogate_fine_tuning_cnn_spatial25_anti_overfit_v1/
```

Inside it, keep search-level summary files:

```text
search_results.jsonl
search_results.csv
best_results.json
```

Each trial must live under the parent:

```text
3_CNN/results/cnn_surrogate_fine_tuning_cnn_spatial25_anti_overfit_v1/
  trials/
    cnn_spatial25_anti_overfit_v1_0001/
      results/
        checkpoint.pt
        metrics.json
        predictions.csv
        split_manifest.csv
        train_history.csv
        trial_config.json
        trial_result.json
      figures/
        loss_curve.png
        stiffness_pred_vs_true.png
        local_strain_pred_vs_true.png
```

When figures are disabled, the `figures/` directory must not be created for that trial.

Use `trial_dir`, `output_dir`, and `figure_dir` explicitly in records:

```python
record = {
    "trial_dir": paths["trial_dir"],
    "output_dir": paths["output_dir"],
    "figure_dir": paths["figure_dir"],
}
```

When figures are disabled, `figure_dir` must be `None` in the record.

## 4. CLI Contract

Update `3_CNN/scripts/grid_search_surrogate.py`:

```text
--model cnn
--model cnn,distilled
--device cuda
--figure
--save-all
--save-model
```

Behavior:

- `--figure` is opt-in. Without it, no plot files or per-trial figure directories are created.
- `--save-all` is opt-in. Without it, only best trial artifacts are retained.
- `--save-model` replaces `--save`. It retrains the `best_average` parameter set into the official result directory, matching the old `--save` behavior.
- The old `--save` flag is no longer accepted by argparse.
- Default device remains `cuda`.
- `TRAIN_TEST_SPLIT = 180` and `EARLY_STOPPING_PATIENCE = 50` remain internal script constants.

Expected parsing:

```python
args = parse_args(["--model", "cnn"])
assert args.figure is False
assert args.save_all is False
assert args.save_model is False

args = parse_args(["--model", "cnn", "--figure", "--save-all", "--save-model"])
assert args.figure is True
assert args.save_all is True
assert args.save_model is True
```

## 5. Figure Control

Add `save_figures: bool` to all training config dataclasses:

```python
save_figures: bool
```

For compatibility with existing direct training scripts, set `save_figures=True` in:

- `3_CNN/scripts/train_cnn_surrogate.py`
- `3_CNN/scripts/train_coordinate_surrogate.py`
- `3_CNN/scripts/train_distilled_surrogate.py`

For grid search, set `save_figures=args.figure`.

In `pipeline.py`, guard figure directory creation and plotting:

```python
if config.save_figures:
    ensure_directory(config.figure_dir)
```

and:

```python
if config.save_figures:
    plot_loss_curve(history, config.figure_dir)
    plot_cnn_stiffness_pred_vs_true(predictions, config.figure_dir)
    plot_cnn_local_strain_pred_vs_true(predictions, config.figure_dir)
    plot_cnn_local_strain_error_distribution(predictions, config.figure_dir)
```

Apply the same pattern to coordinate and distilled pipelines. Keep writing metrics, predictions, scalers, split manifests, checkpoints, and model packages independent of `save_figures`.

## 6. Retention Contract

`--save-all` controls trial artifact retention, not search summaries.

Always keep:

```text
search_results.jsonl
search_results.csv
best_results.json
```

When `--save-all` is passed:

- Keep every completed trial directory.
- Keep failed trial directories if they were created.
- Keep all per-trial `results/` and `figures/` subdirectories.

When `--save-all` is not passed:

- Keep only trial directories referenced by the current best records.
- Treat the retained best set as the union of:
  - `best_stiffness`
  - `best_local_strain`
  - `best_strain`
  - `best_average`
- Remove completed non-best trial directories after `best_results.json` is updated.
- Remove failed trial directories after their failure record is appended to `search_results.jsonl`.
- Do not remove a currently running trial directory.
- Do not remove a partial trial directory that has no appended record yet; this preserves per-trial checkpoint warm start after an interrupted run.

The cleanup helper must refuse to delete paths outside the current `search_dir`:

```python
def _is_within_directory(parent, child):
    parent_path = os.path.abspath(parent)
    child_path = os.path.abspath(child)
    try:
        return os.path.commonpath([parent_path, child_path]) == parent_path
    except ValueError:
        return False
```

Use `shutil.rmtree(path)` only after `_is_within_directory(search_dir, path)` returns `True`.

## 7. Warm Start Behavior

Keep the existing search-level warm start:

```python
records = _read_jsonl(jsonl_path)
completed = _existing_completed_by_hash(records)
```

Completed parameter hashes in `search_results.jsonl` must still be skipped even when their non-best trial directory was pruned. This is acceptable because `search_results.jsonl` is the search ledger.

Keep per-trial checkpoint warm start for trials whose directories still exist:

```python
updates["checkpoint_path"] = os.path.join(paths["output_dir"], "checkpoint.pt")
```

If a trial was completed and pruned because it was not best, rerunning the same search should skip it from the ledger rather than retraining it.

If a trial was interrupted before a record was appended, rerunning the same search should rebuild the same `checkpoint_path` and allow the existing training `warm_start` logic to resume it.

## 8. Backend API Changes

Update `run_grid_search` signature:

```python
def run_grid_search(
    model_name,
    search_config,
    base_config_builder,
    training_runner,
    results_root,
    temp_root,
    device,
    train_test_split,
    early_stopping_patience,
    save_figures=False,
    save_all=False,
    save_best_model=False,
):
```

Remove `figures_root` from the call path. Figures are now nested under each `trial_dir` when enabled.

Replace `_trial_paths` with a function that receives `search_dir`:

```python
def _trial_paths(search_config, trial_id, search_dir):
    trial_dir = os.path.join(search_dir, "trials", trial_id)
    return {
        "trial_dir": trial_dir,
        "output_dir": os.path.join(trial_dir, "results"),
        "figure_dir": os.path.join(trial_dir, "figures"),
        "temp_dir": os.path.join(trial_dir, "temp"),
    }
```

In `build_trial_config`, set:

```python
updates["figure_dir"] = paths["figure_dir"] if save_figures else None
updates["save_figures"] = save_figures
```

In `_completed_record` and `_failed_record`, write `figure_dir=None` when figures are disabled.

Rename `_rerun_best_average` argument behavior from `save_best` to `save_best_model` in the public orchestration path. The final official model may still save figures because direct official result directories are meant for inspection; set `save_figures=True` for the final retrain unless a later requirement says official output should also honor `--figure`.

## 9. Test Plan

### Task 1: CLI Argument Tests

**Files:**

- Modify: `3_CNN/scripts/grid_search_surrogate.py`
- Modify: `3_CNN/tests/test_grid_search_script.py`

- [ ] **Step 1: Update argument parser tests**

Add or update:

```python
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
```

Add:

```python
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
```

Add:

```python
def test_script_rejects_removed_save_argument(self):
    module = _load_script()

    with self.assertRaises(SystemExit):
        module.parse_args(["--model", "cnn", "--save"])
```

- [ ] **Step 2: Run the script parser tests and verify failure before implementation**

Run:

```powershell
python -m unittest 3_CNN.tests.test_grid_search_script -v
```

Expected before implementation: failures mentioning missing `figure`, `save_all`, or `save_model`.

- [ ] **Step 3: Implement CLI arguments**

Replace the old `--save` parser argument with:

```python
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
```

Pass flags into `run_grid_search`:

```python
save_figures=args.figure,
save_all=args.save_all,
save_best_model=args.save_model,
```

- [ ] **Step 4: Run the script parser tests**

Run:

```powershell
python -m unittest 3_CNN.tests.test_grid_search_script -v
```

Expected: PASS.

### Task 2: Search Directory Layout Tests

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/grid_search.py`
- Modify: `3_CNN/tests/test_grid_search.py`

- [ ] **Step 1: Add a nested directory layout test**

Add:

```python
def test_run_grid_search_places_trial_artifacts_under_search_parent(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        captured_configs = []

        def fake_runner(config):
            captured_configs.append(config)
            os.makedirs(config.output_dir, exist_ok=True)
            with open(os.path.join(config.output_dir, "marker.txt"), "w", encoding="utf-8") as handle:
                handle.write("ran")
            return _cnn_metric_result(0.1, 0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_grid_search(
                model_name="cnn",
                search_config=SearchConfig,
                base_config_builder=_base_config,
                training_runner=fake_runner,
                results_root=os.path.join(temp_dir, "results"),
                temp_root=os.path.join(temp_dir, "temp"),
                device="cuda",
                train_test_split=180,
                early_stopping_patience=50,
                save_figures=False,
                save_all=True,
                save_best_model=False,
            )

        search_dir = os.path.join(temp_dir, "results", "unit_fine_tuning_unit_v1")
        self.assertEqual(summary["search_dir"], search_dir)
        self.assertTrue(captured_configs[0].output_dir.startswith(search_dir))
        self.assertIn(os.path.join("trials", "unit_v1_0001", "results"), captured_configs[0].output_dir)
        self.assertIsNone(captured_configs[0].figure_dir)
        self.assertTrue(os.path.isfile(os.path.join(search_dir, "search_results.jsonl")))
```

- [ ] **Step 2: Run the grid search tests and verify failure before implementation**

Run:

```powershell
python -m unittest 3_CNN.tests.test_grid_search -v
```

Expected before implementation: failure from the changed `run_grid_search` signature and old sibling path layout.

- [ ] **Step 3: Implement nested trial paths**

Change `_trial_paths` to build `trial_dir`, `output_dir`, `figure_dir`, and `temp_dir` under `search_dir`.

Update `run_grid_search` to call:

```python
paths = _trial_paths(search_config, trial_id, search_dir)
```

Remove `figures_root` from `run_grid_search`.

- [ ] **Step 4: Run the grid search tests**

Run:

```powershell
python -m unittest 3_CNN.tests.test_grid_search -v
```

Expected: tests that do not yet cover pruning may pass after adapting call signatures.

### Task 3: Figure Disable Tests

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/config.py`
- Modify: `3_CNN/src/cnn_surrogate/pipeline.py`
- Modify: `3_CNN/tests/test_pipeline.py`
- Modify: `3_CNN/tests/test_coordinate_pipeline.py`
- Modify: `3_CNN/tests/test_distillation_pipeline.py`

- [ ] **Step 1: Add config field in tests**

Update test config builders to include:

```python
save_figures=True
```

For a new pipeline test, set:

```python
config.save_figures = False
config.figure_dir = os.path.join(temp_dir, "figures")
```

Assert:

```python
self.assertFalse(os.path.exists(config.figure_dir))
```

after running a tiny fake or fixture-backed pipeline test.

- [ ] **Step 2: Add `save_figures` to dataclasses**

In `config.py`, add `save_figures: bool` to `BaselineTrainingConfig` and `CoordinateTrainingConfig`.

Because `DistillationTrainingConfig` subclasses `BaselineTrainingConfig`, it inherits the field.

- [ ] **Step 3: Update direct training script config builders**

In each direct training script, define:

```python
SAVE_FIGURES = True
```

and pass:

```python
save_figures=SAVE_FIGURES,
```

- [ ] **Step 4: Guard plotting in all pipelines**

Wrap every `ensure_directory(config.figure_dir)` and every plot call with:

```python
if config.save_figures:
    ensure_directory(config.figure_dir)
```

and:

```python
if config.save_figures:
    plot_loss_curve(...)
```

Do not guard metrics, predictions, scaler writes, model saves, or split manifest writes.

- [ ] **Step 5: Run pipeline tests**

Run:

```powershell
python -m unittest 3_CNN.tests.test_pipeline 3_CNN.tests.test_coordinate_pipeline 3_CNN.tests.test_distillation_pipeline -v
```

Expected: PASS.

### Task 4: Retention And Pruning Tests

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/grid_search.py`
- Modify: `3_CNN/tests/test_grid_search.py`

- [ ] **Step 1: Add `save_all=True` retention test**

Use the fake runner to create `marker.txt` in each `config.output_dir`.

Assert after two trials:

```python
self.assertTrue(os.path.isfile(os.path.join(
    search_dir, "trials", "unit_v1_0001", "results", "marker.txt"
)))
self.assertTrue(os.path.isfile(os.path.join(
    search_dir, "trials", "unit_v1_0002", "results", "marker.txt"
)))
```

- [ ] **Step 2: Add default pruning test**

Make the second trial best:

```python
def fake_runner(config):
    calls.append(config)
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "marker.txt"), "w", encoding="utf-8") as handle:
        handle.write("ran")
    if len(calls) == 1:
        return _cnn_metric_result(0.4, 0.4)
    return _cnn_metric_result(0.1, 0.1)
```

Run with:

```python
save_all=False
```

Assert:

```python
self.assertFalse(os.path.exists(os.path.join(search_dir, "trials", "unit_v1_0001")))
self.assertTrue(os.path.exists(os.path.join(search_dir, "trials", "unit_v1_0002")))
self.assertTrue(os.path.isfile(os.path.join(search_dir, "search_results.jsonl")))
self.assertTrue(os.path.isfile(os.path.join(search_dir, "best_results.json")))
```

- [ ] **Step 3: Implement pruning helpers**

Add:

```python
def _best_trial_ids(best_results):
    trial_ids = set()
    for record in best_results.values():
        if record and record.get("trial_id"):
            trial_ids.add(record["trial_id"])
    return trial_ids
```

Add:

```python
def _prune_trial_artifacts(search_dir, records, best_results, save_all):
    if save_all:
        return
    keep_trial_ids = _best_trial_ids(best_results)
    for record in records:
        trial_id = record.get("trial_id")
        trial_dir = record.get("trial_dir")
        if not trial_dir or trial_id in keep_trial_ids:
            continue
        if _is_within_directory(search_dir, trial_dir) and os.path.isdir(trial_dir):
            shutil.rmtree(trial_dir)
```

Call pruning after `_record_progress_files(search_dir, records)` updates `best_results`.

- [ ] **Step 4: Run retention tests**

Run:

```powershell
python -m unittest 3_CNN.tests.test_grid_search -v
```

Expected: PASS.

### Task 5: Save Model Rename Tests

**Files:**

- Modify: `3_CNN/src/cnn_surrogate/grid_search.py`
- Modify: `3_CNN/scripts/grid_search_surrogate.py`
- Modify: `3_CNN/tests/test_grid_search.py`
- Modify: `3_CNN/tests/test_grid_search_script.py`

- [ ] **Step 1: Update backend save test names and arguments**

Rename tests from `save_best` wording to `save_model` wording.

Call:

```python
save_best_model=True
```

instead of:

```python
save_best=True
```

Assert the final official retrain still uses:

```python
self.assertEqual(calls[-1].output_dir, os.path.join("3_CNN", "results", "cnn_surrogate"))
self.assertTrue(calls[-1].save_model)
```

- [ ] **Step 2: Update script dispatch tests**

In `test_main_dispatches_distilled_search`, call:

```python
module.main(["--model", "distilled", "--device", "cpu", "--save-model"])
```

Assert:

```python
self.assertTrue(calls[0]["save_best_model"])
```

and:

```python
self.assertFalse(calls[0]["save_all"])
self.assertFalse(calls[0]["save_figures"])
```

- [ ] **Step 3: Run script and backend tests**

Run:

```powershell
python -m unittest 3_CNN.tests.test_grid_search 3_CNN.tests.test_grid_search_script -v
```

Expected: PASS.

## 10. Manual Verification Commands

After implementation, run:

```powershell
python -m unittest 3_CNN.tests.test_grid_search 3_CNN.tests.test_grid_search_script -v
```

Then run:

```powershell
python -m unittest 3_CNN.tests.test_pipeline 3_CNN.tests.test_coordinate_pipeline 3_CNN.tests.test_distillation_pipeline -v
```

For a small manual dry run, temporarily use a tiny test grid or a local fixture search config, then run:

```powershell
python 3_CNN\scripts\grid_search_surrogate.py --model cnn --device cpu
```

Expected without `--figure`:

```text
3_CNN/results/<result_prefix>_<search_id>/search_results.jsonl
3_CNN/results/<result_prefix>_<search_id>/search_results.csv
3_CNN/results/<result_prefix>_<search_id>/best_results.json
```

Expected:

- No new per-trial folders under `3_CNN/figures/`
- No per-trial `figures/` directory under the search parent
- Non-best completed trial directories pruned unless `--save-all` is passed

Then run with figures enabled:

```powershell
python 3_CNN\scripts\grid_search_surrogate.py --model cnn --device cpu --figure --save-all
```

Expected:

- Each retained trial directory has a `figures/` child directory.
- Every trial directory remains under the search parent.

## 11. Self-Review Checklist

- Every requested CLI change is covered: `--figure`, `--save-all`, and `--save-model`.
- The old `--save` flag is removed from the command-line parser.
- The search parent directory owns all trial result, figure, and temp children.
- Figure generation is disabled by default for grid search and remains enabled for direct training scripts.
- Search-level warm start still works from `search_results.jsonl`.
- Per-trial checkpoint warm start still works for interrupted trials whose directories remain.
- Cleanup cannot remove directories outside the active search parent.
- README files are not modified.
