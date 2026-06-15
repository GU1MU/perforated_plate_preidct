# ODB ML Data Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one Abaqus Python 2.7-compatible extraction script that converts `.odb` files under `2_FEM/temp/odb` into machine-learning-ready summary tables.

**Architecture:** The new script owns the full extraction workflow: ODB discovery, reference normalization from `solid_1_plate.odb`, equivalent stiffness calculation, local strain concentration extraction around each circular hole, warm-start continuation, group-count limiting, and CSV/JSON output. It must not import `compute_odb_stiffness.py`. The script runs with `2_FEM/temp` as its working directory, reads grouped-model hole geometry from a selected-layout JSON file configured by `LAYOUT_FILE`, and uses pure-Python fake ODB objects for unit tests while reserving real ODB access for Abaqus Python.

**Tech Stack:** Python 2.7-compatible standard library, Abaqus `odbAccess`, CSV, JSON, `unittest`, PowerShell.

---

## 1. Scope

Create:

- `2_FEM/scripts/extract_odb_ml_data.py`
- `2_FEM/tests/test_extract_odb_ml_data.py`

Do not modify:

- `2_FEM/scripts/compute_odb_stiffness.py`
- `2_FEM/scripts/generate_roi_indicator_stratified_inp.py`
- Any README file

The script must expose configuration through top-level constants, not command-line arguments.

## 2. Required Behavior

The script defaults to running from `2_FEM/temp`.

Inputs:

- ODB files: `2_FEM/temp/odb/*.odb`
- Selected-layout JSON: `2_FEM/results/pilot_indicator_sampling/seed_20260609_candidates_100000_target_200/selected_layouts.json` by default
- Reference ODB: `2_FEM/temp/odb/solid_1_plate.odb`

Outputs:

- `2_FEM/results/odb_ml_data/odb_ml_summary.csv`
- `2_FEM/results/odb_ml_data/odb_ml_summary.json`
- `2_FEM/results/odb_ml_data/odb_ml_failures.csv`

Configuration constants:

```python
REF = None
LAYOUT_FILE = os.path.join("..", "results", "pilot_indicator_sampling", "seed_20260609_candidates_100000_target_200", "selected_layouts.json")
GROUP_COUNT = None
WARM_START = True
DEFAULT_STEP = "Load"
DEFAULT_FRAME_INDEX = -1
DEFAULT_INSTANCE = "PlateInstance"
DEFAULT_TOP_SET = "TopEdge"
DEFAULT_BOTTOM_SET = "BottomEdge"
DEFAULT_COMPONENT = 2
NEAR_HOLE_BAND_MM = 1.0
```

`REF` behavior:

- `REF = None`: process only grouped models named `{group_index}_{instance_index}_plate.odb`.
- `REF = "solid"`: process only `solid_*_plate.odb`.
- `REF = "uniform"`: process only `uniform_*_plate.odb`.

`GROUP_COUNT` behavior:

- `GROUP_COUNT = None`: no per-group limit.
- `GROUP_COUNT = n`: when `REF = None`, process at most `n` unfinished ODB files per group in this run.
- Apply `GROUP_COUNT` after warm-start filtering, so completed rows do not consume the current run's quota.
- Ignore `GROUP_COUNT` when `REF` is `"solid"` or `"uniform"`.

`WARM_START` behavior:

- If `WARM_START = True`, read existing `odb_ml_summary.csv`.
- Skip rows whose `odb_name` exists in that file with `status == "ok"`.
- Append new successful rows after the existing rows.
- Failed rows must not be considered complete.

## 3. Definitions

For each model, compute the maximum principal strain field value \(\varepsilon_1\).

The solid reference strain is the full-field mean of maximum principal strain from `solid_1_plate.odb`:

$$
\overline{\varepsilon}_{1,\mathrm{solid}}
=
\operatorname{mean}\left(\varepsilon_{1,\mathrm{solid}}\right)
$$

The local strain concentration factor for hole \(i\) is:

$$
K_{\varepsilon,i}
=
\frac{\max(\varepsilon_{1,i})}
{\overline{\varepsilon}_{1,\mathrm{solid}}}
$$

The local region for hole \(i\) is the annular band from the hole boundary to 1.0 mm outside the boundary:

$$
r_i \le d((x,y),(x_i,y_i)) \le r_i + 1.0\mathrm{mm}
$$

The global maximum strain concentration factor is:

$$
K_{\varepsilon,\max}
=
\frac{\max(\varepsilon_{1,\mathrm{model}})}
{\overline{\varepsilon}_{1,\mathrm{solid}}}
$$

The equivalent stiffness must match the formula used by `compute_odb_stiffness.py`, reimplemented locally:

$$
k
=
\frac{
\operatorname{mean}\left(|\overline{RF}_{\mathrm{top}}|,|\overline{RF}_{\mathrm{bottom}}|\right)
}{
\operatorname{mean}\left(|\overline{U}_{\mathrm{top}}|,|\overline{U}_{\mathrm{bottom}}|\right)
}
$$

The relative equivalent stiffness is:

$$
K_{\mathrm{stiff}}
=
\frac{k_{\mathrm{model}}}{k_{\mathrm{solid}}}
$$

## 4. Hole Coordinate Source

Grouped model names use this format:

```text
{group_index}_{instance_index}_plate.odb
```

For grouped models:

1. Parse `group_index` and `instance_index` from the ODB filename.
2. Read the selected-layout JSON configured by `LAYOUT_FILE`.
3. Group layouts using the same nine bins as `generate_roi_indicator_stratified_inp.py`.
4. Use `group_index` to select the layout group and `instance_index - 1` to select the layout within that group.
5. Use that layout's `holes` array as the source of hole centers and radii.
6. If the layout entry or `holes` array is missing, fail that ODB explicitly instead of guessing geometry from the ODB mesh.

For `uniform_*_plate.odb`:

- Reconstruct the 4 by 6 uniform layout used by `generate_roi_indicator_stratified_inp.py`.
- Use \(x\) coordinates from 7.0 to 73.0 and \(y\) coordinates from 7.0 to 153.0.
- Iterate rows by \(y\), then columns by \(x\).

For `solid_*_plate.odb`:

- Use an empty hole list.
- Use `solid_1_plate.odb` as the normalization reference.

`LAYOUT_FILE` must point directly to a selected-layout JSON file with a top-level `layouts` array. It does not accept `pilot_summary.json` and does not read `inp/group_manifest.json`.

## 5. Output Contract

`odb_ml_summary.csv` is a wide table with one row per model. Include these base columns:

```text
odb_name
odb_path
status
ref
group_index
instance_index
step
frame_index
hole_count
solid_mean_max_principal_strain
solid_equivalent_stiffness
model_max_principal_strain
max_strain_concentration_factor
equivalent_stiffness
relative_equivalent_stiffness
top_node_count
bottom_node_count
top_mean_reaction
bottom_mean_reaction
top_mean_displacement
bottom_mean_displacement
warning
```

For holes 1 through 24, add:

```text
hole_01_x
hole_01_y
hole_01_local_max_principal_strain
hole_01_strain_concentration_factor
...
hole_24_x
hole_24_y
hole_24_local_max_principal_strain
hole_24_strain_concentration_factor
```

`odb_ml_summary.json` stores the same rows with a nested `holes` array for each model.

`odb_ml_failures.csv` contains:

```text
odb_name
odb_path
status
message
```

## 6. Implementation Tasks

### Task 1: Script Skeleton And Configuration

**Files:**

- Create: `2_FEM/scripts/extract_odb_ml_data.py`
- Create: `2_FEM/tests/test_extract_odb_ml_data.py`

- [ ] **Step 1: Add the initial test module**

Create `2_FEM/tests/test_extract_odb_ml_data.py` with imports, script-path setup, and tests for the default constants:

```python
import os
import sys
import tempfile
import unittest
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = FEM_ROOT / "scripts"
TEMP_DIR = FEM_ROOT / "temp"

sys.path.insert(0, str(SCRIPT_DIR))

import extract_odb_ml_data as extractor


def tearDownModule():
    try:
        sys.path.remove(str(SCRIPT_DIR))
    except ValueError:
        pass


class ExtractOdbMlDataPathTests(unittest.TestCase):
    def test_default_constants(self):
        self.assertEqual(extractor.DEFAULT_STEP, "Load")
        self.assertEqual(extractor.DEFAULT_INSTANCE, "PlateInstance")
        self.assertEqual(extractor.DEFAULT_TOP_SET, "TopEdge")
        self.assertEqual(extractor.DEFAULT_BOTTOM_SET, "BottomEdge")
        self.assertEqual(extractor.DEFAULT_COMPONENT, 2)
        self.assertEqual(extractor.NEAR_HOLE_BAND_MM, 1.0)
        self.assertIsNone(extractor.REF)
        self.assertIsNone(extractor.GROUP_COUNT)
        self.assertTrue(extractor.WARM_START)

    def test_default_paths_are_temp_relative(self):
        self.assertEqual(extractor.ODB_DIR, os.path.join("odb"))
        self.assertEqual(
            extractor.LAYOUT_FILE,
            os.path.join(
                "..",
                "results",
                "pilot_indicator_sampling",
                "seed_20260609_candidates_100000_target_200",
                "selected_layouts.json",
            ),
        )
        self.assertEqual(extractor.OUTPUT_DIR, os.path.join("..", "results", "odb_ml_data"))
```

- [ ] **Step 2: Run the focused test and confirm the missing module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 2_FEM\tests -p test_extract_odb_ml_data.py
```

Expected:

```text
ModuleNotFoundError: No module named 'extract_odb_ml_data'
```

- [ ] **Step 3: Create the script skeleton**

Create `2_FEM/scripts/extract_odb_ml_data.py` with Python 2.7-compatible syntax:

```python
from __future__ import print_function

import csv
import io
import json
import math
import os
import re
import sys


DEFAULT_STEP = "Load"
DEFAULT_FRAME_INDEX = -1
DEFAULT_TOP_SET = "TopEdge"
DEFAULT_BOTTOM_SET = "BottomEdge"
DEFAULT_INSTANCE = "PlateInstance"
DEFAULT_COMPONENT = 2

PLATE_X = 80.0
PLATE_Y = 160.0
HOLE_RADIUS = 4.0
HOLE_COUNT = 24
MIN_CENTER_TO_EDGE = 7.0
NEAR_HOLE_BAND_MM = 1.0

REF = None
VALID_REFS = (None, "solid", "uniform")
GROUP_COUNT = None
WARM_START = True

ODB_DIR = os.path.join("odb")
LAYOUT_FILE = os.path.join(
    "..",
    "results",
    "pilot_indicator_sampling",
    "seed_20260609_candidates_100000_target_200",
    "selected_layouts.json",
)
OUTPUT_DIR = os.path.join("..", "results", "odb_ml_data")
SUMMARY_CSV = "odb_ml_summary.csv"
SUMMARY_JSON = "odb_ml_summary.json"
FAILURES_CSV = "odb_ml_failures.csv"
REFERENCE_ODB = os.path.join("odb", "solid_1_plate.odb")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
TEMP_DIR = os.path.join(FEM_ROOT, "temp")


def ensure_directory(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)
    return path


def enter_temp_work_dir():
    ensure_directory(TEMP_DIR)
    os.chdir(TEMP_DIR)
    return TEMP_DIR


def main():
    enter_temp_work_dir()
    return 0


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the focused test and confirm it passes**

Expected:

```text
OK
```

### Task 2: ODB Discovery, REF Filtering, And GROUP_COUNT

**Files:**

- Modify: `2_FEM/scripts/extract_odb_ml_data.py`
- Modify: `2_FEM/tests/test_extract_odb_ml_data.py`

- [ ] **Step 1: Add tests for name parsing, REF filtering, and group limiting**

Add tests that assert:

```python
extractor.parse_odb_name("9_150_plate.odb") == {
    "odb_name": "9_150_plate.odb",
    "ref": None,
    "group_index": 9,
    "instance_index": 150,
}
```

Also assert:

```python
extractor.filter_odb_names(
    ["1_1_plate.odb", "2_3_plate.odb", "solid_1_plate.odb", "uniform_1_plate.odb"],
    ref=None,
) == ["1_1_plate.odb", "2_3_plate.odb"]
```

And:

```python
extractor.limit_group_paths_per_group(
    [
        "odb/1_1_plate.odb",
        "odb/1_2_plate.odb",
        "odb/1_3_plate.odb",
        "odb/2_1_plate.odb",
        "odb/2_2_plate.odb",
    ],
    group_count=2,
    ref=None,
) == [
    "odb/1_1_plate.odb",
    "odb/1_2_plate.odb",
    "odb/2_1_plate.odb",
    "odb/2_2_plate.odb",
]
```

- [ ] **Step 2: Run the focused test and confirm missing-function failures**

Expected failure:

```text
AttributeError: module 'extract_odb_ml_data' has no attribute 'parse_odb_name'
```

- [ ] **Step 3: Implement discovery helpers**

Implement:

```python
GROUP_ODB_RE = re.compile(r"^([0-9]+)_([0-9]+)_plate[.]odb$")
REF_ODB_RE = re.compile(r"^(solid|uniform)_([0-9]+)_plate[.]odb$")


def parse_odb_name(name):
    base = os.path.basename(str(name))
    match = GROUP_ODB_RE.match(base)
    if match:
        return {
            "odb_name": base,
            "ref": None,
            "group_index": int(match.group(1)),
            "instance_index": int(match.group(2)),
        }
    match = REF_ODB_RE.match(base)
    if match:
        return {
            "odb_name": base,
            "ref": match.group(1),
            "group_index": None,
            "instance_index": int(match.group(2)),
        }
    return None


def filter_odb_names(names, ref=None):
    if ref not in VALID_REFS:
        raise ValueError("unknown REF: %s" % ref)
    selected = []
    for name in names:
        parsed = parse_odb_name(name)
        if parsed is None:
            continue
        if ref is None and parsed["ref"] is None:
            selected.append(os.path.basename(str(name)))
        elif ref is not None and parsed["ref"] == ref:
            selected.append(os.path.basename(str(name)))
    selected.sort(key=lambda item: _sort_key(item))
    return selected


def limit_group_paths_per_group(odb_paths, group_count=GROUP_COUNT, ref=REF):
    if group_count is None or ref is not None:
        return list(odb_paths)
    count = int(group_count)
    if count < 1:
        raise ValueError("GROUP_COUNT must be a positive integer or None")
    selected = []
    counts = {}
    for path in odb_paths:
        parsed = parse_odb_name(os.path.basename(str(path)))
        if parsed is None or parsed["ref"] is not None:
            continue
        group_index = parsed["group_index"]
        used = counts.get(group_index, 0)
        if used < count:
            selected.append(path)
            counts[group_index] = used + 1
    return selected
```

Also implement `_sort_key()` and `discover_odb_paths()`.

- [ ] **Step 4: Run the focused test and confirm it passes**

### Task 3: Selected Layout JSON And Hole Coordinates

**Files:**

- Modify: `2_FEM/scripts/extract_odb_ml_data.py`
- Modify: `2_FEM/tests/test_extract_odb_ml_data.py`

- [ ] **Step 1: Add tests for hole recovery**

Test these cases:

- `build_uniform_holes()` returns 24 holes.
- The first uniform hole is `(7.0, 7.0)`.
- The last uniform hole is `(73.0, 153.0)`.
- Grouped holes are found by grouping selected layouts by bin and then selecting `instance_index - 1` within the requested group.
- Solid models return an empty hole list.
- `load_layout_payload()` accepts a selected-layout JSON file with a top-level `layouts` array.

- [ ] **Step 2: Implement selected-layout and hole helpers**

Implement:

```python
def read_json_file(path):
    with io.open(path, "r", encoding="utf-8-sig") as json_file:
        return json.load(json_file)


def linspace(start, stop, count):
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [start]
    step = (stop - start) / float(count - 1)
    return [start + step * index for index in range(count)]


def build_uniform_holes():
    holes = []
    for y in linspace(MIN_CENTER_TO_EDGE, PLATE_Y - MIN_CENTER_TO_EDGE, 6):
        for x in linspace(MIN_CENTER_TO_EDGE, PLATE_X - MIN_CENTER_TO_EDGE, 4):
            holes.append({"x": x, "y": y, "r": HOLE_RADIUS})
    return holes


def load_layout_payload(path=LAYOUT_FILE):
    payload = read_json_file(path)
    if not isinstance(payload.get("layouts"), list):
        raise ValueError("layout file has no layouts array: %s" % path)
    return payload
```

Implement bin-to-group helpers and `holes_for_model()` with the rules in Section 4.

- [ ] **Step 3: Run the focused test and confirm it passes**

### Task 4: ODB Field Utilities And Equivalent Stiffness

**Files:**

- Modify: `2_FEM/scripts/extract_odb_ml_data.py`
- Modify: `2_FEM/tests/test_extract_odb_ml_data.py`

- [ ] **Step 1: Add fake ODB classes**

Create fake objects for:

- `rootAssembly.instances`
- instance `nodeSets`
- steps and frames
- `RF` and `U` field outputs
- field subsets returned by `getSubset(region=...)`

- [ ] **Step 2: Add a stiffness test**

Use fake top and bottom values:

```python
top RF: 12.0, 18.0
bottom RF: -14.0, -16.0
top U: 0.42, 0.38
bottom U: -0.41, -0.39
```

Expected:

```python
top_mean_reaction == 15.0
bottom_mean_reaction == -15.0
top_mean_displacement == 0.4
bottom_mean_displacement == -0.4
equivalent_stiffness == 37.5
```

- [ ] **Step 3: Implement local ODB utilities**

Implement local versions of:

- `_case_lookup()`
- `_step()`
- `_frame()`
- `_field_output()`
- `resolve_node_set()`
- `_component_values()`
- `_mean()`
- `_boundary_summary()`
- `compute_equivalent_stiffness()`

Do not import `compute_odb_stiffness.py`.

- [ ] **Step 4: Run the focused test and confirm it passes**

### Task 5: Maximum Principal Strain And Local Ring Extraction

**Files:**

- Modify: `2_FEM/scripts/extract_odb_ml_data.py`
- Modify: `2_FEM/tests/test_extract_odb_ml_data.py`

- [ ] **Step 1: Add strain tests**

Test:

- `max_principal_strain()` prefers `value.maxPrincipal` when present.
- For planar strain data `(e11, e22, e12)`, it computes:

$$
\varepsilon_1
=
\frac{e_{11}+e_{22}}{2}
+
\sqrt{\left(\frac{e_{11}-e_{22}}{2}\right)^2+e_{12}^2}
$$

- `local_hole_max_principal()` only selects values with distance between \(r\) and \(r+1.0\mathrm{mm}\).
- `strain_field_summary()` returns both the full-field mean and the full-field maximum.

- [ ] **Step 2: Implement strain helpers**

Implement:

- `max_principal_strain(value)`
- `value_xy(value)`
- `strain_field_summary(values)`
- `local_hole_max_principal(values, hole, band_mm=NEAR_HOLE_BAND_MM)`
- `strain_values_from_frame(frame)`

- [ ] **Step 3: Add centroid fallback if Abaqus field values do not expose coordinates**

If real ODB execution reports missing coordinates, implement:

```python
def element_centroid_lookup(instance):
    node_coords = {}
    for node in instance.nodes:
        node_coords[int(node.label)] = node.coordinates
    centroids = {}
    for element in instance.elements:
        coords = [node_coords[int(label)] for label in element.connectivity]
        x = sum(float(coord[0]) for coord in coords) / float(len(coords))
        y = sum(float(coord[1]) for coord in coords) / float(len(coords))
        centroids[int(element.label)] = (x, y)
    return centroids
```

Then attach centroid coordinates to strain values by `value.elementLabel`.

- [ ] **Step 4: Run the focused test and confirm it passes**

### Task 6: Reference Normalization And Model Rows

**Files:**

- Modify: `2_FEM/scripts/extract_odb_ml_data.py`
- Modify: `2_FEM/tests/test_extract_odb_ml_data.py`

- [ ] **Step 1: Add row-building tests**

Use:

```python
reference = {
    "solid_mean_max_principal_strain": 0.01,
    "solid_equivalent_stiffness": 100.0,
}
```

For a model with:

```python
model_max_principal_strain = 0.03
local_hole_peak = 0.02
equivalent_stiffness = 40.0
```

Expected:

```python
max_strain_concentration_factor == 3.0
hole_1_strain_concentration_factor == 2.0
relative_equivalent_stiffness == 0.4
```

- [ ] **Step 2: Implement row assembly**

Implement:

- `_safe_ratio(numerator, denominator)`
- `build_model_result(parsed, odb_path, holes, strain_values, stiffness, reference)`

The row must include all base output fields and a nested `holes` array.

- [ ] **Step 3: Run the focused test and confirm it passes**

### Task 7: CSV, JSON, Failures, And Warm Start

**Files:**

- Modify: `2_FEM/scripts/extract_odb_ml_data.py`
- Modify: `2_FEM/tests/test_extract_odb_ml_data.py`

- [ ] **Step 1: Add output tests**

Test:

- `completed_odb_names()` returns only rows with `status == "ok"`.
- `flatten_row()` writes `hole_01_x`, `hole_01_y`, `hole_01_local_max_principal_strain`, and `hole_01_strain_concentration_factor`.
- `paths_after_warm_start()` removes completed ODBs.
- `GROUP_COUNT` is applied after warm-start filtering.

- [ ] **Step 2: Implement output helpers**

Implement:

- `csv_fields()`
- `flatten_row(row)`
- `completed_odb_names(rows)`
- `read_existing_summary(path)`
- `write_summary_csv(rows, output_path)`
- `write_summary_json(rows, output_path)`
- `failure_row(path, message)`
- `write_failures_csv(rows, output_path)`
- `paths_after_warm_start(odb_paths, existing_rows, warm_start=WARM_START)`

- [ ] **Step 3: Run the focused test and confirm it passes**

### Task 8: Batch Runner

**Files:**

- Modify: `2_FEM/scripts/extract_odb_ml_data.py`
- Modify: `2_FEM/tests/test_extract_odb_ml_data.py`

- [ ] **Step 1: Add orchestration tests**

Test that:

- The script text does not contain `import compute_odb_stiffness`.
- The script text does not contain `from compute_odb_stiffness`.
- `paths_after_warm_start()` and `limit_group_paths_per_group()` compose correctly.

- [ ] **Step 2: Implement ODB opening and batch functions**

Implement:

```python
def open_odb(path):
    try:
        from odbAccess import openOdb
    except ImportError:
        raise RuntimeError("odbAccess is unavailable. Run this script with Abaqus Python.")
    return openOdb(path=str(path), readOnly=True)
```

Implement:

- `compute_reference(open_odb_func=open_odb)`
- `process_one_odb(path, layout_payload, reference, open_odb_func=open_odb)`
- `run_configured_analysis(open_odb_func=None)`

`run_configured_analysis()` must:

1. Enter `2_FEM/temp`.
2. Create the output directory.
3. Read existing CSV rows if warm start is enabled.
4. Discover ODB paths using `REF`.
5. Apply warm-start filtering.
6. Apply `GROUP_COUNT`.
7. Load the selected-layout payload from `LAYOUT_FILE` when `REF = None`.
8. Compute the solid reference once.
9. Process each remaining ODB.
10. Write summary CSV, summary JSON, and failures CSV.

- [ ] **Step 3: Wire `main()` to `run_configured_analysis()`**

Use:

```python
def main():
    run_configured_analysis()
    return 0
```

- [ ] **Step 4: Run the focused test and confirm it passes**

### Task 9: Verification

**Files:**

- Verify: `2_FEM/scripts/extract_odb_ml_data.py`
- Verify: `2_FEM/tests/test_extract_odb_ml_data.py`

- [ ] **Step 1: Run the focused test**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 2_FEM\tests -p test_extract_odb_ml_data.py
```

Expected:

```text
OK
```

- [ ] **Step 2: Run the FEM test suite**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 2_FEM\tests
```

Expected:

```text
OK
```

- [ ] **Step 3: Run Python syntax verification**

```powershell
.\.venv\Scripts\python.exe -m py_compile 2_FEM\scripts\extract_odb_ml_data.py
```

Expected: no output and exit code 0.

- [ ] **Step 4: Run an Abaqus smoke extraction**

From the repository root:

```powershell
cd 2_FEM\temp
abaqus python ..\scripts\extract_odb_ml_data.py
```

Expected output files:

```text
2_FEM/results/odb_ml_data/odb_ml_summary.csv
2_FEM/results/odb_ml_data/odb_ml_summary.json
2_FEM/results/odb_ml_data/odb_ml_failures.csv
```

- [ ] **Step 5: Inspect the CSV header**

```powershell
Get-Content 2_FEM\results\odb_ml_data\odb_ml_summary.csv -TotalCount 1
```

The header must include:

```text
max_strain_concentration_factor
relative_equivalent_stiffness
hole_01_x
hole_01_y
hole_01_local_max_principal_strain
hole_01_strain_concentration_factor
```

## 7. Acceptance Checklist

- [ ] `2_FEM/scripts/extract_odb_ml_data.py` exists.
- [ ] `2_FEM/tests/test_extract_odb_ml_data.py` exists.
- [ ] The script does not import `compute_odb_stiffness.py`.
- [ ] The script has top-level `REF`, `GROUP_COUNT`, and `WARM_START` constants.
- [ ] The default working directory is `2_FEM/temp`.
- [ ] `REF = None` processes only grouped models.
- [ ] `REF = "solid"` processes only `solid_*_plate.odb`.
- [ ] `REF = "uniform"` processes only `uniform_*_plate.odb`.
- [ ] `GROUP_COUNT` limits the number of unfinished ODBs processed per group.
- [ ] The grouped-model hole coordinates come from the selected-layout JSON configured by `LAYOUT_FILE`.
- [ ] `solid_1_plate.odb` provides both normalization references.
- [ ] Local strain concentration uses the 1.0 mm annular band outside each hole boundary.
- [ ] Output includes global maximum strain concentration, relative equivalent stiffness, all hole centers, and all local strain concentration factors.
- [ ] Warm start skips existing successful rows.
- [ ] Unit tests pass.
- [ ] Abaqus Python smoke extraction passes or records explicit failure rows.
