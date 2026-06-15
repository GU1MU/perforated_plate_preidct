# ROI Indicator Stratified FEM Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增两个独立脚本，在不修改现有FEM随机建模方案的前提下，先用pilot脚本完成指标分层采样，再用Abaqus兼容脚本对筛选出的孔分布写出真实INP模型。

**Architecture:** `pilot_roi_indicator_sampling.py`只使用Python标准库，负责生成满足几何约束的候选孔分布、计算\(C\)和\(A_{\mathrm{orient}}\)、按pilot分位数分层，并把结果写入`2_FEM/results/pilot_indicator_sampling`。`generate_roi_indicator_stratified_inp.py`读取pilot输出的`selected_layouts.json`，复核几何和指标，在Abaqus/CAE的Python2.7环境中创建\(80\mathrm{mm}\times160\mathrm{mm}\times2\mathrm{mm}\)中心研究区模型并写出INP。两个脚本都保持Python2.7兼容语法，不使用f-string、dataclass、pathlib、类型注解或仅Python3可用的标准库API。

**Tech Stack:** Python2.7兼容标准库、Abaqus/CAE noGUI、CSV、JSON、PowerShell、项目虚拟环境中的Python用于pilot和dry-run验证。

---

## 1. Scope And Constraints

本计划只新增以下两个脚本：

- Create: `2_FEM/scripts/pilot_roi_indicator_sampling.py`
- Create: `2_FEM/scripts/generate_roi_indicator_stratified_inp.py`

本计划不修改以下既有文件：

- `2_FEM/scripts/generate_perforated_plate_inp.py`
- `2_FEM/scripts/visualize_sampling_domains.py`
- `2_FEM/tests/test_generate_perforated_plate_inp.py`
- `0_proposal/pre-research plan.md`
- `0_proposal/outline.md`
- 任意README文件

pilot结果写入：

- `2_FEM/results/pilot_indicator_sampling/<run_id>/pilot_samples.csv`
- `2_FEM/results/pilot_indicator_sampling/<run_id>/pilot_summary.json`
- `2_FEM/results/pilot_indicator_sampling/<run_id>/selected_layouts.json`

真实建模脚本写入：

- `2_FEM/temp/roi_indicator_stratified_inp/<layout_id>_plate.inp`
- `2_FEM/temp/roi_indicator_stratified_inp/roi_indicator_manifest.json`

几何常量采用：

$$
L_x=80\mathrm{mm},\quad L_y=160\mathrm{mm},\quad t=2\mathrm{mm}
$$

$$
r=4\mathrm{mm},\quad n=24,\quad d_{\min}=10\mathrm{mm},\quad e_{\min}=7\mathrm{mm}
$$

其中\(d_{\min}\)是孔心最小距离，\(e_{\min}\)是孔心到研究区边界的最小距离。若实施前确认\(d_{\min}\)和\(e_{\min}\)表示孔边间距，则把脚本常量改为\(d_{\min}=18\mathrm{mm}\)、\(e_{\min}=11\mathrm{mm}\)。

## 2. Sampling Design

每个候选孔分布都保存连续指标。聚集程度为：

$$
C=\frac{d_{\mathrm{eq}}}{d_{\mathrm{NN,avg}}}
$$

本项目孔径固定，因此：

$$
d_{\mathrm{eq}}=8\mathrm{mm}
$$

方向性使用ROI归一化坐标计算：

$$
u=\frac{x}{80},\qquad v=\frac{y}{160}
$$

$$
A_{\mathrm{orient}}=
\frac{\operatorname{Var}(u)-\operatorname{Var}(v)}
{\operatorname{Var}(u)+\operatorname{Var}(v)}
$$

pilot先生成候选样本，再用分位数确定标签边界：

- \(C\le Q_{33}(C)\)：`cluster_low`
- \(Q_{33}(C)<C\le Q_{67}(C)\)：`cluster_medium`
- \(C>Q_{67}(C)\)：`cluster_high`
- \(A_{\mathrm{orient}}\le Q_{33}(A)\)：`orient_y`
- \(Q_{33}(A)<A_{\mathrm{orient}}<Q_{67}(A)\)：`orient_none`
- \(A_{\mathrm{orient}}\ge Q_{67}(A)\)：`orient_x`

正式选择阶段按`cluster_label × orient_label`组成九个箱，每个箱选取相同数量样本。标签只用于平衡数据集，论文和后处理应优先报告真实\(C\)和\(A_{\mathrm{orient}}\)。

## 3. File Responsibilities

`2_FEM/scripts/pilot_roi_indicator_sampling.py`负责：

- 解析命令行参数。
- 生成满足几何约束的候选孔分布。
- 计算最近邻距离、\(C\)、\(A_{\mathrm{orient}}\)。
- 使用pilot分位数创建九个分层箱。
- 按每箱目标数量筛选样本。
- 写出CSV、JSON和可复用的`selected_layouts.json`。

`2_FEM/scripts/generate_roi_indicator_stratified_inp.py`负责：

- 读取`selected_layouts.json`。
- 复核孔数量、孔心边界约束、孔心间距约束。
- 复核并记录\(C\)、\(A_{\mathrm{orient}}\)。
- 在Abaqus环境中创建二维平面应力板模型。
- 写出INP和建模manifest。
- 支持`--dry-run`，在没有Abaqus模块时只执行读取、复核和manifest预览。

## 4. Implementation Tasks

### Task 1: Create The Pilot Script Skeleton

**Files:**

- Create: `2_FEM/scripts/pilot_roi_indicator_sampling.py`

- [ ] **Step 1: Create the Python2.7-compatible module header and constants**

Use this exact file header and constants:

```python
from __future__ import print_function

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import sys


PLATE_X = 80.0
PLATE_Y = 160.0
PLATE_THICKNESS = 2.0
HOLE_RADIUS = 4.0
HOLE_DIAMETER = 2.0 * HOLE_RADIUS
HOLE_COUNT = 24
MIN_CENTER_DISTANCE = 10.0
MIN_CENTER_TO_EDGE = 7.0

DEFAULT_CANDIDATE_COUNT = 10000
DEFAULT_TARGET_PER_BIN = 20
DEFAULT_SEED = 20260609
MAX_ATTEMPTS_PER_HOLE = 5000
MAX_LAYOUT_RESTARTS = 200

RESULTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "results", "pilot_indicator_sampling")
)
```

- [ ] **Step 2: Add path and JSON helpers**

Add these helpers after constants:

```python
def ensure_directory(path):
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


def write_json_file(path, data):
    ensure_directory(os.path.dirname(os.path.abspath(path)))
    text = json.dumps(data, indent=2, sort_keys=True)
    if sys.version_info[0] < 3:
        text = text.decode("ascii")
    with io.open(path, "w", encoding="utf-8") as json_file:
        json_file.write(text)
        json_file.write(u"\n")
    return path


def read_json_file(path):
    with io.open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)
```

- [ ] **Step 3: Run import smoke check**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import importlib.util; spec=importlib.util.spec_from_file_location('pilot','2_FEM/scripts/pilot_roi_indicator_sampling.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); print(module.PLATE_X, module.HOLE_COUNT)"
```

Expected:

```text
80.0 24
```

### Task 2: Implement Geometry Generation And Validation

**Files:**

- Modify: `2_FEM/scripts/pilot_roi_indicator_sampling.py`

- [ ] **Step 1: Add deterministic seed helpers**

```python
def seed_for_candidate(base_seed, candidate_index, restart_index):
    text = "%s:%s:%s" % (base_seed, candidate_index, restart_index)
    digest = hashlib.md5(text.encode("ascii")).hexdigest()
    return int(digest[:12], 16)
```

- [ ] **Step 2: Add distance and validation functions**

```python
def squared_distance(left, right):
    dx = left["x"] - right["x"]
    dy = left["y"] - right["y"]
    return dx * dx + dy * dy


def is_far_enough(candidate, holes):
    min_distance_sq = MIN_CENTER_DISTANCE * MIN_CENTER_DISTANCE
    for hole in holes:
        if squared_distance(candidate, hole) < min_distance_sq:
            return False
    return True


def validate_holes(holes):
    if len(holes) != HOLE_COUNT:
        raise ValueError("expected %d holes, got %d" % (HOLE_COUNT, len(holes)))

    xmin = MIN_CENTER_TO_EDGE
    xmax = PLATE_X - MIN_CENTER_TO_EDGE
    ymin = MIN_CENTER_TO_EDGE
    ymax = PLATE_Y - MIN_CENTER_TO_EDGE

    for index, hole in enumerate(holes):
        x = hole["x"]
        y = hole["y"]
        if x < xmin or x > xmax or y < ymin or y > ymax:
            raise ValueError("hole %d outside allowed bounds" % index)

    min_distance_sq = MIN_CENTER_DISTANCE * MIN_CENTER_DISTANCE
    for left_index in range(len(holes)):
        for right_index in range(left_index + 1, len(holes)):
            if squared_distance(holes[left_index], holes[right_index]) < min_distance_sq:
                raise ValueError(
                    "holes %d and %d violate minimum center distance"
                    % (left_index, right_index)
                )
    return None
```

- [ ] **Step 3: Add candidate generator**

```python
def generate_candidate_holes(base_seed, candidate_index):
    xmin = MIN_CENTER_TO_EDGE
    xmax = PLATE_X - MIN_CENTER_TO_EDGE
    ymin = MIN_CENTER_TO_EDGE
    ymax = PLATE_Y - MIN_CENTER_TO_EDGE

    best_count = 0
    for restart_index in range(MAX_LAYOUT_RESTARTS):
        rng = random.Random(seed_for_candidate(base_seed, candidate_index, restart_index))
        holes = []
        while len(holes) < HOLE_COUNT:
            accepted = False
            for _attempt in range(MAX_ATTEMPTS_PER_HOLE):
                candidate = {
                    "x": rng.uniform(xmin, xmax),
                    "y": rng.uniform(ymin, ymax),
                    "r": HOLE_RADIUS,
                }
                if is_far_enough(candidate, holes):
                    holes.append(candidate)
                    accepted = True
                    break
            if not accepted:
                break

        if len(holes) == HOLE_COUNT:
            validate_holes(holes)
            return holes
        if len(holes) > best_count:
            best_count = len(holes)

    raise ValueError(
        "could not generate candidate %s: accepted %d holes"
        % (candidate_index, best_count)
    )
```

- [ ] **Step 4: Run geometry smoke check**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import importlib.util; spec=importlib.util.spec_from_file_location('pilot','2_FEM/scripts/pilot_roi_indicator_sampling.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); holes=m.generate_candidate_holes(1,1); m.validate_holes(holes); print(len(holes), round(holes[0]['r'], 1))"
```

Expected:

```text
24 4.0
```

### Task 3: Implement Continuous Metrics

**Files:**

- Modify: `2_FEM/scripts/pilot_roi_indicator_sampling.py`

- [ ] **Step 1: Add nearest-neighbor and variance helpers**

```python
def nearest_neighbor_distances(holes):
    distances = []
    for left_index, left in enumerate(holes):
        nearest_sq = None
        for right_index, right in enumerate(holes):
            if left_index == right_index:
                continue
            dist_sq = squared_distance(left, right)
            if nearest_sq is None or dist_sq < nearest_sq:
                nearest_sq = dist_sq
        distances.append(math.sqrt(nearest_sq))
    return distances


def mean(values):
    return sum(values) / float(len(values))


def variance(values):
    avg = mean(values)
    return sum((value - avg) * (value - avg) for value in values) / float(len(values))
```

- [ ] **Step 2: Add metric calculation**

```python
def calculate_metrics(holes):
    nearest = nearest_neighbor_distances(holes)
    d_nn_avg = mean(nearest)
    cluster_index = HOLE_DIAMETER / d_nn_avg

    normalized_x = [hole["x"] / PLATE_X for hole in holes]
    normalized_y = [hole["y"] / PLATE_Y for hole in holes]
    var_x = variance(normalized_x)
    var_y = variance(normalized_y)
    denominator = var_x + var_y
    if denominator <= 0.0:
        orientation_index = 0.0
    else:
        orientation_index = (var_x - var_y) / denominator

    return {
        "d_eq": HOLE_DIAMETER,
        "d_nn_avg": d_nn_avg,
        "cluster_index": cluster_index,
        "orientation_index": orientation_index,
        "var_x_norm": var_x,
        "var_y_norm": var_y,
    }
```

- [ ] **Step 3: Run metric smoke check**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import importlib.util; spec=importlib.util.spec_from_file_location('pilot','2_FEM/scripts/pilot_roi_indicator_sampling.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); holes=m.generate_candidate_holes(1,1); metrics=m.calculate_metrics(holes); print(round(metrics['d_eq'],1), metrics['cluster_index'] > 0, -1.0 <= metrics['orientation_index'] <= 1.0)"
```

Expected:

```text
8.0 True True
```

### Task 4: Implement Pilot Binning And Balanced Selection

**Files:**

- Modify: `2_FEM/scripts/pilot_roi_indicator_sampling.py`

- [ ] **Step 1: Add quantile and label helpers**

```python
def quantile(values, fraction):
    if not values:
        raise ValueError("cannot calculate quantile for empty values")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def label_cluster(value, thresholds):
    if value <= thresholds["cluster_q33"]:
        return "low"
    if value <= thresholds["cluster_q67"]:
        return "medium"
    return "high"


def label_orientation(value, thresholds):
    if value <= thresholds["orientation_q33"]:
        return "y"
    if value >= thresholds["orientation_q67"]:
        return "x"
    return "none"


def bin_key(record):
    return "%s_%s" % (record["cluster_label"], record["orientation_label"])
```

- [ ] **Step 2: Add candidate record construction**

```python
def layout_id_for_record(seed, candidate_index):
    return "roi_%s_%05d" % (seed, candidate_index)


def build_candidate_records(candidate_count, seed):
    records = []
    for candidate_index in range(1, candidate_count + 1):
        holes = generate_candidate_holes(seed, candidate_index)
        metrics = calculate_metrics(holes)
        record = {
            "layout_id": layout_id_for_record(seed, candidate_index),
            "candidate_index": candidate_index,
            "holes": holes,
            "metrics": metrics,
        }
        records.append(record)
    return records
```

- [ ] **Step 3: Add threshold calculation and selection**

```python
def calculate_thresholds(records):
    cluster_values = [record["metrics"]["cluster_index"] for record in records]
    orientation_values = [record["metrics"]["orientation_index"] for record in records]
    return {
        "cluster_q33": quantile(cluster_values, 1.0 / 3.0),
        "cluster_q67": quantile(cluster_values, 2.0 / 3.0),
        "orientation_q33": quantile(orientation_values, 1.0 / 3.0),
        "orientation_q67": quantile(orientation_values, 2.0 / 3.0),
    }


def attach_labels(records, thresholds):
    for record in records:
        metrics = record["metrics"]
        record["cluster_label"] = label_cluster(metrics["cluster_index"], thresholds)
        record["orientation_label"] = label_orientation(metrics["orientation_index"], thresholds)
        record["bin"] = bin_key(record)
    return records


def select_balanced_records(records, target_per_bin):
    selected = []
    counts = {}
    for cluster_label in ("low", "medium", "high"):
        for orientation_label in ("x", "none", "y"):
            counts["%s_%s" % (cluster_label, orientation_label)] = 0

    for record in records:
        key = record["bin"]
        if counts[key] >= target_per_bin:
            continue
        selected.append(record)
        counts[key] += 1
        if min(counts.values()) >= target_per_bin:
            break

    return selected, counts
```

- [ ] **Step 4: Run binning smoke check**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import importlib.util; spec=importlib.util.spec_from_file_location('pilot','2_FEM/scripts/pilot_roi_indicator_sampling.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); records=m.build_candidate_records(120,1); thresholds=m.calculate_thresholds(records); m.attach_labels(records,thresholds); selected,counts=m.select_balanced_records(records,2); print(len(selected) >= 12, len(counts))"
```

Expected:

```text
True 9
```

### Task 5: Implement Pilot Outputs

**Files:**

- Modify: `2_FEM/scripts/pilot_roi_indicator_sampling.py`

- [ ] **Step 1: Add CSV writer**

```python
def write_pilot_csv(path, records):
    ensure_directory(os.path.dirname(os.path.abspath(path)))
    with open(path, "wb" if sys.version_info[0] < 3 else "w") as csv_file:
        if sys.version_info[0] < 3:
            writer = csv.writer(csv_file)
        else:
            writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow([
            "layout_id",
            "candidate_index",
            "cluster_label",
            "orientation_label",
            "bin",
            "cluster_index",
            "orientation_index",
            "d_nn_avg",
            "var_x_norm",
            "var_y_norm",
        ])
        for record in records:
            metrics = record["metrics"]
            writer.writerow([
                record["layout_id"],
                record["candidate_index"],
                record.get("cluster_label", ""),
                record.get("orientation_label", ""),
                record.get("bin", ""),
                "%.12g" % metrics["cluster_index"],
                "%.12g" % metrics["orientation_index"],
                "%.12g" % metrics["d_nn_avg"],
                "%.12g" % metrics["var_x_norm"],
                "%.12g" % metrics["var_y_norm"],
            ])
    return path
```

- [ ] **Step 2: Add output payload builders**

```python
def selected_layout_payload(selected, thresholds, counts, seed, candidate_count, target_per_bin):
    return {
        "schema": "roi_indicator_stratified_layouts_v1",
        "seed": seed,
        "candidate_count": candidate_count,
        "target_per_bin": target_per_bin,
        "geometry": {
            "plate_x": PLATE_X,
            "plate_y": PLATE_Y,
            "plate_thickness": PLATE_THICKNESS,
            "hole_radius": HOLE_RADIUS,
            "hole_count": HOLE_COUNT,
            "min_center_distance": MIN_CENTER_DISTANCE,
            "min_center_to_edge": MIN_CENTER_TO_EDGE,
        },
        "thresholds": thresholds,
        "bin_counts": counts,
        "layouts": selected,
    }


def write_pilot_outputs(records, selected, thresholds, counts, output_dir, seed, candidate_count, target_per_bin):
    ensure_directory(output_dir)
    samples_csv = os.path.join(output_dir, "pilot_samples.csv")
    selected_json = os.path.join(output_dir, "selected_layouts.json")
    summary_json = os.path.join(output_dir, "pilot_summary.json")

    write_pilot_csv(samples_csv, records)
    payload = selected_layout_payload(selected, thresholds, counts, seed, candidate_count, target_per_bin)
    write_json_file(selected_json, payload)
    write_json_file(summary_json, {
        "schema": "roi_indicator_pilot_summary_v1",
        "seed": seed,
        "candidate_count": candidate_count,
        "selected_count": len(selected),
        "target_per_bin": target_per_bin,
        "thresholds": thresholds,
        "bin_counts": counts,
        "outputs": {
            "pilot_samples_csv": os.path.abspath(samples_csv),
            "selected_layouts_json": os.path.abspath(selected_json),
        },
    })
    return {
        "pilot_samples_csv": samples_csv,
        "selected_layouts_json": selected_json,
        "pilot_summary_json": summary_json,
    }
```

- [ ] **Step 3: Add CLI entrypoint**

```python
def default_run_id(seed, candidate_count, target_per_bin):
    return "seed_%s_candidates_%s_target_%s" % (seed, candidate_count, target_per_bin)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Pilot indicator-stratified ROI hole sampling.")
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--target-per-bin", type=int, default=DEFAULT_TARGET_PER_BIN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.candidate_count < 9:
        raise ValueError("candidate-count must be at least 9")
    if args.target_per_bin < 1:
        raise ValueError("target-per-bin must be at least 1")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            RESULTS_ROOT,
            default_run_id(args.seed, args.candidate_count, args.target_per_bin),
        )

    records = build_candidate_records(args.candidate_count, args.seed)
    thresholds = calculate_thresholds(records)
    attach_labels(records, thresholds)
    selected, counts = select_balanced_records(records, args.target_per_bin)
    outputs = write_pilot_outputs(
        records,
        selected,
        thresholds,
        counts,
        output_dir,
        args.seed,
        args.candidate_count,
        args.target_per_bin,
    )
    print("Wrote pilot samples: %s" % outputs["pilot_samples_csv"])
    print("Wrote selected layouts: %s" % outputs["selected_layouts_json"])
    print("Wrote pilot summary: %s" % outputs["pilot_summary_json"])
    return outputs


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run pilot smoke command**

Run:

```powershell
.\.venv\Scripts\python.exe 2_FEM\scripts\pilot_roi_indicator_sampling.py --candidate-count 120 --target-per-bin 2 --seed 1 --output-dir 2_FEM\results\pilot_indicator_sampling\smoke_seed_1
```

Expected output contains:

```text
Wrote pilot samples:
Wrote selected layouts:
Wrote pilot summary:
```

Expected files:

```text
2_FEM\results\pilot_indicator_sampling\smoke_seed_1\pilot_samples.csv
2_FEM\results\pilot_indicator_sampling\smoke_seed_1\selected_layouts.json
2_FEM\results\pilot_indicator_sampling\smoke_seed_1\pilot_summary.json
```

### Task 6: Create The Abaqus-Compatible Real Modeling Script

**Files:**

- Create: `2_FEM/scripts/generate_roi_indicator_stratified_inp.py`

- [ ] **Step 1: Create Python2.7-compatible header, constants, and IO helpers**

Use this skeleton:

```python
from __future__ import print_function

import argparse
import io
import json
import math
import os
import sys


PLATE_X = 80.0
PLATE_Y = 160.0
PLATE_THICKNESS = 2.0
HOLE_RADIUS = 4.0
HOLE_DIAMETER = 2.0 * HOLE_RADIUS
HOLE_COUNT = 24
MIN_CENTER_DISTANCE = 10.0
MIN_CENTER_TO_EDGE = 7.0

DEFAULT_E = 2100.0
DEFAULT_NU = 0.33
DEFAULT_U = 2.0
DEFAULT_MESH_SIZE = 2.0
ELEMENT_TYPE = "CPS6"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
TEMP_DIR = os.path.join(FEM_ROOT, "temp")
DEFAULT_OUTPUT_DIR = os.path.join(TEMP_DIR, "roi_indicator_stratified_inp")


def ensure_directory(path):
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


def read_json_file(path):
    with io.open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def write_json_file(path, data):
    ensure_directory(os.path.dirname(os.path.abspath(path)))
    text = json.dumps(data, indent=2, sort_keys=True)
    if sys.version_info[0] < 3:
        text = text.decode("ascii")
    with io.open(path, "w", encoding="utf-8") as json_file:
        json_file.write(text)
        json_file.write(u"\n")
    return path
```

- [ ] **Step 2: Add validation and metrics copied to keep this file standalone**

Add standalone versions of `squared_distance`、`validate_holes`、`nearest_neighbor_distances`、`mean`、`variance`、`calculate_metrics` from `Task 2` and `Task 3`. Do not import the pilot脚本 because Abaqus路径解析和项目虚拟环境路径可能不同.

- [ ] **Step 3: Add layout loading**

```python
def load_layout_payload(path):
    payload = read_json_file(path)
    if payload.get("schema") != "roi_indicator_stratified_layouts_v1":
        raise ValueError("unsupported selected layout schema: %s" % payload.get("schema"))
    layouts = payload.get("layouts", [])
    if not layouts:
        raise ValueError("selected layout file has no layouts")
    for layout in layouts:
        holes = layout.get("holes")
        if holes is None:
            raise ValueError("layout %s has no holes" % layout.get("layout_id"))
        validate_holes(holes)
    return payload
```

- [ ] **Step 4: Add run-plan construction**

```python
def safe_job_stem(name):
    chars = []
    for char in name:
        if char.isalnum() or char == "_":
            chars.append(char)
        else:
            chars.append("_")
    stem = "".join(chars)
    if not stem:
        stem = "roi_layout"
    if not stem[0].isalpha():
        stem = "job_" + stem
    return stem


def build_run_plan(payload, output_dir):
    ensure_directory(output_dir)
    plan = []
    for layout in payload["layouts"]:
        layout_id = layout["layout_id"]
        job_stem = safe_job_stem(layout_id + "_plate")
        holes = layout["holes"]
        metrics = calculate_metrics(holes)
        plan.append({
            "layout_id": layout_id,
            "cluster_label": layout.get("cluster_label"),
            "orientation_label": layout.get("orientation_label"),
            "bin": layout.get("bin"),
            "holes": holes,
            "metrics": metrics,
            "job_name": job_stem,
            "inp_name": job_stem + ".inp",
            "inp_path": os.path.abspath(os.path.join(output_dir, job_stem + ".inp")),
        })
    return plan
```

- [ ] **Step 5: Add manifest writer**

```python
def build_manifest(payload, plan, material_e, displacement_u, mesh_size):
    return {
        "schema": "roi_indicator_abaqus_manifest_v1",
        "source": {
            "pilot_schema": payload.get("schema"),
            "pilot_seed": payload.get("seed"),
            "pilot_candidate_count": payload.get("candidate_count"),
            "pilot_target_per_bin": payload.get("target_per_bin"),
            "thresholds": payload.get("thresholds"),
            "bin_counts": payload.get("bin_counts"),
        },
        "geometry": {
            "plate_x": PLATE_X,
            "plate_y": PLATE_Y,
            "plate_thickness": PLATE_THICKNESS,
            "hole_radius": HOLE_RADIUS,
            "hole_count": HOLE_COUNT,
            "min_center_distance": MIN_CENTER_DISTANCE,
            "min_center_to_edge": MIN_CENTER_TO_EDGE,
        },
        "material": {
            "E": material_e,
            "nu": DEFAULT_NU,
        },
        "load": {
            "u": displacement_u,
        },
        "mesh_size": mesh_size,
        "element_type": ELEMENT_TYPE,
        "runs": [
            {
                "layout_id": item["layout_id"],
                "cluster_label": item["cluster_label"],
                "orientation_label": item["orientation_label"],
                "bin": item["bin"],
                "inp_name": item["inp_name"],
                "inp_path": item["inp_path"],
                "metrics": item["metrics"],
                "hole_count": len(item["holes"]),
            }
            for item in plan
        ],
    }
```

- [ ] **Step 6: Add dry-run CLI**

```python
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Generate ROI indicator-stratified Abaqus INP files.")
    parser.add_argument("selected_layouts_json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--material-e", type=float, default=DEFAULT_E)
    parser.add_argument("--displacement-u", type=float, default=DEFAULT_U)
    parser.add_argument("--mesh-size", type=float, default=DEFAULT_MESH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    payload = load_layout_payload(args.selected_layouts_json)
    plan = build_run_plan(payload, args.output_dir)
    manifest = build_manifest(payload, plan, args.material_e, args.displacement_u, args.mesh_size)
    manifest_path = os.path.join(args.output_dir, "roi_indicator_manifest.json")
    write_json_file(manifest_path, manifest)
    if args.dry_run:
        print("Dry run validated %d layouts" % len(plan))
        print("Wrote manifest: %s" % manifest_path)
        return manifest_path
    for item in plan:
        saved_path = write_inp_with_abaqus(item, args.material_e, args.displacement_u, args.mesh_size)
        print("Saved inp: %s" % saved_path)
    return manifest_path


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run dry-run after pilot smoke output exists**

Run:

```powershell
.\.venv\Scripts\python.exe 2_FEM\scripts\generate_roi_indicator_stratified_inp.py 2_FEM\results\pilot_indicator_sampling\smoke_seed_1\selected_layouts.json --output-dir 2_FEM\temp\roi_indicator_stratified_inp_smoke --dry-run
```

Expected output contains:

```text
Dry run validated
Wrote manifest:
```

Expected file:

```text
2_FEM\temp\roi_indicator_stratified_inp_smoke\roi_indicator_manifest.json
```

### Task 7: Implement Abaqus INP Writing

**Files:**

- Modify: `2_FEM/scripts/generate_roi_indicator_stratified_inp.py`

- [ ] **Step 1: Add Abaqus job helper**

```python
def move_generated_inp(source_path, target_path):
    source_abs = os.path.abspath(source_path)
    target_abs = os.path.abspath(target_path)
    ensure_directory(os.path.dirname(target_abs))
    if source_abs == target_abs:
        return target_abs
    if os.path.exists(target_abs):
        os.remove(target_abs)
    os.rename(source_abs, target_abs)
    return target_abs
```

- [ ] **Step 2: Add Abaqus writer**

```python
def write_inp_with_abaqus(run_item, material_e, displacement_u, mesh_size):
    ensure_directory(DEFAULT_OUTPUT_DIR)
    ensure_directory(TEMP_DIR)
    os.chdir(TEMP_DIR)

    from abaqus import mdb
    from abaqusConstants import TWO_D_PLANAR, DEFORMABLE_BODY, ON, OFF, UNSET, STANDARD, TRI, CPS6
    from mesh import ElemType

    job_name = run_item["job_name"]
    model_name = "Model_%s" % job_name
    if model_name in mdb.models:
        del mdb.models[model_name]
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]

    model = mdb.Model(name=model_name)
    sketch = model.ConstrainedSketch(
        name="PlateProfile",
        sheetSize=2.0 * max(PLATE_X, PLATE_Y),
    )
    sketch.rectangle(point1=(0.0, 0.0), point2=(PLATE_X, PLATE_Y))
    for hole in run_item["holes"]:
        x = hole["x"]
        y = hole["y"]
        r = hole.get("r", HOLE_RADIUS)
        sketch.CircleByCenterPerimeter(center=(x, y), point1=(x + r, y))

    part = model.Part(
        name="Plate",
        dimensionality=TWO_D_PLANAR,
        type=DEFORMABLE_BODY,
    )
    part.BaseShell(sketch=sketch)
    faces = part.faces[:]

    material = model.Material(name="PlateMaterial")
    material.Elastic(table=((material_e, DEFAULT_NU),))
    model.HomogeneousSolidSection(
        name="PlateSection",
        material="PlateMaterial",
        thickness=PLATE_THICKNESS,
    )
    face_set = part.Set(faces=faces, name="PlateFace")
    part.SectionAssignment(region=face_set, sectionName="PlateSection")

    tol = 1.0e-6
    bottom_edges = part.edges.getByBoundingBox(
        xMin=-tol,
        yMin=-tol,
        zMin=-tol,
        xMax=PLATE_X + tol,
        yMax=tol,
        zMax=tol,
    )
    top_edges = part.edges.getByBoundingBox(
        xMin=-tol,
        yMin=PLATE_Y - tol,
        zMin=-tol,
        xMax=PLATE_X + tol,
        yMax=PLATE_Y + tol,
        zMax=tol,
    )
    lower_left_vertices = part.vertices.getByBoundingBox(
        xMin=-tol,
        yMin=-tol,
        zMin=-tol,
        xMax=tol,
        yMax=tol,
        zMax=tol,
    )
    part.Set(edges=bottom_edges, name="BottomEdge")
    part.Set(edges=top_edges, name="TopEdge")
    part.Set(vertices=lower_left_vertices, name="LowerLeftVertex")

    assembly = model.rootAssembly
    instance = assembly.Instance(name="PlateInstance", part=part, dependent=ON)
    model.StaticStep(name="Load", previous="Initial")
    model.fieldOutputRequests["F-Output-1"].setValues(variables=("S", "E", "U", "RF"))
    model.DisplacementBC(
        name="BottomUY",
        createStepName="Load",
        region=instance.sets["BottomEdge"],
        u1=UNSET,
        u2=-0.5 * displacement_u,
    )
    model.DisplacementBC(
        name="TopUY",
        createStepName="Load",
        region=instance.sets["TopEdge"],
        u1=UNSET,
        u2=0.5 * displacement_u,
    )
    model.DisplacementBC(
        name="FixLowerLeftUX",
        createStepName="Load",
        region=instance.sets["LowerLeftVertex"],
        u1=0.0,
        u2=UNSET,
    )

    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.setMeshControls(regions=faces, elemShape=TRI)
    elem_type = ElemType(elemCode=CPS6, elemLibrary=STANDARD)
    part.setElementType(regions=(faces,), elemTypes=(elem_type,))
    part.generateMesh()

    job = mdb.Job(name=job_name, model=model.name)
    job.writeInput(consistencyChecking=OFF)
    return move_generated_inp(job_name + ".inp", run_item["inp_path"])
```

- [ ] **Step 3: Run Abaqus command manually when Abaqus is available**

Run:

```powershell
abaqus cae noGUI=2_FEM\scripts\generate_roi_indicator_stratified_inp.py -- 2_FEM\results\pilot_indicator_sampling\smoke_seed_1\selected_layouts.json --output-dir 2_FEM\temp\roi_indicator_stratified_inp_smoke
```

Expected output contains one line per selected layout:

```text
Saved inp:
```

Expected outputs include:

```text
2_FEM\temp\roi_indicator_stratified_inp_smoke\roi_indicator_manifest.json
```

and at least one file matching:

```text
2_FEM\temp\roi_indicator_stratified_inp_smoke\*_plate.inp
```

### Task 8: Final Verification

**Files:**

- Verify: `2_FEM/scripts/pilot_roi_indicator_sampling.py`
- Verify: `2_FEM/scripts/generate_roi_indicator_stratified_inp.py`
- Verify outputs under: `2_FEM/results/pilot_indicator_sampling`

- [ ] **Step 1: Run a larger pilot without Abaqus**

Run:

```powershell
.\.venv\Scripts\python.exe 2_FEM\scripts\pilot_roi_indicator_sampling.py --candidate-count 1000 --target-per-bin 5 --seed 20260609
```

Expected output contains:

```text
Wrote selected layouts:
```

Expected selected count in`pilot_summary.json`is at least\(5\times9=45\).

- [ ] **Step 2: Dry-run the real modeling script against the largerpilot**

Run:

```powershell
.\.venv\Scripts\python.exe 2_FEM\scripts\generate_roi_indicator_stratified_inp.py 2_FEM\results\pilot_indicator_sampling\seed_20260609_candidates_1000_target_5\selected_layouts.json --dry-run
```

Expected output contains:

```text
Dry run validated
```

- [ ] **Step 3: Inspect output JSON contract**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import json; p='2_FEM/results/pilot_indicator_sampling/seed_20260609_candidates_1000_target_5/selected_layouts.json'; data=json.load(open(p)); print(data['schema'], len(data['layouts']), sorted(data['bin_counts']))"
```

Expected output starts with:

```text
roi_indicator_stratified_layouts_v1
```

and prints\(9\)bin keys.

- [ ] **Step 4: Confirm no existing方案file was modified**

Run:

```powershell
Get-ChildItem -LiteralPath 2_FEM\scripts | Select-Object Name
```

Expected list includes the two new scripts and still includes`generate_perforated_plate_inp.py`.

## 5. Risk Controls

- Python2.7兼容性通过语法选择控制：不用f-string、dataclass、pathlib、typehint、`exist_ok=True`、`encoding`参数传给内置`open`。
- pilot和真实建模脚本都独立保存几何常量，避免导入旧方案脚本造成旧几何混入。
- Abaqus导入只放在`write_inp_with_abaqus`内部，`--dry-run`可在普通Python环境下验证布局文件和manifest。
- 分层阈值来自pilot分位数，可降低某一类样本天然稀缺导致的等待时间。
- 若某个bin在pilot中数量不足，先增大`--candidate-count`；若仍不足，再在pilot脚本中加入定向候选生成模式，最终标签仍由真实\(C\)和\(A_{\mathrm{orient}}\)计算结果决定。

## 6. Acceptance Criteria

- 实施后只新增两个脚本文件，不改旧FEM建模脚本。
- pilot脚本能生成`pilot_samples.csv`、`pilot_summary.json`和`selected_layouts.json`。
- pilot输出位于`2_FEM/results/pilot_indicator_sampling`。
- `selected_layouts.json`中每个布局都有\(24\)个孔，并满足\(d_{\min}=10\mathrm{mm}\)、\(e_{\min}=7\mathrm{mm}\)。
- 每个布局保存真实\(C\)和\(A_{\mathrm{orient}}\)。
- 真实建模脚本在`--dry-run`下可用项目虚拟环境Python完成验证。
- 真实建模脚本在Abaqus noGUI中可写出INP和manifest。
- 所有新增Python代码保持Abaqus Python2.7兼容语法。

## 7. Self-Review Checklist

- [ ] 计划覆盖pilot采样、指标量化、分层选择、结果落盘、真实建模和dry-run验证。
- [ ] 计划没有要求修改现有FEM方案文件。
- [ ] 计划明确`d_min=10mm`和`e_min=7mm`按孔心约束实现。
- [ ] 计划明确若两项表示孔边约束时需要换算。
- [ ] 计划明确Abaqus脚本使用Python2.7兼容语法。
- [ ] 计划明确pilot结果写入`2_FEM/results/pilot_indicator_sampling`。
