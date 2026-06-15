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

DEFAULT_CANDIDATE_COUNT = 100000
DEFAULT_TARGET_PER_BIN = 200
DEFAULT_SEED = 20260609
MAX_ATTEMPTS_PER_HOLE = 50000
MAX_LAYOUT_RESTARTS = 2000
CLUSTER_LABELS = ("low", "medium", "high")
ORIENTATION_LABELS = ("x", "none", "y")

RESULTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "results", "pilot_indicator_sampling")
)


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
    with io.open(path, "r", encoding="utf-8-sig") as json_file:
        return json.load(json_file)


def seed_for_candidate(base_seed, candidate_index, restart_index):
    text = "%s:%s:%s" % (base_seed, candidate_index, restart_index)
    digest = hashlib.md5(text.encode("ascii")).hexdigest()
    return int(digest[:12], 16)


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


def expected_bin_keys():
    keys = []
    for cluster_label in CLUSTER_LABELS:
        for orientation_label in ORIENTATION_LABELS:
            keys.append("%s_%s" % (cluster_label, orientation_label))
    return keys


def validate_balanced_counts(counts, target_per_bin):
    underfilled = []
    for key in expected_bin_keys():
        if counts.get(key, 0) < target_per_bin:
            underfilled.append("%s=%s" % (key, counts.get(key, 0)))
    if underfilled:
        raise ValueError(
            "underfilled sampling bins for target %d: %s"
            % (target_per_bin, ", ".join(underfilled))
        )
    return None


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
    for cluster_label in CLUSTER_LABELS:
        for orientation_label in ORIENTATION_LABELS:
            counts["%s_%s" % (cluster_label, orientation_label)] = 0

    for record in records:
        key = record["bin"]
        if counts[key] >= target_per_bin:
            continue
        selected.append(record)
        counts[key] += 1
        if min(counts.values()) >= target_per_bin:
            break

    validate_balanced_counts(counts, target_per_bin)
    return selected, counts


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
    minimum_candidate_count = len(expected_bin_keys()) * args.target_per_bin
    if args.candidate_count < minimum_candidate_count:
        raise ValueError(
            "candidate-count must be at least %d for target-per-bin %d"
            % (minimum_candidate_count, args.target_per_bin)
        )

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
