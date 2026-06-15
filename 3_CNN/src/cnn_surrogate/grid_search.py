import csv
import hashlib
import itertools
import json
import math
import os
import traceback
from dataclasses import asdict, replace


STIFFNESS_TARGET = "relative_equivalent_stiffness"
STRAIN_TARGET = "max_strain_concentration_factor"

FINAL_OUTPUTS = {
    "cnn": {
        "output_dir": os.path.join("3_CNN", "results", "cnn_surrogate"),
        "figure_dir": os.path.join("3_CNN", "figures", "cnn_surrogate"),
    },
    "distilled": {
        "output_dir": os.path.join("3_CNN", "results", "distilled_surrogate"),
        "figure_dir": os.path.join("3_CNN", "figures", "distilled_surrogate"),
    },
    "coordinate": {
        "output_dir": os.path.join("3_CNN", "results", "coordinate_surrogate"),
        "figure_dir": os.path.join("3_CNN", "figures", "coordinate_surrogate"),
    },
}


def ensure_directory(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)
    return path


def grid_parameter_combinations(param_grid):
    keys = list(param_grid.keys())
    combinations = []
    for values in itertools.product(*[param_grid[key] for key in keys]):
        combinations.append(dict(zip(keys, values)))
    return combinations


def build_trial_id(search_id, index):
    return "%s_%04d" % (search_id, index)


def parameter_hash(params):
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def average_rmse(stiffness_rmse, strain_rmse):
    if stiffness_rmse is None or strain_rmse is None:
        return None
    return (float(stiffness_rmse) + float(strain_rmse)) / 2.0


def _target_rmse(metrics, split_name, target_name):
    try:
        value = metrics[split_name]["targets"][target_name]["rmse"]
    except KeyError:
        return None
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _local_strain_rmse(metrics, split_name):
    try:
        value = metrics[split_name]["local_strain_summary"]["rmse"]
    except KeyError:
        return None
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def extract_scores(model_name, result):
    if model_name == "distilled":
        metrics = result["student_metrics"]
        stiffness_rmse = _target_rmse(metrics, "val", STIFFNESS_TARGET)
        strain_rmse = _target_rmse(metrics, "val", STRAIN_TARGET)
        return {
            "stiffness_rmse": stiffness_rmse,
            "strain_rmse": strain_rmse,
            "average_rmse": average_rmse(stiffness_rmse, strain_rmse),
        }
    else:
        metrics = result["metrics"]
        stiffness_rmse = _target_rmse(metrics, "val", STIFFNESS_TARGET)
        local_strain_rmse = _local_strain_rmse(metrics, "val")
        return {
            "stiffness_rmse": stiffness_rmse,
            "local_strain_rmse": local_strain_rmse,
            "strain_rmse": local_strain_rmse,
            "average_rmse": average_rmse(stiffness_rmse, local_strain_rmse),
        }


def _score_is_number(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _best_record(records, score_name):
    candidates = [
        record for record in records
        if record.get("status") == "completed"
        and _score_is_number(record.get("scores", {}).get(score_name))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda record: record["scores"][score_name])


def select_best_records(records):
    return {
        "best_stiffness": _best_record(records, "stiffness_rmse"),
        "best_local_strain": _best_record(records, "local_strain_rmse"),
        "best_strain": _best_record(records, "strain_rmse"),
        "best_average": _best_record(records, "average_rmse"),
    }


def _read_jsonl(path):
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _write_results_csv(path, records):
    columns = [
        "trial_id",
        "param_hash",
        "status",
        "stiffness_rmse",
        "local_strain_rmse",
        "strain_rmse",
        "average_rmse",
        "output_dir",
        "figure_dir",
        "params_json",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            scores = record.get("scores", {})
            writer.writerow({
                "trial_id": record.get("trial_id"),
                "param_hash": record.get("param_hash"),
                "status": record.get("status"),
                "stiffness_rmse": scores.get("stiffness_rmse"),
                "local_strain_rmse": scores.get("local_strain_rmse"),
                "strain_rmse": scores.get("strain_rmse"),
                "average_rmse": scores.get("average_rmse"),
                "output_dir": record.get("output_dir"),
                "figure_dir": record.get("figure_dir"),
                "params_json": json.dumps(record.get("params", {}), sort_keys=True),
                "error": record.get("error"),
            })
    return path


def _existing_completed_by_hash(records):
    completed = {}
    for record in records:
        if record.get("status") == "completed":
            completed[record.get("param_hash")] = record
    return completed


def _trial_paths(search_config, trial_id, results_root, figures_root, temp_root):
    trial_name = "%s_%s" % (search_config.RESULT_PREFIX, trial_id)
    return {
        "output_dir": os.path.join(results_root, trial_name),
        "figure_dir": os.path.join(figures_root, trial_name),
        "temp_dir": os.path.join(temp_root, trial_name),
    }


def build_trial_config(
    model_name,
    base_config,
    params,
    paths,
    device,
    train_test_split,
    early_stopping_patience,
    save_model,
    progress_description,
):
    updates = dict(params)
    if model_name == "distilled" and "student_epochs" in updates:
        updates["epochs"] = updates["student_epochs"]
    updates.update({
        "output_dir": paths["output_dir"],
        "figure_dir": paths["figure_dir"],
        "temp_dir": paths["temp_dir"],
        "train_test_split": train_test_split,
        "early_stopping_patience": early_stopping_patience,
        "device": device,
        "show_progress": True,
        "progress_description": progress_description,
        "save_model": save_model,
    })
    if hasattr(base_config, "checkpoint_path"):
        updates["checkpoint_path"] = os.path.join(paths["output_dir"], "checkpoint.pt")
    return replace(base_config, **updates)


def _record_progress_files(search_dir, records):
    _write_results_csv(os.path.join(search_dir, "search_results.csv"), records)
    best_results = select_best_records(records)
    _write_json(os.path.join(search_dir, "best_results.json"), best_results)
    return best_results


def _completed_record(search_config, model_name, trial_id, params, paths, result):
    scores = extract_scores(model_name, result)
    record = {
        "search_id": search_config.SEARCH_ID,
        "trial_id": trial_id,
        "param_hash": parameter_hash(params),
        "status": "completed",
        "params": params,
        "scores": scores,
        "output_dir": paths["output_dir"],
        "figure_dir": paths["figure_dir"],
    }
    return record


def _failed_record(search_config, trial_id, params, paths, error):
    return {
        "search_id": search_config.SEARCH_ID,
        "trial_id": trial_id,
        "param_hash": parameter_hash(params),
        "status": "failed",
        "params": params,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "output_dir": paths["output_dir"],
        "figure_dir": paths["figure_dir"],
    }


def _write_trial_files(paths, config, record):
    ensure_directory(paths["output_dir"])
    _write_json(os.path.join(paths["output_dir"], "trial_config.json"), asdict(config))
    _write_json(os.path.join(paths["output_dir"], "trial_result.json"), record)


def _rerun_best_average(
    model_name,
    search_config,
    best_average,
    base_config_builder,
    training_runner,
    device,
    train_test_split,
    early_stopping_patience,
):
    final_paths = dict(FINAL_OUTPUTS[model_name])
    final_paths["temp_dir"] = os.path.join("3_CNN", "temp", "%s_best_average" % search_config.RESULT_PREFIX)
    config = build_trial_config(
        model_name=model_name,
        base_config=base_config_builder(),
        params=best_average["params"],
        paths=final_paths,
        device=device,
        train_test_split=train_test_split,
        early_stopping_patience=early_stopping_patience,
        save_model=True,
        progress_description="Training %s best_average" % model_name,
    )
    print("saving best_average %s to %s" % (best_average["trial_id"], final_paths["output_dir"]), flush=True)
    training_runner(config)
    return config


def run_grid_search(
    model_name,
    search_config,
    base_config_builder,
    training_runner,
    results_root,
    figures_root,
    temp_root,
    device,
    train_test_split,
    early_stopping_patience,
    save_best=False,
):
    search_dir = os.path.join(results_root, "%s_%s" % (search_config.RESULT_PREFIX, search_config.SEARCH_ID))
    ensure_directory(search_dir)
    jsonl_path = os.path.join(search_dir, "search_results.jsonl")
    records = _read_jsonl(jsonl_path)
    completed = _existing_completed_by_hash(records)
    combinations = grid_parameter_combinations(search_config.PARAM_GRID)
    total = len(combinations)

    for index, params in enumerate(combinations, start=1):
        trial_id = build_trial_id(search_config.SEARCH_ID, index)
        current_hash = parameter_hash(params)
        if current_hash in completed:
            print("[%d/%d] skipped %s" % (index, total, trial_id), flush=True)
            continue

        paths = _trial_paths(search_config, trial_id, results_root, figures_root, temp_root)
        print("[%d/%d] running %s" % (index, total, trial_id), flush=True)
        try:
            config = build_trial_config(
                model_name=model_name,
                base_config=base_config_builder(),
                params=params,
                paths=paths,
                device=device,
                train_test_split=train_test_split,
                early_stopping_patience=early_stopping_patience,
                save_model=False,
                progress_description="Training %s %s" % (model_name, trial_id),
            )
            result = training_runner(config)
            record = _completed_record(search_config, model_name, trial_id, params, paths, result)
            print("[%d/%d] finished %s" % (index, total, trial_id), flush=True)
        except Exception as error:
            record = _failed_record(search_config, trial_id, params, paths, error)
            print("[%d/%d] failed %s: %s" % (index, total, trial_id, error), flush=True)

        _append_jsonl(jsonl_path, record)
        records.append(record)
        if record["status"] == "completed":
            completed[current_hash] = record
        _write_trial_files(paths, config if record["status"] == "completed" else base_config_builder(), record)
        best_results = _record_progress_files(search_dir, records)

    best_results = _record_progress_files(search_dir, records)
    if save_best:
        best_average = best_results.get("best_average")
        if best_average is None:
            raise RuntimeError("cannot save best model because no completed best_average result exists")
        _rerun_best_average(
            model_name=model_name,
            search_config=search_config,
            best_average=best_average,
            base_config_builder=base_config_builder,
            training_runner=training_runner,
            device=device,
            train_test_split=train_test_split,
            early_stopping_patience=early_stopping_patience,
        )

    return {
        "search_dir": search_dir,
        "records": records,
        "best_results": best_results,
    }
