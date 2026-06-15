import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from cnn_surrogate.data import TARGET_COLUMNS, cnn_target_columns, coordinate_target_columns
from cnn_surrogate.devices import move_tensor_to_device, resolve_device, should_pin_memory


PREDICTION_COLUMNS = [
    "odb_name",
    "group_index",
    "instance_index",
    "split",
    "relative_equivalent_stiffness_true",
    "max_strain_concentration_factor_true",
    "relative_equivalent_stiffness_pred",
    "max_strain_concentration_factor_pred",
]


def cnn_spatial_prediction_columns():
    columns = ["odb_name", "group_index", "instance_index", "split"]
    for target_column in cnn_target_columns():
        columns.append(target_column + "_true")
        columns.append(target_column + "_pred")
    return columns


def _coordinate_prediction_columns():
    columns = ["odb_name", "group_index", "instance_index", "split"]
    for target_column in coordinate_target_columns():
        columns.append(target_column + "_true")
        columns.append(target_column + "_pred")
    return columns


def _inverse_coordinate_targets(dataset, values):
    stiffness = dataset.stiffness_scaler.inverse_transform(values[:, :1])
    local = dataset.local_strain_scaler.inverse_transform(values[:, 1:].reshape(-1, 1)).reshape(values[:, 1:].shape)
    return np.concatenate([stiffness, local], axis=1)


def _prediction_columns(prediction_suffix):
    return [
        "odb_name",
        "group_index",
        "instance_index",
        "split",
        "relative_equivalent_stiffness_true",
        "max_strain_concentration_factor_true",
        "relative_equivalent_stiffness_%s_pred" % prediction_suffix,
        "max_strain_concentration_factor_%s_pred" % prediction_suffix,
    ]


def _clamp_pixel(value, low, high):
    return max(low, min(high, value))


def _dataset_pixel_size(dataset):
    pixel_size = getattr(dataset, "pixel_size", None)
    if pixel_size is None:
        return None
    return float(pixel_size)


def _hole_pixel(row, hole_index, pixel_size, height, width):
    x = float(row["hole_%02d_x" % hole_index])
    y = float(row["hole_%02d_y" % hole_index])
    r = _clamp_pixel(int(math.floor(y / pixel_size)), 0, height - 1)
    c = _clamp_pixel(int(math.floor(x / pixel_size)), 0, width - 1)
    return r, c


def _target_map_hole_pixel(dataset, row_position, hole_index):
    target_column = cnn_target_columns()[hole_index]
    raw_value = float(dataset.frame.iloc[row_position][target_column])
    scaled_value = float(dataset.local_strain_scaler.transform(np.asarray([[raw_value]], dtype=np.float32))[0, 0])
    target_map = dataset.local_maps[row_position, 0]
    mask = dataset.local_masks[row_position, 0] > 0.5
    candidates = np.argwhere(mask & np.isclose(target_map, scaled_value, rtol=1.0e-5, atol=1.0e-6))
    if len(candidates) == 0:
        masked_difference = np.where(mask, np.abs(target_map - scaled_value), np.inf)
        candidates = np.argwhere(masked_difference == np.nanmin(masked_difference))
    if len(candidates) == 0:
        raise ValueError("could not locate encoded local target pixel for hole_%02d" % hole_index)
    return int(candidates[0, 0]), int(candidates[0, 1])


def _inverse_local_values(local_strain_scaler, values):
    flat_values = values.reshape(-1, 1)
    return local_strain_scaler.inverse_transform(flat_values).reshape(values.shape)


def predict_frame(model, dataset, batch_size, split_name=None, device="auto"):
    if hasattr(dataset, "coordinates"):
        return predict_coordinate_frame(model, dataset, batch_size, split_name=split_name, device=device)

    if hasattr(dataset, "local_maps") and hasattr(dataset, "local_masks"):
        return predict_cnn_spatial_frame(model, dataset, batch_size, split_name=split_name, device=device)

    if len(dataset) == 0:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    device = resolve_device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=should_pin_memory(device),
    )
    predictions = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = move_tensor_to_device(inputs, device)
            predictions.append(model(inputs).detach().cpu().numpy())
    scaled_predictions = np.vstack(predictions)
    true_values = dataset.frame[TARGET_COLUMNS].values.astype(np.float32)
    pred_values = dataset.target_scaler.inverse_transform(scaled_predictions)

    result = dataset.frame[["odb_name", "group_index", "instance_index"]].copy()
    if "split" in dataset.frame.columns:
        result["split"] = dataset.frame["split"].values
    else:
        result["split"] = split_name
    for target_index, target_column in enumerate(TARGET_COLUMNS):
        result[target_column + "_true"] = true_values[:, target_index]
        result[target_column + "_pred"] = pred_values[:, target_index]
    return result[PREDICTION_COLUMNS]


def predict_cnn_spatial_frame(model, dataset, batch_size, split_name=None, device="auto"):
    columns = cnn_spatial_prediction_columns()
    if len(dataset) == 0:
        return pd.DataFrame(columns=columns)

    device = resolve_device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=should_pin_memory(device),
    )
    stiffness_predictions = []
    local_map_predictions = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for images, _, _, _ in loader:
            images = move_tensor_to_device(images, device)
            stiffness_prediction, local_map_prediction = model(images)
            stiffness_predictions.append(stiffness_prediction.detach().cpu().numpy())
            local_map_predictions.append(local_map_prediction.detach().cpu().numpy())

    scaled_stiffness_predictions = np.vstack(stiffness_predictions)
    scaled_local_map_predictions = np.concatenate(local_map_predictions, axis=0)
    stiffness_true = dataset.stiffness_scaler.inverse_transform(
        np.asarray(dataset.stiffness_targets, dtype=np.float32)
    )[:, 0]
    stiffness_pred = dataset.stiffness_scaler.inverse_transform(scaled_stiffness_predictions)[:, 0]

    local_target_columns = cnn_target_columns()[1:]
    sample_count = len(dataset)
    local_true_scaled = np.zeros((sample_count, len(local_target_columns)), dtype=np.float32)
    local_pred_scaled = np.zeros((sample_count, len(local_target_columns)), dtype=np.float32)
    height = int(scaled_local_map_predictions.shape[-2])
    width = int(scaled_local_map_predictions.shape[-1])
    pixel_size = _dataset_pixel_size(dataset)
    for row_position, (_, row) in enumerate(dataset.frame.iterrows()):
        for column_position, hole_index in enumerate(range(1, len(local_target_columns) + 1)):
            if pixel_size is None:
                r, c = _target_map_hole_pixel(dataset, row_position, hole_index)
            else:
                r, c = _hole_pixel(row, hole_index, pixel_size, height, width)
            local_true_scaled[row_position, column_position] = dataset.local_maps[row_position, 0, r, c]
            local_pred_scaled[row_position, column_position] = scaled_local_map_predictions[row_position, 0, r, c]
    local_true = _inverse_local_values(dataset.local_strain_scaler, local_true_scaled)
    local_pred = _inverse_local_values(dataset.local_strain_scaler, local_pred_scaled)

    result = dataset.frame[["odb_name", "group_index", "instance_index"]].copy()
    if split_name is not None:
        result["split"] = split_name
    elif "split" in dataset.frame.columns:
        result["split"] = dataset.frame["split"].values
    else:
        result["split"] = None
    result["relative_equivalent_stiffness_true"] = stiffness_true
    result["relative_equivalent_stiffness_pred"] = stiffness_pred
    for column_position, target_column in enumerate(local_target_columns):
        result[target_column + "_true"] = local_true[:, column_position]
        result[target_column + "_pred"] = local_pred[:, column_position]
    return result[columns]


def predict_coordinate_frame(model, dataset, batch_size, split_name=None, device="cpu"):
    columns = _coordinate_prediction_columns()
    if len(dataset) == 0:
        return pd.DataFrame(columns=columns)

    device = resolve_device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=should_pin_memory(device),
    )
    predictions = []
    targets = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for coordinates, stiffness_target, local_targets in loader:
            coordinates = move_tensor_to_device(coordinates, device)
            outputs = model(coordinates)
            predictions.append(outputs.detach().cpu().numpy())
            targets.append(torch.cat([stiffness_target, local_targets], dim=1).detach().cpu().numpy())

    prediction_values = _inverse_coordinate_targets(dataset, np.vstack(predictions))
    target_values = _inverse_coordinate_targets(dataset, np.vstack(targets))
    result = dataset.frame[["odb_name", "group_index", "instance_index"]].copy()
    if split_name is not None:
        result["split"] = split_name
    elif "split" in dataset.frame.columns:
        result["split"] = dataset.frame["split"].values
    else:
        result["split"] = None
    for target_index, target_column in enumerate(coordinate_target_columns()):
        result[target_column + "_true"] = target_values[:, target_index]
        result[target_column + "_pred"] = prediction_values[:, target_index]
    return result[columns]


def _distillation_predict_frame(model, dataset, batch_size, prediction_suffix, uses_local_features, split_name=None, device="auto"):
    columns = _prediction_columns(prediction_suffix)
    if len(dataset) == 0:
        return pd.DataFrame(columns=columns)

    device = resolve_device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=should_pin_memory(device),
    )
    predictions = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = move_tensor_to_device(batch[0], device)
            if uses_local_features:
                local_features = move_tensor_to_device(batch[1], device)
                batch_prediction = model(images, local_features)
            else:
                batch_prediction = model(images)
            predictions.append(batch_prediction.detach().cpu().numpy())
    scaled_predictions = np.vstack(predictions)
    true_values = dataset.frame[TARGET_COLUMNS].values.astype(np.float32)
    pred_values = dataset.target_scaler.inverse_transform(scaled_predictions)

    result = dataset.frame[["odb_name", "group_index", "instance_index"]].copy()
    if "split" in dataset.frame.columns:
        result["split"] = dataset.frame["split"].values
    else:
        result["split"] = split_name
    for target_index, target_column in enumerate(TARGET_COLUMNS):
        result[target_column + "_true"] = true_values[:, target_index]
        result[target_column + "_%s_pred" % prediction_suffix] = pred_values[:, target_index]
    return result[columns]


def predict_teacher_frame(model, dataset, batch_size, split_name=None, device="auto"):
    return _distillation_predict_frame(
        model,
        dataset,
        batch_size,
        prediction_suffix="teacher",
        uses_local_features=True,
        split_name=split_name,
        device=device,
    )


def predict_student_frame(model, dataset, batch_size, split_name=None, device="auto"):
    return _distillation_predict_frame(
        model,
        dataset,
        batch_size,
        prediction_suffix="student",
        uses_local_features=False,
        split_name=split_name,
        device=device,
    )


def _metric_block(frame, target_column, prediction_suffix=None):
    if len(frame) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    true_values = frame[target_column + "_true"].values
    if prediction_suffix is None:
        pred_column = target_column + "_pred"
    else:
        pred_column = target_column + "_%s_pred" % prediction_suffix
    pred_values = frame[pred_column].values
    rmse = math.sqrt(mean_squared_error(true_values, pred_values))
    r2 = None
    if len(frame) >= 2:
        r2 = float(r2_score(true_values, pred_values))
    return {
        "mae": float(mean_absolute_error(true_values, pred_values)),
        "rmse": float(rmse),
        "r2": r2,
    }


def _coordinate_metric_values(true_values, pred_values):
    if len(true_values) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    rmse = math.sqrt(mean_squared_error(true_values, pred_values))
    r2 = None
    if len(true_values) >= 2:
        r2 = float(r2_score(true_values, pred_values))
    return {
        "mae": float(mean_absolute_error(true_values, pred_values)),
        "rmse": float(rmse),
        "r2": r2,
    }


def _coordinate_metric_block(frame, target_column):
    if len(frame) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    return _coordinate_metric_values(
        frame[target_column + "_true"].values,
        frame[target_column + "_pred"].values,
    )


def _local_strain_summary(frame):
    if len(frame) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    true_values = []
    pred_values = []
    for target_column in coordinate_target_columns()[1:]:
        true_values.append(frame[target_column + "_true"].values)
        pred_values.append(frame[target_column + "_pred"].values)
    return _coordinate_metric_values(np.concatenate(true_values), np.concatenate(pred_values))


def _cnn_local_strain_summary(frame):
    if len(frame) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    true_values = []
    pred_values = []
    for target_column in cnn_target_columns()[1:]:
        true_values.append(frame[target_column + "_true"].values)
        pred_values.append(frame[target_column + "_pred"].values)
    return _coordinate_metric_values(np.concatenate(true_values), np.concatenate(pred_values))


def _cnn_local_strain_errors(frame):
    if len(frame) == 0:
        return np.asarray([], dtype=np.float64)
    true_values = []
    pred_values = []
    for target_column in cnn_target_columns()[1:]:
        true_values.append(frame[target_column + "_true"].values.astype(float))
        pred_values.append(frame[target_column + "_pred"].values.astype(float))
    return np.concatenate(pred_values) - np.concatenate(true_values)


def _cnn_local_strain_error_quantiles(frame):
    errors = _cnn_local_strain_errors(frame)
    if len(errors) == 0:
        return {
            "q00": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q100": None,
        }
    quantiles = np.percentile(errors, [0, 25, 50, 75, 100])
    return {
        "q00": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q100": float(quantiles[4]),
    }


def compute_coordinate_metrics(predictions):
    metrics = {}
    local_target_columns = coordinate_target_columns()[1:]
    for split in ["train", "val", "test"]:
        split_frame = predictions[predictions["split"] == split]
        local_metrics = {}
        for target_column in local_target_columns:
            local_metrics[target_column] = _coordinate_metric_block(split_frame, target_column)
        metrics[split] = {
            "count": int(len(split_frame)),
            "targets": {
                "relative_equivalent_stiffness": _coordinate_metric_block(
                    split_frame,
                    "relative_equivalent_stiffness",
                ),
            },
            "local_strain_summary": _local_strain_summary(split_frame),
            "local_strain_by_hole": local_metrics,
        }
    return metrics


def compute_cnn_spatial_metrics(predictions):
    metrics = {}
    for split in ["train", "val", "test"]:
        split_frame = predictions[predictions["split"] == split]
        metrics[split] = {
            "count": int(len(split_frame)),
            "targets": {
                "relative_equivalent_stiffness": _metric_block(
                    split_frame,
                    "relative_equivalent_stiffness",
                ),
            },
            "local_strain_summary": _cnn_local_strain_summary(split_frame),
            "local_strain_error_quantiles": _cnn_local_strain_error_quantiles(split_frame),
        }
    return metrics


def compute_metrics_for_suffix(predictions_frame, prediction_suffix=None):
    metrics = {}
    for split in ["train", "val", "test"]:
        split_frame = predictions_frame[predictions_frame["split"] == split]
        metrics[split] = {"count": int(len(split_frame)), "targets": {}}
        for target_column in TARGET_COLUMNS:
            metrics[split]["targets"][target_column] = _metric_block(
                split_frame,
                target_column,
                prediction_suffix=prediction_suffix,
            )
    return metrics


def compute_metrics(predictions_frame):
    if all(
        column + "_true" in predictions_frame.columns and column + "_pred" in predictions_frame.columns
        for column in cnn_target_columns()[1:]
    ):
        return compute_cnn_spatial_metrics(predictions_frame)
    return compute_metrics_for_suffix(predictions_frame)
