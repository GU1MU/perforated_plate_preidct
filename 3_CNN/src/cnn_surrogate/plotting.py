import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnn_surrogate.data import TARGET_COLUMNS, cnn_target_columns, coordinate_target_columns


def _ensure_directory(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)
    return path


def plot_loss_curve(history, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "loss_curve.png")
    history_frame = pd.DataFrame(history)
    plt.figure(figsize=(6, 4))
    if len(history_frame) > 0:
        plt.plot(history_frame["epoch"], history_frame["train_loss"], label="train")
        if "val_loss" in history_frame and history_frame["val_loss"].notna().any():
            plt.plot(history_frame["epoch"], history_frame["val_loss"], label="val")
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Weighted MSE")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_pred_vs_true(predictions, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "pred_vs_true.png")
    plt.figure(figsize=(8, 4))
    for index, target_column in enumerate(TARGET_COLUMNS):
        axis = plt.subplot(1, 2, index + 1)
        true_column = target_column + "_true"
        pred_column = target_column + "_pred"
        if len(predictions) > 0:
            axis.scatter(predictions[true_column], predictions[pred_column], s=18, alpha=0.75)
            values = np.concatenate([
                predictions[true_column].values.astype(float),
                predictions[pred_column].values.astype(float),
            ])
            lower = float(np.nanmin(values))
            upper = float(np.nanmax(values))
            if lower == upper:
                lower -= 0.5
                upper += 0.5
            axis.plot([lower, upper], [lower, upper], color="black", linewidth=1.0)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
        axis.set_xlabel("FEM")
        axis.set_ylabel("CNN")
        axis.set_title(target_column)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_identity_scatter(true_values, pred_values, xlabel, ylabel):
    if len(true_values) > 0:
        plt.scatter(true_values, pred_values, s=18, alpha=0.75)
        values = np.concatenate([
            np.asarray(true_values, dtype=float),
            np.asarray(pred_values, dtype=float),
        ])
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        plt.plot([lower, upper], [lower, upper], color="black", linewidth=1.0)
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_coordinate_stiffness_pred_vs_true(predictions, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "stiffness_pred_vs_true.png")
    true_column = "relative_equivalent_stiffness_true"
    pred_column = "relative_equivalent_stiffness_pred"
    plt.figure(figsize=(5, 4))
    if len(predictions) > 0:
        _plot_identity_scatter(
            predictions[true_column].values,
            predictions[pred_column].values,
            "FEM",
            "CoordinateSurrogate",
        )
    else:
        plt.xlabel("FEM")
        plt.ylabel("CoordinateSurrogate")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_cnn_stiffness_pred_vs_true(predictions, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "stiffness_pred_vs_true.png")
    true_column = "relative_equivalent_stiffness_true"
    pred_column = "relative_equivalent_stiffness_pred"
    plt.figure(figsize=(5, 4))
    if len(predictions) > 0:
        _plot_identity_scatter(
            predictions[true_column].values,
            predictions[pred_column].values,
            "FEM",
            "CNN",
        )
    else:
        plt.xlabel("FEM")
        plt.ylabel("CNN")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_coordinate_local_strain_pred_vs_true(predictions, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "local_strain_pred_vs_true.png")
    plt.figure(figsize=(5, 4))
    if len(predictions) > 0:
        true_values = []
        pred_values = []
        for target_column in coordinate_target_columns()[1:]:
            true_values.append(predictions[target_column + "_true"].values.astype(float))
            pred_values.append(predictions[target_column + "_pred"].values.astype(float))
        _plot_identity_scatter(
            np.concatenate(true_values),
            np.concatenate(pred_values),
            "FEM",
            "CoordinateSurrogate",
        )
    else:
        plt.xlabel("FEM")
        plt.ylabel("CoordinateSurrogate")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_cnn_local_strain_pred_vs_true(predictions, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "local_strain_pred_vs_true.png")
    plt.figure(figsize=(5, 4))
    if len(predictions) > 0:
        true_values = []
        pred_values = []
        for target_column in cnn_target_columns()[1:]:
            true_values.append(predictions[target_column + "_true"].values.astype(float))
            pred_values.append(predictions[target_column + "_pred"].values.astype(float))
        _plot_identity_scatter(
            np.concatenate(true_values),
            np.concatenate(pred_values),
            "FEM",
            "CNN",
        )
    else:
        plt.xlabel("FEM")
        plt.ylabel("CNN")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_coordinate_local_strain_rmse_by_hole(metrics, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "local_strain_rmse_by_hole.png")
    split_name = "val"
    if metrics.get("val", {}).get("count", 0) <= 0:
        split_name = "train"
    by_hole = metrics.get(split_name, {}).get("local_strain_by_hole", {})
    columns = coordinate_target_columns()[1:]
    rmse_values = [by_hole.get(column, {}).get("rmse") for column in columns]
    rmse_values = [np.nan if value is None else value for value in rmse_values]

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(1, len(columns) + 1), rmse_values)
    plt.xlabel("Hole index")
    plt.ylabel("RMSE")
    plt.xticks(np.arange(1, len(columns) + 1, 2))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_cnn_local_strain_error_distribution(predictions, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "local_strain_error_distribution.png")
    columns = cnn_target_columns()[1:]
    errors = []
    if len(predictions) > 0:
        for target_column in columns:
            true_values = predictions[target_column + "_true"].values.astype(float)
            pred_values = predictions[target_column + "_pred"].values.astype(float)
            errors.append(pred_values - true_values)

    plt.figure(figsize=(8, 4))
    if errors:
        error_values = np.concatenate(errors)
        plt.hist(error_values, bins=min(30, max(5, int(np.sqrt(len(error_values))))), alpha=0.8)
        plt.axvline(0.0, color="black", linewidth=1.0)
    plt.xlabel("CNN local strain prediction error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_teacher_vs_student(teacher_predictions, student_predictions, figure_dir):
    _ensure_directory(figure_dir)
    path = os.path.join(figure_dir, "teacher_vs_student.png")
    key_columns = ["odb_name", "group_index", "instance_index", "split"]
    merged = pd.merge(
        teacher_predictions,
        student_predictions,
        on=key_columns + [target_column + "_true" for target_column in TARGET_COLUMNS],
        how="inner",
    )

    plt.figure(figsize=(8, 4))
    for index, target_column in enumerate(TARGET_COLUMNS):
        axis = plt.subplot(1, 2, index + 1)
        teacher_column = target_column + "_teacher_pred"
        student_column = target_column + "_student_pred"
        if len(merged) > 0:
            axis.scatter(merged[teacher_column], merged[student_column], s=18, alpha=0.75)
            values = np.concatenate([
                merged[teacher_column].values.astype(float),
                merged[student_column].values.astype(float),
            ])
            lower = float(np.nanmin(values))
            upper = float(np.nanmax(values))
            if lower == upper:
                lower -= 0.5
                upper += 0.5
            axis.plot([lower, upper], [lower, upper], color="black", linewidth=1.0)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
        axis.set_xlabel("Teacher")
        axis.set_ylabel("Student")
        axis.set_title(target_column)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
