import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

from cnn_surrogate.data import (
    CnnSpatialLayoutDataset,
    CoordinateLayoutDataset,
    DistillationLayoutDataset,
    TARGET_COLUMNS,
    assign_splits,
    filter_valid_rows,
    load_coordinate_dataset_table,
    local_feature_columns,
    required_cnn_columns,
    required_distillation_columns,
    validate_distillation_columns,
)
from cnn_surrogate.evaluation import (
    compute_cnn_spatial_metrics,
    compute_coordinate_metrics,
    compute_metrics,
    compute_metrics_for_suffix,
    predict_cnn_spatial_frame,
    predict_coordinate_frame,
    predict_student_frame,
    predict_teacher_frame,
)
from cnn_surrogate.io import (
    ensure_directory,
    save_coordinate_model_package,
    save_cnn_spatial_model_package,
    save_distillation_package,
    write_metrics,
    write_predictions,
    write_split_manifest,
    write_train_history,
)
from cnn_surrogate.plotting import (
    plot_cnn_local_strain_error_distribution,
    plot_cnn_local_strain_pred_vs_true,
    plot_cnn_stiffness_pred_vs_true,
    plot_coordinate_local_strain_pred_vs_true,
    plot_coordinate_local_strain_rmse_by_hole,
    plot_coordinate_stiffness_pred_vs_true,
    plot_loss_curve,
    plot_pred_vs_true,
    plot_teacher_vs_student,
)
from cnn_surrogate.training import train_coordinate_model, train_model, train_student_model, train_teacher_model


def _attach_cnn_spatial_metadata(dataset, config):
    dataset.pixel_size = config.pixel_size
    dataset.image_height = config.image_height
    dataset.image_width = config.image_width
    return dataset


def _build_cnn_spatial_dataset_for_split(frame, split_name, stiffness_scaler, local_strain_scaler, config):
    split_frame = frame[frame["split"] == split_name].copy()
    dataset = CnnSpatialLayoutDataset(
        split_frame,
        image_height=config.image_height,
        image_width=config.image_width,
        pixel_size=config.pixel_size,
        stiffness_scaler=stiffness_scaler,
        local_strain_scaler=local_strain_scaler,
        fit_scalers=False,
    )
    return _attach_cnn_spatial_metadata(dataset, config)


def _build_coordinate_dataset_for_split(frame, split_name, stiffness_scaler, local_strain_scaler, config):
    split_frame = frame[frame["split"] == split_name].copy()
    return CoordinateLayoutDataset(
        split_frame,
        domain_width=config.coordinate_domain_width,
        domain_height=config.coordinate_domain_height,
        stiffness_scaler=stiffness_scaler,
        local_strain_scaler=local_strain_scaler,
        fit_scalers=False,
    )


def _build_distillation_dataset_for_split(frame, split_name, target_scaler, local_feature_scaler, config):
    split_frame = frame[frame["split"] == split_name].copy()
    return DistillationLayoutDataset(
        split_frame,
        image_height=config.image_height,
        image_width=config.image_width,
        pixel_size=config.pixel_size,
        target_scaler=target_scaler,
        local_feature_scaler=local_feature_scaler,
        fit_scaler=False,
        fit_local_feature_scaler=False,
    )


def _combine_prediction_frames(prediction_frames):
    non_empty = [item for item in prediction_frames if len(item) > 0]
    if non_empty:
        return pd.concat(non_empty, axis=0, ignore_index=True)
    return prediction_frames[0].copy()


def _load_cnn_spatial_dataset_table(path):
    frame = pd.read_csv(path)
    missing = [column for column in required_cnn_columns() if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(missing))
    frame = filter_valid_rows(frame)
    frame = frame.dropna(subset=required_cnn_columns())
    frame["group_index"] = frame["group_index"].astype(int)
    frame["instance_index"] = frame["instance_index"].astype(int)
    return frame.reset_index(drop=True)


def _load_distillation_dataset_table(path):
    frame = pd.read_csv(path)
    validate_distillation_columns(frame)
    frame = filter_valid_rows(frame)
    frame = frame.dropna(subset=required_distillation_columns())
    frame["group_index"] = frame["group_index"].astype(int)
    frame["instance_index"] = frame["instance_index"].astype(int)
    return frame.reset_index(drop=True)


def _write_named_history(history, output_dir, filename):
    ensure_directory(output_dir)
    path = "%s/%s" % (output_dir, filename)
    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss"]).to_csv(path, index=False)
    return path


def _write_named_predictions(predictions, output_dir, filename):
    ensure_directory(output_dir)
    path = "%s/%s" % (output_dir, filename)
    predictions.to_csv(path, index=False)
    return path


def _write_named_metrics(metrics, output_dir, filename):
    ensure_directory(output_dir)
    path = "%s/%s" % (output_dir, filename)
    with open(path, "w", encoding="utf-8") as metrics_file:
        import json
        json.dump(metrics, metrics_file, indent=2)
    return path


def _plot_distillation_pred_vs_true(predictions, figure_dir, prediction_suffix, filename):
    plot_frame = predictions.copy()
    for target_column in TARGET_COLUMNS:
        plot_frame[target_column + "_pred"] = plot_frame[target_column + "_%s_pred" % prediction_suffix]
    temp_figure_dir = "%s/%s_pred_vs_true_tmp" % (figure_dir, prediction_suffix)
    ensure_directory(temp_figure_dir)
    temp_path = plot_pred_vs_true(plot_frame, temp_figure_dir)
    final_path = "%s/%s" % (figure_dir, filename)
    import os
    os.replace(temp_path, final_path)
    try:
        os.rmdir(temp_figure_dir)
    except OSError:
        pass
    return final_path


def run_baseline_training(config):
    ensure_directory(config.output_dir)
    if config.save_figures:
        ensure_directory(config.figure_dir)
    ensure_directory(config.temp_dir)
    if config.warm_start and not config.checkpoint_path:
        config.checkpoint_path = os.path.join(config.output_dir, "checkpoint.pt")

    frame = _load_cnn_spatial_dataset_table(config.data_csv)
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
    train_dataset = CnnSpatialLayoutDataset(
        train_frame,
        image_height=config.image_height,
        image_width=config.image_width,
        pixel_size=config.pixel_size,
        stiffness_scaler=stiffness_scaler,
        local_strain_scaler=local_strain_scaler,
        fit_scalers=True,
    )
    train_dataset = _attach_cnn_spatial_metadata(train_dataset, config)
    val_dataset = _build_cnn_spatial_dataset_for_split(
        frame,
        "val",
        train_dataset.stiffness_scaler,
        train_dataset.local_strain_scaler,
        config,
    )
    test_dataset = _build_cnn_spatial_dataset_for_split(
        frame,
        "test",
        train_dataset.stiffness_scaler,
        train_dataset.local_strain_scaler,
        config,
    )

    model, history = train_model(train_dataset, val_dataset, config)
    write_train_history(history, config.output_dir)

    prediction_frames = [
        predict_cnn_spatial_frame(model, train_dataset, config.batch_size, split_name="train", device=config.device),
        predict_cnn_spatial_frame(model, val_dataset, config.batch_size, split_name="val", device=config.device),
        predict_cnn_spatial_frame(model, test_dataset, config.batch_size, split_name="test", device=config.device),
    ]
    predictions = _combine_prediction_frames(prediction_frames)
    metrics = compute_cnn_spatial_metrics(predictions)

    write_predictions(predictions, config.output_dir)
    write_metrics(metrics, config.output_dir)
    if config.save_figures:
        plot_loss_curve(history, config.figure_dir)
        plot_cnn_stiffness_pred_vs_true(predictions, config.figure_dir)
        plot_cnn_local_strain_pred_vs_true(predictions, config.figure_dir)
        plot_cnn_local_strain_error_distribution(predictions, config.figure_dir)
    save_cnn_spatial_model_package(
        model,
        train_dataset.stiffness_scaler,
        train_dataset.local_strain_scaler,
        config.output_dir,
        config,
    )

    return {"history": history, "metrics": metrics, "predictions": predictions}


def run_coordinate_training(config):
    ensure_directory(config.output_dir)
    if config.save_figures:
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
    if config.save_figures:
        plot_loss_curve(history, config.figure_dir)
        plot_coordinate_stiffness_pred_vs_true(predictions, config.figure_dir)
        plot_coordinate_local_strain_pred_vs_true(predictions, config.figure_dir)
        plot_coordinate_local_strain_rmse_by_hole(metrics, config.figure_dir)
    save_coordinate_model_package(
        model,
        train_dataset.stiffness_scaler,
        train_dataset.local_strain_scaler,
        config.output_dir,
        config,
    )

    return {"history": history, "metrics": metrics, "predictions": predictions}


def run_distillation_training(config):
    ensure_directory(config.output_dir)
    if config.save_figures:
        ensure_directory(config.figure_dir)
    ensure_directory(config.temp_dir)

    frame = _load_distillation_dataset_table(config.data_csv)
    frame = assign_splits(
        frame,
        train_count=config.train_test_split,
        shuffle=config.split_shuffle,
        random_seed=config.random_seed,
    )
    write_split_manifest(frame, config.output_dir)

    train_frame = frame[frame["split"] == "train"].copy()
    target_scaler = StandardScaler()
    local_feature_scaler = StandardScaler()
    train_dataset = DistillationLayoutDataset(
        train_frame,
        image_height=config.image_height,
        image_width=config.image_width,
        pixel_size=config.pixel_size,
        target_scaler=target_scaler,
        local_feature_scaler=local_feature_scaler,
        fit_scaler=True,
        fit_local_feature_scaler=True,
    )
    val_dataset = _build_distillation_dataset_for_split(
        frame,
        "val",
        train_dataset.target_scaler,
        train_dataset.local_feature_scaler,
        config,
    )
    test_dataset = _build_distillation_dataset_for_split(
        frame,
        "test",
        train_dataset.target_scaler,
        train_dataset.local_feature_scaler,
        config,
    )

    teacher_model, teacher_history = train_teacher_model(train_dataset, val_dataset, config)
    student_model, student_history = train_student_model(train_dataset, val_dataset, teacher_model, config)

    teacher_prediction_frames = [
        predict_teacher_frame(teacher_model, train_dataset, config.batch_size, split_name="train", device=config.device),
        predict_teacher_frame(teacher_model, val_dataset, config.batch_size, split_name="val", device=config.device),
        predict_teacher_frame(teacher_model, test_dataset, config.batch_size, split_name="test", device=config.device),
    ]
    student_prediction_frames = [
        predict_student_frame(student_model, train_dataset, config.batch_size, split_name="train", device=config.device),
        predict_student_frame(student_model, val_dataset, config.batch_size, split_name="val", device=config.device),
        predict_student_frame(student_model, test_dataset, config.batch_size, split_name="test", device=config.device),
    ]
    teacher_predictions = _combine_prediction_frames(teacher_prediction_frames)
    student_predictions = _combine_prediction_frames(student_prediction_frames)
    teacher_metrics = compute_metrics_for_suffix(teacher_predictions, prediction_suffix="teacher")
    student_metrics = compute_metrics_for_suffix(student_predictions, prediction_suffix="student")

    _write_named_history(teacher_history, config.output_dir, "teacher_train_history.csv")
    _write_named_history(student_history, config.output_dir, "student_train_history.csv")
    _write_named_predictions(teacher_predictions, config.output_dir, "teacher_predictions.csv")
    _write_named_predictions(student_predictions, config.output_dir, "student_predictions.csv")
    _write_named_metrics(teacher_metrics, config.output_dir, "teacher_metrics.json")
    _write_named_metrics(student_metrics, config.output_dir, "student_metrics.json")

    if config.save_figures:
        _plot_distillation_pred_vs_true(
            teacher_predictions,
            config.figure_dir,
            prediction_suffix="teacher",
            filename="teacher_true_vs_predict.png",
        )
        _plot_distillation_pred_vs_true(
            student_predictions,
            config.figure_dir,
            prediction_suffix="student",
            filename="student_true_vs_predict.png",
        )
        plot_teacher_vs_student(teacher_predictions, student_predictions, config.figure_dir)
    save_distillation_package(
        teacher_model,
        student_model,
        train_dataset.target_scaler,
        train_dataset.local_feature_scaler,
        config.output_dir,
        config,
    )

    return {
        "teacher_model": teacher_model,
        "student_model": student_model,
        "teacher_history": teacher_history,
        "student_history": student_history,
        "teacher_predictions": teacher_predictions,
        "student_predictions": student_predictions,
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
        "target_scaler": train_dataset.target_scaler,
        "local_feature_scaler": train_dataset.local_feature_scaler,
    }
