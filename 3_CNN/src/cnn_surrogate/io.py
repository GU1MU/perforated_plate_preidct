import json
import os
import pickle

import pandas as pd
import torch

from cnn_surrogate.data import (
    TARGET_COLUMNS,
    cnn_target_columns,
    coordinate_feature_names,
    coordinate_target_columns,
    local_feature_columns,
)


def ensure_directory(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)
    return path


def remove_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)
    return None


def write_split_manifest(frame, output_dir):
    ensure_directory(output_dir)
    path = os.path.join(output_dir, "split_manifest.csv")
    columns = ["odb_name", "group_index", "instance_index", "split"]
    frame[columns].to_csv(path, index=False)
    return path


def write_train_history(history, output_dir):
    ensure_directory(output_dir)
    path = os.path.join(output_dir, "train_history.csv")
    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss"]).to_csv(path, index=False)
    return path


def write_metrics(metrics, output_dir):
    ensure_directory(output_dir)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    return path


def write_predictions(predictions, output_dir):
    ensure_directory(output_dir)
    path = os.path.join(output_dir, "predictions.csv")
    predictions.to_csv(path, index=False)
    return path


def save_model_package(model, target_scaler, output_dir, config):
    if not config.save_model:
        for filename in ["model.pt", "target_scaler.pkl"]:
            artifact_path = os.path.join(output_dir, filename)
            if os.path.isfile(artifact_path):
                os.remove(artifact_path)
        return None
    ensure_directory(output_dir)
    model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "image_height": config.image_height,
        "image_width": config.image_width,
        "pixel_size": config.pixel_size,
        "target_columns": TARGET_COLUMNS,
    }, model_path)
    with open(os.path.join(output_dir, "target_scaler.pkl"), "wb") as scaler_file:
        pickle.dump(target_scaler, scaler_file)
    return model_path


def save_cnn_spatial_model_package(model, stiffness_scaler, local_strain_scaler, output_dir, config):
    filenames = [
        "model.pt",
        "stiffness_scaler.pkl",
        "local_strain_scaler.pkl",
        "target_scaler.pkl",
    ]
    if not config.save_model:
        for filename in filenames:
            remove_if_exists(os.path.join(output_dir, filename))
        return None

    ensure_directory(output_dir)
    remove_if_exists(os.path.join(output_dir, "target_scaler.pkl"))
    model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "image_height": config.image_height,
        "image_width": config.image_width,
        "pixel_size": config.pixel_size,
        "target_columns": cnn_target_columns(),
        "embedding_dim": config.embedding_dim,
        "spatial_pool_height": config.spatial_pool_height,
        "spatial_pool_width": config.spatial_pool_width,
    }, model_path)
    with open(os.path.join(output_dir, "stiffness_scaler.pkl"), "wb") as scaler_file:
        pickle.dump(stiffness_scaler, scaler_file)
    with open(os.path.join(output_dir, "local_strain_scaler.pkl"), "wb") as scaler_file:
        pickle.dump(local_strain_scaler, scaler_file)
    return model_path


def save_distillation_package(teacher_model, student_model, target_scaler, local_feature_scaler, output_dir, config):
    filenames = [
        "teacher_model.pt",
        "student_model.pt",
        "target_scaler.pkl",
        "local_feature_scaler.pkl",
    ]
    if not config.save_model:
        for filename in filenames:
            artifact_path = os.path.join(output_dir, filename)
            if os.path.isfile(artifact_path):
                os.remove(artifact_path)
        return None

    ensure_directory(output_dir)
    base_metadata = {
        "image_height": config.image_height,
        "image_width": config.image_width,
        "pixel_size": config.pixel_size,
        "target_columns": TARGET_COLUMNS,
        "distill_weight": config.distill_weight,
    }
    teacher_metadata = dict(base_metadata)
    teacher_metadata["local_feature_columns"] = local_feature_columns()
    teacher_metadata["model_role"] = "teacher"
    student_metadata = dict(base_metadata)
    student_metadata["model_role"] = "student"
    teacher_path = os.path.join(output_dir, "teacher_model.pt")
    student_path = os.path.join(output_dir, "student_model.pt")
    torch.save(dict(teacher_metadata, model_state_dict=teacher_model.state_dict()), teacher_path)
    torch.save(dict(student_metadata, model_state_dict=student_model.state_dict()), student_path)
    with open(os.path.join(output_dir, "target_scaler.pkl"), "wb") as scaler_file:
        pickle.dump(target_scaler, scaler_file)
    with open(os.path.join(output_dir, "local_feature_scaler.pkl"), "wb") as scaler_file:
        pickle.dump(local_feature_scaler, scaler_file)
    return teacher_path


def save_coordinate_model_package(model, stiffness_scaler, local_strain_scaler, output_dir, config):
    ensure_directory(output_dir)
    model_path = os.path.join(output_dir, "model.pt")
    stiffness_scaler_path = os.path.join(output_dir, "stiffness_scaler.pkl")
    local_strain_scaler_path = os.path.join(output_dir, "local_strain_scaler.pkl")
    remove_if_exists(model_path)
    remove_if_exists(stiffness_scaler_path)
    remove_if_exists(local_strain_scaler_path)
    remove_if_exists(os.path.join(output_dir, "target_scaler.pkl"))
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
    }, model_path)
    with open(stiffness_scaler_path, "wb") as scaler_file:
        pickle.dump(stiffness_scaler, scaler_file)
    with open(local_strain_scaler_path, "wb") as scaler_file:
        pickle.dump(local_strain_scaler, scaler_file)
    return model_path
