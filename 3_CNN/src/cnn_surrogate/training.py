import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_surrogate.data import cnn_target_columns, coordinate_target_columns
from cnn_surrogate.devices import move_tensor_to_device, resolve_device, should_pin_memory
from cnn_surrogate.losses import (
    weighted_mse_loss,
    distillation_loss,
    coordinate_weighted_mse_loss,
    cnn_spatial_supervision_loss,
)
from cnn_surrogate.models import CnnSurrogate, TeacherSurrogate, StudentSurrogate, CoordinateSurrogate


try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _plain_progress(iterable, description):
    values = list(iterable)
    total = len(values)
    for position, value in enumerate(values, start=1):
        print("%s: %d/%d" % (description, position, total))
        yield value


def iter_progress(iterable, description, enabled=True):
    if not enabled:
        return iterable
    if tqdm is None:
        return _plain_progress(iterable, description)
    return tqdm(iterable, desc=description, unit="epoch", file=sys.stdout, dynamic_ncols=True, leave=True)


def _validated_early_stopping_patience(config):
    patience = config.early_stopping_patience
    if patience is not None and patience < 0:
        raise ValueError("early_stopping_patience must be None or non-negative")
    return patience


def _update_early_stopping(val_loss, best_val_loss, stale_epoch_count, patience):
    if patience is None or val_loss is None:
        return best_val_loss, stale_epoch_count, False
    if best_val_loss is None or val_loss < best_val_loss:
        return val_loss, 0, False
    stale_epoch_count += 1
    return best_val_loss, stale_epoch_count, stale_epoch_count >= patience


def run_epoch(
    model,
    loader,
    config,
    optimizer=None,
    device=None,
):
    if loader is None:
        return None
    if device is None:
        device = torch.device("cpu")
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    for images, stiffness_targets, local_maps, local_masks in loader:
        images = move_tensor_to_device(images, device)
        stiffness_targets = move_tensor_to_device(stiffness_targets, device)
        local_maps = move_tensor_to_device(local_maps, device)
        local_masks = move_tensor_to_device(local_masks, device)
        if training:
            optimizer.zero_grad()
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
        if training:
            loss.backward()
            optimizer.step()
        batch_count = int(images.shape[0])
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
    if total_count == 0:
        return None
    return total_loss / float(total_count)


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


def coordinate_checkpoint_signature(config):
    return {
        "data_csv": config.data_csv,
        "train_test_split": config.train_test_split,
        "split_shuffle": config.split_shuffle,
        "random_seed": config.random_seed,
        "coordinate_domain_width": config.coordinate_domain_width,
        "coordinate_domain_height": config.coordinate_domain_height,
        "coordinate_feature_dim": config.coordinate_feature_dim,
        "point_hidden_dim": config.point_hidden_dim,
        "context_hidden_dim": config.context_hidden_dim,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "loss_weight_stiffness": config.loss_weight_stiffness,
        "loss_weight_local_strain": config.loss_weight_local_strain,
        "early_stopping_patience": config.early_stopping_patience,
        "target_columns": coordinate_target_columns(),
    }


def _checkpoint_error_message():
    return "checkpoint is incompatible; remove checkpoint.pt or set WARM_START=False"


def _scaler_state(scaler):
    if scaler is None or not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        return None
    return {
        "mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
    }


def _cnn_scaler_states(train_dataset):
    stiffness_state = _scaler_state(getattr(train_dataset, "stiffness_scaler", None))
    local_state = _scaler_state(getattr(train_dataset, "local_strain_scaler", None))
    if stiffness_state is None and local_state is None:
        return None
    return {
        "stiffness_scaler": stiffness_state,
        "local_strain_scaler": local_state,
    }


def _scaler_state_matches(saved_state, expected_state):
    if expected_state is None:
        return True
    if saved_state is None:
        return False
    for field_name in ["mean", "scale"]:
        saved_values = np.asarray(saved_state.get(field_name), dtype=np.float64)
        expected_values = np.asarray(expected_state.get(field_name), dtype=np.float64)
        if saved_values.shape != expected_values.shape:
            return False
        if not np.allclose(saved_values, expected_values):
            return False
    return True


def _cnn_scalers_match_checkpoint(checkpoint, train_dataset):
    expected_states = _cnn_scaler_states(train_dataset)
    if expected_states is None:
        return True
    saved_states = checkpoint.get("scaler_states")
    if saved_states is None:
        return False
    return (
        _scaler_state_matches(saved_states.get("stiffness_scaler"), expected_states.get("stiffness_scaler"))
        and _scaler_state_matches(saved_states.get("local_strain_scaler"), expected_states.get("local_strain_scaler"))
    )


def _load_cnn_checkpoint_if_available(model, optimizer, train_dataset, config, device):
    if not config.warm_start or not config.checkpoint_path or not os.path.exists(config.checkpoint_path):
        return 0, [], None, 0
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    if checkpoint.get("config_signature") != cnn_checkpoint_signature(config):
        raise ValueError(_checkpoint_error_message())
    if not _cnn_scalers_match_checkpoint(checkpoint, train_dataset):
        raise ValueError(_checkpoint_error_message())
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint.get("epoch", 0)),
        list(checkpoint.get("history", [])),
        checkpoint.get("best_val_loss"),
        int(checkpoint.get("stale_epoch_count", 0)),
    )


def _save_cnn_checkpoint(
    model,
    optimizer,
    train_dataset,
    config,
    epoch,
    history,
    best_val_loss,
    stale_epoch_count,
):
    if not config.warm_start or not config.checkpoint_path:
        return None
    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    temp_path = config.checkpoint_path + ".tmp"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "stale_epoch_count": stale_epoch_count,
        "scaler_states": _cnn_scaler_states(train_dataset),
        "config_signature": cnn_checkpoint_signature(config),
    }, temp_path)
    os.replace(temp_path, config.checkpoint_path)
    return config.checkpoint_path


def _coordinate_scalers_match_checkpoint(checkpoint, train_dataset):
    if (
        "stiffness_scaler_mean" not in checkpoint
        or "stiffness_scaler_scale" not in checkpoint
        or "local_strain_scaler_mean" not in checkpoint
        or "local_strain_scaler_scale" not in checkpoint
    ):
        return False
    pairs = [
        ("stiffness_scaler_mean", train_dataset.stiffness_scaler.mean_),
        ("stiffness_scaler_scale", train_dataset.stiffness_scaler.scale_),
        ("local_strain_scaler_mean", train_dataset.local_strain_scaler.mean_),
        ("local_strain_scaler_scale", train_dataset.local_strain_scaler.scale_),
    ]
    for key, expected in pairs:
        saved_values = np.asarray(checkpoint.get(key), dtype=np.float64)
        expected_values = np.asarray(expected, dtype=np.float64)
        if saved_values.shape != expected_values.shape:
            return False
        if not np.allclose(saved_values, expected_values):
            return False
    return True


def _load_coordinate_checkpoint_if_available(model, optimizer, train_dataset, config, device):
    if not config.warm_start or not config.checkpoint_path or not os.path.exists(config.checkpoint_path):
        return 0, [], None, 0
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    if checkpoint.get("config_signature") != coordinate_checkpoint_signature(config):
        raise ValueError(_checkpoint_error_message())
    if not _coordinate_scalers_match_checkpoint(checkpoint, train_dataset):
        raise ValueError(_checkpoint_error_message())
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint.get("epoch", 0)),
        list(checkpoint.get("history", [])),
        checkpoint.get("best_val_loss"),
        int(checkpoint.get("stale_epoch_count", 0)),
    )


def _save_coordinate_checkpoint(
    model,
    optimizer,
    train_dataset,
    config,
    epoch,
    history,
    best_val_loss,
    stale_epoch_count,
):
    if not config.warm_start or not config.checkpoint_path:
        return None
    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    temp_path = config.checkpoint_path + ".tmp"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "stale_epoch_count": stale_epoch_count,
        "stiffness_scaler_mean": train_dataset.stiffness_scaler.mean_.tolist(),
        "stiffness_scaler_scale": train_dataset.stiffness_scaler.scale_.tolist(),
        "local_strain_scaler_mean": train_dataset.local_strain_scaler.mean_.tolist(),
        "local_strain_scaler_scale": train_dataset.local_strain_scaler.scale_.tolist(),
        "config_signature": coordinate_checkpoint_signature(config),
    }, temp_path)
    os.replace(temp_path, config.checkpoint_path)
    return config.checkpoint_path


def run_coordinate_epoch(model, loader, config, optimizer=None, device=None):
    if loader is None:
        return None
    if device is None:
        device = torch.device("cpu")
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    for coordinates, stiffness_target, local_targets in loader:
        coordinates = move_tensor_to_device(coordinates, device)
        stiffness_target = move_tensor_to_device(stiffness_target, device)
        local_targets = move_tensor_to_device(local_targets, device)
        if training:
            optimizer.zero_grad()
        predictions = model(coordinates)
        loss = coordinate_weighted_mse_loss(
            predictions,
            stiffness_target,
            local_targets,
            stiffness_weight=config.loss_weight_stiffness,
            local_strain_weight=config.loss_weight_local_strain,
        )
        if training:
            loss.backward()
            optimizer.step()
        batch_count = int(coordinates.shape[0])
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
    if total_count == 0:
        return None
    return total_loss / float(total_count)


def train_coordinate_model(train_dataset, val_dataset, config):
    if len(train_dataset) == 0:
        raise ValueError("training split is empty")
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    device = resolve_device(config.device)
    model = CoordinateSurrogate(
        point_feature_dim=config.coordinate_feature_dim,
        point_hidden_dim=config.point_hidden_dim,
        context_hidden_dim=config.context_hidden_dim,
        dropout=config.dropout,
    ).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=should_pin_memory(device),
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=should_pin_memory(device),
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    early_stopping_patience = _validated_early_stopping_patience(config)
    start_epoch, history, best_val_loss, stale_epoch_count = _load_coordinate_checkpoint_if_available(
        model,
        optimizer,
        train_dataset,
        config,
        device,
    )
    if start_epoch >= config.epochs:
        return model, history

    epoch_iterator = iter_progress(
        range(start_epoch + 1, config.epochs + 1),
        config.progress_description,
        enabled=config.show_progress,
    )
    for epoch in epoch_iterator:
        train_loss = run_coordinate_epoch(
            model,
            train_loader,
            config,
            optimizer=optimizer,
            device=device,
        )
        with torch.no_grad():
            val_loss = run_coordinate_epoch(
                model,
                val_loader,
                config,
                optimizer=None,
                device=device,
            )
        if hasattr(epoch_iterator, "set_postfix"):
            postfix = {"train_loss": train_loss}
            if val_loss is not None:
                postfix["val_loss"] = val_loss
            epoch_iterator.set_postfix(postfix)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        best_val_loss, stale_epoch_count, should_stop = _update_early_stopping(
            val_loss,
            best_val_loss,
            stale_epoch_count,
            early_stopping_patience,
        )
        _save_coordinate_checkpoint(
            model,
            optimizer,
            train_dataset,
            config,
            epoch,
            history,
            best_val_loss,
            stale_epoch_count,
        )
        if should_stop:
            break
    return model, history


def train_model(train_dataset, val_dataset, config):
    if len(train_dataset) == 0:
        raise ValueError("training split is empty")
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    device = resolve_device(config.device)
    model = CnnSurrogate(
        dropout=config.dropout,
        embedding_dim=config.embedding_dim,
        pooled_height=config.spatial_pool_height,
        pooled_width=config.spatial_pool_width,
    ).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=should_pin_memory(device),
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=should_pin_memory(device),
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    early_stopping_patience = _validated_early_stopping_patience(config)
    start_epoch, history, best_val_loss, stale_epoch_count = _load_cnn_checkpoint_if_available(
        model,
        optimizer,
        train_dataset,
        config,
        device,
    )
    if start_epoch >= config.epochs:
        return model, history

    epoch_iterator = iter_progress(
        range(start_epoch + 1, config.epochs + 1),
        config.progress_description,
        enabled=config.show_progress,
    )
    for epoch in epoch_iterator:
        train_loss = run_epoch(
            model,
            train_loader,
            config,
            optimizer=optimizer,
            device=device,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                config,
                optimizer=None,
                device=device,
            )
        if hasattr(epoch_iterator, "set_postfix"):
            postfix = {"train_loss": train_loss}
            if val_loss is not None:
                postfix["val_loss"] = val_loss
            epoch_iterator.set_postfix(postfix)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        best_val_loss, stale_epoch_count, should_stop = _update_early_stopping(
            val_loss,
            best_val_loss,
            stale_epoch_count,
            early_stopping_patience,
        )
        _save_cnn_checkpoint(
            model,
            optimizer,
            train_dataset,
            config,
            epoch,
            history,
            best_val_loss,
            stale_epoch_count,
        )
        if should_stop:
            break
    return model, history


def _run_teacher_epoch(
    model,
    loader,
    optimizer=None,
    stiffness_weight=1.0,
    strain_weight=1.0,
    device=None,
):
    if loader is None:
        return None
    if device is None:
        device = torch.device("cpu")
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    for images, local_features, targets in loader:
        images = move_tensor_to_device(images, device)
        local_features = move_tensor_to_device(local_features, device)
        targets = move_tensor_to_device(targets, device)
        if training:
            optimizer.zero_grad()
        predictions = model(images, local_features)
        loss = weighted_mse_loss(
            predictions,
            targets,
            stiffness_weight=stiffness_weight,
            strain_weight=strain_weight,
        )
        if training:
            loss.backward()
            optimizer.step()
        batch_count = int(images.shape[0])
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
    if total_count == 0:
        return None
    return total_loss / float(total_count)


def _run_student_epoch(model, teacher_model, loader, config, optimizer=None, device=None):
    if loader is None:
        return None
    if device is None:
        device = torch.device("cpu")
    training = optimizer is not None
    model.train(training)
    teacher_model.eval()
    total_loss = 0.0
    total_count = 0
    for images, local_features, targets in loader:
        images = move_tensor_to_device(images, device)
        local_features = move_tensor_to_device(local_features, device)
        targets = move_tensor_to_device(targets, device)
        if training:
            optimizer.zero_grad()
        predictions = model(images)
        with torch.no_grad():
            teacher_predictions = teacher_model(images, local_features)
        loss = distillation_loss(predictions, targets, teacher_predictions, config)
        if training:
            loss.backward()
            optimizer.step()
        batch_count = int(images.shape[0])
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
    if total_count == 0:
        return None
    return total_loss / float(total_count)


def _distillation_loader(dataset, batch_size, shuffle, device):
    if dataset is None or len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=should_pin_memory(device),
    )


def train_teacher_model(train_dataset, val_dataset, config):
    if len(train_dataset) == 0:
        raise ValueError("training split is empty")
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    device = resolve_device(config.device)
    model = TeacherSurrogate(dropout=config.dropout).to(device)
    train_loader = _distillation_loader(train_dataset, config.batch_size, shuffle=True, device=device)
    val_loader = _distillation_loader(val_dataset, config.batch_size, shuffle=False, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.teacher_learning_rate,
        weight_decay=config.weight_decay,
    )
    early_stopping_patience = _validated_early_stopping_patience(config)
    best_val_loss = None
    stale_epoch_count = 0
    history = []
    epoch_iterator = iter_progress(
        range(1, config.teacher_epochs + 1),
        config.progress_description,
        enabled=config.show_progress,
    )
    for epoch in epoch_iterator:
        train_loss = _run_teacher_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            stiffness_weight=config.loss_weight_stiffness,
            strain_weight=config.loss_weight_strain,
            device=device,
        )
        with torch.no_grad():
            val_loss = _run_teacher_epoch(
                model,
                val_loader,
                optimizer=None,
                stiffness_weight=config.loss_weight_stiffness,
                strain_weight=config.loss_weight_strain,
                device=device,
            )
        if hasattr(epoch_iterator, "set_postfix"):
            postfix = {"train_loss": train_loss}
            if val_loss is not None:
                postfix["val_loss"] = val_loss
            epoch_iterator.set_postfix(postfix)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        best_val_loss, stale_epoch_count, should_stop = _update_early_stopping(
            val_loss,
            best_val_loss,
            stale_epoch_count,
            early_stopping_patience,
        )
        if should_stop:
            break
    return model, history


def train_student_model(train_dataset, val_dataset, teacher_model, config):
    if len(train_dataset) == 0:
        raise ValueError("training split is empty")
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    device = resolve_device(config.device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    model = StudentSurrogate(dropout=config.dropout).to(device)
    train_loader = _distillation_loader(train_dataset, config.batch_size, shuffle=True, device=device)
    val_loader = _distillation_loader(val_dataset, config.batch_size, shuffle=False, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.student_learning_rate,
        weight_decay=config.weight_decay,
    )
    early_stopping_patience = _validated_early_stopping_patience(config)
    best_val_loss = None
    stale_epoch_count = 0
    history = []
    epoch_iterator = iter_progress(
        range(1, config.student_epochs + 1),
        config.progress_description,
        enabled=config.show_progress,
    )
    for epoch in epoch_iterator:
        train_loss = _run_student_epoch(
            model,
            teacher_model,
            train_loader,
            config,
            optimizer=optimizer,
            device=device,
        )
        with torch.no_grad():
            val_loss = _run_student_epoch(
                model,
                teacher_model,
                val_loader,
                config,
                optimizer=None,
                device=device,
            )
        if hasattr(epoch_iterator, "set_postfix"):
            postfix = {"train_loss": train_loss}
            if val_loss is not None:
                postfix["val_loss"] = val_loss
            epoch_iterator.set_postfix(postfix)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        best_val_loss, stale_epoch_count, should_stop = _update_early_stopping(
            val_loss,
            best_val_loss,
            stale_epoch_count,
            early_stopping_patience,
        )
        if should_stop:
            break
    return model, history
