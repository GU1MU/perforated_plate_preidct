import math

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


HOLE_COUNT = 24
LEGACY_GLOBAL_TARGET_COLUMNS = ["relative_equivalent_stiffness", "max_strain_concentration_factor"]
TARGET_COLUMNS = LEGACY_GLOBAL_TARGET_COLUMNS


def required_columns():
    columns = ["odb_name", "status", "group_index", "instance_index"]
    for index in range(1, HOLE_COUNT + 1):
        columns.append("hole_%02d_x" % index)
        columns.append("hole_%02d_y" % index)
    columns.extend(TARGET_COLUMNS)
    return columns


def local_feature_columns():
    return ["hole_%02d_strain_concentration_factor" % index for index in range(1, HOLE_COUNT + 1)]


def cnn_target_columns():
    return ["relative_equivalent_stiffness"] + local_feature_columns()


def required_cnn_columns():
    columns = ["odb_name", "status", "group_index", "instance_index"]
    for index in range(1, HOLE_COUNT + 1):
        columns.append("hole_%02d_x" % index)
        columns.append("hole_%02d_y" % index)
    columns.extend(cnn_target_columns())
    return columns


def required_distillation_columns():
    return required_columns() + local_feature_columns()


def coordinate_feature_names():
    return [
        "x_norm",
        "y_norm",
        "left_distance_norm",
        "right_distance_norm",
        "bottom_distance_norm",
        "top_distance_norm",
    ]


def coordinate_target_columns():
    return ["relative_equivalent_stiffness"] + local_feature_columns()


def required_coordinate_columns():
    columns = ["odb_name", "status", "group_index", "instance_index"]
    for index in range(1, HOLE_COUNT + 1):
        columns.append("hole_%02d_x" % index)
        columns.append("hole_%02d_y" % index)
    columns.extend(coordinate_target_columns())
    return columns


def filter_valid_rows(frame):
    return frame[frame["status"] == "ok"].copy()


def validate_columns(frame):
    missing = [column for column in required_columns() if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(missing))
    return None


def validate_distillation_columns(frame):
    missing = [column for column in required_distillation_columns() if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(missing))
    return None


def validate_cnn_columns(frame):
    missing = [column for column in required_cnn_columns() if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(missing))
    return None


def validate_coordinate_columns(frame):
    missing = [column for column in required_coordinate_columns() if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(missing))
    return None


def load_dataset_table(path):
    frame = pd.read_csv(path)
    validate_columns(frame)
    frame = filter_valid_rows(frame)
    frame = frame.dropna(subset=required_columns())
    frame["group_index"] = frame["group_index"].astype(int)
    frame["instance_index"] = frame["instance_index"].astype(int)
    return frame.reset_index(drop=True)


def load_cnn_dataset_table(path):
    frame = pd.read_csv(path)
    validate_cnn_columns(frame)
    frame = filter_valid_rows(frame)
    frame = frame.dropna(subset=required_cnn_columns())
    frame["group_index"] = frame["group_index"].astype(int)
    frame["instance_index"] = frame["instance_index"].astype(int)
    return frame.reset_index(drop=True)


def load_coordinate_dataset_table(path):
    frame = pd.read_csv(path)
    validate_coordinate_columns(frame)
    frame = filter_valid_rows(frame)
    frame = frame.dropna(subset=required_coordinate_columns())
    frame["group_index"] = frame["group_index"].astype(int)
    frame["instance_index"] = frame["instance_index"].astype(int)
    return frame.reset_index(drop=True)


def assign_splits(frame, train_count, shuffle, random_seed):
    pieces = []
    rng = np.random.RandomState(random_seed)
    for _, group in frame.groupby("group_index"):
        group = group.sort_values("instance_index").copy()
        indices = list(group.index)
        if shuffle:
            rng.shuffle(indices)
        split_by_index = {}
        train_indices = indices[:train_count]
        remaining = indices[train_count:]
        val_count = int(math.ceil(len(remaining) / 2.0))
        val_indices = remaining[:val_count]
        test_indices = remaining[val_count:]
        for index in train_indices:
            split_by_index[index] = "train"
        for index in val_indices:
            split_by_index[index] = "val"
        for index in test_indices:
            split_by_index[index] = "test"
        group["split"] = [split_by_index[index] for index in group.index]
        pieces.append(group)
    if not pieces:
        result = frame.copy()
        result["split"] = []
        return result
    return pd.concat(pieces, axis=0).sort_values(["group_index", "instance_index"]).reset_index(drop=True)


def _clamp(value, low, high):
    return max(low, min(high, value))


def encode_hole_image(row, pixel_size, height, width):
    image = np.zeros((1, height, width), dtype=np.float32)
    for index in range(1, HOLE_COUNT + 1):
        x = float(row["hole_%02d_x" % index])
        y = float(row["hole_%02d_y" % index])
        r = _clamp(int(math.floor(y / pixel_size)), 0, height - 1)
        c = _clamp(int(math.floor(x / pixel_size)), 0, width - 1)
        image[0, r, c] = 1.0
    return image


def encode_local_strain_map(row, pixel_size, height, width, local_strain_scaler=None):
    target_map = np.zeros((1, height, width), dtype=np.float32)
    mask = np.zeros((1, height, width), dtype=np.float32)
    values = []
    positions = []
    occupied = set()
    for index in range(1, HOLE_COUNT + 1):
        x = float(row["hole_%02d_x" % index])
        y = float(row["hole_%02d_y" % index])
        r = _clamp(int(math.floor(y / pixel_size)), 0, height - 1)
        c = _clamp(int(math.floor(x / pixel_size)), 0, width - 1)
        if (r, c) in occupied:
            raise ValueError("multiple holes map to the same pixel: row=%d, column=%d" % (r, c))
        occupied.add((r, c))
        value = float(row["hole_%02d_strain_concentration_factor" % index])
        values.append([value])
        positions.append((r, c))
    values = np.asarray(values, dtype=np.float32)
    if local_strain_scaler is not None:
        values = local_strain_scaler.transform(values)
    for offset, (r, c) in enumerate(positions):
        target_map[0, r, c] = float(values[offset, 0])
        mask[0, r, c] = 1.0
    return target_map, mask


def encode_coordinate_features(row, domain_width, domain_height):
    features = np.zeros((HOLE_COUNT, len(coordinate_feature_names())), dtype=np.float32)
    for index in range(1, HOLE_COUNT + 1):
        x = float(row["hole_%02d_x" % index])
        y = float(row["hole_%02d_y" % index])
        x_norm = x / float(domain_width)
        y_norm = y / float(domain_height)
        features[index - 1] = np.array([
            x_norm,
            y_norm,
            x_norm,
            (float(domain_width) - x) / float(domain_width),
            y_norm,
            (float(domain_height) - y) / float(domain_height),
        ], dtype=np.float32)
    return features


class HoleLayoutDataset(Dataset):
    def __init__(self, frame, image_height, image_width, pixel_size, target_scaler=None, fit_scaler=False):
        self.frame = frame.reset_index(drop=True)
        if len(self.frame) == 0:
            self.images = np.zeros((0, 1, image_height, image_width), dtype=np.float32)
            self.targets = np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32)
            self.target_scaler = target_scaler
            return

        self.images = np.stack([
            encode_hole_image(row, pixel_size=pixel_size, height=image_height, width=image_width)
            for _, row in self.frame.iterrows()
        ])
        targets = self.frame[TARGET_COLUMNS].values.astype(np.float32)
        if target_scaler is None:
            target_scaler = StandardScaler()
        if fit_scaler:
            targets = target_scaler.fit_transform(targets)
        else:
            targets = target_scaler.transform(targets)
        self.targets = targets.astype(np.float32)
        self.target_scaler = target_scaler

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[index]), torch.from_numpy(self.targets[index])


class CnnSpatialLayoutDataset(Dataset):
    def __init__(
        self,
        frame,
        image_height,
        image_width,
        pixel_size,
        stiffness_scaler=None,
        local_strain_scaler=None,
        fit_scalers=False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_height = image_height
        self.image_width = image_width
        self.pixel_size = pixel_size
        if stiffness_scaler is None:
            stiffness_scaler = StandardScaler()
        if local_strain_scaler is None:
            local_strain_scaler = StandardScaler()
        self.stiffness_scaler = stiffness_scaler
        self.local_strain_scaler = local_strain_scaler

        if len(self.frame) == 0:
            self.images = np.zeros((0, 1, image_height, image_width), dtype=np.float32)
            self.stiffness_targets = np.zeros((0, 1), dtype=np.float32)
            self.local_maps = np.zeros((0, 1, image_height, image_width), dtype=np.float32)
            self.local_masks = np.zeros((0, 1, image_height, image_width), dtype=np.float32)
            return

        self.images = np.stack([
            encode_hole_image(row, pixel_size=pixel_size, height=image_height, width=image_width)
            for _, row in self.frame.iterrows()
        ]).astype(np.float32)

        stiffness_targets = self.frame[["relative_equivalent_stiffness"]].values.astype(np.float32)
        if fit_scalers:
            stiffness_targets = self.stiffness_scaler.fit_transform(stiffness_targets)
        else:
            stiffness_targets = self.stiffness_scaler.transform(stiffness_targets)
        self.stiffness_targets = stiffness_targets.astype(np.float32)

        local_values = self.frame[local_feature_columns()].values.astype(np.float32).reshape(-1, 1)
        if fit_scalers:
            self.local_strain_scaler.fit(local_values)

        local_targets = [
            encode_local_strain_map(
                row,
                pixel_size=pixel_size,
                height=image_height,
                width=image_width,
                local_strain_scaler=self.local_strain_scaler,
            )
            for _, row in self.frame.iterrows()
        ]
        self.local_maps = np.stack([target_map for target_map, _ in local_targets]).astype(np.float32)
        self.local_masks = np.stack([mask for _, mask in local_targets]).astype(np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.images[index]),
            torch.from_numpy(self.stiffness_targets[index]),
            torch.from_numpy(self.local_maps[index]),
            torch.from_numpy(self.local_masks[index]),
        )


class CoordinateLayoutDataset(Dataset):
    def __init__(
        self,
        frame,
        domain_width,
        domain_height,
        stiffness_scaler=None,
        local_strain_scaler=None,
        fit_scalers=False,
    ):
        self.frame = frame.reset_index(drop=True)
        if stiffness_scaler is None:
            stiffness_scaler = StandardScaler()
        if local_strain_scaler is None:
            local_strain_scaler = StandardScaler()
        self.stiffness_scaler = stiffness_scaler
        self.local_strain_scaler = local_strain_scaler

        if len(self.frame) == 0:
            self.coordinates = np.zeros((0, HOLE_COUNT, len(coordinate_feature_names())), dtype=np.float32)
            self.stiffness_targets = np.zeros((0, 1), dtype=np.float32)
            self.local_targets = np.zeros((0, HOLE_COUNT), dtype=np.float32)
            return

        self.coordinates = np.stack([
            encode_coordinate_features(row, domain_width=domain_width, domain_height=domain_height)
            for _, row in self.frame.iterrows()
        ])

        stiffness_targets = self.frame[["relative_equivalent_stiffness"]].values.astype(np.float32)
        local_targets = self.frame[local_feature_columns()].values.astype(np.float32)
        if fit_scalers:
            stiffness_targets = self.stiffness_scaler.fit_transform(stiffness_targets)
            local_targets = self.local_strain_scaler.fit_transform(local_targets.reshape(-1, 1)).reshape(local_targets.shape)
        else:
            stiffness_targets = self.stiffness_scaler.transform(stiffness_targets)
            local_targets = self.local_strain_scaler.transform(local_targets.reshape(-1, 1)).reshape(local_targets.shape)
        self.stiffness_targets = stiffness_targets.astype(np.float32)
        self.local_targets = local_targets.astype(np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.coordinates[index]),
            torch.from_numpy(self.stiffness_targets[index]),
            torch.from_numpy(self.local_targets[index]),
        )


class DistillationLayoutDataset(Dataset):
    def __init__(
        self,
        frame,
        image_height,
        image_width,
        pixel_size,
        target_scaler=None,
        local_feature_scaler=None,
        fit_scaler=False,
        fit_local_feature_scaler=False,
        fit_scalers=None,
    ):
        if fit_scalers is not None:
            fit_scaler = fit_scalers
            fit_local_feature_scaler = fit_scalers

        self.frame = frame.reset_index(drop=True)
        if target_scaler is None:
            target_scaler = StandardScaler()
        if local_feature_scaler is None:
            local_feature_scaler = StandardScaler()
        self.target_scaler = target_scaler
        self.local_feature_scaler = local_feature_scaler

        if len(self.frame) == 0:
            self.images = np.zeros((0, 1, image_height, image_width), dtype=np.float32)
            self.local_features = np.zeros((0, len(local_feature_columns())), dtype=np.float32)
            self.targets = np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32)
            return

        self.images = np.stack([
            encode_hole_image(row, pixel_size=pixel_size, height=image_height, width=image_width)
            for _, row in self.frame.iterrows()
        ])

        local_features = self.frame[local_feature_columns()].values.astype(np.float32)
        if fit_local_feature_scaler:
            local_features = self.local_feature_scaler.fit_transform(local_features)
        else:
            local_features = self.local_feature_scaler.transform(local_features)
        self.local_features = local_features.astype(np.float32)

        targets = self.frame[TARGET_COLUMNS].values.astype(np.float32)
        if fit_scaler:
            targets = self.target_scaler.fit_transform(targets)
        else:
            targets = self.target_scaler.transform(targets)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.images[index]),
            torch.from_numpy(self.local_features[index]),
            torch.from_numpy(self.targets[index]),
        )
