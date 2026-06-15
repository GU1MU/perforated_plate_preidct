import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate import data


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _sample_row(group=1, instance=1, status="ok"):
    row = {
        "odb_name": "%d_%d_plate.odb" % (group, instance),
        "status": status,
        "group_index": group,
        "instance_index": instance,
        "relative_equivalent_stiffness": 0.5 + 0.01 * instance + 0.02 * group,
        "max_strain_concentration_factor": 1.5 + 0.03 * instance + 0.04 * group,
    }
    for index in range(1, 25):
        row["hole_%02d_x" % index] = float((index + instance) % 40)
        row["hole_%02d_y" % index] = float((2 * index + group) % 80)
    return row


class DataColumnTests(unittest.TestCase):
    def test_filter_valid_rows_keeps_only_ok_rows(self):
        frame = pd.DataFrame([
            {"odb_name": "a.odb", "status": "ok"},
            {"odb_name": "b.odb", "status": "failed"},
        ])
        result = data.filter_valid_rows(frame)
        self.assertEqual(list(result["odb_name"]), ["a.odb"])

    def test_required_columns_include_all_hole_centers_and_targets(self):
        columns = data.required_columns()
        self.assertIn("hole_01_x", columns)
        self.assertIn("hole_24_y", columns)
        self.assertIn("relative_equivalent_stiffness", columns)
        self.assertIn("max_strain_concentration_factor", columns)

    def test_validate_columns_reports_missing_columns(self):
        with self.assertRaises(ValueError):
            data.validate_columns(pd.DataFrame([{"odb_name": "a.odb"}]))

    def test_cnn_target_columns_are_stiffness_plus_24_local_strain_targets(self):
        columns = data.cnn_target_columns()
        self.assertEqual(len(columns), 25)
        self.assertEqual(columns[0], "relative_equivalent_stiffness")
        self.assertEqual(columns[1], "hole_01_strain_concentration_factor")
        self.assertEqual(columns[-1], "hole_24_strain_concentration_factor")

    def test_legacy_target_columns_remain_two_targets_for_distillation(self):
        self.assertEqual(data.TARGET_COLUMNS, [
            "relative_equivalent_stiffness",
            "max_strain_concentration_factor",
        ])

    def test_required_cnn_columns_include_cnn_targets_without_legacy_max_target(self):
        columns = data.required_cnn_columns()
        self.assertIn("hole_01_x", columns)
        self.assertIn("hole_24_y", columns)
        self.assertIn("relative_equivalent_stiffness", columns)
        self.assertIn("hole_24_strain_concentration_factor", columns)
        self.assertNotIn("max_strain_concentration_factor", columns)


class DataSplitTests(unittest.TestCase):
    def test_split_by_group_assigns_odd_remainder_extra_sample_to_validation(self):
        rows = []
        for instance in range(1, 7):
            rows.append({"odb_name": "1_%d_plate.odb" % instance, "group_index": 1, "instance_index": instance})
        frame = pd.DataFrame(rows)

        split = data.assign_splits(frame, train_count=3, shuffle=False, random_seed=1)

        self.assertEqual(list(split["split"]), ["train", "train", "train", "val", "val", "test"])

    def test_split_by_group_uses_train_count_and_splits_remainder(self):
        rows = []
        for group in [1, 2]:
            for instance in range(1, 6):
                rows.append({"odb_name": "%d_%d_plate.odb" % (group, instance), "group_index": group, "instance_index": instance})
        frame = pd.DataFrame(rows)

        split = data.assign_splits(frame, train_count=3, shuffle=False, random_seed=1)

        group1 = split[split["group_index"] == 1]
        self.assertEqual(list(group1["split"]), ["train", "train", "train", "val", "test"])
        self.assertEqual(split["split"].value_counts().to_dict(), {"train": 6, "val": 2, "test": 2})

    def test_split_by_group_keeps_small_group_in_train(self):
        frame = pd.DataFrame([
            {"odb_name": "1_1_plate.odb", "group_index": 1, "instance_index": 1},
            {"odb_name": "1_2_plate.odb", "group_index": 1, "instance_index": 2},
        ])
        split = data.assign_splits(frame, train_count=3, shuffle=False, random_seed=1)
        self.assertEqual(list(split["split"]), ["train", "train"])


class DataImageTests(unittest.TestCase):
    def test_encode_row_sets_hole_center_pixels(self):
        row = {}
        for index in range(1, 25):
            row["hole_%02d_x" % index] = 0.0
            row["hole_%02d_y" % index] = 0.0
        row["hole_01_x"] = 4.0
        row["hole_01_y"] = 6.0
        image = data.encode_hole_image(row, pixel_size=2.0, height=80, width=40)
        self.assertEqual(image.shape, (1, 80, 40))
        self.assertEqual(image[0, 3, 2], 1.0)

    def test_encode_row_clamps_boundary_pixels(self):
        row = {}
        for index in range(1, 25):
            row["hole_%02d_x" % index] = 999.0
            row["hole_%02d_y" % index] = 999.0
        image = data.encode_hole_image(row, pixel_size=2.0, height=80, width=40)
        self.assertEqual(image[0, 79, 39], 1.0)

    def test_encode_local_strain_map_sets_only_hole_pixels(self):
        row = _sample_row(instance=1)
        for index in range(1, 25):
            row["hole_%02d_x" % index] = float((index - 1) * 2)
            row["hole_%02d_y" % index] = 0.0
            row["hole_%02d_strain_concentration_factor" % index] = float(index)
        row["hole_01_x"] = 4.0
        row["hole_01_y"] = 6.0

        target_map, mask = data.encode_local_strain_map(
            row,
            pixel_size=2.0,
            height=80,
            width=40,
            local_strain_scaler=None,
        )

        self.assertEqual(target_map.shape, (1, 80, 40))
        self.assertEqual(mask.shape, (1, 80, 40))
        self.assertEqual(mask.sum(), 24.0)
        self.assertEqual(target_map[0, 3, 2], 1.0)

    def test_encode_local_strain_map_rejects_duplicate_hole_pixels(self):
        row = _sample_row(instance=1)
        for index in range(1, 25):
            row["hole_%02d_x" % index] = float(index * 2)
            row["hole_%02d_y" % index] = 0.0
            row["hole_%02d_strain_concentration_factor" % index] = float(index)
        row["hole_02_x"] = row["hole_01_x"]
        row["hole_02_y"] = row["hole_01_y"]

        with self.assertRaisesRegex(ValueError, "multiple holes map to the same pixel"):
            data.encode_local_strain_map(
                row,
                pixel_size=2.0,
                height=80,
                width=40,
                local_strain_scaler=None,
            )


class HoleLayoutDatasetTests(unittest.TestCase):
    def test_dataset_uses_configured_image_shape(self):
        frame = pd.DataFrame([_sample_row(instance=1), _sample_row(instance=2)])
        scaler = StandardScaler()
        dataset = data.HoleLayoutDataset(
            frame,
            image_height=80,
            image_width=40,
            pixel_size=2.0,
            target_scaler=scaler,
            fit_scaler=True,
        )
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (1, 80, 40))
        self.assertEqual(tuple(target.shape), (2,))

    def test_empty_dataset_uses_configured_image_shape(self):
        frame = pd.DataFrame(columns=data.required_columns())
        scaler = StandardScaler()
        scaler.fit(np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32))

        dataset = data.HoleLayoutDataset(
            frame,
            image_height=12,
            image_width=7,
            pixel_size=2.0,
            target_scaler=scaler,
            fit_scaler=False,
        )

        self.assertEqual(dataset.images.shape, (0, 1, 12, 7))
        self.assertEqual(dataset.targets.shape, (0, 2))


class CnnSpatialLayoutDatasetTests(unittest.TestCase):
    def test_cnn_spatial_dataset_returns_image_stiffness_target_local_map_and_mask(self):
        frame = pd.DataFrame([_sample_row(instance=1), _sample_row(instance=2)])
        for index in range(1, 25):
            frame["hole_%02d_strain_concentration_factor" % index] = 2.0 + index * 0.01

        dataset = data.CnnSpatialLayoutDataset(
            frame,
            image_height=80,
            image_width=40,
            pixel_size=2.0,
            stiffness_scaler=StandardScaler(),
            local_strain_scaler=StandardScaler(),
            fit_scalers=True,
        )

        image, stiffness_target, local_map, local_mask = dataset[0]
        self.assertEqual(tuple(image.shape), (1, 80, 40))
        self.assertEqual(tuple(stiffness_target.shape), (1,))
        self.assertEqual(tuple(local_map.shape), (1, 80, 40))
        self.assertEqual(tuple(local_mask.shape), (1, 80, 40))

    def test_cnn_spatial_dataset_fits_scalers_on_training_targets(self):
        frame = pd.DataFrame([_sample_row(instance=1), _sample_row(instance=2)])
        for index in range(1, 25):
            frame["hole_%02d_strain_concentration_factor" % index] = [
                float(index),
                float(index + 100),
            ]

        dataset = data.CnnSpatialLayoutDataset(
            frame,
            image_height=80,
            image_width=40,
            pixel_size=2.0,
            stiffness_scaler=StandardScaler(),
            local_strain_scaler=StandardScaler(),
            fit_scalers=True,
        )

        stiffness_values = frame[["relative_equivalent_stiffness"]].values.astype(np.float32)
        local_values = frame[data.local_feature_columns()].values.astype(np.float32).reshape(-1, 1)
        self.assertAlmostEqual(dataset.stiffness_scaler.mean_[0], float(stiffness_values.mean()))
        self.assertAlmostEqual(dataset.local_strain_scaler.mean_[0], float(local_values.mean()))

    def test_empty_cnn_spatial_dataset_uses_stable_shapes(self):
        frame = pd.DataFrame(columns=data.required_cnn_columns())

        dataset = data.CnnSpatialLayoutDataset(
            frame,
            image_height=12,
            image_width=7,
            pixel_size=2.0,
            stiffness_scaler=StandardScaler(),
            local_strain_scaler=StandardScaler(),
            fit_scalers=True,
        )

        self.assertEqual(dataset.images.shape, (0, 1, 12, 7))
        self.assertEqual(dataset.stiffness_targets.shape, (0, 1))
        self.assertEqual(dataset.local_maps.shape, (0, 1, 12, 7))
        self.assertEqual(dataset.local_masks.shape, (0, 1, 12, 7))


if __name__ == "__main__":
    unittest.main()
