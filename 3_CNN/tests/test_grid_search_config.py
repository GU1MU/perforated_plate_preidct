import sys
import unittest
from pathlib import Path


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate import grid_search_config_cnn, grid_search_config_coordinate, grid_search_config_distilled


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


class GridSearchConfigTests(unittest.TestCase):
    def test_cnn_grid_config_keeps_split_and_patience_out_of_search_grid(self):
        self.assertEqual(grid_search_config_cnn.SEARCH_ID, "cnn_spatial25_anti_overfit_v1")
        self.assertEqual(grid_search_config_cnn.RESULT_PREFIX, "cnn_surrogate_fine_tuning")
        self.assertNotIn("train_test_split", grid_search_config_cnn.PARAM_GRID)
        self.assertNotIn("early_stopping_patience", grid_search_config_cnn.PARAM_GRID)
        self.assertNotIn("loss_weight_strain", grid_search_config_cnn.PARAM_GRID)
        self.assertEqual(grid_search_config_cnn.PARAM_GRID["spatial_pool_height"], [6])
        self.assertEqual(grid_search_config_cnn.PARAM_GRID["spatial_pool_width"], [3])
        self.assertEqual(grid_search_config_cnn.PARAM_GRID["embedding_dim"], [64])
        self.assertEqual(grid_search_config_cnn.PARAM_GRID["loss_weight_local_strain"], [0.5, 1.0, 1.5])

    def test_distilled_grid_config_keeps_split_and_patience_out_of_search_grid(self):
        self.assertEqual(grid_search_config_distilled.SEARCH_ID, "distilled_v1")
        self.assertEqual(grid_search_config_distilled.RESULT_PREFIX, "distilled_surrogate_fine_tuning")
        self.assertNotIn("train_test_split", grid_search_config_distilled.PARAM_GRID)
        self.assertNotIn("early_stopping_patience", grid_search_config_distilled.PARAM_GRID)
        self.assertIn("distill_weight", grid_search_config_distilled.PARAM_GRID)

    def test_coordinate_grid_config_searches_coordinate_model_parameters(self):
        self.assertEqual(grid_search_config_coordinate.SEARCH_ID, "coordinate_v1")
        self.assertEqual(grid_search_config_coordinate.RESULT_PREFIX, "coordinate_surrogate_fine_tuning")
        self.assertNotIn("train_test_split", grid_search_config_coordinate.PARAM_GRID)
        self.assertNotIn("early_stopping_patience", grid_search_config_coordinate.PARAM_GRID)
        self.assertNotIn("coordinate_domain_width", grid_search_config_coordinate.PARAM_GRID)
        self.assertNotIn("coordinate_domain_height", grid_search_config_coordinate.PARAM_GRID)
        self.assertEqual(grid_search_config_coordinate.PARAM_GRID["point_hidden_dim"], [64, 128, 256])
        self.assertEqual(grid_search_config_coordinate.PARAM_GRID["context_hidden_dim"], [128, 256, 384])
        self.assertEqual(grid_search_config_coordinate.PARAM_GRID["loss_weight_local_strain"], [0.5, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
