"""Grid-search defaults for the baseline CNN surrogate."""


SEARCH_ID = "cnn_spatial25_anti_overfit_v1"
RESULT_PREFIX = "cnn_surrogate_fine_tuning"

# 2 * 2 * 3 * 3 * 3 * 3 = 324 combinations.
PARAM_GRID = {
    "batch_size": [32, 64],
    "epochs": [60, 120],
    "learning_rate": [5.0e-4, 2.5e-4, 1.0e-4],
    "weight_decay": [3.0e-4, 1.0e-3, 3.0e-3],
    "dropout": [0.35, 0.5, 0.65],
    "loss_weight_stiffness": [1.0],
    "spatial_pool_height": [6],
    "spatial_pool_width": [3],
    "embedding_dim": [64],
    "loss_weight_local_strain": [0.5, 1.0, 1.5],
}
