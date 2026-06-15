"""Grid-search defaults for the baseline CNN surrogate."""


SEARCH_ID = "cnn_spatial25_v1"
RESULT_PREFIX = "cnn_surrogate_fine_tuning"

PARAM_GRID = {
    "batch_size": [32, 64],
    "epochs": [300, 500],
    "learning_rate": [1.0e-3, 5.0e-4, 2.5e-4],
    "weight_decay": [0.0, 1.0e-5, 1.0e-4],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "loss_weight_stiffness": [1.0, 2.0],
    "spatial_pool_height": [8, 10, 12],
    "spatial_pool_width": [4, 5, 6],
    "embedding_dim": [128, 256, 384],
    "loss_weight_local_strain": [0.5, 1.0, 2.0],
}
