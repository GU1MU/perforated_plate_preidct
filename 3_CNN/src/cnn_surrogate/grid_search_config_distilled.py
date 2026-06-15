"""Grid-search defaults for the distilled CNN surrogate."""


SEARCH_ID = "distilled_v1"
RESULT_PREFIX = "distilled_surrogate_fine_tuning"

PARAM_GRID = {
    "batch_size": [32, 64],
    "teacher_epochs": [300],
    "student_epochs": [300, 500],
    "teacher_learning_rate": [1.0e-3, 5.0e-4, 2.5e-4],
    "student_learning_rate": [1.0e-3, 5.0e-4, 2.5e-4],
    "weight_decay": [0.0, 1.0e-5, 1.0e-4],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "loss_weight_stiffness": [1.0, 2.0],
    "loss_weight_strain": [1.0, 2.0],
    "distill_weight": [0.1, 0.3],
}
