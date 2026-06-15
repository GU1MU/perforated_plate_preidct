from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaselineTrainingConfig(object):
    data_csv: str
    output_dir: str
    figure_dir: str
    temp_dir: str
    train_test_split: int
    split_shuffle: bool
    random_seed: int
    pixel_size: float
    image_height: int
    image_width: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    loss_weight_stiffness: float
    loss_weight_strain: float
    early_stopping_patience: Optional[int]
    device: str
    show_progress: bool
    progress_description: str
    save_model: bool
    spatial_pool_height: int = field(default=10, kw_only=True)
    spatial_pool_width: int = field(default=5, kw_only=True)
    embedding_dim: int = field(default=256, kw_only=True)
    loss_weight_local_strain: float = field(default=1.0, kw_only=True)
    warm_start: bool = field(default=False, kw_only=True)
    checkpoint_path: Optional[str] = field(default=None, kw_only=True)


@dataclass
class DistillationTrainingConfig(BaselineTrainingConfig):
    teacher_epochs: int
    student_epochs: int
    teacher_learning_rate: float
    student_learning_rate: float
    distill_weight: float


@dataclass
class CoordinateTrainingConfig(object):
    data_csv: str
    output_dir: str
    figure_dir: str
    temp_dir: str
    train_test_split: int
    split_shuffle: bool
    random_seed: int
    coordinate_domain_width: float
    coordinate_domain_height: float
    coordinate_feature_dim: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    point_hidden_dim: int
    context_hidden_dim: int
    loss_weight_stiffness: float
    loss_weight_local_strain: float
    early_stopping_patience: Optional[int]
    device: str
    show_progress: bool
    progress_description: str
    save_model: bool
    warm_start: bool
    checkpoint_path: str
