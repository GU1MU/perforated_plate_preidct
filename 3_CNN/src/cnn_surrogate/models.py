import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class SpatialImageEncoder(nn.Module):
    def __init__(self, embedding_dim=256, pooled_height=10, pooled_width=5, dropout=0.2):
        super(SpatialImageEncoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((pooled_height, pooled_width))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pooled_height * pooled_width, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, images):
        spatial_features = self.backbone(images)
        pooled_features = self.spatial_pool(spatial_features)
        embedding = self.embedding(pooled_features)
        return spatial_features, embedding


class CnnSurrogate(nn.Module):
    def __init__(self, dropout=0.2, embedding_dim=256, pooled_height=10, pooled_width=5):
        super(CnnSurrogate, self).__init__()
        self.encoder = SpatialImageEncoder(
            embedding_dim=embedding_dim,
            pooled_height=pooled_height,
            pooled_width=pooled_width,
            dropout=dropout,
        )
        hidden_dim = max(1, embedding_dim // 2)
        self.stiffness_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.local_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, inputs):
        spatial_features, embedding = self.encoder(inputs)
        stiffness = self.stiffness_head(embedding)
        local_low_res = self.local_head(spatial_features)
        local_map = F.interpolate(
            local_low_res,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return stiffness, local_map


class TeacherSurrogate(nn.Module):
    def __init__(self, dropout=0.2):
        super(TeacherSurrogate, self).__init__()
        self.image_encoder = ImageEncoder()
        self.local_encoder = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, images, local_features):
        image_embedding = self.image_encoder(images)
        local_embedding = self.local_encoder(local_features)
        return self.regressor(torch.cat([image_embedding, local_embedding], dim=1))


class StudentSurrogate(nn.Module):
    def __init__(self, dropout=0.2):
        super(StudentSurrogate, self).__init__()
        self.image_encoder = ImageEncoder()
        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, images):
        return self.regressor(self.image_encoder(images))


class CoordinateSurrogate(nn.Module):
    def __init__(self, point_feature_dim=6, point_hidden_dim=128, context_hidden_dim=256, dropout=0.2):
        super(CoordinateSurrogate, self).__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(point_feature_dim, point_hidden_dim),
            nn.ReLU(),
            nn.Linear(point_hidden_dim, point_hidden_dim),
            nn.ReLU(),
        )
        self.context_head = nn.Sequential(
            nn.Linear(point_hidden_dim * 2, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.stiffness_head = nn.Linear(context_hidden_dim, 1)
        self.local_head = nn.Sequential(
            nn.Linear(point_hidden_dim + context_hidden_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_hidden_dim, 1),
        )

    def forward(self, coordinates):
        point_features = self.point_encoder(coordinates)
        mean_features = point_features.mean(dim=1)
        max_features = point_features.max(dim=1).values
        context = self.context_head(torch.cat([mean_features, max_features], dim=1))
        stiffness = self.stiffness_head(context)
        repeated_context = context.unsqueeze(1).expand(-1, point_features.shape[1], -1)
        local_inputs = torch.cat([point_features, repeated_context], dim=2)
        local_strain = self.local_head(local_inputs).squeeze(-1)
        return torch.cat([stiffness, local_strain], dim=1)
