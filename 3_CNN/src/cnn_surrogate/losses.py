import torch


def weighted_mse_loss(prediction, target, stiffness_weight, strain_weight):
    weights = torch.tensor(
        [stiffness_weight, strain_weight],
        dtype=prediction.dtype,
        device=prediction.device,
    )
    return ((prediction - target) ** 2 * weights).mean()


def coordinate_weighted_mse_loss(prediction, stiffness_target, local_targets, stiffness_weight, local_strain_weight):
    stiffness_loss = ((prediction[:, :1] - stiffness_target) ** 2).mean()
    local_loss = ((prediction[:, 1:] - local_targets) ** 2).mean()
    return stiffness_weight * stiffness_loss + local_strain_weight * local_loss


def cnn_spatial_supervision_loss(
    stiffness_prediction,
    stiffness_target,
    local_map_prediction,
    local_map_target,
    local_map_mask,
    stiffness_weight,
    local_strain_weight,
):
    stiffness_loss = ((stiffness_prediction - stiffness_target) ** 2).mean()
    local_squared_error = (local_map_prediction - local_map_target) ** 2 * local_map_mask
    local_denominator = local_map_mask.sum().clamp_min(1.0)
    local_loss = local_squared_error.sum() / local_denominator
    return stiffness_weight * stiffness_loss + local_strain_weight * local_loss


def distillation_loss(student_prediction, true_target, teacher_prediction, config):
    supervised = weighted_mse_loss(
        student_prediction,
        true_target,
        config.loss_weight_stiffness,
        config.loss_weight_strain,
    )
    distilled = weighted_mse_loss(
        student_prediction,
        teacher_prediction,
        config.loss_weight_stiffness,
        config.loss_weight_strain,
    )
    return supervised + config.distill_weight * distilled
