import torch


def resolve_device(requested_device="auto"):
    if requested_device is None:
        requested_device = "auto"
    if isinstance(requested_device, torch.device):
        device = requested_device
    else:
        requested = str(requested_device).strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def should_pin_memory(device):
    return torch.device(device).type == "cuda"


def move_tensor_to_device(tensor, device):
    return tensor.to(device, non_blocking=should_pin_memory(device))
