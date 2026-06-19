from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import AVWhisperConfig


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    model_config: AVWhisperConfig,
    optimizer: torch.optim.Optimizer | None = None,
    **metadata: Any,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config.to_dict(),
        **metadata,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)
