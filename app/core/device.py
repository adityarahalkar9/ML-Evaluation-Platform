from __future__ import annotations

import logging

import torch

from app.config import get_settings

logger = logging.getLogger("app.device")


def resolve_device() -> torch.device:
    pref = get_settings().device.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and pref in ("auto", "cuda"):
        return torch.device("cuda")
    if pref == "cuda":
        logger.warning("device=cuda requested but CUDA is unavailable; falling back to CPU")
    return torch.device("cpu")