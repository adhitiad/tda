"""
Checkpoint utilities for saving/loading model artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch


def save_checkpoint(
    model,
    scaler: object,
    metadata: dict,
    checkpoint_dir: str,
    timeframe: str,
    is_best: bool = False,
):
    """Save model checkpoint with versioned filenames."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "best" if is_best else "latest"
    model_file = checkpoint_path / f"{timeframe}_{suffix}_{timestamp}.pt"
    scaler_file = checkpoint_path / f"{timeframe}_{suffix}_{timestamp}_scaler.pkl"
    meta_file = checkpoint_path / f"{timeframe}_{suffix}_{timestamp}_meta.json"

    if hasattr(model, "state_dict"):
        torch.save(model.state_dict(), model_file)
    else:
        joblib.dump(model, model_file)

    joblib.dump(scaler, scaler_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return str(model_file), str(scaler_file), str(meta_file)
