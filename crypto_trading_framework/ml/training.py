import numpy as np
import torch
import torch.nn as nn

from crypto_trading_framework.ml.model import FocalLoss, create_model
from crypto_trading_framework.core.logging import get_logger

logger = get_logger("training")


def get_class_weights(y: np.ndarray, method: str = "balanced") -> torch.Tensor:
    """
    Menghitung class weights untuk handling imbalanced data.

    :param y: Array target
    :param method: 'balanced' atau 'uniform'
    :return: Tensor class weights
    """
    if method == "uniform":
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    n_samples = len(y)
    n_classes = 2
    class_counts = np.bincount(y.astype(int), minlength=n_classes)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights = n_samples / (n_classes * class_counts)
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def get_loss_function(loss_type: str, class_weights: torch.Tensor | None = None) -> nn.Module:
    """
    Mendapatkan loss function berdasarkan nama dan weights.
    """
    if loss_type == "focal":
        return FocalLoss()
    elif loss_type == "bce":
        if class_weights is not None:
            pos_weight = class_weights[1] / class_weights[0]
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Loss type '{loss_type}' tidak dikenali.")


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model: nn.Module,
    epochs: int,
    lr: float,
    device: torch.device | str,
    loss_fn: nn.Module | None = None,
    early_stopping_patience: int = 10,
) -> tuple[nn.Module, float]:
    """Melatih model dengan early stopping dan mengembalikan model terbaik + validation loss."""
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    device_obj = torch.device(device) if isinstance(device, str) else device
    model = model.to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device_obj)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device_obj)

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train_tensor)
        loss = loss_fn(output, y_train_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_loss


def evaluate_model(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray, device: torch.device | str) -> float:
    """Evaluasi model dan mengembalikan akurasi."""
    device_obj = torch.device(device) if isinstance(device, str) else device
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device_obj)
        logits = model(x_test_tensor)
        preds = torch.sigmoid(logits).cpu().numpy().flatten()
        pred_labels = (preds >= 0.5).astype(int)
        accuracy = float(np.mean(pred_labels == y_test))
    return accuracy
