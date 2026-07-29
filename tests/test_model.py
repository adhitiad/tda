import pytest
import torch

from crypto_trading_framework.ml.model import FocalLoss, create_model


def test_create_lstm():
    model = create_model("lstm", input_size=11)
    x = torch.randn(4, 60, 11)
    out = model(x)
    assert out.shape == (4, 1)
    probs = torch.sigmoid(out)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_create_gru():
    model = create_model("gru", input_size=11)
    x = torch.randn(4, 60, 11)
    out = model(x)
    assert out.shape == (4, 1)


def test_create_attention_lstm():
    model = create_model("attention_lstm", input_size=11)
    x = torch.randn(4, 60, 11)
    out = model(x)
    assert out.shape == (4, 1)


def test_create_tft():
    model = create_model("tft", input_size=11)
    x = torch.randn(4, 60, 11)
    out = model(x)
    assert out.shape == (4, 1)


def test_focal_loss():
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    preds = torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    loss = loss_fn(preds, targets)
    assert loss.item() > 0


def test_invalid_model_type():
    with pytest.raises(ValueError):
        create_model("invalid_model", input_size=11)
