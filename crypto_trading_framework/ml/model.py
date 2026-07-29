
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss untuk handling class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        pt = torch.where(targets == 1, torch.sigmoid(inputs), 1 - torch.sigmoid(inputs))
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CryptoLSTM(nn.Module):
    """
    Arsitektur Deep Learning sederhana untuk prediksi arah harga crypto.

    Struktur:
    - Input: (batch_size, time_steps, num_features)
    - LSTM layer(s)
    - Dropout untuk mencegah overfitting
    - Dense hidden layer
    - Output Sigmoid untuk probabilitas (0 - 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 32,
    ):
        """
        Inisialisasi model CryptoLSTM.

        :param input_size: Jumlah fitur per time step
        :param hidden_size: Ukuran hidden state LSTM
        :param num_layers: Jumlah layer LSTM
        :param dropout: Dropout probability
        :param fc_hidden: Ukuran hidden layer Dense setelah LSTM
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Mengembalikan raw logits (BUKAN probabilitas).
        Gunakan torch.sigmoid() secara eksplisit jika butuh probabilitas,
        atau gunakan BCEWithLogitsLoss yang sudah menerapkan sigmoid secara internal.

        :param x: Input tensor shape (batch_size, time_steps, input_size)
        :return: Output logits shape (batch_size, 1)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        x = self.dropout(last_hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CryptoGRU(nn.Module):
    """
    Alternatif arsitektur menggunakan GRU.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 32,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, h_n = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        x = self.dropout(last_hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TemporalAttention(nn.Module):
    """Temporal attention layer untuk sequence data."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, time_steps, hidden)
        :return: (batch, hidden)
        """
        attn_weights = F.softmax(self.attention(x).squeeze(-1), dim=-1)
        context = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return context


class CryptoAttentionLSTM(nn.Module):
    """
    Arsitektur LSTM dengan Temporal Attention.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 32,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = TemporalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class VariableSelection(nn.Module):
    """Variable selection network untuk TFT-style feature selection."""

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.weights = nn.Linear(num_features, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, time_steps, num_features)
        :return: (batch, time_steps, hidden_size)
        """
        weights = F.softmax(self.weights(x), dim=-1)
        return x * weights


class CryptoTFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer (TFT) inspired architecture.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 32,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.var_selection = VariableSelection(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = TemporalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.var_selection(x)
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_model(
    model_type: str = "lstm",
    input_size: int = 11,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    fc_hidden: int = 32,
) -> nn.Module:
    """
    Factory function untuk membuat model.

    :param model_type: 'lstm', 'gru', 'attention_lstm', atau 'tft'
    :param input_size: Jumlah fitur
    :param hidden_size: Ukuran hidden state
    :param num_layers: Jumlah layer
    :param dropout: Dropout probability
    :param fc_hidden: Ukuran hidden Dense layer
    :return: Instance model PyTorch
    """
    if model_type == "lstm":
        return CryptoLSTM(input_size, hidden_size, num_layers, dropout, fc_hidden)
    elif model_type == "gru":
        return CryptoGRU(input_size, hidden_size, num_layers, dropout, fc_hidden)
    elif model_type == "attention_lstm":
        return CryptoAttentionLSTM(input_size, hidden_size, num_layers, dropout, fc_hidden)
    elif model_type == "tft":
        return CryptoTFT(input_size, hidden_size, num_layers, dropout, fc_hidden)
    else:
        raise ValueError(f"Model type '{model_type}' tidak dikenali. Gunakan 'lstm', 'gru', 'attention_lstm', atau 'tft'.")
