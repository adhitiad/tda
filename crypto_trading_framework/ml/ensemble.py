import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from crypto_trading_framework.ml.model import create_model


class EnsembleModel:
    """Ensemble model yang menggabungkan LSTM, Random Forest, dan XGBoost."""

    def __init__(
        self,
        input_size: int,
        device: torch.device,
        weights: Optional[List[float]] = None,
        voting: str = "soft",
    ):
        self.input_size = input_size
        self.device = device
        self.weights = weights or [0.5, 0.3, 0.2]
        self.voting = voting
        self.lstm_model = None
        self.rf_model = None
        self.xgb_model = None
        self.meta_model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        n_samples = X_train.shape[0]
        X_train_2d = X_train.reshape(n_samples, -1)

        self.lstm_model = create_model("lstm", input_size=self.input_size).to(self.device)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)

        self.lstm_model.train()
        for _ in range(50):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.rf_model.fit(X_train_2d, y_train)

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.xgb_model.fit(X_train_2d, y_train)

        if self.voting == "stacking":
            lstm_probs = self._get_lstm_probs(X_train).reshape(-1, 1)
            rf_probs = self.rf_model.predict_proba(X_train_2d)[:, 1].reshape(-1, 1)
            xgb_probs = self.xgb_model.predict_proba(X_train_2d)[:, 1].reshape(-1, 1)

            meta_features = np.hstack([lstm_probs, rf_probs, xgb_probs])
            self.meta_model = LogisticRegression()
            self.meta_model.fit(meta_features, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_2d = X.reshape(n_samples, -1)

        lstm_probs = self._get_lstm_probs(X).reshape(-1, 1)
        rf_probs = self.rf_model.predict_proba(X_2d)[:, 1].reshape(-1, 1)
        xgb_probs = self.xgb_model.predict_proba(X_2d)[:, 1].reshape(-1, 1)

        if self.voting == "stacking" and self.meta_model is not None:
            meta_features = np.hstack([lstm_probs, rf_probs, xgb_probs])
            return self.meta_model.predict_proba(meta_features)
        else:
            weighted_probs = (
                self.weights[0] * lstm_probs
                + self.weights[1] * rf_probs
                + self.weights[2] * xgb_probs
            )
            return np.hstack([1 - weighted_probs, weighted_probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def _get_lstm_probs(self, X: np.ndarray) -> np.ndarray:
        self.lstm_model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.lstm_model(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs
