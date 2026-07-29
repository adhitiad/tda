from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MLPipeline:
    """Pipeline untuk mempersiapkan data sebelum masuk ke model ML/DL."""

    def __init__(self, scaler_type: str = "minmax"):
        """
        Inisialisasi MLPipeline.

        :param scaler_type: Jenis scaler ('minmax' atau 'standard')
        """
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Scaler '{scaler_type}' tidak dikenali. Gunakan 'minmax' atau 'standard'.")

    def prepare_features(self, df: pl.DataFrame, feature_cols: list | None = None) -> tuple[np.ndarray, list]:
        """
        Menyiapkan fitur-fitur indikator untuk model.

        :param df: Polars DataFrame dengan indikator
        :param feature_cols: Daftar kolom fitur yang digunakan
        :return: (numpy array fitur, list nama kolom fitur)
        """
        if feature_cols is None:
            feature_cols = [
                "close", "volume",
                "ema_20", "ema_50",
                "bb_width",
                "rsi", "stoch_k", "stoch_d",
                "atr", "macd_hist",
                "volume_ratio",
            ]

        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            raise ValueError("Tidak ada kolom fitur yang ditemukan di DataFrame.")

        df_clean = df.drop_nulls(subset=available_cols)
        features = df_clean.select(available_cols).to_numpy()

        return features, available_cols

    def scale_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Melakukan feature scaling.

        :param features: Array numpy fitur
        :param fit: True untuk fit scaler (data training), False untuk transform saja (data test)
        :return: Array numpy yang sudah di-scale
        """
        if fit:
            return self.scaler.fit_transform(features)
        return self.scaler.transform(features)

    def save_scaler(self, path: str):
        """Menyimpan scaler ke disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str):
        """Memuat scaler dari disk."""
        self.scaler = joblib.load(path)

    def define_target(self, df: pl.DataFrame, forward_periods: int = 5, target_type: str = "binary") -> pl.DataFrame:
        """
        Mendefinisikan target klasifikasi.

        :param df: Polars DataFrame dengan kolom close, atr
        :param forward_periods: Jumlah candle ke depan
        :param target_type: 'binary', 'atr_adjusted', atau 'regime'
        :return: DataFrame dengan kolom 'target' baru
        """
        if target_type == "binary":
            future_close = pl.col("close").shift(-forward_periods)
            df = df.with_columns(
                pl.when(future_close > pl.col("close"))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("target")
            )
        elif target_type == "atr_adjusted":
            future_close = pl.col("close").shift(-forward_periods)
            atr = pl.col("atr")
            raw_return = (future_close - pl.col("close")) / pl.col("close")
            adjusted_return = raw_return / (atr + 1e-9)
            df = df.with_columns(
                pl.when(adjusted_return > 1.0)
                .then(pl.lit(1))
                .when(adjusted_return < -1.0)
                .then(pl.lit(0))
                .otherwise(pl.lit(0.5))
                .alias("target")
            )
        elif target_type == "regime":
            if "adx" in df.columns:
                df = df.with_columns(
                    pl.when(pl.col("adx") > 25.0)
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias("target")
                )
            else:
                volatility = pl.col("close").pct_change().rolling_std(window_size=20)
                momentum = (pl.col("close").ewm_mean(span=20, adjust=False) - pl.col("close").ewm_mean(span=50, adjust=False)).abs()
                trend_strength = momentum / (volatility * pl.col("close") + 1e-9)
                df = df.with_columns(
                    pl.when(trend_strength > 0.5)
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias("target")
                )
        else:
            raise ValueError(f"Target type '{target_type}' tidak dikenali.")

        return df

    def create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        time_steps: int = 60,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mengubah data tabular 2D menjadi bentuk 3D (samples, time_steps, features) untuk model Deep Learning (LSTM/GRU).

        :param features: Array numpy fitur yang sudah di-scale [N, num_features]
        :param targets: Array numpy target [N]
        :param time_steps: Panjang window/sequence untuk setiap sampel
        :return: (X_3d, y_1d) dengan shape [samples, time_steps, features] dan [samples]
        """
        if len(features) <= time_steps:
            raise ValueError(
                f"Data terlalu sedikit ({len(features)}) untuk time_steps={time_steps}"
            )

        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        for i in range(time_steps, len(features)):
            x_list.append(features[i - time_steps : i])
            y_list.append(np.array([targets[i]]))

        x_arr = np.array(x_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32).flatten()
        return x_arr, y_arr

    def walk_forward_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> list:
        """
        Walk-forward validation untuk time series.

        :param X: Fitur 3D
        :param y: Target
        :param n_splits: Jumlah split
        :return: List of (X_train, X_val, y_train, y_val) tuples
        """
        splits = []
        fold_size = len(X) // n_splits
        for i in range(1, n_splits):
            train_end = i * fold_size
            val_end = min((i + 1) * fold_size, len(X))
            X_train, X_val = X[:train_end], X[train_end:val_end]
            y_train, y_val = y[:train_end], y[train_end:val_end]
            splits.append((X_train, X_val, y_train, y_val))
        return splits

    def train_test_split_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        shuffle: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Memisahkan data menjadi train dan test.

        :param X: Fitur 3D
        :param y: Target
        :param test_size: Proporsi data test
        :param shuffle: Apakah mengacak (untuk time series sebaiknya False)
        :return: X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test
