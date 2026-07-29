import polars as pl
import numpy as np


def compute_ema(df: pl.DataFrame, column: str = "close", periods: list = [20, 50, 200]) -> pl.DataFrame:
    result = df
    for period in periods:
        ema_col = f"ema_{period}"
        result = result.with_columns(
            pl.col(column).ewm_mean(span=period, adjust=False).alias(ema_col)
        )
    return result


def compute_bollinger_bands(
    df: pl.DataFrame, column: str = "close", period: int = 20, multiplier: float = 2.0
) -> pl.DataFrame:
    sma = pl.col(column).rolling_mean(window_size=period)
    std = pl.col(column).rolling_std(window_size=period)

    df = df.with_columns(
        sma.alias("bb_middle"),
        (sma + multiplier * std).alias("bb_upper"),
        (sma - multiplier * std).alias("bb_lower"),
        (multiplier * std / sma).alias("bb_width"),
    )
    return df


def compute_rsi(df: pl.DataFrame, column: str = "close", period: int = 14) -> pl.DataFrame:
    delta = pl.col(column).diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df = df.with_columns(rsi.alias("rsi"))
    return df


def compute_stochastic(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pl.DataFrame:
    lowest_low = pl.col(low_col).rolling_min(window_size=k_period)
    highest_high = pl.col(high_col).rolling_max(window_size=k_period)

    raw_k = 100 * (pl.col(close_col) - lowest_low) / (highest_high - lowest_low)
    stoch_k = raw_k.rolling_mean(window_size=smooth_k)
    stoch_d = stoch_k.rolling_mean(window_size=d_period)

    df = df.with_columns(
        stoch_k.alias("stoch_k"),
        stoch_d.alias("stoch_d"),
    )
    return df


def compute_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    high = pl.col("high")
    low = pl.col("low")
    close = pl.col("close")
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pl.max_horizontal(tr1, tr2, tr3)

    atr = tr.rolling_mean(window_size=period)
    df = df.with_columns(atr.alias("atr"))
    return df


def compute_macd(
    df: pl.DataFrame,
    column: str = "close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    ema_fast = pl.col(column).ewm_mean(span=fast, adjust=False)
    ema_slow = pl.col(column).ewm_mean(span=slow, adjust=False)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm_mean(span=signal, adjust=False)
    macd_hist = macd - macd_signal

    df = df.with_columns(
        macd.alias("macd"),
        macd_signal.alias("macd_signal"),
        macd_hist.alias("macd_hist"),
    )
    return df


def compute_ichimoku(
    df: pl.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_span_b: int = 52,
) -> pl.DataFrame:
    high = pl.col("high")
    low = pl.col("low")

    tenkan_high = high.rolling_max(window_size=tenkan)
    tenkan_low = low.rolling_min(window_size=tenkan)
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    kijun_high = high.rolling_max(window_size=kijun)
    kijun_low = low.rolling_min(window_size=kijun)
    kijun_sen = (kijun_high + kijun_low) / 2

    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b_high = high.rolling_max(window_size=senkou_span_b)
    senkou_span_b_low = low.rolling_min(window_size=senkou_span_b)
    senkou_span_b_val = (senkou_span_b_high + senkou_span_b_low) / 2

    chikou_span = pl.col("close").shift(-kijun)

    df = df.with_columns(
        tenkan_sen.alias("ichimoku_tenkan"),
        kijun_sen.alias("ichimoku_kijun"),
        senkou_span_a.alias("ichimoku_senkou_a"),
        senkou_span_b_val.alias("ichimoku_senkou_b"),
        chikou_span.alias("ichimoku_chikou"),
    )
    return df


def compute_volume_profile(df: pl.DataFrame, bins: int = 50) -> pl.DataFrame:
    close = pl.col("close")
    volume = pl.col("volume")

    price_min = close.min()
    price_max = close.max()
    price_range = price_max - price_min

    bin_width = price_range / bins
    bin_idx = ((close - price_min) / bin_width).floor().clip(0, bins - 1)

    volume_by_bin = volume.sum().over(bin_idx)

    df = df.with_columns(
        bin_idx.alias("volume_bin"),
        volume_by_bin.alias("volume_at_bin"),
    )

    total_volume = volume.sum()
    volume_share = pl.col("volume_at_bin") / total_volume

    df = df.with_columns(
        volume_share.alias("volume_share"),
    )

    volume_profile = df.group_by("volume_bin").agg(
        pl.col("volume").sum().alias("bin_volume"),
        pl.col("close").mean().alias("bin_price"),
    )

    volume_profile = volume_profile.sort("volume_bin")
    cum_vol = volume_profile["bin_volume"].cum_sum()
    total_vol = volume_profile["bin_volume"].sum()
    volume_profile = volume_profile.with_columns(
        (cum_vol / total_vol).alias("cum_volume_share")
    )

    f20 = volume_profile.filter(pl.col("cum_volume_share") >= 0.2)
    p20 = f20.head(1)["bin_price"].item() if len(f20) > 0 else float(close.mean())

    f50 = volume_profile.filter(pl.col("cum_volume_share") >= 0.5)
    p50 = f50.head(1)["bin_price"].item() if len(f50) > 0 else float(close.mean())

    f80 = volume_profile.filter(pl.col("cum_volume_share") >= 0.8)
    p80 = f80.head(1)["bin_price"].item() if len(f80) > 0 else float(close.mean())

    df = df.with_columns(
        pl.lit(p20).alias("volume_profile_p20"),
        pl.lit(p50).alias("volume_profile_p50"),
        pl.lit(p80).alias("volume_profile_p80"),
        (close - p50).abs().alias("distance_from_poc"),
    )

    skewness = df.group_by("volume_bin").agg(
        pl.col("volume").sum().alias("bin_volume"),
        pl.col("close").mean().alias("bin_price"),
    ).with_columns(
        ((pl.col("bin_price") - pl.col("bin_price").mean()) / (pl.col("bin_price").std() + 1e-9)).alias("z_score")
    ).select(
        (pl.col("z_score") ** 3).mean().alias("volume_profile_skew")
    )

    kurtosis = df.group_by("volume_bin").agg(
        pl.col("volume").sum().alias("bin_volume"),
        pl.col("close").mean().alias("bin_price"),
    ).with_columns(
        ((pl.col("bin_price") - pl.col("bin_price").mean()) / (pl.col("bin_price").std() + 1e-9)).alias("z_score")
    ).select(
        (pl.col("z_score") ** 4).mean().alias("volume_profile_kurtosis")
    )

    df = df.with_columns(
        pl.lit(skewness["volume_profile_skew"].item()).alias("volume_profile_skew"),
        pl.lit(kurtosis["volume_profile_kurtosis"].item()).alias("volume_profile_kurtosis"),
    )

    return df


def compute_microstructure_features(df: pl.DataFrame) -> pl.DataFrame:
    close = pl.col("close")
    volume = pl.col("volume")

    returns = close.pct_change()
    volatility = returns.rolling_std(window_size=20)

    autocorr = returns.rolling_map(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0.0,
        window_size=20,
    )

    kurt = returns.rolling_map(
        lambda x: ((x - x.mean()) / x.std()).pow(4).mean() - 3 if x.std() > 0 else 0.0,
        window_size=20,
    )

    tick_direction = pl.when(returns > 0).then(1).when(returns < 0).then(-1).otherwise(0)
    tick_imbalance = tick_direction.rolling_sum(window_size=20) / 20

    trade_velocity = volume.rolling_std(window_size=10) / (volume.rolling_mean(window_size=10) + 1e-9)

    spread_proxy = (close - close.shift(1)).abs()
    spread_dynamics = spread_proxy.rolling_std(window_size=5) / (spread_proxy.rolling_mean(window_size=5) + 1e-9)

    df = df.with_columns(
        volatility.alias("volatility"),
        autocorr.alias("autocorrelation"),
        kurt.alias("kurtosis"),
        tick_imbalance.alias("tick_imbalance"),
        trade_velocity.alias("trade_velocity"),
        spread_dynamics.alias("spread_dynamics"),
    )

    return df


def compute_returns(df: pl.DataFrame) -> pl.DataFrame:
    close = pl.col("close")
    df = df.with_columns(
        close.pct_change(1).alias("returns_1d"),
        close.pct_change(5).alias("returns_5d"),
    )
    return df


def compute_vwap(df: pl.DataFrame) -> pl.DataFrame:
    high = pl.col("high")
    low = pl.col("low")
    close = pl.col("close")
    volume = pl.col("volume")

    typical_price = (high + low + close) / 3
    cumulative_tpv = (typical_price * volume).cum_sum()
    cumulative_vol = volume.cum_sum()

    vwap = cumulative_tpv / (cumulative_vol + 1e-9)

    df = df.with_columns(
        vwap.alias("vwap"),
        (close - vwap).alias("vwap_deviation"),
    )

    return df


def compute_obv(df: pl.DataFrame) -> pl.DataFrame:
    close = pl.col("close")
    volume = pl.col("volume")

    obv = pl.when(close.diff() > 0).then(volume).when(close.diff() < 0).then(-volume).otherwise(0).cum_sum()

    df = df.with_columns(
        obv.alias("obv"),
        obv.diff().alias("obv_change"),
    )

    return df


def compute_volume_spike(df: pl.DataFrame, window: int = 20, threshold: float = 2.0) -> pl.DataFrame:
    volume = pl.col("volume")
    volume_sma = volume.rolling_mean(window_size=window)
    volume_std = volume.rolling_std(window_size=window)

    volume_spike = (volume - volume_sma) / (volume_std + 1e-9)

    df = df.with_columns(
        volume_spike.alias("volume_spike"),
        (volume_spike > threshold).cast(pl.Int64).alias("volume_spike_flag"),
    )

    return df


def compute_regime(df: pl.DataFrame, lookback: int = 20, threshold: float = 25.0) -> pl.DataFrame:
    high = pl.col("high")
    low = pl.col("low")
    close = pl.col("close")

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = pl.when(plus_dm > minus_dm).then(plus_dm).otherwise(0)
    minus_dm = pl.when(minus_dm > plus_dm).then(minus_dm).otherwise(0)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pl.max_horizontal(tr1, tr2, tr3)

    atr = tr.rolling_mean(window_size=lookback)

    plus_di = 100 * plus_dm.rolling_mean(window_size=lookback) / (atr + 1e-9)
    minus_di = 100 * minus_dm.rolling_mean(window_size=lookback) / (atr + 1e-9)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.rolling_mean(window_size=lookback)

    regime = pl.when(adx > threshold).then(pl.lit(1)).otherwise(pl.lit(0))

    df = df.with_columns(
        regime.alias("regime_trending"),
        adx.alias("adx"),
        plus_di.alias("plus_di"),
        minus_di.alias("minus_di"),
    )

    return df


def add_all_indicators(df: pl.DataFrame) -> pl.DataFrame:
    df = compute_ema(df, column="close", periods=[20, 50, 200])
    df = compute_bollinger_bands(df, column="close", period=20, multiplier=2.0)
    df = compute_rsi(df, column="close", period=14)
    df = compute_stochastic(df, high_col="high", low_col="low", close_col="close")
    df = compute_atr(df, period=14)
    df = compute_macd(df, column="close", fast=12, slow=26, signal=9)
    df = compute_ichimoku(df, tenkan=9, kijun=26, senkou_span_b=52)
    df = compute_volume_profile(df, bins=50)
    df = compute_microstructure_features(df)
    df = compute_returns(df)
    df = compute_vwap(df)
    df = compute_obv(df)
    df = compute_volume_spike(df)
    df = compute_regime(df)
    return df
