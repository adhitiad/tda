import polars as pl
from typing import Dict, List, Optional


def resample_to_base(df: pl.DataFrame, base_timeframe: str) -> pl.DataFrame:
    tf_minutes = {"m5": 5, "m15": 15, "m30": 30, "h1": 60, "h4": 240, "d1": 1440}
    base_minutes = tf_minutes.get(base_timeframe, 15)

    df = df.sort("timestamp")
    df = df.with_columns(
        pl.col("timestamp").dt.truncate(f"{base_minutes}m")
    )

    df_agg = df.group_by("timestamp").agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    )

    return df_agg.sort("timestamp")


def forward_fill_join(
    base_df: pl.DataFrame,
    aux_df: pl.DataFrame,
    aux_suffix: str,
    base_timeframe: str,
) -> pl.DataFrame:
    tf_minutes = {"m5": 5, "m15": 15, "m30": 30, "h1": 60, "h4": 240, "d1": 1440}
    base_minutes = tf_minutes.get(base_timeframe, 15)
    aux_minutes = tf_minutes.get(aux_suffix.replace("_features", ""), 60)

    if aux_minutes <= base_minutes:
        return base_df

    aux_resampled = resample_to_base(aux_df, base_timeframe)

    feature_cols = [c for c in aux_resampled.columns if c not in ("timestamp", "open", "high", "low", "close", "volume")]
    aux_features = aux_resampled.select(["timestamp"] + feature_cols).rename({c: f"{c}_{aux_suffix}" for c in feature_cols})

    result = base_df.join_asof(
        aux_features,
        on="timestamp",
        strategy="forward",
    )

    return result


def fuse_multi_timeframe(
    data: Dict[str, pl.DataFrame],
    primary_timeframe: str = "m15",
    auxiliary_timeframes: Optional[List[str]] = None,
) -> pl.DataFrame:
    if auxiliary_timeframes is None:
        auxiliary_timeframes = [tf for tf in data.keys() if tf != primary_timeframe]

    primary_df = data[primary_timeframe].sort("timestamp")

    for aux_tf in auxiliary_timeframes:
        if aux_tf not in data:
            continue
        aux_df = data[aux_tf].sort("timestamp")
        primary_df = forward_fill_join(primary_df, aux_df, aux_tf, primary_timeframe)

    return primary_df
