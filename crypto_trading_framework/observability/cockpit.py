from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import polars as pl
import yaml
from plotly.subplots import make_subplots
from streamlit.runtime.scriptrunner import RerunData, RerunException

from crypto_trading_framework.core.config_schema import validate_config
from crypto_trading_framework.observability.provider import ObservabilityDataProvider

st = None
try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("Streamlit tidak terinstall. Install dengan: pip install streamlit plotly kaleido") from exc

st.set_page_config(
    page_title="Quantuis Kokpit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REFRESH_INTERVAL = 15


def load_config(path: str = "config/base.yaml") -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return validate_config(raw)


@st.cache_resource
def get_provider(config_path: str) -> ObservabilityDataProvider:
    config = load_config(config_path)
    return ObservabilityDataProvider(config)


def page_market(provider: ObservabilityDataProvider, symbol: str, timeframe: str):
    st.header(f"📊 Market Overview: {symbol} ({timeframe})")
    ohlcv = provider.get_ohlcv(symbol, timeframe, limit=500)
    indicators = provider.get_indicators(symbol, timeframe, limit=500)

    if ohlcv.is_empty():
        st.warning("Data OHLCV belum tersedia untuk simbol/timeframe ini.")
        return

    df = ohlcv.to_pandas()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLCV",
            increasing_line_color="#00c853",
            decreasing_line_color="#ff3d00",
        ),
        row=1,
        col=1,
    )

    for col, color in [("ema_20", "#2979ff"), ("ema_50", "#ffea00"), ("ema_200", "#ff1744")]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df[col], name=col, line={"color": color, "width": 1}), row=1, col=1
            )

    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bb_upper"],
                name="BB Upper",
                line={"color": "#b0bec5", "width": 1, "dash": "dot"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bb_lower"],
                name="BB Lower",
                line={"color": "#b0bec5", "width": 1, "dash": "dot"},
            ),
            row=1,
            col=1,
        )

    if "volume" in df.columns:
        colors = ["#00c853" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ff3d00" for i in range(len(df))]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], name="Volume", marker_color=colors), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=700,
        showlegend=True,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)

    if not indicators.is_empty():
        ind_df = indicators.to_pandas()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("RSI (14)", f"{ind_df['rsi'].iloc[-1]:.2f}" if "rsi" in ind_df.columns else "N/A")
        with c2:
            st.metric("ATR (14)", f"{ind_df['atr'].iloc[-1]:.2f}" if "atr" in ind_df.columns else "N/A")
        with c3:
            st.metric("ADX", f"{ind_df['adx'].iloc[-1]:.2f}" if "adx" in ind_df.columns else "N/A")


def page_signals(provider: ObservabilityDataProvider):
    st.header("🎯 Model Signals")
    signals = provider.get_signals()

    if not signals:
        st.info("Belum ada sinyal aktif di Redis.")
        return

    for sig in signals:
        symbol = sig.get("symbol", "UNKNOWN")
        direction = sig.get("direction", "N/A")
        prob = sig.get("probability_float", sig.get("probability", 0.0))
        veto = sig.get("smart_money_analysis", {}).get("veto_status", "NONE")
        entry = (
            sig.get("entry_zone", [0.0])[0] if isinstance(sig.get("entry_zone"), list) else sig.get("entry_zone", 0.0)
        )
        tp = (
            sig.get("take_profit", [0.0])[0]
            if isinstance(sig.get("take_profit"), list)
            else sig.get("take_profit", 0.0)
        )
        sl = sig.get("stop_loss_atr", 0.0)

        color = "#00c853" if direction == "LONG" else "#ff3d00" if direction == "SHORT" else "#ffea00"
        veto_badge = f"🔴 {veto}" if veto != "NONE" else "🟢 None"

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            c1.markdown(f"### {symbol}")
            c2.markdown(f"<span style='color:{color};font-weight:bold'>{direction}</span>", unsafe_allow_html=True)
            c3.markdown(f"Prob: **{prob * 100:.1f}%**")
            c4.markdown(f"Veto: {veto_badge}")

            c5, c6, c7 = st.columns(3)
            c5.metric("Entry", f"{entry:.2f}")
            c6.metric("TP", f"{tp:.2f}")
            c7.metric("SL (ATR)", f"{sl:.2f}")


def page_smart_money(provider: ObservabilityDataProvider):
    st.header("💰 Smart Money Tracker")
    symbols = provider.config.get("data", {}).get("symbols", [])
    for symbol in symbols:
        snapshot = provider.get_smart_money_snapshot(symbol)
        with st.container(border=True):
            st.subheader(symbol)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Composite", snapshot.get("composite_signal", "NEUTRAL"))
            c2.metric("Veto", snapshot.get("veto_status", "NONE"))
            c3.metric("OI Change 24h", f"{snapshot.get('oi_change_24h_pct', 0.0):.2f}%")
            c4.metric("Stablecoin Bal", f"{snapshot.get('total_stablecoin_balance', 0.0):,.0f}")

            c5, c6 = st.columns(2)
            c5.metric("Long Liq (USD)", f"{snapshot.get('long_liquidation_usd', 0.0):,.0f}")
            c6.metric("Short Liq (USD)", f"{snapshot.get('short_liquidation_usd', 0.0):,.0f}")


def page_shadow_trader(provider: ObservabilityDataProvider):
    st.header("🕯️ Shadow Trader")
    perf = provider.get_shadow_trader_performance()
    if perf:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Balance", f"${perf.get('current_balance', 0.0):,.2f}")
        c2.metric("ROI", perf.get("roi_percentage", "0.00%"))
        c3.metric("Win Rate", perf.get("win_rate", "0.00%"))
        c4.metric("Total Trades", perf.get("total_trades", 0))

    open_trades = provider.get_open_trades()
    if open_trades:
        st.subheader("Open Positions")
        for trade in open_trades:
            with st.container(border=True):
                st.write(f"**{trade.get('symbol')}** {trade.get('side')} | Size: {trade.get('size')}")
                st.caption(
                    f"Entry: {trade.get('entry_price')} | SL: {trade.get('stop_loss')} | TP: {trade.get('take_profit')}"
                )
    else:
        st.info("Tidak ada posisi terbuka.")

    closed_trades = provider.get_closed_trades(limit=20)
    if closed_trades:
        st.subheader("Closed Trades (Last 20)")
        trade_df = pl.DataFrame(closed_trades)
        if "closed_at" in trade_df.columns:
            trade_df = trade_df.sort("closed_at", descending=True)
        st.dataframe(trade_df.to_pandas(), use_container_width=True)


def page_backtest(provider: ObservabilityDataProvider, symbol: str, timeframe: str):
    st.header("🧪 Backtest Results")
    result = provider.get_latest_backtest(symbol, timeframe)
    if not result:
        st.info("Belum ada hasil backtest untuk simbol/timeframe ini.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", result.get("total_trades", 0))
    c2.metric("Win Rate", f"{result.get('winrate', 0.0):.1f}%")
    c3.metric("Total PnL", f"{result.get('total_pnl', 0.0):.2f}")
    c4.metric("Max Drawdown", f"{result.get('max_drawdown_pct', 0.0):.2f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Return", f"{result.get('return_pct', 0.0):.2f}%")
    c6.metric("Profit Factor", f"{result.get('profit_factor', 0.0):.2f}")
    c7.metric("Final Capital", f"{result.get('final_capital', 0.0):,.2f}")

    trades = result.get("trades", [])
    if trades:
        st.subheader("Trade History")
        trade_df = pl.DataFrame(trades)
        if "exit_time" in trade_df.columns:
            trade_df = trade_df.sort("exit_time", descending=True)
        st.dataframe(trade_df.to_pandas(), use_container_width=True)


def page_health(provider: ObservabilityDataProvider):
    st.header("🏥 System Health")
    db_health = provider.get_db_health()
    redis_health = provider.get_redis_health()

    c1, c2 = st.columns(2)
    with c1:
        status = db_health.get("status", "unknown")
        color = "🟢" if status == "healthy" else "🔴"
        st.metric("TimescaleDB", f"{color} {status}", db_health.get("dialect", ""))

    with c2:
        status = redis_health.get("status", "unknown")
        color = "🟢" if status == "healthy" else "🟡" if status == "unavailable" else "🔴"
        st.metric("Redis", f"{color} {status}", f"{redis_health.get('latency_ms', 'N/A')}ms")


def main():
    config_path = st.sidebar.text_input("Config Path", value="config/base.yaml")
    try:
        provider = get_provider(config_path)
    except (OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
        st.error(f"Gagal memuat konfigurasi: {exc}")
        st.stop()

    symbols = provider.config.get("data", {}).get("symbols", ["BTC/USDT:USDT"])
    timeframes = provider.config.get("data", {}).get("timeframes", ["h1"])

    st.sidebar.title("Quantuis Kokpit")
    selected_symbol = st.sidebar.selectbox("Symbol", symbols)
    selected_tf = st.sidebar.selectbox("Timeframe", timeframes)
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Interval (detik)", 5, 120, REFRESH_INTERVAL)

    page = st.sidebar.radio(
        "Halaman",
        ["Market Overview", "Model Signals", "Smart Money", "Shadow Trader", "Backtest", "System Health"],
    )

    if page == "Market Overview":
        page_market(provider, selected_symbol, selected_tf)
    elif page == "Model Signals":
        page_signals(provider)
    elif page == "Smart Money":
        page_smart_money(provider)
    elif page == "Shadow Trader":
        page_shadow_trader(provider)
    elif page == "Backtest":
        page_backtest(provider, selected_symbol, selected_tf)
    elif page == "System Health":
        page_health(provider)

    st.sidebar.caption(f"Last updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")

    if auto_refresh:
        time.sleep(refresh_interval)
        raise RerunException(RerunData(None))


if __name__ == "__main__":
    main()
