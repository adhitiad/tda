# Quantuis CLI Reference

Quantuis is a Trading Data Analysis Framework. The CLI provides commands for
backtesting, live trading, signal generation, model training, and observability.

## Usage

All commands are run through the Typer CLI entry point:

```bash
python -m crypto_trading_framework.cli <command> [options]
```

Or via the thin wrapper:

```bash
python main.py <command> [options]
```

### Quick Reference

```bash
# Backtest
python -m crypto_trading_framework.cli backtest
python main.py backtest

# Live trading
python -m crypto_trading_framework.cli live
python main.py live

# Observability cockpit
python -m crypto_trading_framework.cli cockpit
python main.py cockpit

# Generate signals
python -m crypto_trading_framework.cli signals
python main.py signals

# Train models
python -m crypto_trading_framework.cli train
python main.py train

# Version
python -m crypto_trading_framework.cli version
python main.py version
```

## Commands

### `backtest`

Run a backtest on historical data.

```bash
python -m crypto_trading_framework.cli backtest
```

**Options:**

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `config/base.yaml` | Path to base config YAML |
| `--overlay` | `-o` | `None` | Path to overlay config YAML |
| `--symbol` | `-s` | `None` | Specific symbol to backtest |

**Examples:**

```bash
# Run backtest on all symbols
python -m crypto_trading_framework.cli backtest

# Backtest a specific symbol
python -m crypto_trading_framework.cli backtest -s BTC/USDT:USDT

# Use a custom config with an overlay
python -m crypto_trading_framework.cli backtest -c config/base.yaml -o config/overlays/crypto.yaml
```

---

### `live`

Start the live trading bot loop.

```bash
python -m crypto_trading_framework.cli live
```

**Options:**

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `config/base.yaml` | Path to base config YAML |
| `--overlay` | `-o` | `None` | Path to overlay config YAML |

**Examples:**

```bash
# Start live trading with default config
python -m crypto_trading_framework.cli live

# Start with IDX overlay
python -m crypto_trading_framework.cli live -o config/overlays/idx.yaml
```

---

### `cockpit`

Launch the Streamlit observability dashboard.

```bash
python -m crypto_trading_framework.cli cockpit
```

**Options:**

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `config/base.yaml` | Path to base config YAML |
| `--overlay` | `-o` | `None` | Path to overlay config YAML |

**Examples:**

```bash
# Launch cockpit with default config
python -m crypto_trading_framework.cli cockpit

# Launch with custom config
python -m crypto_trading_framework.cli cockpit -c config/base.yaml
```

The dashboard provides real-time monitoring of:
- OHLCV candles with indicators
- Model ensemble signals (LSTM/RF/XGBoost)
- Smart Money veto status
- Shadow trader performance
- Backtest results
- System health (TimescaleDB, Redis)

---

### `signals`

Generate and display trading signals summary.

```bash
python -m crypto_trading_framework.cli signals
```

**Options:**

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `config/base.yaml` | Path to base config YAML |
| `--overlay` | `-o` | `None` | Path to overlay config YAML |

**Examples:**

```bash
# Generate signals with default config
python -m crypto_trading_framework.cli signals
```

---

### `train`

Train models for all symbols.

```bash
python -m crypto_trading_framework.cli train
```

**Options:**

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `config/base.yaml` | Path to base config YAML |
| `--overlay` | `-o` | `None` | Path to overlay config YAML |

**Note:** Training is currently not fully implemented as a standalone command.
Model training is handled by the `AutomatedTradingBot` in live mode or via
the `backtest` command's training pipeline.

---

### `version`

Print version information.

```bash
python -m crypto_trading_framework.cli version
```

**Output example:**

```
Quantuis v5.0.0
```

---

## Global Options

All commands share the same `--config` and `--overlay` options.

### Config Loading

Quantuis uses a layered config system:

1. **Base config** (`config/base.yaml`) — shared defaults for all environments
2. **Overlay config** (`config/overlays/crypto.yaml` or `config/overlays/idx.yaml`) — environment-specific overrides
3. **Environment variables** (`.env` file) — secrets and sensitive values

The overlay config is deep-merged on top of the base config, and environment
variables are interpolated using `${VAR_NAME}` syntax.

**Example directory structure:**

```
config/
├── base.yaml              # Shared defaults
└── overlays/
    ├── crypto.yaml        # Crypto-specific overrides
    └── idx.yaml           # IDX-specific overrides
```

---

## Configuration Reference

### Key config paths in `config/base.yaml`

| Path | Description |
|---|---|
| `data.exchange_id` | Exchange ID (e.g., `okx`, `idx`) |
| `data.symbols` | List of trading symbols |
| `data.timeframes` | List of candle timeframes |
| `data.lookback` | Number of candles to fetch |
| `trading.dry_run` | Enable dry-run mode (no real orders) |
| `ml.feature_cols` | List of feature columns for ML pipeline |
| `model.type` | Model type (`lstm`, `ensemble`, etc.) |
| `signal.threshold` | Probability threshold for signal generation |
| `task_queue.max_concurrency` | Max concurrent training tasks |
| `paths.data_dir` | Data directory |
| `paths.models_dir` | Models directory |
| `paths.logs_dir` | Logs directory |

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |

---

## Examples — Common Workflows

### 1. Quick start with dry-run backtest

```bash
python -m crypto_trading_framework.cli backtest -s BTC/USDT:USDT
```

### 2. Live trading with IDX overlay

```bash
python -m crypto_trading_framework.cli live -o config/overlays/idx.yaml
```

### 3. Monitor signals in cockpit

```bash
python -m crypto_trading_framework.cli cockpit
```

### 4. Check version and config

```bash
python -m crypto_trading_framework.cli version
python -m crypto_trading_framework.cli signals
```

---

## See Also

- [AGENTS.md](../AGENTS.md) — Repository overview and architecture
- [config/base.yaml](../config/base.yaml) — Default configuration
- [config/overlays/crypto.yaml](../config/overlays/crypto.yaml) — Crypto overlay
- [config/overlays/idx.yaml](../config/overlays/idx.yaml) — IDX overlay