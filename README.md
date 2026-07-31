# Quantuis — Trading Data Analysis Framework

Quantuis is a config-driven, multi-asset trading framework for crypto and
Indonesian stock indices (IDX). It provides signal generation, model training,
backtesting, live trading, and real-time observability — all orchestrated
through a clean layered architecture.

## Features

- **Multi-asset support** — Crypto (BTC, ETH, etc.) and IDX (IHSG, LQ45, etc.)
- **Config-driven** — YAML base config + overlays + `.env` for secrets
- **ML pipeline** — LSTM, ensemble (LSTM/RF/XGBoost), with walk-forward validation
- **Backtesting** — Event-driven backtester with TP/SL, trailing stop, fees
- **Observability** — Streamlit cockpit, Prometheus metrics, structlog JSON logging
- **Smart Money VETO** — On-chain analysis to filter signals
- **Continuous Learning** — Feedback loop with golden memory and drift detection

## Quick Start

### Prerequisites

- Python 3.12+
- pip
- Git

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd tda

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
# Backtest
python -m crypto_trading_framework.cli backtest

# Live trading
python -m crypto_trading_framework.cli live

# Observability cockpit
python -m crypto_trading_framework.cli cockpit

# Generate signals
python -m crypto_trading_framework.cli signals

# Train models
python -m crypto_trading_framework.cli train

# Version
python -m crypto_trading_framework.cli version
```

Or via the thin wrapper:

```bash
python main.py backtest
python main.py live
python main.py cockpit
python main.py signals
python main.py train
python main.py version
```

## Commands Reference

| Command | Description |
|---|---|
| `backtest` | Run backtest on historical data |
| `live` | Start live trading bot |
| `cockpit` | Launch Streamlit observability dashboard |
| `signals` | Generate and display trading signals |
| `train` | Train models for all symbols |
| `version` | Print version information |

See [docs/cli.md](docs/cli.md) for full CLI documentation with options, examples, and configuration details.

## Project Structure

```
tda/
├── config/
│   ├── base.yaml              # Shared defaults
│   ├── settings.py            # Pydantic BaseSettings loader
│   └── overlays/
│       ├── crypto.yaml        # Crypto-specific overrides
│       └── idx.yaml           # IDX-specific overrides
├── crypto_trading_framework/
│   ├── __init__.py
│   ├── cli.py                 # Typer CLI entry point
│   ├── main.py                # Thin wrapper → cli.py
│   ├── config/
│   │   └── settings.py        # Pydantic BaseSettings (package-internal)
│   ├── core/
│   │   ├── bot.py             # AutomatedTradingBot
│   │   ├── config_schema.py   # Pydantic AppConfig models
│   │   ├── indicators.py      # Technical indicators
│   │   ├── logging.py         # Logger setup
│   │   └── kill_switch.py     # Emergency stop
│   ├── data/
│   │   └── ...                # Data ingestion layer
│   ├── tda_engine/
│   │   ├── __init__.py
│   │   └── cache.py           # TDAFeatureCache
│   ├── strategies/
│   │   └── __init__.py        # BaseStrategy ABC
│   ├── risk/
│   │   └── __init__.py
│   ├── execution/
│   │   └── __init__.py
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── provider.py        # ObservabilityDataProvider
│   │   └── cockpit.py         # Streamlit dashboard
│   ├── ml/
│   │   ├── backtest.py        # Event-driven backtester
│   │   ├── vectorized_backtest.py  # NumPy vectorized backtest
│   │   ├── ml_pipeline.py     # Feature prep + scaling
│   │   ├── model.py           # PyTorch model factories
│   │   ├── training.py        # Train/eval loop
│   │   ├── signals.py         # Signal generation
│   │   └── ...
│   ├── db/
│   └── ...
├── tests/
├── docs/
│   └── cli.md                 # CLI reference
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── mypy.ini
```

## Configuration

Quantuis uses a three-layer config system:

1. **`config/base.yaml`** — Shared defaults
2. **`config/overlays/crypto.yaml`** or **`config/overlays/idx.yaml`** — Environment-specific overrides
3. **`.env`** — Secrets and sensitive values (API keys, DB passwords)

Environment variables in YAML are interpolated with `${VAR_NAME}` or
`${VAR_NAME:-default}` syntax.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
make lint

# Auto-format
make format

# Run all checks
make check

# Run tests
pytest tests/ -v
```

## Docker

```bash
# Start full stack (app + Redis + TimescaleDB + Grafana)
make docker-up

# Stop
make docker-down
```

## License

See [LICENSE](LICENSE).

## Architecture

Quantuis follows a clean layered architecture:

```
config → data → features → tda_engine → models → strategy → risk → execution
```

- **Strategy Pattern** — Add new strategies without editing core
- **Pydantic v2** — Config validation with `extra="allow"` for extensibility
- **Polars-first** — All DataFrames are Polars for performance
- **Async I/O** — Data fetching and monitoring use asyncio
- **Celery** — Async inference via task queue

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `make check` before submitting
5. Open a pull request

## Requirements

- Python 3.12+
- pip
- Git

### Optional (for advanced features)

- `ccxt.pro` — Concurrent multi-symbol data fetching
- `vectorbt` — Vectorized backtesting engine
- `streamlit` + `plotly` + `kaleido` — Observability cockpit

## Support

For issues and questions, please open a GitHub issue.