# AGENTS.md

## Repo overview
- Single Python package `crypto_trading_framework/` plus a CLI entry point in `cli.py` (thin wrapper via `main.py`).
- Config-driven: `config/base.yaml` + overlays validated by Pydantic v2 in `core/config_schema.py`. All new feature flags must be added to `AppConfig` there.
- Settings loading via `config/settings.py` using `pydantic-settings` (BaseSettings) with YAML + .env support.
- Tests run with `pytest`; lint with `ruff`. No Makefile previously, now has Makefile.

## Commands
```bash
python -m crypto_trading_framework.cli backtest   # run backtest
python -m crypto_trading_framework.cli live       # live trading bot loop
python -m crypto_trading_framework.cli cockpit    # launch Streamlit observability dashboard
python -m crypto_trading_framework.cli signals    # generate trading signals
python -m crypto_trading_framework.cli train      # train models
python main.py            # alias for cli.py (thin wrapper)
make backtest             # via Makefile
make lint                 # ruff check
make format               # ruff format + black
make check                # lint + format check + mypy
pytest                    # run all tests
```

## Architecture gotchas
- **Polars-first**: DataFrames are Polars. Do not assume Pandas semantics. Known pitfall: `pl.Series.slice(start, length)`, not `slice(start, end)`.
- **Timezone-aware timestamps**: Tests must use `datetime(..., tzinfo=timezone.utc)`. Naive datetimes break Polars/Pandas conversions. `_serialize_timestamp` uses `strftime("%Y-%m-%dT%H:%M:%S%z")` (produces `+0000` without colon) for compatibility with `strptime("%z")`.
- **Lazy imports in `cli.py`**: Several modules are imported inside functions to avoid circular deps. Follow that pattern for new cross-module imports.
- **SQLAlchemy sessions**: In tests using SQLite in-memory, `session.commit()` is required inside iterator loops or data won't persist.
- **Pydantic `extra="allow"`**: Every `BaseModel` in `config_schema.py` uses `extra="allow"`. Unknown YAML keys are tolerated; new keys still need a typed field for IDE support and validation.
- **Quantuis (Fase 5)**: `cli.py` is the Typer CLI entry point. FastAPI server runs on `0.0.0.0:8000` via `uvicorn main:app`. Models auto-loaded from `models/` at startup via `lifespan`. Background scheduler monitors coins every 15 min. Single endpoint `GET /api/v1/signal` with optional `search` query param. Continuous Learning from Fase 4 runs as a background task within the same process.

## Clean layers
| Layer | Path | Role |
|---|---|---|
| config | `crypto_trading_framework/config/` | Pydantic BaseSettings loader, YAML base + overlays |
| data | `crypto_trading_framework/data/` | CCXT + yfinance fetching; converts to Polars |
| tda_engine | `crypto_trading_framework/tda_engine/` | Topological Data Analysis (ripser wrapper + cache) |
| strategies | `crypto_trading_framework/strategies/` | Strategy Pattern base + concrete strategies |
| risk | `crypto_trading_framework/risk/` | Risk management (position sizing, drawdown limits) |
| execution | `crypto_trading_framework/execution/` | Order execution (exchange API, dry_run mode) |
| observability | `crypto_trading_framework/observability/` | Data provider + Streamlit cockpit + Prometheus metrics |
| core | `crypto_trading_framework/core/` | Config schema, logging, bot, indicators, kill switch |
| ml | `crypto_trading_framework/ml/` | ML pipeline, models, training, inference, drift detection |
| db | `crypto_trading_framework/db/` | TimescaleDB, Redis cache, ledger |
| cli | `crypto_trading_framework/cli.py` | Typer CLI entry point |

## Testing conventions
- Tests live in `tests/` and mirror module names (`test_<module>.py`).
- Async tests: use `pytest-asyncio` (already installed).
- Fixtures: defined per-module or per-class; no global `conftest.py` yet.
- No external services required for tests: DB uses in-memory SQLite, Redis is mocked, exchanges are mocked.
- Use `unittest.mock` / `MagicMock` for FastAPI and exchange clients.

## Style & lint
- `ruff` is the formatter/linter. Auto-fix with `ruff check --fix`.
- `black` for formatting (added to pre-commit).
- `mypy` for type checking (strict mode).
- Pre-existing warnings: `BLE001` (blind-except) in `database.py` and `bot.py`; `DTZ001` (naive datetime) in a few places. Do not churn unrelated legacy warnings.
- Logging: use `get_logger("module_name")` from `core.logging`. Do not use `print()`.

## Key modules
| Module | Role |
|---|---|
| `cli.py` | Typer CLI entry point: backtest, live, cockpit, signals, train |
| `main.py` | Thin wrapper delegating to `cli.py` |
| `bot.py` | `AutomatedTradingBot` — training scheduler + monitoring/trade loop |
| `config/settings.py` | Pydantic BaseSettings loader with YAML base + overlay + .env support |
| `data_ingestion.py` | CCXT + yfinance fetching; converts to Polars |
| `indicators.py` | Technical indicators (RSI, MACD, Bollinger, ATR, etc.) |
| `ml_pipeline.py` | Feature prep, scaling, sequence creation, train/test split |
| `model.py` | PyTorch model factories (LSTM, GRU, Attention LSTM, TFT) |
| `training.py` | Train/eval loop with class weights and early stopping |
| `backtest.py` | Event-driven backtester with TP/SL, trailing stop, fees |
| `feature_store.py` | TimescaleDB + Redis cache for computed features |
| `drift_detector.py` | KS test + feature z-score drift detection |
| `walk_forward.py` | Walk-forward validation with expanding/rolling windows |
| `ml/continuous_learning.py` | Fase 4: FeedbackLoopEngine, GoldenMemoryManager, SignalBuffer, ModelManager |
| `core/config_schema.py` | All Pydantic config models |
| `observability/provider.py` | `ObservabilityDataProvider` — aggregates data from DB, Redis, model registry |
| `observability/cockpit.py` | Streamlit dashboard for real-time monitoring |

## Operational notes
- `pyproject.toml` is the source of truth for dependencies (pinned versions + new deps). `requirements.txt` has been removed.
- `INSTRUCTIONS.md` contains generic architecture/security guidance in Indonesian. Follow its data-validation and dependency-pinning principles, but treat executable code/config as the source of truth over prose.
- Default exchange is OKX (`okx`); symbols use CCXT format with `:USDT` suffix for swap markets.
- `dry_run: true` is the default trading mode.
- **Config overlays**: `config/base.yaml` for shared defaults, `config/overlays/crypto.yaml` for crypto-specific overrides, `config/overlays/idx.yaml` for IDX-specific overrides. Use `Settings(config_path=..., overlay_path=...)` to load.
- **Quantuis (Fase 5)**: FastAPI server runs on `0.0.0.0:8000`. Models auto-loaded from `models/` at startup via `lifespan`. Background scheduler monitors coins every 15 min. Single endpoint `GET /api/v1/signal` with optional `search` query param. Returns HTTP 200 with `{"status": "pending", "message": "Sinyal belum tersedia.\nQuantuis sedang menganalisis.\nHarap tunggu"}` when coin not yet processed.
- **Docker**: `docker-compose up` starts app + Redis + TimescaleDB + Grafana.