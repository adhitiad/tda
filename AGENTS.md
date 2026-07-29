# AGENTS.md

## Repo overview
- Single Python package `crypto_trading_framework/` plus a procedural orchestrator in `main.py`.
- Config-driven: `config.yaml` is validated by Pydantic v2 in `core/config_schema.py`. All new feature flags must be added to `AppConfig` there.
- Tests run with `pytest`; lint with `ruff`. No Makefile, no CI workflows present.

## Commands
```bash
python main.py            # full pipeline: ingest → indicators → train → backtest → signals
python main.py bot        # live trading bot loop
uvicorn main:app --host 0.0.0.0 --port 8000  # Quantuis FastAPI server (Fase 5)
pytest                    # run all tests (173 passing)
pytest tests/test_foo.py  # run single file
ruff check .              # lint
```

## Architecture gotchas
- **Polars-first**: DataFrames are Polars. Do not assume Pandas semantics. Known pitfall: `pl.Series.slice(start, length)`, not `slice(start, end)`.
- **Timezone-aware timestamps**: Tests must use `datetime(..., tzinfo=timezone.utc)`. Naive datetimes break Polars/Pandas conversions. `_serialize_timestamp` uses `strftime("%Y-%m-%dT%H:%M:%S%z")` (produces `+0000` without colon) for compatibility with `strptime("%z")`.
- **Lazy imports in `main.py`**: Several modules are imported inside functions to avoid circular deps. Follow that pattern for new cross-module imports.
- **SQLAlchemy sessions**: In tests using SQLite in-memory, `session.commit()` is required inside iterator loops or data won't persist.
- **Pydantic `extra="allow"`**: Every `BaseModel` in `config_schema.py` uses `extra="allow"`. Unknown YAML keys are tolerated; new keys still need a typed field for IDE support and validation.
- **Quantuis (Fase 5)**: `main.py` is now the FastAPI+Uvicorn entry point. Uses `lifespan` for model loading, background scheduler for 15-min coin monitoring loops, and a single `GET /api/v1/signal` endpoint with `search` query param. Continuous Learning from Fase 4 runs as a background task within the same process.

## Testing conventions
- Tests live in `tests/` and mirror module names (`test_<module>.py`).
- Async tests: use `pytest-asyncio` (already installed).
- Fixtures: defined per-module or per-class; no global `conftest.py` yet.
- No external services required for tests: DB uses in-memory SQLite, Redis is mocked, exchanges are mocked.
- Use `unittest.mock` / `MagicMock` for FastAPI and exchange clients.

## Style & lint
- `ruff` is the formatter/linter. Auto-fix with `ruff check --fix`.
- Pre-existing warnings: `BLE001` (blind-except) in `database.py` and `bot.py`; `DTZ001` (naive datetime) in a few places. Do not churn unrelated legacy warnings.
- Logging: use `get_logger("module_name")` from `core.logging`. Do not use `print()`.

## Key modules
| Module | Role |
|---|---|
| `main.py` | Entrypoint: FastAPI+Uvicorn server (Quantuis Fase 5), pipeline orchestration, and bot runner |
| `bot.py` | `AutomatedTradingBot` — training scheduler + monitoring/trade loop |
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

## Operational notes
- `requirements.txt` is the source of truth for dependencies (pinned versions). `pyproject.toml` is minimal and only used for build metadata.
- `INSTRUCTIONS.md` contains generic architecture/security guidance in Indonesian. Follow its data-validation and dependency-pinning principles, but treat executable code/config as the source of truth over prose.
- Default exchange is OKX (`okx`); symbols use CCXT format with `:USDT` suffix for swap markets.
- `dry_run: true` is the default trading mode.
- **Quantuis (Fase 5)**: FastAPI server runs on `0.0.0.0:8000`. Models auto-loaded from `models/` at startup via `lifespan`. Background scheduler monitors coins every 15 min. Single endpoint `GET /api/v1/signal` with optional `search` query param. Returns HTTP 200 with `{"status": "pending", "message": "Sinyal belum tersedia.\nQuantuis sedang menganalisis.\nHarap tunggu"}` when coin not yet processed.
