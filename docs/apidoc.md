# Quantuis API Documentation

Quantuis exposes a FastAPI HTTP server for signal retrieval and health
monitoring. The server is started via `uvicorn main:app` or the
`cockpit` CLI command.

## Server

- **Host**: `0.0.0.0`
- **Port**: `8000`
- **Title**: Quantuis
- **Version**: 5.0.0

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or via the CLI:

```bash
python -m crypto_trading_framework.cli cockpit
```

## Endpoints

### GET /api/v1/health

Returns the server health status and uptime.

**Response (200):**

```json
{
  "status": "ok",
  "uptime_seconds": 123.45
}
```

### GET /api/v1/signal

Returns the current trading signal for the monitored coin.

When the coin has not yet been processed by the background scheduler,
the endpoint returns a pending status.

**Response (200) — Pending:**

```json
{
  "status": "pending",
  "message": "Sinyal belum tersedia.\nQuantuis sedang menganalisis.\nHarap tunggu"
}
```

**Response (200) — Ready:**

```json
{
  "status": "ready",
  "signal": "BUY",
  "symbol": "BTC/USDT:USDT",
  "confidence": 0.78,
  "price": 123450.0,
  "take_profit": 127153.5,
  "stop_loss": 122182.5,
  "model": "ensemble",
  "timestamp": "2026-07-30T09:00:00+0000"
}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `search` | string | `None` | Filter signals by coin symbol |

## WebSocket

The WebSocket endpoint is configurable via `config/base.yaml` under
`websocket.enabled`. When enabled, the server streams real-time
market data and signal updates.

**Configuration:**

```yaml
websocket:
  enabled: true
  buffer_size: 100
  buffer_interval: 5.0
```

## Configuration

The API server is configured through `config/base.yaml`:

| Path | Type | Default | Description |
|---|---|---|---|
| `api.enabled` | bool | `false` | Enable the FastAPI server |
| `api.host` | string | `"127.0.0.1"` | Bind address |
| `api.port` | int | `8000` | Bind port |
| `api.api_key` | string | `""` | API key for authentication |
| `api.allowed_origins` | list | `["*"]` | CORS allowed origins |
| `websocket.enabled` | bool | `false` | Enable WebSocket streaming |
| `websocket.buffer_size` | int | `100` | WebSocket buffer size |
| `websocket.buffer_interval` | float | `5.0` | WebSocket buffer interval in seconds |

## Data Models

### SignalResponse

| Field | Type | Description |
|---|---|---|
| `status` | string | `"pending"` or `"ready"` |
| `signal` | string | `"BUY"`, `"SELL"`, or `"HOLD"` |
| `symbol` | string | Trading symbol in CCXT format |
| `confidence` | float | Model confidence score (0.0–1.0) |
| `price` | float | Current price |
| `take_profit` | float | Take-profit price level |
| `stop_loss` | float | Stop-loss price level |
| `model` | string | Model type used for signal generation |
| `timestamp` | string | ISO 8601 timestamp with timezone |

### HealthResponse

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` when healthy |
| `uptime_seconds` | float | Server uptime in seconds |

## Error Codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `500` | Internal server error |

## Background Tasks

The FastAPI server runs a background scheduler that monitors coins
every 15 minutes. Models are auto-loaded from `models/` at startup
via the `lifespan` event handler. Continuous Learning (Fase 4) runs
as a background task within the same process.

## Authentication

API key authentication is configurable via `api.api_key` in the
config. When set, requests must include the header:

```
X-API-Key: <your-api-key>
```