# Velo Data Integration

Integrasi data tambahan dari [Velo API](https://docs.velo.xyz/api/python) untuk meningkatkan kualitas sinyal trading.

## Setup

1. **Install dependencies:**
```bash
pip install velodata tenacity
```

2. **Get API Key:**
   - Daftar di https://velo.xyz
   - Buat API key di dashboard
   - Copy API key

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env dan isi VELO_API_KEY
```

## Usage

### Basic Example
```bash
python examples/velo_example.py
```

### Integration with Trading Bot

Import dan gunakan `VeloDataClient` di bot:

```python
from crypto_trading_framework.velo_data import VeloDataClient

# Initialize
client = VeloDataClient()

# Fetch CVD data
cvd_df = client.get_cvd_data(
    exchanges=["binance", "okex-swap"],
    products=["BTCUSDT"],
    hours=24,
    resolution="1m"
)

# Fetch funding rate
funding_df = client.get_funding_rate(
    exchanges=["binance-futures"],
    coins=["BTC"],
    hours=1,
    resolution="1m"
)

# Save to CSV
client.save_to_csv(cvd_df, "data/velo_cvd.csv")
client.save_to_csv(funding_df, "data/velo_funding.csv")
```

## Available Data Types

| Data Type | Method | Description |
|-----------|--------|-------------|
| CVD | `get_cvd_data()` | Cumulative Volume Delta (buy/sell pressure) |
| Funding Rate | `get_funding_rate()` | OI-weighted funding rates |
| Open Interest | `get_open_interest()` | Coin open interest |
| Orderbook Depth | `get_orderbook_depth()` | Orderbook snapshots |
| Futures Basis | `get_futures_basis()` | 3M basis annualized |

## Rate Limiting

- Default delay: 1 second between requests
- Automatic retry with exponential backoff (max 3 attempts)
- Batch requests supported for large datasets

## Error Handling

- Network errors: automatic retry
- Invalid API key: clear error message
- Rate limit: automatic delay
- Empty data: returns None, check before using

## Data Format

All data is returned as pandas DataFrame with:
- `time`: datetime column (millisecond timestamps converted)
- Sorted by time ascending
- Ready for merging with OHLCV data

## Example Output

```
=== VELO DATA EXAMPLE ===

1. Fetch available futures columns...
   Total columns: 156
   Sample columns: ['open_price', 'high_price', 'low_price', 'close_price', 'volume']

2. Fetch CVD (Cumulative Volume Delta) for BTC/ETH...
   Shape: (1440, 5)
   Columns: ['time', 'exchange', 'product', 'buy_dollar_volume', 'sell_dollar_volume']

3. Fetch funding rate...
   Shape: (60, 5)
   Columns: ['time', 'exchange', 'coin', 'funding_rate', 'coin_open_interest_close']

=== DONE ===
```
