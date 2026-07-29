"""
Example script untuk menggunakan Velo data integration.
"""

import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_trading_framework.velo_data import VeloDataClient

load_dotenv()

VELO_API_KEY = os.getenv("VELO_API_KEY", "your_api_key_here")


def main():
    print("=== VELO DATA EXAMPLE ===\n")

    try:
        client = VeloDataClient(api_key=VELO_API_KEY)
    except Exception as e:
        print(f"Gagal inisialisasi Velo client: {e}")
        print("Pastikan VELO_API_KEY sudah diisi di .env atau environment variable")
        return

    print("1. Fetch available futures columns...")
    columns = client.get_futures_columns()
    print(f"   Total columns: {len(columns)}")
    print(f"   Sample columns: {columns[:10]}")

    print("\n2. Fetch CVD (Cumulative Volume Delta) for BTC/ETH...")
    cvd_df = client.get_cvd_data(
        exchanges=["binance", "okex-swap"],
        products=["BTCUSDT", "ETHUSDT"],
        hours=24,
        resolution="1m",
    )
    if cvd_df is not None:
        print(f"   Shape: {cvd_df.shape}")
        print(f"   Columns: {list(cvd_df.columns)}")
        print(cvd_df.head())

    print("\n3. Fetch funding rate...")
    funding_df = client.get_funding_rate(
        exchanges=["binance-futures", "okex-swap"],
        coins=["BTC", "ETH", "SOL"],
        hours=1,
        resolution="1m",
    )
    if funding_df is not None:
        print(f"   Shape: {funding_df.shape}")
        print(funding_df.head())

    print("\n4. Fetch open interest...")
    oi_df = client.get_open_interest(
        exchanges=["binance-futures", "okex-swap"],
        coins=["BTC", "ETH", "SOL"],
        hours=1,
        resolution="1m",
    )
    if oi_df is not None:
        print(f"   Shape: {oi_df.shape}")
        print(oi_df.head())

    print("\n5. Fetch futures basis...")
    basis_df = client.get_futures_basis(
        coins=["BTC", "ETH"],
        hours=24,
        resolution="1h",
    )
    if basis_df is not None:
        print(f"   Shape: {basis_df.shape}")
        print(basis_df.head())

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
