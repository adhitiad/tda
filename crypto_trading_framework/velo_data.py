"""
Velo Data Integration Module
Mengambil data tambahan dari Velo API untuk meningkatkan kualitas sinyal trading.

Dokumentasi: https://docs.velo.xyz/api/python
"""

import logging
import os
import time
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("velo_data")

try:
    from velodata import lib as velo
    VELO_AVAILABLE = True
except ImportError:
    VELO_AVAILABLE = False
    logger.warning("velodata library tidak terinstall. Install dengan: pip install velodata")


class VeloDataClient:
    """Client untuk mengambil data dari Velo API."""

    def __init__(self, api_key: str | None = None):
        if not VELO_AVAILABLE:
            raise ImportError("velodata library tidak terinstall")

        self.api_key = api_key or os.getenv("VELO_API_KEY")
        if not self.api_key:
            raise ValueError("VELO_API_KEY harus diisi")

        self.client = velo.client(self.api_key)
        self.rate_limit_delay = 1.0
        self.last_request_time = 0.0

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(self, params: dict[str, Any]) -> pd.DataFrame | None:
        self._wait_for_rate_limit()
        try:
            df = self.client.get_rows(params)
            if df is not None:
                logger.info(f"[VELO] Request sukses: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"[VELO] Request error: {e}")
            raise

    def get_futures_columns(self) -> list[str]:
        try:
            cols = self.client.get_futures_columns()
            logger.info(f"[VELO] Available futures columns: {len(cols)}")
            return cols
        except Exception as e:
            logger.error(f"[VELO] Gagal fetch futures columns: {e}")
            return []

    def get_spot_columns(self) -> list[str]:
        try:
            cols = self.client.get_spot_columns()
            logger.info(f"[VELO] Available spot columns: {len(cols)}")
            return cols
        except Exception as e:
            logger.error(f"[VELO] Gagal fetch spot columns: {e}")
            return []

    def get_futures(self) -> list[dict]:
        try:
            futures = self.client.get_futures()
            logger.info(f"[VELO] Available futures: {len(futures)}")
            return futures
        except Exception as e:
            logger.error(f"[VELO] Gagal fetch futures list: {e}")
            return []

    def get_spot(self) -> list[dict]:
        try:
            spot = self.client.get_spot()
            logger.info(f"[VELO] Available spot: {len(spot)}")
            return spot
        except Exception as e:
            logger.error(f"[VELO] Gagal fetch spot list: {e}")
            return []

    def get_cvd_data(
        self,
        exchanges: list[str],
        products: list[str],
        columns: list[str] = None,
        hours: int = 24,
        resolution: str = "1m",
    ) -> pd.DataFrame | None:
        if columns is None:
            columns = ["buy_dollar_volume", "sell_dollar_volume"]

        end_ms = self.client.timestamp()
        begin_ms = end_ms - hours * 60 * 60 * 1000

        params = {
            "type": "spot",
            "columns": columns,
            "exchanges": exchanges,
            "products": products,
            "begin": begin_ms,
            "end": end_ms,
            "resolution": resolution,
        }

        try:
            df = self._make_request(params)
            if df is not None and not df.empty:
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df = df.sort_values("time")
                logger.info(f"[VELO] CVD data fetched: {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"[VELO] Gagal fetch CVD data: {e}")
        return None

    def get_funding_rate(
        self,
        exchanges: list[str],
        coins: list[str] = None,
        products: list[str] = None,
        hours: int = 1,
        resolution: str = "1m",
    ) -> pd.DataFrame | None:
        if coins is None and products is None:
            coins = ["BTC", "ETH", "SOL", "BNB", "XRP"]

        end_ms = self.client.timestamp()
        begin_ms = end_ms - hours * 60 * 60 * 1000

        params = {
            "type": "futures",
            "columns": ["funding_rate", "coin_open_interest_close"],
            "exchanges": exchanges,
            "begin": begin_ms,
            "end": end_ms,
            "resolution": resolution,
        }
        if coins:
            params["coins"] = coins
        if products:
            params["products"] = products

        try:
            df = self._make_request(params)
            if df is not None and not df.empty:
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df = df.sort_values("time")
                logger.info(f"[VELO] Funding rate fetched: {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"[VELO] Gagal fetch funding rate: {e}")
        return None

    def get_open_interest(
        self,
        exchanges: list[str],
        coins: list[str] = None,
        products: list[str] = None,
        hours: int = 1,
        resolution: str = "1m",
    ) -> pd.DataFrame | None:
        if coins is None and products is None:
            coins = ["BTC", "ETH", "SOL", "BNB", "XRP"]

        end_ms = self.client.timestamp()
        begin_ms = end_ms - hours * 60 * 60 * 1000

        params = {
            "type": "futures",
            "columns": ["coin_open_interest_close"],
            "exchanges": exchanges,
            "begin": begin_ms,
            "end": end_ms,
            "resolution": resolution,
        }
        if coins:
            params["coins"] = coins
        if products:
            params["products"] = products

        try:
            df = self._make_request(params)
            if df is not None and not df.empty:
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df = df.sort_values("time")
                logger.info(f"[VELO] Open interest fetched: {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"[VELO] Gagal fetch open interest: {e}")
        return None

    def get_orderbook_depth(
        self,
        exchange: str,
        product: str,
        hours: int = 24,
        resolution: str = "5m",
    ) -> pd.DataFrame | None:
        end_ms = self.client.timestamp()
        begin_ms = end_ms - hours * 60 * 60 * 1000

        params = {
            "exchange": exchange,
            "product": product,
            "begin": begin_ms,
            "end": end_ms,
            "resolution": resolution,
        }

        try:
            depth_data = []
            for df in self.client.depth(params):
                depth_data.append(df)
            if depth_data:
                result = pd.concat(depth_data, ignore_index=True)
                logger.info(f"[VELO] Orderbook depth fetched: {len(result)} rows")
                return result
        except Exception as e:
            logger.error(f"[VELO] Gagal fetch orderbook depth: {e}")
        return None

    def get_futures_basis(
        self,
        coins: list[str] = None,
        hours: int = 24,
        resolution: str = "1h",
    ) -> pd.DataFrame | None:
        if coins is None:
            coins = ["BTC", "ETH"]

        end_ms = self.client.timestamp()
        begin_ms = end_ms - hours * 60 * 60 * 1000

        params = {
            "type": "futures",
            "columns": ["3m_basis_ann"],
            "coins": coins,
            "begin": begin_ms,
            "end": end_ms,
            "resolution": resolution,
        }

        try:
            df = self._make_request(params)
            if df is not None and not df.empty:
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df = df.sort_values("time")
                logger.info(f"[VELO] Futures basis fetched: {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"[VELO] Gagal fetch futures basis: {e}")
        return None

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        try:
            df.to_csv(filename, index=False)
            logger.info(f"[VELO] Data saved to {filename}")
        except Exception as e:
            logger.error(f"[VELO] Gagal save CSV: {e}")

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        try:
            df.to_parquet(filename, index=False)
            logger.info(f"[VELO] Data saved to {filename}")
        except Exception as e:
            logger.error(f"[VELO] Gagal save Parquet: {e}")


def main():
    logging.basicConfig(level=logging.INFO)

    try:
        client = VeloDataClient()
    except (ImportError, ValueError) as e:
        logger.error(f"Gagal inisialisasi Velo client: {e}")
        return

    logger.info("=== VELO DATA INTEGRATION ===")

    logger.info("\n1. Fetching available futures...")
    futures = client.get_futures()
    if futures:
        print(f"\nSample futures: {futures[:3]}")
        btc_futures = [f for f in futures if "BTC" in f.get("product", "")]
        print(f"\nBTC futures: {btc_futures[:3]}")

    logger.info("\n2. Fetching CVD data...")
    cvd_df = client.get_cvd_data(
        exchanges=["binance", "okex-swap"],
        products=["BTCUSDT", "ETHUSDT"],
        hours=24,
        resolution="1m",
    )
    if cvd_df is not None:
        print(f"\nCVD data shape: {cvd_df.shape}")
        print(cvd_df.head())
        client.save_to_csv(cvd_df, "velo_cvd_data.csv")

    logger.info("\n3. Fetching funding rate...")
    funding_df = client.get_funding_rate(
        exchanges=["binance-futures", "okex-swap"],
        hours=1,
        resolution="1m",
    )
    if funding_df is not None:
        print(f"\nFunding rate shape: {funding_df.shape}")
        print(funding_df.head())
        client.save_to_csv(funding_df, "velo_funding_rate.csv")

    logger.info("\n4. Fetching open interest...")
    oi_df = client.get_open_interest(
        exchanges=["binance-futures", "okex-swap"],
        hours=1,
        resolution="1m",
    )
    if oi_df is not None:
        print(f"\nOpen interest shape: {oi_df.shape}")
        print(oi_df.head())
        client.save_to_csv(oi_df, "velo_open_interest.csv")

    logger.info("\n5. Fetching futures basis...")
    basis_df = client.get_futures_basis(hours=24, resolution="1h")
    if basis_df is not None:
        print(f"\nFutures basis shape: {basis_df.shape}")
        print(basis_df.head())
        client.save_to_csv(basis_df, "velo_futures_basis.csv")

    logger.info("\n=== VELO DATA INTEGRATION SELESAI ===")


if __name__ == "__main__":
    main()
