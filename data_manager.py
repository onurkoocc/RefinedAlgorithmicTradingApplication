import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, List, Any, Union

try:
    from binance.um_futures import UMFutures

    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False


class DataManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DataManager")

        self.symbol = config.get("data", "symbol", "BTCUSDT")
        self.csv_30m = config.get("data", "csv_30m")
        self.use_api = config.get("data", "use_api", False)
        self.extended_data = config.get("data", "fetch_extended_data", True)
        self.min_candles = config.get("data", "min_candles", 15000)

        self.client = None
        if BINANCE_AVAILABLE and self.use_api:
            api_key = os.environ.get("BINANCE_API_KEY", "")
            api_secret = os.environ.get("BINANCE_API_SECRET", "")
            self.client = UMFutures(key=api_key, secret=api_secret, timeout=30)

        self.data_cache = {}

    def fetch_all_data(self, live: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        self.logger.info("Fetching 30m data...")

        df_30m = self.fetch_30m_data(live=live, extended=self.extended_data, min_candles=self.min_candles)
        if df_30m.empty:
            self.logger.error("No 30m data retrieved; aborting.")
            return pd.DataFrame(), None

        return df_30m

    def fetch_30m_data(self, live: bool = False, extended: bool = True,
                       min_candles: int = 15000) -> pd.DataFrame:
        cache_key = f"30m_{extended}_{min_candles}"
        if not live and cache_key in self.data_cache:
            self.logger.info("Using cached 30m data")
            return self.data_cache[cache_key]

        if os.path.exists(self.csv_30m) and not live:
            self.logger.info(f"Loading 30m data from {self.csv_30m}")
            df = pd.read_csv(self.csv_30m, index_col='timestamp', parse_dates=True)
            if not extended or len(df) >= min_candles:
                self.data_cache[cache_key] = df
                return df
            self.logger.info(f"CSV contains {len(df)} candles but {min_candles} requested. Fetching more data.")

        if not self.client:
            self.logger.warning("No Binance API client available. Cannot fetch data.")
            return pd.DataFrame()

        self.logger.info(f"Fetching 30m data from Binance API {'(extended)' if extended else ''}")

        try:
            lookback_candles = min_candles if extended else 15000
            start_time = int((datetime.now().timestamp() - (lookback_candles * 30 * 60)) * 1000)

            all_klines = []
            current_start_time = start_time
            remaining_candles = lookback_candles
            max_retries = 5
            retry_delay = 1.0

            while remaining_candles > 0:
                max_limit = min(1000, remaining_candles)
                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    try:
                        klines = self.client.klines(
                            symbol=self.symbol,
                            interval=self.config.get("data", "interval_30m"),
                            limit=max_limit,
                            startTime=current_start_time
                        )

                        if not klines:
                            break

                        all_klines.extend(klines)
                        remaining_candles -= len(klines)

                        if len(klines) < max_limit:
                            break

                        current_start_time = int(klines[-1][0]) + 1
                        success = True
                        time.sleep(0.5)

                    except Exception as e:
                        retry_count += 1
                        if "rate limit" in str(e).lower() or "429" in str(e):
                            sleep_time = retry_delay * (2 ** (retry_count - 1))
                            self.logger.warning(f"Rate limit hit, waiting {sleep_time}s")
                            time.sleep(sleep_time)
                        else:
                            self.logger.warning(f"API error: {e}, retry {retry_count}/{max_retries}")
                            time.sleep(retry_delay)

                if not success:
                    break

            if not all_klines:
                self.logger.warning("No data received from Binance API")
                if os.path.exists(self.csv_30m):
                    df = pd.read_csv(self.csv_30m, index_col='timestamp', parse_dates=True)
                    self.data_cache[cache_key] = df
                    return df
                return pd.DataFrame()

            df = pd.DataFrame(
                all_klines,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ]
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # CHANGE HERE: Include taker volume columns in the numeric columns list
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

            df['turnover'] = df['quote_asset_volume']

            # CHANGE HERE: Include taker volume columns in the final DataFrame
            df = df[['open', 'high', 'low', 'close', 'volume', 'turnover',
                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]
            df.sort_index(inplace=True)

            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]

            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                self.logger.warning(f"Found {nan_count} NaN values in data, filling...")
                for col in df.columns:
                    if df[col].isna().any():
                        if col in ['open', 'high', 'low', 'close']:
                            df[col] = df[col].ffill().bfill()
                        else:
                            df[col] = df[col].fillna(0)

            df.to_csv(self.csv_30m)
            self.logger.info(f"Fetched {len(df)} 30m candles and saved to {self.csv_30m}")
            self.data_cache[cache_key] = df
            return df

        except Exception as e:
            self.logger.error(f"Error fetching 30m data: {e}")
            if os.path.exists(self.csv_30m):
                df = pd.read_csv(self.csv_30m, index_col='timestamp', parse_dates=True)
                self.data_cache[cache_key] = df
                return df
            return pd.DataFrame()

    def clear_cache(self) -> None:
        self.data_cache.clear()
        self.logger.info("Data cache cleared")

    def align_timeframes(self, df_source: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        if df_source.empty or len(target_index) == 0:
            return pd.DataFrame(index=target_index)

        aligned = pd.DataFrame(index=target_index)

        for col in df_source.select_dtypes(include=np.number).columns:
            aligned[col] = df_source[col].reindex(target_index, method='ffill')

        aligned.fillna(0, inplace=True)

        return aligned