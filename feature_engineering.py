import logging
import numpy as np
import pandas as pd
import gc
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats

from indicator_util import IndicatorUtil
from market_regime_util import MarketRegimeUtil


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("FeatureEngineer")

        self.use_chunking = config.get("feature_engineering", "use_chunking", True)
        self.chunk_size = config.get("feature_engineering", "chunk_size", 2000)

        self.indicator_util = IndicatorUtil()
        self.market_regime_util = MarketRegimeUtil(config)
        self.indicator_util.market_regime_util = self.market_regime_util

        self.essential_features_config = config.get("feature_engineering", "essential_features", [])
        self.indicators_to_compute_config = config.get("feature_engineering", "indicators_to_compute", [])
        self.lag_periods = config.get("feature_engineering", "lag_periods", [1, 3, 5])

    def process_features(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if df_ohlcv.empty:
            self.logger.warning("Empty input dataframe for feature processing")
            return pd.DataFrame()

        df = df_ohlcv.copy()
        df.columns = [col.lower() for col in df.columns]

        if self.use_chunking and len(df) > self.chunk_size:
            final_df = self._process_data_in_chunks(df, chunk_size=self.chunk_size)
        else:
            final_df = self._process_single_chunk(df)

        if final_df.empty:
            self.logger.warning("Empty dataframe after initial processing")
            return pd.DataFrame()

        final_df = self._add_advanced_features(final_df)
        final_df = self._store_actual_prices(final_df)
        final_df = self._clean_dataframe(final_df)

        self.logger.info(f"Processed {len(final_df)} rows with {len(final_df.columns)} features")
        return final_df

    def _process_single_chunk(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        if df_chunk.empty: return pd.DataFrame()

        df_with_indicators = self.indicator_util.calculate_specific_indicators(df_chunk,
                                                                               self.indicators_to_compute_config)

        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume']:
            if col in df_chunk.columns:
                df_with_indicators[col] = df_chunk[col]

        return self._clean_dataframe(df_with_indicators)

    def _process_data_in_chunks(self, df_full: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
        self.logger.info(f"Processing data in chunks of size {chunk_size}")
        results = []
        overlap = max(250,
                      self.config.get("model", "sequence_length", 60) * 3)  # Increased overlap for longer indicators

        for i in range(0, len(df_full), chunk_size - overlap):
            chunk_start = i
            chunk_end = min(i + chunk_size, len(df_full))
            current_chunk = df_full.iloc[chunk_start:chunk_end].copy()

            try:
                processed_chunk = self._process_single_chunk(current_chunk)

                if not processed_chunk.empty:
                    slice_start = overlap if i > 0 else 0
                    # Ensure slice_end does not exceed processed_chunk length
                    actual_slice_end = len(processed_chunk) - overlap if chunk_end < len(df_full) else len(
                        processed_chunk)

                    if slice_start < actual_slice_end:
                        results.append(processed_chunk.iloc[slice_start:actual_slice_end])
                    elif i == 0 and actual_slice_end > 0:  # First chunk, no left overlap to remove
                        results.append(processed_chunk.iloc[:actual_slice_end])
                    # If slice_start >= actual_slice_end and not the first chunk, this chunk is too small after processing, skip.

                gc.collect()
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")

        if not results: return pd.DataFrame()

        combined_df = pd.concat(results, axis=0)
        combined_df = combined_df[
            ~combined_df.index.duplicated(keep='first')]  # Keep first to maintain order from earlier chunks
        combined_df.sort_index(inplace=True)
        return combined_df

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_adv = df.copy()

        if isinstance(df_adv.index, pd.DatetimeIndex):
            df_adv['hour_sin'] = np.sin(2 * np.pi * df_adv.index.hour / 24)
            df_adv['hour_cos'] = np.cos(2 * np.pi * df_adv.index.hour / 24)
            df_adv['day_of_week_sin'] = np.sin(2 * np.pi * df_adv.index.dayofweek / 7)
            df_adv['day_of_week_cos'] = np.cos(2 * np.pi * df_adv.index.dayofweek / 7)
            df_adv['day_of_month_sin'] = np.sin(
                2 * np.pi * (df_adv.index.day - 1) / (df_adv.index.days_in_month - 1 + 1e-9))
            df_adv['day_of_month_cos'] = np.cos(
                2 * np.pi * (df_adv.index.day - 1) / (df_adv.index.days_in_month - 1 + 1e-9))

        if len(df_adv) >= self.market_regime_util.lookback_period:
            regime_outputs = [self.market_regime_util.detect_regime(
                df_adv.iloc[max(0, i - self.market_regime_util.lookback_period):i + 1])
                # Ensure current row is included
                for i in range(len(df_adv))]

            df_adv['market_regime_type'] = [r['type'] for r in regime_outputs]
            df_adv['market_regime'] = [self._map_regime_to_numeric(r['type'], r['confidence']) for r in regime_outputs]
            df_adv['volatility_regime'] = [r['metrics'].get('atr_pct', 0.01) / 0.03 for r in
                                           regime_outputs]  # Use atr_pct
            df_adv['volatility_regime'] = np.clip(df_adv['volatility_regime'], 0, 1)
        else:
            df_adv['market_regime_type'] = 'neutral'
            df_adv['market_regime'] = 0.0
            df_adv['volatility_regime'] = 0.5

        candle_size = df_adv['high'] - df_adv['low']
        avg_candle_size = candle_size.rolling(window=20, min_periods=5).mean().fillna(candle_size.mean())
        df_adv['relative_candle_size'] = candle_size / (avg_candle_size + 1e-9)
        df_adv['candle_body_ratio'] = abs(df_adv['close'] - df_adv['open']) / (candle_size + 1e-9)

        if 'bb_upper_20' in df_adv.columns and 'bb_lower_20' in df_adv.columns:
            band_width = df_adv['bb_upper_20'] - df_adv['bb_lower_20']
            df_adv['bb_percent_b'] = (df_adv['close'] - df_adv['bb_lower_20']) / (band_width + 1e-9)
            df_adv['bb_percent_b'] = np.clip(df_adv['bb_percent_b'], 0, 1)

        high_20 = df_adv['high'].rolling(window=20, min_periods=5).max()
        low_20 = df_adv['low'].rolling(window=20, min_periods=5).min()
        range_diff = high_20 - low_20
        df_adv['range_position'] = (df_adv['close'] - low_20) / (range_diff + 1e-9)
        df_adv['range_position'] = np.clip(df_adv['range_position'], 0, 1)

        if 'adx_14' in df_adv.columns:
            df_adv['trend_strength'] = np.clip(df_adv['adx_14'] / 50.0, 0, 1)
        elif 'ema_9' in df_adv.columns and 'ema_50' in df_adv.columns:
            df_adv['trend_strength'] = np.clip(abs(df_adv['ema_9'] / df_adv['ema_50'] - 1) * 10, 0, 1)
        else:
            df_adv['trend_strength'] = 0.5

        if 'taker_buy_base_asset_volume' in df_adv.columns and 'volume' in df_adv.columns:
            buy_vol = df_adv['taker_buy_base_asset_volume']
            sell_vol = df_adv['volume'] - buy_vol
            df_adv['cumulative_delta_val'] = (buy_vol - sell_vol).fillna(
                0)  # Renamed to avoid conflict if 'cumulative_delta' is used for rolling sum
            df_adv['cumulative_delta'] = df_adv['cumulative_delta_val'].rolling(window=20, min_periods=5).sum()
            avg_vol_20 = df_adv['volume'].rolling(window=20, min_periods=5).mean().fillna(1)
            df_adv['cumulative_delta'] = df_adv['cumulative_delta'] / (avg_vol_20 + 1e-9)

        if 'close' in df_adv.columns and 'volume' in df_adv.columns:
            typical_price = (df_adv['high'] + df_adv['low'] + df_adv['close']) / 3
            rolling_vwap_window = 20
            df_adv['vwap'] = (typical_price * df_adv['volume']).rolling(window=rolling_vwap_window,
                                                                        min_periods=5).sum() / \
                             (df_adv['volume'].rolling(window=rolling_vwap_window, min_periods=5).sum() + 1e-9)
            df_adv['close_vwap_diff'] = (df_adv['close'] - df_adv['vwap']) / (df_adv['vwap'] + 1e-9)

        if 'rsi_14' in df_adv.columns:
            df_adv['cycle_phase'] = np.sin(2 * np.pi * (df_adv['rsi_14'] - 50) / 100)

        if 'atr_14' in df_adv.columns and 'close' in df_adv.columns:
            atr_norm = df_adv['atr_14'] / (df_adv['close'] + 1e-9)
            df_adv['vol_norm_close_change_5'] = df_adv['close'].pct_change(5).fillna(0) / (atr_norm + 1e-9)
            df_adv['vol_norm_momentum_10'] = df_adv['close'].pct_change(10).fillna(0) / (atr_norm + 1e-9)

        # Lagged Returns
        returns = df_adv['close'].pct_change().fillna(0)
        for lag in self.lag_periods:
            df_adv[f'return_lag_{lag}'] = returns.shift(lag).fillna(0)

        return df_adv

    def _map_regime_to_numeric(self, regime_type: str, confidence: float) -> float:
        mapping = {
            "uptrend": 0.8, "strong_uptrend": 1.0,
            "downtrend": -0.8, "strong_downtrend": -1.0,
            "ranging": 0.0, "tight_consolidation": 0.1,
            "volatile": 0.3, "volatile_consolidation": 0.2, "choppy_mixed": -0.2,
            "neutral": 0.0,
            # Ensure all types from MarketRegimeUtil are covered
            "moderate_uptrend": 0.7,
            "moderate_downtrend": -0.7,
        }
        legacy_map = self.config.get("market_regime", "legacy_regime_mapping", {})
        base_regime_type = legacy_map.get(regime_type, regime_type)

        return mapping.get(base_regime_type, 0.0) * confidence

    def _store_actual_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[f'actual_{col}'] = df[col]
        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # Skip non-numeric like market_regime_type
                continue

            # Fill NaNs: ffill then bfill for price-related, median for most indicators, 0 for volume/delta
            if col in ['open', 'high', 'low', 'close'] or col.startswith('actual_'):
                df_clean[col] = df_clean[col].ffill().bfill()
            elif 'volume' in col or 'turnover' in col or 'delta' in col or 'cmf' in col:  # CMF can be 0
                df_clean[col] = df_clean[col].fillna(0)
            elif 'rsi' in col or 'stoch' in col:  # RSI/Stoch can be 50
                df_clean[col] = df_clean[col].fillna(50)
            else:  # For other indicators, median then 0
                df_clean[col] = df_clean[col].fillna(df_clean[col].median()).fillna(0)

            # Final check for any remaining NaNs after ffill/bfill/median (e.g., if entire column was NaN)
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(0)

        df_clean.dropna(
            subset=[f'actual_{c}' for c in ['open', 'high', 'low', 'close'] if f'actual_{c}' in df_clean.columns],
            inplace=True)
        return df_clean