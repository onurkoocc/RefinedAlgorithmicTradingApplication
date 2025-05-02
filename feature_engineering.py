import logging
import numpy as np
import pandas as pd
import gc
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats

from indicator_util import IndicatorUtil


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("FeatureEngineer")

        self.use_chunking = config.get("feature_engineering", "use_chunking", True)
        self.chunk_size = config.get("feature_engineering", "chunk_size", 2000)
        self.correlation_threshold = config.get("feature_engineering", "correlation_threshold", 0.9)

        self.indicator_util = IndicatorUtil()

        self.indicators_to_compute = [
            "ema_9", "ema_21", "ema_50", "sma_200",
            "rsi_14", "bb_middle_20", "bb_upper_20", "bb_lower_20", "bb_width_20",
            "atr_14",
            "obv", "cmf_20",
            "adx_14", "plus_di_14", "minus_di_14",
            "macd_12_26", "macd_signal_12_26_9", "macd_histogram_12_26_9",
        ]

        self.essential_features = [
            # Core price data
            'open', 'high', 'low', 'close', 'volume',

            # Volume dynamics
            'taker_buy_base_asset_volume', 'cumulative_delta', 'volume_imbalance_ratio',
            'volume_price_momentum',

            # Trend indicators
            'ema_9', 'ema_21', 'ema_50', 'sma_200',
            'adx_14', 'plus_di_14', 'minus_di_14',
            'trend_strength', 'ma_cross_velocity',

            # Momentum oscillators
            'rsi_14', 'rsi_roc_3', 'macd_histogram_12_26_9',

            # Volatility metrics
            'atr_14', 'bb_width_20', 'volatility_regime',

            # Market context
            'market_regime', 'mean_reversion_signal', 'price_impact_ratio',

            # Support/resistance
            'bb_percent_b', 'range_position', 'pullback_strength',

            # Time-based patterns
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'cycle_phase', 'cycle_position',

            # Price action patterns
            'relative_candle_size', 'candle_body_ratio', 'gap',

            # Order flow
            'spread_pct', 'close_vwap_diff',

            # Adaptive volatility features
            'vol_norm_close_change', 'vol_norm_momentum'
        ]

        self.feature_stats = {}

    def process_features(self, df_30m: pd.DataFrame) -> pd.DataFrame:
        if df_30m.empty:
            self.logger.warning("Empty input dataframe for feature processing")
            return pd.DataFrame()

        df_30m = df_30m.copy()
        df_30m.columns = [col.lower() for col in df_30m.columns]

        self.logger.info(f"Available columns in input data: {df_30m.columns.tolist()}")

        if self.use_chunking and len(df_30m) > self.chunk_size:
            final_df = self._process_data_in_chunks(df_30m, chunk_size=self.chunk_size)
        else:
            final_df = self._process_data_combined(df_30m)

        if final_df.empty:
            self.logger.warning("Empty dataframe after initial processing")
            return pd.DataFrame()

        final_df = self._clean_dataframe(final_df)
        final_df = self.compute_advanced_features(final_df)
        final_df = self._store_actual_prices(final_df)
        final_df = self._standardize_column_names(final_df)

        self.logger.info(f"Processed {len(final_df)} rows with {len(final_df.columns)} features")
        return final_df

    def compute_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['market_regime'] = df.apply(
            lambda row: self._map_market_phase_to_regime(self.indicator_util.detect_market_phase(row.to_frame().T)),
            axis=1)
        df['volatility_regime'] = df.apply(lambda row: self.indicator_util.detect_volatility_regime(row.to_frame().T),
                                           axis=1)

        regime_chunks = self._identify_regime_chunks(df['market_regime'])

        for regime_type, indices in regime_chunks.items():
            if not indices:
                continue

            regime_df = df.iloc[indices].copy()

            if regime_type == 'bullish':
                regime_df = self._add_bullish_regime_features(regime_df)
            elif regime_type == 'bearish':
                regime_df = self._add_bearish_regime_features(regime_df)
            else:
                regime_df = self._add_ranging_regime_features(regime_df)

            for col in regime_df.columns:
                if col not in df.columns:
                    df[col] = np.nan
                df.iloc[indices, df.columns.get_loc(col)] = regime_df[col].values

        df = self._add_order_flow_features(df)
        df = self._add_nonlinear_interactions(df)
        df = self._add_adaptive_volatility_features(df)

        df = self._clean_dataframe(df)

        return df

    def _map_market_phase_to_regime(self, market_phase):
        if market_phase == 'uptrend':
            return 0.8
        elif market_phase == 'downtrend':
            return -0.8
        elif market_phase == 'ranging_at_resistance':
            return 0.3
        elif market_phase == 'ranging_at_support':
            return -0.3
        else:
            return 0.0

    def _identify_regime_chunks(self, market_regime):
        chunks = {
            'bullish': [],
            'bearish': [],
            'neutral': []
        }

        for i, value in enumerate(market_regime):
            if value > 0.2:
                chunks['bullish'].append(i)
            elif value < -0.2:
                chunks['bearish'].append(i)
            else:
                chunks['neutral'].append(i)

        return chunks

    def _add_bullish_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'rsi_14' in df.columns:
            df['rsi_bullish_bias'] = np.where(df['rsi_14'] > 50,
                                              (df['rsi_14'] - 50) / 50,
                                              (df['rsi_14'] - 50) / 100)
            df['rsi_roc_3'] = df['rsi_14'].pct_change(3) * 100

        if 'close' in df.columns and 'ema_21' in df.columns:
            df['pullback_strength'] = np.clip((df['ema_21'] - df['close']) / df['ema_21'], -0.05, 0.05)
            df['pullback_opportunity'] = np.where(
                (df['close'] > df['ema_21']) & (df['pullback_strength'] > 0.001),
                df['pullback_strength'] * 10, 0)

        if 'ema_9' in df.columns and 'ema_50' in df.columns:
            df['bull_trend_strength'] = np.clip((df['ema_9'] / df['ema_50'] - 1) * 100, 0, 10)

        if 'volume' in df.columns and 'close' in df.columns:
            close_change = df['close'].pct_change()
            up_volume = df['volume'] * (close_change > 0)
            down_volume = df['volume'] * (close_change < 0)
            df['up_down_vol_ratio'] = up_volume.rolling(10).mean() / (down_volume.rolling(10).mean() + 1e-10)

        return df

    def _add_bearish_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'rsi_14' in df.columns:
            df['rsi_bearish_bias'] = np.where(df['rsi_14'] < 50,
                                              (50 - df['rsi_14']) / 50,
                                              (50 - df['rsi_14']) / 100)
            df['rsi_bearish_momentum'] = -df['rsi_14'].diff(3) * (df['rsi_14'] < 50)

        if 'close' in df.columns and 'ema_21' in df.columns:
            df['resistance_strength'] = np.clip((df['close'] - df['ema_21']) / df['ema_21'], -0.05, 0.05)
            df['resistance_opportunity'] = np.where(
                (df['close'] < df['ema_21']) & (df['resistance_strength'] > 0.001),
                df['resistance_strength'] * 10, 0)

        if 'ema_9' in df.columns and 'ema_50' in df.columns:
            df['bear_trend_strength'] = np.clip((1 - df['ema_9'] / df['ema_50']) * 100, 0, 10)

        if 'rsi_14' in df.columns and 'close' in df.columns:
            df['oversold_condition'] = np.where(df['rsi_14'] < 30, 1, 0)
            df['oversold_exit'] = df['oversold_condition'].rolling(5).sum() >= 2
            df['bounce_signal'] = np.where(
                df['oversold_exit'] & (df['rsi_14'] > df['rsi_14'].shift(1)), 1, 0)

        return df

    def _add_ranging_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'close' in df.columns and 'bb_upper_20' in df.columns and 'bb_lower_20' in df.columns:
            bb_range = df['bb_upper_20'] - df['bb_lower_20']
            valid_range = bb_range > 0
            df['bb_percent_b'] = np.nan
            if any(valid_range):
                df.loc[valid_range, 'bb_percent_b'] = (df.loc[valid_range, 'close'] - df.loc[
                    valid_range, 'bb_lower_20']) / bb_range[valid_range]
            df['bb_percent_b'] = df['bb_percent_b'].fillna(0.5)

            df['mean_reversion_signal'] = np.where(
                df['bb_percent_b'] > 0.8, -(df['bb_percent_b'] - 0.8) * 5,
                np.where(df['bb_percent_b'] < 0.2, (0.2 - df['bb_percent_b']) * 5, 0))

        if 'high' in df.columns and 'low' in df.columns:
            rolling_high = df['high'].rolling(20).max()
            rolling_low = df['low'].rolling(20).min()
            df['range_width'] = (rolling_high - rolling_low) / rolling_low

            if 'close' in df.columns:
                df['range_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)

            high_std = df['high'].rolling(10).std() / df['high']
            low_std = df['low'].rolling(10).std() / df['low']
            df['consolidation_intensity'] = 1 - (high_std + low_std) / 2 * 100

        if 'rsi_14' in df.columns:
            df['range_rsi_signal'] = np.where(
                df['rsi_14'] > 65, -1 * (df['rsi_14'] - 65) / 35,
                np.where(df['rsi_14'] < 35, 1 * (35 - df['rsi_14']) / 35, 0))

        if 'close' in df.columns and len(df) > 20:
            df['upper_range'] = df['high'].rolling(20).max()
            df['lower_range'] = df['low'].rolling(20).min()
            df['breakout_signal'] = np.where(
                df['close'] > df['upper_range'].shift(1), 1,
                np.where(df['close'] < df['lower_range'].shift(1), -1, 0))

        return df

    def _add_nonlinear_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'rsi_14' in df.columns and 'macd_histogram_12_26_9' in df.columns:
            df['rsi_macd_interaction'] = df['rsi_14'] * df['macd_histogram_12_26_9'] / 100
            rsi_centered = df['rsi_14'] - 50
            df['rsi_nonlinear'] = np.sign(rsi_centered) * (rsi_centered ** 2 / 50)
            if len(df) > 3:
                df['macd_acceleration'] = df['macd_histogram_12_26_9'].diff().diff()

        if 'atr_14' in df.columns and 'close' in df.columns:
            df['vol_norm_close_change'] = df['close'].pct_change(5) / (df['atr_14'] / df['close'])
            df['vol_norm_momentum'] = df['close'].pct_change(10) / (df['atr_14'] / df['close'])
            log_return = np.log(df['close'] / df['close'].shift(1))
            df['vol_adjusted_log_return'] = log_return / df['atr_14'].pct_change(20).rolling(10).std()

        if 'ema_21' in df.columns and 'ema_50' in df.columns:
            df['ma_spread'] = (df['ema_21'] / df['ema_50'] - 1) * 100
            df['ma_spread_z'] = (df['ma_spread'] - df['ma_spread'].rolling(50).mean()) / df['ma_spread'].rolling(
                50).std()
            df['ma_cross_velocity'] = df['ma_spread'].diff(3)

        if 'close' in df.columns and len(df) > 50:
            def hurst_exponent(series, max_lag=20):
                lags = range(2, max_lag)
                tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
                return np.polyfit(np.log(lags), np.log(tau), 1)[0]

            window = 50
            df['price_complexity'] = np.nan
            for i in range(window, len(df)):
                df.loc[df.index[i], 'price_complexity'] = hurst_exponent(df['close'].values[i - window:i])
            df['price_complexity'] = df['price_complexity'].fillna(0.5)

        if 'volume' in df.columns and 'close' in df.columns:
            df['volume_price_momentum'] = df['close'].pct_change(5) * (df['volume'] / df['volume'].rolling(20).mean())
            df['volume_climax'] = np.where(
                (df['volume'] > df['volume'].rolling(20).mean() * 2) &
                (abs(df['close'].pct_change()) > df['close'].pct_change().rolling(20).std() * 2),
                np.sign(df['close'].pct_change()) * -1, 0)

        return df

    def _add_adaptive_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'volatility_regime' not in df.columns:
            df['volatility_regime'] = 0.5

        vol_regime = df['volatility_regime'].values
        high_vol_mask = vol_regime > 0.7
        low_vol_mask = vol_regime < 0.3

        if 'close' in df.columns and 'atr_14' in df.columns:
            if np.any(high_vol_mask):
                high_vol_indices = np.where(high_vol_mask)[0]
                if len(high_vol_indices) > 20:
                    close_high_vol = df.loc[df.index[high_vol_indices], 'close']
                    rolling_mean = close_high_vol.rolling(window=20).mean()
                    rolling_std = close_high_vol.rolling(window=20).std()
                    df.loc[df.index[high_vol_indices], 'high_vol_upper_band'] = rolling_mean + (rolling_std * 3)
                    df.loc[df.index[high_vol_indices], 'high_vol_lower_band'] = rolling_mean - (rolling_std * 3)
                    df.loc[df.index[high_vol_indices], 'high_vol_bandwidth'] = (rolling_std * 6) / rolling_mean

            if np.any(low_vol_mask):
                low_vol_indices = np.where(low_vol_mask)[0]
                if len(low_vol_indices) > 20:
                    close_low_vol = df.loc[df.index[low_vol_indices], 'close']
                    rolling_mean = close_low_vol.rolling(window=20).mean()
                    rolling_std = close_low_vol.rolling(window=20).std()
                    df.loc[df.index[low_vol_indices], 'low_vol_upper_band'] = rolling_mean + (rolling_std * 1.5)
                    df.loc[df.index[low_vol_indices], 'low_vol_lower_band'] = rolling_mean - (rolling_std * 1.5)
                    df.loc[df.index[low_vol_indices], 'low_vol_bandwidth'] = (rolling_std * 3) / rolling_mean
                    df.loc[df.index[low_vol_indices], 'mean_reversion_score'] = (rolling_mean - close_low_vol) / (
                                rolling_std + 1e-10)

        if len(df) > 5:
            df['vol_regime_change'] = df['volatility_regime'].diff(5)
            df['vol_expanding'] = np.where(df['vol_regime_change'] > 0.15, 1, 0)
            df['vol_contracting'] = np.where(df['vol_regime_change'] < -0.15, 1, 0)

        return df

    def _add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            return df

        df = df.copy()

        if 'taker_buy_base_asset_volume' in df.columns:
            df['buy_sell_ratio'] = df['taker_buy_base_asset_volume'] / df['volume'].replace(0, np.nan)
            df['buy_sell_ratio'].fillna(0.5, inplace=True)
            delta = (2 * df['buy_sell_ratio'] - 1) * df['volume']
            df['cumulative_delta'] = delta.rolling(window=20).sum()
            df['cumulative_delta'].fillna(0, inplace=True)
            avg_volume = df['volume'].rolling(window=20).mean()
            df['cumulative_delta'] = df['cumulative_delta'] / avg_volume.replace(0, 1)
        else:
            price_change = df['close'] - df['open']
            volume_delta = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']
            df['cumulative_delta'] = volume_delta.rolling(window=20).sum()
            df['cumulative_delta'].fillna(0, inplace=True)
            avg_volume = df['volume'].rolling(window=20).mean()
            df['cumulative_delta'] = df['cumulative_delta'] / avg_volume.replace(0, 1)
            df['buy_sell_ratio'] = 0.5 + (price_change / (df['high'] - df['low']).replace(0, 1) * 0.25)

        return df

    def _process_data_in_chunks(self, df_30m: pd.DataFrame, chunk_size: int = 2000) -> pd.DataFrame:
        self.logger.info(f"Processing data in chunks of size {chunk_size}")
        results = []
        overlap = 100

        for i in range(0, len(df_30m), chunk_size - overlap):
            chunk_start = i
            chunk_end = min(i + chunk_size, len(df_30m))
            chunk_30m = df_30m.iloc[chunk_start:chunk_end].copy()

            try:
                chunk_features = self._process_data_combined(chunk_30m)

                if not chunk_features.empty:
                    if i > 0 and chunk_end < len(df_30m):
                        keep_start = overlap // 2
                        keep_end = len(chunk_features) - overlap // 2
                        chunk_features = chunk_features.iloc[keep_start:keep_end]
                    elif i > 0:
                        keep_start = overlap // 2
                        chunk_features = chunk_features.iloc[keep_start:]
                    elif chunk_end < len(df_30m):
                        keep_end = len(chunk_features) - overlap // 2
                        chunk_features = chunk_features.iloc[:keep_end]

                    results.append(chunk_features)
                    self.logger.debug(f"Processed chunk {chunk_start}-{chunk_end}: {len(chunk_features)} rows")

                del chunk_30m
                if 'chunk_features' in locals():
                    del chunk_features
                gc.collect()

            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        if not results:
            return pd.DataFrame()

        try:
            combined = pd.concat(results, axis=0)
            combined = combined[~combined.index.duplicated(keep='first')]
            combined.sort_index(inplace=True)
            del results
            gc.collect()
            return combined

        except Exception as e:
            self.logger.error(f"Error combining chunks: {e}")
            return pd.DataFrame()

    def _process_data_combined(self, df_30m: pd.DataFrame) -> pd.DataFrame:
        if df_30m.empty:
            return pd.DataFrame()

        df_30m = df_30m.copy()
        df_30m.columns = [col.lower() for col in df_30m.columns]
        df_30m = self._clean_dataframe(df_30m)

        if not isinstance(df_30m.index, pd.DatetimeIndex):
            try:
                df_30m.index = pd.to_datetime(df_30m.index)
            except:
                self.logger.warning("Failed to convert index to datetime")

        feat_30m = self.indicator_util.calculate_all_indicators(df_30m)

        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in df_30m.columns:
                feat_30m[col] = df_30m[col]

        taker_columns = [col for col in df_30m.columns if 'taker' in col.lower()]
        for col in taker_columns:
            feat_30m[col] = df_30m[col]

        feat_30m = self._clean_dataframe(feat_30m)
        feat_30m.dropna(subset=['close'], inplace=True)

        if feat_30m.empty:
            self.logger.warning("No data after processing")
            return pd.DataFrame()

        feat_30m = self._clean_dataframe(feat_30m)
        self._update_feature_stats(feat_30m)

        return feat_30m

    def _store_actual_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[f'actual_{col}'] = df[col]
        return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df_standard = df.copy()
        column_mapping = {}
        added_columns = set()

        for col in df.columns:
            if col.startswith('m30_'):
                base_name = col[4:]
                if base_name in self.essential_features and base_name not in df.columns:
                    column_mapping[col] = base_name
                    added_columns.add(base_name)

        for old_col, new_col in column_mapping.items():
            df_standard[new_col] = df[old_col]

        for feature in self.essential_features:
            if feature not in df_standard.columns and feature not in added_columns:
                prefixed = f'm30_{feature}'
                if prefixed in df_standard.columns:
                    df_standard[feature] = df_standard[prefixed]
                    added_columns.add(feature)

        return df_standard

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df_clean.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            if col.startswith('actual_'):
                continue

            if col in ['open', 'high', 'low', 'close']:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            elif col == 'volume' or 'volume' in col.lower():
                df_clean[col] = df_clean[col].fillna(0)
            elif 'rsi' in col.lower():
                df_clean[col] = df_clean[col].fillna(50)
            elif 'macd' in col.lower():
                df_clean[col] = df_clean[col].fillna(0)
            elif 'willr' in col.lower():
                df_clean[col] = df_clean[col].fillna(-50)
            elif 'mfi' in col.lower():
                df_clean[col] = df_clean[col].fillna(50)
            elif 'ratio' in col.lower():
                df_clean[col] = df_clean[col].fillna(0.5)
            elif 'regime' in col.lower():
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna(method='ffill', limit=5)
                median_val = df_clean[col].median()
                if pd.notna(median_val):
                    df_clean[col] = df_clean[col].fillna(median_val)
                else:
                    df_clean[col] = df_clean[col].fillna(0)

        return df_clean

    def _update_feature_stats(self, df: pd.DataFrame) -> None:
        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            if col.startswith('actual_'):
                continue

            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'min': df[col].min(),
                'max': df[col].max()
            }