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

        # Ensure datetime index for time features
        if not isinstance(df_30m.index, pd.DatetimeIndex):
            try:
                df_30m.index = pd.to_datetime(df_30m.index)
            except:
                self.logger.warning("Failed to convert index to datetime")

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

        # Make sure time features exist (may have been added already, but check again)
        if not all(col in final_df.columns for col in ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']):
            final_df = self.indicator_util.calculate_time_features(final_df)

        self.logger.info(f"Processed {len(final_df)} rows with {len(final_df.columns)} features")
        return final_df

    def compute_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate market regimes and volatility using improved methods
        df = self._calculate_improved_market_regime(df)
        df = self._calculate_improved_volatility_regime(df)

        # Calculate rsi_roc_3 for all data
        if 'rsi_14' in df.columns:
            df['rsi_roc_3'] = df['rsi_14'].pct_change(3) * 100

        # Calculate trend_strength
        if 'adx_14' in df.columns:
            df['trend_strength'] = df['adx_14'] / 100

            if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50']):
                ema_alignment = ((df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])) | \
                                ((df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50']))

                df['trend_strength'] = np.where(
                    ema_alignment,
                    df['trend_strength'] * 1.2,
                    df['trend_strength'] * 0.8
                )
                df['trend_strength'] = np.clip(df['trend_strength'], 0, 1)
        else:
            df['trend_strength'] = 0.5

        # Identify market regime segments
        regime_chunks = self._identify_regime_chunks(df['market_regime'])

        # Apply regime-specific feature engineering
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

        # Add additional features
        df = self._add_order_flow_features(df)
        df = self._add_nonlinear_interactions(df)
        df = self._add_adaptive_volatility_features(df)
        df = self._add_liquidity_features(df)
        df = self._add_cyclic_pattern_features(df)
        df = self._add_market_impact_features(df)
        df = self._clean_dataframe(df)

        return df

    def _calculate_improved_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime using a more robust approach with lookback periods"""
        if len(df) < 20 or not all(col in df.columns for col in ['close', 'high', 'low']):
            df['market_regime'] = 0.0
            return df

        # Calculate price movements over multiple timeframes
        df['pct_change_short'] = df['close'].pct_change(5)
        df['pct_change_med'] = df['close'].pct_change(20)
        df['pct_change_long'] = df['close'].pct_change(50)

        # Calculate EMA relationships
        if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50']):
            # Calculate directional alignment score
            df['ema_alignment'] = 0.0

            # Long trend: ema9 > ema21 > ema50
            long_mask = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
            df.loc[long_mask, 'ema_alignment'] = 1.0

            # Short trend: ema9 < ema21 < ema50
            short_mask = (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
            df.loc[short_mask, 'ema_alignment'] = -1.0

            # Calculate slopes to detect momentum
            df['ema9_slope'] = df['ema_9'].pct_change(3) * 100
            df['ema21_slope'] = df['ema_21'].pct_change(3) * 100

            # Combine alignment with slope for stronger signal
            df['ema_trend_score'] = df['ema_alignment'] * df['ema9_slope'].abs()
        else:
            df['ema_alignment'] = 0.0
            df['ema_trend_score'] = 0.0

        # ADX to measure trend strength
        has_adx = 'adx_14' in df.columns and not df['adx_14'].isna().all()
        if has_adx:
            df['trend_intensity'] = df['adx_14'] / 100.0
        else:
            # Fallback: Calculate volatility-adjusted momentum
            df['price_diff'] = df['close'] - df['close'].rolling(10).mean()
            df['volatility'] = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['trend_intensity'] = df['price_diff'] / df['volatility'].where(df['volatility'] > 0, 1)
            df['trend_intensity'] = df['trend_intensity'].clip(-1, 1).fillna(0)

        # Calculate trend strengths with appropriate signs
        # Positive values: bullish regime, Negative values: bearish regime
        # Magnitude indicates strength: 0-0.3 weak, 0.3-0.7 moderate, 0.7-1.0 strong

        # Short-term regime (more reactive)
        df['short_regime'] = df['pct_change_short'].rolling(5).mean() * 50
        df['short_regime'] = df['short_regime'].clip(-1, 1)

        # Medium-term regime
        df['med_regime'] = df['pct_change_med'].rolling(10).mean() * 80
        df['med_regime'] = df['med_regime'].clip(-1, 1)

        # Long-term regime
        df['long_regime'] = df['pct_change_long'].rolling(15).mean() * 120
        df['long_regime'] = df['long_regime'].clip(-1, 1)

        # Check for ranging market using volatility and lack of direction
        if 'bb_width_20' in df.columns:
            df['ranging_signal'] = (df['bb_width_20'] < 0.03).astype(float) * 0.5
        else:
            rolling_std = df['close'].pct_change().rolling(20).std()
            rolling_range = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
            df['ranging_signal'] = ((rolling_std < 0.005) & (rolling_range < 0.03)).astype(float) * 0.5

        # Combine signals with appropriate weighting
        # Give more weight to longer timeframes for stability
        df['market_regime'] = (
                0.2 * df['short_regime'] +
                0.3 * df['med_regime'] +
                0.3 * df['long_regime'] +
                0.2 * df['ema_alignment']
        )

        # Adjust for ranging markets (reduce magnitude when ranging)
        df['market_regime'] = df['market_regime'] * (1 - df['ranging_signal'])

        # Smooth the regime to prevent frequent switching
        df['market_regime'] = df['market_regime'].ewm(span=10, adjust=False).mean()

        # Clean and clip final values
        df['market_regime'] = df['market_regime'].fillna(0).clip(-1, 1)

        # Clear temporary columns
        columns_to_drop = [
            'pct_change_short', 'pct_change_med', 'pct_change_long',
            'short_regime', 'med_regime', 'long_regime',
            'ranging_signal'
        ]
        if 'price_diff' in df.columns:
            columns_to_drop.extend(['price_diff', 'volatility'])

        df = df.drop(columns=columns_to_drop, errors='ignore')
        return df

    def _calculate_improved_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility regime using multiple indicators for robustness"""
        if len(df) < 20 or not all(col in df.columns for col in ['close', 'high', 'low']):
            df['volatility_regime'] = 0.5
            return df

        # 1. Calculate normalized ATR-based volatility if ATR is available
        if 'atr_14' in df.columns:
            df['atr_volatility'] = df['atr_14'] / df['close']

            # Normalize ATR to a 0-1 scale using recent history as context
            lookback = 100
            if len(df) > lookback:
                rolling_min_atr = df['atr_volatility'].rolling(lookback).min()
                rolling_max_atr = df['atr_volatility'].rolling(lookback).max()

                # Avoid division by zero
                denominator = (rolling_max_atr - rolling_min_atr).replace(0, 1e-6)
                df['atr_volatility_normalized'] = (df['atr_volatility'] - rolling_min_atr) / denominator
            else:
                # Simple min-max scaling for shorter datasets
                min_atr = df['atr_volatility'].min()
                max_atr = df['atr_volatility'].max()
                range_atr = max(max_atr - min_atr, 1e-6)
                df['atr_volatility_normalized'] = (df['atr_volatility'] - min_atr) / range_atr
        else:
            # Calculate a simple ATR-like measure manually
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['atr_volatility'] = df['high_low_range'].rolling(14).mean()

            # Normalize
            min_val = df['atr_volatility'].min()
            max_val = df['atr_volatility'].max()
            range_val = max(max_val - min_val, 1e-6)
            df['atr_volatility_normalized'] = (df['atr_volatility'] - min_val) / range_val

        # 2. Calculate return-based volatility (standard deviation of returns)
        df['returns'] = df['close'].pct_change()
        df['return_volatility'] = df['returns'].rolling(20).std() * np.sqrt(20)  # Scaled

        # Normalize return volatility
        return_vol_max = df['return_volatility'].rolling(50).max()
        return_vol_min = df['return_volatility'].rolling(50).min()
        denominator = (return_vol_max - return_vol_min).replace(0, 1e-6)
        df['return_volatility_normalized'] = (df['return_volatility'] - return_vol_min) / denominator

        # 3. Use Bollinger Band width if available
        if 'bb_width_20' in df.columns:
            # BB width is often already normalized, but let's ensure consistency
            bb_width_max = df['bb_width_20'].rolling(50).max()
            bb_width_min = df['bb_width_20'].rolling(50).min()
            denominator = (bb_width_max - bb_width_min).replace(0, 1e-6)
            df['bb_volatility_normalized'] = (df['bb_width_20'] - bb_width_min) / denominator

            # Combine multiple volatility signals (more weighting to BB width as it's effective)
            df['volatility_regime'] = (
                    0.4 * df['bb_volatility_normalized'].fillna(0.5) +
                    0.4 * df['atr_volatility_normalized'].fillna(0.5) +
                    0.2 * df['return_volatility_normalized'].fillna(0.5)
            )
        else:
            # Without BB width, use atr and return volatility
            df['volatility_regime'] = (
                    0.6 * df['atr_volatility_normalized'].fillna(0.5) +
                    0.4 * df['return_volatility_normalized'].fillna(0.5)
            )

        # Exponential smoothing to reduce noise
        df['volatility_regime'] = df['volatility_regime'].ewm(span=5, adjust=False).mean()

        # Ensure the values are within 0-1 range
        df['volatility_regime'] = df['volatility_regime'].clip(0, 1).fillna(0.5)

        # Add volatility regime changes which can be predictive
        df['volatility_increasing'] = (df['volatility_regime'].diff(5) > 0.1).astype(float)
        df['volatility_decreasing'] = (df['volatility_regime'].diff(5) < -0.1).astype(float)

        # Clean up temporary columns
        columns_to_drop = [
            'returns', 'return_volatility', 'return_volatility_normalized',
            'atr_volatility', 'atr_volatility_normalized'
        ]
        if 'high_low_range' in df.columns:
            columns_to_drop.append('high_low_range')
        if 'bb_volatility_normalized' in df.columns:
            columns_to_drop.append('bb_volatility_normalized')

        df = df.drop(columns=columns_to_drop, errors='ignore')
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

    def _add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that measure market liquidity and imbalance"""
        if len(df) < 20 or 'volume' not in df.columns:
            return df

        df_out = df.copy()

        # Calculate volume imbalance ratio
        if 'volume' in df.columns and 'close' in df.columns and 'open' in df.columns:
            up_volume = df['volume'] * (df['close'] > df['open'])
            down_volume = df['volume'] * (df['close'] < df['open'])

            # Avoid division by zero
            up_volume = up_volume.replace(0, 1e-10)
            down_volume = down_volume.replace(0, 1e-10)

            df_out['volume_imbalance_ratio'] = up_volume / down_volume
            df_out['volume_imbalance_ratio'] = df_out['volume_imbalance_ratio'].replace([np.inf, -np.inf], 10).clip(-10,
                                                                                                                    10)

            # Calculate rolling volume imbalance
            df_out['volume_imbalance_10'] = df_out['volume_imbalance_ratio'].rolling(10).mean().fillna(1)

            # Calculate volume acceleration
            df_out['volume_acceleration'] = df['volume'].pct_change(3).fillna(0)

        # Calculate bid-ask liquidity proxies (using high-low spread as a proxy)
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            # Calculate spread as a percentage of price
            df_out['spread_pct'] = (df['high'] - df['low']) / df['close']

            # Calculate moving average of spread
            df_out['avg_spread_10'] = df_out['spread_pct'].rolling(10).mean().fillna(df_out['spread_pct'])

            # Calculate spread volatility
            df_out['spread_volatility'] = df_out['spread_pct'].rolling(20).std().fillna(0)

            # Calculate spread acceleration
            df_out['spread_acceleration'] = df_out['spread_pct'].pct_change(3).fillna(0)

        # Calculate volume-weighted average price (VWAP)
        if 'volume' in df.columns and 'close' in df.columns:
            df_out['vwap_daily'] = (df['close'] * df['volume']).rolling(48).sum() / df['volume'].rolling(48).sum()
            df_out['vwap_daily'].fillna(df['close'], inplace=True)

            # Calculate close - VWAP (deviation from fair value)
            df_out['close_vwap_diff'] = (df['close'] - df_out['vwap_daily']) / df_out['vwap_daily']

        return df_out

    def _add_cyclic_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that capture cyclic patterns in the data"""
        if len(df) < 50:
            return df

        df_out = df.copy()

        # Add Fourier transform features for price cycles
        if 'close' in df.columns:
            try:
                # Get normalized price data
                price_data = df['close'].values
                if len(price_data) > 100:
                    # Compute fast Fourier transform
                    from scipy.fft import fft

                    # Use the last 128 data points for FFT
                    n = min(128, len(price_data))
                    price_segment = price_data[-n:]

                    # Normalize and detrend
                    norm_price = (price_segment - np.mean(price_segment)) / np.std(price_segment)

                    # Compute FFT
                    fft_values = fft(norm_price)
                    fft_magnitudes = np.abs(fft_values)[:n // 2]

                    # Get the dominant frequency and its power
                    dominant_idx = np.argmax(fft_magnitudes[1:]) + 1

                    # Calculate the period length (in candles)
                    period_length = n / dominant_idx if dominant_idx > 0 else n

                    # Store the period length and magnitude
                    df_out['cycle_period'] = period_length
                    df_out['cycle_strength'] = fft_magnitudes[dominant_idx] / np.sum(fft_magnitudes) if np.sum(
                        fft_magnitudes) > 0 else 0

                    # Calculate the sine wave of the dominant cycle
                    cycles_completed = np.arange(len(df_out)) / period_length
                    df_out['cycle_phase'] = np.sin(2 * np.pi * cycles_completed)

                    # Calculate the position in the dominant cycle (0-1 range)
                    df_out['cycle_position'] = (cycles_completed % 1)

            except Exception as e:
                # If FFT fails, use a simpler approach
                df_out['cycle_period'] = 0
                df_out['cycle_strength'] = 0
                df_out['cycle_phase'] = 0
                df_out['cycle_position'] = 0

        # Add time-based cyclical features
        if isinstance(df.index, pd.DatetimeIndex):
            # Hour of day as sine/cosine (already added by indicator_util)
            if 'hour_sin' not in df.columns:
                hours = df.index.hour
                df_out['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
                df_out['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

            # Day of week features (already added by indicator_util)
            if 'day_of_week_sin' not in df.columns:
                day_of_week = df.index.dayofweek
                df_out['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
                df_out['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)

            # Add day of month features (new)
            day_of_month = df.index.day
            max_days = 31
            df_out['day_of_month_sin'] = np.sin(2 * np.pi * day_of_month / max_days)
            df_out['day_of_month_cos'] = np.cos(2 * np.pi * day_of_month / max_days)

            # Add month features (new)
            months = df.index.month
            df_out['month_sin'] = np.sin(2 * np.pi * months / 12.0)
            df_out['month_cos'] = np.cos(2 * np.pi * months / 12.0)

        return df_out

    def _add_market_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features related to market impact and price reactions"""
        if len(df) < 20:
            return df

        df_out = df.copy()

        # Calculate price impact ratio
        if all(col in df.columns for col in ['close', 'volume']):
            # Calculate price changes
            price_changes = df['close'].pct_change()

            # Calculate normalized volume
            volume_ma = df['volume'].rolling(20).mean()
            norm_volume = df['volume'] / volume_ma

            # Calculate price impact ratio (price change per unit volume)
            # Higher values mean more price impact with less volume
            df_out['price_impact_ratio'] = abs(price_changes) / norm_volume
            df_out['price_impact_ratio'] = df_out['price_impact_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

            # Rolling average of price impact
            df_out['avg_price_impact_10'] = df_out['price_impact_ratio'].rolling(10).mean().fillna(0)

        # Calculate candle size relative to average
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # Candle size
            candle_size = (df['high'] - df['low']) / df['close']

            # Average candle size over 20 periods
            avg_candle_size = candle_size.rolling(20).mean()

            # Relative candle size (compared to average)
            df_out['relative_candle_size'] = candle_size / avg_candle_size
            df_out['relative_candle_size'] = df_out['relative_candle_size'].replace([np.inf, -np.inf], 1).fillna(1)

            # Large candle indicator
            df_out['large_candle'] = (df_out['relative_candle_size'] > 1.5).astype(float)

            # Candle body ratio (close-open relative to high-low)
            body_size = abs(df['close'] - df['open'])
            total_size = df['high'] - df['low']
            df_out['candle_body_ratio'] = body_size / total_size
            df_out['candle_body_ratio'] = df_out['candle_body_ratio'].replace([np.inf, -np.inf], 0.5).fillna(0.5)

        # Calculate gap features
        if 'close' in df.columns and 'open' in df.columns:
            # Previous close to current open gap
            df_out['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df_out['gap'] = df_out['gap'].fillna(0)

            # Gap significance (is it larger than normal)
            df_out['gap_significance'] = abs(df_out['gap']) / df_out['gap'].rolling(20).std()
            df_out['gap_significance'] = df_out['gap_significance'].replace([np.inf, -np.inf], 0).fillna(0)

        return df_out

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

        # Create a dictionary to hold all the actual price columns
        actual_prices = {}
        for col in price_columns:
            if col in df.columns:
                actual_prices[f'actual_{col}'] = df[col]

        # If no columns to add, return original dataframe
        if not actual_prices:
            return df

        # Create a dataframe with just the actual price columns
        actual_df = pd.DataFrame(actual_prices, index=df.index)

        # Concat the actual price columns with the original dataframe
        return pd.concat([df, actual_df], axis=1)

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
