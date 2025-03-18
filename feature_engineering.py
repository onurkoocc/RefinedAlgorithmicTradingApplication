import logging
import numpy as np
import pandas as pd
import gc
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("FeatureEngineer")

        self.use_chunking = config.get("feature_engineering", "use_chunking", True)
        self.chunk_size = config.get("feature_engineering", "chunk_size", 2000)
        self.correlation_threshold = config.get("feature_engineering", "correlation_threshold", 0.9)

        self.indicators_to_compute = [
            "ema_7", "ema_21", "ema_50", "ema_200",
            "ema_cross", "hull_ma_16",
            "macd", "macd_signal", "macd_histogram",
            "parabolic_sar", "adx", "dmi_plus", "dmi_minus",
            "rsi_14", "stoch_k", "stoch_d", "roc_10",
            "cci_20", "willr_14", "mfi",
            "rsi_divergence",
            "atr_14", "bb_width", "bb_percent_b",
            "keltner_width", "donchian_width",
            "true_range", "natr",
            "obv", "taker_buy_ratio", "cmf", "volume_oscillator",
            "vwap", "force_index", "net_taker_flow",
            "vwap_distance",
            "volume_roc",
            "buy_sell_ratio",
            "volume_profile",
            "cumulative_delta",
            "liquidity_index"
        ]

        self.essential_features = [
            "open", "high", "low", "close", "volume",
            "ema_7", "ema_21", "ema_50", "ema_200",
            "ema_cross", "hull_ma_16", "macd_histogram",
            "rsi_14", "stoch_k", "stoch_d", "cci_20", "willr_14",
            "rsi_divergence",
            "atr_14", "bb_width", "bb_percent_b", "keltner_width",
            "natr",
            "obv", "taker_buy_ratio", "cmf", "mfi", "volume_oscillator",
            "net_taker_flow", "vwap_distance", "volume_roc",
            "buy_sell_ratio", "cumulative_delta",
            "market_regime", "volatility_regime",
            "pattern_recognition"
        ]

        self.feature_stats = {}
        self._define_indicator_parameters()

    def _define_indicator_parameters(self):
        self.ema_short_period = self.config.get("feature_engineering", "ema_short_period", 7)
        self.ema_medium_period = self.config.get("feature_engineering", "ema_medium_period", 21)
        self.ema_long_period = 50
        self.ema_vlong_period = 200

        self.hull_ma_period = self.config.get("feature_engineering", "hull_ma_period", 16)
        self.macd_fast = self.config.get("feature_engineering", "macd_fast", 12)
        self.macd_slow = self.config.get("feature_engineering", "macd_slow", 26)
        self.macd_signal_period = self.config.get("feature_engineering", "macd_signal", 9)
        self.adx_period = self.config.get("feature_engineering", "adx_period", 14)
        self.parabolic_sar_acceleration = self.config.get("feature_engineering", "parabolic_sar_acceleration", 0.02)
        self.parabolic_sar_max = self.config.get("feature_engineering", "parabolic_sar_max", 0.2)

        self.rsi_period = self.config.get("feature_engineering", "rsi_period", 14)
        self.stoch_period = self.config.get("feature_engineering", "stoch_period", 14)
        self.stoch_smooth = self.config.get("feature_engineering", "stoch_smooth", 3)
        self.roc_period = self.config.get("feature_engineering", "roc_period", 10)
        self.cci_period = self.config.get("feature_engineering", "cci_period", 20)
        self.willr_period = self.config.get("feature_engineering", "willr_period", 14)
        self.mfi_period = self.config.get("feature_engineering", "mfi_period", 14)

        self.atr_period = self.config.get("feature_engineering", "atr_period", 14)
        self.bb_period = self.config.get("feature_engineering", "bb_period", 20)
        self.bb_stddev = self.config.get("feature_engineering", "bb_stddev", 2)
        self.keltner_period = self.config.get("feature_engineering", "keltner_period", 20)
        self.keltner_atr_multiple = self.config.get("feature_engineering", "keltner_atr_multiple", 2)
        self.donchian_period = self.config.get("feature_engineering", "donchian_period", 20)

        self.cmf_period = self.config.get("feature_engineering", "cmf_period", 14)
        self.force_index_period = self.config.get("feature_engineering", "force_index_period", 13)
        self.vwap_period = self.config.get("feature_engineering", "vwap_period", 14)
        self.volume_oscillator_short = self.config.get("feature_engineering", "volume_oscillator_short", 5)
        self.volume_oscillator_long = self.config.get("feature_engineering", "volume_oscillator_long", 20)

        self.rsi_divergence_period = 14
        self.volume_roc_period = 5
        self.vwap_distance_period = 20
        self.cumulative_delta_period = 20

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

        # Detect market regime first
        df = self._add_market_regime(df)
        df = self._add_volatility_regime(df)

        # Apply regime-specific feature engineering
        regimes = df['market_regime'].values

        # Process data in chunks by regime for better adaptability
        regime_chunks = self._identify_regime_chunks(regimes)

        for regime_type, indices in regime_chunks.items():
            if not indices:
                continue

            regime_df = df.iloc[indices].copy()

            if regime_type == 'bullish':
                regime_df = self._add_bullish_regime_features(regime_df)
            elif regime_type == 'bearish':
                regime_df = self._add_bearish_regime_features(regime_df)
            else:  # neutral/ranging regime
                regime_df = self._add_ranging_regime_features(regime_df)

            # Update original dataframe with regime-specific features
            for col in regime_df.columns:
                if col not in df.columns:
                    df[col] = np.nan
                df.iloc[indices, df.columns.get_loc(col)] = regime_df[col].values

        # Add standard features that are valuable in all regimes
        df = self._add_rsi_divergence(df)
        df = self._add_order_flow_features(df)
        df = self._add_liquidity_features(df)

        # Add non-linear feature interactions
        df = self._add_nonlinear_interactions(df)

        # Add adaptive volatility features
        df = self._add_adaptive_volatility_features(df)

        # Fill any missing values from regime-specific processing
        df = self._clean_dataframe(df)

        return df

    def _identify_regime_chunks(self, regimes, threshold=0.2):
        chunks = {
            'bullish': [],
            'bearish': [],
            'neutral': []
        }

        for i, value in enumerate(regimes):
            if value > threshold:
                regime_type = 'bullish'
            elif value < -threshold:
                regime_type = 'bearish'
            else:
                regime_type = 'neutral'

            chunks[regime_type].append(i)

        return chunks

    def _add_bullish_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Enhanced momentum indicators for bullish regimes
        if 'rsi_14' in df.columns:
            # Higher weightage to RSI readings above 50
            df['rsi_bullish_bias'] = np.where(df['rsi_14'] > 50,
                                              (df['rsi_14'] - 50) / 50,
                                              (df['rsi_14'] - 50) / 100)

            # RSI rate of change - more sensitive in bullish regime
            df['rsi_roc_3'] = df['rsi_14'].pct_change(3) * 100

        # Price pullback detection in uptrend
        if 'close' in df.columns and 'ema_21' in df.columns:
            # Pullback to moving average (buying opportunity in uptrend)
            df['pullback_strength'] = np.clip((df['ema_21'] - df['close']) / df['ema_21'], -0.05, 0.05)
            df['pullback_opportunity'] = np.where(
                (df['close'] > df['ema_21']) & (df['pullback_strength'] > 0.001),
                df['pullback_strength'] * 10,
                0
            )

        # Advanced trend strength for bullish market
        if 'ema_7' in df.columns and 'ema_50' in df.columns:
            df['bull_trend_strength'] = np.clip((df['ema_7'] / df['ema_50'] - 1) * 100, 0, 10)

        # Volume confirmation features for bullish regime
        if 'volume' in df.columns and 'close' in df.columns:
            # Volume on up days vs down days
            close_change = df['close'].pct_change()
            up_volume = df['volume'] * (close_change > 0)
            down_volume = df['volume'] * (close_change < 0)
            df['up_down_vol_ratio'] = up_volume.rolling(10).mean() / (down_volume.rolling(10).mean() + 1e-10)

        return df

    def _add_bearish_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Enhanced momentum indicators for bearish regimes
        if 'rsi_14' in df.columns:
            # Higher weightage to RSI readings below 50
            df['rsi_bearish_bias'] = np.where(df['rsi_14'] < 50,
                                              (50 - df['rsi_14']) / 50,
                                              (50 - df['rsi_14']) / 100)

            # RSI momentum - look for bearish acceleration
            df['rsi_bearish_momentum'] = -df['rsi_14'].diff(3) * (df['rsi_14'] < 50)

        # Resistance bounce detection
        if 'close' in df.columns and 'ema_21' in df.columns:
            # Bounce off resistance (short opportunity in downtrend)
            df['resistance_strength'] = np.clip((df['close'] - df['ema_21']) / df['ema_21'], -0.05, 0.05)
            df['resistance_opportunity'] = np.where(
                (df['close'] < df['ema_21']) & (df['resistance_strength'] > 0.001),
                df['resistance_strength'] * 10,
                0
            )

        # Advanced trend strength for bearish market
        if 'ema_7' in df.columns and 'ema_50' in df.columns:
            df['bear_trend_strength'] = np.clip((1 - df['ema_7'] / df['ema_50']) * 100, 0, 10)

        # Oversold bounce detection
        if 'rsi_14' in df.columns and 'close' in df.columns:
            # Oversold condition followed by strength
            df['oversold_condition'] = np.where(df['rsi_14'] < 30, 1, 0)
            df['oversold_exit'] = df['oversold_condition'].rolling(5).sum() >= 2
            df['bounce_signal'] = np.where(
                df['oversold_exit'] & (df['rsi_14'] > df['rsi_14'].shift(1)),
                1, 0
            )

        return df

    def _add_ranging_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Mean reversion metrics
        if 'close' in df.columns and 'bb_percent_b' in df.columns:
            # Distance from mean with stronger signal at extremes
            df['mean_reversion_signal'] = np.where(
                df['bb_percent_b'] > 0.8,
                -(df['bb_percent_b'] - 0.8) * 5,  # Sell signal near upper band
                np.where(
                    df['bb_percent_b'] < 0.2,
                    (0.2 - df['bb_percent_b']) * 5,  # Buy signal near lower band
                    0
                )
            )

        # Range identification features
        if 'high' in df.columns and 'low' in df.columns:
            # Identify local highs and lows
            rolling_high = df['high'].rolling(20).max()
            rolling_low = df['low'].rolling(20).min()
            df['range_width'] = (rolling_high - rolling_low) / rolling_low

            # Position within range
            if 'close' in df.columns:
                df['range_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)

            # Detect consolidation patterns
            high_std = df['high'].rolling(10).std() / df['high']
            low_std = df['low'].rolling(10).std() / df['low']
            df['consolidation_intensity'] = 1 - (high_std + low_std) / 2 * 100

        # Oscillator sensitivity adjustment for ranging markets
        if 'rsi_14' in df.columns:
            # More sensitive oscillator levels for ranging markets
            df['range_rsi_signal'] = np.where(
                df['rsi_14'] > 65,
                -1 * (df['rsi_14'] - 65) / 35,  # Stronger sell in range
                np.where(
                    df['rsi_14'] < 35,
                    1 * (35 - df['rsi_14']) / 35,  # Stronger buy in range
                    0
                )
            )

        # Range breakout detection
        if 'close' in df.columns and len(df) > 20:
            df['upper_range'] = df['high'].rolling(20).max()
            df['lower_range'] = df['low'].rolling(20).min()
            df['breakout_signal'] = np.where(
                df['close'] > df['upper_range'].shift(1),
                1,  # Upside breakout
                np.where(
                    df['close'] < df['lower_range'].shift(1),
                    -1,  # Downside breakout
                    0
                )
            )

        return df

    def _add_nonlinear_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Polynomial features of key indicators
        if 'rsi_14' in df.columns and 'macd_histogram' in df.columns:
            # RSI & MACD interaction - captures momentum confirmation
            df['rsi_macd_interaction'] = df['rsi_14'] * df['macd_histogram'] / 100

            # Non-linear RSI scaling - higher sensitivity at extremes
            rsi_centered = df['rsi_14'] - 50
            df['rsi_nonlinear'] = np.sign(rsi_centered) * (rsi_centered ** 2 / 50)

            # MACD acceleration - second derivative of price action
            if len(df) > 3:
                df['macd_acceleration'] = df['macd_histogram'].diff().diff()

        # Volatility-adjusted momentum features
        if 'atr_14' in df.columns and 'close' in df.columns:
            # Normalize price changes by volatility for more consistent signals
            df['vol_norm_close_change'] = df['close'].pct_change(5) / (df['atr_14'] / df['close'])
            df['vol_norm_momentum'] = df['close'].pct_change(10) / (df['atr_14'] / df['close'])

            # Volatility-adjusted logarithmic returns - better statistical properties
            log_return = np.log(df['close'] / df['close'].shift(1))
            df['vol_adjusted_log_return'] = log_return / df['atr_14'].pct_change(20).rolling(10).std()

        # Cross-timeframe interaction features
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            # Moving average convergence/divergence dynamics
            df['ma_spread'] = (df['ema_20'] / df['ema_50'] - 1) * 100
            df['ma_spread_z'] = (df['ma_spread'] - df['ma_spread'].rolling(50).mean()) / df['ma_spread'].rolling(
                50).std()

            # MA crossover velocity - how quickly MAs are crossing
            df['ma_cross_velocity'] = df['ma_spread'].diff(3)

        # Price pattern complexity features
        if 'close' in df.columns and len(df) > 50:
            # Fractal dimension approximation - measure price complexity
            def hurst_exponent(series, max_lag=20):
                lags = range(2, max_lag)
                tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
                return np.polyfit(np.log(lags), np.log(tau), 1)[0]

            # Calculate rolling Hurst exponent (or fractal dimension)
            window = 50
            df['price_complexity'] = np.nan
            for i in range(window, len(df)):
                df.loc[df.index[i], 'price_complexity'] = hurst_exponent(df['close'].values[i - window:i])

            # Fill NaN values in the beginning
            df['price_complexity'] = df['price_complexity'].fillna(0.5)

        # Oscillator interaction features
        if 'rsi_14' in df.columns and 'stoch_k' in df.columns:
            # Combined oscillator - stronger signal when multiple indicators align
            df['combined_oscillator'] = (
                                                (df['rsi_14'] - 50) / 50 +
                                                (df['stoch_k'] - 50) / 50
                                        ) / 2

            # Oscillator divergence - when RSI and stochastic disagree
            df['oscillator_divergence'] = abs((df['rsi_14'] - 50) / 50 - (df['stoch_k'] - 50) / 50)

        # Volume and price interaction
        if 'volume' in df.columns and 'close' in df.columns:
            # Volume-weighted price momentum
            df['volume_price_momentum'] = df['close'].pct_change(5) * (df['volume'] / df['volume'].rolling(20).mean())

            # Detect volume climax events - high volume with price reversal
            df['volume_climax'] = np.where(
                (df['volume'] > df['volume'].rolling(20).mean() * 2) &
                (abs(df['close'].pct_change()) > df['close'].pct_change().rolling(20).std() * 2),
                np.sign(df['close'].pct_change()) * -1,  # Reversal signal
                0
            )

        return df

    def _add_adaptive_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        vol_regime = df['volatility_regime'].values

        # Create different features based on volatility regime
        high_vol_mask = vol_regime > 0.7
        low_vol_mask = vol_regime < 0.3
        medium_vol_mask = ~high_vol_mask & ~low_vol_mask

        if 'close' in df.columns and 'atr_14' in df.columns:
            # For high volatility - use wider channels and faster indicators
            if np.any(high_vol_mask):
                high_vol_indices = np.where(high_vol_mask)[0]

                # Wider Bollinger Band calculation for high volatility
                if len(high_vol_indices) > 20:
                    close_high_vol = df.loc[df.index[high_vol_indices], 'close']
                    rolling_mean = close_high_vol.rolling(window=20).mean()
                    rolling_std = close_high_vol.rolling(window=20).std()

                    # Create 3-sigma bands for high volatility
                    df.loc[df.index[high_vol_indices], 'high_vol_upper_band'] = rolling_mean + (rolling_std * 3)
                    df.loc[df.index[high_vol_indices], 'high_vol_lower_band'] = rolling_mean - (rolling_std * 3)
                    df.loc[df.index[high_vol_indices], 'high_vol_bandwidth'] = (rolling_std * 6) / rolling_mean

            # For low volatility - use mean reversion and tighter channels
            if np.any(low_vol_mask):
                low_vol_indices = np.where(low_vol_mask)[0]

                # Tighter Bollinger Band calculation for low volatility
                if len(low_vol_indices) > 20:
                    close_low_vol = df.loc[df.index[low_vol_indices], 'close']
                    rolling_mean = close_low_vol.rolling(window=20).mean()
                    rolling_std = close_low_vol.rolling(window=20).std()

                    # Create 1.5-sigma bands for low volatility
                    df.loc[df.index[low_vol_indices], 'low_vol_upper_band'] = rolling_mean + (rolling_std * 1.5)
                    df.loc[df.index[low_vol_indices], 'low_vol_lower_band'] = rolling_mean - (rolling_std * 1.5)
                    df.loc[df.index[low_vol_indices], 'low_vol_bandwidth'] = (rolling_std * 3) / rolling_mean

                    # Mean reversion score - stronger in low volatility
                    df.loc[df.index[low_vol_indices], 'mean_reversion_score'] = (
                            (rolling_mean - close_low_vol) / (rolling_std + 1e-10)
                    )

        # Volatility regime transition detection
        if len(df) > 5:
            df['vol_regime_change'] = df['volatility_regime'].diff(5)

            # Volatility expansion/contraction signals
            df['vol_expanding'] = np.where(df['vol_regime_change'] > 0.15, 1, 0)
            df['vol_contracting'] = np.where(df['vol_regime_change'] < -0.15, 1, 0)

        return df

    def _add_rsi_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'rsi_14' not in df.columns and 'm30_rsi_14' not in df.columns:
            return df

        df = df.copy()

        rsi_col = 'rsi_14' if 'rsi_14' in df.columns else 'm30_rsi_14'
        rsi = df[rsi_col].values

        df['rsi_divergence'] = 0
        lookback = 5

        if len(df) <= 2 * lookback + 1:
            return df

        for i in range(lookback, len(df) - lookback):
            price_window = df['close'].values[i - lookback:i + lookback + 1]
            rsi_window = rsi[i - lookback:i + lookback + 1]

            price_peak_idx = i - lookback + np.argmax(price_window)
            price_trough_idx = i - lookback + np.argmin(price_window)
            rsi_peak_idx = i - lookback + np.argmax(rsi_window)
            rsi_trough_idx = i - lookback + np.argmin(rsi_window)

            if (price_trough_idx == i and
                    abs(rsi_trough_idx - i) <= lookback and
                    price_window[lookback] < price_window[lookback - 3] and
                    rsi_window[lookback] > rsi_window[lookback - 3]):
                df.iloc[i, df.columns.get_loc('rsi_divergence')] = 1

            elif (price_peak_idx == i and
                  abs(rsi_peak_idx - i) <= lookback and
                  price_window[lookback] > price_window[lookback - 3] and
                  rsi_window[lookback] < rsi_window[lookback - 3]):
                df.iloc[i, df.columns.get_loc('rsi_divergence')] = -1

        return df

    def _add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            return df

        df = df.copy()

        if 'taker_buy_base_asset_volume' in df.columns:
            df['buy_sell_ratio'] = df['taker_buy_base_asset_volume'] / df['volume'].replace(0, np.nan)
            df['buy_sell_ratio'].fillna(0.5, inplace=True)

            delta = (2 * df['buy_sell_ratio'] - 1) * df['volume']
            df['cumulative_delta'] = delta.rolling(window=self.cumulative_delta_period).sum()
            df['cumulative_delta'].fillna(0, inplace=True)

            avg_volume = df['volume'].rolling(window=self.cumulative_delta_period).mean()
            df['cumulative_delta'] = df['cumulative_delta'] / avg_volume.replace(0, 1)

        else:
            price_change = df['close'] - df['open']
            volume_delta = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']
            df['cumulative_delta'] = volume_delta.rolling(window=self.cumulative_delta_period).sum()
            df['cumulative_delta'].fillna(0, inplace=True)

            avg_volume = df['volume'].rolling(window=self.cumulative_delta_period).mean()
            df['cumulative_delta'] = df['cumulative_delta'] / avg_volume.replace(0, 1)

            df['buy_sell_ratio'] = 0.5 + (price_change / (df['high'] - df['low']).replace(0, 1) * 0.25)

        return df

    def _add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            return df

        df = df.copy()

        df['volume_roc'] = df['volume'].pct_change(periods=self.volume_roc_period) * 100
        df['volume_roc'].fillna(0, inplace=True)

        if 'vwap' in df.columns or 'm30_vwap' in df.columns:
            vwap_col = 'vwap' if 'vwap' in df.columns else 'm30_vwap'
            df['vwap_distance'] = (df['close'] - df[vwap_col]) / df[vwap_col]
        else:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).rolling(window=self.vwap_period).sum() / \
                   df['volume'].rolling(window=self.vwap_period).sum()
            df['vwap_distance'] = (df['close'] - vwap) / vwap

        df['vwap_distance'].fillna(0, inplace=True)

        spread_proxy = (df['high'] - df['low']) / df['close']
        volume_normalized = df['volume'] / df['volume'].rolling(window=20).mean()
        df['liquidity_index'] = volume_normalized / (1 + spread_proxy * 10)
        df['liquidity_index'].fillna(1, inplace=True)

        return df

    def _add_derivative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 5:
            return df

        df = df.copy()

        key_indicators = ['rsi_14', 'macd_histogram', 'bb_width', 'atr_14']

        for indicator in key_indicators:
            if indicator in df.columns:
                df[f'{indicator}_roc'] = df[indicator].pct_change(periods=3) * 100
                df[f'{indicator}_roc'].fillna(0, inplace=True)

        if 'macd_histogram' in df.columns:
            df['momentum_acceleration'] = df['macd_histogram'].diff().diff()
            df['momentum_acceleration'].fillna(0, inplace=True)

        return df

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

    def _process_data_in_chunks(self, df_30m: pd.DataFrame, chunk_size: int = 2000) -> pd.DataFrame:
        self.logger.info(f"Processing data in chunks of size {chunk_size}")
        results = []

        overlap = max(self.bb_period, self.macd_slow, self.donchian_period,
                      self.keltner_period, self.rsi_period, self.adx_period) * 2

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

        feat_30m = self._compute_indicators(df_30m)

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

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        out = pd.DataFrame(index=df.index)
        prefix = 'm30_'

        self._compute_trend_indicators(df, out, prefix)
        self._compute_momentum_indicators(df, out, prefix)
        self._compute_volatility_indicators(df, out, prefix)
        self._compute_volume_indicators(df, out, prefix)

        out = self._clean_dataframe(out)

        return out

    def _compute_trend_indicators(self, df: pd.DataFrame, out: pd.DataFrame, prefix: str = '') -> None:
        out[f'{prefix}ema_{self.ema_short_period}'] = df['close'].ewm(span=self.ema_short_period, adjust=False).mean()
        out[f'{prefix}ema_{self.ema_medium_period}'] = df['close'].ewm(span=self.ema_medium_period, adjust=False).mean()
        out[f'{prefix}ema_cross'] = out[f'{prefix}ema_{self.ema_short_period}'] - out[
            f'{prefix}ema_{self.ema_medium_period}']

        half_period = self.hull_ma_period // 2
        sqrt_period = int(np.sqrt(self.hull_ma_period))

        wma_half = df['close'].rolling(window=half_period).apply(
            lambda x: np.sum(x * np.arange(1, half_period + 1)) / np.sum(np.arange(1, half_period + 1)), raw=True
        )
        wma_full = df['close'].rolling(window=self.hull_ma_period).apply(
            lambda x: np.sum(x * np.arange(1, self.hull_ma_period + 1)) / np.sum(np.arange(1, self.hull_ma_period + 1)),
            raw=True
        )

        temp = 2 * wma_half - wma_full
        out[f'{prefix}hull_ma_{self.hull_ma_period}'] = temp.rolling(window=sqrt_period).apply(
            lambda x: np.sum(x * np.arange(1, sqrt_period + 1)) / np.sum(np.arange(1, sqrt_period + 1)), raw=True
        )

        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        out[f'{prefix}macd'] = ema_fast - ema_slow
        out[f'{prefix}macd_signal'] = out[f'{prefix}macd'].ewm(span=self.macd_signal_period, adjust=False).mean()
        out[f'{prefix}macd_histogram'] = out[f'{prefix}macd'] - out[f'{prefix}macd_signal']

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        sar = np.zeros_like(close)
        trend = np.ones_like(close)
        acceleration_factor = self.parabolic_sar_acceleration
        acceleration_max = self.parabolic_sar_max
        extreme_point = high[0]
        acceleration = acceleration_factor

        sar[0] = low[0]

        for i in range(1, len(close)):
            if i == 1:
                if close[i] > close[i - 1]:
                    trend[i] = 1
                    sar[i] = min(low[i - 1], low[i])
                    extreme_point = high[i]
                else:
                    trend[i] = -1
                    sar[i] = max(high[i - 1], high[i])
                    extreme_point = low[i]
                acceleration = acceleration_factor
            else:
                sar[i] = sar[i - 1] + acceleration * (extreme_point - sar[i - 1])

                if trend[i - 1] == 1:
                    if sar[i] > min(low[i - 1], low[i]):
                        trend[i] = -1
                        sar[i] = max(high[max(0, i - 2):i + 1])
                        extreme_point = low[i]
                        acceleration = acceleration_factor
                    else:
                        trend[i] = 1
                        if high[i] > extreme_point:
                            extreme_point = high[i]
                            acceleration = min(acceleration + acceleration_factor, acceleration_max)
                        sar[i] = min(sar[i], min(low[max(0, i - 2):i]))
                else:
                    if sar[i] < max(high[i - 1], high[i]):
                        trend[i] = 1
                        sar[i] = min(low[max(0, i - 2):i + 1])
                        extreme_point = high[i]
                        acceleration = acceleration_factor
                    else:
                        trend[i] = -1
                        if low[i] < extreme_point:
                            extreme_point = low[i]
                            acceleration = min(acceleration + acceleration_factor, acceleration_max)
                        sar[i] = max(sar[i], max(high[max(0, i - 2):i]))

        out[f'{prefix}parabolic_sar'] = sar
        out[f'{prefix}parabolic_sar_direction'] = trend

        tr = pd.DataFrame(index=df.index)
        tr['hl'] = df['high'] - df['low']
        tr['hc'] = (df['high'] - df['close'].shift(1)).abs()
        tr['lc'] = (df['low'] - df['close'].shift(1)).abs()
        tr = tr.max(axis=1)

        plus_dm = df['high'] - df['high'].shift(1)
        minus_dm = df['low'].shift(1) - df['low']

        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

        smoothed_tr = tr.rolling(window=self.adx_period).sum()
        smoothed_plus_dm = plus_dm.rolling(window=self.adx_period).sum()
        smoothed_minus_dm = minus_dm.rolling(window=self.adx_period).sum()

        plus_di = np.zeros_like(smoothed_tr.values)
        minus_di = np.zeros_like(smoothed_tr.values)

        valid_tr = ~np.isnan(smoothed_tr) & (smoothed_tr > 0)
        if np.any(valid_tr):
            plus_di[valid_tr] = 100 * (smoothed_plus_dm.values[valid_tr] / smoothed_tr.values[valid_tr])
            minus_di[valid_tr] = 100 * (smoothed_minus_dm.values[valid_tr] / smoothed_tr.values[valid_tr])

        out[f'{prefix}plus_di'] = plus_di
        out[f'{prefix}minus_di'] = minus_di

        dx = np.zeros_like(plus_di)
        valid_sum = (plus_di + minus_di) > 0
        if np.any(valid_sum):
            dx[valid_sum] = 100 * np.abs(plus_di[valid_sum] - minus_di[valid_sum]) / (
                    plus_di[valid_sum] + minus_di[valid_sum])

        out[f'{prefix}adx'] = pd.Series(dx, index=df.index).ewm(span=self.adx_period, adjust=False).mean().fillna(25)

    def _compute_momentum_indicators(self, df: pd.DataFrame, out: pd.DataFrame, prefix: str = '') -> None:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        rs = np.zeros_like(avg_gain.values)
        valid_loss = ~np.isnan(avg_loss) & (avg_loss > 0)
        if np.any(valid_loss):
            rs[valid_loss] = avg_gain.values[valid_loss] / avg_loss.values[valid_loss]

        rsi = 100 - (100 / (1 + rs))
        out[f'{prefix}rsi_{self.rsi_period}'] = pd.Series(rsi, index=df.index).fillna(50)

        low_period = df['low'].rolling(window=self.stoch_period).min()
        high_period = df['high'].rolling(window=self.stoch_period).max()

        k_percent = np.zeros_like(df['close'].values)
        valid_range = (high_period - low_period) > 0

        if np.any(valid_range):
            k_percent[valid_range] = 100 * ((df['close'].values[valid_range] - low_period.values[valid_range]) /
                                            (high_period.values[valid_range] - low_period.values[valid_range]))

        out[f'{prefix}stoch_k'] = pd.Series(k_percent, index=df.index).fillna(50)
        out[f'{prefix}stoch_d'] = out[f'{prefix}stoch_k'].rolling(window=self.stoch_smooth).mean().fillna(50)

        out[f'{prefix}roc_{self.roc_period}'] = ((df['close'] / df['close'].shift(self.roc_period)) - 1) * 100

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = typical_price.rolling(window=self.cci_period).mean()

        mad = pd.Series(
            [abs(typical_price.iloc[max(0, i - self.cci_period + 1):i + 1] - tp_sma.iloc[i]).mean()
             for i in range(len(typical_price))],
            index=typical_price.index
        )

        cci = np.zeros_like(typical_price.values)
        valid_mad = (mad > 0) & ~np.isnan(mad) & ~np.isnan(tp_sma)

        if np.any(valid_mad):
            cci[valid_mad] = (typical_price.values[valid_mad] - tp_sma.values[valid_mad]) / (
                    0.015 * mad.values[valid_mad])

        out[f'{prefix}cci_{self.cci_period}'] = pd.Series(cci, index=df.index).fillna(0)

        highest_high = df['high'].rolling(window=self.willr_period).max()
        lowest_low = df['low'].rolling(window=self.willr_period).min()

        wr = np.zeros_like(df['close'].values) - 50
        valid_range = (highest_high - lowest_low) > 0

        if np.any(valid_range):
            wr[valid_range] = -100 * (highest_high.values[valid_range] - df['close'].values[valid_range]) / (
                    highest_high.values[valid_range] - lowest_low.values[valid_range])

        out[f'{prefix}willr_{self.willr_period}'] = pd.Series(wr, index=df.index).fillna(-50)

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        price_change = typical_price.diff()

        positive_flow = (price_change > 0) * money_flow
        negative_flow = (price_change < 0) * money_flow

        positive_sum = positive_flow.rolling(window=self.mfi_period).sum()
        negative_sum = negative_flow.rolling(window=self.mfi_period).sum()

        mfi = np.zeros_like(positive_sum.values)
        valid_negative = ~np.isnan(negative_sum) & (negative_sum > 0)
        if np.any(valid_negative):
            money_ratio = positive_sum.values[valid_negative] / negative_sum.values[valid_negative]
            mfi[valid_negative] = 100 - (100 / (1 + money_ratio))

        out[f'{prefix}mfi'] = pd.Series(mfi, index=df.index).fillna(50)

    def _compute_volatility_indicators(self, df: pd.DataFrame, out: pd.DataFrame, prefix: str = '') -> None:
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        out[f'{prefix}atr_{self.atr_period}'] = tr.rolling(window=self.atr_period).mean()

        bb_sma = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std(ddof=0)

        out[f'{prefix}bb_middle'] = bb_sma
        out[f'{prefix}bb_upper'] = bb_sma + (self.bb_stddev * bb_std)
        out[f'{prefix}bb_lower'] = bb_sma - (self.bb_stddev * bb_std)

        bb_width = np.zeros_like(bb_sma.values)
        valid_sma = ~np.isnan(bb_sma) & (bb_sma > 0)
        if np.any(valid_sma):
            bb_width[valid_sma] = (out[f'{prefix}bb_upper'].values[valid_sma] - out[f'{prefix}bb_lower'].values[
                valid_sma]) / \
                                  bb_sma.values[valid_sma]

        out[f'{prefix}bb_width'] = pd.Series(bb_width, index=df.index)

        bb_percent_b = np.zeros_like(bb_sma.values)
        valid_range = ~np.isnan(out[f'{prefix}bb_upper']) & ~np.isnan(out[f'{prefix}bb_lower']) & (
                out[f'{prefix}bb_upper'] != out[f'{prefix}bb_lower'])
        if np.any(valid_range):
            bb_percent_b[valid_range] = (
                    (df['close'].values[valid_range] - out[f'{prefix}bb_lower'].values[valid_range]) /
                    (out[f'{prefix}bb_upper'].values[valid_range] - out[f'{prefix}bb_lower'].values[valid_range]))

        out[f'{prefix}bb_percent_b'] = pd.Series(bb_percent_b, index=df.index).fillna(0.5)

        keltner_middle = df['close'].ewm(span=self.keltner_period, adjust=False).mean()
        atr = out[f'{prefix}atr_{self.atr_period}']

        out[f'{prefix}keltner_upper'] = keltner_middle + (self.keltner_atr_multiple * atr)
        out[f'{prefix}keltner_lower'] = keltner_middle - (self.keltner_atr_multiple * atr)

        keltner_width = np.zeros_like(keltner_middle.values)
        valid_middle = ~np.isnan(keltner_middle) & (keltner_middle > 0)

        if np.any(valid_middle):
            keltner_width[valid_middle] = (out[f'{prefix}keltner_upper'].values[valid_middle] -
                                           out[f'{prefix}keltner_lower'].values[valid_middle]) / \
                                          keltner_middle.values[valid_middle]

        out[f'{prefix}keltner_width'] = pd.Series(keltner_width, index=df.index)

        highest_high = df['high'].rolling(window=self.donchian_period).max()
        lowest_low = df['low'].rolling(window=self.donchian_period).min()

        donchian_width = np.zeros_like(df['close'].values)
        valid_price = ~np.isnan(df['close']) & (df['close'] > 0)

        if np.any(valid_price):
            donchian_width[valid_price] = (highest_high.values[valid_price] -
                                           lowest_low.values[valid_price]) / df['close'].values[valid_price]

        out[f'{prefix}donchian_width'] = pd.Series(donchian_width, index=df.index)

    def _compute_volume_indicators(self, df: pd.DataFrame, out: pd.DataFrame, prefix: str = '') -> None:
        if 'volume' not in df.columns or df['volume'].isna().all():
            self.logger.warning("Volume data not available for volume indicators")
            for indicator in ['obv', 'taker_buy_ratio', 'cmf', 'volume_oscillator', 'vwap', 'force_index',
                              'net_taker_flow']:
                out[f'{prefix}{indicator}'] = 0
            return

        close_diff = df['close'].diff()
        obv = np.zeros(len(df))

        for i in range(1, len(df)):
            if np.isnan(close_diff.iloc[i]) or np.isnan(df['volume'].iloc[i]):
                obv[i] = obv[i - 1]
            elif close_diff.iloc[i] > 0:
                obv[i] = obv[i - 1] + df['volume'].iloc[i]
            elif close_diff.iloc[i] < 0:
                obv[i] = obv[i - 1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i - 1]

        out[f'{prefix}obv'] = obv

        if 'taker_buy_base_asset_volume' in df.columns and 'volume' in df.columns:
            safe_volume = df['volume'].replace(0, np.nan)
            out[f'{prefix}taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / safe_volume
            out[f'{prefix}taker_buy_ratio'] = out[f'{prefix}taker_buy_ratio'].fillna(0.5).clip(0, 1)
        else:
            price_change = df['close'].pct_change()
            out[f'{prefix}taker_buy_ratio'] = 0.5 + (np.sign(price_change) * 0.1)

        high_low_diff = df['high'] - df['low']
        money_flow_multiplier = np.zeros_like(high_low_diff.values)

        valid_range = ~np.isnan(high_low_diff) & (high_low_diff > 0)
        if np.any(valid_range):
            money_flow_multiplier[valid_range] = ((df['close'].values[valid_range] - df['low'].values[valid_range]) -
                                                  (df['high'].values[valid_range] - df['close'].values[valid_range])) / \
                                                 high_low_diff.values[valid_range]

        money_flow_volume = money_flow_multiplier * df['volume'].values

        cmf_sum = pd.Series(money_flow_volume, index=df.index).rolling(window=self.cmf_period).sum()
        vol_sum = df['volume'].rolling(window=self.cmf_period).sum()

        cmf = np.zeros_like(cmf_sum.values)
        valid_vol_sum = ~np.isnan(vol_sum) & (vol_sum > 0)
        if np.any(valid_vol_sum):
            cmf[valid_vol_sum] = cmf_sum.values[valid_vol_sum] / vol_sum.values[valid_vol_sum]

        out[f'{prefix}cmf'] = pd.Series(cmf, index=df.index)

        vol_ema_short = df['volume'].ewm(span=self.volume_oscillator_short, adjust=False).mean()
        vol_ema_long = df['volume'].ewm(span=self.volume_oscillator_long, adjust=False).mean()

        vol_osc = np.zeros_like(df['volume'].values)
        valid_vol = ~np.isnan(vol_ema_long) & (vol_ema_long > 0)

        if np.any(valid_vol):
            vol_osc[valid_vol] = 100 * (vol_ema_short.values[valid_vol] - vol_ema_long.values[valid_vol]) / \
                                 vol_ema_long.values[valid_vol]

        out[f'{prefix}volume_oscillator'] = pd.Series(vol_osc, index=df.index)

        force = (df['close'] - df['close'].shift(1)) * df['volume']
        out[f'{prefix}force_index'] = force.ewm(span=self.force_index_period, adjust=False).mean()

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_volume = typical_price * df['volume']

        cumulative_tp_volume = tp_volume.rolling(window=self.vwap_period).sum()
        cumulative_volume = df['volume'].rolling(window=self.vwap_period).sum()

        vwap = np.zeros_like(cumulative_volume.values)
        valid_vol = ~np.isnan(cumulative_volume) & (cumulative_volume > 0)
        if np.any(valid_vol):
            vwap[valid_vol] = cumulative_tp_volume.values[valid_vol] / cumulative_volume.values[valid_vol]

        out[f'{prefix}vwap'] = pd.Series(vwap, index=df.index).fillna(df['close'])

        if 'taker_buy_base_asset_volume' in df.columns:
            taker_sell_volume = df['volume'] - df['taker_buy_base_asset_volume']
            out[f'{prefix}net_taker_flow'] = df['taker_buy_base_asset_volume'] - taker_sell_volume
        else:
            price_change = df['close'].pct_change()
            out[f'{prefix}net_taker_flow'] = df['volume'] * np.sign(price_change)

    def _add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'market_regime' in df.columns:
            return df

        df = df.copy()
        df['market_regime'] = np.zeros(len(df))

        try:
            ema8 = self._calculate_ema(df['close'].values, 8)
            ema21 = self._calculate_ema(df['close'].values, 21)
            ema55 = self._calculate_ema(df['close'].values, 55)
            ema100 = self._calculate_ema(df['close'].values, 100)

            # Calculate EMA slopes with multiple lookbacks for better trend detection
            ema8_slope_short = self._calculate_slope(ema8, 3)
            ema8_slope_medium = self._calculate_slope(ema8, 5)

            ema21_slope_short = self._calculate_slope(ema21, 5)
            ema21_slope_medium = self._calculate_slope(ema21, 10)

            ema55_slope = self._calculate_slope(ema55, 15)
            ema100_slope = self._calculate_slope(ema100, 20)

            # Combine multiple timeframe slopes with adaptive weighting
            trend_direction = (
                    ema8_slope_short * 0.25 +
                    ema8_slope_medium * 0.25 +
                    ema21_slope_short * 0.2 +
                    ema21_slope_medium * 0.15 +
                    ema55_slope * 0.1 +
                    ema100_slope * 0.05
            )

            # Enhance trend strength calculation
            trend_strength = np.tanh(abs(trend_direction) * 2)  # Use tanh for smoother scaling

            # Calculate price volatility with multiple lookbacks
            short_vol = np.std(df['close'].pct_change().dropna().values[-10:]) * 100
            medium_vol = np.std(df['close'].pct_change().dropna().values[-20:]) * 100

            # Use weighted volatility measure
            price_volatility = short_vol * 0.6 + medium_vol * 0.4

            # Dynamic scaling factor based on recent volatility regime
            adaptive_scale = 5.0 / max(0.1, price_volatility)

            # Enhanced EMA structure analysis
            valid_idx = ~np.isnan(df['close'].values) & ~np.isnan(ema21) & (ema21 > 0)

            if np.any(valid_idx):
                deviation = np.zeros_like(df['close'].values)

                # More nuanced EMA structure scoring
                if len(df) > 1:
                    closes = df['close'].values

                    # Strong uptrend formation
                    strong_uptrend = (closes[-1] > ema8[-1] > ema21[-1] > ema55[-1] > ema100[-1])

                    # Strong downtrend formation
                    strong_downtrend = (closes[-1] < ema8[-1] < ema21[-1] < ema55[-1] < ema100[-1])

                    # Weaker but still valid trend formations
                    moderate_uptrend = (closes[-1] > ema8[-1] > ema21[-1]) and (ema21[-1] > ema55[-1])
                    moderate_downtrend = (closes[-1] < ema8[-1] < ema21[-1]) and (ema21[-1] < ema55[-1])

                    # Potential reversal patterns
                    uptrend_weakening = (closes[-1] < ema8[-1]) and (ema8[-1] > ema21[-1] > ema55[-1])
                    downtrend_weakening = (closes[-1] > ema8[-1]) and (ema8[-1] < ema21[-1] < ema55[-1])

                    # Assign EMA structure score based on pattern
                    if strong_uptrend:
                        ema_structure = 0.8
                    elif strong_downtrend:
                        ema_structure = -0.8
                    elif moderate_uptrend:
                        ema_structure = 0.5
                    elif moderate_downtrend:
                        ema_structure = -0.5
                    elif uptrend_weakening:
                        ema_structure = 0.2
                    elif downtrend_weakening:
                        ema_structure = -0.2
                    else:
                        ema_structure = 0

                # Calculate price deviation from medium-term EMA (21)
                deviation[valid_idx] = ((df['close'].values[valid_idx] / ema21[valid_idx]) - 1) * adaptive_scale

                # Combine deviation with EMA structure for more accurate regime detection
                deviation = np.clip(deviation + ema_structure, -1.0, 1.0)

                df['market_regime'] = deviation

                # Enhance with rate-of-change data for more accurate momentum
                if len(df) >= 30:
                    # Multiple timeframe ROC for better momentum measurement
                    roc3 = df['close'].pct_change(3).values[-1] * 100
                    roc5 = df['close'].pct_change(5).values[-1] * 100
                    roc10 = df['close'].pct_change(10).values[-1] * 100
                    roc20 = df['close'].pct_change(20).values[-1] * 100

                    # Weighted momentum factor with more weight on shorter timeframes in high vol,
                    # longer timeframes in low vol
                    if price_volatility > 1.5:
                        # Higher weight to short-term momentum in high volatility
                        weights = [0.4, 0.3, 0.2, 0.1]
                    elif price_volatility < 0.5:
                        # Higher weight to longer-term momentum in low volatility
                        weights = [0.1, 0.2, 0.3, 0.4]
                    else:
                        # Balanced weights for normal volatility
                        weights = [0.25, 0.3, 0.25, 0.2]

                    momentum_factor = (
                            weights[0] * np.sign(roc3) * min(1.0, abs(roc3) / 1.0) +
                            weights[1] * np.sign(roc5) * min(1.0, abs(roc5) / 1.5) +
                            weights[2] * np.sign(roc10) * min(1.0, abs(roc10) / 2.0) +
                            weights[3] * np.sign(roc20) * min(1.0, abs(roc20) / 3.0)
                    )

                    # Combine base regime with momentum for final score
                    # Adaptive weighting based on trend strength
                    base_weight = 0.7 - (0.2 * trend_strength)  # Stronger trends get less base regime weight
                    momentum_weight = 1.0 - base_weight

                    df['market_regime'] = base_weight * df['market_regime'] + momentum_weight * momentum_factor
                    df['market_regime'] = np.clip(df['market_regime'], -1.0, 1.0)

        except Exception as e:
            self.logger.warning(f"Error in enhanced market regime detection: {e}")

        return df

    def _calculate_slope(self, values, lookback):
        """Calculate normalized slope of values over lookback period"""
        if len(values) <= lookback:
            return 0

        recent = values[-1]
        past = values[-lookback]

        if past <= 0:
            return 0

        return (recent / past - 1) * 100

    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'volatility_regime' in df.columns:
            return df

        df = df.copy()
        df['volatility_regime'] = np.full(len(df), 0.5)

        try:
            # Combine multiple volatility metrics for more robust regime detection
            volatility_signals = []

            # 1. ATR-based volatility
            if 'atr_14' in df.columns or 'm30_atr_14' in df.columns:
                atr_col = 'atr_14' if 'atr_14' in df.columns else 'm30_atr_14'
                atr = df[atr_col].values
                close = df['close'].values

                valid_idx = ~np.isnan(atr) & ~np.isnan(close) & (close > 0)
                if np.any(valid_idx):
                    # Normalized ATR
                    atr_pct = np.zeros_like(atr)
                    atr_pct[valid_idx] = atr[valid_idx] / close[valid_idx]

                    # Calculate ATR percentiles using recent history
                    if len(df) > 100:
                        atr_pct_valid = atr_pct[valid_idx]
                        p10 = np.percentile(atr_pct_valid, 10)
                        p90 = np.percentile(atr_pct_valid, 90)

                        val_range = max(p90 - p10, 0.001)

                        vol_signal = np.full_like(atr, 0.5)
                        vol_signal[valid_idx] = 0.1 + 0.8 * np.clip(
                            (atr_pct[valid_idx] - p10) / val_range, 0, 1
                        )

                        volatility_signals.append((vol_signal, 0.4))  # 40% weight to ATR

            # 2. Bollinger Band width
            if 'bb_width' in df.columns or 'm30_bb_width' in df.columns:
                bb_col = 'bb_width' if 'bb_width' in df.columns else 'm30_bb_width'
                bb_width = df[bb_col].values

                valid_idx = ~np.isnan(bb_width)
                if np.any(valid_idx):
                    if len(df) > 100:
                        bb_width_valid = bb_width[valid_idx]
                        p10 = np.percentile(bb_width_valid, 10)
                        p90 = np.percentile(bb_width_valid, 90)

                        val_range = max(p90 - p10, 0.001)

                        vol_signal = np.full_like(bb_width, 0.5)
                        vol_signal[valid_idx] = 0.1 + 0.8 * np.clip(
                            (bb_width[valid_idx] - p10) / val_range, 0, 1
                        )

                        volatility_signals.append((vol_signal, 0.3))  # 30% weight to BB width

            # 3. Historical volatility (standard deviation of returns)
            if 'close' in df.columns:
                returns = df['close'].pct_change().values
                hist_vol = np.zeros_like(returns)

                for i in range(20, len(returns)):
                    hist_vol[i] = np.std(returns[i - 20:i]) * np.sqrt(252 * 48)  # Annualized

                valid_idx = ~np.isnan(hist_vol)
                if np.any(valid_idx):
                    if len(df) > 100:
                        hist_vol_valid = hist_vol[valid_idx]
                        hist_vol_valid = hist_vol_valid[hist_vol_valid > 0]

                        if len(hist_vol_valid) > 0:
                            p10 = np.percentile(hist_vol_valid, 10)
                            p90 = np.percentile(hist_vol_valid, 90)

                            val_range = max(p90 - p10, 0.001)

                            vol_signal = np.full_like(hist_vol, 0.5)
                            vol_signal[valid_idx] = 0.1 + 0.8 * np.clip(
                                (hist_vol[valid_idx] - p10) / val_range, 0, 1
                            )

                            volatility_signals.append((vol_signal, 0.3))  # 30% weight to hist vol

            # Combine all volatility signals with their weights
            if volatility_signals:
                combined_signal = np.zeros(len(df))
                combined_weight = 0

                for signal, weight in volatility_signals:
                    combined_signal += signal * weight
                    combined_weight += weight

                if combined_weight > 0:
                    combined_signal /= combined_weight

                    # Apply smoothing to prevent rapid regime changes
                    if len(df) > 5:
                        for i in range(4, len(combined_signal)):
                            combined_signal[i] = 0.6 * combined_signal[i] + 0.4 * combined_signal[i - 1]

                    df['volatility_regime'] = combined_signal

            # Add volatility trend information
            if len(df) > 20:
                df['volatility_expanding'] = np.zeros(len(df))
                df['volatility_contracting'] = np.zeros(len(df))

                for i in range(10, len(df)):
                    recent_vol = df['volatility_regime'].values[i]
                    past_vol = df['volatility_regime'].values[i - 10]

                    if recent_vol > past_vol + 0.15:
                        df.iloc[i, df.columns.get_loc('volatility_expanding')] = 1
                    elif recent_vol < past_vol - 0.15:
                        df.iloc[i, df.columns.get_loc('volatility_contracting')] = 1

        except Exception as e:
            self.logger.warning(f"Error in enhanced volatility regime detection: {e}")

        return df

    def _calculate_ema(self, values: np.ndarray, period: int) -> np.ndarray:
        if len(values) < period:
            return np.zeros_like(values)

        values = np.copy(values)

        mask = np.isnan(values) | np.isinf(values)
        if np.any(mask):
            indices = np.arange(len(values))
            valid_indices = indices[~mask]
            if len(valid_indices) > 0:
                valid_values = values[valid_indices]
                values = np.interp(indices, valid_indices, valid_values)
            else:
                return np.zeros_like(values)

        ema = np.zeros_like(values)
        alpha = 2.0 / (period + 1.0)
        ema[:period] = np.mean(values[:period])

        for i in range(period, len(values)):
            ema[i] = values[i] * alpha + ema[i - 1] * (1.0 - alpha)

        ema = np.nan_to_num(ema, nan=0.0, posinf=0.0, neginf=0.0)

        return ema

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
            elif 'stoch' in col.lower():
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