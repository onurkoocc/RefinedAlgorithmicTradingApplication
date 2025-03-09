import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("FeatureEngineer")

        self.feature_scaling = config.get("feature_engineering", "use_feature_scaling", True)
        self.use_chunking = config.get("feature_engineering", "use_chunking", True)
        self.chunk_size = config.get("feature_engineering", "chunk_size", 2000)

        self.indicators = [
            "ema_20", "obv", "bb_width", "bb_upper", "bb_lower", "bb_mid",
            "atr_14", "rsi_14", "macd", "macd_signal", "macd_histogram",
            "cmf", "mfi", "vwap"
        ]

        self.essential_features = [
            "open", "high", "low", "close", "volume",
            "ema_20", "obv", "bb_width", "atr_14",
            "rsi_14", "macd", "macd_signal", "macd_histogram",
            "cmf", "mfi", "vwap"
        ]

        self.ema_period = 20
        self.bb_periods = (20, 2)
        self.atr_period = 14
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

        self.cmf_period = 20
        self.mfi_period = 14
        self.vwap_period = 14

    def process_features(self, df_30m: pd.DataFrame) -> pd.DataFrame:
        if self.use_chunking and len(df_30m) > self.chunk_size:
            final_df = self._process_data_in_chunks(df_30m, chunk_size=self.chunk_size)
        else:
            final_df = self._process_data_combined(df_30m)

        # Validate dataframe before further processing
        if final_df.empty:
            self.logger.warning("Empty dataframe after initial processing")
            return pd.DataFrame()

        # Check for inf values before filtering indicators
        inf_check = np.isinf(final_df.select_dtypes(include=[np.number]).values).sum()
        if inf_check > 0:
            self.logger.warning(f"Found {inf_check} infinite values before filtering indicators, replacing with NaN")
            final_df = final_df.replace([np.inf, -np.inf], np.nan)

        final_df = self._filter_indicators(final_df)

        final_df = self.compute_advanced_features(final_df)

        # Validate again before scaling
        if self.feature_scaling:
            # Check for problematic values
            numeric_cols = final_df.select_dtypes(include=[np.number]).columns
            inf_count = np.isinf(final_df[numeric_cols].values).sum()
            nan_count = np.isnan(final_df[numeric_cols].values).sum()

            if inf_count > 0 or nan_count > 0:
                self.logger.warning(f"Found {inf_count} infinite values and {nan_count} NaN values before scaling")
                # Replace inf with NaN first
                final_df = final_df.replace([np.inf, -np.inf], np.nan)
                # Handle remaining issues in _scale_features

            final_df = self._scale_features(final_df)

        final_df.columns = [col.lower() for col in final_df.columns]

        self.logger.info(f"Processed {len(final_df)} rows with {len(final_df.columns)} features")
        return final_df

    def compute_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = self._add_swing_points(df)
        df = self._add_market_structure(df)
        df = self._add_volume_patterns(df)
        df = self._add_vwap_analysis(df)
        df = self._add_momentum_factors(df)
        df = self._add_volatility_regime_features(df)
        df = self._add_mean_reversion_indicators(df)

        return df

    def _add_swing_points(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Identify swing high/low points using only past data.
        A true swing point can only be confirmed after a retracement,
        so we need to look for past formations without using future data.
        """
        highs = df['high'].values
        lows = df['low'].values

        swing_highs = np.zeros(len(df))
        swing_lows = np.zeros(len(df))

        # Need enough data to look back 2*window periods
        for i in range(window * 2, len(df)):
            # Check if the point (window) bars ago was the highest in its surrounding range
            # and the price has since moved down (confirming it was a swing high)
            if (i >= window + 1 and
                    highs[i - window] == max(highs[i - 2 * window:i]) and
                    highs[i - 1] < highs[i - window]):
                swing_highs[i] = 1

            # Check if the point (window) bars ago was the lowest in its surrounding range
            # and the price has since moved up (confirming it was a swing low)
            if (i >= window + 1 and
                    lows[i - window] == min(lows[i - 2 * window:i]) and
                    lows[i - 1] > lows[i - window]):
                swing_lows[i] = 1

        df['swing_high'] = swing_highs
        df['swing_low'] = swing_lows

        return df

    def _add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
            df = self._add_swing_points(df)

        swing_high_indices = np.where(df['swing_high'] == 1)[0]
        swing_low_indices = np.where(df['swing_low'] == 1)[0]

        hh = np.zeros(len(df))
        lh = np.zeros(len(df))
        hl = np.zeros(len(df))
        ll = np.zeros(len(df))

        for i in range(1, len(swing_high_indices)):
            current_idx = swing_high_indices[i]
            prev_idx = swing_high_indices[i - 1]

            if df['high'].iloc[current_idx] > df['high'].iloc[prev_idx]:
                hh[current_idx] = 1
            else:
                lh[current_idx] = 1

        for i in range(1, len(swing_low_indices)):
            current_idx = swing_low_indices[i]
            prev_idx = swing_low_indices[i - 1]

            if df['low'].iloc[current_idx] > df['low'].iloc[prev_idx]:
                hl[current_idx] = 1
            else:
                ll[current_idx] = 1

        df['higher_high'] = hh
        df['lower_high'] = lh
        df['higher_low'] = hl
        df['lower_low'] = ll

        market_state = np.zeros(len(df), dtype=int)
        window = 5

        for i in range(max(20, window * 2), len(df)):
            recent = slice(i - 20, i)

            recent_hh = sum(hh[recent])
            recent_lh = sum(lh[recent])
            recent_hl = sum(hl[recent])
            recent_ll = sum(ll[recent])

            if recent_hh > 0 and recent_hl > 0 and recent_hh + recent_hl > recent_lh + recent_ll:
                market_state[i] = 2
            elif recent_hl > recent_ll and recent_lh > recent_hh:
                market_state[i] = 1
            elif recent_lh > 0 and recent_ll > 0 and recent_lh + recent_ll > recent_hh + recent_hl:
                market_state[i] = -2
            elif recent_lh > recent_hh and recent_hl > recent_ll:
                market_state[i] = -1
            else:
                market_state[i] = 0

        df['market_structure'] = market_state

        # NEW: Add a reversion potential indicator
        reversion_potential = np.zeros(len(df))

        for i in range(20, len(df)):
            # Check for overextended conditions using RSI or similar indicators
            rsi_overextended = False
            if 'rsi_14' in df.columns and not np.isnan(df['rsi_14'].iloc[i]):
                if df['rsi_14'].iloc[i] > 70 or df['rsi_14'].iloc[i] < 30:
                    rsi_overextended = True
            elif 'm30_rsi_14' in df.columns and not np.isnan(df['m30_rsi_14'].iloc[i]):
                if df['m30_rsi_14'].iloc[i] > 70 or df['m30_rsi_14'].iloc[i] < 30:
                    rsi_overextended = True

            # Check for price overextension from moving averages
            price_overextended = False
            if 'ema_20' in df.columns and not np.isnan(df['ema_20'].iloc[i]):
                dev_pct = abs(df['close'].iloc[i] / df['ema_20'].iloc[i] - 1)
                if dev_pct > 0.03:  # 3% away from EMA20
                    price_overextended = True
            elif 'm30_ema_20' in df.columns and not np.isnan(df['m30_ema_20'].iloc[i]):
                dev_pct = abs(df['close'].iloc[i] / df['m30_ema_20'].iloc[i] - 1)
                if dev_pct > 0.03:  # 3% away from EMA20
                    price_overextended = True

            # Combined indicator
            if rsi_overextended and price_overextended:
                # Direction of potential reversion
                if ('rsi_14' in df.columns and df['rsi_14'].iloc[i] > 70) or \
                        ('m30_rsi_14' in df.columns and df['m30_rsi_14'].iloc[i] > 70):
                    reversion_potential[i] = -1  # Potential bearish reversion
                else:
                    reversion_potential[i] = 1  # Potential bullish reversion

        df['reversion_potential'] = reversion_potential

        # NEW: Add support/resistance encounter tracking
        if len(df) > 50:
            support_resistance = self._identify_support_resistance_zones(df)
            df['at_key_level'] = self._mark_key_level_proximity(df, support_resistance)
        else:
            df['at_key_level'] = 0

        return df

    # NEW: Add support/resistance zone identification
    def _identify_support_resistance_zones(self, df: pd.DataFrame, lookback: int = 100, window: int = 5) -> list:
        price_points = []

        # Find swing highs and lows
        for i in range(window, min(lookback, len(df) - window)):
            # Swing high
            if df['high'].iloc[i] == df['high'].iloc[i - window:i + window + 1].max():
                price_points.append({
                    'price': df['high'].iloc[i],
                    'type': 'resistance'
                })

            # Swing low
            if df['low'].iloc[i] == df['low'].iloc[i - window:i + window + 1].min():
                price_points.append({
                    'price': df['low'].iloc[i],
                    'type': 'support'
                })

        # Cluster nearby levels
        clusters = []
        if price_points:
            sorted_points = sorted(price_points, key=lambda x: x['price'])

            current_cluster = [sorted_points[0]]
            for i in range(1, len(sorted_points)):
                last_price = current_cluster[-1]['price']
                current_price = sorted_points[i]['price']

                # If within 0.5% of the last price, add to current cluster
                if abs(current_price / last_price - 1) < 0.005:
                    current_cluster.append(sorted_points[i])
                else:
                    # Determine if the cluster is primarily support or resistance
                    support_count = sum(1 for point in current_cluster if point['type'] == 'support')
                    resistance_count = len(current_cluster) - support_count

                    cluster_type = 'support' if support_count > resistance_count else 'resistance'

                    # Calculate the average price of the cluster
                    avg_price = sum(point['price'] for point in current_cluster) / len(current_cluster)

                    clusters.append({
                        'price': avg_price,
                        'type': cluster_type,
                        'strength': len(current_cluster)
                    })

                    # Start a new cluster
                    current_cluster = [sorted_points[i]]

            # Don't forget the last cluster
            if current_cluster:
                support_count = sum(1 for point in current_cluster if point['type'] == 'support')
                resistance_count = len(current_cluster) - support_count

                cluster_type = 'support' if support_count > resistance_count else 'resistance'

                avg_price = sum(point['price'] for point in current_cluster) / len(current_cluster)

                clusters.append({
                    'price': avg_price,
                    'type': cluster_type,
                    'strength': len(current_cluster)
                })

        return clusters

    # NEW: Mark proximity to key support/resistance levels
    def _mark_key_level_proximity(self, df: pd.DataFrame, support_resistance: list) -> np.ndarray:
        proximity = np.zeros(len(df))

        for i in range(len(df)):
            current_price = df['close'].iloc[i]

            for level in support_resistance:
                distance_pct = abs(current_price / level['price'] - 1)

                # If within 1% of a key level
                if distance_pct < 0.01:
                    # Positive for support, negative for resistance
                    if level['type'] == 'support' and current_price > level['price']:
                        proximity[i] = 1 * min(level['strength'], 5)  # Scale by strength, max 5
                    elif level['type'] == 'resistance' and current_price < level['price']:
                        proximity[i] = -1 * min(level['strength'], 5)  # Scale by strength, max 5

        return proximity

    def _add_volume_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        volumes = df['volume'].values
        closes = df['close'].values

        vol_ma = np.zeros_like(volumes)
        vol_std = np.zeros_like(volumes)
        norm_vol = np.zeros_like(volumes)

        for i in range(20, len(volumes)):
            vol_ma[i] = np.mean(volumes[i - 20:i])
            vol_std[i] = np.std(volumes[i - 20:i])

            if vol_std[i] > 0:
                norm_vol[i] = (volumes[i] - vol_ma[i]) / vol_std[i]

        df['normalized_volume'] = norm_vol

        price_change = np.zeros_like(closes)
        for i in range(1, len(closes)):
            price_change[i] = (closes[i] / closes[i - 1]) - 1

        df['volume_price_trend'] = price_change * norm_vol

        volume_delta = np.zeros_like(volumes)
        for i in range(1, len(df)):
            if closes[i] > closes[i - 1]:
                volume_delta[i] = volumes[i]
            else:
                volume_delta[i] = -volumes[i]

        df['volume_delta'] = volume_delta
        df['cumulative_volume_delta'] = np.cumsum(volume_delta)

        return df

    def _add_vwap_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df.index).date
        grouped = df.groupby('date')

        df['vwap_daily'] = np.nan
        df['vwap_stdev_daily'] = np.nan
        df['vwap_upper1_daily'] = np.nan
        df['vwap_lower1_daily'] = np.nan
        df['vwap_upper2_daily'] = np.nan
        df['vwap_lower2_daily'] = np.nan

        for date, group in grouped:
            typical_price = (group['high'] + group['low'] + group['close']) / 3
            pv = typical_price * group['volume']
            vwap = pv.cumsum() / np.maximum(group['volume'].cumsum(), 1e-10)  # Prevent division by zero

            # Safe calculation of standard deviation
            squared_diff = (typical_price - vwap) ** 2 * group['volume']
            vwap_stdev = np.sqrt(np.maximum(squared_diff.cumsum() / np.maximum(group['volume'].cumsum(), 1e-10), 0))

            vwap_upper1 = vwap + vwap_stdev
            vwap_lower1 = vwap - vwap_stdev
            vwap_upper2 = vwap + 2 * vwap_stdev
            vwap_lower2 = vwap - 2 * vwap_stdev

            df.loc[group.index, 'vwap_daily'] = vwap.values
            df.loc[group.index, 'vwap_stdev_daily'] = vwap_stdev.values
            df.loc[group.index, 'vwap_upper1_daily'] = vwap_upper1.values
            df.loc[group.index, 'vwap_lower1_daily'] = vwap_lower1.values
            df.loc[group.index, 'vwap_upper2_daily'] = vwap_upper2.values
            df.loc[group.index, 'vwap_lower2_daily'] = vwap_lower2.values

        # Prevent division by zero or very small values
        df['vwap_deviation'] = (df['close'] - df['vwap_daily']) / np.maximum(df['vwap_stdev_daily'], 1e-10)
        df['vwap_deviation'] = df['vwap_deviation'].replace([np.inf, -np.inf], 0)

        df = df.drop('date', axis=1)

        return df

    def _add_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in [1, 3, 5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)

        # Improved momentum score calculation with more weight on recent periods
        df['momentum_score'] = (
                df['roc_1'] * 0.25 +  # Increased from 0.15 to 0.25
                df['roc_3'] * 0.3 +  # Increased from 0.25 to 0.3
                df['roc_5'] * 0.25 +  # Decreased from 0.3 to 0.25
                df['roc_10'] * 0.15 +  # Decreased from 0.2 to 0.15
                df['roc_20'] * 0.05  # Decreased from 0.1 to 0.05
        )

        # Safe calculation of z-score using std with min value
        momentum_rolling_mean = df['momentum_score'].rolling(20).mean()
        momentum_rolling_std = df['momentum_score'].rolling(20).std()
        momentum_rolling_std = np.maximum(momentum_rolling_std, 1e-10)  # Prevent division by zero
        df['momentum_z'] = (df['momentum_score'] - momentum_rolling_mean) / momentum_rolling_std

        # Add shorter-term RSI for quick momentum detection
        df['rsi_2'] = self._calculate_rsi(df['close'], 2)
        df['rsi_5'] = self._calculate_rsi(df['close'], 5)

        # Add extreme RSI detection for overbought/oversold
        df['rsi_extreme'] = 0
        if 'rsi_14' in df.columns:
            df.loc[df['rsi_14'] > 75, 'rsi_extreme'] = 1
            df.loc[df['rsi_14'] < 25, 'rsi_extreme'] = -1
        elif 'm30_rsi_14' in df.columns:
            df.loc[df['m30_rsi_14'] > 75, 'rsi_extreme'] = 1
            df.loc[df['m30_rsi_14'] < 25, 'rsi_extreme'] = -1

        # Handle RSI divergence calculation properly
        rsi_column = None
        if 'rsi_14' in df.columns:
            rsi_column = 'rsi_14'
        elif 'm30_rsi_14' in df.columns:
            rsi_column = 'm30_rsi_14'
            # Also create rsi_14 column for other methods that might need it
            df['rsi_14'] = df['m30_rsi_14']

        if rsi_column is not None:
            df['rsi_divergence'] = self._calculate_divergence(df['close'], df[rsi_column])
        else:
            # If no RSI column is available, create a default divergence column filled with zeros
            df['rsi_divergence'] = np.zeros(len(df))
            self.logger.warning("No RSI-14 column found for divergence calculation")

        return df
    def _calculate_rsi(self, series, window):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        # Safely calculate RS
        rs = np.zeros_like(gain)
        valid_loss = ~np.isnan(loss) & (loss > 0)
        rs[valid_loss] = gain[valid_loss] / loss[valid_loss]

        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_divergence(self, price, indicator, window=14):
        """
        Calculate RSI/price divergence using only past data.
        Looks for confirmed divergences that have already shown a reversal pattern.
        """
        divergence = np.zeros(len(price))

        for i in range(window * 2, len(price)):
            past_window = slice(i - 2 * window, i)

            # Bearish divergence: price made higher high but indicator did not
            # We need the price to show some retracement to confirm the pattern
            if (i >= 3 and
                    price.iloc[i - 2] > price.iloc[past_window].max() and
                    indicator.iloc[i - 2] < indicator.iloc[past_window].max() and
                    price.iloc[i - 1] < price.iloc[i - 2]):  # Confirmation of reversal
                divergence[i] = -1

            # Bullish divergence: price made lower low but indicator did not
            # We need the price to show some retracement to confirm the pattern
            elif (i >= 3 and
                  price.iloc[i - 2] < price.iloc[past_window].min() and
                  indicator.iloc[i - 2] > indicator.iloc[past_window].min() and
                  price.iloc[i - 1] > price.iloc[i - 2]):  # Confirmation of reversal
                divergence[i] = 1

        return divergence

    def _add_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate Parkinson volatility with safety checks
        log_high_low = np.log(np.maximum(df['high'] / np.maximum(df['low'], 1e-10), 1e-10))
        df['parkinson_vol'] = np.sqrt(
            np.maximum((1.0 / (4.0 * np.log(2.0))) * (log_high_low ** 2), 0)
        ).rolling(window=10).mean()

        # Calculate Garman-Klass volatility with safety checks
        high_low = log_high_low ** 2
        close_open = np.log(np.maximum(df['close'] / np.maximum(df['open'], 1e-10), 1e-10)) ** 2

        gk_vol = 0.5 * high_low - (2.0 * np.log(2.0) - 1.0) * close_open
        df['garman_klass_vol'] = np.sqrt(np.maximum(gk_vol, 0)).rolling(window=10).mean()

        for period in [10, 20, 50]:
            # Safe relative volatility calculation
            atr_mean = df['atr_14'].rolling(period).mean()
            df[f'relative_vol_{period}'] = df['atr_14'] / np.maximum(atr_mean, 1e-10)
            df[f'relative_vol_{period}'] = df[f'relative_vol_{period}'].replace([np.inf, -np.inf], 1.0)

        vol_regimes = np.zeros(len(df), dtype=int)

        for i in range(20, len(df)):
            if 'relative_vol_20' not in df.columns:
                continue

            rel_vol = df['relative_vol_20'].iloc[i]

            if np.isnan(rel_vol):
                continue

            if rel_vol < 0.8:
                vol_regimes[i] = 0
            elif rel_vol > 1.2:
                vol_regimes[i] = 2
            else:
                vol_regimes[i] = 1

        df['volatility_regime_class'] = vol_regimes

        df['volatility_trend'] = df['atr_14'].pct_change(5)

        return df

    def _add_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        for ma_period in [20, 50, 100]:
            ma_col = f'ma_{ma_period}'
            df[ma_col] = df['close'].rolling(window=ma_period).mean()
            # Safe percentage calculation
            df[f'dist_from_{ma_col}'] = (df['close'] / np.maximum(df[ma_col], 1e-10) - 1) * 100
            df[f'dist_from_{ma_col}'] = df[f'dist_from_{ma_col}'].replace([np.inf, -np.inf], 0)

        bb_period = 20
        bb_std = 2

        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()

        df['bb_upper'] = df['bb_middle'] + bb_std * bb_std_val
        df['bb_lower'] = df['bb_middle'] - bb_std * bb_std_val

        # Safe calculation of %B to avoid division by zero
        bb_width = df['bb_upper'] - df['bb_lower']
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / np.maximum(bb_width, 1e-10)
        df['bb_percent_b'] = df['bb_percent_b'].replace([np.inf, -np.inf], 0.5)

        df['mean_reversion_probability'] = np.zeros(len(df))

        for i in range(bb_period, len(df)):
            bb_val = df['bb_percent_b'].iloc[i]

            if np.isnan(bb_val):
                continue

            if bb_val > 1.0:
                df.loc[df.index[i], 'mean_reversion_probability'] = min(0.9, (bb_val - 1.0) * 5 + 0.5)
            elif bb_val < 0.0:
                df.loc[df.index[i], 'mean_reversion_probability'] = min(0.9, (0.0 - bb_val) * 5 + 0.5)
            else:
                df.loc[df.index[i], 'mean_reversion_probability'] = max(0.1, 0.5 - abs(bb_val - 0.5))

        return df

    def _process_data_combined(self, df_30m: pd.DataFrame) -> pd.DataFrame:
        df_30m = df_30m.copy()

        df_30m.replace([np.inf, -np.inf], np.nan, inplace=True)
        if not isinstance(df_30m.index, pd.DatetimeIndex):
            df_30m.index = pd.to_datetime(df_30m.index)

        feat_30m = self._compute_core_indicators(df_30m).add_prefix('m30_')
        feat_30m[['open', 'high', 'low', 'close', 'volume']] = df_30m[['open', 'high', 'low', 'close', 'volume']]

        if 'm30_obv' in feat_30m.columns:
            feat_30m['obv'] = feat_30m['m30_obv']

        # Ensure all necessary columns are copied from prefixed versions if they exist
        for base, prefixed in [('ema_20', 'm30_ema_20'), ('rsi_14', 'm30_rsi_14'),
                               ('macd', 'm30_macd'), ('macd_signal', 'm30_macd_signal'),
                               ('macd_histogram', 'm30_macd_histogram'),
                               ('atr_14', 'm30_atr_14'), ('bb_width', 'm30_bb_width')]:
            if prefixed in feat_30m.columns:
                feat_30m[base] = feat_30m[prefixed]

        feat_30m = self._fill_nans(feat_30m)

        combined = self._fill_nans(feat_30m, critical_cols=['open', 'high', 'low', 'close', 'volume'])
        combined.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

        if combined.empty:
            self.logger.warning("No data after processing")
            return pd.DataFrame()

        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = combined.fillna(0)

        combined['market_regime'] = self._compute_market_regime(combined)
        combined['volatility_regime'] = self._compute_volatility_regime(combined)

        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = combined.fillna(0)

        return combined

    def _process_data_in_chunks(self, df_30m: pd.DataFrame,
                                chunk_size: int = 2000) -> pd.DataFrame:
        self.logger.info(f"Processing data in chunks of size {chunk_size}")
        results = []

        for i in range(0, len(df_30m), chunk_size):
            end_idx = min(i + chunk_size, len(df_30m))
            chunk_30m = df_30m.iloc[i:end_idx].copy()

            try:
                chunk_features = self._process_data_combined(chunk_30m)
                results.append(chunk_features)
            except Exception as e:
                self.logger.error(f"Error processing chunk: {e}")

        if not results:
            return pd.DataFrame()

        combined = pd.concat(results, axis=0)
        return combined

    def _compute_core_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        out = pd.DataFrame(index=df.index)

        # Copy price data
        out['open'] = df['open']
        out['high'] = df['high']
        out['low'] = df['low']
        out['close'] = df['close']
        out['volume'] = df['volume']

        # Calculate EMA (Exponential Moving Average)
        out['ema_20'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()

        # Calculate OBV (On-Balance Volume)
        sign = np.sign(df['close'].diff())
        out['obv'] = (sign * df['volume']).cumsum()
        out['obv'] = out['obv'].fillna(0)

        # Calculate Bollinger Bands
        mid = df['close'].rolling(self.bb_periods[0]).mean()
        std = df['close'].rolling(self.bb_periods[0]).std(ddof=0)
        upper = mid + self.bb_periods[1] * std
        lower = mid - self.bb_periods[1] * std

        # Handle potential division by zero in Bollinger Bands width calculation
        bb_width = np.zeros_like(mid.values)
        valid_mid = ~np.isnan(mid) & (mid > 0)
        if np.any(valid_mid):
            # Safely calculate width
            bb_width_valid = (upper - lower) / np.maximum(mid, 1e-10)
            bb_width_valid = np.where(np.isfinite(bb_width_valid), bb_width_valid, 0)
            bb_width[valid_mid] = bb_width_valid[valid_mid]

        out['bb_width'] = bb_width
        out['bb_upper'] = upper
        out['bb_lower'] = lower
        out['bb_mid'] = mid

        # Calculate ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        out[f'atr_{self.atr_period}'] = tr.rolling(self.atr_period).mean()

        # Calculate RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        # Safe division for RSI calculation
        rs = np.zeros_like(gain.values)
        valid_loss = ~np.isnan(loss) & (loss > 0)
        if np.any(valid_loss):
            rs_valid = gain[valid_loss] / loss[valid_loss]
            rs_valid = np.where(np.isfinite(rs_valid), rs_valid, 0)
            rs[valid_loss] = rs_valid

        out['rsi_14'] = 100 - (100 / (1 + rs))
        out['rsi_14'] = out['rsi_14'].fillna(50)  # Fill NaN with neutral RSI value

        # Calculate MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_26 = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        out['macd'] = ema_12 - ema_26
        out['macd_signal'] = out['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        out['macd_histogram'] = out['macd'] - out['macd_signal']

        # Calculate CMF (Chaikin Money Flow)
        multiplier = np.zeros_like(df['high'].values)
        valid_range = (df['high'] != df['low'])
        if np.any(valid_range):
            # Safely calculate money flow multiplier
            high_low_diff = np.maximum(df['high'] - df['low'], 1e-10)
            multiplier_valid = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
            multiplier_valid = np.where(np.isfinite(multiplier_valid), multiplier_valid, 0)
            multiplier[valid_range] = multiplier_valid[valid_range]

        money_flow_volume = multiplier * df['volume']
        cmf_sum = money_flow_volume.rolling(self.cmf_period).sum()
        volume_sum = df['volume'].rolling(self.cmf_period).sum()

        cmf = np.zeros_like(cmf_sum.values)
        valid_volume = ~np.isnan(volume_sum) & (volume_sum > 0)
        if np.any(valid_volume):
            cmf_valid = cmf_sum[valid_volume] / volume_sum[valid_volume]
            cmf_valid = np.where(np.isfinite(cmf_valid), cmf_valid, 0)
            cmf[valid_volume] = cmf_valid

        out['cmf'] = cmf
        out['cmf'] = out['cmf'].fillna(0)

        # Calculate MFI (Money Flow Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        price_diff = typical_price.diff()
        positive_flow = price_diff.copy()
        positive_flow[price_diff <= 0] = 0
        negative_flow = price_diff.copy()
        negative_flow[price_diff >= 0] = 0

        positive_sum = (positive_flow * raw_money_flow).rolling(self.mfi_period).sum()
        negative_sum = (-negative_flow * raw_money_flow).rolling(self.mfi_period).sum()

        mfi = np.full_like(positive_sum.values, 50)  # Default to neutral 50
        valid_sums = ~np.isnan(positive_sum) & ~np.isnan(negative_sum)
        if np.any(valid_sums):
            sum_total = np.maximum(positive_sum + negative_sum, 1e-10)
            mfi_valid = 100 * (positive_sum / sum_total)
            mfi_valid = np.where(np.isfinite(mfi_valid), mfi_valid, 50)
            mfi[valid_sums] = mfi_valid[valid_sums]

        out['mfi'] = mfi
        out['mfi'] = out['mfi'].fillna(50)  # Fill NaN with neutral MFI value

        # Calculate VWAP (Volume Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_price_volume = (typical_price * df['volume']).rolling(self.vwap_period).sum()
        cumulative_volume = df['volume'].rolling(self.vwap_period).sum()

        vwap = df['close'].copy()  # Default to close price
        valid_volume = ~np.isnan(cumulative_volume) & (cumulative_volume > 0)
        if np.any(valid_volume):
            vwap_valid = cumulative_price_volume / np.maximum(cumulative_volume, 1e-10)
            vwap_valid = np.where(np.isfinite(vwap_valid), vwap_valid, df['close'])
            vwap[valid_volume] = vwap_valid[valid_volume]

        out['vwap'] = vwap
        out['vwap'] = out['vwap'].fillna(df['close'])  # Fill NaN with close price

        # Final sanity check for any remaining infinities
        out = out.replace([np.inf, -np.inf], np.nan)

        # Make sure column names are lowercase
        out.columns = [col.lower() for col in out.columns]

        return out

    def _compute_market_regime(self, df: pd.DataFrame) -> np.ndarray:
        if 'market_regime' in df.columns:
            return df['market_regime']

        regime = np.zeros(len(df))

        if 'm30_ema_20' in df.columns:
            price = df['close'].values
            ema_20 = df['m30_ema_20'].values

            for i in range(len(df)):
                if pd.notna(price[i]) and pd.notna(ema_20[i]) and ema_20[i] > 0:
                    deviation = (price[i] / ema_20[i]) - 1
                    strength = min(1.0, max(-1.0, deviation * 7))
                    regime[i] = strength

        elif 'ema_20' in df.columns:
            price = df['close'].values
            ema_20 = df['ema_20'].values

            for i in range(len(df)):
                if pd.notna(price[i]) and pd.notna(ema_20[i]) and ema_20[i] > 0:
                    deviation = (price[i] / ema_20[i]) - 1
                    strength = min(1.0, max(-1.0, deviation * 7))
                    regime[i] = strength

        elif 'close' in df.columns and len(df) >= 20:
            ema_values = df['close'].ewm(span=20, adjust=False).mean().values
            for i in range(20, len(df)):
                price = df['close'].iloc[i]
                ema_20 = ema_values[i]
                if ema_20 > 0:
                    deviation = (price / ema_20) - 1
                    strength = min(1.0, max(-1.0, deviation * 7))
                    regime[i] = strength

        return regime

    def _compute_volatility_regime(self, df: pd.DataFrame) -> np.ndarray:
        if 'volatility_regime' in df.columns:
            return df['volatility_regime']

        regime = np.full(len(df), 0.5)

        if 'm30_bb_width' in df.columns:
            bb_width = df['m30_bb_width'].values

            for i in range(len(df)):
                if pd.notna(bb_width[i]):
                    scaled_vol = min(0.9, max(0.2, (bb_width[i] - 0.01) * 20))
                    regime[i] = scaled_vol

        elif 'atr_14' in df.columns:
            atr = df['atr_14'].values
            close = df['close'].values

            for i in range(len(df)):
                if pd.notna(atr[i]) and pd.notna(close[i]) and close[i] > 0:
                    atr_pct = atr[i] / close[i]
                    if atr_pct < 0.01:
                        regime[i] = 0.3
                    elif atr_pct > 0.03:
                        regime[i] = 0.8
                    else:
                        regime[i] = 0.3 + 0.5 * ((atr_pct - 0.01) / 0.02)

        return regime

    def _filter_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.lower() for col in df.columns]

        required_features = [
            'm30_ema_20', 'm30_bb_lower', 'm30_bb_width',
            'm30_bb_upper', 'm30_obv', 'm30_bb_mid',
            'open', 'high', 'low', 'close', 'volume', 'atr_14',
            'm30_rsi_14', 'm30_macd', 'm30_macd_signal', 'm30_macd_histogram',
            'm30_cmf', 'm30_mfi', 'm30_vwap'
        ]

        self.required_model_features = required_features.copy()

        if 'atr_14' not in df.columns and 'm30_atr_14' in df.columns:
            df['atr_14'] = df['m30_atr_14']

        actual_columns = [col for col in df.columns if col.startswith('actual_')]

        existing_columns = [col for col in required_features if col in df.columns]

        if len(existing_columns) < len(required_features):
            missing_columns = set(required_features) - set(existing_columns)
            self.logger.warning(f"Some required columns are missing from the dataframe: {missing_columns}")

        self.logger.info(f"Using indicators for model: {existing_columns}")

        filtered_df = df[existing_columns + actual_columns].copy()

        return filtered_df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Save actual values for reference
        df['actual_open'] = df['open']
        df['actual_high'] = df['high']
        df['actual_low'] = df['low']
        df['actual_close'] = df['close']

        # Select numeric columns excluding actual values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col.startswith('actual_')]

        # Replace inf/-inf with NaN and log the occurrence
        inf_count = np.isinf(df[numeric_cols].values).sum()
        if inf_count > 0:
            self.logger.warning(f"Found {inf_count} infinite values before scaling, replacing with NaN")
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Check for NaN values
        nan_count = df[numeric_cols].isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values before scaling, filling them")
            # Fill NaN values with appropriate methods
            for col in numeric_cols:
                if df[col].isna().any():
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        # Use forward fill for price data
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    else:
                        # Use median for indicators (more robust than mean)
                        median_val = df[col].median()
                        if pd.isna(median_val):  # If median is also NaN
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna(median_val)

        # Handle extreme values by clipping to reasonable percentiles
        for col in numeric_cols:
            try:
                q_low = df[col].quantile(0.001)
                q_high = df[col].quantile(0.999)
                if not pd.isna(q_low) and not pd.isna(q_high):
                    df[col] = df[col].clip(q_low, q_high)
            except Exception as e:
                self.logger.warning(f"Error clipping column {col}: {e}")

        # Final check for any remaining inf or NaN values
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Now perform scaling with proper error handling
        try:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        except Exception as e:
            self.logger.error(f"Error during scaling: {e}, falling back to manual scaling")
            # If standard scaling fails, do manual z-score scaling
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0  # For constant columns

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        return df

    def _fill_nans(self, df: pd.DataFrame, critical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df_copy = df.copy()

        df_copy = df_copy.ffill().bfill().fillna(0)

        if critical_cols:
            df_clean = df_copy.dropna(subset=critical_cols)
            if len(df_clean) < len(df_copy):
                self.logger.info(f"Dropped {len(df_copy) - len(df_clean)} rows with NaN in critical columns")
                return df_clean

        df_copy.columns = [col.lower() for col in df_copy.columns]

        return df_copy