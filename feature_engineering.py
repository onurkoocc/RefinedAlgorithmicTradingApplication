import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from scipy import stats


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("FeatureEngineer")

        self.feature_scaling = config.get("feature_engineering", "use_feature_scaling", True)
        self.use_chunking = config.get("feature_engineering", "use_chunking", True)
        self.chunk_size = config.get("feature_engineering", "chunk_size", 2000)

        self.indicators = [
            # Trend-based indicators
            "ema_5", "ema_10", "ema_20",
            "macd", "macd_signal", "macd_histogram",
            "macd_fast", "macd_fast_signal", "macd_fast_histogram",
            "parabolic_sar",
            "adx", "plus_di", "minus_di",
            "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b", "ichimoku_chikou",
            "linear_regression_slope",

            # Momentum-based indicators
            "rsi_14", "rsi_7",
            "stoch_k", "stoch_d",
            "stoch_k_fast", "stoch_d_fast",
            "mfi",

            # Volatility indicators
            "atr_14",
            "bb_upper", "bb_lower", "bb_middle", "bb_width", "bb_percent_b",
            "donchian_high", "donchian_low", "donchian_middle",
            "keltner_upper", "keltner_lower", "keltner_middle",

            # Volume indicators
            "obv",
            "volume_oscillator",
            "cmf",
            "force_index",
            "vwap",
            "net_taker_flow"
        ]

        self.essential_features = [
            "open", "high", "low", "close", "volume",
            "ema_5", "ema_10", "ema_20",
            "macd", "macd_signal", "macd_histogram",
            "macd_fast", "macd_fast_signal", "macd_fast_histogram",
            "parabolic_sar", "adx",
            "rsi_14", "rsi_7", "stoch_k", "stoch_d",
            "atr_14", "bb_width", "bb_percent_b",
            "donchian_middle", "keltner_middle",
            "obv", "cmf", "vwap", "net_taker_flow"
        ]

    def process_features(self, df_30m: pd.DataFrame) -> pd.DataFrame:
        df_30m.columns = [col.lower() for col in df_30m.columns]

        self.logger.info(f"Available columns in input data: {df_30m.columns.tolist()}")

        taker_volume_columns = []
        for col in df_30m.columns:
            if 'taker' in col.lower():
                taker_volume_columns.append(col)
                self.logger.info(f"Found taker volume column: {col}")

        if self.use_chunking and len(df_30m) > self.chunk_size:
            final_df = self._process_data_in_chunks(df_30m, chunk_size=self.chunk_size)
        else:
            final_df = self._process_data_combined(df_30m)

        if final_df.empty:
            self.logger.warning("Empty dataframe after initial processing")
            return pd.DataFrame()

        inf_check = np.isinf(final_df.select_dtypes(include=[np.number]).values).sum()
        if inf_check > 0:
            self.logger.warning(f"Found {inf_check} infinite values before filtering indicators, replacing with NaN")
            final_df = final_df.replace([np.inf, -np.inf], np.nan)

        if taker_volume_columns:
            self.logger.info(f"Preserving taker volume columns: {taker_volume_columns}")
            for col in taker_volume_columns:
                if col in df_30m.columns and col not in final_df.columns:
                    final_df[col] = df_30m[col]

        final_df = self._filter_indicators(final_df)
        final_df = self.compute_advanced_features(final_df)

        if self.feature_scaling:
            numeric_cols = final_df.select_dtypes(include=[np.number]).columns
            inf_count = np.isinf(final_df[numeric_cols].values).sum()
            nan_count = np.isnan(final_df[numeric_cols].values).sum()

            if inf_count > 0 or nan_count > 0:
                self.logger.warning(f"Found {inf_count} infinite values and {nan_count} NaN values before scaling")
                final_df = final_df.replace([np.inf, -np.inf], np.nan)

            final_df = self._scale_features(final_df)

        final_df.columns = [col.lower() for col in final_df.columns]

        self.logger.info(f"Processed {len(final_df)} rows with {len(final_df.columns)} features")
        return final_df

    def compute_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_market_regime(df)
        df = self._add_volatility_regime(df)
        return df

    def _add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'market_regime' in df.columns:
            return df

        df['market_regime'] = np.zeros(len(df))

        if 'ema_20' in df.columns:
            price = df['close'].values
            ema_20 = df['ema_20'].values

            valid_idx = ~np.isnan(price) & ~np.isnan(ema_20) & (ema_20 > 0)
            if np.any(valid_idx):
                deviation = np.zeros_like(price)
                deviation[valid_idx] = (price[valid_idx] / ema_20[valid_idx]) - 1
                df['market_regime'] = np.clip(deviation * 7, -1.0, 1.0)

        return df

    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'volatility_regime' in df.columns:
            return df

        df['volatility_regime'] = np.full(len(df), 0.5)

        if 'bb_width' in df.columns:
            bb_width = df['bb_width'].values
            valid_idx = ~np.isnan(bb_width)
            if np.any(valid_idx):
                vol_regime = np.full_like(bb_width, 0.5)
                vol_regime[valid_idx] = np.clip((bb_width[valid_idx] - 0.01) * 20, 0.2, 0.9)
                df['volatility_regime'] = vol_regime
        elif 'atr_14' in df.columns:
            atr = df['atr_14'].values
            close = df['close'].values

            valid_idx = ~np.isnan(atr) & ~np.isnan(close) & (close > 0)
            if np.any(valid_idx):
                atr_pct = np.zeros_like(atr)
                atr_pct[valid_idx] = atr[valid_idx] / close[valid_idx]

                vol_regime = np.full_like(atr, 0.5)
                vol_regime[valid_idx & (atr_pct < 0.01)] = 0.3
                vol_regime[valid_idx & (atr_pct > 0.03)] = 0.8
                vol_regime[valid_idx & (atr_pct >= 0.01) & (atr_pct <= 0.03)] = 0.3 + 0.5 * (
                            (atr_pct[valid_idx & (atr_pct >= 0.01) & (atr_pct <= 0.03)] - 0.01) / 0.02)

                df['volatility_regime'] = vol_regime

        return df

    def _process_data_combined(self, df_30m: pd.DataFrame) -> pd.DataFrame:
        df_30m = df_30m.copy()
        df_30m.columns = [col.lower() for col in df_30m.columns]

        self.logger.info(f"Columns in process_data_combined input: {df_30m.columns.tolist()}")

        df_30m.replace([np.inf, -np.inf], np.nan, inplace=True)
        if not isinstance(df_30m.index, pd.DatetimeIndex):
            df_30m.index = pd.to_datetime(df_30m.index)

        feat_30m = self._compute_core_indicators(df_30m).add_prefix('m30_')
        feat_30m[['open', 'high', 'low', 'close', 'volume']] = df_30m[['open', 'high', 'low', 'close', 'volume']]

        taker_columns = [col for col in df_30m.columns if 'taker' in col.lower()]
        for col in taker_columns:
            feat_30m[col] = df_30m[col]
            self.logger.info(f"Preserved taker column: {col}")

        for base, prefixed in [
            ('ema_5', 'm30_ema_5'), ('ema_10', 'm30_ema_10'), ('ema_20', 'm30_ema_20'),
            ('rsi_7', 'm30_rsi_7'), ('rsi_14', 'm30_rsi_14'),
            ('macd', 'm30_macd'), ('macd_signal', 'm30_macd_signal'), ('macd_histogram', 'm30_macd_histogram'),
            ('atr_14', 'm30_atr_14'), ('bb_width', 'm30_bb_width'), ('adx', 'm30_adx')
        ]:
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

        return combined

    def _process_data_in_chunks(self, df_30m: pd.DataFrame, chunk_size: int = 2000) -> pd.DataFrame:
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

        out['open'] = df['open']
        out['high'] = df['high']
        out['low'] = df['low']
        out['close'] = df['close']
        out['volume'] = df['volume']

        # Trend-based indicators
        self._compute_trend_indicators(df, out)

        # Momentum-based indicators
        self._compute_momentum_indicators(df, out)

        # Volatility indicators
        self._compute_volatility_indicators(df, out)

        # Volume indicators
        self._compute_volume_indicators(df, out)

        # Final sanity check for infinities
        out = out.replace([np.inf, -np.inf], np.nan)
        out.columns = [col.lower() for col in out.columns]

        return out

    def _compute_trend_indicators(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        # EMA calculations (5, 10, 20)
        out['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        out['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        out['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

        # MACD standard (12, 26, 9)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        out['macd'] = ema_12 - ema_26
        out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
        out['macd_histogram'] = out['macd'] - out['macd_signal']

        # MACD Fast (5, 10, 5)
        ema_5 = df['close'].ewm(span=5, adjust=False).mean()
        ema_10 = df['close'].ewm(span=10, adjust=False).mean()
        out['macd_fast'] = ema_5 - ema_10
        out['macd_fast_signal'] = out['macd_fast'].ewm(span=5, adjust=False).mean()
        out['macd_fast_histogram'] = out['macd_fast'] - out['macd_fast_signal']

        # Parabolic SAR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        acceleration_factor = 0.02
        acceleration_max = 0.2
        sar = np.zeros_like(close)
        trend = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
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

                if trend[i - 1] == 1:  # Uptrend
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
                else:  # Downtrend
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

        out['parabolic_sar'] = sar

        # ADX (14)
        tr = pd.DataFrame(index=df.index)
        tr['hl'] = df['high'] - df['low']
        tr['hc'] = (df['high'] - df['close'].shift(1)).abs()
        tr['lc'] = (df['low'] - df['close'].shift(1)).abs()
        tr = tr.max(axis=1)

        plus_dm = df['high'] - df['high'].shift(1)
        minus_dm = df['low'].shift(1) - df['low']

        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

        smoothed_tr = tr.rolling(window=14).sum()
        smoothed_plus_dm = plus_dm.rolling(window=14).sum()
        smoothed_minus_dm = minus_dm.rolling(window=14).sum()

        # Safe division for +DI and -DI
        plus_di = np.zeros_like(smoothed_tr.values)
        minus_di = np.zeros_like(smoothed_tr.values)

        valid_tr = ~np.isnan(smoothed_tr) & (smoothed_tr > 0)
        if np.any(valid_tr):
            plus_di[valid_tr] = 100 * (smoothed_plus_dm.values[valid_tr] / smoothed_tr.values[valid_tr])
            minus_di[valid_tr] = 100 * (smoothed_minus_dm.values[valid_tr] / smoothed_tr.values[valid_tr])

        out['plus_di'] = plus_di
        out['minus_di'] = minus_di

        # Safe calculation of DX and ADX
        dx = np.zeros_like(plus_di)
        valid_sum = (plus_di + minus_di) > 0
        if np.any(valid_sum):
            dx[valid_sum] = 100 * np.abs(plus_di[valid_sum] - minus_di[valid_sum]) / (
                        plus_di[valid_sum] + minus_di[valid_sum])

        out['adx'] = pd.Series(dx, index=df.index).ewm(span=14, adjust=False).mean().fillna(25)

        # Ichimoku Cloud (5, 10, 20)
        tenkan_high = df['high'].rolling(window=5).max()
        tenkan_low = df['low'].rolling(window=5).min()
        out['ichimoku_tenkan'] = (tenkan_high + tenkan_low) / 2

        kijun_high = df['high'].rolling(window=10).max()
        kijun_low = df['low'].rolling(window=10).min()
        out['ichimoku_kijun'] = (kijun_high + kijun_low) / 2

        out['ichimoku_senkou_a'] = ((out['ichimoku_tenkan'] + out['ichimoku_kijun']) / 2).shift(10)

        senkou_b_high = df['high'].rolling(window=20).max()
        senkou_b_low = df['low'].rolling(window=20).min()
        out['ichimoku_senkou_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(10)

        out['ichimoku_chikou'] = df['close'].shift(-10)

        # Linear Regression Slope (14)
        window = 14
        slopes = np.zeros(len(df))

        for i in range(window, len(df)):
            y = df['close'].values[i - window:i]
            x = np.arange(window)
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes[i] = slope

        out['linear_regression_slope'] = slopes

    def _compute_momentum_indicators(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        # RSI (14 and 7)
        for period in [7, 14]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = np.zeros_like(avg_gain.values)
            valid_loss = ~np.isnan(avg_loss) & (avg_loss > 0)
            if np.any(valid_loss):
                rs[valid_loss] = avg_gain.values[valid_loss] / avg_loss.values[valid_loss]

            rsi = 100 - (100 / (1 + rs))
            out[f'rsi_{period}'] = pd.Series(rsi, index=df.index).fillna(50)

        # Stochastic Oscillator (14,3,3)
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()

        k_fast = np.zeros_like(df['close'].values)
        valid_range = ~np.isnan(high_14) & ~np.isnan(low_14) & (high_14 != low_14)
        if np.any(valid_range):
            k_fast[valid_range] = 100 * ((df['close'].values[valid_range] - low_14.values[valid_range]) /
                                         (high_14.values[valid_range] - low_14.values[valid_range]))

        out['stoch_k'] = pd.Series(k_fast, index=df.index).rolling(window=3).mean()
        out['stoch_d'] = out['stoch_k'].rolling(window=3).mean()

        # Stochastic Oscillator Fast (5,3,3)
        high_5 = df['high'].rolling(window=5).max()
        low_5 = df['low'].rolling(window=5).min()

        k_fast_5 = np.zeros_like(df['close'].values)
        valid_range_5 = ~np.isnan(high_5) & ~np.isnan(low_5) & (high_5 != low_5)
        if np.any(valid_range_5):
            k_fast_5[valid_range_5] = 100 * ((df['close'].values[valid_range_5] - low_5.values[valid_range_5]) /
                                             (high_5.values[valid_range_5] - low_5.values[valid_range_5]))

        out['stoch_k_fast'] = pd.Series(k_fast_5, index=df.index).rolling(window=3).mean()
        out['stoch_d_fast'] = out['stoch_k_fast'].rolling(window=3).mean()

        # MFI (14)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        price_change = typical_price.diff()

        positive_flow = (price_change > 0) * money_flow
        negative_flow = (price_change < 0) * money_flow

        positive_sum = positive_flow.rolling(window=14).sum()
        negative_sum = negative_flow.rolling(window=14).sum()

        mfi = np.zeros_like(positive_sum.values)
        valid_negative = ~np.isnan(negative_sum) & (negative_sum > 0)
        if np.any(valid_negative):
            money_ratio = positive_sum.values[valid_negative] / negative_sum.values[valid_negative]
            mfi[valid_negative] = 100 - (100 / (1 + money_ratio))

        out['mfi'] = pd.Series(mfi, index=df.index).fillna(50)

    def _compute_volatility_indicators(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        # ATR (14)
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        out['atr_14'] = tr.rolling(window=14).mean()

        # Bollinger Bands (20,2)
        bb_sma = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std(ddof=0)

        out['bb_middle'] = bb_sma
        out['bb_upper'] = bb_sma + (2 * bb_std)
        out['bb_lower'] = bb_sma - (2 * bb_std)

        # Safe calculation of BB Width
        bb_width = np.zeros_like(bb_sma.values)
        valid_sma = ~np.isnan(bb_sma) & (bb_sma > 0)
        if np.any(valid_sma):
            bb_width[valid_sma] = (out['bb_upper'].values[valid_sma] - out['bb_lower'].values[valid_sma]) / \
                                  bb_sma.values[valid_sma]

        out['bb_width'] = pd.Series(bb_width, index=df.index)

        # %B Indicator
        bb_percent_b = np.zeros_like(bb_sma.values)
        valid_range = ~np.isnan(out['bb_upper']) & ~np.isnan(out['bb_lower']) & (out['bb_upper'] != out['bb_lower'])
        if np.any(valid_range):
            bb_percent_b[valid_range] = ((df['close'].values[valid_range] - out['bb_lower'].values[valid_range]) /
                                         (out['bb_upper'].values[valid_range] - out['bb_lower'].values[valid_range]))

        out['bb_percent_b'] = pd.Series(bb_percent_b, index=df.index).fillna(0.5)

        # Donchian Channel (20)
        out['donchian_high'] = df['high'].rolling(window=20).max()
        out['donchian_low'] = df['low'].rolling(window=20).min()
        out['donchian_middle'] = (out['donchian_high'] + out['donchian_low']) / 2

        # Keltner Channel (20, ATR multiplier 2)
        keltner_middle = df['close'].rolling(window=20).mean()
        keltner_atr = out['atr_14']

        out['keltner_middle'] = keltner_middle
        out['keltner_upper'] = keltner_middle + (2 * keltner_atr)
        out['keltner_lower'] = keltner_middle - (2 * keltner_atr)

    def _compute_volume_indicators(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        # OBV (On-Balance Volume)
        close_diff = df['close'].diff()
        obv = np.zeros(len(df))

        obv[0] = df['volume'].iloc[0]
        for i in range(1, len(df)):
            if close_diff.iloc[i] > 0:
                obv[i] = obv[i - 1] + df['volume'].iloc[i]
            elif close_diff.iloc[i] < 0:
                obv[i] = obv[i - 1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i - 1]

        out['obv'] = obv

        # Volume Oscillator (10,20)
        vol_sma_10 = df['volume'].rolling(window=10).mean()
        vol_sma_20 = df['volume'].rolling(window=20).mean()

        vol_osc = np.zeros_like(vol_sma_20.values)
        valid_vol = ~np.isnan(vol_sma_20) & (vol_sma_20 > 0)
        if np.any(valid_vol):
            vol_osc[valid_vol] = ((vol_sma_10.values[valid_vol] - vol_sma_20.values[valid_vol]) /
                                  vol_sma_20.values[valid_vol]) * 100

        out['volume_oscillator'] = pd.Series(vol_osc, index=df.index)

        # CMF (14)
        high_low_diff = df['high'] - df['low']
        money_flow_multiplier = np.zeros_like(high_low_diff.values)

        valid_range = ~np.isnan(high_low_diff) & (high_low_diff > 0)
        if np.any(valid_range):
            money_flow_multiplier[valid_range] = ((df['close'].values[valid_range] - df['low'].values[valid_range]) -
                                                  (df['high'].values[valid_range] - df['close'].values[valid_range])) / \
                                                 high_low_diff.values[valid_range]

        money_flow_volume = money_flow_multiplier * df['volume'].values

        cmf_sum = pd.Series(money_flow_volume, index=df.index).rolling(window=14).sum()
        vol_sum = df['volume'].rolling(window=14).sum()

        cmf = np.zeros_like(cmf_sum.values)
        valid_vol_sum = ~np.isnan(vol_sum) & (vol_sum > 0)
        if np.any(valid_vol_sum):
            cmf[valid_vol_sum] = cmf_sum.values[valid_vol_sum] / vol_sum.values[valid_vol_sum]

        out['cmf'] = pd.Series(cmf, index=df.index)

        # Force Index (13-period EMA of (Close - Prior Close) * Volume)
        force = (df['close'] - df['close'].shift(1)) * df['volume']
        out['force_index'] = force.ewm(span=13, adjust=False).mean()

        # VWAP (Volume Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_volume = typical_price * df['volume']

        cumulative_tp_volume = tp_volume.rolling(window=14).sum()
        cumulative_volume = df['volume'].rolling(window=14).sum()

        vwap = np.zeros_like(cumulative_volume.values)
        valid_vol = ~np.isnan(cumulative_volume) & (cumulative_volume > 0)
        if np.any(valid_vol):
            vwap[valid_vol] = cumulative_tp_volume.values[valid_vol] / cumulative_volume.values[valid_vol]

        out['vwap'] = pd.Series(vwap, index=df.index).fillna(df['close'])

        # Net Taker Flow = 2*taker_buy_base_asset_volume - volume
        if 'taker_buy_base_asset_volume' in df.columns:
            out['net_taker_flow'] = 2 * df['taker_buy_base_asset_volume'] - df['volume']
        else:
            # Estimate if actual taker volume not available
            price_change = df['close'].pct_change()
            out['net_taker_flow'] = df['volume'] * np.sign(price_change)

    def _filter_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.lower() for col in df.columns]

        required_features = [
            'm30_ema_5', 'm30_ema_10', 'm30_ema_20',
            'm30_macd', 'm30_macd_signal', 'm30_macd_histogram',
            'm30_macd_fast', 'm30_macd_fast_signal', 'm30_macd_fast_histogram',
            'm30_parabolic_sar', 'm30_adx', 'm30_plus_di', 'm30_minus_di',
            'm30_rsi_14', 'm30_rsi_7', 'm30_stoch_k', 'm30_stoch_d',
            'm30_atr_14', 'm30_bb_width', 'm30_bb_percent_b',
            'm30_donchian_middle', 'm30_keltner_middle',
            'm30_obv', 'm30_cmf', 'm30_vwap', 'm30_net_taker_flow',
            'open', 'high', 'low', 'close', 'volume'
        ]

        taker_columns = [col for col in df.columns if 'taker' in col.lower()]
        for col in taker_columns:
            if col not in required_features:
                required_features.append(col)
                self.logger.info(f"Added taker column to required features: {col}")

        self.required_model_features = required_features.copy()

        existing_columns = [col for col in required_features if col in df.columns]
        for col in taker_columns:
            if col not in existing_columns:
                existing_columns.append(col)

        if len(existing_columns) < len(required_features):
            missing_columns = set(required_features) - set(existing_columns)
            self.logger.warning(f"Some required columns are missing from the dataframe: {missing_columns}")

        self.logger.info(f"Using indicators for model: {existing_columns}")

        actual_columns = [col for col in df.columns if col.startswith('actual_')]
        all_columns = existing_columns + actual_columns + taker_columns

        filtered_columns = []
        for col in all_columns:
            if col not in filtered_columns:
                filtered_columns.append(col)

        filtered_df = df[filtered_columns].copy()
        self.logger.info(f"Filtered dataframe columns: {filtered_df.columns.tolist()}")

        return filtered_df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['actual_open'] = df['open']
        df['actual_high'] = df['high']
        df['actual_low'] = df['low']
        df['actual_close'] = df['close']

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col.startswith('actual_')]

        inf_count = np.isinf(df[numeric_cols].values).sum()
        if inf_count > 0:
            self.logger.warning(f"Found {inf_count} infinite values before scaling, replacing with NaN")
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        nan_count = df[numeric_cols].isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values before scaling, filling them")
            for col in numeric_cols:
                if df[col].isna().any():
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    else:
                        median_val = df[col].median()
                        if pd.isna(median_val):
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna(median_val)

        for col in numeric_cols:
            try:
                q_low = df[col].quantile(0.001)
                q_high = df[col].quantile(0.999)
                if not pd.isna(q_low) and not pd.isna(q_high):
                    df[col] = df[col].clip(q_low, q_high)
            except Exception as e:
                self.logger.warning(f"Error clipping column {col}: {e}")

        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        df[numeric_cols] = df[numeric_cols].fillna(0)

        try:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        except Exception as e:
            self.logger.error(f"Error during scaling: {e}, falling back to manual scaling")
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0

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