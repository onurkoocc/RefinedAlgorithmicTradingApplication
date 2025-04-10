import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any


class IndicatorUtil:
    def __init__(self):
        self.indicators_cache = {}

    def reset_cache(self):
        self.indicators_cache = {}

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df = self.calculate_time_features(df)
        df = self.calculate_ema(df, [9, 21, 50])
        df = self.calculate_rsi(df, 14)
        df = self.calculate_bollinger_bands(df, 20, 2)
        df = self.calculate_atr(df, 14)
        df = self.calculate_obv(df)
        df = self.calculate_cmf(df, 20)
        df = self.calculate_adx(df, 14)
        df = self.calculate_macd(df, 12, 26, 9)
        df = self.calculate_sma(df, [200])
        df = self.calculate_bb_width(df)

        return df

    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        if not isinstance(df_out.index, pd.DatetimeIndex):
            try:
                df_out.index = pd.to_datetime(df_out.index)
            except:
                return df_out

        hours = df_out.index.hour
        df_out['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
        df_out['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

        day_of_week = df_out.index.dayofweek
        df_out['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
        df_out['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)

        return df_out

    def calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        df_out = df.copy()
        for period in periods:
            col_name = f'ema_{period}'
            if col_name in self.indicators_cache and len(self.indicators_cache[col_name]) == len(df):
                df_out[col_name] = self.indicators_cache[col_name]
                continue

            df_out[col_name] = df['close'].ewm(span=period, adjust=False).mean()
            self.indicators_cache[col_name] = df_out[col_name].values

        return df_out

    def calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        df_out = df.copy()
        for period in periods:
            col_name = f'sma_{period}'
            if col_name in self.indicators_cache and len(self.indicators_cache[col_name]) == len(df):
                df_out[col_name] = self.indicators_cache[col_name]
                continue

            df_out[col_name] = df['close'].rolling(window=period).mean()
            self.indicators_cache[col_name] = df_out[col_name].values

        return df_out

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df_out = df.copy()
        col_name = f'rsi_{period}'

        if col_name in self.indicators_cache and len(self.indicators_cache[col_name]) == len(df):
            df_out[col_name] = self.indicators_cache[col_name]
            return df_out

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
        df_out[col_name] = pd.Series(rsi, index=df.index).fillna(50)

        self.indicators_cache[col_name] = df_out[col_name].values
        return df_out

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        df_out = df.copy()
        bb_middle = f'bb_middle_{period}'
        bb_upper = f'bb_upper_{period}'
        bb_lower = f'bb_lower_{period}'

        if (bb_middle in self.indicators_cache and bb_upper in self.indicators_cache and
                bb_lower in self.indicators_cache and len(self.indicators_cache[bb_middle]) == len(df)):
            df_out[bb_middle] = self.indicators_cache[bb_middle]
            df_out[bb_upper] = self.indicators_cache[bb_upper]
            df_out[bb_lower] = self.indicators_cache[bb_lower]
            return df_out

        df_out[bb_middle] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std(ddof=0)

        df_out[bb_upper] = df_out[bb_middle] + (std_dev * bb_std)
        df_out[bb_lower] = df_out[bb_middle] - (std_dev * bb_std)

        self.indicators_cache[bb_middle] = df_out[bb_middle].values
        self.indicators_cache[bb_upper] = df_out[bb_upper].values
        self.indicators_cache[bb_lower] = df_out[bb_lower].values
        return df_out

    def calculate_bb_width(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        if 'bb_width_20' in self.indicators_cache and len(self.indicators_cache['bb_width_20']) == len(df):
            df_out['bb_width_20'] = self.indicators_cache['bb_width_20']
            return df_out

        if 'bb_upper_20' not in df_out.columns or 'bb_lower_20' not in df_out.columns or 'bb_middle_20' not in df_out.columns:
            df_out = self.calculate_bollinger_bands(df_out)

        bb_width = np.zeros_like(df_out['bb_middle_20'].values)
        valid_sma = ~np.isnan(df_out['bb_middle_20']) & (df_out['bb_middle_20'] > 0)

        if np.any(valid_sma):
            bb_width[valid_sma] = (df_out['bb_upper_20'].values[valid_sma] - df_out['bb_lower_20'].values[valid_sma]) / \
                                df_out['bb_middle_20'].values[valid_sma]

        df_out['bb_width_20'] = pd.Series(bb_width, index=df.index)
        self.indicators_cache['bb_width_20'] = df_out['bb_width_20'].values
        return df_out

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df_out = df.copy()
        col_name = f'atr_{period}'

        if col_name in self.indicators_cache and len(self.indicators_cache[col_name]) == len(df):
            df_out[col_name] = self.indicators_cache[col_name]
            return df_out

        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        df_out[col_name] = tr.rolling(window=period).mean()
        self.indicators_cache[col_name] = df_out[col_name].values
        return df_out

    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        if 'obv' in self.indicators_cache and len(self.indicators_cache['obv']) == len(df):
            df_out['obv'] = self.indicators_cache['obv']
            return df_out

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

        df_out['obv'] = obv
        self.indicators_cache['obv'] = df_out['obv'].values
        return df_out

    def calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df_out = df.copy()
        col_name = f'cmf_{period}'

        if col_name in self.indicators_cache and len(self.indicators_cache[col_name]) == len(df):
            df_out[col_name] = self.indicators_cache[col_name]
            return df_out

        high_low_diff = df['high'] - df['low']
        money_flow_multiplier = np.zeros_like(high_low_diff.values)

        valid_range = ~np.isnan(high_low_diff) & (high_low_diff > 0)
        if np.any(valid_range):
            money_flow_multiplier[valid_range] = ((df['close'].values[valid_range] - df['low'].values[valid_range]) -
                                                (df['high'].values[valid_range] - df['close'].values[valid_range])) / \
                                                high_low_diff.values[valid_range]

        money_flow_volume = money_flow_multiplier * df['volume'].values

        cmf_sum = pd.Series(money_flow_volume, index=df.index).rolling(window=period).sum()
        vol_sum = df['volume'].rolling(window=period).sum()

        cmf = np.zeros_like(cmf_sum.values)
        valid_vol_sum = ~np.isnan(vol_sum) & (vol_sum > 0)
        if np.any(valid_vol_sum):
            cmf[valid_vol_sum] = cmf_sum.values[valid_vol_sum] / vol_sum.values[valid_vol_sum]

        df_out[col_name] = pd.Series(cmf, index=df.index)
        self.indicators_cache[col_name] = df_out[col_name].values
        return df_out

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df_out = df.copy()
        col_name = f'adx_{period}'
        plus_di = f'plus_di_{period}'
        minus_di = f'minus_di_{period}'

        if (col_name in self.indicators_cache and plus_di in self.indicators_cache and
                minus_di in self.indicators_cache and len(self.indicators_cache[col_name]) == len(df)):
            df_out[col_name] = self.indicators_cache[col_name]
            df_out[plus_di] = self.indicators_cache[plus_di]
            df_out[minus_di] = self.indicators_cache[minus_di]
            return df_out

        tr = pd.DataFrame(index=df.index)
        tr['hl'] = df['high'] - df['low']
        tr['hc'] = (df['high'] - df['close'].shift(1)).abs()
        tr['lc'] = (df['low'] - df['close'].shift(1)).abs()
        tr = tr.max(axis=1)

        plus_dm = df['high'] - df['high'].shift(1)
        minus_dm = df['low'].shift(1) - df['low']

        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

        smoothed_tr = tr.rolling(window=period).sum()
        smoothed_plus_dm = plus_dm.rolling(window=period).sum()
        smoothed_minus_dm = minus_dm.rolling(window=period).sum()

        plus_di_vals = np.zeros_like(smoothed_tr.values)
        minus_di_vals = np.zeros_like(smoothed_tr.values)

        valid_tr = ~np.isnan(smoothed_tr) & (smoothed_tr > 0)
        if np.any(valid_tr):
            plus_di_vals[valid_tr] = 100 * (smoothed_plus_dm.values[valid_tr] / smoothed_tr.values[valid_tr])
            minus_di_vals[valid_tr] = 100 * (smoothed_minus_dm.values[valid_tr] / smoothed_tr.values[valid_tr])

        df_out[plus_di] = plus_di_vals
        df_out[minus_di] = minus_di_vals

        dx = np.zeros_like(plus_di_vals)
        valid_sum = (plus_di_vals + minus_di_vals) > 0
        if np.any(valid_sum):
            dx[valid_sum] = 100 * np.abs(plus_di_vals[valid_sum] - minus_di_vals[valid_sum]) / (
                        plus_di_vals[valid_sum] + minus_di_vals[valid_sum])

        df_out[col_name] = pd.Series(dx, index=df.index).ewm(span=period, adjust=False).mean().fillna(25)

        self.indicators_cache[col_name] = df_out[col_name].values
        self.indicators_cache[plus_di] = df_out[plus_di].values
        self.indicators_cache[minus_di] = df_out[minus_di].values
        return df_out

    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                      signal_period: int = 9) -> pd.DataFrame:
        df_out = df.copy()
        macd_col = f'macd_{fast_period}_{slow_period}'
        macd_signal_col = f'macd_signal_{fast_period}_{slow_period}_{signal_period}'
        macd_hist_col = f'macd_histogram_{fast_period}_{slow_period}_{signal_period}'

        if (macd_col in self.indicators_cache and macd_signal_col in self.indicators_cache and
                macd_hist_col in self.indicators_cache and len(self.indicators_cache[macd_col]) == len(df)):
            df_out[macd_col] = self.indicators_cache[macd_col]
            df_out[macd_signal_col] = self.indicators_cache[macd_signal_col]
            df_out[macd_hist_col] = self.indicators_cache[macd_hist_col]
            return df_out

        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()

        df_out[macd_col] = ema_fast - ema_slow
        df_out[macd_signal_col] = df_out[macd_col].ewm(span=signal_period, adjust=False).mean()
        df_out[macd_hist_col] = df_out[macd_col] - df_out[macd_signal_col]

        self.indicators_cache[macd_col] = df_out[macd_col].values
        self.indicators_cache[macd_signal_col] = df_out[macd_signal_col].values
        self.indicators_cache[macd_hist_col] = df_out[macd_hist_col].values
        return df_out

    def detect_market_phase(self, df: pd.DataFrame) -> str:
        if df.empty or len(df) < 50:
            return "neutral"

        required_indicators = ['ema_9', 'ema_21', 'ema_50', 'adx_14', 'bb_width_20']
        for indicator in required_indicators:
            if indicator not in df.columns:
                df = self.calculate_all_indicators(df)
                break

        recent_close = df['close'].iloc[-1]
        recent_ema9 = df['ema_9'].iloc[-1]
        recent_ema21 = df['ema_21'].iloc[-1]
        recent_ema50 = df['ema_50'].iloc[-1]
        adx = df['adx_14'].iloc[-1]
        bb_width = df['bb_width_20'].iloc[-1]
        ema9_slope = (df['ema_9'].iloc[-1] / df['ema_9'].iloc[-5] - 1) * 100
        ema21_slope = (df['ema_21'].iloc[-1] / df['ema_21'].iloc[-5] - 1) * 100
        price_momentum = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100

        volume_trend = 0
        if 'volume' in df.columns:
            recent_vol = df['volume'].iloc[-3:].mean()
            past_vol = df['volume'].iloc[-8:-3].mean() if len(df) >= 8 else df['volume'].iloc[:5].mean()
            if past_vol > 0:
                volume_trend = (recent_vol / past_vol) - 1

        strong_uptrend = False
        strong_downtrend = False
        price_above_emas = recent_close > recent_ema9 > recent_ema21 > recent_ema50
        price_below_emas = recent_close < recent_ema9 < recent_ema21 < recent_ema50

        if price_above_emas and ema9_slope > 0.18 and ema21_slope > 0.08:
            if adx > 25 or (price_momentum > 0.2 and volume_trend > 0.1):
                strong_uptrend = True

        if price_below_emas and ema9_slope < -0.18 and ema21_slope < -0.08:
            if adx > 25 or (price_momentum < -0.2 and volume_trend > 0.1):
                strong_downtrend = True

        support_resistance_levels = self.identify_support_resistance(df)
        nearest_level, distance = self.get_nearest_level(recent_close, support_resistance_levels)

        is_ranging = abs(ema21_slope) < 0.18 and adx < 20 and distance < 0.022

        if is_ranging:
            if 'volume' in df.columns:
                last_10_volume = df['volume'].iloc[-10:].values
                volume_stable = np.std(last_10_volume) / np.mean(last_10_volume) < 0.5
                if not volume_stable:
                    is_ranging = False

            last_5_candles = df.iloc[-5:]
            range_size = (last_5_candles['high'].max() - last_5_candles['low'].min()) / recent_close
            if range_size > 0.03:
                is_ranging = False

        if strong_uptrend:
            return "uptrend"
        elif strong_downtrend:
            return "downtrend"
        elif is_ranging:
            if nearest_level > recent_close:
                return "ranging_at_support"
            else:
                return "ranging_at_resistance"
        else:
            return "neutral"

    def identify_support_resistance(self, df: pd.DataFrame, window: int = 10) -> List[float]:
        if len(df) < 100:
            return []

        levels = []
        pivots = self.find_pivot_points(df, window)

        for pivot in pivots:
            price_level = pivot['price']
            is_close_to_existing = False

            for existing_level in levels:
                proximity = abs(price_level / existing_level - 1)
                if proximity < 0.005:
                    is_close_to_existing = True
                    break

            if not is_close_to_existing:
                levels.append(price_level)

        return sorted(levels)

    def find_pivot_points(self, df: pd.DataFrame, window: int = 10) -> List[Dict]:
        pivots = []

        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == df['high'].iloc[i - window:i + window + 1].max():
                pivots.append({
                    'type': 'high',
                    'price': df['high'].iloc[i],
                    'index': i,
                    'strength': 1
                })

            if df['low'].iloc[i] == df['low'].iloc[i - window:i + window + 1].min():
                pivots.append({
                    'type': 'low',
                    'price': df['low'].iloc[i],
                    'index': i,
                    'strength': 1
                })

        return pivots

    def get_nearest_level(self, current_price: float, levels: List[float]) -> Tuple[float, float]:
        if not levels:
            return 0.0, 1.0

        nearest_level = levels[0]
        min_distance = abs(current_price / nearest_level - 1)

        for level in levels:
            distance = abs(current_price / level - 1)
            if distance < min_distance:
                min_distance = distance
                nearest_level = level

        return nearest_level, min_distance

    def detect_volatility_regime(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.5

        if 'bb_width_20' not in df.columns:
            df = self.calculate_bb_width(df)

        bb_width = df['bb_width_20'].iloc[-1]

        if np.isnan(bb_width):
            return 0.5

        if len(df) >= 20:
            returns = df['close'].pct_change().values[-20:]
            returns = returns[~np.isnan(returns)]
            hist_vol = np.std(returns) * np.sqrt(48)
        else:
            hist_vol = 0.01

        bb_vol = min(0.9, max(0.1, (bb_width - 0.01) * 10))
        hist_vol_scaled = min(0.9, max(0.1, hist_vol * 10))
        volatility_regime = 0.7 * bb_vol + 0.3 * hist_vol_scaled

        return volatility_regime