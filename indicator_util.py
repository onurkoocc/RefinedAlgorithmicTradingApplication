import numpy as np
import pandas as pd
import pandas_ta as ta
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

        # EMAs and SMAs
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.sma(length=200, append=True)

        # RSI
        df.ta.rsi(length=14, append=True)

        # Bollinger Bands
        bbands = df.ta.bbands(length=20, std=2, append=True)

        # ATR
        df.ta.atr(length=14, append=True)

        # OBV
        df.ta.obv(append=True)

        # CMF
        df.ta.cmf(length=20, append=True)

        # ADX
        adx_data = df.ta.adx(length=14, append=True)

        # MACD
        macd_data = df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # Rename columns to match expected names
        rename_map = {
            'EMA_9': 'ema_9',
            'EMA_21': 'ema_21',
            'EMA_50': 'ema_50',
            'SMA_200': 'sma_200',
            'RSI_14': 'rsi_14',
            'BBL_20_2.0': 'bb_lower_20',
            'BBM_20_2.0': 'bb_middle_20',
            'BBU_20_2.0': 'bb_upper_20',
            'BBB_20_2.0': 'bb_width_20',
            'BBP_20_2.0': 'bb_percent_b',
            'ATRr_14': 'atr_14',  # pandas-ta uses ATRr_14 for ATR
            'OBV': 'obv',
            'CMF_20': 'cmf_20',
            'ADX_14': 'adx_14',
            'DMP_14': 'plus_di_14',
            'DMN_14': 'minus_di_14',
            'MACD_12_26_9': 'macd_12_26',
            'MACDs_12_26_9': 'macd_signal_12_26_9',
            'MACDh_12_26_9': 'macd_histogram_12_26_9'
        }

        # Apply renaming
        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

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

    def detect_market_phase(self, df: pd.DataFrame) -> str:
        if df.empty or len(df) < 50:
            return "neutral"

        if not all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50', 'adx_14']):
            df = self.calculate_all_indicators(df)

        recent_close = df['close'].iloc[-1]
        recent_ema9 = df['ema_9'].iloc[-1]
        recent_ema21 = df['ema_21'].iloc[-1]
        recent_ema50 = df['ema_50'].iloc[-1]

        adx = df['adx_14'].iloc[-1]
        bb_width = df['bb_width_20'].iloc[-1] if 'bb_width_20' in df.columns else 0.02

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

    def get_signal_reason(self, ema_cross_up, ema_cross_down, macd_cross_up, macd_cross_down, rsi_oversold,
                          rsi_overbought) -> str:
        reasons = []

        if ema_cross_up:
            reasons.append("EMA9 crossed above EMA21")
        if ema_cross_down:
            reasons.append("EMA9 crossed below EMA21")
        if macd_cross_up:
            reasons.append("MACD crossed above Signal")
        if macd_cross_down:
            reasons.append("MACD crossed below Signal")
        if rsi_oversold:
            reasons.append("RSI oversold")
        if rsi_overbought:
            reasons.append("RSI overbought")

        if not reasons:
            return "No clear signal"

        return ", ".join(reasons)

    def check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        if len(df) < 5:
            return False

        if 'obv' not in df.columns:
            df.ta.obv(append=True)
            df.rename(columns={'OBV': 'obv'}, inplace=True)

        volumes = df['volume'].values[-5:]
        closes = df['close'].values[-5:]
        obv_values = df['obv'].values[-5:]

        price_change = closes[-1] / closes[-2] - 1
        obv_change = obv_values[-1] - obv_values[-2]

        short_avg_volume = np.mean(volumes[-4:-1])
        current_volume = volumes[-1]
        volume_sufficient = current_volume > short_avg_volume * 0.85

        if price_change > 0 and obv_change > 0 and volume_sufficient:
            return True

        if price_change < 0 and obv_change < 0 and volume_sufficient:
            return True

        return False

    def detect_volatility_regime(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.5

        if 'bb_width_20' not in df.columns:
            df.ta.bbands(length=20, std=2, append=True)
            df.rename(columns={'BBB_20_2.0': 'bb_width_20'}, inplace=True)

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