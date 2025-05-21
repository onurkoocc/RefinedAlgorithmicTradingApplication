import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any


class IndicatorUtil:
    def __init__(self):
        self.market_regime_util = None

    def calculate_specific_indicators(self, df: pd.DataFrame, indicator_configs: List[str]) -> pd.DataFrame:
        if df.empty: return df
        df_out = df.copy()

        required_base_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_base_cols:
            if col not in df_out.columns:
                # self.logger.warning(f"Missing base column {col} for indicator calculation.") # Assuming logger is not available here
                if col == 'volume':
                    df_out[col] = 0
                elif 'close' in df_out.columns:
                    df_out[col] = df_out['close']
                else: # If close is also missing, this is problematic
                    df_out[col] = 0


        for indicator_config in indicator_configs:
            parts = indicator_config.split('_')
            name = parts[0]
            params = [int(p) for p in parts[1:] if p.isdigit()]

            if name == "ema":
                if params: df_out = self.calculate_ema(df_out, params[0])
            elif name == "sma":
                if params: df_out = self.calculate_sma(df_out, params[0])
            elif name == "rsi":
                if params: df_out = self.calculate_rsi(df_out, params[0])
            elif name == "bb":
                if len(params) >= 1:
                    period = params[0]
                    std_dev = params[1] if len(params) > 1 else 2
                    df_out = self.calculate_bollinger_bands(df_out, period, std_dev)
                    df_out = self.calculate_bb_width(df_out, period) # Ensure bb_width is also calculated
            elif name == "atr":
                if params: df_out = self.calculate_atr(df_out, params[0])
            elif name == "obv":
                df_out = self.calculate_obv(df_out)
            elif name == "adx": # ADX also calculates +DI, -DI
                if params: df_out = self.calculate_adx(df_out, params[0])
            elif name == "macd":
                fast = params[0] if len(params) > 0 else 12
                slow = params[1] if len(params) > 1 else 26
                signal = params[2] if len(params) > 2 else 9
                df_out = self.calculate_macd(df_out, fast, slow, signal)
            elif name == "stoch": # Stochastic Oscillator
                k_period = params[0] if len(params) > 0 else 14
                d_period = params[1] if len(params) > 1 else 3
                df_out = self.calculate_stochastic_oscillator(df_out, k_period, d_period)
            elif name == "roc": # Rate of Change
                if params: df_out = self.calculate_roc(df_out, params[0])
            elif name == "cmf": # Chaikin Money Flow
                if params: df_out = self.calculate_cmf(df_out, params[0])
            elif name == "histvol": # Historical Volatility
                if params: df_out = self.calculate_historical_volatility(df_out, params[0])

        return df_out

    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        col_name = f'ema_{period}'
        if col_name not in df.columns:
            df[col_name] = df['close'].ewm(span=period, adjust=False, min_periods=max(1, period // 2)).mean()
        return df

    def calculate_sma(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        col_name = f'sma_{period}'
        if col_name not in df.columns:
            df[col_name] = df['close'].rolling(window=period, min_periods=max(1, period // 2)).mean()
        return df

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        col_name = f'rsi_{period}'
        if col_name not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=max(1, period // 2)).mean()
            avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=max(1, period // 2)).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            df[col_name] = 100 - (100 / (1 + rs))
            df[col_name] = df[col_name].fillna(50)
        return df

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        bb_middle_col = f'bb_middle_{period}'
        bb_upper_col = f'bb_upper_{period}'
        bb_lower_col = f'bb_lower_{period}'

        if bb_middle_col not in df.columns:
            df[bb_middle_col] = df['close'].rolling(window=period, min_periods=max(1, period // 2)).mean()
            bb_std = df['close'].rolling(window=period, min_periods=max(1, period // 2)).std(ddof=0)
            df[bb_upper_col] = df[bb_middle_col] + (std_dev * bb_std)
            df[bb_lower_col] = df[bb_middle_col] - (std_dev * bb_std)
        return df

    def calculate_bb_width(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        col_name = f'bb_width_{period}'
        bb_upper_col = f'bb_upper_{period}'
        bb_lower_col = f'bb_lower_{period}'
        bb_middle_col = f'bb_middle_{period}'

        if col_name not in df.columns and all(c in df.columns for c in [bb_upper_col, bb_lower_col, bb_middle_col]):
            df[col_name] = (df[bb_upper_col] - df[bb_lower_col]) / (df[bb_middle_col] + 1e-9)
        return df

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        col_name = f'atr_{period}'
        if col_name not in df.columns:
            high_low = df['high'] - df['low']
            high_close_prev = (df['high'] - df['close'].shift(1)).abs()
            low_close_prev = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
            df[col_name] = tr.ewm(alpha=1 / period, adjust=False, min_periods=max(1, period // 2)).mean()
        return df

    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'obv' not in df.columns:
            obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv'] = obv
        return df

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        col_name = f'adx_{period}'
        plus_di_col = f'plus_di_{period}'
        minus_di_col = f'minus_di_{period}'

        if col_name not in df.columns:
            high_low = df['high'] - df['low']
            high_close_prev = abs(df['high'] - df['close'].shift(1))
            low_close_prev = abs(df['low'] - df['close'].shift(1))
            tr_series = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
            atr = tr_series.ewm(alpha=1 / period, adjust=False, min_periods=max(1, period // 2)).mean()

            move_up = df['high'].diff()
            move_down = -df['low'].diff()
            plus_dm = ((move_up > move_down) & (move_up > 0)) * move_up
            minus_dm = ((move_down > move_up) & (move_down > 0)) * move_down
            plus_dm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=max(1, period // 2)).mean()
            minus_dm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=max(1, period // 2)).mean()

            df[plus_di_col] = 100 * (plus_dm_smooth / (atr + 1e-9))
            df[minus_di_col] = 100 * (minus_dm_smooth / (atr + 1e-9))

            dx = 100 * (abs(df[plus_di_col] - df[minus_di_col]) / (abs(df[plus_di_col] + df[minus_di_col]) + 1e-9))
            df[col_name] = dx.ewm(alpha=1 / period, adjust=False, min_periods=max(1, period // 2)).mean()
            df[col_name] = df[col_name].fillna(25)
        return df

    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                       signal_period: int = 9) -> pd.DataFrame:
        macd_col = f'macd_{fast_period}_{slow_period}'
        signal_col = f'macd_signal_{fast_period}_{slow_period}_{signal_period}'
        hist_col = f'macd_histogram_{fast_period}_{slow_period}_{signal_period}'

        if macd_col not in df.columns:
            ema_fast = df['close'].ewm(span=fast_period, adjust=False, min_periods=max(1, fast_period // 2)).mean()
            ema_slow = df['close'].ewm(span=slow_period, adjust=False, min_periods=max(1, slow_period // 2)).mean()
            df[macd_col] = ema_fast - ema_slow
            df[signal_col] = df[macd_col].ewm(span=signal_period, adjust=False,
                                              min_periods=max(1, signal_period // 2)).mean()
            df[hist_col] = df[macd_col] - df[signal_col]
        return df

    def calculate_stochastic_oscillator(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        stoch_k_col = f'stoch_k_{k_period}'
        stoch_d_col = f'stoch_d_{k_period}_{d_period}'
        if stoch_k_col not in df.columns:
            low_min = df['low'].rolling(window=k_period, min_periods=max(1, k_period // 2)).min()
            high_max = df['high'].rolling(window=k_period, min_periods=max(1, k_period // 2)).max()
            df[stoch_k_col] = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-9))
            df[stoch_k_col] = df[stoch_k_col].fillna(50)
        if stoch_d_col not in df.columns and stoch_k_col in df.columns:
            df[stoch_d_col] = df[stoch_k_col].rolling(window=d_period, min_periods=max(1, d_period // 2)).mean()
            df[stoch_d_col] = df[stoch_d_col].fillna(df[stoch_k_col])
        return df

    def calculate_roc(self, df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        col_name = f'roc_{period}'
        if col_name not in df.columns:
            df[col_name] = (df['close'].diff(period) / df['close'].shift(period)) * 100
        return df

    def calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        col_name = f'cmf_{period}'
        if col_name not in df.columns:
            mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9) * df['volume']
            mfv = mfv.fillna(0) # Handle potential NaNs from division by zero if high == low
            df[col_name] = mfv.rolling(window=period, min_periods=max(1, period // 2)).sum() / \
                           df['volume'].rolling(window=period, min_periods=max(1, period // 2)).sum()
            df[col_name] = df[col_name].fillna(0)
        return df

    def calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        col_name = f'hist_vol_{period}'
        if col_name not in df.columns:
            log_returns = np.log(df['close'] / df['close'].shift(1))
            df[col_name] = log_returns.rolling(window=period, min_periods=max(1, period // 2)).std() * np.sqrt(period) # Annualize by sqrt(period) if period is days, adjust if other interval
        return df


    def detect_market_phase(self, df_window: pd.DataFrame) -> str:
        if self.market_regime_util:
            return self.market_regime_util.detect_regime(df_window)["type"]

        if len(df_window) < 20: return "neutral"

        df = self.calculate_specific_indicators(df_window.copy(), ["adx_14", "ema_21", "ema_50"])
        adx = df[f'adx_14'].iloc[-1] if f'adx_14' in df.columns else 20
        close = df['close'].iloc[-1]
        ema21 = df[f'ema_21'].iloc[-1] if f'ema_21' in df.columns else close
        ema50 = df[f'ema_50'].iloc[-1] if f'ema_50' in df.columns else close

        if adx > 25:
            if close > ema21 and ema21 > ema50: return "uptrend"
            if close < ema21 and ema21 < ema50: return "downtrend"
        elif adx < 20:
            return "ranging"
        return "neutral"

    def detect_volatility_regime(self, df_window: pd.DataFrame) -> float:
        if len(df_window) < 20: return 0.5

        df = self.calculate_specific_indicators(df_window.copy(), ["atr_14", "bb_width_20"])
        atr = df[f'atr_14'].iloc[-1] if f'atr_14' in df.columns else (df['close'].iloc[-1] * 0.01)
        bb_width = df[f'bb_width_20'].iloc[-1] if f'bb_width_20' in df.columns else 0.02
        close = df['close'].iloc[-1]

        atr_pct = (atr / (close + 1e-9)) if close > 0 else 0.02

        norm_atr = np.clip((atr_pct - 0.005) / (0.03 - 0.005), 0, 1)
        norm_bbw = np.clip((bb_width - 0.01) / (0.08 - 0.01), 0, 1)

        return np.clip((norm_atr * 0.6 + norm_bbw * 0.4), 0, 1)

    def check_volume_confirmation(self, df_window: pd.DataFrame, direction: str) -> bool:
        if len(df_window) < 5 or 'volume' not in df_window.columns: return False

        recent_volume = df_window['volume'].iloc[-1]
        avg_volume_short = df_window['volume'].iloc[-5:-1].mean()

        price_change = df_window['close'].iloc[-1] - df_window['close'].iloc[-2] if len(df_window) > 1 else 0

        if direction == "long":
            return price_change > 0 and recent_volume > avg_volume_short * 0.9
        elif direction == "short":
            return price_change < 0 and recent_volume > avg_volume_short * 0.9
        return False