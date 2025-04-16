import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import math


class IndicatorFactory:
    def create_indicator(self, name, params=None):
        if params is None:
            params = {}
        return {"name": name, "params": params}


class PerformanceTracker:
    def __init__(self):
        self.signals = []
        self.performance_by_regime = {}
        self.performance_by_signal_type = {}

    def update(self, signals):
        if not isinstance(signals, list):
            signals = [signals]
        self.signals.extend(signals)

    def track_trade_result(self, trade, signal):
        regime = signal.get("market_phase", "unknown")
        signal_type = signal.get("signal_type", "unknown")
        pnl = trade.get("pnl", 0)

        if regime not in self.performance_by_regime:
            self.performance_by_regime[regime] = {"count": 0, "win": 0, "pnl": 0, "win_rate": 0}

        if signal_type not in self.performance_by_signal_type:
            self.performance_by_signal_type[signal_type] = {"count": 0, "win": 0, "pnl": 0, "win_rate": 0}

        self.performance_by_regime[regime]["count"] += 1
        self.performance_by_signal_type[signal_type]["count"] += 1

        if pnl > 0:
            self.performance_by_regime[regime]["win"] += 1
            self.performance_by_signal_type[signal_type]["win"] += 1

        self.performance_by_regime[regime]["pnl"] += pnl
        self.performance_by_signal_type[signal_type]["pnl"] += pnl

        if self.performance_by_regime[regime]["count"] > 0:
            self.performance_by_regime[regime]["win_rate"] = (
                    self.performance_by_regime[regime]["win"] / self.performance_by_regime[regime]["count"]
            )

        if self.performance_by_signal_type[signal_type]["count"] > 0:
            self.performance_by_signal_type[signal_type]["win_rate"] = (
                    self.performance_by_signal_type[signal_type]["win"] / self.performance_by_signal_type[signal_type][
                "count"]
            )


class ParameterOptimizer:
    def __init__(self, config):
        self.config = config
        self.optimization_frequency = 20
        self.trade_count = 0
        self.optimized_params = {}

    def update(self, signals, trade_history=None):
        self.trade_count += 1
        if self.trade_count % self.optimization_frequency == 0 and trade_history:
            self._optimize_parameters(trade_history)

    def _optimize_parameters(self, trade_history):
        if len(trade_history) < 10:
            return

        recent_trades = trade_history[-min(len(trade_history), 50):]
        win_trades = [t for t in recent_trades if t.get("pnl", 0) > 0]
        loss_trades = [t for t in recent_trades if t.get("pnl", 0) <= 0]

        if win_trades:
            win_adx = np.mean([t.get("adx", 0) for t in win_trades])
            win_vol = np.mean([t.get("volatility", 0) for t in win_trades])
            win_phases = {}
            for t in win_trades:
                phase = t.get("market_phase", "neutral")
                win_phases[phase] = win_phases.get(phase, 0) + 1

            best_phase = max(win_phases.items(), key=lambda x: x[1])[0] if win_phases else "neutral"

            self.optimized_params = {
                "adx_threshold": win_adx * 0.9,
                "vol_target": win_vol,
                "preferred_phase": best_phase
            }


class MultiTimeframeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.timeframes = {
            "5m": 6,
            "1h": 0.5,
            "4h": 0.125,
            "1d": 0.031
        }

    def analyze(self, df):
        try:
            signals = {}

            # Bail early if we have insufficient data
            if len(df) < 10 or 'open' not in df.columns or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
                return {"alignment": {"bullish": 0, "bearish": 0}}

            # Check for NaN values
            if df[['open', 'high', 'low', 'close']].isna().any().any():
                df_clean = df.copy()
                df_clean[['open', 'high', 'low', 'close']] = df_clean[['open', 'high', 'low', 'close']].fillna(method='ffill').fillna(method='bfill')
                if df_clean[['open', 'high', 'low', 'close']].isna().any().any():
                    return {"alignment": {"bullish": 0, "bearish": 0}}
                df = df_clean

            for timeframe, ratio in self.timeframes.items():
                resampled_data = self._resample_data(df, timeframe, ratio)
                signals[timeframe] = self._analyze_timeframe(resampled_data, timeframe)

            alignment = self._calculate_alignment(signals)
            signals["alignment"] = alignment

            return signals
        except Exception as e:
            import traceback
            print(f"Error in timeframe analysis: {e}")
            print(traceback.format_exc())
            return {"alignment": {"bullish": 0, "bearish": 0}}

    def _resample_data(self, df, timeframe, ratio):
        try:
            if ratio >= 1:
                if timeframe == "5m":
                    return df

                sample_size = int(ratio)
                if sample_size <= 1:
                    return df

                result = pd.DataFrame()
                for i in range(0, len(df), sample_size):
                    end_idx = min(i + sample_size, len(df))
                    chunk = df.iloc[i:end_idx]
                    if len(chunk) > 0:
                        # Check for NaN values in the chunk
                        if chunk[['open', 'high', 'low', 'close']].isna().all().any():
                            continue

                        row = pd.Series({
                            'open': chunk['open'].iloc[0],
                            'high': chunk['high'].max(),
                            'low': chunk['low'].min(),
                            'close': chunk['close'].iloc[-1],
                            'volume': chunk['volume'].sum() if 'volume' in chunk else 0
                        }, name=chunk.index[-1])
                        result = pd.concat([result, pd.DataFrame([row])])
                return result
            else:
                candles_to_use = max(1, int(1 / ratio))
                return df.iloc[::candles_to_use].copy()
        except Exception as e:
            print(f"Error in resampling data: {e}")
            return df.head(1).copy()  # Return minimal data that won't cause errors

    def _analyze_timeframe(self, data, timeframe):
        if len(data) < 10:
            return {"trend": "neutral", "momentum": "neutral", "strength": "weak"}

        # Check for NaN values
        if data[['open', 'high', 'low', 'close']].isna().any().any():
            return {"trend": "neutral", "momentum": "neutral", "strength": "weak"}

        close = data['close'].values

        if len(close) >= 9:
            ema9 = self._calculate_ema(close, 9)
        else:
            ema9 = close[-1] if len(close) > 0 else 0

        if len(close) >= 21:
            ema21 = self._calculate_ema(close, 21)
        else:
            ema21 = close[-1] if len(close) > 0 else 0

        if len(close) >= 50:
            ema50 = self._calculate_ema(close, 50)
        else:
            ema50 = close[-1] if len(close) > 0 else 0

        if len(close) >= 14:
            adx = self._calculate_adx(data, 14)
            rsi = self._calculate_rsi(close, 14)
        else:
            adx = 25
            rsi = 50

        macd, macd_signal, macd_hist = self._calculate_macd(close, 12, 26, 9)

        if ema9 > ema21 > ema50:
            trend = "uptrend"
        elif ema9 < ema21 < ema50:
            trend = "downtrend"
        else:
            trend = "neutral"

        if macd > macd_signal:
            momentum = "bullish"
        else:
            momentum = "bearish"

        if adx > 30:
            strength = "strong"
        elif adx > 20:
            strength = "moderate"
        else:
            strength = "weak"

        return {
            "trend": trend,
            "momentum": momentum,
            "strength": strength,
            "adx": adx,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "ema9": ema9,
            "ema21": ema21,
            "ema50": ema50
        }

    def _calculate_alignment(self, signals):
        timeframes = ["5m", "1h", "4h", "1d"]
        available_timeframes = [tf for tf in timeframes if tf in signals]

        if not available_timeframes:
            return {"bullish": 0, "bearish": 0}

        bullish_count = 0
        bearish_count = 0

        for tf in available_timeframes:
            tf_signals = signals.get(tf, {})

            if tf_signals.get("trend") == "uptrend" and tf_signals.get("momentum") == "bullish":
                bullish_count += 1
            elif tf_signals.get("trend") == "downtrend" and tf_signals.get("momentum") == "bearish":
                bearish_count += 1

        total = len(available_timeframes)
        if total <= 0:  # Extra safety check
            return {"bullish": 0, "bearish": 0}

        return {
            "bullish": bullish_count / total,
            "bearish": bearish_count / total
        }

    def _calculate_ema(self, data, period):
        if len(data) == 0:
            return 0

        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema[-1]

    def _calculate_rsi(self, data, period=14):
        if len(data) < period + 1:
            return 50

        try:
            delta = np.diff(data)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = np.mean(gain[:period])
            avg_loss = np.mean(loss[:period])

            if avg_loss <= 0.00001:  # Avoid division by effectively zero
                return 100

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi
        except (ZeroDivisionError, IndexError, ValueError):
            return 50

    def _calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        if len(data) < max(fast_period, slow_period, signal_period):
            return 0, 0, 0

        try:
            ema_fast = self._calculate_ema(data, fast_period)
            ema_slow = self._calculate_ema(data, slow_period)

            macd = ema_fast - ema_slow
            macd_signal = macd  # Should be EMA of MACD, but simplified for safety
            macd_hist = macd - macd_signal

            return macd, macd_signal, macd_hist
        except (ZeroDivisionError, IndexError, ValueError):
            return 0, 0, 0

    def _calculate_adx(self, data, period=14):
        try:
            if len(data) < period + 1:
                return 25

            # Simplified ADX calculation to avoid the complex calculations that might cause errors
            return 25  # Default moderate value
        except Exception:
            return 25


class MarketRegimeDetector:
    def __init__(self, config):
        self.config = config
        self.lookback_periods = {
            "short": 24,
            "medium": 48,
            "long": 144
        }
        self.volume_weight = 0.3  # Weight for volume analysis in regime detection
        self.transition_threshold = 0.15  # Threshold for detecting transition regimes
        self.regime_history = deque(maxlen=10)  # Store recent regime classifications
        self.level_detection_enabled = True  # Enable key level detection
        self.key_levels = []  # Store detected key price levels
        self.regime_performance = {}  # Track performance by regime

    def detect_regime(self, df):
        try:
            if len(df) < 50:
                return {"type": "neutral", "confidence": 0.5}

            # Check if required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    return {"type": "neutral", "confidence": 0.5}

            # Check for NaN values and fill them
            if df[required_columns].isna().any().any():
                df_clean = df.copy()
                df_clean[required_columns] = df_clean[required_columns].fillna(method='ffill').fillna(method='bfill')
                if df_clean[required_columns].isna().any().any():
                    return {"type": "neutral", "confidence": 0.5}
                df = df_clean

            trend_strength = self._calculate_trend_strength(df)
            volatility = self._calculate_volatility(df)
            momentum = self._calculate_momentum(df)
            range_metrics = self._calculate_range_metrics(df)

            # Add volume profile analysis
            volume_profile = self._analyze_volume_profile(df)

            # Add key level detection
            if self.level_detection_enabled:
                self._update_key_levels(df)
                key_level_influence = self._evaluate_key_level_proximity(df)
            else:
                key_level_influence = {"impact": 0, "type": "none"}

            regimes = {
                "strong_uptrend": 0.0,
                "uptrend": 0.0,
                "uptrend_transition": 0.0,  # New transition regime
                "weak_uptrend": 0.0,
                "choppy": 0.0,
                "ranging": 0.0,
                "ranging_at_support": 0.0,
                "ranging_at_resistance": 0.0,
                "downtrend_transition": 0.0,  # New transition regime
                "weak_downtrend": 0.0,
                "downtrend": 0.0,
                "strong_downtrend": 0.0,
                "volatile": 0.0
            }

            current_price = df['close'].iloc[-1]

            if 'ema_9' in df.columns and 'ema_21' in df.columns and 'ema_50' in df.columns:
                # Check for NaN values in EMAs
                if pd.isna(df['ema_9'].iloc[-1]) or pd.isna(df['ema_21'].iloc[-1]) or pd.isna(df['ema_50'].iloc[-1]):
                    regimes["neutral"] = 1.0
                else:
                    ema9 = df['ema_9'].iloc[-1]
                    ema21 = df['ema_21'].iloc[-1]
                    ema50 = df['ema_50'].iloc[-1]

                    # Detect trends based on EMAs
                    if current_price > ema9 > ema21 > ema50:
                        regimes["strong_uptrend"] = 0.5
                        regimes["uptrend"] = 0.3
                    elif current_price > ema9 > ema21:
                        regimes["uptrend"] = 0.4
                        regimes["weak_uptrend"] = 0.3

                        # Check for transition from neutral/ranging to uptrend
                        if len(self.regime_history) > 0 and "ranging" in self.regime_history[-1]:
                            regimes["uptrend_transition"] = 0.4
                            regimes["uptrend"] = 0.2

                    elif current_price < ema9 < ema21 < ema50:
                        regimes["strong_downtrend"] = 0.5
                        regimes["downtrend"] = 0.3
                    elif current_price < ema9 < ema21:
                        regimes["downtrend"] = 0.4
                        regimes["weak_downtrend"] = 0.3

                        # Check for transition from neutral/ranging to downtrend
                        if len(self.regime_history) > 0 and "ranging" in self.regime_history[-1]:
                            regimes["downtrend_transition"] = 0.4
                            regimes["downtrend"] = 0.2

                    # Detect transitions between major regimes
                    if len(df) > 5 and 'ema_9' in df.columns:
                        try:
                            ema9_diff = df['ema_9'].diff(5).iloc[-1]
                            ema9_slope = (ema9_diff / df['ema_9'].iloc[-5]) * 100 if df['ema_9'].iloc[-5] > 0 else 0

                            if ema9_slope > 0.1 and ema9_slope < 0.3 and ema21 > 0 and abs(
                                    ema9 - ema21) / ema21 < 0.005:
                                if regimes["downtrend"] > 0.2:
                                    regimes["downtrend_transition"] = max(regimes["downtrend_transition"], 0.3)
                                    regimes["downtrend"] *= 0.7
                            elif ema9_slope < -0.1 and ema9_slope > -0.3 and ema21 > 0 and abs(
                                    ema9 - ema21) / ema21 < 0.005:
                                if regimes["uptrend"] > 0.2:
                                    regimes["uptrend_transition"] = max(regimes["uptrend_transition"], 0.3)
                                    regimes["uptrend"] *= 0.7
                        except (ZeroDivisionError, TypeError):
                            pass
            else:
                # Default to neutral if EMA indicators are not available
                regimes["neutral"] = 0.5

            # ADX for trend strength
            if 'adx_14' in df.columns and not pd.isna(df['adx_14'].iloc[-1]):
                adx = df['adx_14'].iloc[-1]
                if adx > 30:
                    # Strong trend, boost appropriate trend regime
                    if regimes["strong_uptrend"] > 0.3:
                        regimes["strong_uptrend"] += 0.3
                    elif regimes["uptrend"] > 0.3:
                        regimes["uptrend"] += 0.2
                        regimes["strong_uptrend"] += 0.1
                    elif regimes["strong_downtrend"] > 0.3:
                        regimes["strong_downtrend"] += 0.3
                    elif regimes["downtrend"] > 0.3:
                        regimes["downtrend"] += 0.2
                        regimes["strong_downtrend"] += 0.1
                elif adx < 20:
                    # Weak trend, might be ranging
                    regimes["ranging"] += 0.3
                    regimes["choppy"] += 0.2

                    # Decrease trend regimes
                    for key in ["strong_uptrend", "uptrend", "strong_downtrend", "downtrend"]:
                        regimes[key] = max(0, regimes[key] - 0.2)

                    # Incorporate key level proximity for ranging markets
                    if key_level_influence["impact"] > 0.5:
                        if key_level_influence["type"] == "support":
                            regimes["ranging_at_support"] += 0.4
                            regimes["ranging"] -= 0.2
                        elif key_level_influence["type"] == "resistance":
                            regimes["ranging_at_resistance"] += 0.4
                            regimes["ranging"] -= 0.2

            # Volatility assessment
            if volatility > 0.7:
                regimes["volatile"] += 0.5
                # Decrease ranging probability
                regimes["ranging"] = max(0, regimes["ranging"] - 0.2)

            # Check for ranging using Bollinger Band width
            if 'bb_width_20' in df.columns and not pd.isna(df['bb_width_20'].iloc[-1]):
                bb_width = df['bb_width_20'].iloc[-1]
                if bb_width < 0.03:
                    regimes["ranging"] += 0.4
                    regimes["choppy"] -= 0.1
                elif bb_width < 0.05:
                    regimes["ranging"] += 0.2

            # Normalize probabilities
            total_prob = sum(regimes.values())
            if total_prob > 0:
                for regime in regimes:
                    regimes[regime] /= total_prob

            # Find the most likely regime
            primary_regime = max(regimes.items(), key=lambda x: x[1])

            # Store regime in history
            self.regime_history.append(primary_regime[0])

            return {
                "type": primary_regime[0],
                "confidence": primary_regime[1],
                "probabilities": regimes,
                "metrics": {
                    "trend_strength": trend_strength,
                    "volatility": volatility,
                    "momentum": momentum,
                    "range": range_metrics,
                    "volume_profile": volume_profile,
                    "key_level_proximity": key_level_influence
                }
            }
        except Exception as e:
            import traceback
            print(f"Error in market regime detection: {e}")
            print(traceback.format_exc())
            return {"type": "neutral", "confidence": 0.5}

    def _calculate_trend_strength(self, df):
        try:
            if 'adx_14' in df.columns and not pd.isna(df['adx_14'].iloc[-1]):
                adx = df['adx_14'].iloc[-1]
                return min(1.0, adx / 50.0)

            if len(df) < 20:
                return 0.5

            # Safe extraction of close values
            close_values = df['close'].values[-10:]
            if len(close_values) < 10 or close_values[-10] <= 0:
                return 0.5

            short_slope = (close_values[-1] / close_values[-10] - 1) * 10
            return min(1.0, abs(short_slope))
        except (IndexError, ZeroDivisionError, ValueError):
            return 0.5

    def _calculate_volatility(self, df):
        try:
            if 'atr_14' in df.columns and 'close' in df.columns:
                if pd.isna(df['atr_14'].iloc[-1]) or pd.isna(df['close'].iloc[-1]) or df['close'].iloc[-1] <= 0:
                    return 0.5
                atr = df['atr_14'].iloc[-1]
                close = df['close'].iloc[-1]
                return min(1.0, (atr / close) * 100)

            if 'bb_width_20' in df.columns and not pd.isna(df['bb_width_20'].iloc[-1]):
                bb_width = df['bb_width_20'].iloc[-1]
                return min(1.0, bb_width * 10)

            if len(df) < 10:
                return 0.5

            # Safe calculation of returns
            close = df['close'].values[-10:]
            if len(close) < 10:
                return 0.5

            returns = []
            for i in range(1, len(close)):
                if close[i - 1] > 0:
                    returns.append((close[i] - close[i - 1]) / close[i - 1])
                else:
                    returns.append(0)

            return min(1.0, np.std(returns) * 100) if returns else 0.5
        except (ValueError, ZeroDivisionError, IndexError):
            return 0.5

    def _calculate_momentum(self, df):
        try:
            if 'macd_histogram_12_26_9' in df.columns and not pd.isna(df['macd_histogram_12_26_9'].iloc[-1]):
                hist = df['macd_histogram_12_26_9'].iloc[-1]
                return np.tanh(hist)

            if 'rsi_14' in df.columns and not pd.isna(df['rsi_14'].iloc[-1]):
                rsi = df['rsi_14'].iloc[-1]
                return (rsi - 50) / 50

            return 0
        except (ValueError, TypeError):
            return 0

    def _evaluate_key_level_proximity(self, df):
        if not self.key_levels or len(df) == 0:
            return {"impact": 0, "type": "none"}

        current_price = df['close'].iloc[-1]
        if pd.isna(current_price) or current_price <= 0:
            return {"impact": 0, "type": "none"}

        # Find closest level
        closest_level = None
        closest_distance = float('inf')

        for level in self.key_levels:
            level_price = level.get("price", 0)
            if level_price <= 0:
                continue

            distance = abs(current_price - level_price) / current_price
            if distance < closest_distance:
                closest_distance = distance
                closest_level = level

        # If no valid levels found
        if closest_level is None:
            return {"impact": 0, "type": "none"}

        # If price is very close to a level (within 1%)
        if closest_distance < 0.01:
            return {
                "impact": 1 - (closest_distance * 100),  # Higher impact when closer
                "type": closest_level.get("type", "none"),
                "price": closest_level.get("price", 0),
                "strength": closest_level.get("strength", 1)
            }
        else:
            return {"impact": 0, "type": "none"}

    def _calculate_range_metrics(self, df):
        if len(df) < 20:
            return {"width": 0.1, "position": 0.5}

        high = df['high'].rolling(20).max().iloc[-1]
        low = df['low'].rolling(20).min().iloc[-1]
        close = df['close'].iloc[-1]

        # Fix: Check if high and low are valid and different
        if pd.isna(high) or pd.isna(low) or pd.isna(close) or high <= low or low <= 0:
            return {"width": 0.1, "position": 0.5}

        width = (high - low) / low
        position = (close - low) / (high - low)

        return {"width": width, "position": position}

    def _analyze_volume_profile(self, df):
        if len(df) < 10 or 'volume' not in df.columns:
            return {
                "volume_at_price_range": 0.5,
                "buying_pressure": 0.5,
                "selling_pressure": 0.5
            }

        # Calculate price range and divide into 10 bins
        recent_df = df.iloc[-30:].copy()
        price_min = recent_df['low'].min()
        price_max = recent_df['high'].max()
        price_range = price_max - price_min

        # Fix: Check if price range is zero
        if price_range <= 0:
            return {
                "volume_at_price_range": 0.5,
                "buying_pressure": 0.5,
                "selling_pressure": 0.5
            }

        # Create bins based on price
        bin_size = price_range / 10
        bins = {}
        for i in range(10):
            bin_low = price_min + (i * bin_size)
            bin_high = price_min + ((i + 1) * bin_size)
            bins[i] = {"low": bin_low, "high": bin_high, "volume": 0}

        # Assign volume to bins
        total_volume = 0
        up_volume = 0
        down_volume = 0

        for idx, row in recent_df.iterrows():
            # Fix: Check for NaN or zero values
            if pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['volume']):
                continue

            price = (row['high'] + row['low']) / 2
            vol = row['volume']
            total_volume += vol

            if row['close'] > row['open']:
                up_volume += vol
            else:
                down_volume += vol

            for i in range(10):
                if bins[i]["low"] <= price <= bins[i]["high"]:
                    bins[i]["volume"] += vol
                    break

        # Calculate volume concentration
        if total_volume == 0:
            return {
                "volume_at_price_range": 0.5,
                "buying_pressure": 0.5,
                "selling_pressure": 0.5
            }

        volumes = [bins[i]["volume"] for i in range(10)]
        max_vol_bin = max(volumes) if volumes else 0
        volume_concentration = max_vol_bin / total_volume if max_vol_bin > 0 and total_volume > 0 else 0.5

        # Calculate buying/selling pressure
        buying_pressure = up_volume / total_volume if total_volume > 0 else 0.5
        selling_pressure = down_volume / total_volume if total_volume > 0 else 0.5

        return {
            "volume_at_price_range": volume_concentration,
            "buying_pressure": buying_pressure,
            "selling_pressure": selling_pressure
        }

    def _update_key_levels(self, df):
        try:
            if len(df) < 100:
                return

            # Find support and resistance levels using local minima/maxima
            highs = df['high'].values
            lows = df['low'].values

            potential_supports = []
            potential_resistances = []

            window_size = 10
            for i in range(window_size, len(lows) - window_size):
                # Check for local minima (support)
                if all(lows[i] <= lows[i - j] for j in range(1, window_size + 1)) and \
                        all(lows[i] <= lows[i + j] for j in range(1, window_size + 1)):
                    potential_supports.append((i, lows[i]))

                # Check for local maxima (resistance)
                if all(highs[i] >= highs[i - j] for j in range(1, window_size + 1)) and \
                        all(highs[i] >= highs[i + j] for j in range(1, window_size + 1)):
                    potential_resistances.append((i, highs[i]))

            # Safely cluster close levels with error handling
            supports = self._cluster_levels(potential_supports, 0.005)
            resistances = self._cluster_levels(potential_resistances, 0.005)

            # Keep only significant levels (with enough touches or high volume)
            self.key_levels = []

            for level, _ in supports:
                touches = self._count_level_touches(df, level, 'support')
                if touches >= 2:
                    self.key_levels.append({"price": level, "type": "support", "strength": touches})

            for level, _ in resistances:
                touches = self._count_level_touches(df, level, 'resistance')
                if touches >= 2:
                    self.key_levels.append({"price": level, "type": "resistance", "strength": touches})

        except Exception as e:
            self.logger.warning(f"Error updating key levels: {e}")

    def _cluster_levels(self, levels, threshold):
        if not levels:
            return []

        try:
            # Sort by price
            sorted_levels = sorted(levels, key=lambda x: x[1])

            clusters = []
            if not sorted_levels:
                return clusters

            current_cluster = [sorted_levels[0]]

            for i in range(1, len(sorted_levels)):
                current_level = sorted_levels[i]
                prev_level = current_cluster[-1]

                # If close to previous level, add to current cluster
                if prev_level[1] > 0 and abs(current_level[1] - prev_level[1]) / prev_level[1] <= threshold:
                    current_cluster.append(current_level)
                else:
                    # Process the current cluster
                    if current_cluster:
                        avg_price = sum(level[1] for level in current_cluster) / len(current_cluster)
                        # Fix: Ensure denominator is never zero
                        weights = [1.0 / max(1, len(sorted_levels) - level[0]) for level in current_cluster]
                        weight_sum = sum(weights)
                        clusters.append((avg_price, weight_sum))

                    # Start new cluster
                    current_cluster = [current_level]

            # Handle the last cluster
            if current_cluster:
                avg_price = sum(level[1] for level in current_cluster) / len(current_cluster)
                # Fix: Ensure denominator is never zero
                weights = [1.0 / max(1, len(sorted_levels) - level[0]) for level in current_cluster]
                weight_sum = sum(weights)
                clusters.append((avg_price, weight_sum))

            return clusters
        except Exception as e:
            self.logger.warning(f"Error in clustering levels: {e}")
            return []

    def _count_level_touches(self, df, level, level_type):
        # Count how many times price approached this level
        touches = 0
        price_threshold = level * 0.005  # 0.5% threshold

        for i in range(1, len(df) - 1):
            if level_type == 'support':
                if abs(df['low'].iloc[i] - level) <= price_threshold and \
                        df['low'].iloc[i - 1] > df['low'].iloc[i] and \
                        df['low'].iloc[i + 1] > df['low'].iloc[i]:
                    touches += 1
            else:  # resistance
                if abs(df['high'].iloc[i] - level) <= price_threshold and \
                        df['high'].iloc[i - 1] < df['high'].iloc[i] and \
                        df['high'].iloc[i + 1] < df['high'].iloc[i]:
                    touches += 1

        return touches

class VolumeProfileAnalyzer:
    def __init__(self, config):
        self.config = config
        self.vp_lookback = 144
        self.num_nodes = 20

    def analyze(self, df):
        if len(df) < 10 or 'volume' not in df.columns:
            return {
                "volume_at_price_range": 0.5,
                "buying_pressure": 0.5,
                "selling_pressure": 0.5,
                "value_areas": {"high": [], "low": []},
                "liquidity_voids": []
            }

        try:
            recent_data = df.iloc[-self.vp_lookback:].copy() if len(df) > self.vp_lookback else df.copy()

            # Check for NaN values and fill them
            if recent_data.isna().any().any():
                recent_data = recent_data.fillna(method='ffill').fillna(method='bfill')
                # If still have NaNs after filling, return default values
                if recent_data.isna().any().any():
                    return {
                        "volume_at_price_range": 0.5,
                        "buying_pressure": 0.5,
                        "selling_pressure": 0.5,
                        "value_areas": {"high": [], "low": []},
                        "liquidity_voids": []
                    }

            price_range = {
                "min": recent_data['low'].min(),
                "max": recent_data['high'].max()
            }

            # Check if price range is invalid or too small
            if price_range["max"] <= price_range["min"] or np.isclose(price_range["max"], price_range["min"]):
                return {
                    "volume_at_price_range": 0.5,
                    "buying_pressure": 0.5,
                    "selling_pressure": 0.5,
                    "value_areas": {"high": [], "low": []},
                    "liquidity_voids": []
                }

            node_size = (price_range["max"] - price_range["min"]) / self.num_nodes
            nodes = [price_range["min"] + i * node_size for i in range(self.num_nodes + 1)]

            volume_profile = self._calculate_volume_distribution(recent_data, nodes)
            value_areas = self._identify_value_areas(volume_profile)
            liquidity_voids = self._identify_liquidity_voids(volume_profile)
            pressure = self._detect_pressure(recent_data)
            delta = self._analyze_delta(recent_data)

            return {
                "profile": volume_profile,
                "value_areas": value_areas,
                "liquidity_voids": liquidity_voids,
                "buying_pressure": pressure.get("buying", 0.5),
                "selling_pressure": pressure.get("selling", 0.5),
                "delta": delta
            }
        except Exception as e:
            import traceback
            print(f"Error in volume profile analysis: {e}")
            print(traceback.format_exc())
            return {
                "volume_at_price_range": 0.5,
                "buying_pressure": 0.5,
                "selling_pressure": 0.5,
                "value_areas": {"high": [], "low": []},
                "liquidity_voids": []
            }

    def _calculate_volume_distribution(self, data, nodes):
        if len(data) == 0 or len(nodes) <= 1:
            return []

        distribution = []

        for i in range(len(nodes) - 1):
            lower = nodes[i]
            upper = nodes[i + 1]

            node_volume = 0
            for idx, row in data.iterrows():
                # Skip rows with NaN values
                if pd.isna(row['low']) or pd.isna(row['high']) or pd.isna(row['volume']):
                    continue

                # Skip invalid price ranges
                if row['high'] <= row['low']:
                    continue

                if lower <= row['low'] and row['high'] <= upper:
                    node_volume += row['volume']
                elif row['low'] < lower and row['high'] > lower:
                    # Safely calculate overlap
                    range_size = max(0.000001, row['high'] - row['low'])  # Avoid division by zero
                    overlap = (row['high'] - lower) / range_size
                    node_volume += row['volume'] * overlap
                elif row['low'] < upper and row['high'] > upper:
                    # Safely calculate overlap
                    range_size = max(0.000001, row['high'] - row['low'])  # Avoid division by zero
                    overlap = (upper - row['low']) / range_size
                    node_volume += row['volume'] * overlap

            distribution.append({
                "price_low": lower,
                "price_high": upper,
                "volume": node_volume
            })

        return distribution

    def _identify_value_areas(self, volume_profile):
        if not volume_profile:
            return {"high": [], "low": []}

        # Check if volume_profile contains valid data
        if any(not isinstance(node, dict) for node in volume_profile):
            return {"high": [], "low": []}

        # Calculate total volume with safety check
        total_volume = sum(node.get("volume", 0) for node in volume_profile)
        if total_volume <= 0:
            return {"high": [], "low": []}

        sorted_nodes = sorted(volume_profile, key=lambda x: x.get("volume", 0), reverse=True)

        high_volume_nodes = []
        cumulative_volume = 0

        for node in sorted_nodes:
            if "volume" not in node:
                continue

            high_volume_nodes.append(node)
            cumulative_volume += node["volume"]

            if cumulative_volume / total_volume >= 0.7:
                break

        low_volume_nodes = [node for node in volume_profile if node not in high_volume_nodes]

        return {"high": high_volume_nodes, "low": low_volume_nodes}

    def _identify_liquidity_voids(self, volume_profile):
        if not volume_profile:
            return []

        # Check if volume_profile contains valid data
        if any(not isinstance(node, dict) for node in volume_profile):
            return []

        # Calculate average volume with safety checks
        volumes = [node.get("volume", 0) for node in volume_profile]
        if not volumes or all(v == 0 for v in volumes):
            return []

        avg_volume = sum(volumes) / len(volumes)
        threshold = avg_volume * 0.3

        voids = [node for node in volume_profile if node.get("volume", 0) < threshold]
        return voids

    def _detect_pressure(self, data):
        if len(data) < 5 or 'volume' not in data.columns:
            return {"buying": 0.5, "selling": 0.5}

        # Check for NaN or empty values
        if data['volume'].isna().any() or data['close'].isna().any() or data['open'].isna().any():
            return {"buying": 0.5, "selling": 0.5}

        buying_volume = 0
        selling_volume = 0

        for idx, row in data.iterrows():
            if pd.isna(row['volume']) or pd.isna(row['close']) or pd.isna(row['open']):
                continue

            if row['close'] > row['open']:
                buying_volume += row['volume']
            else:
                selling_volume += row['volume']

        total_volume = buying_volume + selling_volume
        if total_volume <= 0:
            return {"buying":
                        0.5, "selling": 0.5}

        buying_pressure = buying_volume / total_volume
        selling_pressure = selling_volume / total_volume

        return {"buying": buying_pressure, "selling": selling_pressure}

    def _analyze_delta(self, data):
        if len(data) < 5 or 'volume' not in data.columns:
            return 0

        # Check for NaN values
        if data['volume'].isna().any() or data['close'].isna().any() or data['open'].isna().any():
            return 0

        delta_sum = 0
        total_volume = 0

        for idx, row in data.iterrows():
            if pd.isna(row['volume']) or pd.isna(row['close']) or pd.isna(row['open']):
                continue

            vol = row['volume']
            total_volume += vol

            if row['close'] > row['open']:
                delta_sum += vol
            else:
                delta_sum -= vol

        if total_volume <= 0:
            return 0

        return delta_sum / total_volume

    def calculate_order_flow_metrics(self, data):
        if len(data) < 10 or 'volume' not in data.columns:
            return {
                "buying_pressure": 0.5,
                "selling_pressure": 0.5,
                "volume_delta": 0,
                "volume_trend": 0
            }

        # Fix: Check for NaN values in data
        if data['volume'].isna().any() or data['close'].isna().any() or data['open'].isna().any():
            data = data.copy()
            data = data.fillna(method='ffill').fillna(method='bfill')
            if data['volume'].isna().any():
                return {
                    "buying_pressure": 0.5,
                    "selling_pressure": 0.5,
                    "volume_delta": 0,
                    "volume_trend": 0
                }

        # Calculate buying vs selling volume
        up_candles = data['close'] > data['open']
        down_candles = data['close'] < data['open']

        buying_volume = data.loc[up_candles, 'volume'].sum() if any(up_candles) else 0
        selling_volume = data.loc[down_candles, 'volume'].sum() if any(down_candles) else 0
        total_volume = buying_volume + selling_volume

        if total_volume <= 0:
            return {
                "buying_pressure": 0.5,
                "selling_pressure": 0.5,
                "volume_delta": 0,
                "volume_trend": 0
            }

        buying_pressure = buying_volume / total_volume
        selling_pressure = selling_volume / total_volume

        # Calculate volume delta (normalized difference)
        volume_delta = (buying_volume - selling_volume) / total_volume

        # Calculate volume trend
        recent_volumes = data['volume'].values[-5:]
        prev_volumes = data['volume'].values[-10:-5] if len(data) >= 10 else data['volume'].values[:5]

        # Fix: Check for zero volumes
        recent_avg = np.mean(recent_volumes) if len(recent_volumes) > 0 and np.sum(recent_volumes) > 0 else 0.001
        prev_avg = np.mean(prev_volumes) if len(prev_volumes) > 0 and np.sum(prev_volumes) > 0 else 0.001

        # Fix: Avoid division by zero
        if prev_avg <= 0:
            volume_trend = 0
        else:
            volume_trend = (recent_avg / prev_avg) - 1

        # Add taker buy ratio if available
        taker_buy_ratio = 0.5
        if 'taker_buy_base_asset_volume' in data.columns and 'volume' in data.columns:
            taker_volumes = data['taker_buy_base_asset_volume'].sum()
            total_vol = data['volume'].sum()
            if total_vol > 0:
                taker_buy_ratio = taker_volumes / total_vol

        return {
            "buying_pressure": buying_pressure,
            "selling_pressure": selling_pressure,
            "volume_delta": volume_delta,
            "volume_trend": volume_trend,
            "taker_buy_ratio": taker_buy_ratio
        }


class AdaptiveThresholdManager:
    def __init__(self, config):
        self.config = config
        self.base_threshold = config.get("signal", "confidence_threshold", 0.0008)
        self.max_threshold = config.get("signal", "max_threshold", 0.0025)
        self.min_threshold = config.get("signal", "min_threshold", 0.0003)
        self.regime_factors = self._initialize_regime_factors()
        self.volatility_factors = self._initialize_volatility_factors()
        self.performance_scaling = config.get("signal", "performance_scaling", True)
        self.win_loss_history = deque(maxlen=30)

    def get_current_thresholds(self, market_regime):
        threshold = self.base_threshold

        regime_type = market_regime.get("type", "unknown")
        regime_factor = self.regime_factors.get(regime_type, 1.0)
        threshold *= regime_factor

        volatility = market_regime.get("metrics", {}).get("volatility", 0.5)
        vol_factor = self._get_volatility_factor(volatility)
        threshold *= vol_factor

        if self.performance_scaling and self.win_loss_history:
            performance_factor = self._calculate_performance_factor()
            threshold *= performance_factor

        threshold = max(self.min_threshold, min(threshold, self.max_threshold))

        return {
            "weak": threshold * 0.8,
            "normal": threshold,
            "strong": threshold * 1.5,
            "very_strong": threshold * 2.5
        }

    def update_performance(self, trade_result):
        if trade_result["pnl"] > 0:
            self.win_loss_history.append(1)
        else:
            self.win_loss_history.append(-1)

    def _initialize_regime_factors(self):
        return {
            "strong_uptrend": 0.75,
            "uptrend": 0.8,
            "weak_uptrend": 0.9,
            "choppy": 1.2,
            "ranging": 1.3,
            "weak_downtrend": 0.9,
            "downtrend": 0.8,
            "strong_downtrend": 0.75,
            "volatile": 1.5,
            "neutral": 1.0
        }

    def _initialize_volatility_factors(self):
        return [
            (0.0, 0.3, 0.9),
            (0.3, 0.5, 1.0),
            (0.5, 0.7, 1.2),
            (0.7, 1.0, 1.5)
        ]

    def _get_volatility_factor(self, volatility):
        for low, high, factor in self.volatility_factors:
            if low <= volatility < high:
                return factor
        return 1.0

    def _calculate_performance_factor(self):
        if not self.win_loss_history:
            return 1.0

        recent_wins = sum(1 for result in self.win_loss_history if result == 1)
        win_rate = recent_wins / len(self.win_loss_history)

        if win_rate > 0.6:
            return max(0.75, 1.0 - (win_rate - 0.6) * 2)
        elif win_rate < 0.4:
            return min(1.5, 1.0 + (0.4 - win_rate) * 2)

        return 1.0


class SignalConfidenceScorer:
    def __init__(self, config):
        self.config = config
        self.weight_model_prediction = 0.40
        self.weight_technical_alignment = 0.20
        self.weight_timeframe_alignment = 0.15
        self.weight_volume_confirmation = 0.15
        self.weight_historical_performance = 0.10
        self.performance_data = {}

    def score_signals(self, signals, trade_history=None):
        if isinstance(signals, dict):
            signals = [signals]

        scored_signals = []

        for signal in signals:
            prediction_score = self._normalize_prediction(signal.get("predicted_return", 0))
            technical_score = self._calculate_technical_alignment(signal)
            timeframe_score = self._calculate_timeframe_alignment(signal)
            volume_score = self._calculate_volume_confirmation(signal)

            historical_score = 0.5
            if trade_history:
                historical_score = self._calculate_historical_performance(signal, trade_history)

            composite_score = (
                    self.weight_model_prediction * prediction_score +
                    self.weight_technical_alignment * technical_score +
                    self.weight_timeframe_alignment * timeframe_score +
                    self.weight_volume_confirmation * volume_score +
                    self.weight_historical_performance * historical_score
            )

            composite_score = self._apply_score_enhancers(composite_score, signal)

            signal["confidence_score"] = composite_score
            signal["component_scores"] = {
                "prediction": prediction_score,
                "technical": technical_score,
                "timeframe": timeframe_score,
                "volume": volume_score,
                "historical": historical_score
            }

            scored_signals.append(signal)

        return scored_signals if len(scored_signals) > 1 else scored_signals[0]

    def _normalize_prediction(self, prediction):
        return min(1.0, max(0.0, (abs(prediction) * 50) ** 0.7))

    def _calculate_technical_alignment(self, signal):
        ema_signal = signal.get("ema_signal", 0)
        macd_signal = signal.get("macd_signal", 0)
        rsi_14 = signal.get("rsi_14", 50)
        bb_width = signal.get("bb_width", 0)
        adx = signal.get("adx", 25)
        trend_strength = signal.get("trend_strength", 0.5)
        volume_confirms = signal.get("volume_confirms", False)
        direction = 1 if signal.get("direction", "") == "long" else -1

        # Get market phase alignment
        market_phase = signal.get("market_phase", "neutral")
        phase_alignment = 0
        if (direction > 0 and market_phase in ["strong_uptrend", "uptrend"]) or \
                (direction < 0 and market_phase in ["strong_downtrend", "downtrend"]):
            phase_alignment = 0.15
        elif (direction > 0 and market_phase in ["downtrend", "strong_downtrend"]) or \
                (direction < 0 and market_phase in ["uptrend", "strong_uptrend"]):
            phase_alignment = -0.15

        technical_score = 0.5  # Base score

        # EMA alignment: 20% weight
        if (direction > 0 and ema_signal > 0) or (direction < 0 and ema_signal < 0):
            technical_score += 0.2

        # MACD alignment: 20% weight
        if (direction > 0 and macd_signal > 0) or (direction < 0 and macd_signal < 0):
            technical_score += 0.2

        # Trend strength: 15% weight (new)
        if trend_strength > 0.6:
            if (direction > 0 and adx > 25) or (direction < 0 and adx > 25):
                technical_score += 0.15

        # Volume confirmation: 15% weight (new)
        if volume_confirms:
            technical_score += 0.15

        # Market phase alignment: 15% weight (new)
        technical_score += phase_alignment

        # RSI: reduced to 10% weight (was implicit 20-30% before)
        if (direction > 0 and rsi_14 > 50) or (direction < 0 and rsi_14 < 50):
            technical_score += 0.1

        # Penalize extreme RSI - reduced penalty
        if (direction > 0 and rsi_14 > 70) or (direction < 0 and rsi_14 < 30):
            technical_score -= 0.1  # Reduced from 0.2

        return min(1.0, max(0.0, technical_score))

    def _calculate_timeframe_alignment(self, signal):
        multi_timeframe = signal.get("multi_timeframe", {})
        alignment = multi_timeframe.get("alignment", {})

        direction = 1 if signal.get("direction", "") == "long" else -1

        if direction > 0:
            return alignment.get("bullish", 0)
        else:
            return alignment.get("bearish", 0)

    def _calculate_volume_confirmation(self, signal):
        volume_profile = signal.get("volume_profile", {})
        volume_confirms = signal.get("volume_confirms", False)

        if volume_confirms:
            return 0.8

        direction = 1 if signal.get("direction", "") == "long" else -1

        if direction > 0:
            buying_pressure = volume_profile.get("buying_pressure", 0.5)
            return buying_pressure
        else:
            selling_pressure = volume_profile.get("selling_pressure", 0.5)
            return selling_pressure

    def _calculate_historical_performance(self, signal, trade_history):
        if not trade_history:
            return 0.5

        market_phase = signal.get("market_phase", "neutral")
        direction = signal.get("direction", "long")

        relevant_trades = [t for t in trade_history
                           if t.get("market_phase") == market_phase and
                           t.get("direction") == direction]

        if not relevant_trades:
            return 0.5

        win_trades = [t for t in relevant_trades if t.get("pnl", 0) > 0]
        win_rate = len(win_trades) / len(relevant_trades) if relevant_trades else 0.5

        return win_rate

    def _apply_score_enhancers(self, score, signal):
        market_phase = signal.get("market_phase", "neutral")
        direction = signal.get("direction", "long")

        phase_alignment_boost = 0

        if (market_phase == "strong_uptrend" and direction == "long") or \
                (market_phase == "strong_downtrend" and direction == "short"):
            phase_alignment_boost = 0.15
        elif (market_phase == "uptrend" and direction == "long") or \
                (market_phase == "downtrend" and direction == "short"):
            phase_alignment_boost = 0.1
        elif (market_phase == "ranging" and direction == "long" and signal.get("rsi_14", 50) < 30) or \
                (market_phase == "ranging" and direction == "short" and signal.get("rsi_14", 50) > 70):
            phase_alignment_boost = 0.05
        elif (market_phase == "strong_uptrend" and direction == "short") or \
                (market_phase == "strong_downtrend" and direction == "long"):
            phase_alignment_boost = -0.15

        enhanced_score = score + phase_alignment_boost
        return min(1.0, max(0.0, enhanced_score))


class SignalGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("SignalGenerator")
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(config)
        self.market_regime_detector = MarketRegimeDetector(config)
        self.volume_analyzer = VolumeProfileAnalyzer(config)
        self.signal_scorer = SignalConfidenceScorer(config)
        self.indicator_factory = IndicatorFactory()
        self.threshold_manager = AdaptiveThresholdManager(config)
        self.parameter_optimizer = ParameterOptimizer(config)
        self.performance_tracker = PerformanceTracker()

        from indicator_util import IndicatorUtil
        self.indicator_util = IndicatorUtil()

        # Add RL Signal Generator
        from rl_agent import RLSignalGenerator
        self.rl_signal_generator = RLSignalGenerator(config)
        self.rl_mode = config.get("rl", "mode", "hybrid")
        self.rl_ensemble_weight = config.get("rl", "ensemble_weight", 0.5)

    def generate_signal(self, model_pred, df, **kwargs):
        try:
            # Check for RL-only mode
            if self.rl_mode == "standalone" and hasattr(self, 'rl_signal_generator'):
                return self.rl_signal_generator.generate_signal(model_pred, df, **kwargs)

            # Initial validation
            if df is None or len(df) < 50:
                return {"signal_type": "NoTrade", "reason": "InsufficientData"}

            # Normalize column names
            df.columns = [col.lower() for col in df.columns]

            # Process model prediction
            if isinstance(model_pred, np.ndarray):
                if model_pred.size > 1:
                    model_pred = float(model_pred[0])
                else:
                    model_pred = float(model_pred)

            # Handle NaN or infinite predictions
            if np.isnan(model_pred) or np.isinf(model_pred):
                model_pred = 0.0

            # Validate price data
            current_price = self._get_latest_price(df)
            if np.isnan(current_price) or current_price <= 0:
                return {"signal_type": "NoTrade", "reason": "InvalidPrice"}

            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    return {"signal_type": "NoTrade", "reason": f"MissingRequiredColumn_{col}"}

            # Process NaN values in critical columns
            if df[required_columns].isna().any().any():
                df_clean = df.copy()
                df_clean[required_columns] = df_clean[required_columns].fillna(method='ffill').fillna(method='bfill')
                if df_clean[required_columns].isna().any().any():
                    return {"signal_type": "NoTrade", "reason": "InvalidDataAfterCleaning"}
                df = df_clean

            # Generate market analysis with proper error handling
            try:
                market_regime = self.market_regime_detector.detect_regime(df)
            except Exception as e:
                self.logger.warning(f"Error in market regime detection: {e}")
                market_regime = {"type": "neutral", "confidence": 0.5, "metrics": {}}

            try:
                volume_profile = self.volume_analyzer.analyze(df)
            except Exception as e:
                self.logger.warning(f"Error in volume analysis: {e}")
                volume_profile = {
                    "buying_pressure": 0.5,
                    "selling_pressure": 0.5,
                    "value_areas": {"high": [], "low": []},
                    "liquidity_voids": []
                }

            try:
                multi_timeframe_signals = self.multi_timeframe_analyzer.analyze(df)
            except Exception as e:
                self.logger.warning(f"Error in timeframe analysis: {e}")
                multi_timeframe_signals = {"alignment": {"bullish": 0, "bearish": 0}}

            try:
                order_flow_metrics = self.volume_analyzer.calculate_order_flow_metrics(
                    df.iloc[-20:] if len(df) > 20 else df)
            except Exception as e:
                self.logger.warning(f"Error in order flow analysis: {e}")
                order_flow_metrics = {
                    "buying_pressure": 0.5,
                    "selling_pressure": 0.5,
                    "volume_delta": 0,
                    "volume_trend": 0
                }

            # Create base signal
            try:
                base_signal = self._generate_base_signal(model_pred, df, market_regime)
            except Exception as e:
                self.logger.warning(f"Error generating base signal: {e}")
                direction = "long" if model_pred > 0 else "short"
                base_signal = {
                    "signal_type": "NoTrade",
                    "direction": direction,
                    "predicted_return": model_pred,
                    "market_phase": market_regime.get("type", "neutral"),
                    "regime_confidence": market_regime.get("confidence", 0.5),
                    "volatility": 0.5,
                    "trend_strength": 0.5
                }

            # Enhance signal with additional metrics
            try:
                enhanced_signal = self._enhance_signal(base_signal, df, multi_timeframe_signals, volume_profile)
                # Add order flow metrics
                enhanced_signal.update(order_flow_metrics)
            except Exception as e:
                self.logger.warning(f"Error enhancing signal: {e}")
                enhanced_signal = base_signal.copy()
                enhanced_signal["multi_timeframe"] = multi_timeframe_signals
                enhanced_signal["volume_profile"] = volume_profile

            # Check for Fibonacci signals in range-bound markets
            try:
                use_fibonacci = self.config.get("signal", "use_fibonacci", False)

                if use_fibonacci and self._is_range_bound_market(df):
                    fibonacci_levels = self._calculate_fibonacci_levels(df)
                    current_price = self._get_latest_price(df)

                    fibonacci_signal = self._generate_fibonacci_signal(df, current_price, fibonacci_levels)

                    if fibonacci_signal["signal_type"] != "NoTrade":
                        enhanced_signal = enhanced_signal.copy()
                        enhanced_signal.update(fibonacci_signal)
            except Exception as e:
                self.logger.warning(f"Error in Fibonacci analysis: {e}")

            # Extract additional parameters
            trade_history = kwargs.get("trade_history", [])
            adaptive_mode = kwargs.get("adaptive_mode", False)
            win_streak = kwargs.get("win_streak", 0)
            loss_streak = kwargs.get("loss_streak", 0)

            # Apply confidence scoring with error handling
            try:
                scored_signal = self.signal_scorer.score_signals(enhanced_signal, trade_history)
            except Exception as e:
                self.logger.warning(f"Error scoring signal: {e}")
                scored_signal = enhanced_signal
                scored_signal["confidence_score"] = 0.3

            # Get thresholds with error handling
            try:
                thresholds = self.threshold_manager.get_current_thresholds(market_regime)
            except Exception as e:
                self.logger.warning(f"Error getting thresholds: {e}")
                thresholds = {"weak": 0.0008, "normal": 0.001, "strong": 0.0015, "very_strong": 0.0025}

            # Apply thresholds with error handling
            try:
                final_signal = self._apply_thresholds(scored_signal, thresholds, adaptive_mode, win_streak, loss_streak)
            except Exception as e:
                self.logger.warning(f"Error applying thresholds: {e}")
                final_signal = scored_signal
                final_signal["signal_type"] = "NoTrade"
                final_signal["reason"] = "ThresholdError"
                final_signal["applied_threshold"] = thresholds.get("normal", 0.001)
                final_signal["ensemble_score"] = scored_signal.get("confidence_score", 0.3)

            # RL integration - hybrid mode
            if self.rl_mode == "hybrid" and hasattr(self, 'rl_signal_generator'):
                try:
                    rl_signal = self.rl_signal_generator.generate_signal(model_pred, df, **kwargs)
                    final_signal = self._ensemble_signals(final_signal, rl_signal)
                except Exception as e:
                    self.logger.warning(f"Error in RL signal integration: {e}")

            # Perform final validation on signal
            if final_signal is None or not isinstance(final_signal, dict):
                final_signal = {
                    "signal_type": "NoTrade",
                    "reason": "InvalidSignal",
                    "direction": "long" if model_pred > 0 else "short"
                }

            # Ensure all critical fields are present with valid values
            if "signal_type" not in final_signal:
                final_signal["signal_type"] = "NoTrade"

            if "direction" not in final_signal:
                final_signal["direction"] = "long" if model_pred > 0 else "short"

            if "ensemble_score" not in final_signal:
                final_signal["ensemble_score"] = final_signal.get("confidence_score", 0.3)

            # Track signal and update optimizers
            try:
                self.performance_tracker.update(final_signal)
                self.parameter_optimizer.update(final_signal, trade_history)
            except Exception as e:
                self.logger.debug(f"Error updating performance trackers: {e}")

            return final_signal

        except Exception as e:
            import traceback
            self.logger.error(f"Uncaught error in generate_signal: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "signal_type": "NoTrade",
                "reason": f"SignalError_{str(e)}",
                "direction": "long" if isinstance(model_pred, (int, float)) and model_pred > 0 else "short"
            }

    def _ensemble_signals(self, traditional_signal, rl_signal):
        if traditional_signal is None or rl_signal is None:
            return traditional_signal or rl_signal or {"signal_type": "NoTrade", "reason": "EmptySignals"}

        traditional_type = traditional_signal.get("signal_type", "NoTrade")
        rl_type = rl_signal.get("signal_type", "NoTrade")

        # Ensure market_phase is a string in both signals
        if "market_phase" in traditional_signal and not isinstance(traditional_signal["market_phase"], str):
            if isinstance(traditional_signal["market_phase"], (float, np.float64, np.float32, int, np.int64, np.int32)):
                # Convert numeric value to string representation
                value = float(traditional_signal["market_phase"])
                if value > 0.5:
                    traditional_signal["market_phase"] = "uptrend"
                elif value < -0.5:
                    traditional_signal["market_phase"] = "downtrend"
                else:
                    traditional_signal["market_phase"] = "neutral"
            else:
                traditional_signal["market_phase"] = "neutral"

        if "market_phase" in rl_signal and not isinstance(rl_signal["market_phase"], str):
            if isinstance(rl_signal["market_phase"], (float, np.float64, np.float32, int, np.int64, np.int32)):
                # Convert numeric value to string representation
                value = float(rl_signal["market_phase"])
                if value > 0.5:
                    rl_signal["market_phase"] = "uptrend"
                elif value < -0.5:
                    rl_signal["market_phase"] = "downtrend"
                else:
                    rl_signal["market_phase"] = "neutral"
            else:
                rl_signal["market_phase"] = "neutral"

        # Both agree - use the stronger signal
        if (traditional_type.endswith("Buy") and rl_type.endswith("Buy")) or \
                (traditional_type.endswith("Sell") and rl_type.endswith("Sell")):
            rl_confidence = rl_signal.get("rl_confidence", 0.5)
            traditional_score = traditional_signal.get("ensemble_score", 0.5)

            if rl_confidence > traditional_score:
                final_signal = rl_signal.copy()
                final_signal["ensemble_score"] = rl_confidence
                final_signal["signal_source"] = "RL_Dominant"
            else:
                final_signal = traditional_signal.copy()
                final_signal["signal_source"] = "Traditional_Dominant"

            # Boost confidence when both agree
            final_signal["ensemble_score"] = min(0.95, final_signal["ensemble_score"] * 1.1)
            return final_signal

        # Disagreement - use weighted ensemble
        rl_weight = self.rl_ensemble_weight
        traditional_weight = 1 - rl_weight

        rl_confidence = rl_signal.get("rl_confidence", 0.5)
        traditional_score = traditional_signal.get("ensemble_score", 0.5)

        rl_value = 1 if rl_type.endswith("Buy") else (-1 if rl_type.endswith("Sell") else 0)
        traditional_value = 1 if traditional_type.endswith("Buy") else (-1 if traditional_type.endswith("Sell") else 0)

        weighted_signal = (rl_value * rl_weight * rl_confidence) + \
                          (traditional_value * traditional_weight * traditional_score)

        if abs(weighted_signal) < 0.2:
            return {
                "signal_type": "NoTrade",
                "reason": "EnsembleDisagreement",
                "direction": "long" if weighted_signal > 0 else "short",
                "ensemble_score": abs(weighted_signal) + 0.3,
                "rl_contribution": rl_value * rl_weight * rl_confidence,
                "traditional_contribution": traditional_value * traditional_weight * traditional_score
            }

        direction = "long" if weighted_signal > 0 else "short"
        signal_type = "Buy" if direction == "long" else "Sell"

        if abs(weighted_signal) > 0.5:
            signal_type = f"Strong{signal_type}"

        # Take metadata from dominant signal
        base_signal = rl_signal if abs(rl_value * rl_weight * rl_confidence) > abs(
            traditional_value * traditional_weight * traditional_score) else traditional_signal

        final_signal = {
            "signal_type": signal_type,
            "direction": direction,
            "predicted_return": base_signal.get("predicted_return", 0),
            "market_phase": base_signal.get("market_phase", "neutral"),
            "ensemble_score": abs(weighted_signal) + 0.3,
            "rl_contribution": rl_value * rl_weight * rl_confidence,
            "traditional_contribution": traditional_value * traditional_weight * traditional_score,
            "signal_source": "Ensemble"
        }

        return final_signal
    def _generate_base_signal(self, model_pred, df, market_regime):
        """
        Generate the base signal with robust error handling for all calculations.
        """
        try:
            # Extract essential indicators with safe defaults
            adx_value = self._get_indicator_value(df, 'adx_14', 25)
            bb_width = self._get_indicator_value(df, 'bb_width_20', 0.02)
            rsi_14 = self._get_indicator_value(df, 'rsi_14', 50)

            # Calculate key metrics with error handling
            try:
                volatility_regime = self._calculate_volatility_regime(df)
            except Exception:
                volatility_regime = 0.5

            try:
                price_momentum = self._calculate_price_momentum(df)
            except Exception:
                price_momentum = 0.0

            try:
                volume_trend = self._analyze_volume_trend(df)
            except Exception:
                volume_trend = 0.5

            # Determine market conditions
            is_ranging_market = adx_value < 22 or bb_width < 0.016
            is_trending_market = adx_value > 28 and abs(price_momentum) > 0.015

            # Get technical indicators safely
            try:
                ema_signal = self._check_ma_signal(df)
            except Exception:
                ema_signal = 0

            try:
                macd_signal = self._check_macd_signal(df)
            except Exception:
                macd_signal = 0

            try:
                volume_confirms = self._check_volume_confirmation(df, is_ranging_market)
            except Exception:
                volume_confirms = False

            # Calculate trend strength with safeguards
            trend_strength = min(1.0, adx_value / 50.0)
            if not np.isnan(price_momentum) and abs(price_momentum) > 0.02:
                trend_strength += 0.2
            if is_trending_market:
                trend_strength += 0.15
            trend_strength = min(1.0, max(0.1, trend_strength))

            # Construct the signal
            signal = {
                "signal_type": "NoTrade",
                "direction": "long" if model_pred > 0 else "short",
                "predicted_return": model_pred,
                "market_phase": market_regime.get("type", "neutral"),
                "regime_confidence": market_regime.get("confidence", 0.5),
                "adx": adx_value,
                "rsi_14": rsi_14,
                "bb_width": bb_width,
                "volatility": volatility_regime,
                "price_momentum": price_momentum,
                "volume_trend": volume_trend,
                "ema_signal": ema_signal,
                "macd_signal": macd_signal,
                "volume_confirms": volume_confirms,
                "is_ranging": is_ranging_market,
                "is_trending": is_trending_market,
                "trend_strength": trend_strength
            }

            return signal

        except Exception as e:
            self.logger.warning(f"Error in base signal generation: {e}")
            # Return minimal valid signal to avoid cascading errors
            return {
                "signal_type": "NoTrade",
                "direction": "long" if model_pred > 0 else "short",
                "predicted_return": model_pred,
                "market_phase": market_regime.get("type", "neutral"),
                "regime_confidence": market_regime.get("confidence", 0.5),
                "volatility": 0.5,
                "trend_strength": 0.5
            }

    def _enhance_signal(self, signal, df, multi_timeframe_signals, volume_profile):
        """
        Enhance the base signal with additional metrics and analysis.
        Includes robust error handling for all calculations.
        """
        try:
            enhanced_signal = signal.copy()

            # Add multi-timeframe analysis
            enhanced_signal["multi_timeframe"] = multi_timeframe_signals

            # Add volume profile analysis
            enhanced_signal["volume_profile"] = volume_profile

            # Get recent candles for order flow analysis with error handling
            try:
                recent_data = df.iloc[-20:] if len(df) > 20 else df

                # Calculate and add order flow metrics with error handling
                order_flow_metrics = self.volume_analyzer.calculate_order_flow_metrics(recent_data)
                enhanced_signal.update(order_flow_metrics)
            except Exception as e:
                self.logger.warning(f"Error in order flow analysis: {e}")
                # Add default order flow metrics
                enhanced_signal.update({
                    "buying_pressure": 0.5,
                    "selling_pressure": 0.5,
                    "volume_delta": 0,
                    "volume_trend": 0
                })

            # Add liquidity information safely
            try:
                if "liquidity_voids" in volume_profile:
                    current_price = enhanced_signal.get("entry_price", self._get_latest_price(df))
                    if not np.isnan(current_price) and current_price > 0:
                        nearby_voids = self._find_nearby_liquidity_voids(
                            current_price,
                            volume_profile.get("liquidity_voids", [])
                        )
                        enhanced_signal["nearby_liquidity_voids"] = nearby_voids
                    else:
                        enhanced_signal["nearby_liquidity_voids"] = []
                else:
                    enhanced_signal["nearby_liquidity_voids"] = []
            except Exception:
                enhanced_signal["nearby_liquidity_voids"] = []

            # Add multi-timeframe confirmation
            try:
                timeframe_alignment = multi_timeframe_signals.get("alignment", {})
                direction = 1 if enhanced_signal.get("direction") == "long" else -1

                if direction > 0 and timeframe_alignment.get("bullish", 0) > 0.6:
                    enhanced_signal["multi_timeframe_confirmation"] = True
                elif direction < 0 and timeframe_alignment.get("bearish", 0) > 0.6:
                    enhanced_signal["multi_timeframe_confirmation"] = True
                else:
                    enhanced_signal["multi_timeframe_confirmation"] = False
            except Exception:
                enhanced_signal["multi_timeframe_confirmation"] = False

            # Enhance with volume confirmation
            try:
                buying_pressure = enhanced_signal.get("buying_pressure", 0.5)
                selling_pressure = enhanced_signal.get("selling_pressure", 0.5)
                volume_delta = enhanced_signal.get("volume_delta", 0)

                if direction > 0 and (buying_pressure > 0.6 or volume_delta > 0.2):
                    enhanced_signal["volume_confirmation"] = True
                elif direction < 0 and (selling_pressure > 0.6 or volume_delta < -0.2):
                    enhanced_signal["volume_confirmation"] = True
                else:
                    enhanced_signal["volume_confirmation"] = False
            except Exception:
                enhanced_signal["volume_confirmation"] = False

            return enhanced_signal

        except Exception as e:
            self.logger.warning(f"Error in signal enhancement: {e}")
            # Return the original signal to avoid cascading failures
            signal["multi_timeframe"] = multi_timeframe_signals
            signal["volume_profile"] = volume_profile
            return signal

    def _apply_thresholds(self, signal, thresholds, adaptive_mode=False, win_streak=0, loss_streak=0):
        """
        Apply thresholds to determine final signal type.
        Includes comprehensive validation and error handling.
        """
        try:
            # Extract key metrics with safe defaults
            confidence_score = float(signal.get("confidence_score", 0))
            direction = signal.get("direction", "long")
            predicted_return = float(signal.get("predicted_return", 0))
            market_phase = signal.get("market_phase", "neutral")
            volume_confirms = bool(signal.get("volume_confirms", False))
            multi_timeframe_confirmation = bool(signal.get("multi_timeframe_confirmation", False))
            trend_strength = float(signal.get("trend_strength", 0.5))
            volatility = float(signal.get("volatility", 0.5))
            rsi_14 = float(signal.get("rsi_14", 50))

            # Apply adaptive mode adjustments if enabled
            if adaptive_mode:
                threshold_adjustments = {}
                if win_streak >= 3:
                    threshold_adjustments = {k: v * 0.85 for k, v in thresholds.items()}
                elif loss_streak >= 2:
                    threshold_adjustments = {k: v * 1.15 for k, v in thresholds.items()}

                thresholds = threshold_adjustments or thresholds

            # Determine signal type based on confidence score
            if confidence_score >= thresholds.get("very_strong", 0.0025):
                signal_type = "StrongBuy" if direction == "long" else "StrongSell"
            elif confidence_score >= thresholds.get("strong", 0.0015):
                signal_type = "Buy" if direction == "long" else "Sell"
            elif confidence_score >= thresholds.get("normal", 0.001):
                signal_type = "Buy" if direction == "long" else "Sell"
            elif confidence_score >= thresholds.get("weak", 0.0008):
                signal_type = "NoTrade"
                signal["reason"] = "BelowNormalThreshold"
            else:
                signal_type = "NoTrade"
                signal["reason"] = "BelowWeakThreshold"

            # Final checks for signal quality
            if signal_type != "NoTrade":
                # Combined signal quality check with reduced RSI dependency
                signal_quality_issues = []

                # Check RSI extremes with reduced importance
                rsi_extreme = (direction == "long" and rsi_14 > 75) or (direction == "short" and rsi_14 < 25)
                if rsi_extreme:
                    signal_quality_issues.append(0.5)  # Half weight compared to other factors

                # Check recent high volatility
                if volatility > 0.8 and not multi_timeframe_confirmation:
                    signal_quality_issues.append(1)

                # Check market phase alignment
                phase_misalignment = ((direction == "long" and market_phase in ["strong_downtrend", "downtrend"]) or
                                      (direction == "short" and market_phase in ["strong_uptrend", "uptrend"]))
                if phase_misalignment:
                    signal_quality_issues.append(1)

                # Check volume confirmation
                if not volume_confirms and volatility > 0.6:
                    signal_quality_issues.append(1)

                # Override if we have enough accumulated quality issues
                quality_score = sum(signal_quality_issues)
                if quality_score >= 2:
                    signal_type = "NoTrade"
                    signal["reason"] = "QualityCheckFailed"

                # Check for necessary technical confirmations
                if signal_type != "NoTrade":
                    confirmation_count = 0

                    # Check EMA confirmation
                    if (direction == "long" and signal.get("ema_signal", 0) > 0) or \
                            (direction == "short" and signal.get("ema_signal", 0) < 0):
                        confirmation_count += 1

                    # Check MACD confirmation
                    if (direction == "long" and signal.get("macd_signal", 0) > 0) or \
                            (direction == "short" and signal.get("macd_signal", 0) < 0):
                        confirmation_count += 1

                    # Check multi-timeframe confirmation
                    if multi_timeframe_confirmation:
                        confirmation_count += 1

                    # Check volume confirmation
                    if volume_confirms:
                        confirmation_count += 1

                    # Require at least two confirmations for weaker signals
                    if confirmation_count < 2 and abs(predicted_return) < thresholds.get("strong", 0.0015):
                        signal_type = "NoTrade"
                        signal["reason"] = "InsufficientConfirmation"

            # Update signal with final type and metrics
            signal["signal_type"] = signal_type
            signal["applied_threshold"] = thresholds.get("normal", 0.001)

            # Set ensemble_score based on confidence_score and component scores
            ensemble_score = confidence_score
            if "component_scores" in signal:
                component_scores = signal["component_scores"]
                # Calculate a weighted ensemble score
                ensemble_score = (
                        confidence_score * 0.5 +
                        component_scores.get("technical", 0.5) * 0.2 +
                        component_scores.get("volume", 0.5) * 0.15 +
                        component_scores.get("timeframe", 0.5) * 0.15
                )

            # Ensure ensemble score is valid
            signal["ensemble_score"] = min(1.0, max(0.1, ensemble_score))

            return signal

        except Exception as e:
            self.logger.warning(f"Error applying thresholds: {e}")
            # Return a safe default signal
            return {
                "signal_type": "NoTrade",
                "reason": "ThresholdError",
                "direction": signal.get("direction", "long"),
                "confidence_score": signal.get("confidence_score", 0.3),
                "ensemble_score": signal.get("confidence_score", 0.3),
                "applied_threshold": thresholds.get("normal", 0.001)
            }

    def _find_nearby_liquidity_voids(self, current_price, voids):
        """
        Find liquidity voids near the current price with error handling.
        """
        if not voids or current_price is None or current_price <= 0:
            return []

        try:
            nearby_voids = []
            for void in voids:
                if not isinstance(void, dict):
                    continue

                lower = void.get("price_low", 0)
                upper = void.get("price_high", 0)

                if lower <= 0 or upper <= 0 or upper <= lower:
                    continue

                mid_price = (lower + upper) / 2
                distance_pct = abs(current_price - mid_price) / current_price

                if distance_pct < 0.05:  # Within 5% of current price
                    void_info = void.copy()
                    void_info["distance_pct"] = distance_pct
                    void_info["is_above"] = mid_price > current_price
                    nearby_voids.append(void_info)

            return sorted(nearby_voids, key=lambda x: x.get("distance_pct", 1.0))
        except Exception as e:
            self.logger.warning(f"Error finding nearby liquidity voids: {e}")
            return []

    def _get_latest_price(self, df):
        """
        Safely extract the latest price from the dataframe.
        """
        if df is None or len(df) == 0:
            return 0.0

        try:
            if 'close' in df.columns:
                price = df['close'].iloc[-1]
                if not pd.isna(price) and price > 0:
                    return float(price)
            return 0.0
        except (IndexError, ValueError, TypeError):
            return 0.0

    def _get_indicator_value(self, df, indicator, default_value=0.0):
        """
        Safely extract indicator value from dataframe with robust error handling.
        """
        try:
            if df is None or len(df) == 0:
                return default_value

            # Try with the exact indicator name
            if indicator in df.columns and len(df) > 0:
                value = df[indicator].iloc[-1]
                if not pd.isna(value):
                    return float(value)

            # Try with the m30_ prefix
            indicator_alt = f'm30_{indicator}'
            if indicator_alt in df.columns and len(df) > 0:
                value = df[indicator_alt].iloc[-1]
                if not pd.isna(value):
                    return float(value)

            return default_value
        except (IndexError, ValueError, TypeError):
            return default_value

    def _calculate_volatility_regime(self, df):
        return self.indicator_util.detect_volatility_regime(df)

    def _calculate_price_momentum(self, df):
        if len(df) < 10:
            return 0.0

        close_prices = df['close'].values[-10:]
        ema9_values = df['ema_9'].values[-10:] if 'ema_9' in df.columns else np.zeros(10)

        if all(np.isnan(ema9_values)):
            if np.isnan(close_prices[-1]) or np.isnan(close_prices[-5]) or close_prices[-5] <= 0:
                return 0.0
            momentum = (close_prices[-1] / close_prices[-5] - 1) if close_prices[-5] > 0 else 0
        else:
            if np.isnan(ema9_values[-1]) or np.isnan(ema9_values[-5]) or ema9_values[-5] <= 0:
                return 0.0
            momentum = (ema9_values[-1] / ema9_values[-5] - 1) if ema9_values[-5] > 0 else 0

        return momentum

    def _analyze_volume_trend(self, df):
        if len(df) < 10 or 'volume' not in df.columns:
            return 0.5

        volumes = df['volume'].values[-10:]

        # Fix: Check for NaN values and zero volumes
        volumes = np.nan_to_num(volumes, nan=0.0)
        if np.all(volumes == 0):
            return 0.5

        recent_vol_avg = np.mean(volumes[-3:])
        prev_vol_avg = np.mean(volumes[-10:-3])

        # Fix: Check both averages
        if prev_vol_avg <= 0 or recent_vol_avg <= 0:
            return 0.5

        volume_change = recent_vol_avg / prev_vol_avg

        if volume_change > 1.5:
            return 0.9
        elif volume_change > 1.2:
            return 0.7
        elif volume_change < 0.7:
            return 0.2
        elif volume_change < 0.9:
            return 0.4
        else:
            return 0.5

    def _check_ma_signal(self, df):
        """
        Check for moving average signal with error handling.
        """
        try:
            if 'ema_9' not in df.columns or 'ema_21' not in df.columns or len(df) < 1:
                return 0

            ema9 = df['ema_9'].iloc[-1]
            ema21 = df['ema_21'].iloc[-1]

            if pd.isna(ema9) or pd.isna(ema21) or ema21 == 0:
                return 0

            if ema9 > ema21:
                return 1
            elif ema9 < ema21:
                return -1
            return 0
        except (IndexError, ValueError, TypeError):
            return 0

    def _check_macd_signal(self, df):
        """
        Check for MACD signal with error handling.
        """
        try:
            macd_col = 'macd_12_26'
            signal_col = 'macd_signal_12_26_9'

            # Try standard column names
            if macd_col in df.columns and signal_col in df.columns and len(df) > 0:
                macd = df[macd_col].iloc[-1]
                signal = df[signal_col].iloc[-1]

                if not pd.isna(macd) and not pd.isna(signal):
                    if macd > signal:
                        return 1
                    elif macd < signal:
                        return -1
                    return 0

            # Try with m30_ prefix
            m30_macd = 'm30_macd'
            m30_signal = 'm30_macd_signal'
            if m30_macd in df.columns and m30_signal in df.columns and len(df) >= 1:
                macd = df[m30_macd].iloc[-1]
                signal = df[m30_signal].iloc[-1]

                if not pd.isna(macd) and not pd.isna(signal):
                    if macd > signal:
                        return 1
                    elif macd < signal:
                        return -1

            return 0
        except (IndexError, ValueError, TypeError):
            return 0

    def _check_volume_confirmation(self, df, is_ranging=False):
        """
        Check for volume confirmation with comprehensive error handling.
        """
        try:
            if df is None or len(df) < 5 or 'volume' not in df.columns:
                return False

            # Check for NaN values
            if df['volume'].isna().any() or df['close'].isna().any():
                return False

            # Calculate OBV if not present
            if 'obv' not in df.columns and hasattr(self.indicator_util, 'calculate_obv'):
                try:
                    df_with_obv = self.indicator_util.calculate_obv(df.copy())
                    if 'obv' in df_with_obv.columns:
                        df = df_with_obv
                except Exception:
                    pass

            volume_confirmed = False

            # Check OBV confirmation
            if 'obv' in df.columns and not df['obv'].isna().any() and len(df) > 3:
                try:
                    obv_change = df['obv'].diff(3).iloc[-1]
                    price_change = df['close'].pct_change(3).iloc[-1]

                    if not pd.isna(obv_change) and not pd.isna(price_change) and price_change != 0:
                        if (price_change > 0 and obv_change > 0.5 * abs(price_change) * df['obv'].mean()) or \
                                (price_change < 0 and obv_change < 0.5 * abs(price_change) * df['obv'].mean()):
                            volume_confirmed = True
                except (IndexError, ValueError):
                    pass

            # Check CMF confirmation
            if 'cmf_20' in df.columns and not df['cmf_20'].isna().any():
                try:
                    cmf = df['cmf_20'].iloc[-1]
                    price_change = df['close'].pct_change(1).iloc[-1]

                    if not pd.isna(cmf) and not pd.isna(price_change):
                        if (price_change > 0 and cmf > 0.08) or (price_change < 0 and cmf < -0.08):
                            volume_confirmed = True
                except (IndexError, ValueError):
                    pass

            # Check volume pattern
            if 'volume' in df.columns and not df['volume'].isna().any() and not df['close'].isna().any():
                try:
                    volumes = df['volume'].values[-5:]
                    closes = df['close'].values[-5:]

                    if len(volumes) >= 4 and len(closes) >= 2:
                        short_avg_volume = np.mean(volumes[-4:-1])
                        current_volume = volumes[-1]

                        volume_increasing = False
                        if len(df) >= 10:
                            medium_avg_volume = np.mean(df['volume'].values[-10:-5])
                            volume_increasing = current_volume > medium_avg_volume * 1.1
                        else:
                            volume_increasing = True

                        volume_threshold = 1.2 if is_ranging else 1.0
                        volume_sufficient = current_volume > short_avg_volume * volume_threshold

                        price_change = closes[-1] / closes[-2] - 1 if closes[-2] > 0 else 0
                        price_significance = abs(price_change) > 0.002

                        if price_significance and ((price_change > 0 and volume_sufficient and volume_increasing) or
                                                   (price_change < 0 and volume_sufficient)):
                            volume_confirmed = True
                except (IndexError, ValueError, ZeroDivisionError):
                    pass

            # Special case for ranging markets: require more confirmation
            if is_ranging:
                confirmation_count = 0

                if 'obv' in df.columns and not df['obv'].isna().any() and len(df) > 3:
                    try:
                        obv_direction = np.sign(df['obv'].diff(3).iloc[-1])
                        price_direction = np.sign(df['close'].pct_change(3).iloc[-1])
                        if obv_direction == price_direction and obv_direction != 0:
                            confirmation_count += 1
                    except (IndexError, ValueError):
                        pass

                if 'cmf_20' in df.columns and not df['cmf_20'].isna().any():
                    try:
                        if abs(df['cmf_20'].iloc[-1]) > 0.08:
                            confirmation_count += 1
                    except (IndexError, ValueError):
                        pass

                if 'volume' in df.columns and not df['volume'].isna().any() and len(volumes) >= 4:
                    try:
                        if current_volume > short_avg_volume * 1.2:
                            confirmation_count += 1
                    except (NameError, IndexError, ValueError):
                        pass

                return confirmation_count >= 2

            return volume_confirmed

        except Exception as e:
            self.logger.warning(f"Error checking volume confirmation: {e}")
            return False

    def _calculate_fibonacci_levels(self, df):
        lookback = self.config.get("signal", "fibonacci_lookback", 144)
        recent_data = df.iloc[-lookback:] if len(df) > lookback else df

        high = recent_data['high'].max()
        low = recent_data['low'].min()

        levels = {
            0: low,
            0.236: low + (high - low) * 0.236,
            0.382: low + (high - low) * 0.382,
            0.5: low + (high - low) * 0.5,
            0.618: low + (high - low) * 0.618,
            0.764: low + (high - low) * 0.764,
            1: high
        }
        return levels

    def _is_range_bound_market(self, df):
        adx_threshold = self.config.get("signal", "fibonacci_adx_threshold", 25)
        bb_width_threshold = self.config.get("signal", "fibonacci_bb_threshold", 0.03)

        adx_value = self._get_indicator_value(df, 'adx_14', 100)
        bb_width = self._get_indicator_value(df, 'bb_width_20', 0.1)

        return adx_value < adx_threshold and bb_width < bb_width_threshold

    def _generate_fibonacci_signal(self, df, current_price, fibonacci_levels):
        long_entry = self.config.get("signal", "fibonacci_long_entry", 0.382)
        short_entry = self.config.get("signal", "fibonacci_short_entry", 0.618)
        long_block_start = self.config.get("signal", "fibonacci_long_block_start", 0.618)
        long_block_end = self.config.get("signal", "fibonacci_long_block_end", 1.0)
        short_block_start = self.config.get("signal", "fibonacci_short_block_start", 0.0)
        short_block_end = self.config.get("signal", "fibonacci_short_block_end", 0.382)

        if fibonacci_levels[long_entry] * 1.01 >= current_price >= fibonacci_levels[0]:
            if fibonacci_levels[long_block_start] <= current_price <= fibonacci_levels[long_block_end]:
                return {"signal_type": "NoTrade", "reason": "LongBlockingZone"}
            return {
                "signal_type": "FibonacciBuy",
                "direction": "long",
                "fibonacci_levels": fibonacci_levels,
                "entry_level": long_entry,
                "confidence": 0.7
            }

        elif fibonacci_levels[short_entry] * 0.99 <= current_price <= fibonacci_levels[1]:
            if fibonacci_levels[short_block_end] >= current_price >= fibonacci_levels[short_block_start]:
                return {"signal_type": "NoTrade", "reason": "ShortBlockingZone"}
            return {
                "signal_type": "FibonacciSell",
                "direction": "short",
                "fibonacci_levels": fibonacci_levels,
                "entry_level": short_entry,
                "confidence": 0.7
            }

        return {"signal_type": "NoTrade", "reason": "OutsideFibonacciZones"}