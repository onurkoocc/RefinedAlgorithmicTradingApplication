import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from scipy import stats


class SignalProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("SignalProcessor")
        self.confidence_threshold = config.get("signal", "confidence_threshold", 0.0008)
        self.strong_signal_threshold = config.get("signal", "strong_signal_threshold", 0.08)
        self.atr_multiplier_sl = config.get("signal", "atr_multiplier_sl", 2.5)
        self.use_regime_filter = config.get("signal", "use_regime_filter", True)
        self.use_volatility_filter = config.get("signal", "use_volatility_filter", True)
        self.rsi_overbought = config.get("signal", "rsi_overbought", 70)
        self.rsi_oversold = config.get("signal", "rsi_oversold", 30)
        self.return_threshold = config.get("signal", "return_threshold", 0.0001)
        self.phase_return_thresholds = {
            "neutral": 0.0001,
            "uptrend": 0.00008,
            "downtrend": 0.00012,
            "ranging_at_support": 0.00012,
            "ranging_at_resistance": 0.00015
        }
        self.support_resistance_proximity_pct = 0.03
        self.market_phases = ["uptrend", "downtrend", "ranging_at_support", "ranging_at_resistance", "neutral"]
        self.multi_timeframe_lookbacks = {
            "short": 5,
            "medium": 20,
            "long": 60
        }
        self.price_action_patterns = {
            "bullish_engulfing_weight": 0.8,
            "bearish_engulfing_weight": 0.8,
            "doji_weight": 0.4,
            "hammer_weight": 0.7,
            "shooting_star_weight": 0.7
        }
        self.volume_confirmation_requirements = {
            "neutral": 0.4,
            "uptrend": 0.5,
            "downtrend": 0.5,
            "ranging_at_support": 0.6,
            "ranging_at_resistance": 0.7
        }

    def generate_signal(self, model_pred: float, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        df.columns = [col.lower() for col in df.columns]
        if len(df) < 50:
            return {"signal_type": "NoTrade", "reason": "InsufficientData"}

        adaptive_mode = kwargs.get("adaptive_mode", False)
        win_streak = kwargs.get("win_streak", 0)
        loss_streak = kwargs.get("loss_streak", 0)

        if isinstance(model_pred, np.ndarray):
            if model_pred.size > 1:
                model_pred = float(model_pred[0])
            else:
                model_pred = float(model_pred)

        adjusted_threshold = self.return_threshold
        if adaptive_mode:
            if win_streak >= 3:
                adjusted_threshold = max(0.00007, self.return_threshold * 0.7)
            elif loss_streak >= 2:
                adjusted_threshold = min(0.0008, self.return_threshold * 1.1)

        if len(df) > 1:
            current_price = self._get_previous_price(df)
        else:
            return {"signal_type": "NoTrade", "reason": "InsufficientData"}

        if np.isnan(current_price) or current_price <= 0:
            return {"signal_type": "NoTrade", "reason": "InvalidPrice"}

        market_regime = self._detect_market_regime(df)
        volatility_regime = self._detect_volatility_regime(df)
        trend_strength = self._calculate_trend_strength(df)
        timeframe_confirmations = self._analyze_multi_timeframe_confirmations(df, model_pred > 0)
        price_action_score = self._detect_price_action_patterns(df, model_pred > 0)
        momentum = self._analyze_momentum(df)

        min_price_action_score = 0.02
        if price_action_score < min_price_action_score:
            min_price_action_score_adjusted = min_price_action_score
            if win_streak >= 2:
                min_price_action_score_adjusted *= 0.5
            if price_action_score < min_price_action_score_adjusted:
                return {
                    "signal_type": "NoTrade",
                    "reason": "WeakPriceActionPatterns",
                    "price_action_score": price_action_score,
                    "predicted_return": model_pred
                }

        predicted_direction = "bullish" if model_pred > 0 else "bearish"
        volume_confirms = self._check_volume_confirmation(df, predicted_direction == "bullish")
        support_resistance_levels = self._identify_support_resistance(df)
        nearest_level, distance_pct = self._get_nearest_level(current_price, support_resistance_levels)
        market_phase = self._detect_market_phase(df)

        # Make threshold higher for ranging_at_resistance markets
        phase_threshold = self.phase_return_thresholds.get(market_phase, adjusted_threshold)
        if market_phase == "ranging_at_resistance" and model_pred > 0:
            phase_threshold *= 1.2  # Require stronger signals in resistance zones for longs

        if win_streak >= 3:
            phase_threshold *= 0.6
        elif loss_streak >= 2:
            phase_threshold *= 1.1

        # Add volatility filter
        if volatility_regime > 0.7 and abs(model_pred) < phase_threshold * 1.3:
            return {
                "signal_type": "NoTrade",
                "reason": "HighVolatilityWeakSignal",
                "predicted_return": model_pred
            }

        # Avoid trades against strong momentum
        if model_pred > 0 and momentum < -0.3:  # If trying to go long against strong downward momentum
            return {
                "signal_type": "NoTrade",
                "reason": "AgainstMomentum",
                "predicted_return": model_pred
            }

        if model_pred < 0 and momentum > 0.3:  # If trying to go short against strong upward momentum
            return {
                "signal_type": "NoTrade",
                "reason": "AgainstMomentum",
                "predicted_return": model_pred
            }

        if abs(model_pred) < phase_threshold:
            return {
                "signal_type": "NoTrade",
                "confidence": abs(model_pred),
                "reason": "LowConfidence",
                "predicted_return": model_pred
            }

        if model_pred > 0 and distance_pct < self.support_resistance_proximity_pct * 0.3 and nearest_level > current_price:
            if win_streak >= 2:
                pass
            else:
                return {
                    "signal_type": "NoTrade",
                    "reason": "TooCloseToResistance",
                    "distance_pct": distance_pct,
                    "predicted_return": model_pred
                }

        if model_pred < 0 and distance_pct < self.support_resistance_proximity_pct * 0.3 and nearest_level < current_price:
            if win_streak >= 2:
                pass
            else:
                return {
                    "signal_type": "NoTrade",
                    "reason": "TooCloseToSupport",
                    "distance_pct": distance_pct,
                    "predicted_return": model_pred
                }

        if market_phase == "ranging_at_resistance" and model_pred > 0:
            if abs(model_pred) < phase_threshold * 1.1 and win_streak < 2:
                return {
                    "signal_type": "NoTrade",
                    "reason": "RangingAtResistanceNeedStrongerSignal",
                    "market_phase": market_phase,
                    "predicted_return": model_pred
                }

        if model_pred > 0 and market_regime < -0.8:
            if win_streak >= 2:
                pass
            else:
                return {
                    "signal_type": "NoTrade",
                    "reason": "StrongDowntrendForLong",
                    "market_phase": market_phase,
                    "predicted_return": model_pred
                }

        if model_pred < 0 and market_regime > 0.8:
            if win_streak >= 2:
                pass
            else:
                return {
                    "signal_type": "NoTrade",
                    "reason": "StrongUptrendForShort",
                    "market_phase": market_phase,
                    "predicted_return": model_pred
                }

        extra_signals = self._calculate_additional_signals(df)
        ensemble_score = self._compute_ensemble_score(
            model_prediction=model_pred,
            trend_strength=trend_strength,
            volume_confirms=volume_confirms,
            price_action_score=price_action_score,
            timeframe_confirmations=timeframe_confirmations,
            extra_signals=extra_signals
        )

        phase_ensemble_factor = 1.0
        if market_phase == "neutral":
            phase_ensemble_factor = 0.6
        elif market_phase == "ranging_at_resistance":
            phase_ensemble_factor = 0.85

        if win_streak >= 3:
            phase_ensemble_factor *= 0.7

        adjusted_ensemble_threshold = self.return_threshold * 100 * phase_ensemble_factor
        if abs(ensemble_score) < adjusted_ensemble_threshold:
            if win_streak >= 3 and abs(ensemble_score) >= adjusted_ensemble_threshold * 0.7:
                pass
            else:
                self.logger.debug(f"Low ensemble score: {ensemble_score}, threshold: {adjusted_ensemble_threshold}")
                return {
                    "signal_type": "NoTrade",
                    "reason": "LowEnsembleScore",
                    "ensemble_score": ensemble_score,
                    "predicted_return": model_pred
                }

        rsi_14 = self._get_rsi(df)
        if rsi_14 is not None:
            # Check for overbought/oversold in direction of trade - stricter thresholds
            rsi_ob_threshold = self.rsi_overbought - 5  # Reduced threshold (from -15)
            rsi_os_threshold = self.rsi_oversold + 5  # Reduced threshold (from +15)

            if market_phase == "neutral":
                rsi_ob_threshold = self.rsi_overbought + 5
                rsi_os_threshold = self.rsi_oversold - 5

            if model_pred > 0 and rsi_14 >= rsi_ob_threshold and win_streak < 2:
                return {"signal_type": "NoTrade", "reason": "OverboughtRSI", "rsi": rsi_14}

            if model_pred < 0 and rsi_14 <= rsi_os_threshold and win_streak < 2:
                return {"signal_type": "NoTrade", "reason": "OversoldRSI", "rsi": rsi_14}

        macd_hist = self._get_macd_histogram(df)
        if macd_hist is not None:
            if model_pred > 0 and macd_hist < -0.0025 and win_streak < 2:
                return {"signal_type": "NoTrade", "reason": "NegativeMACD", "macd_hist": macd_hist}
            if model_pred < 0 and macd_hist > 0.0025 and win_streak < 2:
                return {"signal_type": "NoTrade", "reason": "PositiveMACD", "macd_hist": macd_hist}

        atr_value = self._compute_atr(df)
        if np.isnan(atr_value) or atr_value <= 0:
            atr_value = current_price * 0.01

        stop_adjustment = 1.0
        if loss_streak >= 2:
            stop_adjustment = 1.3
        elif win_streak >= 3:
            stop_adjustment = 0.8

        if market_phase == "neutral":
            stop_adjustment *= 0.9
        elif market_phase == "ranging_at_resistance":
            stop_adjustment *= 0.8

        if model_pred > 0:
            signal = self._create_bullish_signal(
                current_price, atr_value, abs(model_pred),
                market_regime, volatility_regime,
                stop_adjustment=stop_adjustment
            )
            signal['trend_strength'] = trend_strength
            signal['volume_confirmation'] = volume_confirms
            signal['market_phase'] = market_phase
            signal['ensemble_score'] = ensemble_score
            signal['momentum'] = float(extra_signals.get('momentum', 0))
            signal['nearest_resistance'] = nearest_level if nearest_level > current_price else None
            signal['nearest_support'] = nearest_level if nearest_level < current_price else None
            signal['predicted_return'] = model_pred
            signal['price_action_score'] = price_action_score
            signal['timeframe_confirmations'] = timeframe_confirmations
            if rsi_14 is not None:
                signal['rsi_14'] = rsi_14
            if macd_hist is not None:
                signal['macd_histogram'] = macd_hist
            return signal
        else:
            signal = self._create_bearish_signal(
                current_price, atr_value, abs(model_pred),
                market_regime, volatility_regime,
                stop_adjustment=stop_adjustment
            )
            signal['trend_strength'] = trend_strength
            signal['volume_confirmation'] = volume_confirms
            signal['market_phase'] = market_phase
            signal['ensemble_score'] = ensemble_score
            signal['momentum'] = float(extra_signals.get('momentum', 0))
            signal['nearest_resistance'] = nearest_level if nearest_level > current_price else None
            signal['nearest_support'] = nearest_level if nearest_level < current_price else None
            signal['predicted_return'] = model_pred
            signal['price_action_score'] = price_action_score
            signal['timeframe_confirmations'] = timeframe_confirmations
            if rsi_14 is not None:
                signal['rsi_14'] = rsi_14
            if macd_hist is not None:
                signal['macd_histogram'] = macd_hist
            return signal

    def _get_previous_price(self, df: pd.DataFrame) -> float:
        try:
            if 'actual_close' in df.columns and len(df) > 1:
                return float(df['actual_close'].iloc[-2])
            elif 'close' in df.columns and len(df) > 1:
                return float(df['close'].iloc[-2])
            else:
                return 0.0
        except (IndexError, ValueError):
            return 0.0

    def _analyze_multi_timeframe_confirmations(self, df: pd.DataFrame, is_bullish: bool) -> int:
        if len(df) < self.multi_timeframe_lookbacks["long"]:
            return 0
        confirmations = 0
        try:
            short_window = min(self.multi_timeframe_lookbacks["short"], len(df) - 1)
            short_close_change = df['close'].iloc[-2] / df['close'].iloc[-1 - short_window] - 1
            medium_window = min(self.multi_timeframe_lookbacks["medium"], len(df) - 1)
            medium_close_change = df['close'].iloc[-2] / df['close'].iloc[-1 - medium_window] - 1
            long_window = min(self.multi_timeframe_lookbacks["long"], len(df) - 1)
            long_close_change = df['close'].iloc[-2] / df['close'].iloc[-1 - long_window] - 1
            if (is_bullish and short_close_change > 0) or (not is_bullish and short_close_change < 0):
                confirmations += 1
            if (is_bullish and medium_close_change > 0) or (not is_bullish and medium_close_change < 0):
                confirmations += 1
            if (is_bullish and long_close_change > 0) or (not is_bullish and long_close_change < 0):
                confirmations += 1
            return confirmations
        except Exception as e:
            self.logger.warning(f"Error in multi-timeframe confirmation: {e}")
            return 0

    def _detect_price_action_patterns(self, df: pd.DataFrame, is_bullish: bool) -> float:
        if len(df) < 10:
            return 0.0
        try:
            recent_df = df.iloc[-10:].copy()
            patterns_score = 0.0
            opens = recent_df['open'].values
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            closes = recent_df['close'].values
            if len(closes) >= 2:
                prev_body_size = abs(closes[-2] - opens[-2])
                curr_body_size = abs(closes[-1] - opens[-1])
                if is_bullish and closes[-1] > opens[-1] and closes[-2] < opens[-2] and \
                        opens[-1] < closes[-2] and closes[-1] > opens[-2] and \
                        curr_body_size > prev_body_size * 1.1:
                    patterns_score += self.price_action_patterns["bullish_engulfing_weight"]
                elif not is_bullish and closes[-1] < opens[-1] and closes[-2] > opens[-2] and \
                        opens[-1] > closes[-2] and closes[-1] < opens[-2] and \
                        curr_body_size > prev_body_size * 1.1:
                    patterns_score += self.price_action_patterns["bearish_engulfing_weight"]
            if len(closes) >= 1:
                last_body_size = abs(closes[-1] - opens[-1])
                last_range = highs[-1] - lows[-1]
                if last_range > 0:
                    body_to_range_ratio = last_body_size / last_range
                    if is_bullish and body_to_range_ratio < 0.4 and \
                            min(opens[-1], closes[-1]) - lows[-1] > last_body_size * 2 and \
                            highs[-1] - max(opens[-1], closes[-1]) < last_body_size * 0.5:
                        patterns_score += self.price_action_patterns["hammer_weight"]
                    elif not is_bullish and body_to_range_ratio < 0.4 and \
                            highs[-1] - max(opens[-1], closes[-1]) > last_body_size * 2 and \
                            min(opens[-1], closes[-1]) - lows[-1] < last_body_size * 0.5:
                        patterns_score += self.price_action_patterns["shooting_star_weight"]
            if len(closes) >= 1:
                last_body_size = abs(closes[-1] - opens[-1])
                last_range = highs[-1] - lows[-1]
                if last_range > 0 and last_body_size / last_range < 0.1:
                    patterns_score += self.price_action_patterns["doji_weight"] * 0.5
            return min(1.0, patterns_score / 3.0)
        except Exception as e:
            self.logger.warning(f"Error detecting price action patterns: {e}")
            return 0.0

    def _identify_support_resistance(self, df: pd.DataFrame) -> List[float]:
        levels = []
        if len(df) < 100:
            return levels
        pivot_points = self._find_pivot_points(df, window=10)
        for pivot in pivot_points:
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

    def _find_pivot_points(self, df: pd.DataFrame, window: int = 10) -> List[Dict[str, Any]]:
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

    def _get_nearest_level(self, current_price: float, levels: List[float]) -> Tuple[float, float]:
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

    def _detect_market_phase(self, df: pd.DataFrame) -> str:
        if len(df) < 50:
            return "neutral"
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            ema20 = self._calculate_ema(closes, 20)
            ema50 = self._calculate_ema(closes, 50)
            recent_close = closes[-1]
            recent_ema20 = ema20[-1]
            recent_ema50 = ema50[-1]
            ema20_slope = (ema20[-1] / ema20[-5] - 1) * 100
            ema50_slope = (ema50[-1] / ema50[-10] - 1) * 100
            recent_atr = self._calculate_atr(highs[-20:], lows[-20:], closes[-20:])
            price_volatility = recent_atr / recent_close
            sr_levels = self._identify_support_resistance(df)
            nearest_level, distance = self._get_nearest_level(recent_close, sr_levels)
            if recent_close > recent_ema20 > recent_ema50 and ema20_slope > 0.18 and ema50_slope > 0.08:
                return "uptrend"
            elif recent_close < recent_ema20 < recent_ema50 and ema20_slope < -0.18 and ema50_slope < -0.08:
                return "downtrend"
            elif abs(ema20_slope) < 0.18 and distance < 0.022:
                if nearest_level > recent_close:
                    return "ranging_at_support"
                else:
                    return "ranging_at_resistance"
            else:
                return "neutral"
        except Exception as e:
            self.logger.warning(f"Error in market phase detection: {e}")
            return "neutral"

    def _calculate_ema(self, values: np.ndarray, period: int) -> np.ndarray:
        if len(values) < period:
            return np.zeros_like(values)
        ema = np.zeros_like(values)
        alpha = 2 / (period + 1)
        ema[:period] = np.mean(values[:period])
        for i in range(period, len(values)):
            ema[i] = values[i] * alpha + ema[i - 1] * (1 - alpha)
        return ema

    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        if len(highs) < period + 1:
            return 0.01 * closes[-1]
        true_ranges = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            true_ranges.append(max(high_low, high_close, low_close))
        if not true_ranges:
            return 0.01 * closes[-1]
        return sum(true_ranges[-period:]) / min(len(true_ranges), period)

    def _calculate_additional_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        signals = {}
        try:
            signals['price_action'] = self._analyze_price_action(df)
            if 'volume' in df.columns:
                signals['volume_trend'] = self._analyze_volume_trend(df)
            signals['volatility_signal'] = self._analyze_volatility(df)
            signals['momentum'] = self._analyze_momentum(df)
            signals['time_cycle'] = self._analyze_time_cycle(df)
        except Exception as e:
            self.logger.warning(f"Error calculating additional signals: {e}")
        return signals

    def _analyze_time_cycle(self, df: pd.DataFrame) -> float:
        try:
            if len(df) < 100:
                return 0.0
            if isinstance(df.index, pd.DatetimeIndex):
                hour_of_day = df.index[-1].hour
                if 8 <= hour_of_day <= 12:
                    return 0.2
                elif 12 < hour_of_day <= 16:
                    return 0.1
                elif 20 <= hour_of_day <= 23:
                    return -0.1
                else:
                    return 0.0
            return 0.0
        except Exception as e:
            return 0.0

    def _analyze_price_action(self, df: pd.DataFrame) -> float:
        try:
            closes = df['close'].values[-10:]
            opens = df['open'].values[-10:]
            body_sizes = np.abs(closes - opens)
            ranges = df['high'].values[-10:] - df['low'].values[-10:]
            relative_body_sizes = body_sizes / np.maximum(ranges, 0.0001)
            bullish_candles = sum(1 for i in range(len(closes)) if closes[i] > opens[i])
            bullish_ratio = bullish_candles / len(closes)
            recent_direction = 1 if closes[-1] > closes[-5] else -1
            doji_threshold = 0.15
            doji_count = sum(1 for rb in relative_body_sizes if rb < doji_threshold)
            doji_factor = 1.0 - (doji_count / len(relative_body_sizes) * 0.5)
            recent_momentum = abs(closes[-1] / closes[-3] - 1) * 10
            momentum_factor = min(1.5, max(0.5, 1.0 + recent_momentum))
            signal = recent_direction * (bullish_ratio - 0.5) * 2 * np.mean(
                relative_body_sizes) * doji_factor * momentum_factor
            return min(1.0, max(-1.0, signal))
        except Exception as e:
            return 0.0

    def _analyze_volume_trend(self, df: pd.DataFrame) -> float:
        volumes = df['volume'].values[-20:]
        closes = df['close'].values[-20:]
        if len(volumes) < 5:
            return 0.0
        recent_vol_avg = np.mean(volumes[-5:])
        older_vol_avg = np.mean(volumes[-20:-5])
        vol_change = recent_vol_avg / older_vol_avg if older_vol_avg > 0 else 1.0
        price_changes = np.diff(closes)
        vol_changes = np.diff(volumes)
        if len(price_changes) > 1 and len(vol_changes) > 1:
            recent_weights = np.linspace(0.5, 1.0, min(len(price_changes), len(vol_changes)))
            correlations = []
            for i in range(min(len(price_changes), len(vol_changes))):
                weight = recent_weights[i]
                if price_changes[i] > 0 and vol_changes[i] > 0:
                    correlations.append(1 * weight)
                elif price_changes[i] < 0 and vol_changes[i] > 0:
                    correlations.append(-1 * weight)
                else:
                    correlations.append(0)
            vol_price_corr = np.sum(correlations) / np.sum(recent_weights)
        else:
            vol_price_corr = 0
        recent_vol_max = np.max(volumes[-5:])
        avg_vol = np.mean(volumes[-20:])
        vol_spike_factor = 1.0
        if recent_vol_max > avg_vol * 1.5:
            vol_spike_factor = 1.3
        signal = vol_price_corr * min(2.0, max(0.5, vol_change)) / 2.0 * vol_spike_factor
        return min(1.0, max(-1.0, signal))

    def _analyze_volatility(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.0
        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]
        closes = df['close'].values[-20:]
        atr = self._calculate_atr(highs, lows, closes)
        norm_atr = atr / closes[-1]
        short_atr = self._calculate_atr(highs[-10:], lows[-10:], closes[-10:])
        medium_atr = self._calculate_atr(highs[-15:], lows[-15:], closes[-15:])
        vol_trend = 0.0
        if short_atr > medium_atr:
            vol_trend = -0.2
        else:
            vol_trend = 0.1
        if norm_atr > 0.03:
            return -0.4 + vol_trend
        elif norm_atr < 0.01:
            return 0.3 + vol_trend
        else:
            return vol_trend

    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.0
        closes = df['close'].values
        roc1 = (closes[-1] / closes[-2] - 1) if len(closes) > 2 else 0
        roc5 = (closes[-1] / closes[-6] - 1) if len(closes) > 6 else 0
        roc10 = (closes[-1] / closes[-11] - 1) if len(closes) > 11 else 0
        roc20 = (closes[-1] / closes[-21] - 1) if len(closes) > 21 else 0
        market_phase = self._detect_market_phase(df)
        weights = {
            'roc1': 0.2,
            'roc5': 0.5,
            'roc10': 0.3,
            'roc20': 0.0
        }
        if market_phase == "neutral":
            weights = {
                'roc1': 0.25,
                'roc5': 0.55,
                'roc10': 0.2,
                'roc20': 0.0
            }
        elif market_phase == "uptrend" or market_phase == "downtrend":
            weights = {
                'roc1': 0.15,
                'roc5': 0.35,
                'roc10': 0.3,
                'roc20': 0.2
            }
        elif market_phase == "ranging_at_resistance" or market_phase == "ranging_at_support":
            weights = {
                'roc1': 0.35,
                'roc5': 0.5,
                'roc10': 0.15,
                'roc20': 0.0
            }
        momentum_score = (
                roc1 * weights['roc1'] +
                roc5 * weights['roc5'] +
                roc10 * weights['roc10'] +
                roc20 * weights['roc20']
        ) * 12
        return min(1.0, max(-1.0, momentum_score))

    def _compute_ensemble_score(self, model_prediction: float, trend_strength: float,
                                volume_confirms: bool, extra_signals: Dict[str, float], **kwargs) -> float:
        base_score = model_prediction * 25.0 * np.sign(model_prediction) * (abs(model_prediction) ** 0.7)
        trend_factor = 0.5 + (1.0 * trend_strength ** 0.5)
        base_score *= trend_factor
        volume_factor = 1.45 if volume_confirms else 0.65
        base_score *= volume_factor
        signal_weights = {
            'price_action': 0.40,
            'volume_trend': 0.35,
            'volatility_signal': 0.20,
            'momentum': 0.45,
            'time_cycle': 0.15
        }
        if 'price_action_score' in kwargs and kwargs['price_action_score'] is not None:
            if 'price_action' not in extra_signals:
                extra_signals['price_action'] = float(kwargs['price_action_score'])
        signal_contributions = []
        signal_weight_sum = 0
        for signal_type, signal_value in extra_signals.items():
            if signal_type in signal_weights:
                weight = signal_weights[signal_type]
                scaled_signal = 2.0 / (1.0 + np.exp(-5.0 * signal_value)) - 1.0
                if np.sign(signal_value) == np.sign(model_prediction):
                    alignment_bonus = 0.2
                else:
                    alignment_bonus = -0.15
                contribution = (scaled_signal + alignment_bonus) * weight
                signal_contributions.append(contribution)
                signal_weight_sum += weight
        if signal_contributions and signal_weight_sum > 0:
            extra_contribution = sum(signal_contributions) / signal_weight_sum
            model_weight = 0.65 + (0.15 * abs(model_prediction) / 0.005)
            signal_weight = 1.0 - model_weight
            ensemble_score = (base_score * model_weight) + (extra_contribution * signal_weight)
        else:
            ensemble_score = base_score
        return np.tanh(ensemble_score)

    def _get_current_price(self, df: pd.DataFrame) -> float:
        if 'actual_close' in df.columns:
            return float(df['actual_close'].iloc[-1])
        else:
            return float(df['close'].iloc[-1])

    def _get_rsi(self, df: pd.DataFrame) -> Optional[float]:
        for col in ['rsi_14', 'm30_rsi_14']:
            if col in df.columns:
                try:
                    rsi = float(df[col].iloc[-1])
                    if not np.isnan(rsi) and 0 <= rsi <= 100:
                        return rsi
                except:
                    pass
        return None

    def _get_macd_histogram(self, df: pd.DataFrame) -> Optional[float]:
        for col in ['macd_histogram', 'm30_macd_histogram']:
            if col in df.columns:
                try:
                    hist = float(df[col].iloc[-1])
                    if not np.isnan(hist):
                        return hist
                except:
                    pass
        return None

    def _detect_market_regime(self, df: pd.DataFrame) -> float:
        if 'market_regime' in df.columns:
            return float(df['market_regime'].iloc[-1])
        try:
            if 'm30_ema_20' in df.columns:
                ema_20 = float(df['m30_ema_20'].iloc[-1])
                price = float(df['close'].iloc[-1])
                if ema_20 > 0:
                    deviation = (price / ema_20) - 1
                    strength = min(1.0, max(-1.0, deviation * 7))
                    return strength
            elif 'ema_20' in df.columns:
                ema_20 = float(df['ema_20'].iloc[-1])
                price = float(df['close'].iloc[-1])
                if ema_20 > 0:
                    deviation = (price / ema_20) - 1
                    strength = min(1.0, max(-1.0, deviation * 7))
                    return strength
            elif 'm30_force_index' in df.columns:
                force_idx = float(df['m30_force_index'].iloc[-1])
                close = float(df['close'].iloc[-1])
                if close > 0:
                    norm_force = np.tanh(force_idx / (close * 10000))
                    return norm_force
            elif 'close' in df.columns and len(df) >= 20:
                ema_values = df['close'].ewm(span=20, adjust=False).mean().values
                price = float(df['close'].iloc[-1])
                ema_20 = ema_values[-1]
                if ema_20 > 0:
                    deviation = (price / ema_20) - 1
                    strength = min(1.0, max(-1.0, deviation * 7))
                    return strength
        except Exception as e:
            self.logger.warning(f"Error detecting market regime: {e}")
        return 0.0

    def _detect_volatility_regime(self, df: pd.DataFrame) -> float:
        if 'volatility_regime' in df.columns:
            return float(df['volatility_regime'].iloc[-1])
        try:
            if 'm30_bb_width' in df.columns:
                bb_width = float(df['m30_bb_width'].iloc[-1])
                scaled_vol = min(0.9, max(0.2, (bb_width - 0.01) * 20))
                return scaled_vol
            elif 'bb_width' in df.columns:
                bb_width = float(df['bb_width'].iloc[-1])
                scaled_vol = min(0.9, max(0.2, (bb_width - 0.01) * 20))
                return scaled_vol
            elif 'm30_stoch_k' in df.columns and 'm30_stoch_d' in df.columns:
                stoch_k = float(df['m30_stoch_k'].iloc[-1])
                stoch_d = float(df['m30_stoch_d'].iloc[-1])
                extreme = min(abs(stoch_k - 50), abs(stoch_d - 50)) / 50
                return 0.3 + (0.7 * extreme)
        except Exception as e:
            self.logger.warning(f"Error detecting volatility regime: {e}")
        return 0.5

    def _compute_atr(self, df: pd.DataFrame) -> float:
        price = self._get_previous_price(df)
        default_atr = price * 0.01
        if len(df) < 15:
            return default_atr
        for col in ['atr_14', 'm30_atr_14']:
            if col in df.columns and len(df) > 1:
                try:
                    atr_val = float(df[col].iloc[-2])
                    if not np.isnan(atr_val) and atr_val > 0:
                        return atr_val
                except (IndexError, ValueError):
                    pass
        try:
            highs = df['high'].values[:-1]
            lows = df['low'].values[:-1]
            closes = df['close'].values[:-1]
            if len(closes) < 14:
                return default_atr
            true_ranges = []
            for i in range(1, len(highs)):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i - 1])
                low_close = abs(lows[i] - closes[i - 1])
                true_ranges.append(max(high_low, high_close, low_close))
            if not true_ranges:
                return default_atr
            return sum(true_ranges[-14:]) / min(len(true_ranges), 14)
        except Exception:
            return default_atr

    def _check_volume_confirmation(self, df: pd.DataFrame, is_bullish: bool) -> bool:
        try:
            if len(df) < 5:
                return False
            volumes = df['volume'].values[-5:]
            closes = df['close'].values[-5:]
            short_avg_volume = np.mean(volumes[-4:-1])
            current_volume = volumes[-1]
            if len(df) >= 10:
                medium_avg_volume = np.mean(df['volume'].values[-10:-5])
                medium_volume_increasing = current_volume > medium_avg_volume * 0.9
            else:
                medium_volume_increasing = True
            volume_sufficient = current_volume > short_avg_volume * 0.3
            if is_bullish:
                price_change = closes[-1] / closes[-2] - 1
                if price_change > 0 and (volume_sufficient or medium_volume_increasing):
                    return True
            else:
                price_change = closes[-1] / closes[-2] - 1
                if price_change < 0 and (volume_sufficient or medium_volume_increasing):
                    return True
            if is_bullish and price_change > 0 and len(volumes) >= 3:
                vol_trend = volumes[-1] / np.mean(volumes[-3:-1])
                if price_change > 0.003 and vol_trend > 0.5:
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Error in volume confirmation: {e}")
            return False

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        try:
            if len(df) < 20:
                return 0.5
            closes = df['close'].values
            short_term = np.mean(closes[-5:])
            medium_term = np.mean(closes[-10:])
            long_term = np.mean(closes[-20:])
            very_short_term = np.mean(closes[-3:])
            current_price = closes[-1]
            very_short_trend_up = current_price > very_short_term
            short_trend_up = current_price > short_term
            medium_trend_up = current_price > medium_term
            long_trend_up = current_price > long_term
            vs_pct = abs(current_price / very_short_term - 1) * 1.2
            short_pct = abs(current_price / short_term - 1)
            medium_pct = abs(current_price / medium_term - 1) * 0.8
            long_pct = abs(current_price / long_term - 1) * 0.6
            if very_short_trend_up == short_trend_up == medium_trend_up == long_trend_up:
                weighted_pct = (vs_pct * 1.2 + short_pct * 1.0 + medium_pct * 0.8 + long_pct * 0.6) / 3.6
                strength = min(1.0, weighted_pct * 60)
                return 0.5 + (0.5 * strength if short_trend_up else -0.5 * strength)
            else:
                agreements = sum([1 if t == very_short_trend_up else 0
                                  for t in [short_trend_up, medium_trend_up, long_trend_up]])
                return 0.3 + (0.225 * agreements)
        except Exception as e:
            self.logger.warning(f"Error in trend strength calculation: {e}")
            return 0.5

    def _create_bullish_signal(self, current_price, atr_value, confidence,
                               market_regime, volatility_regime, stop_adjustment=1.0):
        # Use wider initial stops based on volatility
        base_atr_mult = 4.2  # Increased from 3.5 to give wider stops

        if volatility_regime > 0.7:
            vol_adjusted_mult = base_atr_mult * 1.8  # Increased from 1.5
        elif volatility_regime < 0.3:
            vol_adjusted_mult = base_atr_mult * 0.9  # Slightly increased
        else:
            vol_adjusted_mult = base_atr_mult * (1.0 + (volatility_regime - 0.5) * 1.2)

        if confidence > 0.005:
            vol_adjusted_mult *= 0.9
        elif confidence < 0.003:
            vol_adjusted_mult *= 1.2

        if market_regime < -0.25:
            vol_adjusted_mult *= 1.3  # Increased from 1.1 to provide wider stops in downtrends

        vol_adjusted_mult *= stop_adjustment

        # Calculate stop loss with wider initial placement
        stop_loss_price = max(0.001, current_price - (vol_adjusted_mult * atr_value))
        risk = current_price - stop_loss_price

        # Set more aggressive first profit target (for quick exits)
        tp1_multiplier = 1.2 + (confidence * 80)  # Reduced from 1.4 to encourage quicker exits
        tp2_multiplier = 2.2 + (confidence * 180)

        tp1_distance = tp1_multiplier * risk
        tp2_distance = tp2_multiplier * risk

        return {
            "signal_type": "StrongBuy" if confidence >= 0.005 else "Buy",
            "confidence": confidence,
            "direction": "long",
            "entry_price": current_price,
            "stop_loss": stop_loss_price,
            "take_profit1": current_price + tp1_distance,
            "take_profit2": current_price + tp2_distance,
            "regime": market_regime,
            "volatility": volatility_regime,
            "atr": atr_value,
            "risk_reward_ratio1": tp1_multiplier,
            "risk_reward_ratio2": tp2_multiplier
        }

    def _create_bearish_signal(self, current_price, atr_value, confidence,
                               market_regime, volatility_regime, stop_adjustment=1.0):
        # Use wider initial stops based on volatility
        base_atr_mult = 4.2  # Increased from 3.5

        if volatility_regime > 0.7:
            vol_adjusted_mult = base_atr_mult * 1.8  # Increased from 1.5
        elif volatility_regime < 0.3:
            vol_adjusted_mult = base_atr_mult * 0.9  # Slightly increased
        else:
            vol_adjusted_mult = base_atr_mult * (1.0 + (volatility_regime - 0.5) * 1.2)

        if confidence > 0.005:
            vol_adjusted_mult *= 0.9
        elif confidence < 0.003:
            vol_adjusted_mult *= 1.2

        if market_regime > 0.25:
            vol_adjusted_mult *= 1.3  # Increased from 1.1

        vol_adjusted_mult *= stop_adjustment

        # Calculate stop with wider initial placement
        stop_loss_price = current_price + (vol_adjusted_mult * atr_value)
        risk = stop_loss_price - current_price

        # Set more aggressive profit targets
        tp_multiplier = 1.2 + (confidence * 80)  # Reduced from 1.4
        take_profit = current_price - (tp_multiplier * risk)

        return {
            "signal_type": "StrongSell" if confidence >= 0.005 else "Sell",
            "confidence": confidence,
            "direction": "short",
            "entry_price": current_price,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit,
            "reward_risk_ratio": tp_multiplier,
            "regime": market_regime,
            "volatility": volatility_regime,
            "atr": atr_value
        }