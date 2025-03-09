import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple


class SignalProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("SignalProcessor")

        self.confidence_threshold = config.get("signal", "confidence_threshold", 0.35)
        self.strong_signal_threshold = config.get("signal", "strong_signal_threshold", 0.65)
        self.atr_multiplier_sl = config.get("signal", "atr_multiplier_sl", 2.5)
        self.use_regime_filter = config.get("signal", "use_regime_filter", True)
        self.use_volatility_filter = config.get("signal", "use_volatility_filter", True)

        self.rsi_overbought = config.get("signal", "rsi_overbought", 70)
        self.rsi_oversold = config.get("signal", "rsi_oversold", 30)

        # Lowering the ensemble threshold to allow more trades
        self.ensemble_threshold = 0.45
        # Increasing the proximity threshold to be less restrictive
        self.support_resistance_proximity_pct = 0.025

        self.market_phases = ["uptrend", "downtrend", "ranging_at_support", "ranging_at_resistance", "neutral"]

    def generate_signal(self, model_probs: np.ndarray, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        df.columns = [col.lower() for col in df.columns]

        if len(df) < 50:
            return {"signal_type": "NoTrade", "reason": "InsufficientData"}

        model_probs = self._validate_probabilities(model_probs)

        P_bullish = float(model_probs[3] + model_probs[4])
        P_bearish = float(model_probs[0] + model_probs[1])
        max_conf = max(P_bullish, P_bearish)

        # Process adaptive parameters if provided
        adaptive_mode = kwargs.get("adaptive_mode", False)
        win_streak = kwargs.get("win_streak", 0)
        loss_streak = kwargs.get("loss_streak", 0)

        # Adjust confidence threshold based on recent performance
        adjusted_confidence = self.confidence_threshold
        if adaptive_mode:
            if win_streak >= 3:
                # Slightly more aggressive on win streaks
                adjusted_confidence = max(0.35, self.confidence_threshold * 0.9)
            elif loss_streak >= 2:
                # More conservative on loss streaks
                adjusted_confidence = min(0.6, self.confidence_threshold * 1.15)
        else:
            adjusted_confidence = self.confidence_threshold

        if max_conf < adjusted_confidence:
            return {"signal_type": "NoTrade", "confidence": max_conf, "reason": "LowConfidence"}

        current_price = self._get_current_price(df)
        if np.isnan(current_price) or current_price <= 0:
            return {"signal_type": "NoTrade", "reason": "InvalidPrice"}

        market_regime = self._detect_market_regime(df)
        volatility_regime = self._detect_volatility_regime(df)
        trend_strength = self._calculate_trend_strength(df)
        volume_confirms = self._check_volume_confirmation(df, P_bullish > P_bearish)

        support_resistance_levels = self._identify_support_resistance(df)
        nearest_level, distance_pct = self._get_nearest_level(current_price, support_resistance_levels)

        # More restrictive filter near key levels - these tend to result in losing trades
        if P_bullish > P_bearish and distance_pct < (
                self.support_resistance_proximity_pct / 1.5) and nearest_level > current_price:
            return {
                "signal_type": "NoTrade",
                "reason": "TooCloseToResistance",
                "distance_pct": distance_pct
            }

        if P_bearish > P_bullish and distance_pct < (
                self.support_resistance_proximity_pct / 1.5) and nearest_level < current_price:
            return {
                "signal_type": "NoTrade",
                "reason": "TooCloseToSupport",
                "distance_pct": distance_pct
            }

        market_phase = self._detect_market_phase(df)

        # Be more selective in ranging_at_resistance market phase which showed negative results
        if market_phase == "ranging_at_resistance" and P_bullish > P_bearish:
            if max_conf < self.strong_signal_threshold * 1.1 or not volume_confirms:
                return {
                    "signal_type": "NoTrade",
                    "reason": "RangingAtResistanceNeedStrongerSignal",
                    "market_phase": market_phase
                }

        # Making market phase requirements less restrictive
        # Only filter on extreme phase mismatches
        if P_bullish > P_bearish and market_phase == "downtrend" and market_regime < -0.5:
            return {
                "signal_type": "NoTrade",
                "reason": "StrongDowntrendForLong",
                "market_phase": market_phase
            }

        if P_bearish > P_bullish and market_phase == "uptrend" and market_regime > 0.5:
            return {
                "signal_type": "NoTrade",
                "reason": "StrongUptrendForShort",
                "market_phase": market_phase
            }

        extra_signals = self._calculate_additional_signals(df)
        ensemble_score = self._compute_ensemble_score(
            model_prob=(P_bullish if P_bullish > P_bearish else -P_bearish),
            trend_strength=trend_strength,
            volume_confirms=volume_confirms,
            extra_signals=extra_signals
        )

        # Slightly more restrictive ensemble threshold
        if abs(ensemble_score) < self.ensemble_threshold * 0.85:  # Changed from 0.8 to 0.85
            self.logger.debug(f"Low ensemble score: {ensemble_score}, threshold: {self.ensemble_threshold}")
            # Allow trade if model probability is very strong
            if max_conf < self.strong_signal_threshold:
                return {
                    "signal_type": "NoTrade",
                    "reason": "LowEnsembleScore",
                    "ensemble_score": ensemble_score
                }

        # Add special handling for neutral market phase which showed good results
        if market_phase == "neutral":
            # Be slightly more permissive in neutral market phase
            if P_bullish > P_bearish and trend_strength < 0.15:  # Less restrictive for neutral phase
                # Allow if volume confirms or confidence is high
                if not volume_confirms and max_conf < self.strong_signal_threshold:
                    return {
                        "signal_type": "NoTrade",
                        "reason": "WeakBullishTrendInNeutralMarket",
                        "trend_strength": trend_strength
                    }
        else:
            # Standard check for non-neutral phases
            if P_bullish > P_bearish and trend_strength < 0.2:
                return {
                    "signal_type": "NoTrade",
                    "reason": "WeakBullishTrend",
                    "trend_strength": trend_strength
                }

        if P_bearish > P_bullish and trend_strength > 0.8:
            return {
                "signal_type": "NoTrade",
                "reason": "StrongBullishTrend",
                "trend_strength": trend_strength
            }

        # Require volume confirmation only for low confidence signals
        if max_conf < self.confidence_threshold * 1.2 and not volume_confirms:
            return {
                "signal_type": "NoTrade",
                "reason": "LowConfidenceNoVolumeConfirmation",
                "confidence": max_conf,
                "volume_confirmation": False
            }

        if self.use_regime_filter:
            if P_bullish > P_bearish and market_regime < -0.7:
                return {"signal_type": "NoTrade", "reason": "StrongBearishRegime", "regime": market_regime}

            if P_bearish > P_bullish and market_regime > 0.7:
                return {"signal_type": "NoTrade", "reason": "StrongBullishRegime", "regime": market_regime}

        if self.use_volatility_filter and volatility_regime > 0.8:
            if max_conf < self.strong_signal_threshold:
                return {
                    "signal_type": "NoTrade",
                    "reason": "HighVolatility",
                    "volatility": float(volatility_regime)
                }

        rsi_14 = self._get_rsi(df)
        if rsi_14 is not None:
            if P_bullish > P_bearish and rsi_14 >= self.rsi_overbought:
                return {"signal_type": "NoTrade", "reason": "OverboughtRSI", "rsi": rsi_14}

            if P_bearish > P_bullish and rsi_14 <= self.rsi_oversold:
                return {"signal_type": "NoTrade", "reason": "OversoldRSI", "rsi": rsi_14}

        macd_hist = self._get_macd_histogram(df)
        if macd_hist is not None:
            # Making MACD filter less restrictive
            if P_bullish > P_bearish and macd_hist < -0.001:
                return {"signal_type": "NoTrade", "reason": "NegativeMACD", "macd_hist": macd_hist}

            if P_bearish > P_bullish and macd_hist > 0.001:
                return {"signal_type": "NoTrade", "reason": "PositiveMACD", "macd_hist": macd_hist}

        if P_bullish > P_bearish and (rsi_14 is not None and rsi_14 > 25) and (macd_hist is not None and macd_hist < 0):
            return {"signal_type": "NoTrade", "reason": "WeakMomentumEntry",
                    "rsi": rsi_14, "macd_hist": macd_hist}

        atr_value = self._compute_atr(df)
        if np.isnan(atr_value) or atr_value <= 0:
            atr_value = current_price * 0.01

        # Check for loss streak to adjust stop loss settings
        stop_adjustment = 1.0
        if loss_streak >= 2:
            # Wider stops during losing streaks (more conservative)
            stop_adjustment = 1.15
        elif win_streak >= 3:
            # Tighter stops during winning streaks (lock in profits)
            stop_adjustment = 0.9

        if P_bullish > P_bearish:
            signal = self._create_bullish_signal(
                current_price, atr_value, P_bullish,
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

            if rsi_14 is not None:
                signal['rsi_14'] = rsi_14
            if macd_hist is not None:
                signal['macd_histogram'] = macd_hist
            return signal
        else:
            signal = self._create_bearish_signal(
                current_price, atr_value, P_bearish,
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

            if rsi_14 is not None:
                signal['rsi_14'] = rsi_14
            if macd_hist is not None:
                signal['macd_histogram'] = macd_hist
            return signal

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
                if proximity < 0.005:  # 0.5% proximity
                    is_close_to_existing = True
                    break

            if not is_close_to_existing:
                levels.append(price_level)

        return sorted(levels)

    def _find_pivot_points(self, df: pd.DataFrame, window: int = 10) -> List[Dict[str, Any]]:
        pivots = []

        for i in range(window, len(df) - window):
            # Check for pivot high
            if df['high'].iloc[i] == df['high'].iloc[i - window:i + window + 1].max():
                pivots.append({
                    'type': 'high',
                    'price': df['high'].iloc[i],
                    'index': i,
                    'strength': 1
                })

            # Check for pivot low
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
            # Calculate market structure components
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values

            # Short-term EMAs
            ema20 = self._calculate_ema(closes, 20)
            ema50 = self._calculate_ema(closes, 50)

            # Recent price action
            recent_close = closes[-1]
            recent_ema20 = ema20[-1]
            recent_ema50 = ema50[-1]

            # Recent trend direction
            ema20_slope = (ema20[-1] / ema20[-5] - 1) * 100
            ema50_slope = (ema50[-1] / ema50[-10] - 1) * 100

            # Volatility
            recent_atr = self._calculate_atr(highs[-20:], lows[-20:], closes[-20:])
            price_volatility = recent_atr / recent_close

            # Support/resistance proximity
            sr_levels = self._identify_support_resistance(df)
            nearest_level, distance = self._get_nearest_level(recent_close, sr_levels)

            # Determine market phase
            if recent_close > recent_ema20 > recent_ema50 and ema20_slope > 0.2 and ema50_slope > 0.1:
                return "uptrend"
            elif recent_close < recent_ema20 < recent_ema50 and ema20_slope < -0.2 and ema50_slope < -0.1:
                return "downtrend"
            elif abs(ema20_slope) < 0.2 and distance < 0.02:
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

        # Initialize with SMA
        ema[:period] = np.mean(values[:period])

        # Calculate EMA
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
            # Price action signals
            signals['price_action'] = self._analyze_price_action(df)

            # Volume analysis
            if 'volume' in df.columns:
                signals['volume_trend'] = self._analyze_volume_trend(df)

            # Volatility signals
            signals['volatility_signal'] = self._analyze_volatility(df)

            # Momentum signals
            signals['momentum'] = self._analyze_momentum(df)

        except Exception as e:
            self.logger.warning(f"Error calculating additional signals: {e}")

        return signals

    def _analyze_price_action(self, df: pd.DataFrame) -> float:
        try:
            closes = df['close'].values[-10:]
            opens = df['open'].values[-10:]

            # Body size relative to range
            body_sizes = np.abs(closes - opens)
            ranges = df['high'].values[-10:] - df['low'].values[-10:]
            relative_body_sizes = body_sizes / np.maximum(ranges, 0.0001)

            # Bullish candles ratio
            bullish_candles = sum(1 for i in range(len(closes)) if closes[i] > opens[i])
            bullish_ratio = bullish_candles / len(closes)

            # Recent direction
            recent_direction = 1 if closes[-1] > closes[-5] else -1

            # Combine signals
            signal = recent_direction * (bullish_ratio - 0.5) * 2 * np.mean(relative_body_sizes)

            return min(1.0, max(-1.0, signal))

        except Exception as e:
            return 0.0

    def _analyze_volume_trend(self, df: pd.DataFrame) -> float:
        volumes = df['volume'].values[-20:]
        closes = df['close'].values[-20:]

        if len(volumes) < 5:
            return 0.0

        # Volume trend
        recent_vol_avg = np.mean(volumes[-5:])
        older_vol_avg = np.mean(volumes[-20:-5])
        vol_change = recent_vol_avg / older_vol_avg if older_vol_avg > 0 else 1.0

        # Volume-price correlation
        price_changes = np.diff(closes)
        vol_changes = np.diff(volumes)

        if len(price_changes) > 1 and len(vol_changes) > 1:
            correlations = []
            for i in range(min(len(price_changes), len(vol_changes))):
                if price_changes[i] > 0 and vol_changes[i] > 0:
                    correlations.append(1)
                elif price_changes[i] < 0 and vol_changes[i] > 0:
                    correlations.append(-1)
                else:
                    correlations.append(0)

            vol_price_corr = np.mean(correlations)
        else:
            vol_price_corr = 0

        signal = vol_price_corr * min(2.0, max(0.5, vol_change)) / 2.0
        return min(1.0, max(-1.0, signal))

    def _analyze_volatility(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.0

        # Calculate volatility as normalized ATR
        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]
        closes = df['close'].values[-20:]

        atr = self._calculate_atr(highs, lows, closes)
        norm_atr = atr / closes[-1]

        # Higher volatility is typically bearish
        if norm_atr > 0.03:
            return -0.5  # High volatility - bearish
        elif norm_atr < 0.01:
            return 0.3  # Low volatility - slightly bullish
        else:
            return 0.0  # Neutral volatility

    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.0

        closes = df['close'].values

        # Calculate momentum using ROC at different timeframes
        roc1 = (closes[-1] / closes[-2] - 1) if len(closes) > 2 else 0
        roc5 = (closes[-1] / closes[-6] - 1) if len(closes) > 6 else 0
        roc10 = (closes[-1] / closes[-11] - 1) if len(closes) > 11 else 0

        # Weighted momentum score
        momentum_score = (roc1 * 0.2 + roc5 * 0.5 + roc10 * 0.3) * 10

        return min(1.0, max(-1.0, momentum_score))

    def _compute_ensemble_score(self, model_prob: float, trend_strength: float,
                                volume_confirms: bool, extra_signals: Dict[str, float]) -> float:
        base_score = model_prob

        # Adjust based on trend strength - higher weight for strong trends
        trend_factor = 0.2 + (0.6 * trend_strength)
        base_score *= trend_factor

        # Adjust with volume confirmation - volume is important
        volume_factor = 1.3 if volume_confirms else 0.8
        base_score *= volume_factor

        # Add extra signal contributions
        signal_sum = 0
        signal_count = 0

        for signal_type, signal_value in extra_signals.items():
            if signal_type == 'price_action':
                weight = 0.3
            elif signal_type == 'volume_trend':
                weight = 0.25
            elif signal_type == 'volatility_signal':
                weight = 0.15
            elif signal_type == 'momentum':
                weight = 0.3
            else:
                weight = 0.1

            signal_sum += signal_value * weight
            signal_count += weight

        # Combine model and extra signals
        if signal_count > 0:
            extra_contribution = signal_sum / signal_count

            # Ensemble formula: 70% model, 30% other signals - giving more weight to model
            ensemble_score = (base_score * 0.8) + (extra_contribution * 0.2)
        else:
            ensemble_score = base_score

        # Normalize to -1 to 1 range
        return min(1.0, max(-1.0, ensemble_score))

    def _validate_probabilities(self, model_probs: Union[np.ndarray, List[float]]) -> np.ndarray:
        if not isinstance(model_probs, np.ndarray):
            model_probs = np.array(model_probs, dtype=np.float32)

        if len(model_probs) != 5:
            self.logger.warning(f"Expected 5 probabilities, got {len(model_probs)}")
            return np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        if np.isnan(model_probs).any() or np.isinf(model_probs).any():
            model_probs = np.nan_to_num(model_probs, nan=0.0, posinf=0.0, neginf=0.0)

        total = model_probs.sum()
        if total > 0:
            model_probs = model_probs / total
        else:
            model_probs = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        return model_probs

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
        for col in ['atr_14', 'm30_atr_14']:
            if col in df.columns:
                try:
                    atr = float(df[col].iloc[-1])
                    if not np.isnan(atr) and atr > 0:
                        return atr
                except:
                    pass

        price = self._get_current_price(df)
        return price * 0.01

    def _create_bullish_signal(self, current_price, atr_value, confidence,
                               market_regime, volatility_regime, stop_adjustment=1.0):
        base_atr_mult = 3.0

        if volatility_regime > 0.7:
            vol_adjusted_mult = base_atr_mult * 1.5
        elif volatility_regime < 0.3:
            vol_adjusted_mult = base_atr_mult * 0.8
        else:
            vol_adjusted_mult = base_atr_mult * (1.0 + (volatility_regime - 0.5))

        if confidence > 0.8:
            vol_adjusted_mult *= 0.9
        elif confidence < 0.6:
            vol_adjusted_mult *= 1.2

        if market_regime < -0.3:
            vol_adjusted_mult *= 1.2

        # Apply stop adjustment parameter
        vol_adjusted_mult *= stop_adjustment

        stop_loss_price = current_price - (vol_adjusted_mult * atr_value)

        risk = current_price - stop_loss_price
        tp1_multiplier = 1.5 + (confidence * 0.5)
        tp2_multiplier = 2.5 + (confidence * 1.0)

        tp1_distance = tp1_multiplier * risk
        tp2_distance = tp2_multiplier * risk

        return {
            "signal_type": "StrongBuy" if confidence >= self.strong_signal_threshold else "Buy",
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
        base_atr_mult = 3.0

        if volatility_regime > 0.7:
            vol_adjusted_mult = base_atr_mult * 1.5
        elif volatility_regime < 0.3:
            vol_adjusted_mult = base_atr_mult * 0.8
        else:
            vol_adjusted_mult = base_atr_mult * (1.0 + (volatility_regime - 0.5))

        if confidence > 0.8:
            vol_adjusted_mult *= 0.9
        elif confidence < 0.6:
            vol_adjusted_mult *= 1.2

        if market_regime > 0.3:
            vol_adjusted_mult *= 1.2

        # Apply stop adjustment parameter
        vol_adjusted_mult *= stop_adjustment

        stop_loss_price = current_price + (vol_adjusted_mult * atr_value)

        risk = stop_loss_price - current_price
        tp_multiplier = 1.5 + (confidence * 1.0)

        take_profit = current_price - (tp_multiplier * risk)

        return {
            "signal_type": "StrongSell" if confidence >= self.strong_signal_threshold else "Sell",
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

    def _check_volume_confirmation(self, df: pd.DataFrame, is_bullish: bool) -> bool:
        try:
            if len(df) < 5:
                return False

            volumes = df['volume'].values[-5:]
            closes = df['close'].values[-5:]

            avg_volume = np.mean(volumes[:-1])
            current_volume = volumes[-1]

            if current_volume < avg_volume * 0.8:
                return False

            if is_bullish:
                price_change = closes[-1] / closes[-2] - 1
                if price_change > 0 and current_volume > avg_volume:
                    return True

            else:
                price_change = closes[-1] / closes[-2] - 1
                if price_change < 0 and current_volume > avg_volume:
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

            current_price = closes[-1]

            short_trend_up = current_price > short_term
            medium_trend_up = current_price > medium_term
            long_trend_up = current_price > long_term

            if short_trend_up == medium_trend_up == long_trend_up:
                short_pct = abs(current_price / short_term - 1)
                medium_pct = abs(current_price / medium_term - 1)
                long_pct = abs(current_price / long_term - 1)

                avg_pct = (short_pct + medium_pct + long_pct) / 3

                strength = min(1.0, avg_pct * 50)
                return 0.5 + (0.5 * strength if short_trend_up else -0.5 * strength)

            else:
                agreements = sum([1 if t == short_trend_up else 0 for t in [medium_trend_up, long_trend_up]])
                return 0.3 + (0.2 * agreements)

        except Exception as e:
            self.logger.warning(f"Error in trend strength calculation: {e}")
            return 0.5