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

        self.confidence_threshold = config.get("signal", "confidence_threshold", 0.35)
        self.strong_signal_threshold = config.get("signal", "strong_signal_threshold", 0.65)
        self.atr_multiplier_sl = config.get("signal", "atr_multiplier_sl", 2.5)
        self.use_regime_filter = config.get("signal", "use_regime_filter", True)
        self.use_volatility_filter = config.get("signal", "use_volatility_filter", True)

        self.rsi_overbought = config.get("signal", "rsi_overbought", 70)
        self.rsi_oversold = config.get("signal", "rsi_oversold", 30)

        # OPTIMIZATION: Lowered ensemble threshold to allow more trades
        self.ensemble_threshold = 0.42  # Reduced from 0.45

        # OPTIMIZATION: Market phase specific confidence thresholds
        self.phase_confidence_thresholds = {
            "neutral": 0.32,  # Reduced by 8% from base 0.35
            "uptrend": 0.35,  # Standard threshold
            "downtrend": 0.35,  # Standard threshold
            "ranging_at_support": 0.37,  # Slightly increased
            "ranging_at_resistance": 0.39  # Increased by 12% from base 0.35
        }

        # OPTIMIZATION: Increased proximity threshold to be less restrictive for entries
        self.support_resistance_proximity_pct = 0.028  # Increased from 0.025

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

        # OPTIMIZATION: Enhanced adaptive confidence threshold based on recent performance
        # More dynamic range based on win/loss streaks
        adjusted_confidence = self.confidence_threshold
        if adaptive_mode:
            if win_streak >= 3:
                # More aggressive on win streaks
                adjusted_confidence = max(0.32, self.confidence_threshold * 0.88)  # Reduced from 0.35/0.9
            elif loss_streak >= 2:
                # More conservative on loss streaks
                adjusted_confidence = min(0.62, self.confidence_threshold * 1.2)  # Increased from 0.6/1.15
        else:
            adjusted_confidence = self.confidence_threshold

        current_price = self._get_current_price(df)
        if np.isnan(current_price) or current_price <= 0:
            return {"signal_type": "NoTrade", "reason": "InvalidPrice"}

        market_regime = self._detect_market_regime(df)
        volatility_regime = self._detect_volatility_regime(df)
        trend_strength = self._calculate_trend_strength(df)
        volume_confirms = self._check_volume_confirmation(df, P_bullish > P_bearish)

        support_resistance_levels = self._identify_support_resistance(df)
        nearest_level, distance_pct = self._get_nearest_level(current_price, support_resistance_levels)

        # OPTIMIZATION: Market phase detection for phase-specific adjustments
        market_phase = self._detect_market_phase(df)

        # OPTIMIZATION: Apply phase-specific confidence threshold
        phase_threshold = self.phase_confidence_thresholds.get(market_phase, adjusted_confidence)
        if max_conf < phase_threshold:
            return {"signal_type": "NoTrade", "confidence": max_conf, "reason": "LowConfidence"}

        # OPTIMIZATION: More restrictive filter near key levels - these tend to result in losing trades
        # Especially for ranging_at_resistance phase
        sr_proximity_factor = 1.0
        if market_phase == "ranging_at_resistance":
            sr_proximity_factor = 1.8  # More restrictive for difficult phase
        elif market_phase == "neutral":
            sr_proximity_factor = 1.2  # Slightly more restrictive for neutral phase

        if P_bullish > P_bearish and distance_pct < (
                self.support_resistance_proximity_pct / sr_proximity_factor) and nearest_level > current_price:
            return {
                "signal_type": "NoTrade",
                "reason": "TooCloseToResistance",
                "distance_pct": distance_pct
            }

        if P_bearish > P_bullish and distance_pct < (
                self.support_resistance_proximity_pct / sr_proximity_factor) and nearest_level < current_price:
            return {
                "signal_type": "NoTrade",
                "reason": "TooCloseToSupport",
                "distance_pct": distance_pct
            }

        # OPTIMIZATION: Be more selective in ranging_at_resistance market phase which showed negative results
        if market_phase == "ranging_at_resistance" and P_bullish > P_bearish:
            # Higher threshold and strict volume requirement
            if max_conf < self.strong_signal_threshold * 1.15 or not volume_confirms:  # Increased from 1.1
                return {
                    "signal_type": "NoTrade",
                    "reason": "RangingAtResistanceNeedStrongerSignal",
                    "market_phase": market_phase
                }

        # OPTIMIZATION: Less restrictive market phase requirements except for extreme cases
        # Only filter on extreme phase mismatches
        if P_bullish > P_bearish and market_phase == "downtrend" and market_regime < -0.45:  # Less strict (from -0.5)
            return {
                "signal_type": "NoTrade",
                "reason": "StrongDowntrendForLong",
                "market_phase": market_phase
            }

        if P_bearish > P_bullish and market_phase == "uptrend" and market_regime > 0.45:  # Less strict (from 0.5)
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

        # OPTIMIZATION: Phase-specific ensemble threshold adjustments
        phase_ensemble_factor = 1.0
        if market_phase == "neutral":
            phase_ensemble_factor = 0.9  # Reduced threshold for neutral phase (more permissive)
        elif market_phase == "ranging_at_resistance":
            phase_ensemble_factor = 1.15  # Increased threshold for challenging phase

        # OPTIMIZATION: Slightly less restrictive ensemble threshold during win streaks
        if win_streak >= 3:
            phase_ensemble_factor *= 0.95  # More permissive during win streaks

        adjusted_ensemble_threshold = self.ensemble_threshold * phase_ensemble_factor

        # Check against adjusted ensemble threshold
        if abs(ensemble_score) < adjusted_ensemble_threshold * 0.85:  # Changed from 0.8 to 0.85
            self.logger.debug(f"Low ensemble score: {ensemble_score}, threshold: {adjusted_ensemble_threshold}")
            # Allow trade if model probability is very strong
            if max_conf < self.strong_signal_threshold:
                return {
                    "signal_type": "NoTrade",
                    "reason": "LowEnsembleScore",
                    "ensemble_score": ensemble_score
                }

        # OPTIMIZATION: Enhanced handling for neutral market phase which showed good results
        if market_phase == "neutral":
            # Be more permissive in neutral market phase (best performing phase)
            if P_bullish > P_bearish and trend_strength < 0.14:  # Less restrictive (from 0.15)
                # Allow if volume confirms or confidence is high
                if not volume_confirms and max_conf < self.strong_signal_threshold * 0.95:  # Slight reduction
                    return {
                        "signal_type": "NoTrade",
                        "reason": "WeakBullishTrendInNeutralMarket",
                        "trend_strength": trend_strength
                    }
        else:
            # Standard check for non-neutral phases
            if P_bullish > P_bearish and trend_strength < 0.2:
                # OPTIMIZATION: Enhanced trend strength requirement for uptrend phase
                if market_phase == "uptrend" and trend_strength < 0.22:  # More strict for uptrend
                    return {
                        "signal_type": "NoTrade",
                        "reason": "WeakBullishTrendInUptrend",
                        "trend_strength": trend_strength
                    }
                return {
                    "signal_type": "NoTrade",
                    "reason": "WeakBullishTrend",
                    "trend_strength": trend_strength
                }

        if P_bearish > P_bullish and trend_strength > 0.8:
            # OPTIMIZATION: Less strict for downtrend phase
            if market_phase == "downtrend" and trend_strength < 0.85:  # More permissive for downtrend
                pass  # Allow the trade to continue
            else:
                return {
                    "signal_type": "NoTrade",
                    "reason": "StrongBullishTrend",
                    "trend_strength": trend_strength
                }

        # OPTIMIZATION: Enhanced volume confirmation requirements
        # More selective based on market phase and confidence
        if market_phase == "neutral":
            # Less strict volume requirements for best performing phase
            if max_conf < self.confidence_threshold * 1.15 and not volume_confirms:
                return {
                    "signal_type": "NoTrade",
                    "reason": "LowConfidenceNoVolumeConfirmation",
                    "confidence": max_conf,
                    "volume_confirmation": False
                }
        elif market_phase == "ranging_at_resistance":
            # Stricter volume requirements for challenging phase
            if max_conf < self.strong_signal_threshold * 0.9 and not volume_confirms:
                return {
                    "signal_type": "NoTrade",
                    "reason": "NoVolumeConfirmationInRanging",
                    "confidence": max_conf,
                    "volume_confirmation": False
                }
        else:
            # Standard volume check for other phases
            if max_conf < self.confidence_threshold * 1.2 and not volume_confirms:
                return {
                    "signal_type": "NoTrade",
                    "reason": "LowConfidenceNoVolumeConfirmation",
                    "confidence": max_conf,
                    "volume_confirmation": False
                }

        if self.use_regime_filter:
            # OPTIMIZATION: Less strict regime filter
            if P_bullish > P_bearish and market_regime < -0.65:  # Reduced from -0.7
                return {"signal_type": "NoTrade", "reason": "StrongBearishRegime", "regime": market_regime}

            if P_bearish > P_bullish and market_regime > 0.65:  # Reduced from 0.7
                return {"signal_type": "NoTrade", "reason": "StrongBullishRegime", "regime": market_regime}

        if self.use_volatility_filter and volatility_regime > 0.8:
            # OPTIMIZATION: Less strict volatility requirement for confident signals
            if max_conf < self.strong_signal_threshold * 0.95:  # Reduced from 1.0
                return {
                    "signal_type": "NoTrade",
                    "reason": "HighVolatility",
                    "volatility": float(volatility_regime)
                }

        rsi_14 = self._get_rsi(df)
        if rsi_14 is not None:
            # OPTIMIZATION: Enhanced RSI filtering based on market phase
            rsi_ob_threshold = self.rsi_overbought
            rsi_os_threshold = self.rsi_oversold

            # Adjust thresholds based on market phase
            if market_phase == "neutral":
                rsi_ob_threshold = self.rsi_overbought - 3  # Less strict
                rsi_os_threshold = self.rsi_oversold + 3  # Less strict
            elif market_phase == "ranging_at_resistance":
                rsi_ob_threshold = self.rsi_overbought - 4  # Even less strict for overbought in ranging

            if P_bullish > P_bearish and rsi_14 >= rsi_ob_threshold:
                return {"signal_type": "NoTrade", "reason": "OverboughtRSI", "rsi": rsi_14}

            if P_bearish > P_bullish and rsi_14 <= rsi_os_threshold:
                return {"signal_type": "NoTrade", "reason": "OversoldRSI", "rsi": rsi_14}

        macd_hist = self._get_macd_histogram(df)
        if macd_hist is not None:
            # OPTIMIZATION: Less restrictive MACD filter
            if P_bullish > P_bearish and macd_hist < -0.0009:  # Less restrictive (from -0.001)
                return {"signal_type": "NoTrade", "reason": "NegativeMACD", "macd_hist": macd_hist}

            if P_bearish > P_bullish and macd_hist > 0.0009:  # Less restrictive (from 0.001)
                return {"signal_type": "NoTrade", "reason": "PositiveMACD", "macd_hist": macd_hist}

        # OPTIMIZATION: Enhanced momentum entry checks
        if P_bullish > P_bearish and (rsi_14 is not None and rsi_14 > 25) and (macd_hist is not None and macd_hist < 0):
            # Less restrictive in neutral phase - best performer
            if market_phase == "neutral" and macd_hist > -0.0012:  # Relaxed threshold for neutral
                pass  # Allow the trade
            else:
                return {"signal_type": "NoTrade", "reason": "WeakMomentumEntry",
                        "rsi": rsi_14, "macd_hist": macd_hist}

        atr_value = self._compute_atr(df)
        if np.isnan(atr_value) or atr_value <= 0:
            atr_value = current_price * 0.01

        # OPTIMIZATION: Enhanced check for loss streak to adjust stop loss settings
        # More dynamic adjustments based on consecutive results
        stop_adjustment = 1.0
        if loss_streak >= 2:
            # Wider stops during losing streaks (more conservative)
            stop_adjustment = 1.18  # Increased from 1.15
        elif win_streak >= 3:
            # Tighter stops during winning streaks (lock in profits)
            stop_adjustment = 0.88  # Reduced from 0.9

        # OPTIMIZATION: Further adjust stop based on market phase
        if market_phase == "neutral":
            stop_adjustment *= 0.95  # Tighter stops in best performing phase
        elif market_phase == "ranging_at_resistance":
            stop_adjustment *= 0.85  # Even tighter stops in worst performing phase

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

            # OPTIMIZATION: Enhanced market phase detection with more nuanced conditions
            # Determine market phase with refined thresholds
            if recent_close > recent_ema20 > recent_ema50 and ema20_slope > 0.18 and ema50_slope > 0.08:  # Reduced from 0.2/0.1
                return "uptrend"
            elif recent_close < recent_ema20 < recent_ema50 and ema20_slope < -0.18 and ema50_slope < -0.08:  # Reduced from -0.2/-0.1
                return "downtrend"
            elif abs(ema20_slope) < 0.18 and distance < 0.022:  # Relaxed from 0.2/0.02
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

            # OPTIMIZATION: Enhanced momentum analysis with market phase context
            signals['momentum'] = self._analyze_momentum(df)

            # OPTIMIZATION: Add time-based cycle analysis
            signals['time_cycle'] = self._analyze_time_cycle(df)

        except Exception as e:
            self.logger.warning(f"Error calculating additional signals: {e}")

        return signals

    def _analyze_time_cycle(self, df: pd.DataFrame) -> float:
        """Analyze time-based cycles and patterns"""
        try:
            if len(df) < 100:
                return 0.0

            # Get index as datetime if available
            if isinstance(df.index, pd.DatetimeIndex):
                # Extract hour of day (0-23)
                hour_of_day = df.index[-1].hour

                # Simple bias based on hour (this should be customized based on backtest data)
                # Assuming higher values = more bullish
                if 8 <= hour_of_day <= 12:  # Morning session
                    return 0.2  # Slightly bullish
                elif 12 < hour_of_day <= 16:  # Afternoon session
                    return 0.1  # Neutral to slightly bullish
                elif 20 <= hour_of_day <= 23:  # Evening session
                    return -0.1  # Slightly bearish
                else:
                    return 0.0  # Neutral

            return 0.0

        except Exception as e:
            return 0.0

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

            # OPTIMIZATION: Enhanced price action analysis
            # Doji detection (small bodies relative to range)
            doji_threshold = 0.15
            doji_count = sum(1 for rb in relative_body_sizes if rb < doji_threshold)
            doji_factor = 1.0 - (doji_count / len(relative_body_sizes) * 0.5)  # Reduce signal strength based on dojis

            # Recent momentum strength
            recent_momentum = abs(closes[-1] / closes[-3] - 1) * 10
            momentum_factor = min(1.5, max(0.5, 1.0 + recent_momentum))

            # Combine signals with optimized weighting
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

        # Volume trend
        recent_vol_avg = np.mean(volumes[-5:])
        older_vol_avg = np.mean(volumes[-20:-5])
        vol_change = recent_vol_avg / older_vol_avg if older_vol_avg > 0 else 1.0

        # OPTIMIZATION: Enhanced volume-price correlation analysis
        # Volume-price correlation with more weight on recent bars
        price_changes = np.diff(closes)
        vol_changes = np.diff(volumes)

        if len(price_changes) > 1 and len(vol_changes) > 1:
            # Apply recency weighting
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

        # OPTIMIZATION: Enhanced volume spike detection
        recent_vol_max = np.max(volumes[-5:])
        avg_vol = np.mean(volumes[-20:])
        vol_spike_factor = 1.0
        if recent_vol_max > avg_vol * 1.5:  # Volume spike detected
            vol_spike_factor = 1.3  # Increase signal strength

        signal = vol_price_corr * min(2.0, max(0.5, vol_change)) / 2.0 * vol_spike_factor
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

        # OPTIMIZATION: More nuanced volatility analysis
        # Volatility trend (increasing or decreasing)
        short_atr = self._calculate_atr(highs[-10:], lows[-10:], closes[-10:])
        medium_atr = self._calculate_atr(highs[-15:], lows[-15:], closes[-15:])

        vol_trend = 0.0
        if short_atr > medium_atr:
            vol_trend = -0.2  # Increasing volatility: slightly bearish
        else:
            vol_trend = 0.1  # Decreasing volatility: slightly bullish

        # Combine current volatility level with trend
        if norm_atr > 0.03:
            return -0.4 + vol_trend  # High volatility - bearish with trend adjustment
        elif norm_atr < 0.01:
            return 0.3 + vol_trend  # Low volatility - slightly bullish with trend adjustment
        else:
            return vol_trend  # Neutral volatility - use trend signal

    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.0

        closes = df['close'].values

        # OPTIMIZATION: Enhanced momentum analysis with time decay weighting
        # Calculate momentum using ROC at different timeframes with time-based weighting
        roc1 = (closes[-1] / closes[-2] - 1) if len(closes) > 2 else 0
        roc5 = (closes[-1] / closes[-6] - 1) if len(closes) > 6 else 0
        roc10 = (closes[-1] / closes[-11] - 1) if len(closes) > 11 else 0
        roc20 = (closes[-1] / closes[-21] - 1) if len(closes) > 21 else 0

        # Market phase adaptive weighting
        market_phase = self._detect_market_phase(df)

        # Default weights
        weights = {
            'roc1': 0.2,
            'roc5': 0.5,
            'roc10': 0.3,
            'roc20': 0.0
        }

        # Adjust weights based on market phase
        if market_phase == "neutral":
            # In neutral phase, prefer short to medium term momentum
            weights = {
                'roc1': 0.25,
                'roc5': 0.55,
                'roc10': 0.2,
                'roc20': 0.0
            }
        elif market_phase == "uptrend" or market_phase == "downtrend":
            # In trending markets, include longer-term momentum
            weights = {
                'roc1': 0.15,
                'roc5': 0.35,
                'roc10': 0.3,
                'roc20': 0.2
            }
        elif market_phase == "ranging_at_resistance" or market_phase == "ranging_at_support":
            # In ranging markets, focus more on short-term momentum
            weights = {
                'roc1': 0.35,
                'roc5': 0.5,
                'roc10': 0.15,
                'roc20': 0.0
            }

        # Calculate weighted momentum score
        momentum_score = (
                                 roc1 * weights['roc1'] +
                                 roc5 * weights['roc5'] +
                                 roc10 * weights['roc10'] +
                                 roc20 * weights['roc20']
                         ) * 12  # Increased multiplier from 10

        return min(1.0, max(-1.0, momentum_score))

    def _compute_ensemble_score(self, model_prob: float, trend_strength: float,
                                volume_confirms: bool, extra_signals: Dict[str, float]) -> float:
        # OPTIMIZATION: Enhanced ensemble score calculation with better weighting
        base_score = model_prob

        # OPTIMIZATION: Higher weight for trend strength in ensemble score
        trend_factor = 0.25 + (0.65 * trend_strength)  # Increased from 0.2/0.6
        base_score *= trend_factor

        # OPTIMIZATION: Enhanced volume confirmation impact
        volume_factor = 1.35 if volume_confirms else 0.75  # Increased difference (from 1.3/0.8)
        base_score *= volume_factor

        # OPTIMIZATION: Enhanced signal contributions with optimized weights
        signal_sum = 0
        signal_count = 0

        for signal_type, signal_value in extra_signals.items():
            # Updated weights based on signal importance
            if signal_type == 'price_action':
                weight = 0.35  # Increased from 0.3
            elif signal_type == 'volume_trend':
                weight = 0.3  # Increased from 0.25
            elif signal_type == 'volatility_signal':
                weight = 0.15  # Unchanged
            elif signal_type == 'momentum':
                weight = 0.35  # Increased from 0.3
            elif signal_type == 'time_cycle':
                weight = 0.15  # New signal with moderate weight
            else:
                weight = 0.1

            signal_sum += signal_value * weight
            signal_count += weight

        # Combine model and extra signals
        if signal_count > 0:
            extra_contribution = signal_sum / signal_count

            # OPTIMIZATION: Increased weight for model (from 70/30% to 75/25%)
            ensemble_score = (base_score * 0.75) + (extra_contribution * 0.25)
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
                    atr_val = float(df[col].iloc[-1])
                    if not np.isnan(atr_val) and atr_val > 0:
                        return atr_val
                except:
                    pass

        price = self._get_current_price(df)
        return price * 0.01

    def _create_bullish_signal(self, current_price, atr_value, confidence,
                               market_regime, volatility_regime, stop_adjustment=1.0):
        # OPTIMIZATION: Improved stop loss calculation with market-adaptive ATR multiplier
        base_atr_mult = 3.0

        if volatility_regime > 0.7:
            vol_adjusted_mult = base_atr_mult * 1.45  # Reduced from 1.5 for tighter stops
        elif volatility_regime < 0.3:
            vol_adjusted_mult = base_atr_mult * 0.75  # Reduced from 0.8 for tighter stops
        else:
            vol_adjusted_mult = base_atr_mult * (1.0 + (volatility_regime - 0.5) * 0.9)  # Reduced scale

        # OPTIMIZATION: Enhanced confidence-based stop adjustment
        if confidence > 0.8:
            vol_adjusted_mult *= 0.85  # Tighter stops for high confidence (reduced from 0.9)
        elif confidence < 0.6:
            vol_adjusted_mult *= 1.25  # Wider stops for low confidence (increased from 1.2)

        # OPTIMIZATION: Enhanced market regime adjustment
        if market_regime < -0.25:  # Less strict (from -0.3)
            vol_adjusted_mult *= 1.15  # Wider stops when going against trend (reduced from 1.2)

        # Apply stop adjustment parameter from adaptive parameters
        vol_adjusted_mult *= stop_adjustment

        stop_loss_price = current_price - (vol_adjusted_mult * atr_value)

        risk = current_price - stop_loss_price

        # OPTIMIZATION: Enhanced reward/risk ratio based on confidence
        tp1_multiplier = 1.6 + (confidence * 0.6)  # Increased from 1.5/0.5
        tp2_multiplier = 2.6 + (confidence * 1.2)  # Increased from 2.5/1.0

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
        # OPTIMIZATION: Improved stop loss calculation with market-adaptive ATR multiplier
        base_atr_mult = 3.0

        if volatility_regime > 0.7:
            vol_adjusted_mult = base_atr_mult * 1.45  # Reduced from 1.5 for tighter stops
        elif volatility_regime < 0.3:
            vol_adjusted_mult = base_atr_mult * 0.75  # Reduced from 0.8 for tighter stops
        else:
            vol_adjusted_mult = base_atr_mult * (1.0 + (volatility_regime - 0.5) * 0.9)  # Reduced scale

        # OPTIMIZATION: Enhanced confidence-based stop adjustment
        if confidence > 0.8:
            vol_adjusted_mult *= 0.85  # Tighter stops for high confidence (reduced from 0.9)
        elif confidence < 0.6:
            vol_adjusted_mult *= 1.25  # Wider stops for low confidence (increased from 1.2)

        # OPTIMIZATION: Enhanced market regime adjustment
        if market_regime > 0.25:  # Less strict (from 0.3)
            vol_adjusted_mult *= 1.15  # Wider stops when going against trend (reduced from 1.2)

        # Apply stop adjustment parameter from adaptive parameters
        vol_adjusted_mult *= stop_adjustment

        stop_loss_price = current_price + (vol_adjusted_mult * atr_value)

        risk = stop_loss_price - current_price
        # OPTIMIZATION: Enhanced reward/risk ratio based on confidence
        tp_multiplier = 1.6 + (confidence * 1.2)  # Increased from 1.5/1.0

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

            # OPTIMIZATION: Enhanced volume confirmation with multiple timeframes
            # Check shorter timeframe (last 3 bars)
            short_avg_volume = np.mean(volumes[-4:-1])  # Last 3 bars excluding current
            current_volume = volumes[-1]

            # Check medium timeframe (5-10 bars)
            if len(df) >= 10:
                medium_avg_volume = np.mean(df['volume'].values[-10:-5])
                medium_volume_increasing = current_volume > medium_avg_volume * 0.9
            else:
                medium_volume_increasing = True

            # OPTIMIZATION: Less restrictive volume requirement
            volume_sufficient = current_volume > short_avg_volume * 0.75  # Reduced from 0.8

            # Direction confirmation with price
            if is_bullish:
                price_change = closes[-1] / closes[-2] - 1
                if price_change > 0 and (volume_sufficient or medium_volume_increasing):
                    return True
            else:
                price_change = closes[-1] / closes[-2] - 1
                if price_change < 0 and (volume_sufficient or medium_volume_increasing):
                    return True

            # OPTIMIZATION: Check for volume divergence (price up on declining volume)
            if is_bullish and price_change > 0 and len(volumes) >= 3:
                vol_trend = volumes[-1] / np.mean(volumes[-3:-1])
                # Allow confirmation even with slightly declining volume if price change is strong
                if price_change > 0.005 and vol_trend > 0.7:  # Relaxed from strictly increasing
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

            # OPTIMIZATION: Enhanced trend strength with more timeframes
            short_term = np.mean(closes[-5:])
            medium_term = np.mean(closes[-10:])
            long_term = np.mean(closes[-20:])

            # Add very short term for better responsiveness
            very_short_term = np.mean(closes[-3:])

            current_price = closes[-1]

            # Check trend agreement across timeframes
            very_short_trend_up = current_price > very_short_term
            short_trend_up = current_price > short_term
            medium_trend_up = current_price > medium_term
            long_trend_up = current_price > long_term

            # OPTIMIZATION: Enhanced trend strength calculation with weighted timeframes
            # Calculate percentage deviations at each timeframe
            vs_pct = abs(current_price / very_short_term - 1) * 1.2  # Higher weight
            short_pct = abs(current_price / short_term - 1)
            medium_pct = abs(current_price / medium_term - 1) * 0.8
            long_pct = abs(current_price / long_term - 1) * 0.6  # Lower weight

            # Check for agreement across all timeframes
            if very_short_trend_up == short_trend_up == medium_trend_up == long_trend_up:
                # All timeframes agree - strong trend
                weighted_pct = (vs_pct * 1.2 + short_pct * 1.0 + medium_pct * 0.8 + long_pct * 0.6) / 3.6

                strength = min(1.0, weighted_pct * 60)  # Increased from 50
                return 0.5 + (0.5 * strength if short_trend_up else -0.5 * strength)
            else:
                # Count agreements with very short term trend
                agreements = sum([1 if t == very_short_trend_up else 0
                                  for t in [short_trend_up, medium_trend_up, long_trend_up]])
                return 0.3 + (0.225 * agreements)  # Slight increase from 0.2

        except Exception as e:
            self.logger.warning(f"Error in trend strength calculation: {e}")
            return 0.5