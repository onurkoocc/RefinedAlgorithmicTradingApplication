import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque


class SignalConfidenceScorer:
    def __init__(self, config):
        self.config = config
        self.weights = {
            "model_pred": 0.50,
            "technical": 0.20,
            "regime": 0.15,
            "volume": 0.10,
            "volatility": 0.05
        }

    def score_signal(self, signal_components: Dict[str, Any]) -> float:
        score = 0.0

        pred_strength = abs(signal_components.get("predicted_return", 0.0))
        score += self.weights["model_pred"] * pred_strength

        tech_score = 0.0
        direction = np.sign(signal_components.get("predicted_return", 0.0))
        if direction == 0: direction = 1

        if np.sign(signal_components.get("ema_signal", 0)) == direction: tech_score += 0.5
        if np.sign(signal_components.get("macd_signal", 0)) == direction: tech_score += 0.5
        tech_score = np.clip(tech_score, 0, 1)
        score += self.weights["technical"] * tech_score

        market_phase = signal_components.get("market_phase", "neutral")
        regime_score = 0.5
        if (direction > 0 and "uptrend" in market_phase) or \
                (direction < 0 and "downtrend" in market_phase):
            regime_score = 1.0
        elif (direction > 0 and "downtrend" in market_phase) or \
                (direction < 0 and "uptrend" in market_phase):
            regime_score = 0.0
        elif "ranging" in market_phase or "volatile" in market_phase:
            regime_score = 0.3 if pred_strength > 0.5 else 0.6
        score += self.weights["regime"] * regime_score

        if signal_components.get("volume_confirms", False):
            score += self.weights["volume"] * 1.0
        else:
            score += self.weights["volume"] * 0.3

        volatility = signal_components.get("volatility", 0.5)
        if volatility > 0.8:
            score += self.weights["volatility"] * 0.2
        elif volatility < 0.2:
            score += self.weights["volatility"] * 0.6
        else:
            score += self.weights["volatility"] * 0.8

        return np.clip(score, 0.0, 1.0)


class SignalGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("SignalGenerator")

        from indicator_util import IndicatorUtil
        from market_regime_util import MarketRegimeUtil

        self.indicator_util = IndicatorUtil()
        self.market_regime_detector = MarketRegimeUtil(config)
        self.indicator_util.market_regime_util = self.market_regime_detector

        self.signal_scorer = SignalConfidenceScorer(config)

        self.base_buy_threshold = config.get("signal", "buy_threshold", 0.0010)
        self.base_strong_buy_threshold = config.get("signal", "strong_buy_threshold", 0.0018)

    def generate_signal(self, model_pred_scaled: float, df_current_features: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if df_current_features is None or len(df_current_features) < self.market_regime_detector.lookback_period:
            return {"signal_type": "NoTrade", "reason": "InsufficientDataForSignal"}

        market_regime_info = self.market_regime_detector.detect_regime(df_current_features)
        market_phase = market_regime_info["type"]
        regime_confidence = market_regime_info["confidence"]
        regime_params = self.market_regime_detector.get_regime_parameters(market_phase)

        if market_regime_info.get("blended_parameters"):
            blended = market_regime_info["blended_parameters"]
            for key, value in blended.items():
                if key in regime_params and isinstance(regime_params[key], dict) and isinstance(value, dict):
                    regime_params[key].update(value)
                else:
                    regime_params[key] = value

        latest_features = df_current_features.iloc[-1]
        signal_components = {
            "predicted_return": model_pred_scaled,
            "market_phase": market_phase,
            "regime_confidence": regime_confidence,
            "volatility": self.indicator_util.detect_volatility_regime(df_current_features),
            "ema_signal": self._check_ma_signal(latest_features),
            "macd_signal": self._check_macd_signal(latest_features),
            "volume_confirms": self.indicator_util.check_volume_confirmation(df_current_features,
                                                                             "long" if model_pred_scaled > 0 else "short"),
            "atr_at_entry": latest_features.get(f'atr_{self.config.get("feature_engineering", "atr_period", 14)}',
                                                latest_features.get('close', 1) * 0.01),
            "current_price": latest_features.get('close', 0.0)
        }
        signal_components.update(regime_params)
        signal_components.update(market_regime_info.get("metrics", {}))

        confidence_score = self.signal_scorer.score_signal(signal_components)
        signal_components["ensemble_score"] = confidence_score

        threshold_factor = regime_params.get("signal_threshold_factors", 1.0)

        buy_thresh = self.base_buy_threshold * threshold_factor
        strong_buy_thresh = self.base_strong_buy_threshold * threshold_factor

        final_signal_type = "NoTrade"
        reason = "BelowThreshold"
        direction = "long" if model_pred_scaled > 0 else "short"

        abs_model_pred = abs(model_pred_scaled)

        if abs_model_pred > strong_buy_thresh * 0.8 and confidence_score > 0.65:
            final_signal_type = "StrongBuy" if direction == "long" else "StrongSell"
        elif abs_model_pred > buy_thresh * 0.8 and confidence_score > 0.50:
            final_signal_type = "Buy" if direction == "long" else "Sell"
        elif abs_model_pred > buy_thresh * 0.5 and confidence_score > 0.40:
            final_signal_type = "NoTrade"
            reason = "WeakSignalLowConfidence"
        else:
            reason = "LowPredictionOrConfidence"

        if (direction == "long" and market_phase == "downtrend" and regime_confidence > 0.6) or \
                (direction == "short" and market_phase == "uptrend" and regime_confidence > 0.6):
            if "Strong" in final_signal_type:
                final_signal_type = final_signal_type.replace("Strong", "")
            elif final_signal_type != "NoTrade":
                final_signal_type = "NoTrade"
                reason = "RegimeMisalignment"

        output_signal = {
            "signal_type": final_signal_type,
            "reason": reason if final_signal_type == "NoTrade" else "SignalThresholdMet",
            "direction": direction,
            "confidence": confidence_score,
            "ensemble_score": confidence_score,
            "predicted_return_scaled": model_pred_scaled,
            "market_phase": market_phase,
            "volatility": signal_components["volatility"],
            "atr_at_entry": signal_components["atr_at_entry"],
            "current_price": signal_components["current_price"]
        }
        output_signal.update(regime_params)
        if market_regime_info.get("blended_parameters"):
            output_signal["blended_parameters"] = market_regime_info["blended_parameters"]

        return output_signal

    def _check_ma_signal(self, latest_features: pd.Series) -> int:
        ema_short_period = self.config.get("feature_engineering", "ema_short_period", 9)
        ema_medium_period = self.config.get("feature_engineering", "ema_medium_period", 21)

        ema_short = latest_features.get(f'ema_{ema_short_period}', np.nan)
        ema_medium = latest_features.get(f'ema_{ema_medium_period}', np.nan)

        if pd.isna(ema_short) or pd.isna(ema_medium): return 0
        if ema_short > ema_medium: return 1
        if ema_short < ema_medium: return -1
        return 0

    def _check_macd_signal(self, latest_features: pd.Series) -> int:
        mf = self.config.get("feature_engineering", "macd_fast", 12)
        ms = self.config.get("feature_engineering", "macd_slow", 26)
        msig = self.config.get("feature_engineering", "macd_signal", 9)

        macd_val = latest_features.get(f'macd_{mf}_{ms}', np.nan)
        signal_line = latest_features.get(f'macd_signal_{mf}_{ms}_{msig}', np.nan)

        if pd.isna(macd_val) or pd.isna(signal_line): return 0
        if macd_val > signal_line: return 1
        if macd_val < signal_line: return -1
        return 0

    def update_parameters(self, params: Dict[str, Any]):
        if "buy_threshold" in params:
            self.base_buy_threshold = params["buy_threshold"]
        if "strong_buy_threshold" in params:
            self.base_strong_buy_threshold = params["strong_buy_threshold"]
        if "signal_scorer_weights" in params and isinstance(params["signal_scorer_weights"], dict):
            self.signal_scorer.weights.update(params["signal_scorer_weights"])