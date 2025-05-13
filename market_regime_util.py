import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union


class MarketRegimeUtil:
    def __init__(self, config):
        self.config = config
        self.lookback_period = config.get("market_regime", "lookback_period", 60)
        self.regime_definitions = config.get("market_regime", "metrics_thresholds", {})
        self.default_regime = "ranging"
        self.enable_blending = config.get("market_regime", "enable_parameter_blending", True)
        self.blend_factor = config.get("market_regime", "transition_blend_factor", 0.4)
        self.last_detected_regime = None
        self.last_regime_confidence = 0.0

        from indicator_util import IndicatorUtil
        self.indicator_util = IndicatorUtil()

    def detect_regime(self, df_window: pd.DataFrame) -> Dict[str, Any]:
        if len(df_window) < self.lookback_period:
            return {"type": self.default_regime, "confidence": 0.3, "metrics": {}, "blended_parameters": None}

        current_metrics = self._calculate_metrics(df_window.iloc[-self.lookback_period:])

        detected_regime = self.default_regime
        highest_confidence = 0.3

        for regime_name, thresholds in self.regime_definitions.items():
            match_score = 0
            num_conditions = 0

            if "adx" in thresholds and "adx" in current_metrics:
                num_conditions += 1
                if current_metrics["adx"] >= thresholds["adx"]: match_score += 1
            if "adx_lt" in thresholds and "adx" in current_metrics:
                num_conditions += 1
                if current_metrics["adx"] < thresholds["adx_lt"]: match_score += 1

            if "price_above_ema_pct" in thresholds and "price_above_ema_pct" in current_metrics:
                num_conditions += 1
                if current_metrics["price_above_ema_pct"] >= thresholds["price_above_ema_pct"]: match_score += 1
            if "price_above_ema_pct_lt" in thresholds and "price_above_ema_pct" in current_metrics:
                num_conditions += 1
                if current_metrics["price_above_ema_pct"] < thresholds["price_above_ema_pct_lt"]: match_score += 1

            if "di_diff_gt" in thresholds and "di_diff" in current_metrics:
                num_conditions += 1
                if current_metrics["di_diff"] > thresholds["di_diff_gt"]: match_score += 1
            if "di_diff_lt" in thresholds and "di_diff" in current_metrics:
                num_conditions += 1
                if current_metrics["di_diff"] < thresholds["di_diff_lt"]: match_score += 1

            if "bb_width_lt" in thresholds and "bb_width" in current_metrics:
                num_conditions += 1
                if current_metrics["bb_width"] < thresholds["bb_width_lt"]: match_score += 1
            if "bb_width_gt" in thresholds and "bb_width" in current_metrics:
                num_conditions += 1
                if current_metrics["bb_width"] > thresholds["bb_width_gt"]: match_score += 1

            if "atr_pct_gt" in thresholds and "atr_pct" in current_metrics:
                num_conditions += 1
                if current_metrics["atr_pct"] > thresholds["atr_pct_gt"]: match_score += 1

            current_confidence = (match_score / num_conditions) if num_conditions > 0 else 0

            if current_confidence > highest_confidence:
                highest_confidence = current_confidence
                detected_regime = regime_name

        if num_conditions <= 2 and highest_confidence > 0.7:
            highest_confidence = 0.7
        elif num_conditions == 1 and highest_confidence > 0.5:
            highest_confidence = 0.5

        blended_params = None
        if self.enable_blending and self.last_detected_regime and self.last_detected_regime != detected_regime:
            blended_params = self._get_blended_parameters(self.last_detected_regime, detected_regime,
                                                          highest_confidence)

        self.last_detected_regime = detected_regime
        self.last_regime_confidence = highest_confidence

        return {
            "type": detected_regime,
            "confidence": round(highest_confidence, 2),
            "metrics": current_metrics,
            "blended_parameters": blended_params
        }

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        temp_iu = self.indicator_util
        df_with_indicators = temp_iu.calculate_specific_indicators(df.copy(), [
            "adx_14", "plus_di_14", "minus_di_14",
            "ema_21",
            "bb_middle_20", "bb_upper_20", "bb_lower_20",
            "atr_14"
        ])

        metrics = {}
        latest_close = df_with_indicators['close'].iloc[-1]

        metrics["adx"] = df_with_indicators[f'adx_14'].iloc[-1] if f'adx_14' in df_with_indicators.columns else 20.0

        ema21 = df_with_indicators[f'ema_21'].iloc[-1] if f'ema_21' in df_with_indicators.columns else latest_close
        metrics["price_above_ema_pct"] = np.mean(
            df_with_indicators['close'] > ema21) * 100 if ema21 is not None else 50.0

        plus_di = df_with_indicators[f'plus_di_14'].iloc[-1] if f'plus_di_14' in df_with_indicators.columns else 20.0
        minus_di = df_with_indicators[f'minus_di_14'].iloc[-1] if f'minus_di_14' in df_with_indicators.columns else 20.0
        metrics["di_diff"] = plus_di - minus_di

        bb_upper = df_with_indicators[f'bb_upper_20'].iloc[
            -1] if f'bb_upper_20' in df_with_indicators.columns else latest_close * 1.02
        bb_lower = df_with_indicators[f'bb_lower_20'].iloc[
            -1] if f'bb_lower_20' in df_with_indicators.columns else latest_close * 0.98
        bb_middle = df_with_indicators[f'bb_middle_20'].iloc[
            -1] if f'bb_middle_20' in df_with_indicators.columns else latest_close
        metrics["bb_width"] = (bb_upper - bb_lower) / (bb_middle + 1e-9) if bb_middle != 0 else 0.04

        atr = df_with_indicators[f'atr_14'].iloc[-1] if f'atr_14' in df_with_indicators.columns else latest_close * 0.01
        metrics["atr_pct"] = (atr / (latest_close + 1e-9)) if latest_close != 0 else 0.01

        return metrics

    def _get_blended_parameters(self, prev_regime: str, current_regime: str, current_confidence: float) -> Optional[
        Dict[str, Any]]:
        actual_blend_factor = self.blend_factor * (1 - current_confidence)
        if actual_blend_factor < 0.1: return None

        blended_params = {}
        param_categories = ["atr_multipliers", "profit_target_factors", "max_duration_factors",
                            "signal_threshold_factors", "position_sizing_factors"]

        prev_params_conf = self.config.get("market_regime", "regime_parameters", {}).get(prev_regime, {})
        curr_params_conf = self.config.get("market_regime", "regime_parameters", {}).get(current_regime, {})

        for category in param_categories:
            prev_cat_val = self.config.get("market_regime", "regime_parameters", {}).get(category, {}).get(prev_regime)
            curr_cat_val = self.config.get("market_regime", "regime_parameters", {}).get(category, {}).get(
                current_regime)

            default_cat_val_conf = self.config.get("market_regime", "regime_parameters", {}).get(category, {})
            default_val = None
            if "ranging" in default_cat_val_conf:
                default_val = default_cat_val_conf["ranging"]
            elif "neutral" in default_cat_val_conf:
                default_val = default_cat_val_conf["neutral"]
            elif default_cat_val_conf:
                default_val = next(iter(default_cat_val_conf.values()))

            if prev_cat_val is None: prev_cat_val = default_val
            if curr_cat_val is None: curr_cat_val = default_val

            if prev_cat_val is None or curr_cat_val is None: continue

            if isinstance(curr_cat_val, dict) and isinstance(prev_cat_val, dict):
                blended_sub_dict = {}
                for sub_key in curr_cat_val.keys():
                    if sub_key in prev_cat_val:
                        blended_sub_dict[sub_key] = curr_cat_val[sub_key] * (1 - actual_blend_factor) + \
                                                    prev_cat_val[sub_key] * actual_blend_factor
                    else:
                        blended_sub_dict[sub_key] = curr_cat_val[sub_key]
                blended_params[category] = blended_sub_dict
            elif isinstance(curr_cat_val, (float, int)) and isinstance(prev_cat_val, (float, int)):
                blended_params[category] = curr_cat_val * (1 - actual_blend_factor) + \
                                           prev_cat_val * actual_blend_factor

        return blended_params if blended_params else None

    def get_regime_parameters(self, regime_name: str) -> Dict[str, Any]:
        legacy_map = self.config.get("market_regime", "legacy_regime_mapping", {})
        mapped_name = legacy_map.get(regime_name, regime_name)

        base_params = self.config.get("market_regime", "regime_parameters", {})
        final_params = {}

        for category, cat_config in base_params.items():
            if isinstance(cat_config, dict):
                final_params[category] = cat_config.get(mapped_name, cat_config.get(self.default_regime, {}))
            else:
                final_params[category] = cat_config

        if "atr_multipliers" in final_params and isinstance(final_params["atr_multipliers"], dict):
            if "long" not in final_params["atr_multipliers"]:
                final_params["atr_multipliers"]["long"] = base_params.get("atr_multipliers", {}).get(
                    self.default_regime, {}).get("long", 2.0)
            if "short" not in final_params["atr_multipliers"]:
                final_params["atr_multipliers"]["short"] = base_params.get("atr_multipliers", {}).get(
                    self.default_regime, {}).get("short", 2.0)
        elif "atr_multipliers" not in final_params:
            final_params["atr_multipliers"] = {"long": 2.0, "short": 2.0}

        return final_params