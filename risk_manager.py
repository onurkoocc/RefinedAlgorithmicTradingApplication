import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import math


class TimeFrame:
    MICRO = "micro"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    EXTENDED = "extended"


class PerformanceTracker:
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.peak_capital = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.current_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.trade_durations = []  # in hours

    def update_trade(self, pnl: float, duration_hours: float):
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.max_win_streak = max(self.max_win_streak, self.current_streak)
        else:
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.max_loss_streak = max(self.max_loss_streak, abs(self.current_streak))
        if duration_hours > 0:
            self.trade_durations.append(duration_hours)

    def update_capital(self, current_capital: float, peak_capital: float):
        self.peak_capital = peak_capital
        if self.peak_capital > 0:
            self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

    def get_metrics(self) -> Dict[str, Any]:
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        avg_duration = np.mean(self.trade_durations) if self.trade_durations else 0
        return {
            'total_trades': self.total_trades, 'win_rate': win_rate,
            'total_pnl': self.total_pnl, 'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'max_win_streak': self.max_win_streak, 'max_loss_streak': self.max_loss_streak,
            'avg_trade_duration_hours': avg_duration
        }


class AdaptivePositionSizer:
    def __init__(self, config):
        self.config = config
        self.base_risk = config.get("risk", "base_risk_per_trade", 0.01)
        self.max_risk = config.get("risk", "max_risk_per_trade", 0.02)
        self.min_risk = config.get("risk", "min_risk_per_trade", 0.005)
        self.kelly_fraction = config.get("risk", "kelly_fraction", 0.3)
        self.use_adaptive_kelly = config.get("risk", "use_adaptive_kelly", True)
        self.streak_sensitivity = config.get("risk", "streak_sensitivity", 0.10)
        self.recent_pnls = deque(maxlen=20)  # PnL percentages

    def calculate_position_size(self, signal: Dict[str, Any], entry_price: float, stop_loss_price: float,
                                current_capital: float,
                                market_regime_params: Dict[str, Any], performance_metrics: Dict[str, Any]) -> float:
        if entry_price <= 0 or stop_loss_price <= 0: return 0.0

        direction = signal.get('direction', 'long')
        stop_distance_abs = abs(entry_price - stop_loss_price)
        if stop_distance_abs == 0: return 0.0

        risk_pct = self.base_risk
        if self.use_adaptive_kelly and len(self.recent_pnls) >= 10:
            wins = [p for p in self.recent_pnls if p > 0]
            losses = [p for p in self.recent_pnls if p <= 0]
            if wins and losses:  # Ensure both wins and losses exist to calculate ratio
                win_rate = len(wins) / len(self.recent_pnls)
                avg_win_pct = np.mean(wins)
                avg_loss_pct = abs(np.mean(losses))
                if avg_loss_pct > 1e-9:  # Avoid division by zero for win_loss_ratio
                    win_loss_ratio = avg_win_pct / avg_loss_pct
                    kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio
                    risk_pct = max(self.min_risk, min(self.max_risk, kelly_pct * self.kelly_fraction))
                # If avg_loss_pct is zero (no losses or zero-pnl losses), kelly might be problematic, stick to base_risk or adjusted kelly.
                # For simplicity, if avg_loss_pct is effectively zero, kelly might be 1 or undefined.
                # A more robust Kelly might cap the win_loss_ratio or handle this edge case.
                # Here, if avg_loss_pct is 0, we'd fall back to base_risk due to the `if wins and losses` and `if avg_loss_pct > 0` checks.

        risk_multiplier = market_regime_params.get("position_sizing_factors", 1.0)

        confidence = signal.get('ensemble_score', 0.5)
        risk_multiplier *= (0.7 + 0.6 * confidence)

        volatility = signal.get('volatility', 0.5)
        if volatility < 0.3:
            risk_multiplier *= 1.1
        elif volatility > 0.7:
            risk_multiplier *= 0.9

        # Use current streak from performance_metrics, not max_win_streak/max_loss_streak
        current_win_streak = 0
        current_loss_streak = 0
        if performance_metrics.get('current_streak', 0) > 0:
            current_win_streak = performance_metrics.get('current_streak', 0)
        elif performance_metrics.get('current_streak', 0) < 0:
            current_loss_streak = abs(performance_metrics.get('current_streak', 0))

        if current_win_streak >= 2:
            risk_multiplier *= (1.0 + min(current_win_streak, 4) * self.streak_sensitivity)
        elif current_loss_streak >= 2:
            risk_multiplier *= (1.0 - min(current_loss_streak, 4) * self.streak_sensitivity)

        current_drawdown = performance_metrics.get('current_drawdown', 0.0)
        if current_drawdown > 0.15:
            risk_multiplier *= 0.7
        elif current_drawdown > 0.10:
            risk_multiplier *= 0.85

        final_risk_pct = np.clip(risk_pct * risk_multiplier, self.min_risk, self.max_risk)
        risk_amount_usd = current_capital * final_risk_pct

        position_size_asset = risk_amount_usd / stop_distance_abs

        min_trade_usd = self.config.get("risk", "min_trade_size_usd", 25.0)
        if position_size_asset * entry_price < min_trade_usd: return 0.0

        # Ensure position_size_asset is a Python float before rounding
        return round(float(position_size_asset), 6)

    def update_performance(self, pnl_pct: float):
        self.recent_pnls.append(pnl_pct)


class RiskManager:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.get("risk", "initial_capital", 10000.0)
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital

        self.performance_tracker = PerformanceTracker()
        self.position_sizer = AdaptivePositionSizer(config)

        from market_regime_util import MarketRegimeUtil  # Local import
        self.market_regime_util = MarketRegimeUtil(config)

        # Exit strategy parameters from config
        self.min_holding_time_hours = config.get("exit", "min_holding_time_hours", 0.5)
        self.max_holding_time_hours_base = config.get("exit", "max_holding_time_hours", 36.0)
        self.early_loss_threshold = config.get("exit", "early_loss_threshold", -0.015)
        self.early_loss_time_hours = config.get("exit", "early_loss_time_hours", 2.0)
        self.stagnant_threshold_pnl_abs = config.get("exit", "stagnant_threshold_pnl_abs", 0.0025)
        self.stagnant_time_hours = config.get("exit", "stagnant_time_hours", 4.0)
        self.base_profit_targets = config.get("exit", "profit_targets", {})
        self.market_phase_adjustments = config.get("exit", "market_phase_adjustments",
                                                   {})  # For profit target and duration factors
        self.partial_exit_strategy_config = config.get("exit", "partial_exit_strategy", {})
        self.time_decay_factors_config = config.get("exit", "time_decay_factors", {})
        self.rsi_extreme_levels = config.get("exit", "rsi_extreme_levels", {"overbought": 75, "oversold": 25})
        self.quick_profit_base_threshold = config.get("exit", "quick_profit_base_threshold", 0.008)
        self.reward_risk_ratio_target = config.get("exit", "reward_risk_ratio_target", 2.0)

        self.max_trades_per_day = config.get("risk", "max_trades_per_day", 5)
        self.daily_trade_count = {}
        self.min_hours_between_trades = config.get("backtest", "min_hours_between_trades", 1.0)  # From backtest section
        self.last_trade_exit_time = None
        # max_portfolio_risk is more about overall allocation if multiple assets/strategies were run.
        # For a single asset strategy, it's implicitly managed by max_risk_per_trade and drawdown limits.
        self.max_drawdown_limit = config.get("risk", "max_drawdown_percent", 0.25)

    def _get_market_regime_params(self, market_phase_name: str) -> Dict[str, Any]:
        default_params = {
            "atr_multipliers": {"long": 2.0, "short": 2.0},  # Default ATR multiplier for SL
            "profit_target_factors": 1.0,  # Factor to adjust base profit targets
            "max_duration_factors": 1.0,  # Factor to adjust max holding duration
            "position_sizing_factors": 1.0  # Factor to adjust base risk for position sizing
            # Removed signal_threshold_factors as it's more relevant to SignalGenerator
        }
        # Get the broad category of parameters for all regimes
        all_regime_params_config = self.config.get("market_regime", "regime_parameters", {})

        # Map the potentially detailed market_phase_name from signal to a simpler config key if needed
        legacy_map = self.config.get("market_regime", "legacy_regime_mapping", {})
        # First, try direct match for market_phase_name, then try mapped name, then default_regime from MRU
        mapped_phase_name = legacy_map.get(market_phase_name, market_phase_name)

        final_params = default_params.copy()

        for param_category_key, default_category_values in default_params.items():
            category_config = all_regime_params_config.get(param_category_key, {})

            # Try to get specific value for the mapped_phase_name
            specific_value = category_config.get(mapped_phase_name)

            if specific_value is not None:
                final_params[param_category_key] = specific_value
            else:
                # Fallback to a general 'default' or 'ranging' if specific not found for this category
                fallback_value = category_config.get(self.market_regime_util.default_regime, default_category_values)
                final_params[param_category_key] = fallback_value

        # Ensure atr_multipliers has both long and short
        if not isinstance(final_params.get("atr_multipliers"), dict) or \
                "long" not in final_params["atr_multipliers"] or \
                "short" not in final_params["atr_multipliers"]:
            # Fallback to a very basic default if config is malformed for atr_multipliers
            base_atr_default = {"long": 2.0, "short": 2.0}
            current_atr = final_params.get("atr_multipliers", {})
            final_params["atr_multipliers"] = {
                "long": current_atr.get("long", base_atr_default["long"]) if isinstance(current_atr, dict) else
                base_atr_default["long"],
                "short": current_atr.get("short", base_atr_default["short"]) if isinstance(current_atr, dict) else
                base_atr_default["short"]
            }
        return final_params

    def calculate_position_size(self, signal: Dict[str, Any], entry_price: float, stop_loss_price: float) -> float:
        market_phase_name = signal.get('market_phase', 'neutral')  # From SignalGenerator

        # Get blended parameters if in transition, otherwise regular regime parameters
        if signal.get("blended_parameters"):
            market_regime_params = signal["blended_parameters"]
            # Ensure all necessary keys are present in blended_parameters, or fetch from default if not
            default_single_regime_params = self._get_market_regime_params(market_phase_name)
            for key, default_val in default_single_regime_params.items():
                if key not in market_regime_params:
                    market_regime_params[key] = default_val
                # For dicts like atr_multipliers, ensure sub-keys are present
                elif isinstance(default_val, dict) and isinstance(market_regime_params[key], dict):
                    for sub_key, sub_default_val in default_val.items():
                        if sub_key not in market_regime_params[key]:
                            market_regime_params[key][sub_key] = sub_default_val

        else:
            market_regime_params = self._get_market_regime_params(market_phase_name)

        if self.performance_tracker.current_drawdown >= self.max_drawdown_limit:
            # logging.getLogger("RiskManager").info(f"Max drawdown limit reached ({self.performance_tracker.current_drawdown*100:.2f}%). No new trades.")
            return 0.0

        today_str = datetime.now().strftime("%Y-%m-%d")  # Use current time for daily limits
        if self.daily_trade_count.get(today_str, 0) >= self.max_trades_per_day:
            # logging.getLogger("RiskManager").info(f"Max trades per day ({self.max_trades_per_day}) reached.")
            return 0.0

        if self.last_trade_exit_time:
            hours_since_last_trade = (datetime.now() - self.last_trade_exit_time).total_seconds() / 3600
            if hours_since_last_trade < self.min_hours_between_trades:
                # logging.getLogger("RiskManager").info(f"Min hours between trades not met ({hours_since_last_trade:.1f} < {self.min_hours_between_trades:.1f}).")
                return 0.0

        return self.position_sizer.calculate_position_size(
            signal, entry_price, stop_loss_price, self.current_capital,
            market_regime_params, self.performance_tracker.get_metrics()  # Pass current metrics
        )

    def update_after_trade(self, trade_result: Dict[str, Any]):
        pnl_usd = float(trade_result.get('pnl', 0))
        entry_price = float(trade_result.get('entry_price_actual', 1))  # Use actual entry for PnL %
        quantity = float(trade_result.get('quantity', 1))

        # Calculate PnL percentage based on initial investment for that trade (approximate)
        # A more precise way would be to calculate risk_amount_usd for the trade.
        # For now, using entry_price * quantity as the base value for PnL %.
        trade_value_at_entry = entry_price * quantity
        pnl_pct = pnl_usd / trade_value_at_entry if trade_value_at_entry > 0 else 0

        self.current_capital += pnl_usd
        self.current_capital = max(0, self.current_capital)
        self.peak_capital = max(self.peak_capital, self.current_capital)

        duration_hours = (trade_result.get('exit_time') - trade_result.get('entry_time')).total_seconds() / 3600
        self.performance_tracker.update_trade(pnl_usd, duration_hours)  # Pass USD PnL
        self.performance_tracker.update_capital(self.current_capital, self.peak_capital)
        self.position_sizer.update_performance(pnl_pct)  # Pass PnL percentage

        trade_exit_time = trade_result.get('exit_time')
        if isinstance(trade_exit_time, datetime):
            today_str = trade_exit_time.strftime("%Y-%m-%d")
            self.daily_trade_count[today_str] = self.daily_trade_count.get(today_str, 0) + 1
            self.last_trade_exit_time = trade_exit_time

        cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        self.daily_trade_count = {d: c for d, c in self.daily_trade_count.items() if d >= cutoff_date}

    def _calculate_pnl_pct(self, direction: str, entry_price: float, current_price: float) -> float:
        if entry_price == 0: return 0.0
        if direction == 'long':
            return (current_price / entry_price) - 1
        elif direction == 'short':
            return (entry_price / current_price) - 1
        return 0.0

    def _get_timeframe_from_duration(self, duration_hours: float) -> str:
        if duration_hours < 2.0: return TimeFrame.MICRO
        if duration_hours < 6.0: return TimeFrame.SHORT
        if duration_hours < 12.0: return TimeFrame.MEDIUM
        if duration_hours < 24.0: return TimeFrame.LONG
        return TimeFrame.EXTENDED

    def _get_dynamic_profit_target(self, market_phase: str, direction: str, duration_hours: float, volatility: float,
                                   market_regime_params: Dict) -> float:
        timeframe = self._get_timeframe_from_duration(duration_hours)
        base_target_for_timeframe = self.base_profit_targets.get(timeframe, 0.02)

        # Use profit_target_factors from market_regime_params
        profit_factor = market_regime_params.get("profit_target_factors", 1.0)
        # If profit_target_factors is a dict with long/short, use that, else use the general factor
        if isinstance(profit_factor, dict):
            profit_factor = profit_factor.get(direction, 1.0)

        vol_factor = 1.0
        if volatility > 0.7:
            vol_factor = 1.15
        elif volatility < 0.3:
            vol_factor = 0.85

        return base_target_for_timeframe * profit_factor * vol_factor

    def _get_dynamic_max_duration(self, market_phase: str, direction: str, volatility: float, ensemble_score: float,
                                  market_regime_params: Dict) -> float:
        base_duration = self.max_holding_time_hours_base

        # Use max_duration_factors from market_regime_params
        duration_factor = market_regime_params.get("max_duration_factors", 1.0)
        if isinstance(duration_factor, dict):  # Should not be dict based on new config, but for safety
            duration_factor = duration_factor.get(direction, 1.0)

        adjusted_duration = base_duration * duration_factor

        if volatility > 0.7:
            adjusted_duration *= 0.8
        elif volatility < 0.3:
            adjusted_duration *= 1.2

        if ensemble_score > 0.7:
            adjusted_duration *= 1.1
        elif ensemble_score < 0.4:
            adjusted_duration *= 0.9

        return adjusted_duration

    def handle_exit_decision(self, position: Dict[str, Any], current_price: float, current_time: datetime,
                             market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        entry_time = position['entry_time']
        direction = position['direction']
        entry_price = float(position['entry_price_slipped'])  # Use slipped entry for PnL calc against current_price
        current_stop_loss = float(position['current_stop_loss'])
        initial_stop_loss = float(position.get('initial_stop_loss', current_stop_loss))
        atr = float(position.get('atr_at_entry', current_price * 0.015))  # Use ATR from entry

        trade_duration_hours = (current_time - entry_time).total_seconds() / 3600
        pnl_pct = self._calculate_pnl_pct(direction, entry_price, current_price)  # PnL based on slipped entry

        market_phase = market_conditions.get('market_phase', 'neutral')
        volatility = float(market_conditions.get('volatility', 0.5))
        ensemble_score = float(position.get('ensemble_score', 0.5))
        momentum = float(market_conditions.get('momentum', 0))
        rsi = float(market_conditions.get('rsi_14', 50))
        macd_hist = float(market_conditions.get('macd_histogram', 0))

        # Get regime parameters, potentially blended
        if market_conditions.get("blended_parameters"):
            market_regime_params = market_conditions["blended_parameters"]
            # Ensure all necessary keys are present in blended_parameters
            default_single_regime_params = self._get_market_regime_params(market_phase)
            for key, default_val in default_single_regime_params.items():
                if key not in market_regime_params:
                    market_regime_params[key] = default_val
                elif isinstance(default_val, dict) and isinstance(market_regime_params[key], dict):
                    for sub_key, sub_default_val in default_val.items():
                        if sub_key not in market_regime_params[key]: market_regime_params[key][
                            sub_key] = sub_default_val
        else:
            market_regime_params = self._get_market_regime_params(market_phase)

        if trade_duration_hours < self.min_holding_time_hours:
            if (direction == 'long' and current_price <= current_stop_loss) or \
                    (direction == 'short' and current_price >= current_stop_loss):
                return {"exit": True, "reason": "StopLoss", "exit_price": current_stop_loss}
            return {"exit": False, "reason": "MinHoldingTime"}

        if (direction == 'long' and current_price <= current_stop_loss) or \
                (direction == 'short' and current_price >= current_stop_loss):
            return {"exit": True, "reason": "StopLoss", "exit_price": current_stop_loss}

        profit_target_pct = self._get_dynamic_profit_target(market_phase, direction, trade_duration_hours, volatility,
                                                            market_regime_params)

        stop_loss_pct_from_entry = abs(
            entry_price - initial_stop_loss) / entry_price if initial_stop_loss > 0 and entry_price > 0 else float(
            'inf')
        min_sensible_profit = 0.005
        if stop_loss_pct_from_entry != float('inf') and stop_loss_pct_from_entry > 0:
            rr_based_target = stop_loss_pct_from_entry * self.reward_risk_ratio_target
            profit_target_pct = max(min_sensible_profit, min(profit_target_pct, rr_based_target))
        else:
            profit_target_pct = max(min_sensible_profit, profit_target_pct)

        if pnl_pct >= profit_target_pct:
            return {"exit": True, "reason": "ProfitTarget", "exit_price": current_price}

        max_duration_hours = self._get_dynamic_max_duration(market_phase, direction, volatility, ensemble_score,
                                                            market_regime_params)
        if trade_duration_hours > max_duration_hours:
            return {"exit": True, "reason": f"MaxDuration_{market_phase}", "exit_price": current_price}

        if pnl_pct < 0 and trade_duration_hours > self.early_loss_time_hours and pnl_pct < self.early_loss_threshold:
            if not ((direction == 'long' and momentum > 0.15) or (direction == 'short' and momentum < -0.15)):
                return {"exit": True, "reason": "EarlyLoss", "exit_price": current_price}

        if trade_duration_hours > self.stagnant_time_hours and abs(pnl_pct) < self.stagnant_threshold_pnl_abs:
            return {"exit": True, "reason": "StagnantTrade", "exit_price": current_price}

        if pnl_pct > 0.005:
            reversal_signal = False
            if direction == 'long' and momentum < -0.2 and macd_hist < 0: reversal_signal = True
            if direction == 'short' and momentum > 0.2 and macd_hist > 0: reversal_signal = True
            if reversal_signal:
                return {"exit": True, "reason": "MomentumReversal", "exit_price": current_price}

        if pnl_pct > 0.004:
            if direction == 'long' and rsi > self.rsi_extreme_levels["overbought"]: return {"exit": True,
                                                                                            "reason": "RSIExtreme",
                                                                                            "exit_price": current_price}
            if direction == 'short' and rsi < self.rsi_extreme_levels["oversold"]: return {"exit": True,
                                                                                           "reason": "RSIExtreme",
                                                                                           "exit_price": current_price}

        decay_threshold_pct = 0.0
        sorted_decay_times = sorted(self.time_decay_factors_config.keys())
        for hours_key in sorted_decay_times:
            if trade_duration_hours <= hours_key:
                decay_threshold_pct = self.time_decay_factors_config[hours_key]
                break
        if trade_duration_hours > (sorted_decay_times[-1] if sorted_decay_times else 36):
            decay_threshold_pct = self.time_decay_factors_config.get(
                sorted_decay_times[-1] if sorted_decay_times else 36, 0.02)

        if 0 < pnl_pct < decay_threshold_pct and trade_duration_hours > (
        sorted_decay_times[0] if sorted_decay_times else 4):
            return {"exit": True, "reason": "TimeDecay", "exit_price": current_price}

        if pnl_pct > self.quick_profit_base_threshold and trade_duration_hours < (
                self.stagnant_time_hours / 2) and trade_duration_hours > self.min_holding_time_hours:
            if (direction == 'long' and momentum < -0.05) or (direction == 'short' and momentum > 0.05):
                return {"exit": True, "reason": "QuickProfit", "exit_price": current_price}

        partial_exit_levels = self.partial_exit_strategy_config.get(market_phase,
                                                                    self.partial_exit_strategy_config.get("neutral",
                                                                                                          []))
        for i, level_info in enumerate(partial_exit_levels):
            level_id = f"level_{i + 1}"
            partial_exit_flag = f"partial_exit_taken_{level_id}"
            if pnl_pct >= level_info["threshold"] and not position.get(partial_exit_flag, False):
                return {
                    "exit": False, "partial_exit": True,
                    "portion": level_info["portion"], "id": f"PartialExit_{level_id}",
                    "price": current_price, "reason": f"PartialProfit_{int(level_info['portion'] * 100)}pct_L{i + 1}",
                    "update_position_flag": partial_exit_flag
                }

        if pnl_pct > 0.01:  # Start trailing only if in some profit
            atr_multiplier = market_regime_params["atr_multipliers"].get(direction, 2.0)
            if pnl_pct > 0.05:
                atr_multiplier *= 0.7
            elif pnl_pct > 0.03:
                atr_multiplier *= 0.85

            new_potential_stop = 0
            if direction == 'long':
                new_potential_stop = current_price - (atr * atr_multiplier)
                if new_potential_stop > current_stop_loss:
                    if pnl_pct > 0.015: new_potential_stop = max(new_potential_stop,
                                                                 entry_price + (entry_price * 0.001))
                    return {"exit": False, "update_stop": True, "new_stop": new_potential_stop,
                            "reason": "TrailingStopUpdate"}
            elif direction == 'short':
                new_potential_stop = current_price + (atr * atr_multiplier)
                if new_potential_stop < current_stop_loss:
                    if pnl_pct > 0.015: new_potential_stop = min(new_potential_stop,
                                                                 entry_price - (entry_price * 0.001))
                    return {"exit": False, "update_stop": True, "new_stop": new_potential_stop,
                            "reason": "TrailingStopUpdate"}

        return {"exit": False, "reason": "NoExitConditionMet"}

    def get_initial_stop_loss(self, signal: Dict[str, Any], entry_price: float, atr_at_entry: float) -> float:
        direction = signal.get('direction', 'long')
        market_phase_name = signal.get('market_phase', 'neutral')

        # Use blended ATR multipliers if present in the signal (from regime transition)
        if signal.get("blended_parameters") and "atr_multipliers" in signal["blended_parameters"]:
            atr_multipliers_for_regime = signal["blended_parameters"]["atr_multipliers"]
        else:  # Otherwise, get from standard regime parameters
            market_regime_params = self._get_market_regime_params(market_phase_name)
            atr_multipliers_for_regime = market_regime_params.get("atr_multipliers", {"long": 2.0, "short": 2.0})

        atr_multiplier = atr_multipliers_for_regime.get(direction, 2.0)

        volatility = signal.get('volatility', 0.5)
        if volatility > 0.75:
            atr_multiplier *= 1.1
        elif volatility < 0.25:
            atr_multiplier *= 0.9

        stop_distance = atr_at_entry * atr_multiplier

        if direction == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def update_parameters(self, params: Dict[str, Any]):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.position_sizer, key):
                setattr(self.position_sizer, key, value)
