import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import math


class PerformanceTracker:
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.performance_by_regime = {}
        self.performance_by_signal = {}
        self.performance_by_exit = {}
        self.peak_capital = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.drawdown_duration = 0
        self.max_drawdown_duration = 0
        self.current_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.trade_durations = []
        self.optimal_holding_times = {}

    def update_performance(self, trade_result):
        pnl = float(trade_result.get('pnl', 0))
        is_win = pnl > 0

        self.total_trades += 1
        if is_win:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self.total_pnl += pnl

        market_phase = trade_result.get('market_phase', 'neutral')
        exit_reason = trade_result.get('exit_signal', 'unknown')
        signal_type = trade_result.get('entry_signal', 'unknown')

        for key, collection in [
            (market_phase, self.performance_by_regime),
            (exit_reason, self.performance_by_exit),
            (signal_type, self.performance_by_signal)
        ]:
            if key not in collection:
                collection[key] = {'count': 0, 'win_count': 0, 'total_pnl': 0}

            collection[key]['count'] += 1
            collection[key]['total_pnl'] += pnl
            if is_win:
                collection[key]['win_count'] += 1

        duration = trade_result.get('duration_hours', 0)
        if duration > 0:
            self.trade_durations.append(duration)

        self._update_optimal_holding_times()

    def _update_optimal_holding_times(self):
        if len(self.trade_durations) < 10:
            return

        winning_trades = [t for t in self.trade_durations if t > 0]
        if winning_trades:
            self.optimal_holding_times['overall'] = np.mean(winning_trades)

    def get_metrics(self):
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        return {
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'profit_factor': self._calculate_profit_factor(),
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'max_win_streak': self.max_win_streak,
            'max_loss_streak': self.max_loss_streak,
            'optimal_holding_time': self.optimal_holding_times.get('overall', 12)
        }

    def _calculate_profit_factor(self):
        wins = [t for t in self.performance_by_exit.values() if t.get('total_pnl', 0) > 0]
        losses = [t for t in self.performance_by_exit.values() if t.get('total_pnl', 0) <= 0]

        total_wins = sum(t.get('total_pnl', 0) for t in wins)
        total_losses = abs(sum(t.get('total_pnl', 0) for t in losses))

        return total_wins / max(total_losses, 1e-10)


class MarketRegimeAdapter:
    def __init__(self, config):
        self.config = config
        self.volatility_bands = {
            "very_low": 0.0, "low": 0.3, "medium": 0.5,
            "high": 0.7, "very_high": 0.85
        }
        self.trend_threshold = config.get("risk", "trend_threshold", 25)
        self.regime_performance = {}
        self.regime_adjustment_frequency = config.get("risk", "regime_adjustment_frequency", 50)
        self.trade_count = 0
        self.risk_multipliers = {}
        self.current_regime = "neutral"
        self.current_volatility = "medium"

    def update_regime_parameters(self, trade_result):
        self.trade_count += 1
        regime = trade_result.get('market_phase', 'neutral')
        pnl = trade_result.get('pnl', 0)

        if regime not in self.regime_performance:
            self.regime_performance[regime] = {'count': 0, 'win_count': 0, 'total_pnl': 0}

        self.regime_performance[regime]['count'] += 1
        if pnl > 0:
            self.regime_performance[regime]['win_count'] += 1
        self.regime_performance[regime]['total_pnl'] += pnl

        if self.trade_count % self.regime_adjustment_frequency == 0:
            self._optimize_regime_multipliers()

    def _optimize_regime_multipliers(self):
        for regime, stats in self.regime_performance.items():
            if stats['count'] < 5:
                continue

            win_rate = stats['win_count'] / stats['count']
            avg_pnl = stats['total_pnl'] / stats['count']

            if win_rate > 0.6 and avg_pnl > 0:
                self.risk_multipliers[regime] = min(1.2, 1.0 + (win_rate - 0.5))
            elif win_rate < 0.4 or avg_pnl < 0:
                self.risk_multipliers[regime] = max(0.7, 1.0 - (0.5 - win_rate))


class PartialExitTracker:
    def __init__(self, config):
        self.config = config
        self.enable_partial_exits = config.get("exit", "enable_partial_exits", True)
        self.max_partial_exits = config.get("exit", "max_partial_exits", 4)

        self.partial_exit_thresholds = {
            "level1": {"threshold": 0.0075, "portion": 0.25},
            "level2": {"threshold": 0.015, "portion": 0.30},
            "level3": {"threshold": 0.0225, "portion": 0.25},
            "level4": {"threshold": 0.0375, "portion": 0.20}
        }

        self.regime_threshold_adjustments = {
            "strong_uptrend": 1.2, "uptrend": 1.1, "neutral": 1.0,
            "downtrend": 0.9, "strong_downtrend": 0.8,
            "ranging_at_support": 0.8, "ranging_at_resistance": 0.7,
            "volatile": 0.7
        }
        self.exit_performance = {}

        self.volatility_adjustment_enabled = True
        self.volatility_adjustment_factors = {
            "low": 0.9, "medium": 1.0, "high": 1.1, "extreme": 1.2
        }

    def get_partial_exit_levels(self, position, current_price, **market_conditions):
        if not self.enable_partial_exits:
            return None

        direction = position.get('direction', '')
        entry_price = float(position.get('entry_price', 0))

        if direction not in ['long', 'short'] or entry_price <= 0:
            return None

        pnl_pct = (current_price / entry_price) - 1 if direction == 'long' else (entry_price / current_price) - 1

        if pnl_pct <= 0:
            return None

        market_phase = market_conditions.get('market_phase', 'neutral')
        phase_adjustment = self.regime_threshold_adjustments.get(market_phase, 1.0)

        volatility = market_conditions.get('volatility', 0.5)
        volatility_adjustment = 1.0

        if self.volatility_adjustment_enabled:
            if volatility < 0.3:
                volatility_adjustment = self.volatility_adjustment_factors["low"]
            elif volatility < 0.6:
                volatility_adjustment = self.volatility_adjustment_factors["medium"]
            elif volatility < 0.8:
                volatility_adjustment = self.volatility_adjustment_factors["high"]
            else:
                volatility_adjustment = self.volatility_adjustment_factors["extreme"]

        for level_name, level_data in self.partial_exit_thresholds.items():
            threshold = level_data["threshold"] * phase_adjustment * volatility_adjustment

            if pnl_pct >= threshold and not position.get(f"partial_exit_{level_name}", False):
                return {
                    "threshold": threshold,
                    "portion": level_data["portion"],
                    "id": f"partial_exit_{level_name}"
                }

        return None


class DynamicExitStrategy:
    def __init__(self, config):
        self.config = config
        self.base_atr_multiplier = config.get("exit", "base_atr_multiplier", 3.0)
        self.atr_multiplier_map = {
            "strong_uptrend": {"long": 3.5, "short": 2.8}, "uptrend": {"long": 3.2, "short": 2.5},
            "neutral": {"long": 2.8, "short": 2.8}, "downtrend": {"long": 2.5, "short": 3.2},
            "strong_downtrend": {"long": 2.8, "short": 3.5}, "ranging_at_support": {"long": 2.4, "short": 3.0},
            "ranging_at_resistance": {"long": 3.0, "short": 2.4}, "volatile": {"long": 3.6, "short": 3.6}
        }
        self.profit_targets = {
            "strong_uptrend": {"long": 0.04, "short": 0.025}, "uptrend": {"long": 0.035, "short": 0.02},
            "neutral": {"long": 0.025, "short": 0.025}, "downtrend": {"long": 0.02, "short": 0.035},
            "strong_downtrend": {"long": 0.025, "short": 0.04}, "ranging_at_support": {"long": 0.02, "short": 0.015},
            "ranging_at_resistance": {"long": 0.015, "short": 0.02}, "volatile": {"long": 0.03, "short": 0.03}
        }
        self.enable_dynamic_trailing = config.get("exit", "enable_dynamic_trailing", True)
        self.trailing_activation_threshold = config.get("exit", "trailing_activation_threshold", 0.01)
        self.time_based_exits = config.get("exit", "time_based_exits", True)
        self.max_trade_duration_hours = config.get("exit", "max_trade_duration_hours", 24.0)
        self.phase_max_durations = {
            "strong_uptrend": 18.0, "uptrend": 12.0, "neutral": 24.0, "downtrend": 12.0,
            "strong_downtrend": 18.0, "ranging_at_support": 8.0, "ranging_at_resistance": 6.0, "volatile": 8.0
        }
        self.rsi_extreme_exit = config.get("exit", "rsi_extreme_exit", True)
        self.rsi_overbought = config.get("exit", "rsi_overbought", 70)
        self.rsi_oversold = config.get("exit", "rsi_oversold", 30)
        self.macd_reversal_exit = config.get("exit", "macd_reversal_exit", True)
        self.enable_early_loss_exit = config.get("exit", "enable_early_loss_exit", True)
        self.early_loss_threshold = config.get("exit", "early_loss_threshold", -0.012)
        self.early_loss_time = config.get("exit", "early_loss_time", 2.5)
        self.enable_quick_profit_exit = config.get("exit", "enable_quick_profit_exit", True)
        self.quick_profit_threshold = config.get("exit", "quick_profit_threshold", 0.006)
        self.min_holding_time = config.get("exit", "min_holding_time", 0.3)
        self.enable_stagnant_exit = config.get("exit", "enable_stagnant_exit", True)
        self.stagnant_threshold = config.get("exit", "stagnant_threshold", 0.003)
        self.stagnant_time = config.get("exit", "stagnant_time", 3.0)
        self.exit_performance = {}
        self.exit_counter = {}
        self.time_dependent_sl_adjustment = True
        self.sl_time_threshold_hours = 2.0
        self.sl_time_widening_factor = 1.25

        self.enable_trailing_take_profit = True
        self.trailing_tp_activation_ratio = 0.6
        self.trailing_tp_atr_multiplier = 1.5
        self.avg_profitable_duration = 6.0

        self.enable_volatility_tp_scaling = True
        self.volatility_tp_factors = {
            "low": 0.9, "medium": 1.0, "high": 1.2, "extreme": 1.4
        }
        self.enable_fibonacci_partial_exits = config.get("exit", "enable_fibonacci_partial_exits", True)
        self.fibonacci_partial_exit_levels_long = config.get("exit", "fibonacci_partial_exit_levels_long",
                                                             [0.618, 0.764, 1.0])
        self.fibonacci_partial_exit_levels_short = config.get("exit", "fibonacci_partial_exit_levels_short",
                                                              [0.382, 0.236, 0.0])

    def evaluate_exit(self, position, current_price, current_atr, **kwargs):
        if not self._validate_position_data(position):
            return {"exit": False, "reason": "InvalidPosition"}

        current_price = float(current_price)
        current_atr = float(current_atr)
        entry_price = float(position.get('entry_price', 0))
        initial_stop_loss = float(position.get('initial_stop_loss', 0))
        current_stop_loss = float(position.get('current_stop_loss', initial_stop_loss))
        direction = position.get('direction', '')

        if direction not in ['long', 'short']:
            return {"exit": False, "reason": "InvalidDirection"}

        trade_duration = float(kwargs.get('trade_duration', 0))
        market_phase = kwargs.get('market_phase', 'neutral')
        volatility_regime = float(kwargs.get('volatility', 0.5))
        ensemble_score = float(position.get('ensemble_score', 0.5))

        pnl_pct = (current_price - entry_price) / entry_price if direction == 'long' else (
                                                                                                      entry_price - current_price) / entry_price

        rsi_14 = float(kwargs.get('rsi_14', 50))
        macd = float(kwargs.get('macd', 0))
        macd_signal = float(kwargs.get('macd_signal', 0))
        macd_histogram = float(kwargs.get('macd_histogram', 0))
        bb_width = float(kwargs.get('bb_width', 0))
        momentum = float(kwargs.get('momentum', 0))
        trend_strength = float(kwargs.get('trend_strength', 0.5))
        volume_delta = float(kwargs.get('volume_delta', 0))

        max_duration = self.phase_max_durations.get(market_phase, self.max_trade_duration_hours)

        if pnl_pct > 0.015:
            max_duration *= 1.5
        if pnl_pct < 0 and volatility_regime > 0.7:
            max_duration *= 0.7
        if ensemble_score < 0.5:
            max_duration *= 0.8

        if trade_duration > max_duration:
            return {"exit": True, "reason": "MaxDurationExit", "exit_price": current_price}

        base_profit_target = self.profit_targets.get(market_phase, {}).get(direction, 0.025)
        base_profit_target *= 1.2

        if self.enable_volatility_tp_scaling:
            volatility_factor = 1.0
            if volatility_regime < 0.3:
                volatility_factor = self.volatility_tp_factors["low"]
            elif volatility_regime < 0.6:
                volatility_factor = self.volatility_tp_factors["medium"]
            elif volatility_regime < 0.8:
                volatility_factor = self.volatility_tp_factors["high"]
            else:
                volatility_factor = self.volatility_tp_factors["extreme"]
            profit_target = base_profit_target * volatility_factor
        else:
            profit_target = base_profit_target

        if trend_strength > 0.7:
            profit_target *= 1.2
        elif trend_strength < 0.3:
            profit_target *= 0.85

        if ensemble_score < 0.5:
            profit_target *= 0.8
        elif ensemble_score > 0.75:
            profit_target *= 1.15

        if pnl_pct >= profit_target:
            return {"exit": True, "reason": "ProfitTargetReached", "exit_price": current_price}

        if self.enable_trailing_take_profit and trade_duration >= (
                self.avg_profitable_duration * self.trailing_tp_activation_ratio):
            if pnl_pct > (profit_target * 0.45):
                trailing_tp_level = current_price - (
                            self.trailing_tp_atr_multiplier * current_atr) if direction == 'long' else current_price + (
                            self.trailing_tp_atr_multiplier * current_atr)
                if (direction == 'long' and trailing_tp_level > current_stop_loss) or (
                        direction == 'short' and trailing_tp_level < current_stop_loss):
                    return {"exit": False, "update_stop": True, "new_stop": float(trailing_tp_level),
                            "reason": "TrailingTakeProfitUpdate"}

        momentum_factors = []
        if direction == 'long':
            if momentum < -0.2: momentum_factors.append(True)
            if macd_histogram < 0 and macd < macd_signal: momentum_factors.append(True)
            if volume_delta < -1.0: momentum_factors.append(True)
        else:
            if momentum > 0.2: momentum_factors.append(True)
            if macd_histogram > 0 and macd > macd_signal: momentum_factors.append(True)
            if volume_delta > 1.0: momentum_factors.append(True)

        momentum_reversal = sum(momentum_factors) >= 2

        if momentum_reversal and pnl_pct > 0.006:
            return {"exit": True, "reason": "MomentumReversal", "exit_price": current_price}

        if bb_width > 0.05 and trade_duration > 1.0:
            vol_expansion_factors = []
            if direction == 'long':
                if macd < macd_signal: vol_expansion_factors.append(True)
                if momentum < -0.15: vol_expansion_factors.append(True)
                if rsi_14 > 70: vol_expansion_factors.append(True)
            else:
                if macd > macd_signal: vol_expansion_factors.append(True)
                if momentum > 0.15: vol_expansion_factors.append(True)
                if rsi_14 < 30: vol_expansion_factors.append(True)

            volatility_expansion = sum(vol_expansion_factors) >= 2

            if volatility_expansion and pnl_pct > 0.005:
                return {"exit": True, "reason": "VolatilityExpansionExit", "exit_price": current_price}

        if abs(volume_delta) > 2.0 and trade_duration > 0.5:
            volume_climax = (direction == 'long' and volume_delta < -1.5 and momentum < -0.1) or (
                        direction == 'short' and volume_delta > 1.5 and momentum > 0.1)

            if volume_climax and pnl_pct > 0.004:
                return {"exit": True, "reason": "VolumeClimaxExit", "exit_price": current_price}

        if self.rsi_extreme_exit:
            rsi_extreme_condition = (direction == 'long' and rsi_14 > self.rsi_overbought) or (
                        direction == 'short' and rsi_14 < self.rsi_oversold)

            if rsi_extreme_condition:
                tech_factors = []
                if direction == 'long':
                    if macd < macd_signal: tech_factors.append(True)
                    if momentum < 0: tech_factors.append(True)
                    if bb_width > 0.03: tech_factors.append(True)
                else:
                    if macd > macd_signal: tech_factors.append(True)
                    if momentum > 0: tech_factors.append(True)
                    if bb_width > 0.03: tech_factors.append(True)

                technical_confirmation = sum(tech_factors) >= 2

                if technical_confirmation and pnl_pct > 0.006:
                    exit_reason = "OverboughtExit" if direction == 'long' else "OversoldExit"
                    return {"exit": True, "reason": exit_reason, "exit_price": current_price}

        if self.macd_reversal_exit:
            macd_reversal = (direction == 'long' and macd < macd_signal and macd_histogram < 0) or (
                        direction == 'short' and macd > macd_signal and macd_histogram > 0)

            if macd_reversal and pnl_pct > 0.007 and trade_duration > 0.5:
                return {"exit": True, "reason": "MACDReversalExit", "exit_price": current_price}

        fibonacci_exit = self._evaluate_fibonacci_exit(position, current_price)
        if fibonacci_exit.get("exit", False):
            return fibonacci_exit

        stop_hit = ((direction == 'long' and current_price <= current_stop_loss) or (
                    direction == 'short' and current_price >= current_stop_loss))

        if stop_hit:
            emergency_buffer = 0.002
            if direction == 'long':
                emergency_price = current_stop_loss * (1 - emergency_buffer)
                if current_price > emergency_price:
                    return {"exit": False, "update_stop": True, "new_stop": float(emergency_price),
                            "reason": "EmergencyStopAdjustment"}
            else:
                emergency_price = current_stop_loss * (1 + emergency_buffer)
                if current_price < emergency_price:
                    return {"exit": False, "update_stop": True, "new_stop": float(emergency_price),
                            "reason": "EmergencyStopAdjustment"}

            return {"exit": True, "reason": "StopLoss", "exit_price": current_stop_loss}

        trailing_stop = self._calculate_trailing_stop(
            direction, entry_price, current_price, pnl_pct,
            current_stop_loss, current_atr, volatility_regime, market_phase,
            ensemble_score, trade_duration
        )

        if trailing_stop is not None:
            return trailing_stop

        if self.enable_early_loss_exit:
            base_early_loss = self.early_loss_threshold * 1.2
            if "ranging" in market_phase:
                base_early_loss *= 0.75
            elif "strong" in market_phase:
                base_early_loss *= 1.3
            if volatility_regime > 0.7:
                base_early_loss *= 0.9
            if trend_strength < 0.3:
                base_early_loss *= 0.85

            time_threshold = self.early_loss_time
            if "ranging" in market_phase:
                time_threshold *= 0.8
            elif "strong" in market_phase:
                time_threshold *= 1.3
            if (direction == 'long' and momentum < -0.2) or (direction == 'short' and momentum > 0.2):
                base_early_loss *= 0.85
                time_threshold *= 0.85

            if pnl_pct < base_early_loss and trade_duration > time_threshold:
                recovery_detected = (direction == 'long' and momentum > 0.1) or (
                            direction == 'short' and momentum < -0.1)
                if not recovery_detected:
                    return {"exit": True, "reason": "EarlyLossExit", "exit_price": current_price}

        if self.enable_quick_profit_exit:
            quick_profit_threshold = self.quick_profit_threshold * 0.9
            if market_phase == "ranging_at_resistance" or market_phase == "ranging_at_support":
                quick_profit_threshold = 0.0045
            elif market_phase == "uptrend" and direction == "long":
                quick_profit_threshold = 0.007
            elif market_phase == "downtrend" and direction == "short":
                quick_profit_threshold = 0.007

            if trade_duration < 1.0:
                quick_profit_threshold *= 1.25
            elif trade_duration > 4.0:
                quick_profit_threshold *= 0.7

            if ensemble_score < 0.5:
                quick_profit_threshold *= 0.8
            elif ensemble_score > 0.75:
                quick_profit_threshold *= 1.2

            if (direction == 'long' and momentum < -0.1) or (direction == 'short' and momentum > 0.1):
                quick_profit_threshold *= 0.85

            if pnl_pct > quick_profit_threshold and trade_duration > self.min_holding_time:
                return {"exit": True, "reason": "QuickProfitTaken", "exit_price": current_price}

        if self.enable_stagnant_exit:
            stagnant_threshold = self.stagnant_threshold
            if "ranging" in market_phase:
                stagnant_threshold *= 0.85
            elif "strong" in market_phase:
                stagnant_threshold *= 1.2

            min_stagnant_time = self.stagnant_time

            if "ranging" in market_phase:
                if trade_duration > min_stagnant_time * 0.8 and abs(pnl_pct) < stagnant_threshold:
                    return {"exit": True, "reason": "StagnantExit", "exit_price": current_price}
                if trade_duration > min_stagnant_time * 1.2 and abs(pnl_pct) < stagnant_threshold * 1.5:
                    return {"exit": True, "reason": "StagnantExit", "exit_price": current_price}
            else:
                if trade_duration >= min_stagnant_time * 1.5 and abs(pnl_pct) < stagnant_threshold:
                    return {"exit": True, "reason": "StagnantExit", "exit_price": current_price}
                elif trade_duration >= min_stagnant_time * 2.0 and abs(pnl_pct) < stagnant_threshold * 1.5:
                    return {"exit": True, "reason": "StagnantExit", "exit_price": current_price}

        return {"exit": False, "reason": "NoActionNeeded"}

    def update_performance_metrics(self, metrics):
        if 'avg_profitable_duration' in metrics and metrics['avg_profitable_duration'] > 0:
            self.avg_profitable_duration = metrics['avg_profitable_duration']

    def _calculate_trailing_stop(self, direction, entry_price, current_price, pnl_pct,
                                 current_stop, atr_value, volatility_regime, market_phase,
                                 ensemble_score=0.5, trade_duration=0.0):
        if not self.enable_dynamic_trailing:
            return None

        min_profit_for_adjustment = 0.015
        if pnl_pct < min_profit_for_adjustment:
            return None

        atr_multiplier = self.atr_multiplier_map.get(market_phase, {}).get(direction, 2.8)

        if volatility_regime > 0.8:
            atr_multiplier *= 1.5
        elif volatility_regime > 0.7:
            atr_multiplier *= 1.3
        elif volatility_regime > 0.5:
            atr_multiplier *= 1.1
        elif volatility_regime < 0.3:
            atr_multiplier *= 0.85

        if self.time_dependent_sl_adjustment and trade_duration > self.sl_time_threshold_hours:
            time_factor = min(1.6, 1.0 + ((trade_duration - self.sl_time_threshold_hours) / 8.0) * 0.6)
            atr_multiplier *= time_factor

        if pnl_pct > 0.045:
            atr_multiplier *= 0.4
        elif pnl_pct > 0.035:
            atr_multiplier *= 0.5
        elif pnl_pct > 0.025:
            atr_multiplier *= 0.6
        elif pnl_pct > 0.015:
            atr_multiplier *= 0.7

        if ensemble_score > 0.7:
            atr_multiplier *= 0.85
        elif ensemble_score < 0.4:
            atr_multiplier *= 1.3

        if market_phase in ["ranging_at_support", "ranging_at_resistance", "choppy"]:
            atr_multiplier *= 1.35

        profit_levels = [(0.035, 0.02), (0.025, 0.012), (0.018, 0.006), (0.012, 0.002)]

        if direction == 'long':
            new_stop = current_price - (atr_multiplier * atr_value)

            for pnl_level, profit_lock in profit_levels:
                if pnl_pct > pnl_level and (trade_duration > 2.0 or pnl_level > 0.025):
                    new_stop = max(new_stop, entry_price + (profit_lock * entry_price))

            if market_phase == "ranging_at_resistance" and pnl_pct > 0.012:
                new_stop = max(new_stop, entry_price + (0.008 * entry_price))

            if new_stop > current_stop:
                return {"exit": False, "update_stop": True, "new_stop": float(new_stop), "reason": "TrailingStopUpdate"}
        else:
            new_stop = current_price + (atr_multiplier * atr_value)

            for pnl_level, profit_lock in profit_levels:
                if pnl_pct > pnl_level and (trade_duration > 2.0 or pnl_level > 0.025):
                    new_stop = min(new_stop, entry_price - (profit_lock * entry_price))

            if market_phase == "ranging_at_support" and pnl_pct > 0.012:
                new_stop = min(new_stop, entry_price - (0.008 * entry_price))

            if new_stop < current_stop:
                return {"exit": False, "update_stop": True, "new_stop": float(new_stop), "reason": "TrailingStopUpdate"}

        return None

    def _validate_position_data(self, position):
        required_fields = ['entry_price', 'direction', 'initial_stop_loss', 'current_stop_loss']
        for field in required_fields:
            if field not in position:
                return False

        try:
            float(position['entry_price'])
            float(position['initial_stop_loss'])
            float(position['current_stop_loss'])
            return position['direction'] in ['long', 'short']
        except (ValueError, TypeError):
            return False

    def update_exit_performance(self, exit_reason, pnl):
        if exit_reason not in self.exit_performance:
            self.exit_performance[exit_reason] = {'count': 0, 'win_count': 0, 'total_pnl': 0, 'avg_pnl': 0}

        self.exit_counter[exit_reason] = self.exit_counter.get(exit_reason, 0) + 1

        perf = self.exit_performance[exit_reason]
        perf['count'] += 1
        perf['total_pnl'] += pnl

        if pnl > 0:
            perf['win_count'] += 1

        perf['avg_pnl'] = perf['total_pnl'] / perf['count']

    def _evaluate_fibonacci_exit(self, position, current_price):
        if "fibonacci_levels" not in position:
            return {"exit": False}

        fibonacci_levels = position.get("fibonacci_levels", {})
        if not fibonacci_levels:
            return {"exit": False}

        direction = position.get('direction', '')

        if self.enable_fibonacci_partial_exits:
            partial_exit = self._check_fibonacci_partial_exit(position, current_price, fibonacci_levels)
            if partial_exit.get("partial_exit", False):
                return partial_exit

        if direction == 'long':
            for level in [0.618, 0.764, 1]:
                if level in fibonacci_levels and current_price >= fibonacci_levels[level]:
                    return {"exit": True, "reason": f"FibonacciTarget_{level}", "exit_price": current_price}
        elif direction == 'short':
            for level in [0.382, 0.236, 0]:
                if level in fibonacci_levels and current_price <= fibonacci_levels[level]:
                    return {"exit": True, "reason": f"FibonacciTarget_{level}", "exit_price": current_price}

        return {"exit": False}

    def _check_fibonacci_partial_exit(self, position, current_price, fibonacci_levels):
        direction = position.get('direction', '')
        partial_exit_levels = self.fibonacci_partial_exit_levels_long if direction == 'long' else self.fibonacci_partial_exit_levels_short

        if direction not in ['long', 'short']:
            return {"exit": False}

        for i, ratio in enumerate(partial_exit_levels):
            ratio_value = float(ratio) if isinstance(ratio, str) else ratio
            if ratio_value not in fibonacci_levels:
                continue

            level = fibonacci_levels[ratio_value]
            level_id = f"fib_partial_{i + 1}"

            if position.get(f"partial_exit_{level_id}", False):
                continue

            level_reached = (direction == 'long' and current_price >= level) or (
                        direction == 'short' and current_price <= level)

            if level_reached:
                return {
                    "exit": False, "partial_exit": True, "portion": 0.3333, "id": level_id,
                    "price": current_price, "reason": f"FibonacciPartialExit_{ratio_value}",
                    "update_position_flag": f"partial_exit_{level_id}"
                }

        return {"exit": False}


class ExposureManager:
    def __init__(self, config):
        self.config = config
        self.max_portfolio_risk = config.get("risk", "max_portfolio_risk", 0.20)
        self.max_correlation_risk = config.get("risk", "max_correlation_risk", 0.12)
        self.max_single_exposure = config.get("risk", "max_single_exposure", 0.40)
        self.drawdown_exposure_curve = {
            "normal": 1.0, "caution": 0.75, "reduced": 0.5, "minimal": 0.25
        }
        self.current_exposure = 0.0
        self.exposures_by_regime = {}
        self.exposure_timestamps = []
        self.min_hours_between_trades = config.get("risk", "min_hours_between_trades", 1.0)
        self.trade_time_decay = config.get("risk", "trade_time_decay", 3.0)
        self.recent_trade_times = deque(maxlen=30)

    def check_exposure_limits(self, signal, proposed_risk, current_capital, **market_conditions):
        drawdown_state = market_conditions.get('drawdown_state', 'normal')
        market_phase = signal.get('market_phase', 'neutral')

        max_exposure_allowed = self.max_portfolio_risk * self.drawdown_exposure_curve.get(drawdown_state, 1.0)

        if self.current_exposure >= max_exposure_allowed:
            return (False, 0.0)

        if self._check_trade_density():
            return (False, 0.0)

        risk_remaining = max_exposure_allowed - self.current_exposure
        if market_phase in self.exposures_by_regime:
            phase_exposure = self.exposures_by_regime[market_phase]
            if phase_exposure >= self.max_correlation_risk:
                return (False, 0.0)

        available_risk = min(proposed_risk, risk_remaining)
        return (True, available_risk)

    def _check_trade_density(self):
        if not self.recent_trade_times:
            return False
        now = datetime.now()
        most_recent = self.recent_trade_times[-1]
        hours_since_last = (now - most_recent).total_seconds() / 3600
        return hours_since_last < self.min_hours_between_trades

    def update_exposure(self, trade_result, is_entry=True):
        if is_entry:
            self.recent_trade_times.append(datetime.now())
            risk_amount = trade_result.get('risk_amount', 0)
            market_phase = trade_result.get('market_phase', 'neutral')
            self.current_exposure += risk_amount
            self.exposures_by_regime[market_phase] = self.exposures_by_regime.get(market_phase, 0) + risk_amount
        else:
            risk_amount = trade_result.get('risk_amount', 0)
            market_phase = trade_result.get('market_phase', 'neutral')
            self.current_exposure = max(0, self.current_exposure - risk_amount)
            if market_phase in self.exposures_by_regime:
                self.exposures_by_regime[market_phase] = max(0, self.exposures_by_regime[market_phase] - risk_amount)


class AdaptivePositionSizer:
    def __init__(self, config):
        self.config = config
        self.base_risk = config.get("risk", "base_risk_per_trade", 0.015)
        self.max_risk = config.get("risk", "max_risk_per_trade", 0.025)
        self.min_risk = config.get("risk", "min_risk_per_trade", 0.008)
        self.kelly_fraction = config.get("risk", "kelly_fraction", 0.5)
        self.use_adaptive_kelly = config.get("risk", "use_adaptive_kelly", True)
        self.volatility_scaling = config.get("risk", "volatility_scaling", True)
        self.momentum_scaling = config.get("risk", "momentum_scaling", True)
        self.regime_factors = {
            "strong_uptrend": 1.3, "uptrend": 1.2, "uptrend_transition": 0.9,
            "neutral": 1.0, "downtrend": 0.8, "downtrend_transition": 0.9,
            "strong_downtrend": 0.7, "ranging": 0.6, "ranging_at_support": 0.7,
            "ranging_at_resistance": 0.5, "volatile": 0.7
        }
        self.confidence_scaling = config.get("risk", "confidence_scaling", True)
        self.streak_sensitivity = config.get("risk", "streak_sensitivity", 0.12)
        self.recent_trades = deque(maxlen=20)
        self.recent_win_rate = 0.5
        self.recent_profit_factor = 1.0
        self.equity_growth_factor = config.get("risk", "equity_growth_factor", 0.85)
        self.drawdown_risk_factor = config.get("risk", "drawdown_risk_factor", 1.5)
        self.regime_performance = {}
        self.regime_adjustment_frequency = 20

    def calculate_position_size(self, signal, entry_price, stop_loss, current_capital, **kwargs):
        direction = signal.get('direction', 'long')
        market_phase = signal.get('market_phase', 'neutral')
        volatility_regime = float(signal.get('volatility', 0.5))
        ensemble_score = float(signal.get('ensemble_score', 0.5))
        win_streak = kwargs.get('win_streak', 0)
        loss_streak = kwargs.get('loss_streak', 0)
        drawdown_state = kwargs.get('drawdown_state', 'normal')
        recovery_mode = kwargs.get('recovery_mode', False)
        key_level_proximity = signal.get('key_level_proximity', {"impact": 0, "type": "none"})

        risk_multiplier = 1.0

        if self.use_adaptive_kelly:
            kelly_risk = self._calculate_kelly_risk()
            starting_risk = kelly_risk
        else:
            starting_risk = self.base_risk

        if market_phase in self.regime_performance and self.regime_performance[market_phase]['count'] >= 5:
            perf = self.regime_performance[market_phase]
            win_rate = perf['win_count'] / perf['count'] if perf['count'] > 0 else 0.5
            avg_pnl = perf['total_pnl'] / perf['count'] if perf['count'] > 0 else 0

            if win_rate > 0.6 and avg_pnl > 0:
                perf_factor = min(1.3, 1.0 + (win_rate - 0.5) * 0.6)
            elif win_rate < 0.4 or avg_pnl < 0:
                perf_factor = max(0.5, 1.0 - (0.5 - win_rate) * 0.8)
            else:
                perf_factor = 1.0

            risk_multiplier *= perf_factor
        else:
            regime_factor = self.regime_factors.get(market_phase, 1.0)
            risk_multiplier *= regime_factor

        if "ranging" in market_phase:
            risk_multiplier *= 0.8
            if key_level_proximity["impact"] > 0.5:
                if key_level_proximity["type"] == "support" and direction == "long":
                    risk_multiplier *= 0.9
                elif key_level_proximity["type"] == "resistance" and direction == "short":
                    risk_multiplier *= 0.9

        if "transition" in market_phase:
            risk_multiplier *= 0.9

        if self.volatility_scaling:
            volatility_factor = self._get_volatility_factor(volatility_regime)
            risk_multiplier *= volatility_factor

        if self.confidence_scaling and 'confidence_score' in signal:
            confidence = float(signal.get('confidence_score', 0.5))
            confidence_factor = 0.7 + (confidence * 0.6)
            risk_multiplier *= confidence_factor

        streak_factor = 1.0
        if win_streak >= 2:
            streak_factor = 1.0 + (min(win_streak, 5) * self.streak_sensitivity)
        elif loss_streak >= 2:
            streak_factor = 1.0 - (min(loss_streak, 5) * self.streak_sensitivity)
        risk_multiplier *= streak_factor

        drawdown_factors = {"normal": 1.0, "caution": 0.8, "reduced": 0.6, "minimal": 0.4}
        risk_multiplier *= drawdown_factors.get(drawdown_state, 1.0)

        if recovery_mode:
            risk_multiplier *= 0.6

        risk_pct = starting_risk * risk_multiplier
        risk_pct = max(self.min_risk, min(self.max_risk, risk_pct))
        risk_amount = current_capital * risk_pct

        stop_distance = entry_price - stop_loss if direction == 'long' else stop_loss - entry_price
        return risk_amount / stop_distance if stop_distance > 0 else 0.0

    def _calculate_kelly_risk(self):
        if len(self.recent_trades) < 5:
            return self.base_risk

        win_count = sum(1 for trade in self.recent_trades if trade.get('pnl', 0) > 0)
        win_rate = win_count / len(self.recent_trades)

        wins = [trade for trade in self.recent_trades if trade.get('pnl', 0) > 0]
        losses = [trade for trade in self.recent_trades if trade.get('pnl', 0) <= 0]

        if not wins or not losses:
            return self.base_risk

        avg_win = sum(trade.get('pnl', 0) for trade in wins) / len(wins)
        avg_loss = abs(sum(trade.get('pnl', 0) for trade in losses)) / len(losses)

        if avg_loss <= 0:
            return self.base_risk

        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        adjusted_kelly = kelly * self.kelly_fraction

        return max(self.min_risk, min(self.max_risk, adjusted_kelly))

    def _get_volatility_factor(self, volatility):
        if volatility < 0.3:
            return 1.2
        elif volatility < 0.5:
            return 1.0
        elif volatility < 0.7:
            return 0.8
        else:
            return 0.6

    def update_trade_result(self, trade_result):
        self.recent_trades.append(trade_result)
        win_count = sum(1 for trade in self.recent_trades if trade.get('pnl', 0) > 0)
        self.recent_win_rate = win_count / len(self.recent_trades) if self.recent_trades else 0.5

        wins = [trade for trade in self.recent_trades if trade.get('pnl', 0) > 0]
        losses = [trade for trade in self.recent_trades if trade.get('pnl', 0) <= 0]

        total_profit = sum(trade.get('pnl', 0) for trade in wins)
        total_loss = abs(sum(trade.get('pnl', 0) for trade in losses))
        self.recent_profit_factor = total_profit / total_loss if total_loss > 0 else 1.0

    def update_regime_performance(self, trade_result):
        self.recent_trades.append(trade_result)
        regime = trade_result.get('market_phase', 'neutral')
        pnl = trade_result.get('pnl', 0)

        if regime not in self.regime_performance:
            self.regime_performance[regime] = {'count': 0, 'win_count': 0, 'total_pnl': 0}

        perf = self.regime_performance[regime]
        perf['count'] += 1
        perf['total_pnl'] += pnl
        if pnl > 0:
            perf['win_count'] += 1


class RiskManager:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.get("risk", "initial_capital", 10000.0)
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.base_risk_per_trade = config.get("risk", "base_risk_per_trade", 0.015)
        self.max_risk_per_trade = config.get("risk", "max_risk_per_trade", 0.025)
        self.min_risk_per_trade = config.get("risk", "min_risk_per_trade", 0.008)
        self.max_portfolio_risk = config.get("risk", "max_portfolio_risk", 0.20)
        self.position_sizer = AdaptivePositionSizer(config)
        self.exposure_manager = ExposureManager(config)
        self.exit_strategy = DynamicExitStrategy(config)
        self.trade_history = deque(maxlen=100)
        self.performance_tracker = PerformanceTracker()
        self.regime_adapter = MarketRegimeAdapter(config)
        self.current_drawdown = 0.0
        self.max_drawdown = config.get("risk", "max_drawdown_percent", 0.20)
        self.drawdown_state = "normal"
        self.max_trades_per_day = config.get("risk", "max_trades_per_day", 24)
        self.daily_trade_count = {}
        self.open_positions = []
        self.partial_exit_tracker = PartialExitTracker(config)
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.recovery_mode = False
        self.recovery_factor = config.get("risk", "recovery_factor", 0.6)
        self.min_trade_size_usd = config.get("risk", "min_trade_size_usd", 25.0)
        self.min_trade_size_btc = config.get("risk", "min_trade_size_btc", 0.0003)

        from indicator_util import IndicatorUtil
        self.indicator_util = IndicatorUtil()

    def calculate_position_size(self, signal, entry_price, stop_loss):
        if self._validate_trade_parameters(entry_price, stop_loss, signal) is False:
            return 0.0

        self._update_drawdown_state()

        direction = signal.get('direction', 'long')
        market_phase = signal.get('market_phase', 'neutral')

        can_trade, available_risk = self.exposure_manager.check_exposure_limits(
            signal, self.base_risk_per_trade, self.current_capital, drawdown_state=self.drawdown_state
        )

        if not can_trade:
            return 0.0

        position_size = self.position_sizer.calculate_position_size(
            signal, entry_price, stop_loss, self.current_capital,
            drawdown_state=self.drawdown_state, win_streak=self.current_win_streak,
            loss_streak=self.current_loss_streak, recovery_mode=self.recovery_mode
        )

        if "ranging" in market_phase:
            position_size *= 0.6

        if market_phase == "ranging" and direction == "long" and signal.get("rsi_14", 50) > 60:
            position_size *= 0.8

        if market_phase == "ranging" and direction == "short" and signal.get("rsi_14", 50) < 40:
            position_size *= 0.8

        position_value = position_size * entry_price
        if position_value < self.min_trade_size_usd or position_size < self.min_trade_size_btc:
            return 0.0

        max_position_value = self.current_capital
        if position_value > max_position_value:
            position_size = max_position_value / entry_price

        max_btc_position = 1.5
        position_size = min(position_size, max_btc_position)

        return round(float(position_size), 6)

    def handle_exit_decision(self, position, current_price, current_atr, **kwargs):
        if not self._validate_position_data(position):
            return {"exit": False, "reason": "InvalidPosition"}

        exit_decision = self.exit_strategy.evaluate_exit(
            position, current_price, current_atr, **kwargs
        )

        if not exit_decision.get('exit', False) and self.partial_exit_tracker.enable_partial_exits:
            partial_exit = self.partial_exit_tracker.get_partial_exit_levels(
                position, current_price, **kwargs
            )

            if partial_exit:
                return {
                    "exit": False, "partial_exit": True, "portion": partial_exit["portion"],
                    "id": partial_exit["id"], "price": current_price,
                    "reason": f"PartialExit_{int(partial_exit['portion'] * 100)}pct",
                }

        return exit_decision

    def update_after_trade(self, trade_result):
        pnl = float(trade_result.get('pnl', 0))
        is_win = pnl > 0

        self.current_capital = max(0, self.current_capital + pnl)
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.trade_history.append(trade_result)

        if is_win:
            self.current_win_streak += 1
            self.current_loss_streak = 0
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
        else:
            self.current_loss_streak += 1
            self.current_win_streak = 0
            self.max_loss_streak = max(self.max_loss_streak, self.current_loss_streak)

        self._update_drawdown_state()
        self._update_recovery_mode()

        self.performance_tracker.update_performance(trade_result)
        self.regime_adapter.update_regime_parameters(trade_result)
        self.position_sizer.update_regime_performance(trade_result)

        performance_metrics = self.performance_tracker.get_metrics()
        self.exit_strategy.update_performance_metrics(performance_metrics)

        trade_date = trade_result.get('exit_time', datetime.now()).strftime("%Y-%m-%d")
        self.daily_trade_count[trade_date] = self.daily_trade_count.get(trade_date, 0) + 1

        self._clean_daily_counts()

        return self.current_capital

    def check_correlation_risk(self, signal):
        trade_date = datetime.now().strftime("%Y-%m-%d")
        current_day_trades = self.daily_trade_count.get(trade_date, 0)

        if current_day_trades >= self.max_trades_per_day:
            return (False, 0.0)

        if self.drawdown_state == "minimal" and self.current_drawdown > self.max_drawdown * 0.9:
            return (False, 0.0)

        return self.exposure_manager.check_exposure_limits(
            signal, self.base_risk_per_trade, self.current_capital, drawdown_state=self.drawdown_state
        )

    def get_partial_exit_levels(self, direction, entry_price, current_price, **market_conditions):
        position = {'direction': direction, 'entry_price': entry_price}
        return self.partial_exit_tracker.get_partial_exit_levels(position, current_price, **market_conditions)

    def get_performance_metrics(self):
        return self.performance_tracker.get_metrics()

    def _validate_trade_parameters(self, entry_price, stop_loss, signal):
        if entry_price <= 0 or stop_loss <= 0:
            return False

        direction = signal.get('direction', '')
        if direction not in ['long', 'short']:
            return False

        if (direction == 'long' and stop_loss >= entry_price) or \
                (direction == 'short' and stop_loss <= entry_price):
            return False

        return True

    def _validate_position_data(self, position):
        required_fields = ['entry_price', 'direction', 'initial_stop_loss', 'current_stop_loss']
        for field in required_fields:
            if field not in position:
                return False

        try:
            float(position['entry_price'])
            float(position['initial_stop_loss'])
            float(position['current_stop_loss'])
            return position['direction'] in ['long', 'short']
        except (ValueError, TypeError):
            return False

    def _update_drawdown_state(self):
        self.current_drawdown = 1 - (self.current_capital / self.peak_capital) if self.peak_capital > 0 else 0

        if self.current_drawdown < 0.05:
            self.drawdown_state = "normal"
        elif self.current_drawdown < 0.10:
            self.drawdown_state = "caution"
        elif self.current_drawdown < 0.15:
            self.drawdown_state = "reduced"
        else:
            self.drawdown_state = "minimal"

    def _update_recovery_mode(self):
        if self.current_drawdown > 0.12 and not self.recovery_mode:
            self.recovery_mode = True
        elif self.current_drawdown < 0.08 and self.recovery_mode:
            self.recovery_mode = False

    def _clean_daily_counts(self):
        today = datetime.now().date()
        cutoff = today - timedelta(days=7)
        self.daily_trade_count = {
            d: c for d, c in self.daily_trade_count.items()
            if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff
        }