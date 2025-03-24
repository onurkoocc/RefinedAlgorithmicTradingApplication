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


class ExitType:
    PROFIT_TARGET = "profit_target"
    TRAILING_STOP = "trailing_stop"
    MOMENTUM_REVERSAL = "momentum_reversal"
    RSI_EXTREME = "rsi_extreme"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    TIME_DECAY = "time_decay"
    STOP_LOSS = "stop_loss"
    PARTIAL_EXIT = "partial_exit"


class AdaptiveTimeManager:
    def __init__(self, config):
        self.config = config

        self.min_holding_time = config.get("time_management", "min_holding_time", 0.5)
        self.max_holding_time = config.get("time_management", "max_holding_time", 36.0)

        self.profit_targets = {
            TimeFrame.MICRO: config.get("time_management", "micro_profit", 0.004),
            TimeFrame.SHORT: config.get("time_management", "short_profit", 0.009),
            TimeFrame.MEDIUM: config.get("time_management", "medium_profit", 0.018),
            TimeFrame.LONG: config.get("time_management", "long_profit", 0.035),
            TimeFrame.EXTENDED: config.get("time_management", "extended_profit", 0.05)
        }
        # Add these new attributes for early loss and stagnant trade detection
        self.early_loss_threshold = config.get("exit", "early_loss_threshold", -0.012)
        self.early_loss_time = config.get("exit", "early_loss_time", 2.5)
        self.stagnant_threshold = config.get("exit", "stagnant_threshold", 0.003)
        self.stagnant_time = config.get("exit", "stagnant_time", 3.0)
        self.market_phase_adjustments = {
            "uptrend": {"long": 1.2, "short": 0.7},
            "downtrend": {"long": 0.7, "short": 1.2},
            "neutral": {"long": 1.0, "short": 1.0},
            "ranging_at_support": {"long": 1.1, "short": 0.7},
            "ranging_at_resistance": {"long": 0.7, "short": 1.1},
            "volatile": {"long": 0.8, "short": 0.8}
        }

        self.atr_multiplier_map = {
            "uptrend": {"long": 2.8, "short": 2.2},
            "downtrend": {"long": 2.2, "short": 2.8},
            "neutral": {"long": 2.5, "short": 2.5},
            "ranging_at_support": {"long": 2.0, "short": 3.0},
            "ranging_at_resistance": {"long": 3.0, "short": 2.0}
        }

        self.partial_exit_strategy = {
            "uptrend": [
                {"threshold": 0.012, "portion": 0.2},
                {"threshold": 0.022, "portion": 0.3},
                {"threshold": 0.035, "portion": 0.3}
            ],
            "downtrend": [
                {"threshold": 0.01, "portion": 0.25},
                {"threshold": 0.018, "portion": 0.3},
                {"threshold": 0.03, "portion": 0.25}
            ],
            "neutral": [
                {"threshold": 0.009, "portion": 0.2},
                {"threshold": 0.016, "portion": 0.25},
                {"threshold": 0.028, "portion": 0.3}
            ]
        }

        self.exit_stats = {}
        self.optimal_durations = {}
        self.trade_history = deque(maxlen=500)
        self.performance_by_duration = {}
        self.performance_by_phase = {}
        self.time_decay_factors = {
            4: 1.0,
            8: 0.95,
            16: 0.9,
            24: 0.85,
            36: 0.7
        }

        self.volatility_amplifiers = {
            "low": 1.2,
            "medium": 1.0,
            "high": 0.8,
            "extreme": 0.6
        }

        from indicator_util import IndicatorUtil
        self.indicator_util = IndicatorUtil()

    def evaluate_time_based_exit(self, position, current_price, current_time, market_conditions):
        return self.evaluate_time_exit(position, current_price, current_time, market_conditions)

    def evaluate_time_exit(self, position, current_price, current_time, market_conditions):
        if not self._validate_position(position, current_price, current_time):
            return {"exit": False, "reason": "InvalidPosition"}

        entry_time = position['entry_time']
        direction = position['direction']
        entry_price = float(position['entry_price'])
        current_stop = float(position['current_stop_loss'])

        trade_duration = self._calculate_duration(entry_time, current_time)
        pnl_pct = self._calculate_pnl(direction, entry_price, current_price)
        market_phase = market_conditions.get('market_phase', 'neutral')
        volatility = float(market_conditions.get('volatility', 0.5))
        ensemble_score = float(position.get('ensemble_score', 0.5))
        momentum = float(market_conditions.get('momentum', 0))

        # Get volume delta from market conditions
        volume_delta = float(market_conditions.get('volume_delta', 0))

        # Add volume_delta to exit factors analysis
        exit_decision = self._analyze_exit_factors(
            position, current_price, current_time, market_conditions,
            trade_duration, pnl_pct, market_phase, volatility, ensemble_score, momentum
        )

        if not exit_decision.get("exit", False):
            partial_exit = self._check_partial_exit(
                direction, entry_price, current_price, market_phase,
                trade_duration, volatility, momentum
            )

            if partial_exit:
                return partial_exit

            trailing_stop = self._update_trailing_stop(
                position, current_price, market_phase,
                volatility, pnl_pct
            )

            if trailing_stop:
                return trailing_stop

        return exit_decision

    def _analyze_exit_factors(self, position, current_price, current_time, market_conditions,
                              trade_duration, pnl_pct, market_phase, volatility, ensemble_score, momentum):
        direction = position['direction']
        entry_price = float(position['entry_price'])
        current_stop = float(position['current_stop_loss'])

        # Add additional market data from market_conditions
        macd_histogram = float(market_conditions.get('macd_histogram', 0))
        volume_delta = float(market_conditions.get('volume_delta', 0))
        bb_width = float(market_conditions.get('bb_width', 0))
        price_action = market_conditions.get('price_action', {})

        max_duration = self._get_max_duration(market_phase, direction, volatility, ensemble_score)
        if trade_duration > max_duration:
            return {
                "exit": True,
                "reason": f"MaxDuration_{market_phase}",
                "exit_price": current_price
            }

        profit_target = self._get_adjusted_profit_target(market_phase, direction, trade_duration, volatility)
        if pnl_pct >= profit_target:
            return {
                "exit": True,
                "reason": "ProfitTarget",
                "exit_price": current_price
            }

        if self._check_momentum_reversal(direction, momentum, pnl_pct, trade_duration, volume_delta, macd_histogram):
            return {
                "exit": True,
                "reason": "MomentumReversal",
                "exit_price": current_price
            }

        rsi = float(market_conditions.get('rsi_14', 50))
        if self._check_rsi_extreme_exit(direction, rsi, pnl_pct, trade_duration):
            return {
                "exit": True,
                "reason": "RSIExtreme",
                "exit_price": current_price
            }

        # New: Check for volatility expansion exit
        if bb_width > 0.05 and abs(pnl_pct) > 0.005:
            if (direction == 'long' and momentum < -0.1) or (direction == 'short' and momentum > 0.1):
                return {
                    "exit": True,
                    "reason": "VolatilityExpansionExit",
                    "exit_price": current_price
                }

        # New: Check for key level breach
        if price_action and 'key_level_breach' in price_action:
            level_breach = price_action['key_level_breach']
            if level_breach.get('breached', False):
                level_type = level_breach.get('type', '')
                if (direction == 'long' and level_type == 'resistance' and pnl_pct > 0.008) or \
                        (direction == 'short' and level_type == 'support' and pnl_pct > 0.008):
                    return {
                        "exit": True,
                        "reason": "KeyLevelBreachExit",
                        "exit_price": current_price
                    }

        if self._check_quick_profit_exit(pnl_pct, trade_duration, volatility, ensemble_score, market_phase):
            return {
                "exit": True,
                "reason": "QuickProfit",
                "exit_price": current_price
            }

        if self._check_stagnant_trade(pnl_pct, trade_duration, market_phase):
            return {
                "exit": True,
                "reason": "StagnantTrade",
                "exit_price": current_price
            }

        if ((direction == 'long' and current_price <= current_stop) or
                (direction == 'short' and current_price >= current_stop)):
            return {
                "exit": True,
                "reason": "StopLoss",
                "exit_price": current_stop
            }

        # Enhanced early loss detection - get values from config rather than attributes
        early_loss_threshold = self.config.get("exit", "early_loss_threshold", -0.012)
        early_loss_time = self.config.get("exit", "early_loss_time", 2.5)

        if pnl_pct < 0:
            # Scale threshold based on market conditions
            base_early_loss = early_loss_threshold

            # More sensitive early loss detection in ranging markets
            if "ranging" in market_phase:
                base_early_loss *= 0.7

            # Consider volatility for early loss threshold
            if volatility > 0.7:
                base_early_loss *= 0.85  # Less sensitive in high volatility

            # Consider trend strength from market conditions
            trend_strength = float(market_conditions.get('trend_strength', 0.5))
            if trend_strength < 0.3:  # Weak trend
                base_early_loss *= 0.8  # More sensitive when trend is weak

            # Adaptive time threshold - exit earlier in ranging markets
            time_threshold = early_loss_time
            if "ranging" in market_phase:
                time_threshold *= 0.7

            # Factor in momentum direction - exit faster on adverse momentum
            if (direction == 'long' and momentum < -0.2) or (direction == 'short' and momentum > 0.2):
                base_early_loss *= 0.8
                time_threshold *= 0.8

            # Check for early loss exit condition
            if pnl_pct < base_early_loss and trade_duration > time_threshold:
                return {
                    "exit": True,
                    "reason": "EarlyLossExit",
                    "exit_price": current_price
                }

        if self._check_time_decay_exit(pnl_pct, trade_duration, market_phase):
            return {
                "exit": True,
                "reason": "TimeDecay",
                "exit_price": current_price
            }

        return {"exit": False, "reason": "NoExit"}

    def _get_max_duration(self, market_phase, direction, volatility, ensemble_score):
        base_max_duration = {
            "uptrend": {"long": 24.0, "short": 12.0},
            "downtrend": {"long": 12.0, "short": 24.0},
            "neutral": {"long": 18.0, "short": 18.0},
            "ranging_at_support": {"long": 12.0, "short": 8.0},
            "ranging_at_resistance": {"long": 8.0, "short": 12.0},
            "volatile": {"long": 10.0, "short": 10.0}
        }

        phase_duration = base_max_duration.get(market_phase, {}).get(direction, 18.0)

        if volatility > 0.7:
            phase_duration *= 0.8
        elif volatility < 0.3:
            phase_duration *= 1.2

        if ensemble_score > 0.7:
            phase_duration *= 1.2
        elif ensemble_score < 0.5:
            phase_duration *= 0.85

        return min(phase_duration, self.max_holding_time)

    def _get_adjusted_profit_target(self, market_phase, direction, trade_duration, volatility):
        base_targets = self.profit_targets.copy()

        phase_adjustment = self.market_phase_adjustments.get(market_phase, {"long": 1.0, "short": 1.0})
        direction_factor = phase_adjustment.get(direction, 1.0)

        volatility_factor = 1.0
        if volatility > 0.7:
            volatility_factor = 0.85
        elif volatility < 0.3:
            volatility_factor = 1.15

        timeframe = self._get_timeframe_from_duration(trade_duration)

        base_target = base_targets.get(timeframe, 0.01)

        return base_target * direction_factor * volatility_factor

    def _get_timeframe_from_duration(self, duration):
        if duration < 2.0:
            return TimeFrame.MICRO
        elif duration < 6.0:
            return TimeFrame.SHORT
        elif duration < 12.0:
            return TimeFrame.MEDIUM
        elif duration < 24.0:
            return TimeFrame.LONG
        else:
            return TimeFrame.EXTENDED

    def _check_momentum_reversal(self, direction, momentum, pnl_pct, trade_duration, volume_delta=0, macd_histogram=0):
        if pnl_pct <= 0.005 or trade_duration < 1.0:
            return False

        # Check both momentum and MACD histogram for stronger confirmation
        if direction == 'long':
            # Combined signal using momentum, MACD histogram and volume (if available)
            momentum_signal = momentum < -0.2
            macd_signal = macd_histogram < 0
            volume_signal = volume_delta < -1.0 if abs(volume_delta) > 0 else False

            # Need at least 2 confirming signals (reduced dependency on any single indicator)
            signals = [momentum_signal, macd_signal, volume_signal]
            confirming_signals = sum(1 for signal in signals if signal)

            return confirming_signals >= 2
        else:  # short
            momentum_signal = momentum > 0.2
            macd_signal = macd_histogram > 0
            volume_signal = volume_delta > 1.0 if abs(volume_delta) > 0 else False

            signals = [momentum_signal, macd_signal, volume_signal]
            confirming_signals = sum(1 for signal in signals if signal)

            return confirming_signals >= 2

        return False

    def _check_rsi_extreme_exit(self, direction, rsi, pnl_pct, trade_duration):
        if pnl_pct <= 0.004 or trade_duration < 0.5:
            return False

        if direction == 'long' and rsi > 75:
            return True

        if direction == 'short' and rsi < 25:
            return True

        return False

    def _check_quick_profit_exit(self, pnl_pct, trade_duration, volatility, ensemble_score, market_phase):
        phase_thresholds = {
            "uptrend": {"long": 0.008, "short": 0.006},
            "downtrend": {"long": 0.006, "short": 0.008},
            "neutral": {"long": 0.007, "short": 0.007},
            "ranging_at_support": {"long": 0.006, "short": 0.005},
            "ranging_at_resistance": {"long": 0.005, "short": 0.006},
            "volatile": {"long": 0.01, "short": 0.01}
        }

        direction = "long" if pnl_pct > 0 else "short"
        threshold = phase_thresholds.get(market_phase, {}).get(direction, 0.007)

        if volatility > 0.7:
            threshold *= 0.85

        if ensemble_score < 0.5:
            threshold *= 0.9

        if trade_duration < 1.0:
            threshold *= 1.2
        elif trade_duration > 6.0:
            threshold *= 0.8

        return pnl_pct > threshold and trade_duration > self.min_holding_time

    def _check_stagnant_trade(self, pnl_pct, trade_duration, market_phase):
        # Get stagnant threshold from config
        base_threshold = self.config.get("exit", "stagnant_threshold", 0.003)

        # More aggressive thresholds for ranging markets
        phase_thresholds = {
            "ranging": base_threshold * 0.83,  # Increased sensitivity in ranging markets
            "ranging_at_support": base_threshold * 0.66,
            "ranging_at_resistance": base_threshold * 0.66,
            "uptrend": base_threshold,
            "downtrend": base_threshold,
            "neutral": base_threshold * 0.83,
            "volatile": base_threshold * 1.33
        }

        threshold = phase_thresholds.get(market_phase, base_threshold)

        # More dynamic time thresholds - exit earlier in ranging markets
        if "ranging" in market_phase:
            if trade_duration > 3.0 and abs(pnl_pct) < threshold:
                return True
            if trade_duration > 6.0 and abs(pnl_pct) < threshold * 1.5:
                return True
            if trade_duration > 9.0 and abs(pnl_pct) < threshold * 2:
                return True
        else:
            # Standard checks for other market phases
            if trade_duration < 4.0:
                return False
            if trade_duration > 8.0 and abs(pnl_pct) < threshold:
                return True
            if trade_duration > 14.0 and abs(pnl_pct) < threshold * 1.5:
                return True
            if trade_duration > 20.0 and abs(pnl_pct) < threshold * 2:
                return True

        return False

    def _check_time_decay_exit(self, pnl_pct, trade_duration, market_phase):
        decay_thresholds = {4: 0.0, 8: 0.003, 16: 0.007, 24: 0.012, 36: 0.018}

        decay_threshold = 0.0
        for hours, threshold in sorted(decay_thresholds.items()):
            if trade_duration <= hours:
                decay_threshold = threshold
                break

        if trade_duration > 36:
            decay_threshold = 0.02

        phase_adjustments = {
            "uptrend": {"long": 0.8, "short": 1.2},
            "downtrend": {"long": 1.2, "short": 0.8},
            "ranging_at_support": {"long": 0.9, "short": 1.1},
            "ranging_at_resistance": {"long": 1.1, "short": 0.9}
        }

        direction = "long" if pnl_pct > 0 else "short"
        phase_adj = phase_adjustments.get(market_phase, {}).get(direction, 1.0)
        decay_threshold *= phase_adj

        if 0 < pnl_pct < decay_threshold and trade_duration > 8.0:
            return True

        return False

    def _check_partial_exit(self, direction, entry_price, current_price, market_phase,
                            trade_duration, volatility, momentum):
        if direction not in ['long', 'short']:
            return None

        if direction == 'long':
            pnl_pct = (current_price / entry_price) - 1
        else:
            pnl_pct = (entry_price / current_price) - 1

        if pnl_pct <= 0:
            return None

        exit_levels = self.partial_exit_strategy.get(market_phase,
                                                     self.partial_exit_strategy["neutral"])

        volatility_adj = 1.0
        if volatility > 0.7:
            volatility_adj = 0.9
        elif volatility < 0.3:
            volatility_adj = 1.1

        for i, level in enumerate(exit_levels):
            threshold = level["threshold"] * volatility_adj
            portion = level["portion"]

            if (direction == 'long' and momentum < -0.1) or (direction == 'short' and momentum > 0.1):
                threshold *= 0.9

            if pnl_pct >= threshold:
                level_id = f"level_{i + 1}"
                return {
                    "exit": False,
                    "partial_exit": True,
                    "portion": portion,
                    "id": f"PartialExit_{level_id}",
                    "price": current_price,
                    "reason": f"PartialExit_{int(portion * 100)}pct",
                    "update_position_flag": f"partial_exit_{level_id}"
                }

        return None

    def _update_trailing_stop(self, position, current_price, market_phase, volatility, pnl_pct):
        direction = position['direction']
        entry_price = float(position['entry_price'])
        current_stop = float(position['current_stop_loss'])
        atr = float(position.get('atr', current_price * 0.01))

        # Don't trail stops until minimum profit threshold
        min_profit_for_adjustment = 0.008  # Increased from previous value
        if pnl_pct < min_profit_for_adjustment:
            return None

        # Get base ATR multiplier for the market phase and direction
        atr_multiplier = self.atr_multiplier_map.get(market_phase, {}).get(direction, 2.5)

        # For ranging markets, use wider trailing stops
        if "ranging" in market_phase:
            atr_multiplier *= 1.5

        # Implement smarter trailing logic based on trend_strength
        trend_strength = float(position.get('trend_strength', 0.5))
        if trend_strength < 0.3:  # Weak trend
            atr_multiplier *= 1.3  # Wider trailing

        # Calculate new stop loss with sufficient breathing room
        if direction == 'long':
            new_stop = current_price - (atr_multiplier * atr)

            # Add progressive profit lock logic - move to breakeven faster in ranging markets
            if market_phase == "ranging" and pnl_pct > 0.012:
                new_stop = max(new_stop, entry_price + (entry_price * 0.001))

            if new_stop > current_stop:
                return {
                    "exit": False,
                    "update_stop": True,
                    "new_stop": float(new_stop),
                    "reason": "TrailingStopUpdate"
                }
        else:
            new_stop = current_price + (atr_multiplier * atr)

            # Add progressive profit lock logic
            if market_phase == "ranging" and pnl_pct > 0.012:
                new_stop = min(new_stop, entry_price - (entry_price * 0.001))

            if new_stop < current_stop:
                return {
                    "exit": False,
                    "update_stop": True,
                    "new_stop": float(new_stop),
                    "reason": "TrailingStopUpdate"
                }

        return None

    def update_duration_stats(self, trade_result):
        self.update_trade_stats(trade_result)

    def update_trade_stats(self, trade_result):
        if not self._validate_trade_result(trade_result):
            return

        exit_reason = trade_result.get('exit_signal', 'Unknown')
        pnl = float(trade_result.get('pnl', 0))
        duration_hours = trade_result.get('duration_hours', 0)
        market_phase = trade_result.get('market_phase', 'neutral')

        self.trade_history.append(trade_result)

        if exit_reason not in self.exit_stats:
            self.exit_stats[exit_reason] = {
                'count': 0, 'win_count': 0, 'total_pnl': 0,
                'durations': [], 'win_rate': 0, 'avg_pnl': 0
            }

        stats = self.exit_stats[exit_reason]
        stats['count'] += 1
        stats['total_pnl'] += pnl

        if pnl > 0:
            stats['win_count'] += 1

        if duration_hours > 0:
            stats['durations'].append(duration_hours)

        stats['win_rate'] = stats['win_count'] / stats['count'] if stats['count'] > 0 else 0
        stats['avg_pnl'] = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0

        duration_bucket = self._get_duration_bucket(duration_hours)
        if duration_bucket not in self.performance_by_duration:
            self.performance_by_duration[duration_bucket] = {
                'count': 0, 'win_count': 0, 'total_pnl': 0
            }

        dur_stats = self.performance_by_duration[duration_bucket]
        dur_stats['count'] += 1
        dur_stats['total_pnl'] += pnl

        if pnl > 0:
            dur_stats['win_count'] += 1

        if market_phase not in self.performance_by_phase:
            self.performance_by_phase[market_phase] = {
                'count': 0, 'win_count': 0, 'total_pnl': 0,
                'durations': [], 'optimal_duration': 12.0
            }

        phase_stats = self.performance_by_phase[market_phase]
        phase_stats['count'] += 1
        phase_stats['total_pnl'] += pnl

        if pnl > 0:
            phase_stats['win_count'] += 1
            phase_stats['durations'].append(duration_hours)

        self._update_optimal_durations()

    def _get_duration_bucket(self, hours):
        if hours < 1:
            return "0-1h"
        elif hours < 2:
            return "1-2h"
        elif hours < 4:
            return "2-4h"
        elif hours < 8:
            return "4-8h"
        elif hours < 16:
            return "8-16h"
        else:
            return "16h+"

    def _update_optimal_durations(self):
        if len(self.trade_history) < 20:
            return

        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        if not winning_trades:
            return

        win_durations = [t.get('duration_hours', 0) for t in winning_trades]
        self.optimal_durations['overall'] = np.mean(win_durations)

        for phase, stats in self.performance_by_phase.items():
            if not stats.get('durations'):
                continue

            if len(stats['durations']) >= 5:
                stats['optimal_duration'] = np.mean(stats['durations'])
                self.optimal_durations[phase] = stats['optimal_duration']

    def optimize_time_parameters(self):
        return self.optimize_exit_parameters()

    def optimize_exit_parameters(self):
        if len(self.trade_history) < 50:
            return False

        exit_perf = {}
        duration_perf = {}

        for reason, stats in self.exit_stats.items():
            if stats['count'] < 5:
                continue

            score = (stats['win_rate'] * 0.4) + (stats['avg_pnl'] * 100 * 0.6)
            exit_perf[reason] = score

        for bucket, stats in self.performance_by_duration.items():
            if stats['count'] < 5:
                continue

            win_rate = stats['win_count'] / stats['count'] if stats['count'] > 0 else 0
            avg_pnl = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
            score = (win_rate * 0.4) + (avg_pnl * 100 * 0.6)
            duration_perf[bucket] = score

        if len(exit_perf) >= 3:
            best_exits = sorted(exit_perf.items(), key=lambda x: x[1], reverse=True)[:3]

            for exit_type, score in best_exits:
                if exit_type == "ProfitTarget" and score > 1.0:
                    self._boost_profit_targets(1.1)
                elif exit_type == "QuickProfit" and score > 1.0:
                    self._boost_profit_targets(0.95)

        if len(duration_perf) >= 3:
            best_durations = sorted(duration_perf.items(), key=lambda x: x[1], reverse=True)[:2]
            worst_durations = sorted(duration_perf.items(), key=lambda x: x[1])[:2]

            self._adjust_max_durations(best_durations, worst_durations)

        return True

    def _boost_profit_targets(self, factor):
        for timeframe in self.profit_targets:
            self.profit_targets[timeframe] *= factor

    def _adjust_max_durations(self, best_durations, worst_durations):
        best_hours = self._hours_from_buckets(best_durations)
        worst_hours = self._hours_from_buckets(worst_durations)

        for phase in self.market_phase_adjustments:
            for direction in ['long', 'short']:
                if any(h in best_hours for h in [4, 8, 16]):
                    self.market_phase_adjustments[phase][direction] *= 1.05

                if any(h in worst_hours for h in [16, 24, 36]):
                    self.market_phase_adjustments[phase][direction] *= 0.95

    def _hours_from_buckets(self, duration_scores):
        hours = []
        for bucket, _ in duration_scores:
            if bucket == "0-1h":
                hours.append(1)
            elif bucket == "1-2h":
                hours.append(2)
            elif bucket == "2-4h":
                hours.append(4)
            elif bucket == "4-8h":
                hours.append(8)
            elif bucket == "8-16h":
                hours.append(16)
            else:
                hours.append(24)
        return hours

    def get_optimal_holding_time(self, market_phase=None, direction=None):
        if not market_phase:
            return self.optimal_durations.get('overall', 12.0)

        phase_specific = self.optimal_durations.get(market_phase)
        if phase_specific:
            return phase_specific

        return self.optimal_durations.get('overall', 12.0)

    def get_exit_performance_stats(self):
        return self.get_exit_performance()

    def get_exit_performance(self):
        return {
            'exit_stats': self.exit_stats,
            'optimal_durations': self.optimal_durations,
            'performance_by_duration': self.performance_by_duration,
            'performance_by_phase': self.performance_by_phase
        }

    def calculate_optimal_trade_duration(self, trade_history=None):
        trades = trade_history or self.trade_history

        if not trades or len(trades) < 10:
            return {
                "optimal_hold_time": 12,
                "confidence": "low",
                "data_points": len(trades) if trades else 0
            }

        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        if not winning_trades:
            return {
                "optimal_hold_time": 12,
                "confidence": "low",
                "data_points": 0
            }

        trade_durations = [t.get('duration_hours', 0) for t in winning_trades]
        optimal_duration = min(36, max(6, np.mean(trade_durations) * 1.2))

        phases = {}
        for trade in winning_trades:
            phase = trade.get('market_phase', 'neutral')
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(trade.get('duration_hours', 0))

        phase_durations = {}
        for phase, durations in phases.items():
            if len(durations) >= 5:
                phase_durations[phase] = np.mean(durations)

        confidence = "medium"
        if len(winning_trades) > 30:
            confidence = "high"
        elif len(winning_trades) < 10:
            confidence = "low"

        return {
            "optimal_hold_time": optimal_duration,
            "avg_trade_duration": np.mean([t.get('duration_hours', 0) for t in trades]),
            "avg_profitable_duration": np.mean(trade_durations),
            "confidence": confidence,
            "data_points": len(winning_trades),
            "phase_optimal_durations": phase_durations
        }

    def _validate_position(self, position, current_price, current_time):
        required_fields = ['entry_time', 'direction', 'entry_price', 'current_stop_loss']

        for field in required_fields:
            if field not in position:
                return False

        if position['direction'] not in ['long', 'short']:
            return False

        try:
            float(position['entry_price'])
            float(position['current_stop_loss'])
            float(current_price)
        except (ValueError, TypeError):
            return False

        entry_time = position.get('entry_time')
        if not isinstance(entry_time, datetime) or current_time <= entry_time:
            return False

        return True

    def _validate_trade_result(self, trade_result):
        required_fields = ['entry_time', 'exit_time', 'pnl', 'exit_signal']

        for field in required_fields:
            if field not in trade_result:
                return False

        entry_time = trade_result.get('entry_time')
        exit_time = trade_result.get('exit_time')

        if not isinstance(entry_time, datetime) or not isinstance(exit_time, datetime):
            return False

        try:
            float(trade_result['pnl'])
        except (ValueError, TypeError):
            return False

        return True

    def _calculate_duration(self, entry_time, current_time):
        if not isinstance(entry_time, datetime) or not isinstance(current_time, datetime):
            return 0.0

        try:
            if entry_time.tzinfo is not None and current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=entry_time.tzinfo)
            elif entry_time.tzinfo is None and current_time.tzinfo is not None:
                entry_time = entry_time.replace(tzinfo=current_time.tzinfo)
        except Exception:
            pass

        try:
            duration_seconds = (current_time - entry_time).total_seconds()
            return max(0.0, duration_seconds / 3600)
        except Exception:
            return 0.0

    def _calculate_pnl(self, direction, entry_price, current_price):
        if entry_price <= 0 or current_price <= 0:
            return 0.0

        if direction == 'long':
            return (current_price / entry_price) - 1
        elif direction == 'short':
            return (entry_price / current_price) - 1
        else:
            return 0.0