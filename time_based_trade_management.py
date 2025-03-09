import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union


class TimeBasedTradeManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TimeBasedTradeManager")

        # Increasing these parameters to allow trades to stay open longer
        self.min_profit_taking_hours = 2
        self.small_profit_exit_hours = 36
        self.stagnant_exit_hours = 36
        self.max_trade_duration_hours = 72

        self.short_term_lookback = 6
        self.medium_term_lookback = 12
        self.long_term_lookback = 24

        self.initial_stop_wide_factor = 1.2
        self.tight_stop_factor = 0.6

        self.profit_targets = {
            "quick": 0.01,  # 1% quick profit target
            "small": 0.015,  # 1.5% small profit target
            "medium": 0.025,  # 2.5% medium profit target
            "large": 0.04  # 4% large profit target
        }

        self.holding_period_stats = {
            "winning_trades": [],
            "losing_trades": []
        }

        # Increasing position age parameters
        self.max_position_age = {
            "uptrend": 90,    # Significantly increased to prevent early exits
            "downtrend": 72,  # Increased from 48
            "ranging": 60,    # Increased from 36
            "volatile": 36    # Increased from 18
        }

        # Making risk factors more conservative for longer-duration trades
        self.time_based_risk_factors = {
            12: 1.2,  # 0-12 hours: wider stops (120%)
            24: 1.0,  # 12-24 hours: normal stops (100%)
            48: 0.8,  # 24-48 hours: slightly tighter stops (80%)
            72: 0.6,  # 48-72 hours: tighter stops (60%)
            96: 0.5   # 72+ hours: extremely tight stops (50%)
        }

        self.log_performance_data = {}

    def evaluate_time_based_exit(self, position: Dict[str, Any],
                                 current_price: float,
                                 current_time: datetime,
                                 market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        entry_time = position.get('entry_time')
        if not entry_time or not isinstance(entry_time, datetime):
            return {"exit": False, "reason": "InvalidEntryTime"}

        direction = position.get('direction', 'long')
        entry_price = float(position.get('entry_price', 0))
        current_stop = float(position.get('current_stop_loss', 0))

        trade_duration = (current_time - entry_time).total_seconds() / 3600
        market_phase = market_conditions.get('market_phase', 'neutral')
        volatility = float(market_conditions.get('volatility', 0.5))

        if direction == 'long':
            pnl_pct = (current_price / entry_price) - 1
        else:
            pnl_pct = (entry_price / current_price) - 1

        max_duration = self._get_max_duration(market_phase, volatility, pnl_pct)

        # Only exit if trade duration is significantly beyond the max duration
        if trade_duration > max_duration * 1.25:
            return {
                "exit": True,
                "reason": f"MaxDurationReached_{market_phase}",
                "exit_price": current_price
            }

        # Only exit if the position is completely stagnant for a long time
        if trade_duration > self.stagnant_exit_hours * 1.5 and abs(pnl_pct) < 0.002:
            return {
                "exit": True,
                "reason": "CompletelyStagnantPosition",
                "exit_price": current_price
            }

        # Only exit small profits after significant time
        if trade_duration > self.small_profit_exit_hours * 1.5 and 0 < pnl_pct < (self.profit_targets["small"] * 0.8):
            return {
                "exit": True,
                "reason": "SmallProfitLongTimeBasedExit",
                "exit_price": current_price
            }

        # Only take quick profits in specific conditions
        if self._should_take_quick_profit(trade_duration, pnl_pct, market_conditions):
            return {
                "exit": True,
                "reason": "QuickProfitTaken",
                "exit_price": current_price
            }

        new_stop = self._calculate_time_adjusted_stop(
            direction, entry_price, current_price, trade_duration,
            current_stop, market_conditions
        )

        if new_stop:
            if (direction == 'long' and new_stop > current_stop) or \
                    (direction == 'short' and new_stop < current_stop):
                return {
                    "exit": False,
                    "update_stop": True,
                    "new_stop": float(new_stop),
                    "reason": "TimeBasedStopAdjustment"
                }

        return {"exit": False, "reason": "NoTimeBasedActionNeeded"}

    def _get_max_duration(self, market_phase: str, volatility: float, pnl_pct: float) -> float:
        base_duration = self.max_position_age.get(market_phase, 60)  # Changed default to 60 hours

        # More lenient volatility factor
        if volatility > 0.7:
            volatility_factor = 0.8
        elif volatility < 0.3:
            volatility_factor = 1.3
        else:
            volatility_factor = 1.0

        # More lenient profit factor - profitable trades stay open longer
        profit_factor = 1.0
        if pnl_pct > self.profit_targets["medium"]:
            profit_factor = 2.0  # Significantly increased
        elif pnl_pct > 0:
            profit_factor = 1.5  # Increased
        elif pnl_pct < -0.01:
            profit_factor = 0.8

        adjusted_duration = base_duration * volatility_factor * profit_factor

        return min(120, max(12, adjusted_duration))  # Increased min and max durations

    # Modify in time_based_trade_management.py - _should_take_quick_profit method

    def _should_take_quick_profit(self, trade_duration: float, pnl_pct: float,
                                  market_conditions: Dict[str, Any]) -> bool:
        # Need to be past the minimum profit taking hours
        if trade_duration < self.min_profit_taking_hours:
            return False

        volatility = float(market_conditions.get('volatility', 0.5))
        market_phase = market_conditions.get('market_phase', 'neutral')
        momentum = float(market_conditions.get('momentum', 0))

        # Base profit threshold - lower slightly to take profits earlier
        profit_threshold = self.profit_targets["quick"] * 1.1

        # Take profits quicker in volatile conditions or when momentum is weakening
        if market_phase == "ranging_at_resistance" and pnl_pct > profit_threshold:
            return True

        if volatility > 0.6 and pnl_pct > profit_threshold:  # Changed from 0.7 to 0.6
            return True

        # Take profit when momentum is reversing against our position
        if momentum < -0.5 and pnl_pct > profit_threshold:  # Changed from -0.7 to -0.5
            return True

        # Add new case: take profit in neutral markets after good gains
        if market_phase == "neutral" and pnl_pct > profit_threshold * 1.2:
            return True

        return False

    # Modify in time_based_trade_management.py - _calculate_time_adjusted_stop method
    def _calculate_time_adjusted_stop(self, direction: str, entry_price: float,
                                      current_price: float, trade_duration: float,
                                      current_stop: float, market_conditions: Dict[str, Any]) -> Optional[float]:
        # Don't adjust stop too early
        if trade_duration < 3:  # Changed from 4 to 3
            return None

        volatility = float(market_conditions.get('volatility', 0.5))
        atr = float(market_conditions.get('atr', current_price * 0.01))
        market_phase = market_conditions.get('market_phase', 'neutral')

        risk_factor = 1.0
        for hours, factor in sorted(self.time_based_risk_factors.items()):
            if trade_duration <= hours:
                risk_factor = factor
                break

        # More conservative volatility adjustment
        if volatility > 0.7:
            volatility_factor = 1.3
        elif volatility < 0.3:
            volatility_factor = 0.8
        else:
            volatility_factor = 1.0

        # Tighter stops in ranging markets
        if market_phase in ["ranging_at_resistance", "ranging_at_support"]:
            volatility_factor *= 0.8

        if direction == 'long':
            pnl_pct = (current_price / entry_price) - 1

            if pnl_pct <= 0:
                return None

            # Tighter trailing stops for better profit protection
            if pnl_pct > self.profit_targets["medium"] * 1.5:
                new_stop = max(entry_price, current_price - (atr * 2.2 * risk_factor * volatility_factor))  # 2.5 to 2.2
            elif pnl_pct > self.profit_targets["small"] * 1.2:
                new_stop = max(entry_price * 0.998,
                               current_price - (atr * 3.0 * risk_factor * volatility_factor))  # 3.5 to 3.0
            elif pnl_pct > self.profit_targets["small"] * 0.7:  # Add new level for breakeven+ stops
                new_stop = max(entry_price * 1.001, current_price - (atr * 4.0 * risk_factor * volatility_factor))
            else:
                return None

            return new_stop if new_stop > current_stop else None

        else:  # short
            pnl_pct = (entry_price / current_price) - 1

            if pnl_pct <= 0:
                return None

            # Tighter trailing stops
            if pnl_pct > self.profit_targets["medium"] * 1.5:
                new_stop = min(entry_price, current_price + (atr * 2.2 * risk_factor * volatility_factor))  # 2.5 to 2.2
            elif pnl_pct > self.profit_targets["small"] * 1.2:
                new_stop = min(entry_price * 1.002,
                               current_price + (atr * 3.0 * risk_factor * volatility_factor))  # 3.5 to 3.0
            elif pnl_pct > self.profit_targets["small"] * 0.7:  # Add new level for breakeven+ stops
                new_stop = min(entry_price * 0.999, current_price + (atr * 4.0 * risk_factor * volatility_factor))
            else:
                return None

            return new_stop if new_stop < current_stop else None

    def calculate_optimal_trade_duration(self, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trade_history or len(trade_history) < 10:
            return {
                "optimal_hold_time": 36,  # Increased default
                "confidence": "low",
                "data_points": len(trade_history) if trade_history else 0
            }

        trade_durations = []
        profitable_durations = []

        for trade in trade_history:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            pnl = float(trade.get('pnl', 0))

            if not entry_time or not exit_time:
                continue

            if not isinstance(entry_time, datetime) or not isinstance(exit_time, datetime):
                continue

            duration_hours = (exit_time - entry_time).total_seconds() / 3600
            trade_durations.append(duration_hours)

            if pnl > 0:
                profitable_durations.append(duration_hours)

        if not trade_durations or not profitable_durations:
            return {
                "optimal_hold_time": 36,  # Increased default
                "confidence": "low",
                "data_points": 0
            }

        avg_duration = np.mean(trade_durations)
        avg_profitable_duration = np.mean(profitable_durations)

        # Allow for longer optimal durations
        optimal_duration = min(96, max(12, avg_profitable_duration))

        min_profitable = min(profitable_durations)
        max_profitable = max(profitable_durations)

        percentiles = {
            "p25": np.percentile(profitable_durations, 25),
            "p50": np.percentile(profitable_durations, 50),
            "p75": np.percentile(profitable_durations, 75)
        }

        confidence = "medium"
        if len(profitable_durations) > 30:
            confidence = "high"
        elif len(profitable_durations) < 10:
            confidence = "low"

        return {
            "optimal_hold_time": optimal_duration,
            "avg_trade_duration": avg_duration,
            "avg_profitable_duration": avg_profitable_duration,
            "min_profitable_duration": min_profitable,
            "max_profitable_duration": max_profitable,
            "percentiles": percentiles,
            "confidence": confidence,
            "data_points": len(profitable_durations)
        }

    def update_duration_stats(self, trade_result: Dict[str, Any]) -> None:
        entry_time = trade_result.get('entry_time')
        exit_time = trade_result.get('exit_time')
        pnl = float(trade_result.get('pnl', 0))
        exit_reason = trade_result.get('exit_signal', 'Unknown')

        if not entry_time or not exit_time:
            return

        if not isinstance(entry_time, datetime) or not isinstance(exit_time, datetime):
            return

        duration_hours = (exit_time - entry_time).total_seconds() / 3600

        trade_data = {
            "duration": duration_hours,
            "exit_reason": exit_reason,
            "pnl": pnl,
            "pnl_per_hour": pnl / max(1, duration_hours)
        }

        if pnl > 0:
            self.holding_period_stats["winning_trades"].append(trade_data)
        else:
            self.holding_period_stats["losing_trades"].append(trade_data)

        self._update_log_performance(exit_reason, pnl, duration_hours)

    def _update_log_performance(self, exit_reason: str, pnl: float, duration: float) -> None:
        if exit_reason not in self.log_performance_data:
            self.log_performance_data[exit_reason] = {
                "count": 0,
                "win_count": 0,
                "total_pnl": 0,
                "total_duration": 0,
                "avg_pnl": 0,
                "avg_duration": 0,
                "win_rate": 0
            }

        data = self.log_performance_data[exit_reason]
        data["count"] += 1
        data["total_pnl"] += pnl
        data["total_duration"] += duration

        if pnl > 0:
            data["win_count"] += 1

        data["avg_pnl"] = data["total_pnl"] / data["count"]
        data["avg_duration"] = data["total_duration"] / data["count"]
        data["win_rate"] = data["win_count"] / data["count"]

    def get_exit_performance_stats(self) -> Dict[str, Any]:
        return {
            "exit_stats": self.log_performance_data,
            "optimal_durations": self.calculate_optimal_trade_duration(
                self.holding_period_stats["winning_trades"] + self.holding_period_stats["losing_trades"]
            )
        }

    def optimize_time_parameters(self) -> Dict[str, Any]:
        if not self.holding_period_stats["winning_trades"]:
            return {
                "status": "insufficient_data",
                "message": "Not enough winning trades to optimize parameters"
            }

        winning_durations = [t["duration"] for t in self.holding_period_stats["winning_trades"]]

        if len(winning_durations) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Only {len(winning_durations)} winning trades, need at least 10"
            }

        # More conservative duration adjustments
        p25 = np.percentile(winning_durations, 25)
        p50 = np.percentile(winning_durations, 50)
        p75 = np.percentile(winning_durations, 75)

        self.min_profit_taking_hours = max(2, min(4, p25 * 0.5))
        self.small_profit_exit_hours = max(12, min(36, p50))
        self.stagnant_exit_hours = max(24, min(48, p75))

        win_rates_by_duration = {}
        pnl_by_duration = {}

        all_trades = self.holding_period_stats["winning_trades"] + self.holding_period_stats["losing_trades"]

        for duration_bin in [12, 24, 36, 48, 72]:
            trades_in_bin = [t for t in all_trades if t["duration"] <= duration_bin]

            if not trades_in_bin:
                continue

            win_count = sum(1 for t in trades_in_bin if t["pnl"] > 0)
            total_pnl = sum(t["pnl"] for t in trades_in_bin)

            win_rates_by_duration[duration_bin] = win_count / len(trades_in_bin)
            pnl_by_duration[duration_bin] = total_pnl / len(trades_in_bin)

        optimal_by_winrate = max(win_rates_by_duration.items(), key=lambda x: x[1])[0] if win_rates_by_duration else 36
        optimal_by_pnl = max(pnl_by_duration.items(), key=lambda x: x[1])[0] if pnl_by_duration else 36

        self.max_trade_duration_hours = max(optimal_by_winrate, optimal_by_pnl)

        # More conservative max position age adjustments
        for market_phase in self.max_position_age:
            if market_phase == "uptrend":
                self.max_position_age[market_phase] = min(120, self.max_trade_duration_hours * 1.5)
            elif market_phase == "downtrend":
                self.max_position_age[market_phase] = min(96, self.max_trade_duration_hours * 1.2)
            elif market_phase == "ranging":
                self.max_position_age[market_phase] = max(24, self.max_trade_duration_hours * 0.9)
            elif market_phase == "volatile":
                self.max_position_age[market_phase] = max(12, self.max_trade_duration_hours * 0.6)

        return {
            "status": "optimized",
            "parameters": {
                "min_profit_taking_hours": self.min_profit_taking_hours,
                "small_profit_exit_hours": self.small_profit_exit_hours,
                "stagnant_exit_hours": self.stagnant_exit_hours,
                "max_trade_duration_hours": self.max_trade_duration_hours,
                "max_position_age": self.max_position_age
            },
            "stats": {
                "win_rates_by_duration": win_rates_by_duration,
                "pnl_by_duration": pnl_by_duration,
                "optimal_by_winrate": optimal_by_winrate,
                "optimal_by_pnl": optimal_by_pnl
            }
        }