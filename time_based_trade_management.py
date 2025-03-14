import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union


class TimeBasedTradeManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TimeBasedTradeManager")

        # Optimized timing parameters based on trade analysis
        self.min_profit_taking_hours = 1.5  # Reduced from 2 to capture quick profits
        self.small_profit_exit_hours = 24  # Previously 36
        self.stagnant_exit_hours = 30  # Previously 36
        self.max_trade_duration_hours = 60  # Previously 72

        # Enhanced holding period parameters
        self.short_term_lookback = 4  # Reduced from 6
        self.medium_term_lookback = 8  # Reduced from 12
        self.long_term_lookback = 16  # Reduced from 24

        # Optimized stop management parameters
        self.initial_stop_wide_factor = 1.1
        self.tight_stop_factor = 0.5

        # Enhanced profit targets based on backtest analysis
        self.profit_targets = {
            "micro": 0.006,  # New 0.6% micro profit target
            "quick": 0.01,  # 1% quick profit target
            "small": 0.015,  # 1.5% small profit target
            "medium": 0.025,  # 2.5% medium profit target
            "large": 0.04,  # 4% large profit target
            "extended": 0.06  # New 6% extended target
        }

        # Performance tracking
        self.holding_period_stats = {
            "winning_trades": [],
            "losing_trades": []
        }

        # Per-phase position management, optimized for best performers
        self.max_position_age = {
            "neutral": 90,  # Increased - best performing phase
            "uptrend": 72,  # Slightly reduced
            "downtrend": 60,  # Reduced from 72
            "ranging_at_support": 48,  # Reduced from 60
            "ranging_at_resistance": 24,  # Significantly reduced - worst performing phase
            "volatile": 36  # Unchanged
        }

        # Optimized risk factors for duration-based stop adjustment
        self.time_based_risk_factors = {
            4: 1.3,  # 0-4 hours: wider stops (130%)
            8: 1.1,  # 4-8 hours: slightly wider stops (110%)
            16: 1.0,  # 8-16 hours: normal stops (100%)
            24: 0.9,  # 16-24 hours: slightly tighter stops (90%)
            48: 0.7,  # 24-48 hours: tighter stops (70%)
            72: 0.5  # 48-72 hours: very tight stops (50%)
        }

        # Market state tracking
        self.market_volatility_history = []
        self.profit_exit_adaptivity = 0.5  # 0-1 scale, higher = more adaptive
        self.log_performance_data = {}

        # Market phase specific exit preferences
        self.phase_exit_preferences = {
            "neutral": {
                "profit_factor": 1.1,  # Extend profit targets by 10%
                "duration_factor": 1.2  # Extend holding periods by 20%
            },
            "ranging_at_resistance": {
                "profit_factor": 0.8,  # Reduce profit targets by 20%
                "duration_factor": 0.6  # Reduce holding periods by 40%
            }
        }

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
        signal_type = position.get('entry_signal', '')

        trade_duration = (current_time - entry_time).total_seconds() / 3600
        market_phase = market_conditions.get('market_phase', 'neutral')
        volatility = float(market_conditions.get('volatility', 0.5))
        momentum = float(market_conditions.get('momentum', 0.0))
        ensemble_score = float(position.get('ensemble_score', 0.5))

        if direction == 'long':
            pnl_pct = (current_price / entry_price) - 1
        else:
            pnl_pct = (entry_price / current_price) - 1

        # Get phase-specific factors
        phase_settings = self.phase_exit_preferences.get(market_phase, {"profit_factor": 1.0, "duration_factor": 1.0})
        profit_factor = phase_settings["profit_factor"]
        duration_factor = phase_settings["duration_factor"]

        # Calculate max duration with enhanced logic
        max_duration = self._get_max_duration(market_phase, volatility, pnl_pct)
        max_duration *= duration_factor  # Apply phase-specific factor

        # 1. Enhanced time-based exits

        # Exit if trade duration exceeds optimized max duration
        if trade_duration > max_duration:
            return {
                "exit": True,
                "reason": f"MaxDurationReached_{market_phase}",
                "exit_price": current_price
            }

        # Exit completely stagnant positions after optimized stagnant period
        if trade_duration > self.stagnant_exit_hours and abs(pnl_pct) < 0.002:
            return {
                "exit": True,
                "reason": "CompletelyStagnantPosition",
                "exit_price": current_price
            }

        # Exit small profits that haven't developed after significant time
        if trade_duration > self.small_profit_exit_hours and 0 < pnl_pct < (
                self.profit_targets["small"] * profit_factor * 0.8):
            return {
                "exit": True,
                "reason": "SmallProfitLongTimeBasedExit",
                "exit_price": current_price
            }

        # 2. Enhanced QuickProfitTaken logic

        # Check for quick profit opportunity - optimized after backtest analysis
        if self._should_take_quick_profit(trade_duration, pnl_pct, market_conditions, ensemble_score):
            return {
                "exit": True,
                "reason": "QuickProfitTaken",
                "exit_price": current_price
            }

        # 3. Momentum-based exit conditions

        # Exit when momentum starts turning against a profitable position
        if trade_duration > self.min_profit_taking_hours and pnl_pct > self.profit_targets["small"] * profit_factor:
            momentum_threshold = -0.3 if direction == 'long' else 0.3

            if (direction == 'long' and momentum < momentum_threshold) or \
                    (direction == 'short' and momentum > -momentum_threshold):
                return {
                    "exit": True,
                    "reason": "MomentumBasedExit",
                    "exit_price": current_price
                }

        # 4. Enhanced trailing stop logic

        # Calculate time-adjusted stop based on optimized parameters
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
        # Set base duration based on optimized phase-specific values
        base_duration = self.max_position_age.get(market_phase, 60)

        # Volatility factor - more lenient in lower volatility
        if volatility > 0.7:
            volatility_factor = 0.7  # More aggressive exit in high volatility
        elif volatility < 0.3:
            volatility_factor = 1.4  # More lenient in low volatility
        else:
            volatility_factor = 1.0

        # Profit-based adjustment - profitable trades can run longer
        profit_factor = 1.0
        if pnl_pct > self.profit_targets["medium"]:
            profit_factor = 2.0  # Significantly extend duration for good trades
        elif pnl_pct > self.profit_targets["small"]:
            profit_factor = 1.5  # Extend duration for profitable trades
        elif pnl_pct > 0:
            profit_factor = 1.2  # Slightly extend for trades in profit
        elif pnl_pct < -0.015:  # Larger loss
            profit_factor = 0.7  # Exit losing trades faster

        adjusted_duration = base_duration * volatility_factor * profit_factor

        return min(120, max(8, adjusted_duration))

    def _should_take_quick_profit(self, trade_duration: float, pnl_pct: float,
                                  market_conditions: Dict[str, Any], ensemble_score: float = 0.5) -> bool:
        # Require minimum holding period
        if trade_duration < self.min_profit_taking_hours:
            return False

        volatility = float(market_conditions.get('volatility', 0.5))
        market_phase = market_conditions.get('market_phase', 'neutral')
        momentum = float(market_conditions.get('momentum', 0))

        # Get phase-specific profit factor
        phase_profit_factor = self.phase_exit_preferences.get(
            market_phase, {"profit_factor": 1.0}
        )["profit_factor"]

        # Base profit thresholds - adjust based on market phase
        quick_profit = self.profit_targets["quick"] * phase_profit_factor
        small_profit = self.profit_targets["small"] * phase_profit_factor

        # Take profits faster in ranging_at_resistance - it's a problematic phase
        if market_phase == "ranging_at_resistance" and pnl_pct > quick_profit * 0.8:
            return True

        # Take profits in volatile market when we have decent gains
        if volatility > 0.6 and pnl_pct > quick_profit:
            return True

        # Take profit when momentum is reversing against position
        if momentum < -0.4 and pnl_pct > quick_profit:
            return True

        # Enhanced profit taking in neutral phase - based on backtest data
        if market_phase == "neutral":
            # In best performing phase, more selective profit taking
            if pnl_pct > small_profit and trade_duration > 5:
                return True

        # Take larger quick profits after sufficient time
        if pnl_pct > small_profit * 1.2 and trade_duration > 4:
            return True

        # Take very large profits more aggressively - don't let big winners turn to losers
        if pnl_pct > self.profit_targets["medium"] and trade_duration > 2:
            # Scale with ensemble score - higher confidence = hold longer
            confidence_threshold = 0.8 if ensemble_score > 0.7 else 0.5

            if momentum < confidence_threshold:
                return True

        return False

    def _calculate_time_adjusted_stop(self, direction: str, entry_price: float,
                                      current_price: float, trade_duration: float,
                                      current_stop: float, market_conditions: Dict[str, Any]) -> Optional[float]:
        # Don't adjust stop too early
        if trade_duration < 2:  # Reduced from 3
            return None

        volatility = float(market_conditions.get('volatility', 0.5))
        atr = float(market_conditions.get('atr', current_price * 0.01))
        market_phase = market_conditions.get('market_phase', 'neutral')
        momentum = float(market_conditions.get('momentum', 0))

        # Get time-based risk factor from enhanced brackets
        risk_factor = 1.0
        for hours, factor in sorted(self.time_based_risk_factors.items()):
            if trade_duration <= hours:
                risk_factor = factor
                break

        # More aggressive volatility adjustment
        if volatility > 0.7:
            volatility_factor = 1.4  # Higher volatility = wider stops
        elif volatility < 0.3:
            volatility_factor = 0.7  # Lower volatility = tighter stops
        else:
            volatility_factor = 1.0

        # Market phase adjustment based on backtest data
        phase_factor = 1.0
        if market_phase == "neutral":
            phase_factor = 0.9  # Tighter stops in best performing phase
        elif market_phase == "ranging_at_resistance":
            phase_factor = 0.7  # Even tighter stops in worst performing phase

        # Momentum adjustment - tighten stops when momentum weakens
        momentum_factor = 1.0
        if direction == 'long' and momentum < -0.3:
            momentum_factor = 0.8  # Tighter stops when momentum turns negative
        elif direction == 'short' and momentum > 0.3:
            momentum_factor = 0.8  # Tighter stops when momentum turns positive

        if direction == 'long':
            pnl_pct = (current_price / entry_price) - 1

            if pnl_pct <= 0:
                return None

            # Multi-tiered stop adjustment based on profit level
            if pnl_pct > self.profit_targets["large"]:  # Very large profit
                new_stop = max(entry_price * 1.01, current_price - (
                            atr * 1.5 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["medium"]:  # Large profit
                new_stop = max(entry_price * 1.005, current_price - (
                            atr * 2.0 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["small"]:  # Medium profit
                new_stop = max(entry_price * 1.002, current_price - (
                            atr * 2.5 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["quick"]:  # Small profit
                new_stop = max(entry_price, current_price - (
                            atr * 3.0 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["micro"]:  # Micro profit
                new_stop = max(entry_price * 0.998, current_price - (
                            atr * 3.5 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            else:
                return None

            return new_stop if new_stop > current_stop else None

        else:  # short
            pnl_pct = (entry_price / current_price) - 1

            if pnl_pct <= 0:
                return None

            # Multi-tiered stop adjustment based on profit level
            if pnl_pct > self.profit_targets["large"]:  # Very large profit
                new_stop = min(entry_price * 0.99, current_price + (
                            atr * 1.5 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["medium"]:  # Large profit
                new_stop = min(entry_price * 0.995, current_price + (
                            atr * 2.0 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["small"]:  # Medium profit
                new_stop = min(entry_price * 0.998, current_price + (
                            atr * 2.5 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["quick"]:  # Small profit
                new_stop = min(entry_price, current_price + (
                            atr * 3.0 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["micro"]:  # Micro profit
                new_stop = min(entry_price * 1.002, current_price + (
                            atr * 3.5 * risk_factor * volatility_factor * phase_factor * momentum_factor))
            else:
                return None

            return new_stop if new_stop < current_stop else None

    def calculate_optimal_trade_duration(self, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trade_history or len(trade_history) < 10:
            return {
                "optimal_hold_time": 24,
                "confidence": "low",
                "data_points": len(trade_history) if trade_history else 0
            }

        trade_durations = []
        profitable_durations = []

        # Also track by market phase
        phase_durations = {}

        for trade in trade_history:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            pnl = float(trade.get('pnl', 0))
            market_phase = trade.get('market_phase', 'neutral')

            if not entry_time or not exit_time:
                continue

            if not isinstance(entry_time, datetime) or not isinstance(exit_time, datetime):
                continue

            duration_hours = (exit_time - entry_time).total_seconds() / 3600
            trade_durations.append(duration_hours)

            # Track by market phase
            if market_phase not in phase_durations:
                phase_durations[market_phase] = {
                    "all": [],
                    "profitable": []
                }

            phase_durations[market_phase]["all"].append(duration_hours)

            if pnl > 0:
                profitable_durations.append(duration_hours)
                phase_durations[market_phase]["profitable"].append(duration_hours)

        if not trade_durations or not profitable_durations:
            return {
                "optimal_hold_time": 24,
                "confidence": "low",
                "data_points": 0
            }

        avg_duration = np.mean(trade_durations)
        avg_profitable_duration = np.mean(profitable_durations)

        # Calculate optimal duration based on average of profitable trades
        optimal_duration = min(60, max(8, avg_profitable_duration))

        percentiles = {
            "p25": np.percentile(profitable_durations, 25),
            "p50": np.percentile(profitable_durations, 50),
            "p75": np.percentile(profitable_durations, 75)
        }

        # Calculate phase-specific optimal durations
        phase_optimal_durations = {}
        for phase, data in phase_durations.items():
            if data["profitable"]:
                phase_optimal_durations[phase] = np.mean(data["profitable"])
            else:
                phase_optimal_durations[phase] = avg_profitable_duration

        confidence = "medium"
        if len(profitable_durations) > 30:
            confidence = "high"
        elif len(profitable_durations) < 10:
            confidence = "low"

        return {
            "optimal_hold_time": optimal_duration,
            "avg_trade_duration": avg_duration,
            "avg_profitable_duration": avg_profitable_duration,
            "percentiles": percentiles,
            "confidence": confidence,
            "data_points": len(profitable_durations),
            "phase_optimal_durations": phase_optimal_durations
        }

    def update_duration_stats(self, trade_result: Dict[str, Any]) -> None:
        entry_time = trade_result.get('entry_time')
        exit_time = trade_result.get('exit_time')
        pnl = float(trade_result.get('pnl', 0))
        exit_reason = trade_result.get('exit_signal', 'Unknown')
        market_phase = trade_result.get('market_phase', 'neutral')

        if not entry_time or not exit_time:
            return

        if not isinstance(entry_time, datetime) or not isinstance(exit_time, datetime):
            return

        duration_hours = (exit_time - entry_time).total_seconds() / 3600

        trade_data = {
            "duration": duration_hours,
            "exit_reason": exit_reason,
            "market_phase": market_phase,
            "pnl": pnl,
            "pnl_per_hour": pnl / max(1, duration_hours)
        }

        if pnl > 0:
            self.holding_period_stats["winning_trades"].append(trade_data)
        else:
            self.holding_period_stats["losing_trades"].append(trade_data)

        self._update_log_performance(exit_reason, pnl, duration_hours, market_phase)

    def _update_log_performance(self, exit_reason: str, pnl: float, duration: float, market_phase: str) -> None:
        # Track by exit reason
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

        # Track by exit reason and market phase
        key = f"{exit_reason}_{market_phase}"
        if key not in self.log_performance_data:
            self.log_performance_data[key] = {
                "count": 0,
                "win_count": 0,
                "total_pnl": 0,
                "total_duration": 0,
                "avg_pnl": 0,
                "avg_duration": 0,
                "win_rate": 0
            }

        phase_data = self.log_performance_data[key]
        phase_data["count"] += 1
        phase_data["total_pnl"] += pnl
        phase_data["total_duration"] += duration

        if pnl > 0:
            phase_data["win_count"] += 1

        phase_data["avg_pnl"] = phase_data["total_pnl"] / phase_data["count"]
        phase_data["avg_duration"] = phase_data["total_duration"] / phase_data["count"]
        phase_data["win_rate"] = phase_data["win_count"] / phase_data["count"]

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

        winning_trades = self.holding_period_stats["winning_trades"]
        if len(winning_trades) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Only {len(winning_trades)} winning trades, need at least 10"
            }

        # Calculate optimal durations from winning trades
        durations = [t["duration"] for t in winning_trades]
        p25 = np.percentile(durations, 25)
        p50 = np.percentile(durations, 50)
        p75 = np.percentile(durations, 75)

        # Optimize timing parameters
        self.min_profit_taking_hours = max(1.5, min(3, p25 * 0.4))  # More aggressive
        self.small_profit_exit_hours = max(12, min(30, p50 * 1.1))  # Slightly longer than median
        self.stagnant_exit_hours = max(16, min(36, p75 * 0.8))  # Slightly more aggressive

        # Analyze by exit reason
        exit_reason_performance = {}
        for trade in winning_trades:
            reason = trade["exit_reason"]
            if reason not in exit_reason_performance:
                exit_reason_performance[reason] = {
                    "count": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                    "durations": []
                }

            perf = exit_reason_performance[reason]
            perf["count"] += 1
            perf["total_pnl"] += trade["pnl"]
            perf["durations"].append(trade["duration"])
            perf["avg_pnl"] = perf["total_pnl"] / perf["count"]

        # Calculate win rates by duration
        win_rates_by_duration = {}
        pnl_by_duration = {}

        # Use all trades for win rate analysis
        all_trades = self.holding_period_stats["winning_trades"] + self.holding_period_stats["losing_trades"]

        # Analyze by duration bracket
        for duration_bin in [4, 8, 12, 16, 24, 36, 48]:
            trades_in_bin = [t for t in all_trades if t["duration"] <= duration_bin]

            if not trades_in_bin:
                continue

            win_count = sum(1 for t in trades_in_bin if t["pnl"] > 0)
            total_pnl = sum(t["pnl"] for t in trades_in_bin)

            win_rates_by_duration[duration_bin] = win_count / len(trades_in_bin)
            pnl_by_duration[duration_bin] = total_pnl / len(trades_in_bin)

        # Find optimal durations by win rate and pnl
        optimal_by_winrate = max(win_rates_by_duration.items(), key=lambda x: x[1])[0] if win_rates_by_duration else 24
        optimal_by_pnl = max(pnl_by_duration.items(), key=lambda x: x[1])[0] if pnl_by_duration else 24

        # Set max trade duration based on best performance metrics
        self.max_trade_duration_hours = max(optimal_by_winrate, optimal_by_pnl)

        # Analyze by market phase
        market_phase_stats = {}
        for trade in all_trades:
            phase = trade.get("market_phase", "neutral")
            is_win = trade["pnl"] > 0

            if phase not in market_phase_stats:
                market_phase_stats[phase] = {
                    "count": 0,
                    "win_count": 0,
                    "total_pnl": 0,
                    "win_rate": 0,
                    "avg_pnl": 0,
                    "durations": []
                }

            stats = market_phase_stats[phase]
            stats["count"] += 1
            if is_win:
                stats["win_count"] += 1
            stats["total_pnl"] += trade["pnl"]
            stats["durations"].append(trade["duration"])

            if stats["count"] > 0:
                stats["win_rate"] = stats["win_count"] / stats["count"]
                stats["avg_pnl"] = stats["total_pnl"] / stats["count"]

        # Optimize phase-specific parameters
        for phase, stats in market_phase_stats.items():
            if stats["count"] < 5:
                continue

            # Set phase-specific profit factors based on performance
            if phase not in self.phase_exit_preferences:
                self.phase_exit_preferences[phase] = {"profit_factor": 1.0, "duration_factor": 1.0}

            # Adjust profit targets based on phase performance
            if stats["win_rate"] > 0.65 and stats["avg_pnl"] > 0:
                # More profitable phases get more room to run
                self.phase_exit_preferences[phase]["profit_factor"] = 1.2
                self.phase_exit_preferences[phase]["duration_factor"] = 1.2
            elif stats["win_rate"] < 0.45 or stats["avg_pnl"] < 0:
                # Challenging phases get more conservative parameters
                self.phase_exit_preferences[phase]["profit_factor"] = 0.7
                self.phase_exit_preferences[phase]["duration_factor"] = 0.6

            # Set phase-specific max position age
            if len(stats["durations"]) >= 5:
                optimal_phase_duration = np.percentile(stats["durations"], 60)  # 60th percentile
                self.max_position_age[phase] = max(12, min(90, optimal_phase_duration * 1.5))

        return {
            "status": "optimized",
            "parameters": {
                "min_profit_taking_hours": self.min_profit_taking_hours,
                "small_profit_exit_hours": self.small_profit_exit_hours,
                "stagnant_exit_hours": self.stagnant_exit_hours,
                "max_trade_duration_hours": self.max_trade_duration_hours,
                "max_position_age": self.max_position_age,
                "phase_exit_preferences": self.phase_exit_preferences
            },
            "stats": {
                "win_rates_by_duration": win_rates_by_duration,
                "pnl_by_duration": pnl_by_duration,
                "optimal_by_winrate": optimal_by_winrate,
                "optimal_by_pnl": optimal_by_pnl,
                "exit_reason_performance": exit_reason_performance,
                "market_phase_stats": market_phase_stats
            }
        }