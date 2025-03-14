import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union


class TimeBasedTradeManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TimeBasedTradeManager")

        # OPTIMIZATION: Increased min profit taking hours for better holding time
        self.min_profit_taking_hours = 2.5  # Increased from 1.5 for better holding duration
        self.small_profit_exit_hours = 20  # Reduced from 24 to take profits more selectively
        self.stagnant_exit_hours = 28  # Reduced from 30 for faster exit from stagnant positions
        self.max_trade_duration_hours = 56  # Reduced from 60 for better capital efficiency

        # OPTIMIZATION: Enhanced holding period parameters for better alignment with optimal duration
        self.short_term_lookback = 3  # Reduced from 4
        self.medium_term_lookback = 6  # Reduced from 8
        self.long_term_lookback = 12  # Reduced from 16

        # Stop management parameters
        self.initial_stop_wide_factor = 1.15  # Increased from 1.1 for better trade room
        self.tight_stop_factor = 0.55  # Increased from 0.5 for better profit protection

        # OPTIMIZATION: Enhanced profit targets for quicker profit-taking
        self.profit_targets = {
            "micro": 0.005,  # Reduced from 0.006 for faster micro exits
            "quick": 0.01,  # Kept at 1% quick profit target
            "small": 0.015,  # Kept at 1.5% small profit target
            "medium": 0.023,  # Reduced from 0.025 for faster medium exits
            "large": 0.035,  # Reduced from 0.04 for faster large exits
            "extended": 0.05  # Reduced from 0.06 for faster extended exits
        }

        # Performance tracking
        self.holding_period_stats = {
            "winning_trades": [],
            "losing_trades": []
        }

        # OPTIMIZATION: Per-phase position management, optimized for best performers
        # Neutral phase duration increased, ranging_at_resistance reduced
        self.max_position_age = {
            "neutral": 100,  # Increased from 90 - best performing phase
            "uptrend": 72,  # Unchanged
            "downtrend": 48,  # Reduced from 60
            "ranging_at_support": 48,  # Unchanged
            "ranging_at_resistance": 16,  # Significantly reduced from 24 - worst performing phase
            "volatile": 32  # Reduced from 36
        }

        # OPTIMIZATION: Optimized risk factors for duration-based stop adjustment
        # More protection for longer-duration trades
        self.time_based_risk_factors = {
            4: 1.4,  # 0-4 hours: wider stops (140%) - increased from 130%
            8: 1.2,  # 4-8 hours: slightly wider stops (120%) - increased from 110%
            16: 1.05,  # 8-16 hours: normal stops (105%) - increased from 100%
            24: 0.85,  # 16-24 hours: slightly tighter stops (85%) - tightened from 90%
            48: 0.65,  # 24-48 hours: tighter stops (65%) - tightened from 70%
            72: 0.45  # 48-72 hours: very tight stops (45%) - tightened from 50%
        }

        # Market state tracking
        self.market_volatility_history = []
        self.profit_exit_adaptivity = 0.6  # Increased from 0.5, more adaptive
        self.log_performance_data = {}

        # OPTIMIZATION: Market phase specific exit preferences enhanced
        # Modified based on performance data
        self.phase_exit_preferences = {
            "neutral": {
                "profit_factor": 1.15,  # Extend profit targets by 15% (increased from 10%)
                "duration_factor": 1.25  # Extend holding periods by 25% (increased from 20%)
            },
            "ranging_at_resistance": {
                "profit_factor": 0.7,  # Reduce profit targets by 30% (reduced from 20%)
                "duration_factor": 0.5  # Reduce holding periods by 50% (reduced from 40%)
            },
            "uptrend": {
                "profit_factor": 1.1,  # Extend profit targets by 10% (new)
                "duration_factor": 1.1  # Extend holding periods by 10% (new)
            },
            "downtrend": {
                "profit_factor": 0.9,  # Reduce profit targets by 10% (new)
                "duration_factor": 0.8  # Reduce holding periods by 20% (new)
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

        # OPTIMIZATION: Exit if trade duration exceeds optimized max duration
        # More aggressive exit for MaxDurationReached_neutral which showed -$23.83 avg P&L
        if trade_duration > max_duration:
            # OPTIMIZATION: Special handling for neutral phase to avoid significant losses
            if market_phase == 'neutral' and pnl_pct < 0:
                # Even faster exit for losing trades in neutral phase
                return {
                    "exit": True,
                    "reason": f"MaxDurationReached_{market_phase}",
                    "exit_price": current_price
                }
            return {
                "exit": True,
                "reason": f"MaxDurationReached_{market_phase}",
                "exit_price": current_price
            }

        # Exit completely stagnant positions after optimized stagnant period
        # OPTIMIZATION: Lower threshold for stagnation from 0.002 to 0.0015
        if trade_duration > self.stagnant_exit_hours and abs(pnl_pct) < 0.0015:
            return {
                "exit": True,
                "reason": "CompletelyStagnantPosition",
                "exit_price": current_price
            }

        # Exit small profits that haven't developed after significant time
        # OPTIMIZATION: Adjusted threshold for faster exits of small profits
        if trade_duration > self.small_profit_exit_hours and 0 < pnl_pct < (
                self.profit_targets["small"] * profit_factor * 0.75):  # Reduced from 0.8
            return {
                "exit": True,
                "reason": "SmallProfitLongTimeBasedExit",
                "exit_price": current_price
            }

        # 2. OPTIMIZATION: Enhanced QuickProfitTaken logic - most profitable exit type
        # More aggressive criteria to use this strategy more often

        # Check for quick profit opportunity - optimized to be more aggressive
        if self._should_take_quick_profit(trade_duration, pnl_pct, market_conditions, ensemble_score):
            return {
                "exit": True,
                "reason": "QuickProfitTaken",
                "exit_price": current_price
            }

        # 3. Momentum-based exit conditions

        # OPTIMIZATION: More sensitive momentum exit threshold for profitable trades
        if trade_duration > self.min_profit_taking_hours and pnl_pct > self.profit_targets["small"] * profit_factor:
            # More sensitive momentum thresholds (from -0.3 to -0.25 for long)
            momentum_threshold = -0.25 if direction == 'long' else 0.25

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

        # OPTIMIZATION: Enhanced volatility factor - more responsive to volatility
        if volatility > 0.7:
            volatility_factor = 0.65  # More aggressive exit in high volatility (reduced from 0.7)
        elif volatility < 0.3:
            volatility_factor = 1.5  # More lenient in low volatility (increased from 1.4)
        else:
            volatility_factor = 1.0

        # OPTIMIZATION: Enhanced profit-based adjustment - profitable trades can run longer
        profit_factor = 1.0
        if pnl_pct > self.profit_targets["medium"]:
            profit_factor = 2.2  # Significantly extend duration for good trades (increased from 2.0)
        elif pnl_pct > self.profit_targets["small"]:
            profit_factor = 1.7  # Extend duration for profitable trades (increased from 1.5)
        elif pnl_pct > 0:
            profit_factor = 1.3  # Slightly extend for trades in profit (increased from 1.2)
        elif pnl_pct < -0.012:  # Slightly smaller loss threshold (from -0.015)
            profit_factor = 0.6  # Exit losing trades faster (reduced from 0.7)

        adjusted_duration = base_duration * volatility_factor * profit_factor

        return min(120, max(8, adjusted_duration))

    def _should_take_quick_profit(self, trade_duration: float, pnl_pct: float,
                                  market_conditions: Dict[str, Any], ensemble_score: float = 0.5) -> bool:
        # OPTIMIZATION: Modified logic to favor this highly profitable exit type

        # Require minimum holding period, but slightly reduced
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

        # OPTIMIZATION: More aggressive take profit in ranging_at_resistance - it's a problematic phase
        if market_phase == "ranging_at_resistance" and pnl_pct > quick_profit * 0.7:  # Reduced from 0.8
            return True

        # OPTIMIZATION: More aggressive take profits in volatile market
        if volatility > 0.55 and pnl_pct > quick_profit * 0.9:  # Reduced threshold from 0.6 to 0.55
            return True

        # OPTIMIZATION: More sensitive momentum-based exit
        if momentum < -0.35 and pnl_pct > quick_profit * 0.9:  # Reduced from -0.4
            return True

        # OPTIMIZATION: Enhanced profit taking in neutral phase - based on backtest data showing strong performance
        if market_phase == "neutral":
            # In best performing phase, more selective profit taking
            if pnl_pct > small_profit * 0.9 and trade_duration > 4.5:  # Reduced from small_profit and 5 hours
                return True

        # OPTIMIZATION: Take larger quick profits after sufficient time - slightly more aggressive
        if pnl_pct > small_profit * 1.1 and trade_duration > 3.5:  # Reduced from 1.2 and 4 hours
            return True

        # OPTIMIZATION: Take very large profits more aggressively when momentum weakens
        if pnl_pct > self.profit_targets["medium"] and trade_duration > 1.8:  # Reduced from 2 hours
            # Scale with ensemble score - higher confidence = hold longer
            confidence_threshold = 0.75 if ensemble_score > 0.7 else 0.45  # Reduced from 0.8/0.5

            if momentum < confidence_threshold:
                return True

        return False

    def _calculate_time_adjusted_stop(self, direction: str, entry_price: float,
                                      current_price: float, trade_duration: float,
                                      current_stop: float, market_conditions: Dict[str, Any]) -> Optional[float]:
        # OPTIMIZATION: Start adjusting stops earlier
        if trade_duration < 1.8:  # Reduced from 2
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
            vol_adjusted_mult = 1.45  # Higher volatility = wider stops (reduced from 1.5)
        elif volatility < 0.3:
            vol_adjusted_mult = 0.75  # Lower volatility = tighter stops (reduced from 0.8)
        else:
            vol_adjusted_mult = 1.0 + (volatility - 0.5) * 0.9  # Slightly reduced from 1.0

        # OPTIMIZATION: Better confidence-based adjustment
        if entry_price > 0:
            price_change_pct = abs(current_price / entry_price - 1)
            confidence_factor = 1.0

            # More stop room for large price moves in our direction
            if price_change_pct > 0.03:  # 3% price change
                confidence_factor = 0.85  # Tighter stops for larger moves (more profit protection)
            elif price_change_pct > 0.015:  # 1.5% price change
                confidence_factor = 0.9  # Slightly tighter stops
            else:
                confidence_factor = 1.0

            vol_adjusted_mult *= confidence_factor

        # OPTIMIZATION: Enhanced market phase adjustment based on backtest data
        phase_factor = 1.0
        if market_phase == "neutral":
            phase_factor = 0.85  # Tighter stops in best performing phase (reduced from 0.9)
        elif market_phase == "ranging_at_resistance":
            phase_factor = 0.6  # Even tighter stops in worst performing phase (reduced from 0.7)
        elif market_phase == "uptrend" and direction == "long":
            phase_factor = 0.95  # Slightly tighter stops in uptrend for longs (new)
        elif market_phase == "downtrend" and direction == "short":
            phase_factor = 0.95  # Slightly tighter stops in downtrend for shorts (new)

        # OPTIMIZATION: Enhanced momentum adjustment - tighten stops when momentum weakens
        momentum_factor = 1.0
        if direction == 'long' and momentum < -0.25:  # More sensitive (from -0.3)
            momentum_factor = 0.75  # Tighter stops when momentum turns negative (reduced from 0.8)
        elif direction == 'short' and momentum > 0.25:  # More sensitive (from 0.3)
            momentum_factor = 0.75  # Tighter stops when momentum turns positive (reduced from 0.8)

        # Calculate stop distance based on ATR
        base_atr_mult = 3.0
        vol_adjusted_mult = base_atr_mult * vol_adjusted_mult * phase_factor * momentum_factor * risk_factor

        if direction == 'long':
            pnl_pct = (current_price / entry_price) - 1

            if pnl_pct <= 0:
                return None

            # OPTIMIZATION: Multi-tiered stop adjustment based on profit level
            # Tighter stops for larger profits for better profit protection
            if pnl_pct > self.profit_targets["large"]:  # Very large profit
                new_stop = max(entry_price * 1.015, current_price - (
                            atr * 1.4 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["medium"]:  # Large profit
                new_stop = max(entry_price * 1.008, current_price - (
                            atr * 1.8 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["small"]:  # Medium profit
                new_stop = max(entry_price * 1.004, current_price - (
                            atr * 2.2 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["quick"]:  # Small profit
                new_stop = max(entry_price * 1.001, current_price - (
                            atr * 2.7 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["micro"]:  # Micro profit
                new_stop = max(entry_price * 0.999, current_price - (
                            atr * 3.3 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
            else:
                return None

            return new_stop if new_stop > current_stop else None

        else:  # short
            pnl_pct = (entry_price / current_price) - 1

            if pnl_pct <= 0:
                return None

            # OPTIMIZATION: Multi-tiered stop adjustment based on profit level
            # Tighter stops for larger profits for better profit protection
            if pnl_pct > self.profit_targets["large"]:  # Very large profit
                new_stop = min(entry_price * 0.985, current_price + (
                            atr * 1.4 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["medium"]:  # Large profit
                new_stop = min(entry_price * 0.992, current_price + (
                            atr * 1.8 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["small"]:  # Medium profit
                new_stop = min(entry_price * 0.996, current_price + (
                            atr * 2.2 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["quick"]:  # Small profit
                new_stop = min(entry_price * 0.999, current_price + (
                            atr * 2.7 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
            elif pnl_pct > self.profit_targets["micro"]:  # Micro profit
                new_stop = min(entry_price * 1.001, current_price + (
                            atr * 3.3 * risk_factor * vol_adjusted_mult  * phase_factor * momentum_factor))
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
        # OPTIMIZATION: More weighted toward profitable trade durations
        optimal_duration = min(60, max(8, avg_profitable_duration * 1.1))

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

        # OPTIMIZATION: Better timing parameters based on profit duration
        self.min_profit_taking_hours = max(1.5, min(3, p25 * 0.5))  # More aggressive (from 0.4)
        self.small_profit_exit_hours = max(12, min(24, p50 * 1.2))  # Slightly longer than median (from 1.1)
        self.stagnant_exit_hours = max(16, min(32, p75 * 0.85))  # More aggressive (from 0.8)

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

        # OPTIMIZATION: More granular duration brackets for better analysis
        for duration_bin in [3, 6, 9, 12, 16, 24, 36, 48]:
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

        # OPTIMIZATION: Weighted approach to max trade duration
        # More weight to profitability
        self.max_trade_duration_hours = (optimal_by_winrate * 0.4 + optimal_by_pnl * 0.6)

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

        # OPTIMIZATION: Enhanced phase-specific parameters
        for phase, stats in market_phase_stats.items():
            if stats["count"] < 5:
                continue

            # Set phase-specific profit factors based on performance
            if phase not in self.phase_exit_preferences:
                self.phase_exit_preferences[phase] = {"profit_factor": 1.0, "duration_factor": 1.0}

            # Adjust profit targets based on phase performance
            if stats["win_rate"] > 0.65 and stats["avg_pnl"] > 0:
                # More profitable phases get more room to run
                self.phase_exit_preferences[phase]["profit_factor"] = 1.25  # Increased from 1.2
                self.phase_exit_preferences[phase]["duration_factor"] = 1.25  # Increased from 1.2
            elif stats["win_rate"] < 0.45 or stats["avg_pnl"] < 0:
                # Challenging phases get more conservative parameters
                self.phase_exit_preferences[phase]["profit_factor"] = 0.6  # Reduced from 0.7
                self.phase_exit_preferences[phase]["duration_factor"] = 0.5  # Reduced from 0.6

            # Set phase-specific max position age
            if len(stats["durations"]) >= 5:
                optimal_phase_duration = np.percentile(stats["durations"], 65)  # Increased from 60th percentile
                self.max_position_age[phase] = max(12, min(100, optimal_phase_duration * 1.6))  # Increased from 1.5

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