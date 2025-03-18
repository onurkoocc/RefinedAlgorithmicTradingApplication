import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union


class TimeBasedTradeManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TimeBasedTradeManager")

        self.min_profit_taking_hours = self._get_config_float("time_management", "min_profit_taking_hours", 1.8)
        self.small_profit_exit_hours = self._get_config_float("time_management", "small_profit_exit_hours", 8.0)
        self.stagnant_exit_hours = self._get_config_float("time_management", "stagnant_exit_hours", 10.0)
        self.max_trade_duration_hours = self._get_config_float("time_management", "max_trade_duration_hours", 36.0)

        self.short_term_lookback = 3
        self.medium_term_lookback = 6
        self.long_term_lookback = 12

        self.profit_targets = {
            "micro": 0.004,
            "quick": 0.008,
            "small": 0.012,
            "medium": 0.018,
            "large": 0.03,
            "extended": 0.045
        }

        self.max_position_age = {
            "neutral": 24.0,
            "uptrend": 18.0,
            "downtrend": 16.0,
            "ranging_at_support": 12.0,
            "ranging_at_resistance": 6.0,
            "volatile": 10.0
        }

        self.phase_exit_preferences = {
            "neutral": {
                "profit_factor": 1.0,
                "duration_factor": 1.1
            },
            "ranging_at_resistance": {
                "profit_factor": 0.6,
                "duration_factor": 0.4
            },
            "uptrend": {
                "profit_factor": 1.0,
                "duration_factor": 1.0
            },
            "downtrend": {
                "profit_factor": 0.8,
                "duration_factor": 0.7
            }
        }

        self.time_based_risk_factors = {
            2: 1.4,
            4: 1.3,
            8: 1.1,
            12: 1.0,
            16: 0.9,
            24: 0.8,
            36: 0.6,
            48: 0.5,
            72: 0.4
        }

        self.holding_period_stats = {
            "winning_trades": [],
            "losing_trades": []
        }
        self.log_performance_data = {}

    def _get_config_float(self, section: str, key: str, default: float) -> float:
        try:
            value = self.config.get(section, key, default)
            return float(value)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid config value for {section}.{key}, using default {default}")
            return default

    def _load_max_position_age(self) -> Dict[str, float]:
        try:
            default_values = {
                "neutral": 90.0,
                "uptrend": 72.0,
                "downtrend": 48.0,
                "ranging_at_support": 48.0,
                "ranging_at_resistance": 16.0,
                "volatile": 32.0
            }

            config_values = self.config.get("time_management", "max_position_age", {})

            result = {}
            for phase, default in default_values.items():
                if phase in config_values:
                    try:
                        value = float(config_values[phase])
                        if value <= 0:
                            raise ValueError("Value must be positive")
                        result[phase] = value
                    except (ValueError, TypeError):
                        result[phase] = default
                else:
                    result[phase] = default

            return result
        except Exception as e:
            self.logger.error(f"Error loading max position age settings: {e}")
            return {
                "neutral": 90.0,
                "uptrend": 72.0,
                "downtrend": 48.0,
                "ranging_at_support": 48.0,
                "ranging_at_resistance": 16.0,
                "volatile": 32.0
            }

    def _load_phase_exit_preferences(self) -> Dict[str, Dict[str, float]]:
        try:
            default_values = {
                "neutral": {
                    "profit_factor": 1.15,
                    "duration_factor": 1.25
                },
                "ranging_at_resistance": {
                    "profit_factor": 0.7,
                    "duration_factor": 0.5
                },
                "uptrend": {
                    "profit_factor": 1.1,
                    "duration_factor": 1.1
                },
                "downtrend": {
                    "profit_factor": 0.9,
                    "duration_factor": 0.8
                }
            }

            config_values = self.config.get("time_management", "phase_exit_preferences", {})

            result = {}
            for phase, default_dict in default_values.items():
                if phase in config_values:
                    result[phase] = {}
                    for key, default in default_dict.items():
                        if key in config_values[phase]:
                            try:
                                value = float(config_values[phase][key])
                                if value <= 0:
                                    raise ValueError("Value must be positive")
                                result[phase][key] = value
                            except (ValueError, TypeError):
                                result[phase][key] = default
                        else:
                            result[phase][key] = default
                else:
                    result[phase] = default_dict.copy()

            return result
        except Exception as e:
            self.logger.error(f"Error loading phase exit preferences: {e}")
            return {
                "neutral": {
                    "profit_factor": 1.15,
                    "duration_factor": 1.25
                },
                "ranging_at_resistance": {
                    "profit_factor": 0.7,
                    "duration_factor": 0.5
                },
                "uptrend": {
                    "profit_factor": 1.1,
                    "duration_factor": 1.1
                },
                "downtrend": {
                    "profit_factor": 0.9,
                    "duration_factor": 0.8
                }
            }

    def _load_time_based_risk_factors(self) -> Dict[int, float]:
        try:
            default_values = {
                4: 1.4,
                8: 1.2,
                16: 1.05,
                24: 0.85,
                48: 0.65,
                72: 0.45
            }

            config_values = self.config.get("time_management", "time_based_risk_factors", {})

            result = {}
            for hours, default in default_values.items():
                str_hours = str(hours)
                if str_hours in config_values:
                    try:
                        value = float(config_values[str_hours])
                        if value <= 0:
                            raise ValueError("Value must be positive")
                        result[hours] = value
                    except (ValueError, TypeError):
                        result[hours] = default
                else:
                    result[hours] = default

            return result
        except Exception as e:
            self.logger.error(f"Error loading time-based risk factors: {e}")
            return {
                4: 1.4,
                8: 1.2,
                16: 1.05,
                24: 0.85,
                48: 0.65,
                72: 0.45
            }

    def evaluate_time_based_exit(self, position: Dict[str, Any],
                                 current_price: float,
                                 current_time: datetime,
                                 market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self._validate_position_data(position, current_price, current_time):
                return {"exit": False, "reason": "InvalidPositionData"}

            entry_time = position['entry_time']
            direction = position['direction']
            entry_price = float(position['entry_price'])
            current_stop = float(position['current_stop_loss'])
            signal_type = position.get('entry_signal', '')

            trade_duration = self._calculate_trade_duration(entry_time, current_time)

            market_phase = market_conditions.get('market_phase', 'neutral')
            volatility = float(market_conditions.get('volatility', 0.5))
            momentum = float(market_conditions.get('momentum', 0.0))
            ensemble_score = float(position.get('ensemble_score', 0.5))

            pnl_pct = self._calculate_pnl_percentage(direction, entry_price, current_price)

            phase_settings = self.phase_exit_preferences.get(
                market_phase,
                {"profit_factor": 1.0, "duration_factor": 1.0}
            )
            profit_factor = phase_settings["profit_factor"]
            duration_factor = phase_settings["duration_factor"]

            max_duration = self._calculate_max_duration(market_phase, volatility, pnl_pct)
            max_duration *= duration_factor

            quick_profit_threshold = self.profit_targets["quick"] * profit_factor * (1.0 - (0.15 * abs(momentum)))

            if pnl_pct > self.profit_targets["medium"] * 1.2:
                momentum_factor = 1.0 + min(0.4, max(-0.3, momentum))

                if (direction == 'long' and momentum > 0.15) or (direction == 'short' and momentum < -0.15):
                    if trade_duration > self.min_profit_taking_hours:
                        return {
                            "exit": True,
                            "reason": "HighProfitMomentumExit",
                            "exit_price": current_price
                        }

            if self._should_take_quick_profit(trade_duration, pnl_pct, market_conditions, ensemble_score):
                return {
                    "exit": True,
                    "reason": "QuickProfitTaken",
                    "exit_price": current_price
                }

            if trade_duration > max_duration:
                return {
                    "exit": True,
                    "reason": f"MaxDurationReached_{market_phase}",
                    "exit_price": current_price
                }

            stagnant_threshold = 0.0008
            if market_phase == "ranging_at_resistance" or market_phase == "ranging_at_support":
                stagnant_threshold = 0.0006

            if trade_duration > self.stagnant_exit_hours and abs(pnl_pct) < stagnant_threshold:
                return {
                    "exit": True,
                    "reason": "StagnantPosition",
                    "exit_price": current_price
                }

            small_profit_threshold = self.profit_targets["small"] * profit_factor * 0.7
            small_profit_duration = self.small_profit_exit_hours

            if market_phase == "neutral":
                small_profit_duration *= 0.75

            if trade_duration > small_profit_duration and 0 < pnl_pct < small_profit_threshold:
                return {
                    "exit": True,
                    "reason": "SmallProfitLongTimeBasedExit",
                    "exit_price": current_price
                }

            if trade_duration > self.min_profit_taking_hours and pnl_pct > self.profit_targets["micro"] * profit_factor:
                momentum_threshold = -0.15 if direction == 'long' else 0.15

                if (direction == 'long' and momentum < momentum_threshold) or \
                        (direction == 'short' and momentum > -momentum_threshold):
                    return {
                        "exit": True,
                        "reason": "MomentumBasedExit",
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

        except Exception as e:
            self.logger.error(f"Error in evaluate_time_based_exit: {e}")
            return {"exit": False, "reason": "EvaluationError"}

    def _validate_position_data(self, position: Dict[str, Any],
                                current_price: float,
                                current_time: datetime) -> bool:
        required_fields = ['entry_time', 'direction', 'entry_price', 'current_stop_loss']
        for field in required_fields:
            if field not in position:
                self.logger.warning(f"Missing required field in position: {field}")
                return False

        entry_time = position.get('entry_time')
        if not entry_time or not isinstance(entry_time, datetime):
            self.logger.warning(f"Invalid entry_time: {entry_time}")
            return False

        direction = position.get('direction')
        if direction not in ['long', 'short']:
            self.logger.warning(f"Invalid direction: {direction}")
            return False

        try:
            float(position['entry_price'])
            float(position['current_stop_loss'])
            float(current_price)
        except (ValueError, TypeError):
            self.logger.warning("Invalid price values in position data")
            return False

        if current_time <= entry_time:
            self.logger.warning(f"Current time {current_time} not after entry time {entry_time}")
            return False

        return True

    def _calculate_trade_duration(self, entry_time: datetime, current_time: datetime) -> float:
        if not isinstance(entry_time, datetime) or not isinstance(current_time, datetime):
            self.logger.warning(f"Invalid datetime objects: entry={entry_time}, current={current_time}")
            return 0.0

        try:
            if entry_time.tzinfo is not None and current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=entry_time.tzinfo)
            elif entry_time.tzinfo is None and current_time.tzinfo is not None:
                entry_time = entry_time.replace(tzinfo=current_time.tzinfo)
        except Exception as e:
            self.logger.warning(f"Timezone handling error: {e}")

        try:
            duration_seconds = (current_time - entry_time).total_seconds()
            if duration_seconds < 0:
                self.logger.warning(f"Negative duration calculated: {duration_seconds}s")
                return 0.0
            return duration_seconds / 3600
        except Exception as e:
            self.logger.error(f"Error calculating trade duration: {e}")
            return 0.0

    def _calculate_pnl_percentage(self, direction: str, entry_price: float, current_price: float) -> float:
        try:
            if direction == 'long':
                return (current_price / entry_price) - 1
            elif direction == 'short':
                return (entry_price / current_price) - 1
            else:
                self.logger.warning(f"Invalid direction for PnL calculation: {direction}")
                return 0.0
        except (ZeroDivisionError, TypeError):
            self.logger.warning(f"Error calculating PnL percentage: entry={entry_price}, current={current_price}")
            return 0.0

    def _calculate_max_duration(self, market_phase: str, volatility: float, pnl_pct: float) -> float:
        base_duration = self.max_position_age.get(market_phase, 60)

        if volatility > 0.7:
            volatility_factor = 0.65
        elif volatility < 0.3:
            volatility_factor = 1.5
        else:
            volatility_factor = 1.0

        if pnl_pct > self.profit_targets["medium"]:
            profit_factor = 2.2
        elif pnl_pct > self.profit_targets["small"]:
            profit_factor = 1.7
        elif pnl_pct > 0:
            profit_factor = 1.3
        elif pnl_pct < -0.012:
            profit_factor = 0.6
        else:
            profit_factor = 1.0

        adjusted_duration = base_duration * volatility_factor * profit_factor

        return min(120, max(8, adjusted_duration))

    def _should_take_quick_profit(self, trade_duration: float, pnl_pct: float,
                                  market_conditions: Dict[str, Any],
                                  ensemble_score: float = 0.5) -> bool:
        # Minimum holding period before taking profits
        if trade_duration < self.min_profit_taking_hours * 0.7:  # Slightly increased from 0.6
            return False

        volatility = float(market_conditions.get('volatility', 0.5))
        market_phase = market_conditions.get('market_phase', 'neutral')
        momentum = float(market_conditions.get('momentum', 0))

        phase_profit_factor = self.phase_exit_preferences.get(
            market_phase, {"profit_factor": 1.0}
        )["profit_factor"]

        # Further reduced profit thresholds based on backtesting results
        quick_profit = self.profit_targets["quick"] * phase_profit_factor * 0.8  # Reduced from 0.85
        small_profit = self.profit_targets["small"] * phase_profit_factor * 0.8  # Reduced from 0.85

        # More aggressive exit in ranging at resistance
        if market_phase == "ranging_at_resistance" and pnl_pct > quick_profit * 0.4:  # Reduced from 0.5
            return True

        # More aggressive exit in high volatility
        if volatility > 0.4 and pnl_pct > quick_profit * 0.7:  # Reduced from 0.75
            return True

        # More aggressive exit on weakening momentum
        if momentum < -0.15 and pnl_pct > quick_profit * 0.7:  # Increased sensitivity (was -0.2)
            return True

        # Even faster exit in neutral markets
        if market_phase == "neutral" and pnl_pct > small_profit * 0.6 and trade_duration > 3.0:  # Reduced from 0.7 and 3.5
            return True

        # Lower general profit target threshold
        if pnl_pct > small_profit * 0.8 and trade_duration > 2.0:  # Reduced from 0.9 and 2.5
            return True

        # More aggressive medium profit taking
        if pnl_pct > self.profit_targets["medium"] * 0.7 and trade_duration > 1.0:  # Reduced from 0.75 and 1.2
            return True

        return False

    def _calculate_time_adjusted_stop(self, direction: str, entry_price: float,
                                      current_price: float, trade_duration: float,
                                      current_stop: float,
                                      market_conditions: Dict[str, Any]) -> Optional[float]:
        # Only adjust stops after minimum time has passed
        if trade_duration < 1.0:  # Reduced from 1.5
            return None

        volatility = float(market_conditions.get('volatility', 0.5))
        atr = float(market_conditions.get('atr', current_price * 0.01))
        market_phase = market_conditions.get('market_phase', 'neutral')
        momentum = float(market_conditions.get('momentum', 0))

        if direction == 'long':
            pnl_pct = (current_price / entry_price) - 1
        else:
            pnl_pct = (entry_price / current_price) - 1

        # Only adjust stops when in profit - MORE CONSERVATIVE
        min_profit_for_adjustment = 0.003  # Only adjust after 0.3% profit
        if pnl_pct <= min_profit_for_adjustment:
            return None

        # Get a more conservative risk factor for time-based adjustments
        risk_factor = self._get_time_based_risk_factor(trade_duration) * 1.2  # 20% more conservative

        if volatility > 0.7:
            vol_adjusted_mult = 1.5  # Increased from 1.35 (wider stops)
        elif volatility < 0.3:
            vol_adjusted_mult = 0.8  # Increased from 0.7 (still tighter, but less extreme)
        else:
            vol_adjusted_mult = 1.0 + (volatility - 0.5) * 0.6  # Less aggressive than before (was 0.8)

        phase_factor = self._get_phase_adjustment_factor(market_phase, direction)
        momentum_factor = self._get_momentum_adjustment_factor(direction, momentum)
        base_atr_mult = self._get_profit_tier_atr_multiple(pnl_pct)

        # More conservative overall adjustment
        adjusted_atr_mult = base_atr_mult * vol_adjusted_mult * phase_factor * momentum_factor * risk_factor * 1.1  # Added 10% extra buffer

        if direction == 'long':
            new_stop = current_price - (adjusted_atr_mult * atr)
            new_stop = self._apply_breakeven_protection_long(new_stop, entry_price, pnl_pct, atr)
            return new_stop if new_stop > current_stop else None
        else:
            new_stop = current_price + (adjusted_atr_mult * atr)
            new_stop = self._apply_breakeven_protection_short(new_stop, entry_price, pnl_pct, atr)
            return new_stop if new_stop < current_stop else None

    # Updated profit tier ATR multiple - more conservative
    def _get_profit_tier_atr_multiple(self, pnl_pct: float) -> float:
        if pnl_pct > self.profit_targets["large"]:
            return 1.3  # Increased from 1.1
        elif pnl_pct > self.profit_targets["medium"]:
            return 1.8  # Increased from 1.5
        elif pnl_pct > self.profit_targets["small"]:
            return 2.2  # Increased from 1.9
        elif pnl_pct > self.profit_targets["quick"]:
            return 2.8  # Increased from 2.4
        elif pnl_pct > self.profit_targets["micro"]:
            return 3.5  # Increased from 3.0
        else:
            return 4.0  # Increased from 3.3

    def _get_time_based_risk_factor(self, trade_duration: float) -> float:
        for hours, factor in sorted(self.time_based_risk_factors.items()):
            if trade_duration <= hours:
                return factor
        return self.time_based_risk_factors.get(max(self.time_based_risk_factors.keys()), 0.4)

    def _get_phase_adjustment_factor(self, market_phase: str, direction: str) -> float:
        if market_phase == "neutral":
            return 0.8
        elif market_phase == "ranging_at_resistance" and direction == "long":
            return 0.6
        elif market_phase == "uptrend" and direction == "long":
            return 0.9
        elif market_phase == "downtrend" and direction == "short":
            return 0.9
        else:
            return 1.0

    def _get_momentum_adjustment_factor(self, direction: str, momentum: float) -> float:
        if direction == 'long' and momentum < -0.2:
            return 0.7
        elif direction == 'short' and momentum > 0.2:
            return 0.7
        elif direction == 'long' and momentum > 0.3:
            return 1.1
        elif direction == 'short' and momentum < -0.3:
            return 1.1
        else:
            return 1.0

    def _get_profit_tier_atr_multiple(self, pnl_pct: float) -> float:
        if pnl_pct > self.profit_targets["large"]:
            return 1.1
        elif pnl_pct > self.profit_targets["medium"]:
            return 1.5
        elif pnl_pct > self.profit_targets["small"]:
            return 1.9
        elif pnl_pct > self.profit_targets["quick"]:
            return 2.4
        elif pnl_pct > self.profit_targets["micro"]:
            return 3.0
        else:
            return 3.3

    def _apply_breakeven_protection_long(self, new_stop: float, entry_price: float,
                                         pnl_pct: float, atr: float) -> float:
        # Only move to breakeven at higher profit levels
        if pnl_pct > 0.02:  # Increased from 0.015
            breakeven_buffer = atr * 0.3  # Increased buffer from 0.25
            return max(new_stop, entry_price + breakeven_buffer)
        elif pnl_pct > 0.01:  # Increased from 0.008
            buffer_factor = min(0.15, pnl_pct * 6)  # Reduced buffer (was 0.2, pnl*10)
            return max(new_stop, entry_price + (buffer_factor * atr))
        else:
            return new_stop

    def _apply_breakeven_protection_short(self, new_stop: float, entry_price: float,
                                          pnl_pct: float, atr: float) -> float:
        # Only move to breakeven at higher profit levels
        if pnl_pct > 0.02:  # Increased from 0.015
            breakeven_buffer = atr * 0.3  # Increased buffer from 0.25
            return min(new_stop, entry_price - breakeven_buffer)
        elif pnl_pct > 0.01:  # Increased from 0.008
            buffer_factor = min(0.15, pnl_pct * 6)  # Reduced buffer (was 0.2, pnl*10)
            return min(new_stop, entry_price - (buffer_factor * atr))
        else:
            return new_stop

    def update_duration_stats(self, trade_result: Dict[str, Any]) -> None:
        try:
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

        except Exception as e:
            self.logger.error(f"Error updating duration stats: {e}")

    def _update_log_performance(self, exit_reason: str, pnl: float,
                                duration: float, market_phase: str) -> None:
        try:
            if exit_reason not in self.log_performance_data:
                self.log_performance_data[exit_reason] = self._create_empty_stats()

            self._update_stats(self.log_performance_data[exit_reason], pnl, duration)

            key = f"{exit_reason}_{market_phase}"
            if key not in self.log_performance_data:
                self.log_performance_data[key] = self._create_empty_stats()

            self._update_stats(self.log_performance_data[key], pnl, duration)

        except Exception as e:
            self.logger.warning(f"Error updating performance log: {e}")

    def _create_empty_stats(self) -> Dict[str, Any]:
        return {
            "count": 0,
            "win_count": 0,
            "total_pnl": 0,
            "total_duration": 0,
            "avg_pnl": 0,
            "avg_duration": 0,
            "win_rate": 0
        }

    def _update_stats(self, stats: Dict[str, Any], pnl: float, duration: float) -> None:
        stats["count"] += 1
        stats["total_pnl"] += pnl
        stats["total_duration"] += duration

        if pnl > 0:
            stats["win_count"] += 1

        if stats["count"] > 0:
            stats["avg_pnl"] = stats["total_pnl"] / stats["count"]
            stats["avg_duration"] = stats["total_duration"] / stats["count"]
            stats["win_rate"] = stats["win_count"] / stats["count"]

    def get_exit_performance_stats(self) -> Dict[str, Any]:
        return {
            "exit_stats": self.log_performance_data,
            "optimal_durations": self.calculate_optimal_trade_duration(
                self.holding_period_stats["winning_trades"] + self.holding_period_stats["losing_trades"]
            )
        }

    def calculate_optimal_trade_duration(self, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trade_history or len(trade_history) < 10:
            return {
                "optimal_hold_time": 24,
                "confidence": "low",
                "data_points": len(trade_history) if trade_history else 0
            }

        try:
            trade_durations = []
            profitable_durations = []
            phase_durations = {}

            for trade in trade_history:
                duration_hours = trade.get('duration')
                pnl = trade.get('pnl', 0)
                market_phase = trade.get('market_phase', 'neutral')

                if duration_hours is None or not isinstance(duration_hours, (int, float)):
                    continue

                trade_durations.append(duration_hours)

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

            optimal_duration = min(60, max(8, avg_profitable_duration * 1.1))

            percentiles = {
                "p25": np.percentile(profitable_durations, 25),
                "p50": np.percentile(profitable_durations, 50),
                "p75": np.percentile(profitable_durations, 75)
            }

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

        except Exception as e:
            self.logger.error(f"Error calculating optimal trade duration: {e}")
            return {
                "optimal_hold_time": 24,
                "confidence": "low",
                "data_points": 0,
                "error": str(e)
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

        try:
            durations = [t["duration"] for t in winning_trades]
            p25 = np.percentile(durations, 25)
            p50 = np.percentile(durations, 50)
            p75 = np.percentile(durations, 75)

            self.min_profit_taking_hours = max(1.5, min(3, p25 * 0.5))
            self.small_profit_exit_hours = max(12, min(24, p50 * 1.2))
            self.stagnant_exit_hours = max(16, min(32, p75 * 0.85))

            exit_reason_performance = self._analyze_exit_reason_performance()

            duration_performance = self._analyze_duration_performance()

            optimal_by_winrate = max(duration_performance.get("win_rates", {}).items(),
                                     key=lambda x: x[1],
                                     default=(24, 0))[0]

            optimal_by_pnl = max(duration_performance.get("pnl", {}).items(),
                                 key=lambda x: x[1],
                                 default=(24, 0))[0]

            self.max_trade_duration_hours = (optimal_by_winrate * 0.4 + optimal_by_pnl * 0.6)

            self._update_market_phase_settings()

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
                    "exit_reason_performance": exit_reason_performance,
                    "duration_performance": duration_performance,
                    "optimal_by_winrate": optimal_by_winrate,
                    "optimal_by_pnl": optimal_by_pnl
                }
            }

        except Exception as e:
            self.logger.error(f"Error optimizing time parameters: {e}")
            return {
                "status": "error",
                "message": f"Error during optimization: {e}"
            }

    def _analyze_exit_reason_performance(self) -> Dict[str, Dict[str, Any]]:
        result = {}

        for trade in self.holding_period_stats["winning_trades"]:
            reason = trade["exit_reason"]

            if reason not in result:
                result[reason] = {
                    "count": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                    "durations": []
                }

            perf = result[reason]
            perf["count"] += 1
            perf["total_pnl"] += trade["pnl"]
            perf["durations"].append(trade["duration"])
            perf["avg_pnl"] = perf["total_pnl"] / perf["count"]

        return result

    def _analyze_duration_performance(self) -> Dict[str, Dict[int, float]]:
        win_rates_by_duration = {}
        pnl_by_duration = {}

        all_trades = self.holding_period_stats["winning_trades"] + self.holding_period_stats["losing_trades"]

        for duration_bin in [3, 6, 9, 12, 16, 24, 36, 48]:
            trades_in_bin = [t for t in all_trades if t["duration"] <= duration_bin]

            if not trades_in_bin:
                continue

            win_count = sum(1 for t in trades_in_bin if t["pnl"] > 0)
            total_pnl = sum(t["pnl"] for t in trades_in_bin)

            win_rates_by_duration[duration_bin] = win_count / len(trades_in_bin)
            pnl_by_duration[duration_bin] = total_pnl / len(trades_in_bin)

        return {
            "win_rates": win_rates_by_duration,
            "pnl": pnl_by_duration
        }

    def _update_market_phase_settings(self) -> None:
        all_trades = self.holding_period_stats["winning_trades"] + self.holding_period_stats["losing_trades"]

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

        for phase, stats in market_phase_stats.items():
            if stats["count"] < 5:
                continue

            if phase not in self.phase_exit_preferences:
                self.phase_exit_preferences[phase] = {"profit_factor": 1.0, "duration_factor": 1.0}

            if stats["win_rate"] > 0.65 and stats["avg_pnl"] > 0:
                self.phase_exit_preferences[phase]["profit_factor"] = 1.25
                self.phase_exit_preferences[phase]["duration_factor"] = 1.25
            elif stats["win_rate"] < 0.45 or stats["avg_pnl"] < 0:
                self.phase_exit_preferences[phase]["profit_factor"] = 0.6
                self.phase_exit_preferences[phase]["duration_factor"] = 0.5

            if len(stats["durations"]) >= 5:
                optimal_phase_duration = np.percentile(stats["durations"], 65)
                self.max_position_age[phase] = max(12, min(100, optimal_phase_duration * 1.6))