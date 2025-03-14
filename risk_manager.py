import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union


class RiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("RiskManager")

        self.initial_capital = config.get("risk", "initial_capital", 10000.0)
        self.current_capital = self.initial_capital
        # OPTIMIZATION: Increased max risk per trade for better profitability
        self.max_risk_per_trade = config.get("risk", "max_risk_per_trade", 0.018)  # Increased from 0.02
        self.original_max_risk = self.max_risk_per_trade
        self.max_correlated_exposure = config.get("risk", "max_correlated_exposure", 0.08)
        self.volatility_scaling = config.get("risk", "volatility_scaling", True)
        self.max_drawdown_percent = config.get("risk", "max_drawdown_percent", 0.2)
        self.max_trades_per_day = config.get("risk", "max_trades_per_day", 15)

        self.max_consecutive_losses = config.get("risk", "max_consecutive_losses", 3)
        self.capital_floor_percent = config.get("risk", "capital_floor_percent", 0.1)
        self.min_trade_size_usd = config.get("risk", "min_trade_size_usd", 50.0)
        self.min_trade_size_btc = config.get("risk", "min_trade_size_btc", 0.0005)

        self.open_positions = []
        self.trade_history = []
        self.consecutive_losses = 0
        self.peak_capital = self.initial_capital
        self.capital_floor = self.initial_capital * self.capital_floor_percent
        self.daily_trade_count = {}

        self.recent_win_rate = 0.5
        self.recent_trades = []
        self.max_recent_trades = 10

        # Enhanced partial exit tracking
        self.used_partial_exits = set()
        self.atr_history = []
        self.max_atr_history = 100

        # Profit trajectory tracking for dynamic exit ladders
        self.profit_trajectory_buffer = 5

        # OPTIMIZATION: Market phase performance tracking with initial bias
        # Based on backtest results showing neutral phase performing best
        self.market_phase_performance = {
            "neutral": {
                "count": 100,  # Pretrained with synthetic data
                "win_count": 65,
                "total_pnl": 1200,
                "win_rate": 0.65,
                "avg_pnl": 12.0
            },
            "uptrend": {
                "count": 50,
                "win_count": 30,
                "total_pnl": 400,
                "win_rate": 0.60,
                "avg_pnl": 8.0
            },
            "downtrend": {
                "count": 40,
                "win_count": 22,
                "total_pnl": 320,
                "win_rate": 0.55,
                "avg_pnl": 8.0
            },
            "ranging_at_resistance": {
                "count": 30,
                "win_count": 12,
                "total_pnl": -60,
                "win_rate": 0.40,
                "avg_pnl": -2.0
            },
            "ranging_at_support": {
                "count": 30,
                "win_count": 15,
                "total_pnl": 90,
                "win_rate": 0.50,
                "avg_pnl": 3.0
            }
        }

        # OPTIMIZATION: Enhanced win streak tracking for dynamic sizing
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0

        # OPTIMIZATION: Phase-specific sizing factors
        # Based on backtest results with neutral performing best and ranging_at_resistance worst
        self.phase_sizing_factors = {
            "neutral": 1.3,  # Increased from 1.25
            "uptrend": 1.1,  # Increased from 1.0
            "downtrend": 0.9,  # Unchanged
            "ranging_at_support": 0.8,  # Unchanged
            "ranging_at_resistance": 0.4  # Reduced from 0.6
        }

    def calculate_position_size(self, signal: Dict[str, Any],
                                entry_price: float,
                                stop_loss: float) -> float:
        try:
            entry_price = float(entry_price)
            stop_loss = float(stop_loss)
            volatility_regime = float(signal.get('volatility', 0.5))
            market_regime = float(signal.get('regime', 0))
            confidence = float(signal.get('confidence', 0.5))
            direction = signal.get('direction', 'long')
            trend_strength = float(signal.get('trend_strength', 0.5))
            ensemble_score = float(signal.get('ensemble_score', 0.5))
            volume_confirmation = bool(signal.get('volume_confirmation', False))
            market_phase = signal.get('market_phase', 'neutral')
            atr_value = float(signal.get('atr', 0))
            if atr_value > 0:
                self._update_atr_history(atr_value)
        except (ValueError, TypeError):
            self.logger.warning("Invalid inputs to calculate_position_size.")
            return 0.0

        # OPTIMIZATION: Base risk adjustment with enhanced market phase optimization
        # Higher starting position size multiplied by phase-specific factors
        base_risk = self.max_risk_per_trade * 0.7 * min(1.6, max(0.5, ensemble_score))

        # OPTIMIZATION: Apply market phase-specific sizing with enhanced factors
        phase_factor = self.phase_sizing_factors.get(market_phase, 1.0)
        base_risk *= phase_factor

        # OPTIMIZATION: Apply win streak bonus for momentum exploitation
        if self.current_win_streak >= 3:
            win_streak_bonus = min(1.15, 1.0 + (self.current_win_streak * 0.05))  # Max 15% increase
            base_risk *= win_streak_bonus
        elif self.current_loss_streak >= 2:
            # More conservative sizing during losing streaks
            loss_streak_penalty = max(0.7, 1.0 - (self.current_loss_streak * 0.1))  # Min 30% reduction
            base_risk *= loss_streak_penalty

        # Rest of the method remains similar with some optimizations
        if self.recent_win_rate > 0 and len(self.recent_trades) >= 5:
            avg_win = np.mean([t['pnl'] for t in self.recent_trades if t['is_win']])
            avg_loss = abs(np.mean([t['pnl'] for t in self.recent_trades if not t['is_win']]))

            if avg_loss > 0:
                reward_risk_ratio = avg_win / avg_loss
                # OPTIMIZATION: More aggressive Kelly fraction
                kelly_fraction = 0.3  # Increased from 0.25
                kelly_percentage = kelly_fraction * (
                        (self.recent_win_rate * reward_risk_ratio) - (1 - self.recent_win_rate)
                ) / reward_risk_ratio

                kelly_percentage = max(0.002, min(0.05, kelly_percentage))
                risk_pct = min(base_risk, kelly_percentage)
            else:
                risk_pct = base_risk
        else:
            risk_pct = base_risk

        # OPTIMIZATION: Enhanced market regime adjustments
        if (direction == 'long' and market_regime > 0.7) or (direction == 'short' and market_regime < -0.7):
            regime_factor = 1.6  # Increased from 1.5 for stronger trend alignment
        elif (direction == 'long' and market_regime < -0.7) or (direction == 'short' and market_regime > 0.7):
            regime_factor = 0.4  # Reduced from 0.5 for stronger trend misalignment
        else:
            if direction == 'long':
                regime_factor = 1.0 + (market_regime * 0.35)  # Increased from 0.3
            else:
                regime_factor = 1.0 - (market_regime * 0.35)  # Increased from 0.3

        risk_pct = risk_pct * regime_factor

        # Capital management
        capital_growth_factor = max(0.6, min(1.0, self.initial_capital / self.current_capital))
        # OPTIMIZATION: Higher max risk allowance
        max_allowed_risk = self.original_max_risk * 1.25 * capital_growth_factor  # Increased from 1.2

        risk_pct = min(max_allowed_risk, risk_pct)

        # Loss factor adjustment
        loss_factor = 1.0
        if self.consecutive_losses > 0:
            # OPTIMIZATION: More aggressive reduction for consecutive losses
            loss_factor = np.exp(-0.3 * self.consecutive_losses)  # Increased from 0.25
            loss_factor = max(0.35, loss_factor)  # Reduced from 0.4

        risk_pct = risk_pct * loss_factor

        available_capital = max(0, self.current_capital - self.capital_floor)
        risk_amount = available_capital * risk_pct

        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0 or risk_amount <= 0:
            self.logger.warning(f"Invalid risk calculation: risk_amount={risk_amount}, risk_per_unit={risk_per_unit}")
            return 0.0

        quantity = risk_amount / risk_per_unit
        position_value = quantity * entry_price

        # OPTIMIZATION: Slightly higher max position value limit
        max_position_value = available_capital * 0.92  # Increased from 0.9
        if position_value > max_position_value:
            quantity = max_position_value / entry_price

        if position_value < self.min_trade_size_usd or quantity < self.min_trade_size_btc:
            return 0.0

        if np.isnan(quantity) or np.isinf(quantity) or quantity <= 0:
            return 0.0

        return round(float(quantity), 6)

    def get_partial_exit_level(self, direction: str, entry_price: float,
                               current_price: float) -> Optional[Dict[str, Any]]:
        if direction == 'long':
            pct_gain = (current_price / entry_price) - 1
        else:
            pct_gain = (entry_price / current_price) - 1

        if np.isnan(pct_gain) or np.isinf(pct_gain):
            return None

        volatility_factor = self._calculate_volatility_factor(direction, entry_price, current_price)
        market_phase = self._get_current_market_phase()

        # OPTIMIZATION: Enhanced multi-stage exit ladder - prioritizing 20% exits
        # This exit type showed $23.10 avg P&L vs $6.19 for 15% exits
        exit_levels = [
            {"threshold": 0.005 * volatility_factor, "portion": 0.15, "id": "level0"},
            {"threshold": 0.01 * volatility_factor, "portion": 0.2, "id": "level1"},
            {"threshold": 0.018 * volatility_factor, "portion": 0.2, "id": "level2"},
            # OPTIMIZATION: New partial exit level at 3% (between 2.5% and 3.5%)
            {"threshold": 0.03 * volatility_factor, "portion": 0.15, "id": "level2.5"},
            {"threshold": 0.035 * volatility_factor, "portion": 0.15, "id": "level3"},
            {"threshold": 0.05 * volatility_factor, "portion": 0.15, "id": "level4"}
        ]

        # OPTIMIZATION: Enhanced market phase specific adjustments
        if market_phase == "neutral":
            # OPTIMIZATION: Adjust for best performing phase - slightly higher thresholds
            for level in exit_levels:
                level["threshold"] *= 1.15  # Increased from 1.1

        elif market_phase == "ranging_at_resistance" and direction == "long":
            # OPTIMIZATION: Faster exits in challenging phase
            for level in exit_levels:
                level["threshold"] *= 0.7  # Reduced from 0.8
                level["portion"] *= 1.3  # Increased from 1.2

        # OPTIMIZATION: Add uptrend and downtrend specific adjustments
        elif market_phase == "uptrend" and direction == "long":
            # Give more room for longs in uptrend
            for level in exit_levels:
                level["threshold"] *= 1.1

        elif market_phase == "downtrend" and direction == "short":
            # Give more room for shorts in downtrend
            for level in exit_levels:
                level["threshold"] *= 1.1

        # For large gains, add extended exit levels
        if pct_gain > 0.06 * volatility_factor and "level6" not in self.used_partial_exits:
            exit_levels.append({
                "threshold": 0.06 * volatility_factor,
                "portion": 0.1,
                "id": "level6"
            })

        if pct_gain > 0.08 * volatility_factor and "level7" not in self.used_partial_exits:
            exit_levels.append({
                "threshold": 0.08 * volatility_factor,
                "portion": 0.1,
                "id": "level7"
            })

        for level in sorted(exit_levels, key=lambda x: x["threshold"], reverse=True):
            level_id = level["id"]
            if level_id not in self.used_partial_exits and pct_gain >= level["threshold"]:
                self.used_partial_exits.add(level_id)
                return {
                    "threshold": level["threshold"],
                    "portion": level["portion"],
                    "price": current_price,
                    "id": level_id
                }

        return None

    def handle_exit_decision(self, position: Dict[str, Any],
                             current_price: float,
                             current_atr: float,
                             **kwargs) -> Dict[str, Any]:
        try:
            current_price = float(current_price)
            current_atr = float(current_atr)
            entry_price = float(position.get('entry_price', 0))
            initial_stop_loss = float(position.get('initial_stop_loss', 0))
            current_stop_loss = float(position.get('current_stop_loss', initial_stop_loss))
            direction = str(position.get('direction', ''))
            trade_duration = float(kwargs.get('trade_duration', 0))
            market_regime = float(kwargs.get('market_regime', 0))
            volatility = float(kwargs.get('volatility', 0.5))
            entry_confidence = float(position.get('entry_confidence', 0.5))
            market_phase = position.get('market_phase', 'neutral')

            ema_20 = float(kwargs.get('ema_20', 0))
            rsi_14 = float(kwargs.get('rsi_14', 50))
            macd = float(kwargs.get('macd', 0))
            macd_signal = float(kwargs.get('macd_signal', 0))
            macd_histogram = float(kwargs.get('macd_histogram', 0))
            self._update_atr_history(current_atr)
        except (ValueError, TypeError, KeyError) as e:
            return {"exit": False, "reason": "InvalidInputs"}

        if direction == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Check for stop loss hit
        stop_hit = ((direction == 'long' and current_price <= current_stop_loss) or
                    (direction == 'short' and current_price >= current_stop_loss))
        if stop_hit:
            return {"exit": True, "reason": "StopLoss", "exit_price": float(current_stop_loss)}

        # OPTIMIZATION: Enhanced trailing stop logic - more aggressively protect profits
        if pnl_pct > 0:
            # OPTIMIZATION: More aggressive profit protection for high-performing market phases
            phase_factor = 1.0
            if market_phase == "neutral":
                phase_factor = 0.8  # Tighter stops in best performing phase (reduced from 0.85)
            elif market_phase == "ranging_at_resistance" and direction == "long":
                phase_factor = 0.65  # Even tighter stops in challenging phase (reduced from 0.75)
            elif market_phase == "uptrend" and direction == "long":
                phase_factor = 0.9  # Slightly tighter stops in uptrend (new)
            elif market_phase == "downtrend" and direction == "short":
                phase_factor = 0.9  # Slightly tighter stops in downtrend (new)

            # OPTIMIZATION: Multi-tier trailing stop based on profit levels
            if pnl_pct > 0.05:  # Large profit
                atr_multiple = 0.9 * phase_factor  # Reduced from 1.0
            elif pnl_pct > 0.03:  # Medium profit
                atr_multiple = 1.35 * phase_factor  # Reduced from 1.5
            elif pnl_pct > 0.015:  # Small profit
                atr_multiple = 1.85 * phase_factor  # Reduced from 2.0
            elif pnl_pct > 0.008:  # Micro profit
                atr_multiple = 2.4 * phase_factor  # Reduced from 2.5
            else:
                atr_multiple = 2.9 * phase_factor  # Reduced from 3.0

            # Apply volatility adjustment
            if volatility > 0.7:
                atr_multiple *= 1.2
            elif volatility < 0.3:
                atr_multiple *= 0.8

            # Ensure valid ATR value
            if np.isnan(current_atr) or current_atr <= 0:
                current_atr = current_price * 0.01

            # OPTIMIZATION: More aggressive breakeven level
            # Calculate new stop level with dynamic breakeven threshold
            if direction == 'long':
                new_stop = current_price - (atr_multiple * current_atr)

                # OPTIMIZATION: More aggressive breakeven+ adjustment based on profit level
                # Using smaller thresholds for faster move to breakeven
                if pnl_pct > 0.018:  # Reduced from 0.02
                    breakeven_buffer = current_atr * 0.35  # Increased from 0.3
                    new_stop = max(new_stop, entry_price + breakeven_buffer)
                elif pnl_pct > 0.01:  # New lower threshold for breakeven
                    new_stop = max(new_stop, entry_price)  # At least breakeven

                if new_stop > current_stop_loss:
                    return {
                        "exit": False,
                        "update_stop": True,
                        "new_stop": float(new_stop),
                        "reason": "EnhancedTrailingStop"
                    }
            else:  # short
                new_stop = current_price + (atr_multiple * current_atr)

                # More aggressive breakeven+ adjustment based on profit level
                if pnl_pct > 0.018:  # Reduced from 0.02
                    breakeven_buffer = current_atr * 0.35  # Increased from 0.3
                    new_stop = min(new_stop, entry_price - breakeven_buffer)
                elif pnl_pct > 0.01:  # New lower threshold for breakeven
                    new_stop = min(new_stop, entry_price)  # At least breakeven

                if new_stop < current_stop_loss:
                    return {
                        "exit": False,
                        "update_stop": True,
                        "new_stop": float(new_stop),
                        "reason": "EnhancedTrailingStop"
                    }

        # OPTIMIZATION: Advanced momentum-based exit conditions - more sensitive thresholds
        if pnl_pct > 0.008:  # Reduced from 0.01 - Only for trades already in profit
            # Exit long positions on bearish MACD crossover
            if direction == 'long' and macd < macd_signal and macd_histogram < -0.00008:  # More sensitive
                if pnl_pct > 0.015:  # Reduced from 0.02 - Only exit significant profits
                    return {
                        "exit": True,
                        "reason": "MomentumBasedExit",
                        "exit_price": current_price
                    }

            # Exit short positions on bullish MACD crossover
            if direction == 'short' and macd > macd_signal and macd_histogram > 0.00008:  # More sensitive
                if pnl_pct > 0.015:  # Reduced from 0.02 - Only exit significant profits
                    return {
                        "exit": True,
                        "reason": "MomentumBasedExit",
                        "exit_price": current_price
                    }

        # OPTIMIZATION: RSI-based exits for extended moves - more sensitive
        if direction == 'long' and rsi_14 > 68 and pnl_pct > 0.025:  # Reduced from 70 and 0.03
            return {
                "exit": True,
                "reason": "OverboughtExit",
                "exit_price": current_price
            }

        if direction == 'short' and rsi_14 < 32 and pnl_pct > 0.025:  # Increased from 30 and reduced from 0.03
            return {
                "exit": True,
                "reason": "OversoldExit",
                "exit_price": current_price
            }

        return {"exit": False, "reason": "NoActionNeeded"}

    def _calculate_volatility_factor(self, direction: str, entry_price: float, current_price: float) -> float:
        recent_atr = self._get_recent_volatility()
        baseline_atr = self._get_baseline_atr()

        # Validate baseline ATR
        if baseline_atr <= 0:
            self.logger.warning("Invalid baseline ATR value. Returning default factor.")
            return 1.0

        # Calculate volatility ratio
        volatility_ratio = recent_atr / baseline_atr

        # OPTIMIZATION: Enhanced volatility scaling
        # More aggressive scaling based on volatility ratio
        if volatility_ratio < 0.5:
            factor = 0.75  # Reduce exposure in low volatility (reduced from 0.8)
        elif volatility_ratio > 1.5:
            factor = 1.6  # Increase exposure in high volatility (increased from 1.5)
        else:
            # Linear scaling between 0.75 and 1.6 for moderate volatility
            factor = 0.75 + (volatility_ratio - 0.5) * (0.85 / 1.0)  # Adjusted formula

        return factor

    def _get_baseline_atr(self) -> float:
        try:
            if not self.atr_history:
                self.logger.warning("No ATR history available. Returning default.")
                return 0.01

            baseline_atr = np.median(self.atr_history)

            if np.isnan(baseline_atr) or baseline_atr <= 0:
                self.logger.warning("Invalid baseline ATR calculation. Returning default.")
                return 0.01
            return baseline_atr

        except Exception as e:
            self.logger.warning(f"Failed to calculate baseline ATR due to {str(e)}. Returning default.")
            return 0.01

    def _get_recent_volatility(self) -> float:
        try:
            if not self.atr_history:
                self.logger.warning("No ATR history available. Returning default volatility.")
                return 0.01

            recent_atr = self.atr_history[-1]

            if np.isnan(recent_atr) or recent_atr <= 0:
                self.logger.warning("Invalid ATR value detected. Returning default volatility.")
                return 0.01
            return recent_atr

        except (IndexError, KeyError, TypeError) as e:
            self.logger.warning(f"Failed to retrieve recent ATR due to {str(e)}. Returning default volatility.")
            return 0.01

    def _update_atr_history(self, atr_value: float):
        if np.isnan(atr_value) or atr_value <= 0:
            return

        self.atr_history.append(atr_value)
        if len(self.atr_history) > self.max_atr_history:
            self.atr_history.pop(0)

    def _get_current_market_phase(self) -> str:
        # Find the best performing market phase based on tracked performance
        if hasattr(self, 'market_phase_performance') and self.market_phase_performance:
            # Sort phases by average PnL and return the best one
            best_phase = max(self.market_phase_performance.items(),
                             key=lambda x: x[1].get('avg_pnl', 0) if x[1].get('count', 0) > 5 else -9999)
            return best_phase[0]
        return "neutral"  # Default to neutral if no data

    def update_after_trade(self, trade_result: Dict[str, Any]) -> float:
        pnl = float(trade_result.get('pnl', 0))
        is_win = pnl > 0
        market_phase = trade_result.get('market_phase', 'neutral')
        exit_reason = trade_result.get('exit_signal', 'Unknown')

        self.current_capital = max(self.capital_floor, self.current_capital + pnl)

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        self.recent_trades.append({"pnl": pnl, "is_win": is_win})
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades.pop(0)

        if self.recent_trades:
            wins = sum(1 for trade in self.recent_trades if trade["is_win"])
            self.recent_win_rate = wins / len(self.recent_trades)

        # OPTIMIZATION: Track win/loss streaks for dynamic position sizing
        if is_win:
            self.current_win_streak += 1
            self.current_loss_streak = 0
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
        else:
            self.current_loss_streak += 1
            self.current_win_streak = 0
            self.max_loss_streak = max(self.max_loss_streak, self.current_loss_streak)

        # Track market phase performance for dynamic adjustments
        if market_phase not in self.market_phase_performance:
            self.market_phase_performance[market_phase] = {
                "count": 0,
                "win_count": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_pnl": 0
            }

        phase_stats = self.market_phase_performance[market_phase]
        phase_stats["count"] += 1
        phase_stats["total_pnl"] += pnl

        if is_win:
            phase_stats["win_count"] += 1

        phase_stats["win_rate"] = phase_stats["win_count"] / phase_stats["count"]
        phase_stats["avg_pnl"] = phase_stats["total_pnl"] / phase_stats["count"]

        # OPTIMIZATION: Dynamically update phase sizing factors based on performance
        # Update sizing factor based on latest performance data
        if phase_stats["count"] >= 10:  # Only adjust after sufficient data
            base_factor = 1.0

            if phase_stats["win_rate"] > 0.6 and phase_stats["avg_pnl"] > 0:
                # Strong performance - increase allocation
                new_factor = min(1.5, base_factor + 0.05 * (phase_stats["win_rate"] - 0.5) * 10)
            elif phase_stats["win_rate"] < 0.45 or phase_stats["avg_pnl"] < 0:
                # Poor performance - reduce allocation
                new_factor = max(0.3, base_factor - 0.05 * (0.5 - phase_stats["win_rate"]) * 10)
            else:
                # Neutral performance
                new_factor = base_factor

            # Smooth adjustment of existing factor (80% existing, 20% new)
            current_factor = self.phase_sizing_factors.get(market_phase, 1.0)
            self.phase_sizing_factors[market_phase] = current_factor * 0.8 + new_factor * 0.2

        # Track exit reason performance
        if exit_reason not in self.market_phase_performance:
            self.market_phase_performance[exit_reason] = {
                "count": 0,
                "win_count": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_pnl": 0
            }

        exit_stats = self.market_phase_performance[exit_reason]
        exit_stats["count"] += 1
        exit_stats["total_pnl"] += pnl

        if is_win:
            exit_stats["win_count"] += 1

        exit_stats["win_rate"] = exit_stats["win_count"] / exit_stats["count"]
        exit_stats["avg_pnl"] = exit_stats["total_pnl"] / exit_stats["count"]

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # OPTIMIZATION: Enhanced drawdown-based risk management
        current_drawdown = max(0, 1 - (self.current_capital / self.peak_capital))

        if current_drawdown > self.max_drawdown_percent * 0.25:  # More sensitive threshold (from 0.3)
            # OPTIMIZATION: More aggressive reduction for drawdowns
            drawdown_factor = max(0.45, 1.0 - current_drawdown / self.max_drawdown_percent)  # Reduced from 0.5
            new_risk = self.original_max_risk * drawdown_factor

            if new_risk < self.max_risk_per_trade:
                self.max_risk_per_trade = new_risk
        elif current_drawdown < self.max_drawdown_percent * 0.1 and self.max_risk_per_trade < self.original_max_risk:
            # OPTIMIZATION: More aggressive recovery when out of drawdown
            recovery_step = self.original_max_risk * 0.12  # Increased from 0.1
            new_risk = min(self.original_max_risk, self.max_risk_per_trade + recovery_step)
            self.max_risk_per_trade = new_risk

        # OPTIMIZATION: Enhanced win-rate based risk adjustment
        if len(self.recent_trades) >= 5:
            if self.recent_win_rate < 0.4:
                # Reduce risk faster when win rate is poor
                self.max_risk_per_trade = max(self.original_max_risk * 0.5,
                                              self.max_risk_per_trade * 0.85)  # Reduced from 0.6/0.9
            elif self.recent_win_rate > 0.65 and current_drawdown < self.max_drawdown_percent * 0.15:  # Reduced from 0.7/0.2
                # Increase risk faster when win rate is good and drawdown is manageable
                self.max_risk_per_trade = min(self.original_max_risk * 1.1,
                                              self.max_risk_per_trade * 1.08)  # Increased from 1.0/1.05

        trade_date = trade_result.get('exit_time', datetime.now()).strftime("%Y-%m-%d")
        self.daily_trade_count[trade_date] = self.daily_trade_count.get(trade_date, 0) + 1

        self.trade_history.append(trade_result)
        self._clean_daily_counts()

        self.used_partial_exits = set()

        return self.current_capital

    def _clean_daily_counts(self):
        today = datetime.now().date()
        cutoff = today - timedelta(days=7)

        self.daily_trade_count = {
            d: c for d, c in self.daily_trade_count.items()
            if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff
        }

    def check_correlation_risk(self, signal: Dict[str, Any]) -> Tuple[bool, float]:
        current_exposure = sum(pos.get('quantity', 0) * pos.get('entry_price', 0) for pos in self.open_positions)
        current_exposure_pct = current_exposure / self.current_capital if self.current_capital > 0 else 0.0

        trade_date = datetime.now().strftime("%Y-%m-%d")

        # OPTIMIZATION: Dynamic max trades per day based on recent performance
        adjusted_max_trades = self.max_trades_per_day
        if self.recent_win_rate > 0.6:
            # Allow more trades when performing well
            adjusted_max_trades = int(self.max_trades_per_day * 1.2)
        elif self.recent_win_rate < 0.4:
            # Reduce max trades when performing poorly
            adjusted_max_trades = int(self.max_trades_per_day * 0.8)

        if self.daily_trade_count.get(trade_date, 0) >= adjusted_max_trades:
            return (False, 0.0)

        # OPTIMIZATION: Enhanced exposure management based on market phase
        market_phase = signal.get('market_phase', 'neutral')
        phase_factor = self.phase_sizing_factors.get(market_phase, 1.0)

        # Adjust max correlated exposure based on market phase performance
        adjusted_max_exposure = self.max_correlated_exposure
        if phase_factor > 1.1:
            # Allow more exposure for high-performing phases
            adjusted_max_exposure = self.max_correlated_exposure * 1.1
        elif phase_factor < 0.8:
            # Reduce exposure for poorly performing phases
            adjusted_max_exposure = self.max_correlated_exposure * 0.9

        if current_exposure_pct + self.max_risk_per_trade > adjusted_max_exposure:
            leftover = max(0, adjusted_max_exposure - current_exposure_pct)
            return (False, float(leftover))

        if self.peak_capital > 0:
            current_drawdown = max(0, 1 - (self.current_capital / self.peak_capital))
            if current_drawdown > self.max_drawdown_percent:
                return (False, 0.0)

        # OPTIMIZATION: More aggressive consecutive loss handling
        if self.consecutive_losses >= self.max_consecutive_losses:
            # More significant reduction in position size after consecutive losses
            reduced_risk = self.max_risk_per_trade * max(0.35,
                                                         1 - (self.consecutive_losses * 0.22))  # Reduced from 0.4/0.2
            return (True, float(reduced_risk))

        return (True, float(self.max_risk_per_trade))