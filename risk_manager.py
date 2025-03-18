import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque


class RiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("RiskManager")

        self.initial_capital = config.get("risk", "initial_capital", 10000.0)
        self.current_capital = self.initial_capital
        self.max_risk_per_trade = config.get("risk", "max_risk_per_trade", 0.025)
        self.original_max_risk = self.max_risk_per_trade

        self.max_correlated_exposure = config.get("risk", "max_correlated_exposure", 0.12)
        self.volatility_scaling = config.get("risk", "volatility_scaling", True)
        self.max_drawdown_percent = config.get("risk", "max_drawdown_percent", 0.2)

        self.max_trades_per_day = config.get("risk", "max_trades_per_day", 20)
        self.max_consecutive_losses = config.get("risk", "max_consecutive_losses", 4)
        self.capital_floor_percent = config.get("risk", "capital_floor_percent", 0.1)
        self.min_trade_size_usd = config.get("risk", "min_trade_size_usd", 50.0)
        self.min_trade_size_btc = config.get("risk", "min_trade_size_btc", 0.0005)

        self.open_positions = []
        self.trade_history = deque(maxlen=50)
        self.recent_trades = deque(maxlen=10)
        self.consecutive_losses = 0
        self.peak_capital = self.initial_capital
        self.capital_floor = self.initial_capital * self.capital_floor_percent
        self.daily_trade_count = {}

        self.recent_win_rate = 0.5
        self.max_recent_trades = 10

        self.used_partial_exits = set()
        self.atr_history = deque(maxlen=100)
        self.rolling_pnl = deque(maxlen=20)
        self.volatility_trend = deque(maxlen=10)
        self.market_regime_history = deque(maxlen=20)

        self.market_phase_performance = self._initialize_market_phase_performance()

        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0

        self.phase_sizing_factors = {
            "neutral": 1.7,
            "uptrend": 1.3,
            "downtrend": 1.25,
            "ranging_at_support": 1.2,
            "ranging_at_resistance": 0.5
        }

        self.max_risk_reward_ratio = 3.0
        self.base_atr_multiplier = 2.0

        # Advanced risk management parameters
        self.adaptive_risk_adjustment = True
        self.drawdown_based_sizing = True
        self.momentum_adjusted_risk = True
        self.volatility_sensitive_stops = True
        self.profit_taking_acceleration = True

        # Enhanced trailing stop factors by profit tier
        self._profit_tier_factors = {
            1: 2.8,  # Minimal profit
            2: 2.2,  # Small profit
            3: 1.8,  # Medium profit
            4: 1.4,  # Good profit
            5: 1.0  # Substantial profit
        }

        # Phase-specific trailing factors
        self._phase_trailing_factors = {
            "uptrend": 1.1,
            "downtrend": 1.0,
            "neutral": 0.9,
            "ranging_at_support": 0.8,
            "ranging_at_resistance": 0.7
        }

        # Historical performance by volatility
        self.volatility_performance = {
            "low": {"win_rate": 0.55, "avg_pnl": 0.015},
            "medium": {"win_rate": 0.60, "avg_pnl": 0.020},
            "high": {"win_rate": 0.45, "avg_pnl": 0.025}
        }

    def _initialize_market_phase_performance(self) -> Dict[str, Dict[str, Any]]:
        return {
            "neutral": {
                "count": 100,
                "win_count": 65,
                "total_pnl": 1200,
                "win_rate": 0.65,
                "avg_pnl": 12.0
            },
            "ranging_at_support": {
                "count": 30,
                "win_count": 18,
                "total_pnl": 150,
                "win_rate": 0.60,
                "avg_pnl": 5.0
            },
            "downtrend": {
                "count": 40,
                "win_count": 25,
                "total_pnl": 350,
                "win_rate": 0.62,
                "avg_pnl": 8.75
            },
            "uptrend": {
                "count": 50,
                "win_count": 28,
                "total_pnl": 300,
                "win_rate": 0.56,
                "avg_pnl": 6.0
            },
            "ranging_at_resistance": {
                "count": 30,
                "win_count": 10,
                "total_pnl": -90,
                "win_rate": 0.33,
                "avg_pnl": -3.0
            }
        }

    def calculate_position_size(self, signal: Dict[str, Any],
                                entry_price: float,
                                stop_loss: float) -> float:
        try:
            entry_price = float(entry_price)
            stop_loss = float(stop_loss)
            direction = signal.get('direction', 'long')
            market_phase = signal.get('market_phase', 'neutral')

            if entry_price <= 0 or stop_loss <= 0:
                self.logger.warning("Invalid entry price or stop loss")
                return 0.0

            if (direction == 'long' and stop_loss >= entry_price) or \
                    (direction == 'short' and stop_loss <= entry_price):
                self.logger.warning(f"Stop loss {stop_loss} on wrong side of entry {entry_price} for {direction}")
                return 0.0

            volatility_regime = float(signal.get('volatility', 0.5))
            market_regime = float(signal.get('regime', 0))
            momentum = float(signal.get('momentum', 0))
            atr_value = float(signal.get('atr', 0))

            if atr_value > 0:
                self._update_atr_history(atr_value)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error in position size calculation: {e}")
            return 0.0

        # Enhanced confidence weighting with non-linear scaling
        confidence = float(signal.get('confidence', 0.5))
        ensemble_score = float(signal.get('ensemble_score', 0.5))

        # Sigmoid-based confidence scaling for smoother risk adjustment
        conf_scaling = 1.0 / (1.0 + np.exp(-10 * (confidence - 0.003)))
        ensemble_scaling = 1.0 / (1.0 + np.exp(-8 * (ensemble_score - 0.3)))

        # Combined model confidence factor with non-linear scaling
        model_confidence = (conf_scaling * 0.4) + (ensemble_scaling * 0.6)
        base_risk = self.max_risk_per_trade * min(1.5, max(0.3, model_confidence * 1.8))

        # Advanced market phase adjustment
        phase_factor = self._get_adaptive_phase_factor(
            market_phase,
            direction,
            self.current_win_streak,
            self.current_loss_streak
        )

        # Adaptive volatility-based position sizing
        vol_factor = self._calculate_volatility_factor(volatility_regime, direction)

        # Calculate momentum-based adjustment
        momentum_factor = self._calculate_momentum_factor(market_regime, momentum, direction)

        # Create position size distribution based on success metrics
        position_sizes = []
        weights = []

        # Try multiple position sizes and weight by expected outcome
        for size_factor in [0.7, 1.0, 1.3]:
            size = base_risk * phase_factor * vol_factor * momentum_factor * size_factor
            position_sizes.append(size)

            # Weight based on historical performance with similar parameters
            performance_weight = self._evaluate_position_weight(size, market_phase, volatility_regime)
            weights.append(performance_weight)

        # Use weighted average for final position size
        total_weight = sum(weights) or 1.0
        final_risk = sum(p * w for p, w in zip(position_sizes, weights)) / total_weight

        # Win/loss streak adjustment
        if self.current_win_streak >= 2:
            win_streak_bonus = min(1.6, 1.0 + (self.current_win_streak * 0.12))
            final_risk *= win_streak_bonus
        elif self.current_loss_streak >= 2:
            loss_streak_penalty = max(0.7, 1.0 - (self.current_loss_streak * 0.1))
            final_risk *= loss_streak_penalty

        # Adaptive drawdown-based adjustment
        if self.drawdown_based_sizing:
            current_drawdown = max(0, 1 - (self.current_capital / self.peak_capital))
            drawdown_severity = current_drawdown / self.max_drawdown_percent
            if current_drawdown > self.max_drawdown_percent * 0.25:
                drawdown_factor = max(0.4, 1.0 - drawdown_severity * 1.4)
                final_risk *= drawdown_factor

        # Calculate dollar risk amount
        risk_amount = max(0, self.current_capital - self.capital_floor) * final_risk
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0 or risk_amount <= 0:
            self.logger.warning(f"Invalid risk calculation: risk_amount={risk_amount}, risk_per_unit={risk_per_unit}")
            return 0.0

        # Risk/reward optimization
        take_profit = signal.get('take_profit1', 0.0) if direction == 'long' else signal.get('take_profit', 0.0)
        if take_profit > 0:
            reward_per_unit = abs(take_profit - entry_price)
            risk_reward_ratio = reward_per_unit / max(risk_per_unit, 0.0001)

            # Scale position size based on risk/reward quality
            if risk_reward_ratio < self.max_risk_reward_ratio * 0.6:
                adjusted_risk_amount = risk_amount * (risk_reward_ratio / (self.max_risk_reward_ratio * 0.6))
                risk_amount = min(risk_amount, adjusted_risk_amount)
            elif risk_reward_ratio > self.max_risk_reward_ratio:
                risk_amount *= min(1.3, risk_reward_ratio / self.max_risk_reward_ratio)

        # Calculate final position size
        quantity = risk_amount / risk_per_unit
        position_value = quantity * entry_price

        # Safety checks
        available_capital = max(0, self.current_capital - self.capital_floor)
        max_position_value = available_capital * 0.92
        if position_value > max_position_value:
            quantity = max_position_value / entry_price

        # Minimum trade size checks
        if position_value < self.min_trade_size_usd or quantity < self.min_trade_size_btc:
            return 0.0

        if np.isnan(quantity) or np.isinf(quantity) or quantity <= 0:
            return 0.0

        return round(float(quantity), 6)

    def _get_adaptive_phase_factor(self, market_phase, direction, win_streak, loss_streak):
        """Get adaptive phase factor based on recent performance"""
        base_factor = self.phase_sizing_factors.get(market_phase, 1.0)

        # Dynamically adjust based on win/loss streaks
        streak_adjustment = 1.0
        if win_streak >= 3:
            streak_adjustment = 1.0 + (min(win_streak, 7) - 2) * 0.08
        elif loss_streak >= 2:
            streak_adjustment = 1.0 - (min(loss_streak, 5) - 1) * 0.12

        # Check if we have historical performance for this phase
        if market_phase in self.market_phase_performance:
            stats = self.market_phase_performance[market_phase]
            if stats.get('count', 0) >= 10:
                win_rate = stats.get('win_rate', 0.5)

                # Increase size for historically successful phases
                if win_rate > 0.6:
                    base_factor *= 1.2
                # Decrease size for historically unsuccessful phases
                elif win_rate < 0.4:
                    base_factor *= 0.7

        # Direction-specific adjustments
        if direction == 'long' and market_phase == 'uptrend':
            base_factor *= 1.1
        elif direction == 'short' and market_phase == 'downtrend':
            base_factor *= 1.1
        elif direction == 'long' and market_phase == 'ranging_at_resistance':
            base_factor *= 0.8
        elif direction == 'short' and market_phase == 'ranging_at_support':
            base_factor *= 0.8

        return base_factor * streak_adjustment

    def _evaluate_position_weight(self, size, market_phase, volatility_regime):
        """Evaluate the likelihood of success for a position size"""
        # Default weight
        weight = 1.0

        # Categorize volatility
        vol_category = "medium"
        if volatility_regime > 0.7:
            vol_category = "high"
        elif volatility_regime < 0.3:
            vol_category = "low"

        # Get historical performance data
        vol_performance = self.volatility_performance.get(vol_category, {"win_rate": 0.5, "avg_pnl": 0.01})

        # Get phase performance if available
        phase_performance = None
        if market_phase in self.market_phase_performance:
            stats = self.market_phase_performance[market_phase]
            if stats.get('count', 0) >= 5:
                phase_performance = {
                    "win_rate": stats.get('win_rate', 0.5),
                    "avg_pnl": stats.get('avg_pnl', 0) / 100  # Convert to percentage
                }

        # Calculate expected value based on historical performance
        vol_ev = vol_performance["win_rate"] * vol_performance["avg_pnl"] - (1 - vol_performance["win_rate"]) * 0.01

        # Adjust weight based on expected value
        weight *= (1.0 + vol_ev * 10)

        # Further adjust by phase performance if available
        if phase_performance:
            phase_ev = phase_performance["win_rate"] * phase_performance["avg_pnl"] - (
                        1 - phase_performance["win_rate"]) * 0.01
            weight *= (1.0 + phase_ev * 5)

        # Size factor - penalize extreme sizes
        size_factor = 1.0
        if size > self.max_risk_per_trade * 1.3:
            # Penalize oversized positions
            size_factor = 0.7
        elif size < self.max_risk_per_trade * 0.7:
            # Slightly penalize undersized positions
            size_factor = 0.9

        weight *= size_factor

        return max(0.1, weight)

    def _calculate_momentum_factor(self, market_regime: float, momentum: float, direction: str) -> float:
        # Align trade direction with market regime and momentum
        regime_aligned = (direction == 'long' and market_regime > 0.2) or \
                         (direction == 'short' and market_regime < -0.2)

        momentum_aligned = (direction == 'long' and momentum > 0.1) or \
                           (direction == 'short' and momentum < -0.1)

        base_factor = 1.0

        # Stronger factor when both regime and momentum align
        if regime_aligned and momentum_aligned:
            base_factor = 1.2
        # Decent factor when at least one aligns
        elif regime_aligned or momentum_aligned:
            base_factor = 1.1
        # Reduced factor when both are against position
        elif (direction == 'long' and market_regime < -0.3 and momentum < -0.2) or \
                (direction == 'short' and market_regime > 0.3 and momentum > 0.2):
            base_factor = 0.7

        # Scale based on strength of momentum
        momentum_strength = abs(momentum)
        momentum_scaling = 1.0 + (momentum_strength * 0.5)

        # Apply non-linear scaling for strong momentum
        if momentum_strength > 0.5:
            momentum_scaling = 1.0 + (0.25 + (momentum_strength - 0.5) * 0.8)

        # Adjust factor by momentum strength only when aligned
        if (direction == 'long' and momentum > 0) or (direction == 'short' and momentum < 0):
            base_factor *= min(1.5, momentum_scaling)

        return base_factor

    def _calculate_volatility_factor(self, volatility_regime: float, direction: str) -> float:
        if len(self.volatility_trend) < 3:
            # Default based on current volatility
            if volatility_regime > 0.7:
                return 0.8  # Reduce size in high volatility
            elif volatility_regime < 0.3:
                return 1.2  # Increase size in low volatility
            return 1.0

        # Convert deque to list for analysis
        volatility_trend_list = list(self.volatility_trend)

        # Detect if volatility is increasing or decreasing
        volatility_increasing = sum(1 for i in range(1, len(volatility_trend_list))
                                    if volatility_trend_list[i] > volatility_trend_list[i - 1])
        volatility_trend_up = volatility_increasing > len(volatility_trend_list) / 2

        # Base factor on current volatility level
        if volatility_regime > 0.7:  # High volatility
            base_factor = 0.8

            # Further reduce size if volatility is still increasing
            if volatility_trend_up:
                base_factor *= 0.9

            # Direction-specific adjustment (higher volatility favors shorts)
            if direction == 'short':
                base_factor *= 1.1

        elif volatility_regime < 0.3:  # Low volatility
            base_factor = 1.2

            # Further increase size if volatility is decreasing
            if not volatility_trend_up:
                base_factor *= 1.1

            # Direction-specific adjustment (lower volatility favors longs)
            if direction == 'long':
                base_factor *= 1.1

        else:  # Medium volatility
            base_factor = 1.0

            # Adjust based on trend
            if volatility_trend_up:
                base_factor *= 0.95  # Slightly reduce as volatility increases
            else:
                base_factor *= 1.05  # Slightly increase as volatility decreases

        # Scale based on historical performance in this volatility regime
        vol_category = "medium"
        if volatility_regime > 0.7:
            vol_category = "high"
        elif volatility_regime < 0.3:
            vol_category = "low"

        vol_performance = self.volatility_performance.get(vol_category, {"win_rate": 0.5})
        win_rate = vol_performance.get("win_rate", 0.5)

        # Adjust factor based on historical win rate
        win_rate_factor = 0.8 + (win_rate * 0.4)  # Scale from 0.8 to 1.2

        return base_factor * win_rate_factor

    def _get_adaptive_profit_threshold(self, market_phase, volatility):
        """Get adaptive minimum profit threshold for trailing stops"""
        # Base profit threshold
        base_threshold = 0.004  # 0.4%

        # Adjust by market phase
        if market_phase == "ranging_at_support" or market_phase == "ranging_at_resistance":
            base_threshold *= 0.8  # Lower threshold in ranging markets
        elif market_phase == "uptrend" or market_phase == "downtrend":
            base_threshold *= 1.1  # Higher threshold in trending markets

        # Adjust by volatility
        if volatility > 0.7:
            base_threshold *= 1.2  # Higher threshold in high volatility
        elif volatility < 0.3:
            base_threshold *= 0.9  # Lower threshold in low volatility

        return base_threshold

    def _get_regime_trailing_factor(self):
        """Calculate trailing stop factor based on regime analysis"""
        if len(self.market_regime_history) < 5:
            return 1.0

        recent_regimes = list(self.market_regime_history)[-5:]
        regime_trend = recent_regimes[-1] - recent_regimes[0]

        # If regime is strengthening in direction, use looser stops
        if abs(regime_trend) > 0.2:
            return 1.2
        # If regime is stable, use standard stops
        elif abs(regime_trend) < 0.1:
            return 1.0
        # If regime is weakening, use tighter stops
        else:
            return 0.9

    def handle_exit_decision(self, position: Dict[str, Any],
                             current_price: float,
                             current_atr: float,
                             **kwargs) -> Dict[str, Any]:
        try:
            if not self._validate_position_data(position):
                return {"exit": False, "reason": "InvalidPosition"}

            current_price = float(current_price)
            current_atr = float(current_atr)
            entry_price = float(position.get('entry_price', 0))
            initial_stop_loss = float(position.get('initial_stop_loss', 0))
            current_stop_loss = float(position.get('current_stop_loss', initial_stop_loss))
            direction = str(position.get('direction', ''))
            entry_time = position.get('entry_time')
            current_time = kwargs.get('current_time', datetime.now())

            if direction not in ['long', 'short']:
                return {"exit": False, "reason": "InvalidDirection"}

            trade_duration = float(kwargs.get('trade_duration', 0))
            market_regime = float(kwargs.get('market_regime', 0))
            volatility = float(kwargs.get('volatility', 0.5))
            market_phase = position.get('market_phase', 'neutral')

            # Track market regime for trend analysis
            self.market_regime_history.append(market_regime)

            rsi_14 = float(kwargs.get('rsi_14', 50))
            macd = float(kwargs.get('macd', 0))
            macd_signal = float(kwargs.get('macd_signal', 0))
            macd_histogram = float(kwargs.get('macd_histogram', 0))

            # Update ATR history for volatility tracking
            self._update_atr_history(current_atr)

            # Calculate PnL metrics
            if direction == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Enhanced trail stop logic for positions in profit
            if pnl_pct > 0 and self.volatility_sensitive_stops:
                trailing_stop = self._calculate_dynamic_trailing_stop(
                    direction, entry_price, current_price, current_stop_loss,
                    current_atr, pnl_pct, market_phase, volatility, trade_duration
                )

                if trailing_stop is not None:
                    return trailing_stop

            # Check if stop loss is hit
            stop_hit = ((direction == 'long' and current_price <= current_stop_loss) or
                        (direction == 'short' and current_price >= current_stop_loss))
            if stop_hit:
                return {"exit": True, "reason": "StopLoss", "exit_price": float(current_stop_loss)}

            # Check for momentum-based exit signals when in profit
            if pnl_pct > 0.006:
                if self._should_exit_on_momentum(direction, macd, macd_signal, macd_histogram, pnl_pct, market_regime):
                    return {
                        "exit": True,
                        "reason": "MomentumBasedExit",
                        "exit_price": current_price
                    }

                if self._should_exit_on_rsi(direction, rsi_14, pnl_pct, market_phase):
                    reason = "OverboughtExit" if direction == 'long' else "OversoldExit"
                    return {
                        "exit": True,
                        "reason": reason,
                        "exit_price": current_price
                    }

            # Time-based exit for positions not making progress
            if trade_duration > self._get_max_holdtime(market_phase, volatility):
                if pnl_pct < 0.002:  # Not making meaningful progress
                    return {
                        "exit": True,
                        "reason": f"MaxDurationReached_{market_phase}",
                        "exit_price": current_price
                    }

            # Check for trend reversal exit in profit
            if pnl_pct > 0.01 and len(self.market_regime_history) >= 5:
                if self._detect_trend_reversal(direction, market_regime):
                    return {
                        "exit": True,
                        "reason": "TrendReversalExit",
                        "exit_price": current_price
                    }

            return {"exit": False, "reason": "NoActionNeeded"}
        except Exception as e:
            self.logger.error(f"Error in handle_exit_decision: {e}")
            return {"exit": False, "reason": "ExitDecisionError"}

    def _calculate_dynamic_trailing_stop(self, direction: str, entry_price: float,
                                         current_price: float, current_stop: float,
                                         atr_value: float, pnl_pct: float, market_phase: str,
                                         volatility: float, trade_duration: float) -> Optional[Dict[str, Any]]:
        """Calculate dynamic trailing stop with advanced adaptations"""

        # Only use trailing after sufficient profit
        min_profit_threshold = self._get_adaptive_profit_threshold(market_phase, volatility)

        if pnl_pct < min_profit_threshold:
            return None

        # Profit-based stop adjustment factor (more aggressive with more profit)
        if pnl_pct > 0.05:  # Substantial profit
            profit_tier = 5
        elif pnl_pct > 0.03:
            profit_tier = 4
        elif pnl_pct > 0.02:
            profit_tier = 3
        elif pnl_pct > 0.01:
            profit_tier = 2
        else:
            profit_tier = 1

        # Get base trailing factor from profit lookup table
        base_factor = self._profit_tier_factors[profit_tier]

        # Apply market-phase specific adjustment
        phase_factor = self._phase_trailing_factors.get(market_phase, 1.0)

        # Apply volatility adjustment (wider for higher volatility)
        vol_factor = 0.8 + (volatility * 0.5)

        # Regime-based adjustment
        regime_factor = self._get_regime_trailing_factor()

        # Time-based adjustment (tighten over time)
        time_factor = max(0.7, 1.0 - (min(trade_duration, 36) / 72))

        # Calculate combined adjustment
        dynamic_factor = base_factor * phase_factor * vol_factor * regime_factor * time_factor

        # Calculate new stop level with sophisticated rules
        if direction == 'long':
            # Calculate exponential tightening based on profit level
            tightening = 1.0 - np.exp(-2 * pnl_pct)
            base_distance = dynamic_factor * atr_value
            adjusted_distance = base_distance * (1.0 - (tightening * 0.6))

            new_stop = current_price - adjusted_distance

            # Apply breakeven protection with tiered levels
            if pnl_pct > 0.03:
                # Lock in 50% of profits minimum
                profit_lock = entry_price + ((current_price - entry_price) * 0.5)
                new_stop = max(new_stop, profit_lock)
            elif pnl_pct > 0.015:
                # At least breakeven plus some buffer
                new_stop = max(new_stop, entry_price + (0.3 * atr_value))
            elif pnl_pct > 0.008:
                # At least breakeven
                new_stop = max(new_stop, entry_price)

            # Only update if improving the stop
            if new_stop > current_stop:
                return {
                    "exit": False,
                    "update_stop": True,
                    "new_stop": float(new_stop),
                    "reason": "DynamicTrailingStop"
                }
        else:  # Short direction
            # Calculate exponential tightening based on profit level
            tightening = 1.0 - np.exp(-2 * pnl_pct)
            base_distance = dynamic_factor * atr_value
            adjusted_distance = base_distance * (1.0 - (tightening * 0.6))

            new_stop = current_price + adjusted_distance

            # Apply breakeven protection with tiered levels
            if pnl_pct > 0.03:
                # Lock in 50% of profits minimum
                profit_lock = entry_price - ((entry_price - current_price) * 0.5)
                new_stop = min(new_stop, profit_lock)
            elif pnl_pct > 0.015:
                # At least breakeven plus some buffer
                new_stop = min(new_stop, entry_price - (0.3 * atr_value))
            elif pnl_pct > 0.008:
                # At least breakeven
                new_stop = min(new_stop, entry_price)

            # Only update if improving the stop
            if new_stop < current_stop:
                return {
                    "exit": False,
                    "update_stop": True,
                    "new_stop": float(new_stop),
                    "reason": "DynamicTrailingStop"
                }

        return None

    def _get_max_holdtime(self, market_phase: str, volatility: float) -> float:
        """Calculate maximum hold time based on market conditions"""
        base_holdtime = {
            "neutral": 18.0,
            "uptrend": 24.0,
            "downtrend": 18.0,
            "ranging_at_support": 16.0,
            "ranging_at_resistance": 12.0,
        }.get(market_phase, 20.0)

        # Adjust for volatility
        if volatility > 0.7:
            return base_holdtime * 0.8  # Shorter during high volatility
        elif volatility < 0.3:
            return base_holdtime * 1.2  # Longer during low volatility

        return base_holdtime

    def _detect_trend_reversal(self, direction: str, current_regime: float) -> bool:
        """Detect if market trend is reversing against our position"""
        if len(self.market_regime_history) < 5:
            return False

        # Convert deque to list before slicing
        recent_regimes = list(self.market_regime_history)[-5:]
        regime_trend = recent_regimes[-1] - recent_regimes[0]

        # Check for directional shift against our position
        if direction == 'long' and regime_trend < -0.2 and current_regime < 0:
            return True  # Trend turning bearish against our long
        elif direction == 'short' and regime_trend > 0.2 and current_regime > 0:
            return True  # Trend turning bullish against our short

        return False

    def _validate_position_data(self, position: Dict[str, Any]) -> bool:
        required_fields = ['entry_price', 'direction', 'initial_stop_loss', 'current_stop_loss']
        for field in required_fields:
            if field not in position:
                self.logger.warning(f"Missing required field in position: {field}")
                return False

        try:
            float(position['entry_price'])
            float(position['initial_stop_loss'])
            float(position['current_stop_loss'])

            if position['direction'] not in ['long', 'short']:
                self.logger.warning(f"Invalid direction: {position['direction']}")
                return False

            return True
        except (ValueError, TypeError):
            self.logger.warning("Invalid numeric value in position data")
            return False

    def _should_exit_on_momentum(self, direction: str, macd: float,
                                 macd_signal: float, macd_histogram: float,
                                 pnl_pct: float, market_regime: float) -> bool:
        # Scale based on the profit level
        confidence_scale = min(1.0, pnl_pct * 20)  # Higher with more profit

        # More conservative exit on strong trends in our favor
        regime_aligned = (direction == 'long' and market_regime > 0.5) or (
                direction == 'short' and market_regime < -0.5)
        threshold = -0.00004 if regime_aligned else -0.00002

        if direction == 'long':
            # Look for bearish momentum in macd
            if macd < macd_signal and macd_histogram < threshold * confidence_scale:
                return True
        else:
            # Look for bullish momentum in macd
            if macd > macd_signal and macd_histogram > -threshold * confidence_scale:
                return True

        # Check for momentum divergence
        if self._detect_momentum_divergence(direction, macd_histogram, pnl_pct):
            return True

        return False

    def _detect_momentum_divergence(self, direction: str, macd_histogram: float, pnl_pct: float) -> bool:
        """Detect price-momentum divergence as exit signal"""
        # Only check for significant profits
        if pnl_pct < 0.02:
            return False

        if direction == 'long' and macd_histogram < 0 and abs(macd_histogram) > 0.00006:
            return True  # Price up but momentum down
        elif direction == 'short' and macd_histogram > 0 and abs(macd_histogram) > 0.00006:
            return True  # Price down but momentum up

        return False

    def _should_exit_on_rsi(self, direction: str, rsi_14: float, pnl_pct: float, market_phase: str) -> bool:
        # Adjust exit thresholds based on market phase
        if market_phase == "ranging_at_resistance":
            overbought_threshold = 60  # More sensitive at resistance
            oversold_threshold = 35
        elif market_phase == "ranging_at_support":
            overbought_threshold = 65
            oversold_threshold = 30  # More sensitive at support
        else:
            overbought_threshold = 70
            oversold_threshold = 30

        # Scale thresholds based on profit
        if pnl_pct > 0.03:
            # More sensitive when in good profit
            overbought_threshold -= 8
            oversold_threshold += 8

        if direction == 'long' and rsi_14 > overbought_threshold:
            return True
        elif direction == 'short' and rsi_14 < oversold_threshold:
            return True

        return False

    def _get_baseline_atr(self) -> float:
        try:
            if not self.atr_history:
                return 0.01

            baseline_atr = np.median(self.atr_history)

            if np.isnan(baseline_atr) or baseline_atr <= 0:
                return 0.01

            return baseline_atr
        except Exception as e:
            self.logger.warning(f"Error calculating baseline ATR: {e}")
            return 0.01

    def _get_recent_volatility(self) -> float:
        try:
            if not self.atr_history:
                return 0.01

            recent_atr = self.atr_history[-1]

            if np.isnan(recent_atr) or recent_atr <= 0:
                return 0.01

            return recent_atr
        except Exception as e:
            self.logger.warning(f"Error retrieving recent volatility: {e}")
            return 0.01

    def _update_atr_history(self, atr_value: float) -> None:
        if np.isnan(atr_value) or atr_value <= 0:
            return

        self.atr_history.append(atr_value)

    def _get_current_market_phase(self) -> str:
        if not self.market_phase_performance:
            return "neutral"

        best_phase = max(
            ((phase, data) for phase, data in self.market_phase_performance.items()
             if data.get('count', 0) > 5),
            key=lambda x: x[1].get('avg_pnl', -999),
            default=("neutral", {})
        )[0]

        return best_phase

    def get_partial_exit_level(self, direction: str, entry_price: float,
                               current_price: float) -> Optional[Dict[str, Any]]:
        try:
            if direction not in ['long', 'short']:
                self.logger.warning(f"Invalid direction: {direction}")
                return None

            if direction == 'long':
                pct_gain = (current_price / entry_price) - 1
            else:
                pct_gain = (entry_price / current_price) - 1

            if np.isnan(pct_gain) or np.isinf(pct_gain) or pct_gain <= 0:
                return None

            volatility_factor = self._calculate_volatility_factor_advanced()
            market_phase = self._get_current_market_phase()

            # More aggressive partial exit strategy
            has_quick_exits = "level0" in self.used_partial_exits or "level1" in self.used_partial_exits
            acceleration_factor = 1.0

            if self.profit_taking_acceleration and has_quick_exits:
                # Accelerate profit taking if we've already taken some profit
                acceleration_factor = 0.75  # More aggressive than before (was 0.85)

            # Optimized exit levels based on profitable data analysis
            exit_levels = [
                {"threshold": 0.0025 * volatility_factor, "portion": 0.12, "id": "level0"},  # Earlier first exit
                {"threshold": 0.005 * volatility_factor, "portion": 0.15, "id": "level1"},  # Earlier second exit
                {"threshold": 0.01 * volatility_factor, "portion": 0.18, "id": "level2"},  # Increased portion
                {"threshold": 0.015 * volatility_factor, "portion": 0.20, "id": "level3"},
                # Increased from 0.18 to 0.20
                {"threshold": 0.025 * volatility_factor, "portion": 0.18, "id": "level4"},
                # Increased from 0.15 to 0.18
                {"threshold": 0.035 * volatility_factor, "portion": 0.15, "id": "level5"}  # Slightly increased
            ]

            # Adjust thresholds based on market phase
            for level in exit_levels:
                level["threshold"] *= acceleration_factor

                if market_phase == "neutral":
                    level["threshold"] *= 0.9  # More aggressive (was 1.2)
                    level["portion"] *= 1.2  # Larger portion in neutral market
                elif market_phase == "ranging_at_resistance" and direction == "long":
                    level["threshold"] *= 0.5  # Much more aggressive (was 0.65)
                    level["portion"] *= 1.5  # Larger portion at resistance
                elif market_phase == "downtrend" and direction == "long":
                    level["threshold"] *= 0.6  # More aggressive for longs in downtrend
                    level["portion"] *= 1.4  # Larger portion
                elif market_phase == "uptrend" and direction == "long":
                    level["threshold"] *= 1.0  # Keep thresholds but increase portion
                    level["portion"] *= 1.2

            # Add opportunistic higher exit levels
            if pct_gain > 0.045 * volatility_factor and "level6" not in self.used_partial_exits:
                exit_levels.append({
                    "threshold": 0.045 * volatility_factor,
                    "portion": 0.15,  # Increased from 0.1
                    "id": "level6"
                })

            if pct_gain > 0.06 * volatility_factor and "level7" not in self.used_partial_exits:
                exit_levels.append({
                    "threshold": 0.06 * volatility_factor,
                    "portion": 0.15,  # Increased from 0.1
                    "id": "level7"
                })

            # Special high gain exit with higher portion
            if pct_gain > 0.08 * volatility_factor and "level8" not in self.used_partial_exits:
                exit_levels.append({
                    "threshold": 0.08 * volatility_factor,
                    "portion": 0.25,  # Increased from 0.15
                    "id": "level8"
                })

            # Find best exit level that matches current gain
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
        except Exception as e:
            self.logger.error(f"Error in get_partial_exit_level: {e}")
            return None

    def _calculate_volatility_factor_advanced(self) -> float:
        """Calculate advanced volatility factor using ATR history and market conditions"""
        recent_atr = self._get_recent_volatility()
        baseline_atr = self._get_baseline_atr()

        if baseline_atr <= 0:
            return 1.0

        volatility_ratio = recent_atr / baseline_atr
        self.volatility_trend.append(volatility_ratio)

        # Analyze volatility trend
        if len(self.volatility_trend) >= 5:
            # Convert deque to list before slicing
            volatility_trend_list = list(self.volatility_trend)
            short_term = np.mean(volatility_trend_list[-3:])
            medium_term = np.mean(volatility_trend_list)

            # Rising volatility needs more conservative thresholds
            if short_term > medium_term * 1.2:
                return max(1.6, volatility_ratio * 1.2)
            # Falling volatility allows more aggressive thresholds
            elif short_term < medium_term * 0.8:
                return min(0.8, volatility_ratio * 0.9)

        # Standard calculation
        if volatility_ratio < 0.5:
            factor = 0.75
        elif volatility_ratio > 1.5:
            factor = 1.6
        else:
            factor = 0.75 + (volatility_ratio - 0.5) * (0.85 / 1.0)

        return factor

    def update_after_trade(self, trade_result: Dict[str, Any]) -> float:
        try:
            pnl = float(trade_result.get('pnl', 0))
            is_win = pnl > 0
            market_phase = str(trade_result.get('market_phase', 'neutral'))
            exit_reason = str(trade_result.get('exit_signal', 'Unknown'))

            self.current_capital = max(self.capital_floor, self.current_capital + pnl)
            self.peak_capital = max(self.peak_capital, self.current_capital)

            # Track rolling PnL
            self.rolling_pnl.append(pnl)

            # Update recent trades tracking
            self.recent_trades.append({"pnl": pnl, "is_win": is_win})
            if self.recent_trades:
                wins = sum(1 for trade in self.recent_trades if trade["is_win"])
                self.recent_win_rate = wins / len(self.recent_trades)

            # Update win/loss streaks
            if is_win:
                self.current_win_streak += 1
                self.current_loss_streak = 0
                self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
            else:
                self.current_loss_streak += 1
                self.current_win_streak = 0
                self.max_loss_streak = max(self.max_loss_streak, self.current_loss_streak)

            # Update market phase statistics
            self._update_market_phase_stats(market_phase, is_win, pnl)
            self._update_exit_reason_stats(exit_reason, is_win, pnl)

            # Track consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            # Update risk parameters based on performance
            self._update_risk_parameters(pnl)

            # Update trade history and calendar tracking
            trade_date = trade_result.get('exit_time', datetime.now()).strftime("%Y-%m-%d")
            self.daily_trade_count[trade_date] = self.daily_trade_count.get(trade_date, 0) + 1

            # Store trade in history
            self.trade_history.append(trade_result)

            # Clean up
            self._clean_daily_counts()
            self.used_partial_exits = set()

            return self.current_capital
        except Exception as e:
            self.logger.error(f"Error in update_after_trade: {e}")
            return self.current_capital

    def _update_risk_parameters(self, pnl: float):
        """Update risk parameters based on recent performance"""
        # Drawdown-based risk adjustment
        current_drawdown = max(0, 1 - (self.current_capital / self.peak_capital))
        drawdown_severity = current_drawdown / self.max_drawdown_percent

        # PnL-based adjustment
        if len(self.rolling_pnl) >= 5:
            # Convert deque to list before slicing
            rolling_pnl_list = list(self.rolling_pnl)
            rolling_pnl_sum = sum(rolling_pnl_list[-5:])
        else:
            rolling_pnl_sum = 0

        if current_drawdown > self.max_drawdown_percent * 0.25:
            # In significant drawdown - reduce risk
            drawdown_factor = max(0.45, 1.0 - drawdown_severity * 1.2)
            new_risk = self.original_max_risk * drawdown_factor

            if new_risk < self.max_risk_per_trade:
                self.max_risk_per_trade = new_risk
                self.logger.info(f"Reduced risk to {self.max_risk_per_trade:.4f} due to drawdown")
        elif current_drawdown < self.max_drawdown_percent * 0.1 and rolling_pnl_sum > 0:
            # Recovering from drawdown with positive performance - gradually increase risk
            recovery_step = self.original_max_risk * 0.08
            new_risk = min(self.original_max_risk, self.max_risk_per_trade + recovery_step)
            self.max_risk_per_trade = new_risk

        # Win/loss streak adjustments
        if self.current_win_streak >= 3:
            streak_bonus = min(0.15, self.current_win_streak * 0.03)
            self.max_risk_per_trade = min(self.original_max_risk * 1.2,
                                          self.max_risk_per_trade * (1 + streak_bonus))
        elif self.current_loss_streak >= 2:
            streak_penalty = min(0.3, self.current_loss_streak * 0.1)
            self.max_risk_per_trade = max(self.original_max_risk * 0.5,
                                          self.max_risk_per_trade * (1 - streak_penalty))

    def _update_market_phase_stats(self, market_phase: str, is_win: bool, pnl: float) -> None:
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
        if is_win:
            phase_stats["win_count"] += 1
        phase_stats["total_pnl"] += pnl

        if phase_stats["count"] > 0:
            phase_stats["win_rate"] = phase_stats["win_count"] / phase_stats["count"]
            phase_stats["avg_pnl"] = phase_stats["total_pnl"] / phase_stats["count"]

        if phase_stats["count"] >= 10:
            self._update_phase_sizing_factor(market_phase, phase_stats)

    def _update_exit_reason_stats(self, exit_reason: str, is_win: bool, pnl: float) -> None:
        if exit_reason not in self.market_phase_performance:
            self.market_phase_performance[exit_reason] = {
                "count": 0,
                "win_count": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_pnl": 0
            }

        stats = self.market_phase_performance[exit_reason]
        stats["count"] += 1
        stats["total_pnl"] += pnl

        if is_win:
            stats["win_count"] += 1

        if stats["count"] > 0:
            stats["win_rate"] = stats["win_count"] / stats["count"]
            stats["avg_pnl"] = stats["total_pnl"] / stats["count"]

    def _update_phase_sizing_factor(self, phase: str, stats: Dict[str, Any]) -> None:
        base_factor = 1.0

        if stats["win_rate"] > 0.6 and stats["avg_pnl"] > 0:
            new_factor = min(1.8, base_factor + 0.08 * (stats["win_rate"] - 0.5) * 10)
        elif stats["win_rate"] < 0.45 or stats["avg_pnl"] < 0:
            new_factor = max(0.4, base_factor - 0.08 * (0.5 - stats["win_rate"]) * 10)
        else:
            new_factor = base_factor

        current_factor = self.phase_sizing_factors.get(phase, 1.0)
        # Apply weighted update to smooth changes
        self.phase_sizing_factors[phase] = current_factor * 0.75 + new_factor * 0.25

    def _clean_daily_counts(self) -> None:
        today = datetime.now().date()
        cutoff = today - timedelta(days=7)

        self.daily_trade_count = {
            d: c for d, c in self.daily_trade_count.items()
            if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff
        }

    def check_correlation_risk(self, signal: Dict[str, Any]) -> Tuple[bool, float]:
        try:
            current_exposure = sum(pos.get('quantity', 0) * pos.get('entry_price', 0)
                                   for pos in self.open_positions)
            current_exposure_pct = current_exposure / self.current_capital if self.current_capital > 0 else 0.0

            trade_date = datetime.now().strftime("%Y-%m-%d")
            current_day_trades = self.daily_trade_count.get(trade_date, 0)

            # Adjust max trades based on performance
            adjusted_max_trades = self.max_trades_per_day
            if self.recent_win_rate > 0.5:
                adjusted_max_trades = int(self.max_trades_per_day * 1.4)
            elif self.recent_win_rate < 0.35:
                adjusted_max_trades = int(self.max_trades_per_day * 0.85)

            if current_day_trades >= adjusted_max_trades:
                return (False, 0.0)

            market_phase = signal.get('market_phase', 'neutral')
            phase_factor = self.phase_sizing_factors.get(market_phase, 1.0)

            # Adjust exposure limits based on market phase and performance
            adjusted_max_exposure = self.max_correlated_exposure
            if phase_factor > 1.1:
                adjusted_max_exposure = self.max_correlated_exposure * 1.2
            elif phase_factor < 0.8:
                adjusted_max_exposure = self.max_correlated_exposure * 0.8

            # Further adjust based on recent performance
            if self.current_win_streak >= 3:
                adjusted_max_exposure *= 1.2
            elif self.current_loss_streak >= 2:
                adjusted_max_exposure *= 0.8

            if current_exposure_pct + self.max_risk_per_trade > adjusted_max_exposure:
                leftover = max(0, adjusted_max_exposure - current_exposure_pct)
                return (False, float(leftover))

            # Check drawdown limits
            if self.peak_capital > 0:
                current_drawdown = max(0, 1 - (self.current_capital / self.peak_capital))
                max_allowed_drawdown = self.max_drawdown_percent * (1.1 if self.current_win_streak >= 3 else 1.0)

                if current_drawdown > max_allowed_drawdown:
                    return (False, 0.0)

            # Handle consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                if self.consecutive_losses >= self.max_consecutive_losses + 2:
                    # Complete trading pause after extended losses
                    return (False, 0.0)

                # Reduced position size during losing streak
                reduced_risk = self.max_risk_per_trade * max(0.35, 1 - (self.consecutive_losses * 0.15))
                return (True, float(reduced_risk))

            return (True, float(self.max_risk_per_trade))
        except Exception as e:
            self.logger.error(f"Error in check_correlation_risk: {e}")
            return (False, 0.0)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return {
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "max_drawdown": 0,
                "current_drawdown": 0,
                "sortino_ratio": 0
            }

        # Calculate basic metrics
        wins = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losses = [t for t in self.trade_history if t.get('pnl', 0) <= 0]

        win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0

        profit_sum = sum(t.get('pnl', 0) for t in wins)
        loss_sum = abs(sum(t.get('pnl', 0) for t in losses))
        profit_factor = profit_sum / max(loss_sum, 1e-10)

        avg_win = profit_sum / len(wins) if wins else 0
        avg_loss = loss_sum / len(losses) if losses else 0

        # Calculate current drawdown
        current_drawdown = max(0, 1 - (self.current_capital / self.peak_capital))

        # Calculate Sortino ratio (downside risk-adjusted return)
        # Get all returns
        returns = [t.get('pnl', 0) / self.current_capital for t in self.trade_history]
        avg_return = np.mean(returns) if returns else 0

        # Get only negative returns for downside deviation
        negative_returns = [r for r in returns if r < 0]
        downside_dev = np.std(negative_returns) if negative_returns else 1e-10
        sortino_ratio = avg_return / downside_dev if downside_dev > 0 else 0

        # Get win streak information
        current_streak = self.current_win_streak if self.current_win_streak > 0 else -self.current_loss_streak

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": current_drawdown,
            "sortino_ratio": sortino_ratio,
            "current_streak": current_streak,
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak
        }