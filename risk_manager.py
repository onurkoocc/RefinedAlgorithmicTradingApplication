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
        self.max_risk_per_trade = config.get("risk", "max_risk_per_trade", 0.02)
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

        self.used_partial_exits = set()

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
        except (ValueError, TypeError):
            self.logger.warning("Invalid inputs to calculate_position_size.")
            return 0.0

        # Base risk adjustment - slightly reduced from 0.7 to 0.65 for more conservative sizing
        base_risk = self.max_risk_per_trade * 0.65 * min(1.5, max(0.5, ensemble_score))

        if self.recent_win_rate > 0 and len(self.recent_trades) >= 5:
            avg_win = np.mean([t['pnl'] for t in self.recent_trades if t['is_win']])
            avg_loss = abs(np.mean([t['pnl'] for t in self.recent_trades if not t['is_win']]))

            if avg_loss > 0:
                reward_risk_ratio = avg_win / avg_loss

                # Kelly fraction reduced slightly from 0.3 to 0.25 for more conservative sizing
                kelly_fraction = 0.25
                kelly_percentage = kelly_fraction * (
                        (self.recent_win_rate * reward_risk_ratio) - (1 - self.recent_win_rate)
                ) / reward_risk_ratio

                kelly_percentage = max(0.002, min(0.05, kelly_percentage))
                risk_pct = min(base_risk, kelly_percentage)
            else:
                risk_pct = base_risk
        else:
            risk_pct = base_risk

        # Market regime adjustments
        if (direction == 'long' and market_regime > 0.7) or (direction == 'short' and market_regime < -0.7):
            regime_factor = 1.5
        elif (direction == 'long' and market_regime < -0.7) or (direction == 'short' and market_regime > 0.7):
            regime_factor = 0.5
        else:
            if direction == 'long':
                regime_factor = 1.0 + (market_regime * 0.3)
            else:
                regime_factor = 1.0 - (market_regime * 0.3)

        risk_pct = risk_pct * regime_factor

        # Market phase adjustments - NEW
        if market_phase == "neutral":
            # Slightly higher allocation for neutral phase which performs well
            phase_factor = 1.1
        elif market_phase == "ranging_at_resistance" and direction == 'long':
            # Reduce position size for longs in ranging_at_resistance
            phase_factor = 0.7
        elif market_phase == "ranging_at_support" and direction == 'short':
            # Reduce position size for shorts in ranging_at_support
            phase_factor = 0.7
        else:
            phase_factor = 1.0

        risk_pct = risk_pct * phase_factor

        # Capital management
        capital_growth_factor = max(0.6, min(1.0, self.initial_capital / self.current_capital))
        max_allowed_risk = self.original_max_risk * 1.2 * capital_growth_factor

        risk_pct = min(max_allowed_risk, risk_pct)

        # Loss factor adjustment
        loss_factor = 1.0
        if self.consecutive_losses > 0:
            loss_factor = np.exp(-0.25 * self.consecutive_losses)
            loss_factor = max(0.4, loss_factor)

        risk_pct = risk_pct * loss_factor

        # Rest of the method remains the same...
        available_capital = max(0, self.current_capital - self.capital_floor)
        risk_amount = available_capital * risk_pct

        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0 or risk_amount <= 0:
            self.logger.warning(f"Invalid risk calculation: risk_amount={risk_amount}, risk_per_unit={risk_per_unit}")
            return 0.0

        quantity = risk_amount / risk_per_unit
        position_value = quantity * entry_price

        max_position_value = available_capital * 0.9
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

        # Modified thresholds and order
        exit_levels = [
            {"threshold": 0.005 * volatility_factor, "portion": 0.2, "id": "level0"},
            {"threshold": 0.01 * volatility_factor, "portion": 0.25, "id": "level1"},
            {"threshold": 0.018 * volatility_factor, "portion": 0.25, "id": "level2"},
            {"threshold": 0.03 * volatility_factor, "portion": 0.25, "id": "level3"},
            {"threshold": 0.05 * volatility_factor, "portion": 0.25, "id": "level4"}
        ]

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

    def check_correlation_risk(self, signal: Dict[str, Any]) -> Tuple[bool, float]:
        current_exposure = sum(pos.get('quantity', 0) * pos.get('entry_price', 0) for pos in self.open_positions)
        current_exposure_pct = current_exposure / self.current_capital if self.current_capital > 0 else 0.0

        trade_date = datetime.now().strftime("%Y-%m-%d")
        if self.daily_trade_count.get(trade_date, 0) >= self.max_trades_per_day:
            return (False, 0.0)

        if current_exposure_pct + self.max_risk_per_trade > self.max_correlated_exposure:
            leftover = max(0, self.max_correlated_exposure - current_exposure_pct)
            return (False, float(leftover))

        if self.peak_capital > 0:
            current_drawdown = max(0, 1 - (self.current_capital / self.peak_capital))
            if current_drawdown > self.max_drawdown_percent:
                return (False, 0.0)

        if self.consecutive_losses >= self.max_consecutive_losses:
            reduced_risk = self.max_risk_per_trade * max(0.4, 1 - (self.consecutive_losses * 0.2))
            return (True, float(reduced_risk))

        return (True, float(self.max_risk_per_trade))

    def update_after_trade(self, trade_result: Dict[str, Any]) -> float:
        pnl = float(trade_result.get('pnl', 0))
        is_win = pnl > 0

        self.current_capital = max(self.capital_floor, self.current_capital + pnl)

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        self.recent_trades.append({"pnl": pnl, "is_win": is_win})
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades.pop(0)

        if self.recent_trades:
            wins = sum(1 for trade in self.recent_trades if trade["is_win"])
            self.recent_win_rate = wins / len(self.recent_trades)

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        current_drawdown = max(0, 1 - (self.current_capital / self.peak_capital))

        if current_drawdown > self.max_drawdown_percent * 0.3:
            drawdown_factor = max(0.5, 1.0 - current_drawdown / self.max_drawdown_percent)
            new_risk = self.original_max_risk * drawdown_factor

            if new_risk < self.max_risk_per_trade:
                self.max_risk_per_trade = new_risk
        elif current_drawdown < self.max_drawdown_percent * 0.1 and self.max_risk_per_trade < self.original_max_risk:
            recovery_step = self.original_max_risk * 0.1
            new_risk = min(self.original_max_risk, self.max_risk_per_trade + recovery_step)
            self.max_risk_per_trade = new_risk

        if len(self.recent_trades) >= 5:
            if self.recent_win_rate < 0.4:
                self.max_risk_per_trade = max(self.original_max_risk * 0.6, self.max_risk_per_trade * 0.9)
            elif self.recent_win_rate > 0.7 and current_drawdown < self.max_drawdown_percent * 0.2:
                self.max_risk_per_trade = min(self.original_max_risk, self.max_risk_per_trade * 1.05)

        trade_date = trade_result.get('exit_time', datetime.now()).strftime("%Y-%m-%d")
        self.daily_trade_count[trade_date] = self.daily_trade_count.get(trade_date, 0) + 1

        self.trade_history.append(trade_result)
        self._clean_daily_counts()

        self.used_partial_exits = set()

        return self.current_capital

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
            trade_id = str(position.get('id', ''))

            ema_20 = float(kwargs.get('ema_20', 0))
            rsi_14 = float(kwargs.get('rsi_14', 50))
            macd = float(kwargs.get('macd', 0))
            macd_signal = float(kwargs.get('macd_signal', 0))
            macd_histogram = float(kwargs.get('macd_histogram', 0))

        except (ValueError, TypeError, KeyError) as e:
            return {"exit": False, "reason": "InvalidInputs"}

        if direction == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        stop_hit = ((direction == 'long' and current_price <= current_stop_loss) or
                    (direction == 'short' and current_price >= current_stop_loss))
        if stop_hit:
            return {"exit": True, "reason": "StopLoss", "exit_price": float(current_stop_loss)}

        if trade_duration > 24:
            if pnl_pct > 0 and pnl_pct < 0.01:
                return {
                    "exit": True,
                    "reason": "TimeBasedExit_SmallProfit",
                    "exit_price": current_price
                }

            if pnl_pct < 0 and trade_duration > 36:
                return {
                    "exit": True,
                    "reason": "TimeBasedExit_StagnantLoss",
                    "exit_price": current_price
                }

        if trade_duration > 12 and pnl_pct < 0:
            stop_tightening_factor = min(0.8, max(0.5, 1.0 - (trade_duration / 72)))

            if volatility > 0.7:
                atr_multiple = 2.5
            elif volatility < 0.3:
                atr_multiple = 1.2
            else:
                atr_multiple = 1.8

            if direction == 'long':
                new_stop = current_price - (atr_multiple * current_atr * stop_tightening_factor)
                if new_stop > current_stop_loss:
                    return {
                        "exit": False,
                        "update_stop": True,
                        "new_stop": float(new_stop),
                        "reason": "TimeBasedStopAdjustment"
                    }
            else:
                new_stop = current_price + (atr_multiple * current_atr * stop_tightening_factor)
                if new_stop < current_stop_loss:
                    return {
                        "exit": False,
                        "update_stop": True,
                        "new_stop": float(new_stop),
                        "reason": "TimeBasedStopAdjustment"
                    }

        if pnl_pct > 0:
            if pnl_pct <= 0.01:
                atr_multiple = 2.5
            elif pnl_pct <= 0.02:
                atr_multiple = 2.0
            elif pnl_pct <= 0.03:
                atr_multiple = 1.5
            elif pnl_pct <= 0.05:
                atr_multiple = 1.2
            else:
                atr_multiple = 1.0

            if volatility > 0.7:
                atr_multiple *= 1.3
            elif volatility < 0.3:
                atr_multiple *= 0.8

            if entry_confidence > 0.8:
                atr_multiple *= 0.9

            if np.isnan(current_atr) or current_atr <= 0:
                current_atr = current_price * 0.01

            if direction == 'long':
                new_stop = current_price - (atr_multiple * current_atr)

                if pnl_pct > 0.03 and new_stop < entry_price:
                    break_even_buffer = current_atr * 0.5
                    new_stop = max(new_stop, entry_price + break_even_buffer)

                if new_stop > current_stop_loss:
                    return {
                        "exit": False,
                        "update_stop": True,
                        "new_stop": float(new_stop),
                        "reason": "TrailingStop"
                    }

            else:
                new_stop = current_price + (atr_multiple * current_atr)

                if pnl_pct > 0.03 and new_stop > entry_price:
                    break_even_buffer = current_atr * 0.5
                    new_stop = min(new_stop, entry_price - break_even_buffer)

                if new_stop < current_stop_loss:
                    return {
                        "exit": False,
                        "update_stop": True,
                        "new_stop": float(new_stop),
                        "reason": "TrailingStop"
                    }

        return {"exit": False, "reason": "NoActionNeeded"}


    def _calculate_volatility_factor(self, direction: str, entry_price: float, current_price: float) -> float:
        recent_volatility = self._get_recent_volatility()
        base_factor = 1.0

        if recent_volatility < 0.01:
            return base_factor * 0.8
        elif recent_volatility > 0.03:
            return base_factor * 1.5
        else:
            return base_factor * (1.0 + (recent_volatility - 0.01) * 20)

    def _get_recent_volatility(self) -> float:
        return 0.02  # Placeholder - implement based on recent market data

    def _clean_daily_counts(self):
        today = datetime.now().date()
        cutoff = today - timedelta(days=7)

        self.daily_trade_count = {
            d: c for d, c in self.daily_trade_count.items()
            if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff
        }