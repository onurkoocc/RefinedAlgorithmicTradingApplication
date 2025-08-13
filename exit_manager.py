import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta


class ExitReason(Enum):
    STOP_LOSS = "StopLoss"
    PROFIT_TARGET_1 = "ProfitTarget1"
    PROFIT_TARGET_2 = "ProfitTarget2"
    PROFIT_TARGET_3 = "ProfitTarget3"
    TRAILING_STOP = "TrailingStop"
    TIME_EXIT_MAX = "TimeExitMax"
    TIME_EXIT_FLAT = "TimeExitFlat"
    REGIME_CHANGE = "RegimeChange"
    VOLATILITY_SPIKE = "VolatilitySpike"
    EMERGENCY_EXIT = "EmergencyExit"


@dataclass
class PartialExit:
    timestamp: datetime
    price: float
    size: float  # Percentage of original position (0.0-1.0)
    reason: ExitReason
    pnl: float
    remaining_size: float


@dataclass
class TradePosition:
    entry_timestamp: datetime
    entry_price: float
    direction: int  # 1 for long, -1 for short
    original_size: float
    current_size: float
    atr_at_entry: float
    regime_at_entry: str
    volatility_at_entry: float
    
    # Exit levels
    stop_loss: float
    trailing_stop: float
    profit_target_1: float
    profit_target_2: float
    profit_target_3: float
    
    # Tracking
    partial_exits: List[PartialExit]
    highest_profit: float
    lowest_profit: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    
    def __post_init__(self):
        if not hasattr(self, 'partial_exits'):
            self.partial_exits = []
        if not hasattr(self, 'highest_profit'):
            self.highest_profit = 0.0
        if not hasattr(self, 'lowest_profit'):
            self.lowest_profit = 0.0
        if not hasattr(self, 'max_adverse_excursion'):
            self.max_adverse_excursion = 0.0
        if not hasattr(self, 'max_favorable_excursion'):
            self.max_favorable_excursion = 0.0


class ExitManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("ExitManager")
        
        # Load exit configuration - handle both Config class and dict
        try:
            if hasattr(config, 'config') and isinstance(config.config, dict):
                # Direct access to config dictionary
                self.exit_config = config.config.get('exit_strategy', {})
            elif hasattr(config, 'get'):
                # Dictionary-like access
                self.exit_config = config.get('exit_strategy', {})
            else:
                # Fallback to empty dict
                self.exit_config = {}
                
            # If exit_config is empty, use defaults
            if not self.exit_config:
                self.exit_config = self._get_default_exit_config()
                
        except Exception as e:
            self.logger.warning(f"Error loading exit config: {e}")
            self.exit_config = self._get_default_exit_config()
            
        self.load_exit_parameters()
        
        # Active positions tracking
        self.active_positions: Dict[str, TradePosition] = {}
        self.historical_exits: List[Dict] = []
        
        # Performance tracking
        self.exit_performance = {
            "reward_risk_ratios": [],
            "exit_reason_stats": {},
            "partial_exit_effectiveness": {},
            "time_in_trade_analysis": {}
        }
        
    def _get_default_exit_config(self):
        """Get default exit configuration"""
        return {
            "base_stop_atr_multiplier": 2.0,
            "profit_target_1_atr": 2.0,
            "profit_target_2_atr": 4.0,
            "profit_target_3_atr": 6.0,
            "target_1_exit_size": 0.5,
            "target_2_exit_size": 0.3,
            "target_3_exit_size": 0.2,
            "trailing_start_atr": 1.0,
            "trailing_distance_atr": 1.5,
            "max_hold_hours": 48,
            "flat_position_hours": 24,
            "enable_regime_change_exit": True,
            "volatility_spike_threshold": 2.5,
            "regime_stop_adjustments": {
                "volatile": 1.3,
                "trending": 0.8, 
                "ranging": 1.2,
                "neutral": 1.0,
                "strong_uptrend": 0.7,
                "strong_downtrend": 0.7
            }
        }
        
    def load_exit_parameters(self):
        """Load and validate exit strategy parameters"""
        self.base_stop_atr_multiplier = self.exit_config.get("base_stop_atr_multiplier", 2.0)
        self.profit_target_1_atr = self.exit_config.get("profit_target_1_atr", 1.5)
        self.profit_target_2_atr = self.exit_config.get("profit_target_2_atr", 3.0)
        self.profit_target_3_atr = self.exit_config.get("profit_target_3_atr", 5.0)
        
        # Partial exit sizes
        self.target_1_exit_size = self.exit_config.get("target_1_exit_size", 0.5)  # 50%
        self.target_2_exit_size = self.exit_config.get("target_2_exit_size", 0.3)  # 30%
        self.target_3_exit_size = self.exit_config.get("target_3_exit_size", 0.2)  # 20%
        
        # Time-based exits
        self.max_hold_hours = self.exit_config.get("max_hold_hours", 48)
        self.flat_position_hours = self.exit_config.get("flat_position_hours", 24)
        
        # Trailing stop
        self.trailing_start_atr = self.exit_config.get("trailing_start_atr", 1.0)
        self.trailing_distance_atr = self.exit_config.get("trailing_distance_atr", 1.5)
        
        # Volatility and regime parameters
        self.volatility_spike_threshold = self.exit_config.get("volatility_spike_threshold", 2.5)
        self.regime_change_exit = self.exit_config.get("enable_regime_change_exit", True)
        
        # Regime-specific stop loss adjustments
        self.regime_stop_adjustments = self.exit_config.get("regime_stop_adjustments", {
            "volatile": 1.3,      # Wider stops in volatile markets
            "trending": 0.8,      # Tighter stops in trending markets
            "ranging": 1.2,       # Slightly wider stops in ranging markets
            "neutral": 1.0        # Base stops
        })
        
    def open_position(self, entry_timestamp: datetime, entry_price: float, 
                     direction: int, position_size: float, atr: float,
                     market_regime: str, volatility: float) -> str:
        """Open a new position and calculate exit levels"""
        
        position_id = f"{entry_timestamp.strftime('%Y%m%d_%H%M%S')}_{direction}"
        
        # Calculate dynamic stop loss
        regime_adjustment = self.regime_stop_adjustments.get(market_regime, 1.0)
        volatility_adjustment = min(1.5, max(0.7, volatility / 0.02))  # Scale based on volatility
        
        stop_atr_multiplier = self.base_stop_atr_multiplier * regime_adjustment * volatility_adjustment
        
        if direction == 1:  # Long position
            stop_loss = entry_price - (atr * stop_atr_multiplier)
            profit_target_1 = entry_price + (atr * self.profit_target_1_atr)
            profit_target_2 = entry_price + (atr * self.profit_target_2_atr)
            profit_target_3 = entry_price + (atr * self.profit_target_3_atr)
            trailing_stop = entry_price - (atr * self.trailing_distance_atr)
        else:  # Short position
            stop_loss = entry_price + (atr * stop_atr_multiplier)
            profit_target_1 = entry_price - (atr * self.profit_target_1_atr)
            profit_target_2 = entry_price - (atr * self.profit_target_2_atr)
            profit_target_3 = entry_price - (atr * self.profit_target_3_atr)
            trailing_stop = entry_price + (atr * self.trailing_distance_atr)
        
        position = TradePosition(
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            direction=direction,
            original_size=position_size,
            current_size=position_size,
            atr_at_entry=atr,
            regime_at_entry=market_regime,
            volatility_at_entry=volatility,
            stop_loss=stop_loss,
            trailing_stop=trailing_stop,
            profit_target_1=profit_target_1,
            profit_target_2=profit_target_2,
            profit_target_3=profit_target_3,
            partial_exits=[],
            highest_profit=0.0,
            lowest_profit=0.0,
            max_adverse_excursion=0.0,
            max_favorable_excursion=0.0
        )
        
        self.active_positions[position_id] = position
        
        self.logger.info(f"Opened {market_regime} {direction} position: "
                        f"Entry=${entry_price:.2f}, Stop=${stop_loss:.2f}, "
                        f"T1=${profit_target_1:.2f}, T2=${profit_target_2:.2f}, T3=${profit_target_3:.2f}")
        
        return position_id
        
    def update_position(self, position_id: str, current_timestamp: datetime, 
                       current_price: float, current_atr: float,
                       current_regime: str, current_volatility: float) -> List[Tuple[str, float, float, str]]:
        """
        Update position and check for exit conditions
        Returns: List of (exit_reason, exit_price, exit_size, remaining_position_id)
        """
        
        if position_id not in self.active_positions:
            return []
            
        position = self.active_positions[position_id]
        exits = []
        
        # Update profit tracking
        current_profit = self._calculate_profit(position, current_price)
        position.highest_profit = max(position.highest_profit, current_profit)
        position.lowest_profit = min(position.lowest_profit, current_profit)
        
        # Update MAE and MFE
        if current_profit < 0:
            position.max_adverse_excursion = max(position.max_adverse_excursion, abs(current_profit))
        else:
            position.max_favorable_excursion = max(position.max_favorable_excursion, current_profit)
        
        # Check exit conditions in priority order
        
        # 1. Stop Loss (highest priority)
        if self._check_stop_loss(position, current_price):
            exits.append((ExitReason.STOP_LOSS.value, position.stop_loss, 1.0, None))
            return exits
            
        # 2. Time-based exits
        time_exit = self._check_time_exits(position, current_timestamp, current_price)
        if time_exit:
            exits.append(time_exit)
            return exits
            
        # 3. Volatility spike exit
        if self._check_volatility_spike(position, current_volatility):
            exits.append((ExitReason.VOLATILITY_SPIKE.value, current_price, 1.0, None))
            return exits
            
        # 4. Regime change exit
        if self._check_regime_change(position, current_regime):
            exits.append((ExitReason.REGIME_CHANGE.value, current_price, 0.7, position_id))
            position.current_size *= 0.3  # Keep 30% of position
            
        # 5. Profit targets (partial exits)
        profit_exits = self._check_profit_targets(position, current_price)
        exits.extend(profit_exits)
        
        # 6. Update trailing stop
        self._update_trailing_stop(position, current_price, current_atr)
        
        # 7. Check trailing stop
        if self._check_trailing_stop(position, current_price):
            remaining_size = position.current_size
            exits.append((ExitReason.TRAILING_STOP.value, position.trailing_stop, remaining_size, None))
            
        return exits
        
    def _calculate_profit(self, position: TradePosition, current_price: float) -> float:
        """Calculate current profit/loss percentage"""
        if position.direction == 1:  # Long
            return (current_price - position.entry_price) / position.entry_price
        else:  # Short
            return (position.entry_price - current_price) / position.entry_price
            
    def _check_stop_loss(self, position: TradePosition, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if position.direction == 1:  # Long
            return current_price <= position.stop_loss
        else:  # Short
            return current_price >= position.stop_loss
            
    def _check_time_exits(self, position: TradePosition, current_time: datetime, current_price: float) -> Optional[Tuple]:
        """Check time-based exit conditions"""
        time_in_trade = (current_time - position.entry_timestamp).total_seconds() / 3600  # Hours
        
        # Maximum hold time
        if time_in_trade >= self.max_hold_hours:
            return (ExitReason.TIME_EXIT_MAX.value, current_price, 1.0, None)
            
        # Flat position exit (if no significant profit after time threshold)
        if time_in_trade >= self.flat_position_hours:
            current_profit = self._calculate_profit(position, current_price)
            if abs(current_profit) < 0.005:  # Less than 0.5% profit/loss
                return (ExitReason.TIME_EXIT_FLAT.value, current_price, 1.0, None)
                
        return None
        
    def _check_volatility_spike(self, position: TradePosition, current_volatility: float) -> bool:
        """Check if volatility has spiked beyond threshold"""
        volatility_ratio = current_volatility / position.volatility_at_entry
        return volatility_ratio > self.volatility_spike_threshold
        
    def _check_regime_change(self, position: TradePosition, current_regime: str) -> bool:
        """Check if market regime has changed adversely"""
        if not self.regime_change_exit:
            return False
            
        entry_regime = position.regime_at_entry
        direction = position.direction
        
        # Define adverse regime changes
        adverse_changes = {
            ("trending", "ranging"): True,
            ("trending", "volatile"): True,
            ("ranging", "volatile"): True,
            ("neutral", "volatile"): True
        }
        
        # Check if current regime is adverse for the position direction
        if (entry_regime, current_regime) in adverse_changes:
            return True
            
        # Specific directional checks
        if direction == 1 and current_regime in ["bearish", "strong_downtrend"]:
            return True
        elif direction == -1 and current_regime in ["bullish", "strong_uptrend"]:
            return True
            
        return False
        
    def _check_profit_targets(self, position: TradePosition, current_price: float) -> List[Tuple]:
        """Check profit target levels for partial exits"""
        exits = []
        
        if position.direction == 1:  # Long
            # Target 1
            if (current_price >= position.profit_target_1 and 
                not any(exit.reason == ExitReason.PROFIT_TARGET_1 for exit in position.partial_exits)):
                exits.append((ExitReason.PROFIT_TARGET_1.value, position.profit_target_1, 
                            self.target_1_exit_size, position))
                            
            # Target 2  
            if (current_price >= position.profit_target_2 and 
                not any(exit.reason == ExitReason.PROFIT_TARGET_2 for exit in position.partial_exits)):
                exits.append((ExitReason.PROFIT_TARGET_2.value, position.profit_target_2, 
                            self.target_2_exit_size, position))
                            
            # Target 3
            if (current_price >= position.profit_target_3 and 
                not any(exit.reason == ExitReason.PROFIT_TARGET_3 for exit in position.partial_exits)):
                exits.append((ExitReason.PROFIT_TARGET_3.value, position.profit_target_3, 
                            self.target_3_exit_size, position))
        else:  # Short
            # Target 1
            if (current_price <= position.profit_target_1 and 
                not any(exit.reason == ExitReason.PROFIT_TARGET_1 for exit in position.partial_exits)):
                exits.append((ExitReason.PROFIT_TARGET_1.value, position.profit_target_1, 
                            self.target_1_exit_size, position))
                            
            # Target 2
            if (current_price <= position.profit_target_2 and 
                not any(exit.reason == ExitReason.PROFIT_TARGET_2 for exit in position.partial_exits)):
                exits.append((ExitReason.PROFIT_TARGET_2.value, position.profit_target_2, 
                            self.target_2_exit_size, position))
                            
            # Target 3
            if (current_price <= position.profit_target_3 and 
                not any(exit.reason == ExitReason.PROFIT_TARGET_3 for exit in position.partial_exits)):
                exits.append((ExitReason.PROFIT_TARGET_3.value, position.profit_target_3, 
                            self.target_3_exit_size, position))
        
        return exits
        
    def _update_trailing_stop(self, position: TradePosition, current_price: float, current_atr: float):
        """Update trailing stop based on favorable price movement"""
        current_profit = self._calculate_profit(position, current_price)
        
        # Start trailing after reaching profit threshold
        if current_profit >= (self.trailing_start_atr * position.atr_at_entry / position.entry_price):
            if position.direction == 1:  # Long
                new_trailing_stop = current_price - (current_atr * self.trailing_distance_atr)
                position.trailing_stop = max(position.trailing_stop, new_trailing_stop)
            else:  # Short
                new_trailing_stop = current_price + (current_atr * self.trailing_distance_atr)
                position.trailing_stop = min(position.trailing_stop, new_trailing_stop)
                
    def _check_trailing_stop(self, position: TradePosition, current_price: float) -> bool:
        """Check if trailing stop should be triggered"""
        if position.direction == 1:  # Long
            return current_price <= position.trailing_stop
        else:  # Short
            return current_price >= position.trailing_stop
            
    def execute_partial_exit(self, position_id: str, exit_reason: ExitReason, 
                           exit_price: float, exit_size: float, timestamp: datetime):
        """Execute a partial exit and update position"""
        if position_id not in self.active_positions:
            return
            
        position = self.active_positions[position_id]
        exit_amount = position.current_size * exit_size
        
        # Calculate PnL for this partial exit
        if position.direction == 1:  # Long
            pnl = (exit_price - position.entry_price) * exit_amount / position.entry_price
        else:  # Short
            pnl = (position.entry_price - exit_price) * exit_amount / position.entry_price
            
        # Create partial exit record
        partial_exit = PartialExit(
            timestamp=timestamp,
            price=exit_price,
            size=exit_size,
            reason=exit_reason,
            pnl=pnl,
            remaining_size=position.current_size - exit_amount
        )
        
        position.partial_exits.append(partial_exit)
        position.current_size -= exit_amount
        
        # Log the partial exit
        self.logger.info(f"Partial exit {exit_reason.value}: {exit_size:.1%} at ${exit_price:.2f}, "
                        f"PnL: {pnl:.2%}, Remaining: {position.current_size:.3f}")
        
    def close_position(self, position_id: str, exit_reason: ExitReason, 
                      exit_price: float, timestamp: datetime) -> Dict:
        """Close position completely and record final statistics"""
        if position_id not in self.active_positions:
            return {}
            
        position = self.active_positions[position_id]
        
        # Calculate final PnL
        if position.direction == 1:  # Long
            final_pnl = (exit_price - position.entry_price) * position.current_size / position.entry_price
        else:  # Short
            final_pnl = (position.entry_price - exit_price) * position.current_size / position.entry_price
            
        # Calculate total PnL including partial exits
        total_partial_pnl = sum(exit.pnl for exit in position.partial_exits)
        total_pnl = total_partial_pnl + final_pnl
        
        # Calculate reward-risk ratio
        risk = abs(position.entry_price - position.stop_loss) / position.entry_price
        reward_risk_ratio = total_pnl / risk if risk > 0 else 0
        
        # Calculate time in trade
        time_in_trade = (timestamp - position.entry_timestamp).total_seconds() / 3600
        
        # Record exit statistics
        exit_record = {
            "position_id": position_id,
            "entry_timestamp": position.entry_timestamp,
            "exit_timestamp": timestamp,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "direction": position.direction,
            "original_size": position.original_size,
            "regime_at_entry": position.regime_at_entry,
            "volatility_at_entry": position.volatility_at_entry,
            "exit_reason": exit_reason.value,
            "partial_exits": len(position.partial_exits),
            "total_pnl": total_pnl,
            "reward_risk_ratio": reward_risk_ratio,
            "time_in_trade_hours": time_in_trade,
            "max_adverse_excursion": position.max_adverse_excursion,
            "max_favorable_excursion": position.max_favorable_excursion,
            "stop_loss": position.stop_loss,
            "profit_targets": [position.profit_target_1, position.profit_target_2, position.profit_target_3]
        }
        
        self.historical_exits.append(exit_record)
        self.performance_tracking(exit_record)
        
        # Remove from active positions
        del self.active_positions[position_id]
        
        self.logger.info(f"Closed position {exit_reason.value}: Total PnL {total_pnl:.2%}, "
                        f"R:R {reward_risk_ratio:.2f}, Time {time_in_trade:.1f}h")
        
        return exit_record
        
    def performance_tracking(self, exit_record: Dict):
        """Update performance tracking statistics"""
        reward_risk = exit_record["reward_risk_ratio"]
        self.exit_performance["reward_risk_ratios"].append(reward_risk)
        
        # Exit reason statistics
        reason = exit_record["exit_reason"]
        if reason not in self.exit_performance["exit_reason_stats"]:
            self.exit_performance["exit_reason_stats"][reason] = {
                "count": 0, "total_pnl": 0, "avg_pnl": 0, "avg_reward_risk": 0
            }
        
        stats = self.exit_performance["exit_reason_stats"][reason]
        stats["count"] += 1
        stats["total_pnl"] += exit_record["total_pnl"]
        stats["avg_pnl"] = stats["total_pnl"] / stats["count"]
        stats["avg_reward_risk"] = (stats["avg_reward_risk"] * (stats["count"] - 1) + reward_risk) / stats["count"]
        
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.exit_performance["reward_risk_ratios"]:
            return {"status": "No trades completed yet"}
            
        rr_ratios = self.exit_performance["reward_risk_ratios"]
        
        summary = {
            "total_trades": len(rr_ratios),
            "avg_reward_risk_ratio": np.mean(rr_ratios),
            "median_reward_risk_ratio": np.median(rr_ratios),
            "min_reward_risk_ratio": np.min(rr_ratios),
            "max_reward_risk_ratio": np.max(rr_ratios),
            "trades_above_2_to_1": sum(1 for rr in rr_ratios if rr >= 2.0),
            "percentage_above_2_to_1": sum(1 for rr in rr_ratios if rr >= 2.0) / len(rr_ratios) * 100,
            "exit_reason_breakdown": self.exit_performance["exit_reason_stats"],
            "active_positions": len(self.active_positions)
        }
        
        return summary
        
    def optimize_exit_parameters(self, historical_data: pd.DataFrame) -> Dict:
        """Analyze historical data to optimize exit parameters"""
        if historical_data.empty:
            return {}
            
        # Analyze optimal reward-risk ratios by market regime
        regime_analysis = {}
        
        for regime in historical_data['market_regime'].unique():
            regime_data = historical_data[historical_data['market_regime'] == regime]
            
            # Calculate various ATR multipliers performance
            optimal_stop = self._optimize_stop_loss(regime_data)
            optimal_targets = self._optimize_profit_targets(regime_data)
            
            regime_analysis[regime] = {
                "optimal_stop_atr": optimal_stop,
                "optimal_profit_targets": optimal_targets,
                "sample_size": len(regime_data)
            }
            
        return regime_analysis
        
    def _optimize_stop_loss(self, data: pd.DataFrame) -> float:
        """Find optimal stop loss ATR multiplier"""
        atr_multipliers = np.arange(1.0, 4.0, 0.2)
        best_ratio = 0
        best_multiplier = 2.0
        
        for multiplier in atr_multipliers:
            # Simulate stop losses
            simulated_ratios = []
            for _, row in data.iterrows():
                # Calculate hypothetical reward-risk with this multiplier
                risk = row['atr'] * multiplier / row['entry_price']
                if risk > 0:
                    reward_risk = row['pnl'] / risk
                    simulated_ratios.append(reward_risk)
            
            if simulated_ratios:
                avg_ratio = np.mean(simulated_ratios)
                if avg_ratio > best_ratio:
                    best_ratio = avg_ratio
                    best_multiplier = multiplier
                    
        return best_multiplier
        
    def _optimize_profit_targets(self, data: pd.DataFrame) -> List[float]:
        """Find optimal profit target ATR multipliers"""
        # Test different target combinations
        target_combinations = [
            [1.0, 2.0, 4.0],
            [1.5, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [1.2, 2.5, 4.5]
        ]
        
        best_score = 0
        best_targets = [1.5, 3.0, 5.0]
        
        for targets in target_combinations:
            # Simulate partial exits
            total_score = 0
            for _, row in data.iterrows():
                score = self._simulate_partial_exits(row, targets)
                total_score += score
            
            avg_score = total_score / len(data) if len(data) > 0 else 0
            if avg_score > best_score:
                best_score = avg_score
                best_targets = targets
                
        return best_targets
        
    def _simulate_partial_exits(self, trade_data, profit_targets: List[float]) -> float:
        """Simulate partial exits for a single trade"""
        # Simplified simulation - would need more detailed price data for full simulation
        entry_price = trade_data.get('entry_price', 0)
        max_price = trade_data.get('max_price', entry_price)
        atr = trade_data.get('atr', 0)
        
        if entry_price == 0 or atr == 0:
            return 0
            
        total_profit = 0
        remaining_size = 1.0
        
        for i, target_multiplier in enumerate(profit_targets):
            target_price = entry_price + (atr * target_multiplier)
            
            if max_price >= target_price:  # Target was hit
                exit_size = [0.5, 0.3, 0.2][i]  # Partial exit sizes
                if remaining_size >= exit_size:
                    profit = (target_price - entry_price) / entry_price * exit_size
                    total_profit += profit
                    remaining_size -= exit_size
                    
        return total_profit
        
    def save_performance_data(self, filepath: str):
        """Save performance data to file"""
        performance_data = {
            "exit_performance": self.exit_performance,
            "historical_exits": self.historical_exits,
            "config": self.exit_config
        }
        
        with open(filepath, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
            
    def load_performance_data(self, filepath: str):
        """Load performance data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.exit_performance = data.get("exit_performance", {})
            self.historical_exits = data.get("historical_exits", [])
            self.logger.info(f"Loaded performance data: {len(self.historical_exits)} historical trades")
        except Exception as e:
            self.logger.warning(f"Could not load performance data: {e}")