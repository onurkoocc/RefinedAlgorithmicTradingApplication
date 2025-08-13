#!/usr/bin/env python3
"""
Market Condition Filter - Avoid poor trading environments
Addresses the issue: 92% of trades were in 'neutral' market with poor results
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

class MarketConditionFilter:
    def __init__(self, config: Dict):
        self.config = config.get('signal', {})
        self.logger = logging.getLogger("MarketConditionFilter")
        
        # Track market condition statistics by regime
        self.regime_trade_counts = {
            'neutral': 0,
            'ranging': 0,
            'uptrend': 0,
            'strong_uptrend': 0,
            'weak_uptrend': 0,
            'downtrend': 0,
            'strong_downtrend': 0,
            'weak_downtrend': 0,
            'volatile': 0,
            'choppy': 0
        }
        self.total_trade_count = 0
        self.last_neutral_reject_time = None
        
        # Configuration - UPDATED LIMITS FOR REGIME DIVERSITY
        self.max_neutral_trades_pct = self.config.get('max_neutral_trades_pct', 0.5)  # Max 50% in neutral
        self.max_single_regime_pct = 0.5  # Max 50% in any single regime
        self.min_trending_pct = 0.2  # Minimum 20% in trending regimes
        self.min_volume_ratio = self.config.get('min_volume_ratio', 1.2)  # Lowered from 1.5
        self.volatility_percentile_min = self.config.get('volatility_percentile_min', 0.3)  # Lowered from 0.4
        self.min_trend_strength = self.config.get('trending_threshold', 25)  # Lowered from 30
        
        # Regime-specific confidence thresholds
        self.regime_thresholds = {
            'strong_uptrend': 0.005,
            'uptrend': 0.006,
            'weak_uptrend': 0.008,
            'strong_downtrend': 0.005,
            'downtrend': 0.006,
            'weak_downtrend': 0.008,
            'ranging': 0.012,
            'neutral': 0.015,
            'volatile': 0.020,
            'choppy': 0.018,
            'uptrend_transition': 0.007,
            'downtrend_transition': 0.007,
            'ranging_at_support': 0.010,
            'ranging_at_resistance': 0.010
        }
        
        # Rolling statistics
        self.volatility_history = []
        self.volume_ratio_history = []
        self.max_history_length = 100
        self.recent_regimes = []  # Track recent regime classifications
        
    def should_trade(self, market_data: Dict) -> Tuple[bool, str]:
        """
        Determine if current market conditions are suitable for trading
        
        Args:
            market_data: Dictionary with market indicators and regime
            
        Returns:
            Tuple of (should_trade: bool, reason: str)
        """
        
        # Extract market data
        market_regime = market_data.get('market_regime', 'neutral')
        volume_ratio = market_data.get('volume_ratio', 1.0)
        realized_volatility = market_data.get('realized_volatility', 0.02)
        trend_strength_raw = market_data.get('adx_14', 20)
        # Handle normalized ADX (0-1) vs standard (0-100)
        trend_strength = trend_strength_raw * 100 if trend_strength_raw <= 1.0 else trend_strength_raw
        rsi_raw = market_data.get('rsi_14', 50)
        # Handle normalized RSI (0-1) vs standard (0-100)
        rsi = rsi_raw * 100 if rsi_raw <= 1.0 else rsi_raw
        
        # Normalize regime name (handle variants)
        regime_normalized = self._normalize_regime_name(market_regime)
        
        # Update rolling statistics
        self._update_statistics(realized_volatility, volume_ratio, regime_normalized)
        
        # Filter 1: Enforce regime diversity - prevent over-concentration
        if self.total_trade_count > 10:  # Only enforce after initial trades
            # Check single regime concentration
            regime_pct = self.regime_trade_counts.get(regime_normalized, 0) / max(1, self.total_trade_count)
            if regime_pct >= self.max_single_regime_pct:
                return False, f"Too many {regime_normalized} trades: {regime_pct:.1%} >= {self.max_single_regime_pct:.1%}"
            
            # Ensure minimum trending trades
            trending_regimes = ['uptrend', 'strong_uptrend', 'weak_uptrend', 'downtrend', 'strong_downtrend', 'weak_downtrend']
            trending_count = sum(self.regime_trade_counts.get(r, 0) for r in trending_regimes)
            trending_pct = trending_count / max(1, self.total_trade_count)
            
            # If we don't have enough trending trades and current is not trending, reject
            if trending_pct < self.min_trending_pct and regime_normalized not in trending_regimes:
                # But allow if it's been too long since last trending opportunity
                if len(self.recent_regimes) > 5 and not any(r in trending_regimes for r in self.recent_regimes[-5:]):
                    pass  # Allow trade to prevent complete stagnation
                else:
                    return False, f"Need more trending trades: {trending_pct:.1%} < {self.min_trending_pct:.1%}"
        
        # Filter 2: Volume confirmation (avoid low interest periods)
        if volume_ratio < self.min_volume_ratio:
            return False, f"Low volume: {volume_ratio:.1f} < {self.min_volume_ratio}"
            
        # Filter 3: Volatility filter (avoid dead markets)
        volatility_percentile = self._calculate_volatility_percentile(realized_volatility)
        if volatility_percentile < self.volatility_percentile_min:
            return False, f"Low volatility percentile: {volatility_percentile:.1%} < {self.volatility_percentile_min:.1%}"
            
        # Filter 4: Trend strength - adjusted for regime
        # Be more lenient with trend strength in volatile markets
        if regime_normalized in ['volatile', 'choppy']:
            min_trend_adj = self.min_trend_strength * 0.7
        elif regime_normalized in ['ranging', 'neutral']:
            min_trend_adj = self.min_trend_strength
        else:
            min_trend_adj = self.min_trend_strength * 0.8  # More lenient for trending regimes
            
        if trend_strength < min_trend_adj and regime_normalized in ['ranging', 'neutral', 'choppy']:
            return False, f"Weak trend in {regime_normalized}: ADX {trend_strength:.1f} < {min_trend_adj:.1f}"
            
        # Filter 5: RSI extremes (avoid overextended moves that reverse quickly)
        if rsi > 75 or rsi < 25:
            return False, f"RSI extreme: {rsi:.1f} (avoid overextended moves)"
            
        # Filter 6: Market session filter (avoid low liquidity times)
        if not self._is_good_trading_session():
            return False, "Poor trading session (low liquidity period)"
            
        # All filters passed
        return True, "Market conditions favorable"
    
    def record_trade(self, market_regime: str):
        """Record a trade for statistics tracking"""
        self.total_trade_count += 1
        regime_normalized = self._normalize_regime_name(market_regime)
        
        # Update regime counts
        if regime_normalized in self.regime_trade_counts:
            self.regime_trade_counts[regime_normalized] += 1
        else:
            # Handle new/unknown regimes
            self.regime_trade_counts[regime_normalized] = 1
            
    def _update_statistics(self, volatility: float, volume_ratio: float, regime: str):
        """Update rolling statistics for percentile calculations"""
        self.volatility_history.append(volatility)
        self.volume_ratio_history.append(volume_ratio)
        self.recent_regimes.append(regime)
        
        # Keep only recent history
        if len(self.volatility_history) > self.max_history_length:
            self.volatility_history = self.volatility_history[-self.max_history_length:]
            self.volume_ratio_history = self.volume_ratio_history[-self.max_history_length:]
        if len(self.recent_regimes) > 20:
            self.recent_regimes = self.recent_regimes[-20:]
    
    def _calculate_volatility_percentile(self, current_volatility: float) -> float:
        """Calculate current volatility percentile vs recent history"""
        if len(self.volatility_history) < 10:
            return 0.5  # Neutral if not enough history
            
        volatility_array = np.array(self.volatility_history)
        percentile = np.mean(volatility_array <= current_volatility)
        return percentile
        
    def _is_good_trading_session(self) -> bool:
        """
        Check if current time is during good trading sessions
        Avoid Asian session low liquidity periods
        """
        from datetime import datetime
        
        current_hour = datetime.now().hour
        
        # Good sessions (UTC):
        # London: 8-12
        # NY: 13-17  
        # London-NY overlap: 13-16 (best)
        # Avoid: 22-6 (Asia session - lower volume)
        
        good_sessions = [
            (8, 12),   # London morning
            (13, 17),  # NY session
        ]
        
        for start_hour, end_hour in good_sessions:
            if start_hour <= current_hour < end_hour:
                return True
                
        return False
    
    def _normalize_regime_name(self, regime: str) -> str:
        """Normalize regime names to handle variants"""
        regime_lower = regime.lower()
        
        # Map variations to standard names
        if 'strong' in regime_lower and 'up' in regime_lower:
            return 'strong_uptrend'
        elif 'strong' in regime_lower and 'down' in regime_lower:
            return 'strong_downtrend'
        elif 'weak' in regime_lower and 'up' in regime_lower:
            return 'weak_uptrend'
        elif 'weak' in regime_lower and 'down' in regime_lower:
            return 'weak_downtrend'
        elif 'uptrend' in regime_lower:
            return 'uptrend'
        elif 'downtrend' in regime_lower:
            return 'downtrend'
        elif 'rang' in regime_lower:
            return 'ranging'
        elif 'volatil' in regime_lower:
            return 'volatile'
        elif 'chop' in regime_lower:
            return 'choppy'
        elif 'neutral' in regime_lower:
            return 'neutral'
        else:
            return regime_lower
    
    def get_regime_threshold(self, regime: str) -> float:
        """Get the confidence threshold for a specific regime"""
        regime_normalized = self._normalize_regime_name(regime)
        return self.regime_thresholds.get(regime_normalized, 0.01)  # Default threshold
    
    def get_statistics(self) -> Dict:
        """Get filter statistics"""
        # Calculate percentages for all regimes
        regime_percentages = {}
        for regime, count in self.regime_trade_counts.items():
            regime_percentages[f'{regime}_percentage'] = count / max(1, self.total_trade_count)
        
        # Calculate trending vs non-trending
        trending_regimes = ['uptrend', 'strong_uptrend', 'weak_uptrend', 'downtrend', 'strong_downtrend', 'weak_downtrend']
        trending_count = sum(self.regime_trade_counts.get(r, 0) for r in trending_regimes)
        trending_pct = trending_count / max(1, self.total_trade_count)
        
        return {
            'total_trades': self.total_trade_count,
            'regime_counts': self.regime_trade_counts.copy(),
            'regime_percentages': regime_percentages,
            'trending_percentage': trending_pct,
            'avg_volatility': np.mean(self.volatility_history) if self.volatility_history else 0,
            'avg_volume_ratio': np.mean(self.volume_ratio_history) if self.volume_ratio_history else 0,
            'volatility_samples': len(self.volatility_history),
            'recent_regimes': self.recent_regimes[-10:] if self.recent_regimes else []
        }
    
    def reset_statistics(self):
        """Reset trade statistics (for new backtest runs)"""
        for regime in self.regime_trade_counts:
            self.regime_trade_counts[regime] = 0
        self.total_trade_count = 0
        self.volatility_history = []
        self.volume_ratio_history = []
        self.recent_regimes = []


def create_optimized_signal_processor(config):
    """
    Factory function to create signal processor with market condition filter
    Replaces the old complex signal system with optimized version
    """
    
    class OptimizedSignalProcessor:
        def __init__(self, config):
            self.config = config
            self.market_filter = MarketConditionFilter(config)
            self.logger = logging.getLogger("OptimizedSignalProcessor")
            
        def process_signal(self, features: Dict, model_prediction: float, market_data: Dict):
            """
            Process signal with market condition filtering
            
            Returns:
                signal_strength: float (0.0 if filtered out)
                direction: int (1=long, -1=short, 0=no trade)
                reason: str
            """
            
            # Check market conditions first
            should_trade, filter_reason = self.market_filter.should_trade(market_data)
            
            if not should_trade:
                return 0.0, 0, f"Filtered: {filter_reason}"
                
            # Apply regime-specific confidence thresholds
            signal_config = self.config.get('signal', {})
            default_threshold = signal_config.get('confidence_threshold', 0.008)
            
            # Get regime-specific threshold
            market_regime = market_data.get('market_regime', 'neutral')
            min_confidence = self.market_filter.get_regime_threshold(market_regime)
            
            # Use default if regime threshold is not found
            if min_confidence == 0.01:  # Default fallback value
                min_confidence = default_threshold
            
            abs_prediction = abs(model_prediction)
            
            # Only trade high confidence signals
            if abs_prediction < min_confidence:
                return 0.0, 0, f"Low confidence for {market_regime}: {abs_prediction:.4f} < {min_confidence:.4f}"
                
            # Determine direction and strength
            if model_prediction > min_confidence:
                direction = 1  # Long
                signal_strength = min(1.0, abs_prediction * 2)  # Amplify strong signals
            elif model_prediction < -min_confidence:
                direction = -1  # Short  
                signal_strength = min(1.0, abs_prediction * 2)
            else:
                direction = 0
                signal_strength = 0.0
                
            # Record trade for statistics
            if direction != 0:
                market_regime = market_data.get('market_regime', 'neutral')
                self.market_filter.record_trade(market_regime)
                
            return signal_strength, direction, "Signal accepted"
            
        def get_statistics(self):
            return self.market_filter.get_statistics()
    
    return OptimizedSignalProcessor(config)


if __name__ == "__main__":
    # Test the market condition filter
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.append(str(Path(__file__).parent))
    
    from config import Config
    
    config = Config().config
    filter_obj = MarketConditionFilter(config)
    
    # Test cases based on actual poor performance
    test_cases = [
        {
            'name': 'Neutral Market (should be limited)',
            'data': {
                'market_regime': 'neutral',
                'volume_ratio': 1.0,  # Low volume
                'realized_volatility': 0.015,  # Low volatility
                'adx_14': 18,  # Weak trend
                'rsi_14': 55
            }
        },
        {
            'name': 'Good Trending Market',
            'data': {
                'market_regime': 'uptrend', 
                'volume_ratio': 2.2,  # Good volume
                'realized_volatility': 0.035,  # Good volatility
                'adx_14': 35,  # Strong trend
                'rsi_14': 62
            }
        },
        {
            'name': 'Strong Downtrend',
            'data': {
                'market_regime': 'strong_downtrend',
                'volume_ratio': 2.5,
                'realized_volatility': 0.04,
                'adx_14': 45,
                'rsi_14': 25  # Oversold in downtrend
            }
        }
    ]
    
    print("MARKET CONDITION FILTER TEST")
    print("=" * 40)
    
    for test in test_cases:
        should_trade, reason = filter_obj.should_trade(test['data'])
        status = "ACCEPT" if should_trade else "REJECT"
        print(f"\n{test['name']}:")
        print(f"  {status}: {reason}")
        
        if should_trade:
            filter_obj.record_trade(test['data']['market_regime'])
    
    print(f"\nFilter Statistics:")
    stats = filter_obj.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")