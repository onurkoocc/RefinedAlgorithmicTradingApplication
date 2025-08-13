import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TypeVar, cast, Tuple, Set

T = TypeVar('T')


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"

        self._create_directories()
        self.logger = self._setup_logger("Config", "config.log")
        self.config = self._get_default_config()

        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)

        self._validate_config()

    def _create_directories(self) -> None:
        self.data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "backtest").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "data": {
                "symbol": "BTCUSDT",
                "interval_30m": "30m",
                "min_candles": 15000,
                "use_api": True,
                "fetch_extended_data": True,
                "csv_30m": str(self.data_dir / "btc_30m.csv")
            },
            "feature_engineering": {
                "use_chunking": True,
                "chunk_size": 2000,
                "correlation_threshold": 0.9,
                "use_xgboost_features": False,  # Changed from use_optuna_features
                "xgb_cv_splits": 3,  # XGBoost cross-validation splits
                "xgb_importance_threshold": 0.001,  # Minimum importance threshold
                "xgb_use_gain": True,  # Use gain-based importance
                "feature_selection_method": "importance",
                "use_adaptive_features": False,  # Changed from False
                "dynamic_feature_count": 13,  # Streamlined from 48, added MACD
                "max_features": 13,  # Streamlined from 48, added MACD
                "use_only_essential_features": True,
                "use_streamlined_features": True,  # New flag for streamlined features
                "use_cyclic_features": False,  # Simplified
                "use_liquidity_features": False,  # Simplified
                "use_market_impact_features": False,  # Simplified
                "essential_features": [
                    # Price Action (3 features)
                    'returns',
                    'log_returns', 
                    'realized_volatility',

                    # Volume (2 features)
                    'volume_ratio',
                    'dollar_volume',

                    # Momentum (2 features)
                    'rsi_14',
                    'rate_of_change',

                    # Trend (3 features)
                    'ema_cross_signal',
                    'adx_14',
                    'price_vs_sma',

                    # Market Microstructure (2 features)
                    'high_low_spread',
                    'volume_imbalance',
                    
                    # MACD for momentum (1 feature)
                    'macd_histogram'
                ],
                "indicators_to_compute": [
                    # Only essential indicators for streamlined features
                    "ema_9", "ema_21", "sma_50", "adx_14", "rsi_14"
                ],
                "ema_short_period": 9,
                "ema_medium_period": 21,
                "ema_long_period": 50,
                "ema_vlong_period": 200,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "adx_period": 14,
                "rsi_period": 14,
                "bb_period": 20,
                "bb_stddev": 2,
                "atr_period": 14,
                "cmf_period": 20,
            },
            "risk": {
                "initial_capital": 10000.0,
                # OPTIMIZED: Much more conservative risk after -18% loss
                "base_risk_per_trade": 0.008,  # Halved from 1.5% to 0.8%
                "max_risk_per_trade": 0.015,   # Reduced from 2.5% to 1.5%
                "min_risk_per_trade": 0.005,   # More conservative minimum
                "max_portfolio_risk": 0.12,    # Reduced from 20% to 12%
                "max_drawdown_percent": 0.15,  # Tighter drawdown limit (was 25%)
                
                # OPTIMIZED: More conservative Kelly and position sizing
                "kelly_fraction": 0.3,         # Reduced from 0.5 to 0.3
                "use_adaptive_kelly": True,
                "volatility_scaling": True,
                "momentum_scaling": True,
                "confidence_scaling": True,
                "streak_sensitivity": 0.15,    # More sensitive to losing streaks
                "equity_growth_factor": 0.75,  # More conservative
                "drawdown_risk_factor": 2.0,   # Higher penalty for drawdowns
                "recovery_factor": 0.4,        # More conservative recovery
                
                # OPTIMIZED: Stricter trade limits after 423 trades analysis
                "max_correlation_risk": 0.08,  # Lower correlation risk
                "max_single_exposure": 0.25,   # Reduced from 40% to 25%
                "min_hours_between_trades": 2.0,  # Force longer gaps (was 1.0h)
                "trade_time_decay": 2.0,       # Faster time decay
                "ranging_position_reduction": 0.4,  # Much smaller positions in ranging
                "regime_adjustment_frequency": 30,  # More frequent adjustments
                "trend_threshold": 30,         # Higher trend requirement
                "max_trades_per_day": 8,       # Much stricter (was 24)
                "min_trade_size_usd": 50.0,    # Higher minimum trade size
                "min_trade_size_btc": 0.0003,
                "emergency_stop_buffer": 0.002
            },
            "exit_strategy": {
                # OPTIMIZED: Tighter stops to prevent large losses (-$220 worst trade)
                "base_stop_atr_multiplier": 1.5,  # Tighter than 2.0x to limit damage
                "regime_stop_adjustments": {
                    "volatile": 1.4,      # Wider stops in volatile markets  
                    "trending": 0.7,      # Much tighter in trending - trend should protect
                    "ranging": 1.1,       # Tighter in ranging markets (92% of trades!)
                    "neutral": 0.9,       # Tighter for neutral (main problem area)
                    "strong_uptrend": 0.6,  # Very tight in strong trends
                    "strong_downtrend": 0.6
                },
                
                # OPTIMIZED: Earlier profit taking to capture wins before reversals
                "profit_target_1_atr": 1.2,  # Faster first target - capture early profits
                "profit_target_2_atr": 2.5,  # Reasonable second target
                "profit_target_3_atr": 4.0,  # More conservative third target (was 6x)
                
                # OPTIMIZED: Take more profits early, less late
                "target_1_exit_size": 0.6,  # Exit 60% at first target (was 50%)
                "target_2_exit_size": 0.3,  # Exit 30% at second target
                "target_3_exit_size": 0.1,  # Only 10% rides to final target (was 20%)
                
                # Trailing Stop System
                "trailing_start_atr": 1.0,   # Start trailing after 1x ATR profit
                "trailing_distance_atr": 1.5, # Trail at 1.5x ATR distance
                
                # Time-Based Exits  
                "max_hold_hours": 48,        # Maximum hold: 48 hours
                "flat_position_hours": 24,   # Reduce if flat after 24 hours
                
                # Market Condition Exits
                "enable_regime_change_exit": True,
                "volatility_spike_threshold": 2.5,  # Exit if vol spikes 2.5x
                
                # Risk Management
                "min_reward_risk_ratio": 2.0,  # Target minimum 2:1 R:R
                "max_adverse_excursion_limit": 0.08,  # 8% max adverse move
                
                # Legacy compatibility (keeping old parameters)
                "enable_dynamic_trailing": True,
                "enable_partial_exits": True, 
                "time_based_exits": True,
                "max_trade_duration_hours": 48.0
            },
            "time_management": {
                "min_profit_taking_hours": 1.0,
                "small_profit_exit_hours": 6.0,
                "stagnant_exit_hours": 8.0,
                "max_trade_duration_hours": 24.0,
                "short_term_lookback": 3,
                "medium_term_lookback": 6,
                "long_term_lookback": 12,
                "profit_targets": {
                    "micro": 0.005,  # Increased from 0.003
                    "quick": 0.0075,  # Increased from 0.006
                    "small": 0.012,  # Increased from 0.01
                    "medium": 0.018,  # Increased from 0.015
                    "large": 0.030,  # Increased from 0.025
                    "extended": 0.048  # Increased from 0.04
                },
                "max_position_age": {
                    "neutral": 24.0,  # Increased from 18.0
                    "uptrend": 18.0,  # Increased from 14.0
                    "downtrend": 16.0,  # Increased from 12.0
                    "ranging_at_support": 10.0,  # Increased from 8.0
                    "ranging_at_resistance": 6.0,  # Increased from 4.0
                    "volatile": 10.0  # Increased from 8.0
                },
                "phase_exit_preferences": {
                    "neutral": {
                        "profit_factor": 0.9,
                        "duration_factor": 0.9
                    },
                    "ranging_at_resistance": {
                        "profit_factor": 0.5,
                        "duration_factor": 0.3
                    },
                    "uptrend": {
                        "profit_factor": 1.1,
                        "duration_factor": 0.9
                    },
                    "downtrend": {
                        "profit_factor": 0.7,
                        "duration_factor": 0.6
                    }
                },
                "min_holding_time": 0.4,  # Increased from 0.3
                "trailing_activation_threshold": 0.015  # Increased from 0.01
            },
            "signal": {
                # FIXED: Reasonable thresholds to allow trading while maintaining quality
                "confidence_threshold": 0.0008,  # Reasonable threshold for signal generation
                "ranging_confidence_threshold": 0.0015,  # Slightly higher for ranging markets
                "strong_signal_threshold": 0.12,  # Higher threshold - stronger signals only
                "atr_multiplier_sl": 2.2,
                "use_regime_filter": True,
                "use_volatility_filter": True,
                
                # OPTIMIZED: Stricter RSI filters to avoid overextended moves
                "rsi_overbought": 70,  # More conservative (was 75)
                "rsi_oversold": 30,   # More conservative (was 25)
                
                # OPTIMIZED: Higher return threshold for quality moves
                "return_threshold": 0.0008,  # Much higher quality requirement
                "trending_threshold": 30,    # Higher threshold for trend detection
                "ranging_threshold": 15,     # Lower threshold - stricter ranging detection
                
                # NEW: Market condition filters
                "min_volume_ratio": 1.5,     # Require above-average volume
                "max_neutral_trades_pct": 0.3,  # Limit neutral market trades to 30%
                "volatility_percentile_min": 0.4,  # Avoid low volatility periods
            },
            "model": {
                "sequence_length": 48,  # Reduced from 72 - focus on recent data
                "horizon": 8,           # Reduced from 16 - shorter prediction horizon
                "normalize_method": "feature_specific",
                "train_ratio": 0.8,    # Increased from 0.7 - more training data
                "epochs": 15,          # Further reduced to prevent overfitting
                "batch_size": 32,      # Smaller batch size for better generalization
                "use_mixed_precision": False,  # Disable for stability
                "early_stopping_patience": 4,  # Even more aggressive early stopping
                "dropout_rate": 0.2,   # Reduced dropout - model was too restricted
                "recurrent_dropout": 0.1,  # Much lower recurrent dropout
                "recurrent_units": 32, # Back to 32 - need some complexity
                "dense_units1": 24,    # Increased slightly from 16
                "dense_units2": 12,    # Increased slightly from 8
                "l2_reg": 5e-4,        # Reduced regularization
                "attention_enabled": False,  # Keep disabled for simplicity
                "initial_learning_rate": 5e-5,  # Lower learning rate for stability
                "lr_decay_factor": 0.85,  # More aggressive decay
                "direction_loss_weight": 2.0,  # Emphasize direction prediction
                "clipnorm": 0.5,       # Tighter gradient clipping
                "model_path": "path/to/results_dir/models/best_model.keras",
                "transformer_params": {
                    "projection_size": 12,  # Match feature count
                    "transformer_heads": 2,  # Reduced
                    "transformer_dropout": 0.3,
                    "layers": 1
                },
                "data_augmentation": {
                    "enabled": True,
                    "noise_level": 0.01,
                    "roll_probability": 0.3,
                    "mask_probability": 0.2
                },
                "risk_management": {
                    "max_drawdown_threshold": 0.25,
                    "consecutive_loss_scale": 0.85,
                    "max_position_size": 0.5,
                    "max_trades_per_day": 5,
                    "min_threshold": 0.001
                }
            },
            "backtest": {
                "train_window_size": 4500,
                "test_window_size": 500,
                "walk_forward_steps": 48,
                "slippage": 0.0004,
                "fixed_cost": 0.0009,
                "variable_cost": 0.00045,
                "min_hours_between_trades": 0.5,
                "use_dynamic_slippage": True,
                "adaptive_training": True,
                "train_confidence_threshold": 0.65,
                "use_early_validation": True,
                "track_indicator_metrics": True,
                "enhanced_exit_analysis": True,
                "track_exit_performance": True,
                "optimize_every_n_iterations": 3
            }
        }

    def _setup_logger(self, name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if logger.handlers:
            logger.handlers = []

        console = logging.StreamHandler()
        console.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(console_formatter)
        logger.addHandler(console)

        if log_file:
            log_path = self.results_dir / "logs" / log_file
            try:
                file_handler = logging.FileHandler(log_path)
                file_handler.setLevel(level)
                file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass

        return logger

    def _validate_config(self) -> None:
        if not isinstance(self.config, dict):
            self.config = {}

    def _load_from_file(self, config_path: str) -> None:
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            self._deep_update(self.config, user_config)
        except json.JSONDecodeError:
            pass
        except Exception:
            pass

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        if section not in self.config:
            return default

        if key is None:
            return self.config[section]

        return self.config[section].get(key, default)

    def get_typed(self, section: str, key: str, default: T) -> T:
        value = self.get(section, key, default)
        return cast(T, value)

    def set(self, section: str, key: str, value: Any) -> None:
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def save(self, filepath: str) -> bool:
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception:
            return False

    def get_all(self) -> Dict[str, Any]:
        return self.config
