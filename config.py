import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TypeVar, cast, Tuple, Set
from enum import Enum

T = TypeVar('T')


class MarketPhase(Enum):
    STRONG_UPTREND = "Strong Uptrend"
    UPTREND = "Uptrend"
    WEAK_UPTREND = "Weak Uptrend"
    UPTREND_TRANSITION = "Uptrend Transition"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    DOWNTREND_TRANSITION = "Downtrend Transition"
    WEAK_DOWNTREND = "Weak Downtrend"
    DOWNTREND = "Downtrend"
    STRONG_DOWNTREND = "Strong Downtrend"


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
                "use_optuna_features": False,
                "optuna_n_trials": 100,
                "optuna_timeout": 3600,
                "optuna_metric": "growth_score",
                "feature_selection_method": "importance",
                "use_adaptive_features": False,  # Changed from False
                "dynamic_feature_count": 48,  # Increased from 50
                "max_features": 48,  # Increased from 50
                "use_only_essential_features": True,  # Changed from True
                "use_cyclic_features": True,
                "use_liquidity_features": True,
                "use_market_impact_features": True,
                "essential_features": [
                    # Core price data
                    'open', 'high', 'low', 'close', 'volume',

                    # Volume dynamics
                    'taker_buy_base_asset_volume', 'cumulative_delta', 'volume_imbalance_ratio',
                    'volume_price_momentum',

                    # Trend indicators
                    'ema_9', 'ema_21', 'ema_50', 'sma_200',
                    'adx_14', 'plus_di_14', 'minus_di_14',
                    'trend_strength', 'ma_cross_velocity',

                    # Momentum oscillators
                    'rsi_14', 'rsi_roc_3', 'macd_histogram_12_26_9',

                    # Volatility metrics
                    'atr_14', 'bb_width_20', 'volatility_regime',

                    # Market context
                    'market_regime', 'mean_reversion_signal', 'price_impact_ratio',

                    # Support/resistance
                    'bb_percent_b', 'range_position', 'pullback_strength',

                    # Time-based patterns
                    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                    'cycle_phase', 'cycle_position',

                    # Price action patterns
                    'relative_candle_size', 'candle_body_ratio', 'gap',

                    # Order flow
                    'spread_pct', 'close_vwap_diff',

                    # Adaptive volatility features
                    'vol_norm_close_change', 'vol_norm_momentum'
                ],
                "indicators_to_compute": [
                    "ema_9", "ema_21", "ema_50", "sma_200",
                    "macd_12_26", "macd_signal_12_26_9", "macd_histogram_12_26_9", "adx_14", "plus_di_14",
                    "minus_di_14",
                    "rsi_14", "bb_middle_20", "bb_upper_20", "bb_lower_20", "bb_width_20",
                    "atr_14", "obv", "cmf_20"
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
                "base_risk_per_trade": 0.015,
                "max_risk_per_trade": 0.025,
                "min_risk_per_trade": 0.008,
                "max_portfolio_risk": 0.20,
                "max_drawdown_percent": 0.25,
                "kelly_fraction": 0.5,
                "use_adaptive_kelly": True,
                "volatility_scaling": True,
                "momentum_scaling": True,
                "confidence_scaling": True,
                "streak_sensitivity": 0.12,
                "equity_growth_factor": 0.85,
                "drawdown_risk_factor": 1.5,
                "recovery_factor": 0.6,
                "max_correlation_risk": 0.12,
                "max_single_exposure": 0.40,
                "min_hours_between_trades": 1.0,
                "trade_time_decay": 3.0,
                "ranging_position_reduction": 0.6,
                "regime_adjustment_frequency": 50,
                "trend_threshold": 25,
                "max_trades_per_day": 24,
                "min_trade_size_usd": 25.0,
                "min_trade_size_btc": 0.0003,
                "emergency_stop_buffer": 0.002
            },
            "exit": {
                "base_atr_multiplier": 4.2,
                "enable_dynamic_trailing": True,
                "trailing_activation_threshold": 0.022,
                "enable_partial_exits": True,
                "partial_exit_levels": 4,
                "time_based_exits": True,
                "max_trade_duration_hours": 28.0,
                "rsi_extreme_exit": True,
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "macd_reversal_exit": True,
                "enable_early_loss_exit": True,
                "early_loss_threshold": -0.022,
                "early_loss_time": 4.0,
                "enable_quick_profit_exit": True,
                "quick_profit_threshold": 0.009,
                "min_holding_time": 0.5,
                "enable_stagnant_exit": True,
                "stagnant_threshold": 0.005,
                "stagnant_time": 4.0,
                "enable_trailing_take_profit": True,
                "trailing_tp_activation_ratio": 0.65,
                "trailing_tp_atr_multiplier": 1.7,
                "min_stop_percent": 0.020,
                "enable_volatility_tp_scaling": True,
                "volatility_tp_factors": {
                    "low": 0.95,
                    "medium": 1.0,
                    "high": 1.3,
                    "extreme": 1.6
                },
                "enable_emergency_stop_adjustment": True,
                "atr_multiplier_map": {
                    "strong_uptrend": {"long": 4.5, "short": 3.8},
                    "uptrend": {"long": 4.2, "short": 3.6},
                    "neutral": {"long": 3.8, "short": 3.8},
                    "downtrend": {"long": 3.6, "short": 4.2},
                    "strong_downtrend": {"long": 3.8, "short": 4.5},
                    "ranging_at_support": {"long": 3.6, "short": 4.3},
                    "ranging_at_resistance": {"long": 4.3, "short": 3.6},
                    "volatile": {"long": 5.2, "short": 5.2}
                },
                "profit_targets": {
                    "micro": 0.005,  # Increased from 0.003
                    "quick": 0.0075,  # Increased from 0.006
                    "small": 0.012,  # Increased from 0.01
                    "medium": 0.018,  # Increased from 0.015
                    "large": 0.030,  # Increased from 0.025
                    "extended": 0.048  # Increased from 0.04
                }
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
                "confidence_threshold": 0.001,
                "ranging_confidence_threshold": 0.002,
                "strong_signal_threshold": 0.07,
                "atr_multiplier_sl": 2.2,
                "use_regime_filter": True,
                "use_volatility_filter": True,
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "return_threshold": 0.00015,
                "trending_threshold": 26,
                "ranging_threshold": 18,
                "use_fibonacci": True,
                "fibonacci_lookback": 144,
                "fibonacci_adx_threshold": 25,
                "fibonacci_bb_threshold": 0.03,
                "fibonacci_long_entry": 0.382,
                "fibonacci_short_entry": 0.618,
                "fibonacci_long_block_start": 0.618,
                "fibonacci_long_block_end": 1.0,
                "fibonacci_short_block_start": 0.0,
                "fibonacci_short_block_end": 0.382,
                "enable_fibonacci_partial_exits": True,
                "fibonacci_partial_exit_levels_long": [0.618, 0.764, 1.0],
                "fibonacci_partial_exit_levels_short": [0.382, 0.236, 0.0],
                "ensemble_thresholds": {
                    "STRONG_UPTREND": 0.65,
                    "UPTREND": 0.70,
                    "WEAK_UPTREND": 0.65,
                    "UPTREND_TRANSITION": 0.60,
                    "RANGING": 0.65,
                    "VOLATILE": 0.70,
                    "DOWNTREND_TRANSITION": 0.65,
                    "WEAK_DOWNTREND": 0.60,
                    "DOWNTREND": 0.70,
                    "STRONG_DOWNTREND": 0.75,
                },
                "direction_bias": {
                    "STRONG_UPTREND": 0.8,  # Strong long bias (52.33% win rate)
                    "UPTREND": 0.5,  # Moderate long bias (44.19% win rate but needs improvement)
                    "WEAK_UPTREND": 0.3,  # Slight long bias
                    "UPTREND_TRANSITION": 0.9,  # Strong long bias (66.67% win rate)
                    "RANGING": 0.1,  # Slight bias toward longs
                    "VOLATILE": -0.7,  # Strong short bias (100% win rate in shorts)
                    "DOWNTREND_TRANSITION": -0.2,  # Slight short bias
                    "WEAK_DOWNTREND": 0.3,  # Slightly favor longs surprisingly
                    "DOWNTREND": -0.3,  # Moderate short bias
                    "STRONG_DOWNTREND": 0.6,  # Favor longs (avoid shorts at 12.5% win rate)
                },
                "position_sizing": {
                    "STRONG_UPTREND": 1.2,
                    "UPTREND": 0.8,
                    "WEAK_UPTREND": 0.9,
                    "UPTREND_TRANSITION": 1.3,
                    "RANGING": 1.0,
                    "VOLATILE": 0.7,
                    "DOWNTREND_TRANSITION": 0.9,
                    "WEAK_DOWNTREND": 1.3,  # Best performance
                    "DOWNTREND": 0.8,
                    "STRONG_DOWNTREND": 0.7,
                },
                "duration_limits": {
                    "STRONG_UPTREND": 2.0,  # 1-2 hours optimal
                    "UPTREND": 1.0,  # <1 hour optimal (64.71% win rate)
                    "WEAK_UPTREND": 2.0,
                    "UPTREND_TRANSITION": 2.0,
                    "RANGING": 2.0,
                    "VOLATILE": 3.0,  # 2-4 hours optimal (66.67% win rate)
                    "DOWNTREND_TRANSITION": 2.0,
                    "WEAK_DOWNTREND": 2.0,
                    "DOWNTREND": 2.0,
                    "STRONG_DOWNTREND": 1.5,
                },
                "profit_targets": {
                    "STRONG_UPTREND": 1.5,
                    "UPTREND": 1.0,
                    "WEAK_UPTREND": 1.2,
                    "UPTREND_TRANSITION": 1.8,
                    "RANGING": 1.3,
                    "VOLATILE": 2.0,
                    "DOWNTREND_TRANSITION": 1.5,
                    "WEAK_DOWNTREND": 1.7,
                    "DOWNTREND": 1.2,
                    "STRONG_DOWNTREND": 1.0,
                },
                "stop_losses": {
                    "STRONG_UPTREND": 0.7,
                    "UPTREND": 0.6,
                    "WEAK_UPTREND": 0.7,
                    "UPTREND_TRANSITION": 0.8,
                    "RANGING": 0.7,
                    "VOLATILE": 0.6,
                    "DOWNTREND_TRANSITION": 0.7,
                    "WEAK_DOWNTREND": 0.8,
                    "DOWNTREND": 0.6,
                    "STRONG_DOWNTREND": 0.5,
                },
                "quick_profit_threshold": 0.7,
                "stagnant_threshold": 0.6,
                "enabled_phases": {
                    "STRONG_UPTREND": True,
                    "UPTREND": True,
                    "WEAK_UPTREND": True,
                    "UPTREND_TRANSITION": True,
                    "RANGING": True,
                    "VOLATILE": True,
                    "DOWNTREND_TRANSITION": True,
                    "WEAK_DOWNTREND": True,
                    "DOWNTREND": True,
                    "STRONG_DOWNTREND": True,
                },
                "direction_restrictions": {
                    "STRONG_UPTREND": {"long": True, "short": True},
                    "UPTREND": {"long": True, "short": False},  # Disable shorts in uptrends
                    "WEAK_UPTREND": {"long": True, "short": True},
                    "UPTREND_TRANSITION": {"long": True, "short": False},  # Disable shorts
                    "RANGING": {"long": True, "short": True},
                    "VOLATILE": {"long": False, "short": True},  # Disable longs
                    "DOWNTREND_TRANSITION": {"long": True, "short": True},
                    "WEAK_DOWNTREND": {"long": True, "short": True},
                    "DOWNTREND": {"long": True, "short": True},
                    "STRONG_DOWNTREND": {"long": True, "short": False},  # Disable shorts
                }
            },
            "model": {
                "sequence_length": 72,
                "horizon": 16,
                "normalize_method": "feature_specific",
                "train_ratio": 0.7,
                "epochs": 32,
                "batch_size": 128,
                "use_mixed_precision": True,
                "early_stopping_patience": 12,  # Increased slightly, model found best late
                "dropout_rate": 0.3,  # Keep original
                "recurrent_dropout": 0.2,  # Keep original
                "recurrent_units": 48,  # Keep original
                "dense_units1": 48,  # Keep original
                "dense_units2": 24,  # Keep original
                "l2_reg": 1e-3,  # Adjusted from model.py default, reflects change
                # "attention_enabled": False, # Removed this setting as attention is gone
                "initial_learning_rate": 5e-5,  # Keep original
                "lr_decay_factor": 0.85,  # Keep original
                # "direction_loss_weight": 0.0, # Removed this setting as loss changed
                "clipnorm": 1.0,  # Keep original
                "model_path": "path/to/results_dir/models/best_model.keras",
                "data_augmentation": {
                    "enabled": True,
                    "noise_level": 0.01,
                    "roll_probability": 0.3,
                    "mask_probability": 0.2
                },
                "risk_management": {  # Kept defaults, needs analysis outside model.py
                    "max_drawdown_threshold": 0.25,
                    "consecutive_loss_scale": 0.85,
                    "max_position_size": 0.5,
                    "max_trades_per_day": 5,  # This was 5 in callback, config might need sync
                    "min_threshold": 0.001
                },
                "growth_metric_callback": {  # Kept defaults, needs analysis outside model.py
                    "monthly_target": 0.08,
                    "threshold_pct": 0.6,
                    "transaction_cost": 0.001,
                    "drawdown_weight": 1.8,
                    "avg_return_weight": 2.0,
                    "consistency_weight": 1.0,
                    "adaptive_threshold": True,
                    "min_trades_penalty": True
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
            },
            "rl": {
                "enabled": True,
                "mode": "hybrid",  # hybrid, standalone, or disabled
                "state_size": 24,  # Will be calculated dynamically
                "action_size": 3,  # 0: Hold, 1: Buy, 2: Sell
                "buffer_capacity": 10000,
                "batch_size": 128,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_min": 0.01,
                "epsilon_decay": 0.995,
                "learning_rate": 0.001,
                "update_target_frequency": 10,
                "training_frequency": 5,
                "max_hold_periods": 48,
                "transaction_cost": 0.001,
                "reward_scale": 100.0,
                "use_pretrained": True,
                "train_during_backtest": True,
                "confidence_threshold": 0.65,
                "ensemble_weight": 0.5
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