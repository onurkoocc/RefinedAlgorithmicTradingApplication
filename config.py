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
                "use_optuna_features": True,
                "optuna_n_trials": 30,
                "optuna_timeout": 3600,
                "optuna_metric": "growth_score", # Options: growth_score, r2, directional_accuracy
                "feature_selection_method": "importance",
                "use_adaptive_features": False,
                "essential_features": [
                    "open", "high", "low", "close", "volume",
                    "ema_9", "ema_21", "ema_50", "sma_200",
                    "rsi_14", "bb_middle_20", "bb_upper_20", "bb_lower_20", "bb_width_20",
                    "atr_14", "obv", "cmf_20",
                    "adx_14", "plus_di_14", "minus_di_14",
                    "macd_12_26", "macd_signal_12_26_9", "macd_histogram_12_26_9"
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
                "max_drawdown_percent": 0.20,
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
                "base_atr_multiplier": 3.6,  # Increased from 3.0
                "enable_dynamic_trailing": True,
                "trailing_activation_threshold": 0.015,  # Increased from 0.01
                "enable_partial_exits": True,
                "partial_exit_levels": 4,
                "time_based_exits": True,
                "max_trade_duration_hours": 24.0,
                "rsi_extreme_exit": True,
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "macd_reversal_exit": True,
                "enable_early_loss_exit": True,
                "early_loss_threshold": -0.018,  # Changed from -0.012 for more breathing room
                "early_loss_time": 3.0,  # Increased from 2.5
                "enable_quick_profit_exit": True,
                "quick_profit_threshold": 0.008,  # Increased from 0.006
                "min_holding_time": 0.4,  # Increased from 0.3
                "enable_stagnant_exit": True,
                "stagnant_threshold": 0.004,  # Increased from 0.003
                "stagnant_time": 3.5,  # Increased from 3.0
                "enable_trailing_take_profit": True,
                "trailing_tp_activation_ratio": 0.5,  # When to start trailing (50% of avg profitable duration)
                "trailing_tp_atr_multiplier": 1.3,  # Tighter ATR multiplier for take profit trailing
                "min_stop_percent": 0.015,  # 1.5% minimum stop distance
                "enable_volatility_tp_scaling": True,
                "volatility_tp_factors": {
                    "low": 0.95,    # 5% lower targets in low volatility
                    "medium": 1.0,  # Base level
                    "high": 1.3,    # 30% higher targets in high volatility (increased from 1.2)
                    "extreme": 1.6  # 60% higher targets in extreme volatility (increased from 1.4)
                },
                "enable_emergency_stop_adjustment": True,
                "atr_multiplier_map": {
                    "strong_uptrend": {"long": 3.8, "short": 3.2},  # All increased
                    "uptrend": {"long": 3.5, "short": 3.0},
                    "neutral": {"long": 3.2, "short": 3.2},
                    "downtrend": {"long": 3.0, "short": 3.5},
                    "strong_downtrend": {"long": 3.2, "short": 3.8},
                    "ranging_at_support": {"long": 3.0, "short": 3.6},
                    "ranging_at_resistance": {"long": 3.6, "short": 3.0},
                    "volatile": {"long": 4.3, "short": 4.3}  # Significantly increased for volatile markets
                },
                "profit_targets": {
                    "micro": 0.005,    # Increased from 0.003
                    "quick": 0.0075,   # Increased from 0.006
                    "small": 0.012,    # Increased from 0.01
                    "medium": 0.018,   # Increased from 0.015
                    "large": 0.030,    # Increased from 0.025
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
                    "micro": 0.005,    # Increased from 0.003
                    "quick": 0.0075,   # Increased from 0.006
                    "small": 0.012,    # Increased from 0.01
                    "medium": 0.018,   # Increased from 0.015
                    "large": 0.030,    # Increased from 0.025
                    "extended": 0.048  # Increased from 0.04
                },
                "max_position_age": {
                    "neutral": 24.0,                # Increased from 18.0
                    "uptrend": 18.0,                # Increased from 14.0
                    "downtrend": 16.0,              # Increased from 12.0
                    "ranging_at_support": 10.0,     # Increased from 8.0
                    "ranging_at_resistance": 6.0,   # Increased from 4.0
                    "volatile": 10.0                # Increased from 8.0
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
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "return_threshold": 0.00015,
                "trending_threshold": 26,
                "ranging_threshold": 18
            },
            "model": {
                "sequence_length": 72,
                "horizon": 16,
                "normalize_method": "feature_specific",
                "train_ratio": 0.7,
                "epochs": 20,
                "batch_size": 256,
                "use_mixed_precision": True,
                "early_stopping_patience": 5,
                "dropout_rate": 0.25,  # Increased from previous value (targeting mid-range of 0.15-0.35)
                "dropout_min": 0.15,  # New: Min dropout for Optuna search range
                "dropout_max": 0.35,  # New: Max dropout for Optuna search range
                "recurrent_dropout_max": 0.2,  # New: Max recurrent dropout for LSTM/GRUs
                "l2_reg": 1e-5,  # Base L2 regularization value
                "l2_min": 1e-6,  # New: Min L2 regularization for Optuna search
                "l2_max": 1e-3,  # New: Max L2 regularization for Optuna search
                "attention_enabled": True,
                "use_ensemble": True,
                "optuna_trials": 10,
                "optuna_timeout": 3600,
                "initial_learning_rate": 3e-4,
                "lr_decay_factor": 0.4,  # Updated from 0.5 to 0.4 for faster decay
                "direction_loss_weight": 0.6,
                "max_features": 70,
                "model_path": str(self.results_dir / "models" / "best_model.keras"),
                "data_augmentation": {  # New section for data augmentation parameters
                    "enabled": True,
                    "noise_level": 0.015,  # Increased from 0.01
                    "roll_probability": 0.4,  # Increased from 0.3
                    "mask_probability": 0.25  # Increased from 0.2
                }
            },
            "backtest": {
                "train_window_size": 4500,
                "test_window_size": 500,
                "walk_forward_steps": 50,
                "slippage": 0.0004,
                "fixed_cost": 0.0009,
                "variable_cost": 0.00045,
                "min_hours_between_trades": 1,
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