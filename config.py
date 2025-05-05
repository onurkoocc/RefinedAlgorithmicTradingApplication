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
                "use_optuna_features": False,
                "optuna_n_trials": 30,
                "optuna_timeout": 3600,
                "optuna_metric": "growth_score",  # Options: growth_score, r2, directional_accuracy
                "feature_selection_method": "importance",
                "use_adaptive_features": False,
                "use_only_essential_features": True,
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
                    "low": 0.95,  # 5% lower targets in low volatility
                    "medium": 1.0,  # Base level
                    "high": 1.3,  # 30% higher targets in high volatility (increased from 1.2)
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
                "epochs": 20,  # Increased from original value
                "batch_size": 256,
                "use_mixed_precision": True,
                "early_stopping_patience": 7,  # Increased from 5 to allow more training time
                "dropout_rate": 0.22,  # Adjusted to reduce overfitting
                "dropout_min": 0.15,
                "dropout_max": 0.35,
                "recurrent_dropout_max": 0.18,  # Slightly reduced
                "l2_reg": 5.8e-5,  # Increased from 1e-5 for better regularization
                "l2_min": 1e-6,
                "l2_max": 1e-3,
                "attention_enabled": True,
                "use_ensemble": True,
                "ensemble_size": 3,  # New parameter for number of ensemble models
                "optuna_trials": 10,
                "optuna_timeout": 3600,
                "initial_learning_rate": 4.5e-4,  # Adjusted from 3e-4
                "lr_decay_factor": 0.7,  # Changed from 0.4 to 0.7 for slower decay
                "direction_loss_weight": 0.7,  # Increased from 0.6 to emphasize direction
                "max_features": 70,
                "model_path": str(self.results_dir / "models" / "best_model.keras"),
                "data_augmentation": {
                    "enabled": True,
                    "noise_level": 0.02,  # Increased from 0.015 for better robustness
                    "roll_probability": 0.4,
                    "mask_probability": 0.3,  # Increased from 0.25 for better generalization
                    "scale_probability": 0.2  # New parameter for random scaling
                },
                "growth_metric": {
                    "monthly_target": 0.08,  # Target 8% monthly growth
                    "min_target": 0.06,  # Minimum acceptable growth
                    "max_target": 0.12,  # Maximum desired growth (avoid overfitting)
                    "drawdown_weight": 1.5,  # Increased from previous value to penalize drawdowns
                    "consistency_weight": 1.0,  # Increased to prioritize consistency
                    "avg_return_weight": 0.9,  # Decreased slightly to balance with consistency
                    "threshold_pct": 0.35  # Adjusted to be more selective on trades
                },
                "architecture": {
                    "projection_size": 128,  # Increased from 96
                    "transformer_layers": 3,  # Increased from default 2
                    "transformer_heads": 8,
                    "transformer_dropout": 0.26,
                    "recurrent_units": 64,  # Increased from 60
                    "dense_units1": 80,  # Increased from 68
                    "dense_units2": 32  # Increased from 18
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
