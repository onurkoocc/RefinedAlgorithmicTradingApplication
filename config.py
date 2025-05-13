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
                "min_candles": 7500, # Adjusted for robust feature engineering
                "use_api": True,
                "fetch_extended_data": True,
                "csv_30m": str(self.data_dir / "btc_30m.csv")
            },
            "feature_engineering": {
                "use_chunking": True,
                "chunk_size": 2000,
                "correlation_threshold": 0.95, # Stricter threshold
                "use_optuna_features": False, # Default to False, can be enabled
                "optuna_n_trials": 20, # Reduced for faster iterations if used
                "optuna_timeout": 1800,
                "optuna_metric": "growth_score",
                "feature_selection_method": "importance",
                "use_adaptive_features": False,
                "use_only_essential_features": False, # Allow more features initially
                "essential_features": [
                    'open', 'high', 'low', 'close', 'volume',
                    'taker_buy_base_asset_volume', 'cumulative_delta',
                    'ema_9', 'ema_21', 'ema_50', 'sma_200',
                    'adx_14', 'plus_di_14', 'minus_di_14',
                    'trend_strength',
                    'rsi_14', 'macd_histogram_12_26_9',
                    'atr_14', 'bb_width_20', 'volatility_regime',
                    'market_regime',
                    'bb_percent_b', 'range_position',
                    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                    'cycle_phase',
                    'relative_candle_size', 'candle_body_ratio',
                    'close_vwap_diff',
                    'vol_norm_close_change', 'vol_norm_momentum'
                ],
                "indicators_to_compute": [ # Streamlined list
                    "ema_9", "ema_21", "ema_50", "sma_200",
                    "macd_12_26", "macd_signal_12_26_9", "macd_histogram_12_26_9",
                    "adx_14", "plus_di_14", "minus_di_14",
                    "rsi_14", "bb_middle_20", "bb_upper_20", "bb_lower_20", "bb_width_20",
                    "atr_14", "obv"
                ],
                "ema_short_period": 9, "ema_medium_period": 21, "ema_long_period": 50, "ema_vlong_period": 200,
                "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                "adx_period": 14, "rsi_period": 14, "bb_period": 20, "bb_stddev": 2, "atr_period": 14,
            },
            "risk": {
                "initial_capital": 10000.0,
                "base_risk_per_trade": 0.01, # Reduced for consistency
                "max_risk_per_trade": 0.02,  # Reduced
                "min_risk_per_trade": 0.005, # Reduced
                "max_portfolio_risk": 0.15,  # Reduced
                "max_drawdown_percent": 0.25, # User specified
                "kelly_fraction": 0.3, # More conservative Kelly
                "use_adaptive_kelly": True,
                "volatility_scaling": True,
                "momentum_scaling": True,
                "confidence_scaling": True,
                "streak_sensitivity": 0.10,
                "equity_growth_factor": 0.9,
                "drawdown_risk_factor": 1.2,
                "recovery_factor": 0.7,
                "max_trades_per_day": 5, # To meet 10-30 trades/month
                "min_trade_size_usd": 25.0,
                "min_trade_size_btc": 0.0003,
                "emergency_stop_buffer": 0.0015
            },
            "exit": { # Parameters for the unified exit strategy in RiskManager
                "min_holding_time_hours": 0.5,
                "max_holding_time_hours": 36.0, # Default max
                "early_loss_threshold": -0.015, # Adjusted
                "early_loss_time_hours": 2.0,   # Adjusted
                "stagnant_threshold_pnl_abs": 0.0025, # Adjusted
                "stagnant_time_hours": 4.0,        # Adjusted
                "profit_targets": { # Base targets, will be adjusted by regime
                    "micro": 0.006, "short": 0.012, "medium": 0.020,
                    "long": 0.035, "extended": 0.050
                },
                "market_phase_adjustments": { # Multipliers for profit targets and holding times
                    "uptrend": {"long_profit_factor": 1.2, "short_profit_factor": 0.7, "duration_factor": 1.1},
                    "downtrend": {"long_profit_factor": 0.7, "short_profit_factor": 1.2, "duration_factor": 1.1},
                    "neutral": {"long_profit_factor": 1.0, "short_profit_factor": 1.0, "duration_factor": 1.0},
                    "ranging_at_support": {"long_profit_factor": 1.1, "short_profit_factor": 0.8, "duration_factor": 0.9},
                    "ranging_at_resistance": {"long_profit_factor": 0.8, "short_profit_factor": 1.1, "duration_factor": 0.9},
                    "volatile": {"long_profit_factor": 0.9, "short_profit_factor": 0.9, "duration_factor": 0.8}
                },
                "partial_exit_strategy": { # Simplified
                    "uptrend": [{"threshold": 0.015, "portion": 0.3}, {"threshold": 0.025, "portion": 0.4}],
                    "downtrend": [{"threshold": 0.015, "portion": 0.3}, {"threshold": 0.025, "portion": 0.4}],
                    "neutral": [{"threshold": 0.012, "portion": 0.3}, {"threshold": 0.020, "portion": 0.4}]
                },
                "time_decay_factors": { # PnL must exceed this for given hours
                    4: 0.001, 8: 0.002, 16: 0.004, 24: 0.006, 36: 0.008
                },
                "rsi_extreme_levels": {"overbought": 75, "oversold": 25},
                "quick_profit_base_threshold": 0.008, # Base for quick profit
                "reward_risk_ratio_target": 2.0 # Target R/R
            },
            "signal": {
                "confidence_threshold": 0.0010, # Adjusted
                "ranging_confidence_threshold": 0.0015,
                "strong_signal_threshold": 0.05,
                "atr_multiplier_sl": 2.0, # Tighter SL ATR
                "use_regime_filter": True,
                "use_volatility_filter": True,
                "rsi_overbought": 70, "rsi_oversold": 30,
                "return_threshold": 0.0001,
                "trending_threshold": 25, "ranging_threshold": 20,
                "buy_threshold": 0.0010,
                "strong_buy_threshold": 0.0018
            },
            "model": {
                "sequence_length": 60, # Adjusted
                "horizon": 12,         # Adjusted
                "normalize_method": "feature_specific",
                "train_ratio": 0.75,
                "epochs": 25, # Slightly increased for robust training
                "batch_size": 128, # Adjusted for 4070 Laptop GPU
                "use_mixed_precision": True,
                "early_stopping_patience": 5, # Adjusted
                "dropout_rate": 0.20,
                "l2_reg": 5e-5,
                "attention_enabled": True,
                "use_ensemble": True,
                "ensemble_size": 3,
                "initial_learning_rate": 3e-4, # Adjusted
                "lr_decay_factor": 0.6,
                "direction_loss_weight": 0.6,
                "max_features": 60, # Adjusted
                "model_path": str(self.results_dir / "models" / "best_model.keras"),
                "data_augmentation": {"enabled": True, "noise_level": 0.015, "roll_probability": 0.3, "mask_probability": 0.25, "scale_probability": 0.15},
                "growth_metric": { # For OptimizedGrowthMetricCallback
                    "monthly_target": 0.10, # Mid-point of 8-12%
                    "min_target": 0.07,
                    "max_target": 0.13,
                    "drawdown_weight": 1.8, # Higher penalty for drawdown
                    "consistency_weight": 1.2,
                    "avg_return_weight": 0.8,
                    "threshold_pct": 0.30 # For selecting trades in callback
                },
                "architecture": { # Slightly simplified for faster training if needed
                    "projection_size": 96,
                    "transformer_layers": 2,
                    "transformer_heads": 6,
                    "transformer_dropout": 0.20,
                    "recurrent_units": 50,
                    "dense_units1": 64,
                    "dense_units2": 24
                }
            },
            "backtest": {
                "train_window_size": 3000, # Adjusted
                "test_window_size": 600,   # Adjusted
                "walk_forward_steps": 10, # Sufficient for robust validation
                "slippage": 0.0005,
                "fixed_cost": 0.0010,
                "variable_cost": 0.0005,
                "min_hours_between_trades": 1.0, # Adjusted for trade frequency
                "use_dynamic_slippage": True,
                "adaptive_training": True,
                "train_confidence_threshold": 0.60,
                "optimize_every_n_iterations": 2 # More frequent optimization
            },
            "market_regime": { # Simplified regime detection
                "lookback_period": 60, # Adjusted
                "enable_parameter_blending": True,
                "transition_blend_factor": 0.4,
                "regime_types": ["uptrend", "downtrend", "ranging", "volatile"], # Simplified
                "legacy_regime_mapping": { # Maps more complex internal names to simpler ones if needed
                    "uptrend": "uptrend", "strong_uptrend": "uptrend",
                    "downtrend": "downtrend", "strong_downtrend": "downtrend",
                    "ranging": "ranging", "ranging_at_support": "ranging", "ranging_at_resistance": "ranging",
                    "neutral": "ranging", "choppy": "volatile", "volatile": "volatile",
                    # New mappings from MarketRegimeUtil's 7 types
                    "moderate_uptrend": "uptrend",
                    "moderate_downtrend": "downtrend",
                    "tight_consolidation": "ranging",
                    "volatile_consolidation": "volatile",
                    "choppy_mixed": "volatile"
                },
                "metrics_thresholds": { # Simplified thresholds
                    "uptrend": {"adx": 22, "price_above_ema_pct": 65, "di_diff_gt": 5},
                    "downtrend": {"adx": 22, "price_above_ema_pct_lt": 35, "di_diff_lt": -5},
                    "ranging": {"adx_lt": 20, "bb_width_lt": 0.04}, # bb_width in decimal
                    "volatile": {"atr_pct_gt": 0.015, "bb_width_gt": 0.05} # atr_pct = (atr/close)
                },
                "regime_parameters": { # Centralized parameters for RiskManager and SignalGenerator
                    "atr_multipliers": { # For stop-loss calculations
                        "uptrend": {"long": 2.0, "short": 2.5},
                        "downtrend": {"long": 2.5, "short": 2.0},
                        "ranging": {"long": 1.8, "short": 1.8},
                        "volatile": {"long": 2.8, "short": 2.8}
                    },
                    "profit_target_factors": { # Multiplies base profit targets
                        "uptrend": 1.1, "downtrend": 1.1, "ranging": 0.9, "volatile": 1.0
                    },
                    "max_duration_factors": { # Multiplies base max duration
                        "uptrend": 1.2, "downtrend": 1.2, "ranging": 0.8, "volatile": 0.9
                    },
                    "signal_threshold_factors": { # Multiplies base signal thresholds
                        "uptrend": 0.9, "downtrend": 0.9, "ranging": 1.2, "volatile": 1.1
                    },
                    "position_sizing_factors": { # Multiplies base risk
                        "uptrend": 1.1, "downtrend": 1.1, "ranging": 0.8, "volatile": 0.9
                    }
                }
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

    def get(self, section: str, key: Optional[Any] = None, default: Any = None) -> Any:
        if section not in self.config:
            return default
        if key is None:
            return self.config[section]
        if isinstance(key, str):
            return self.config[section].get(key, default)
        return self.config[section]

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