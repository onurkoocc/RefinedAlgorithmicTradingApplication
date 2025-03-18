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

        self.logger.info("Configuration initialized successfully")

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
                "min_candles": 25000,
                "use_api": True,
                "fetch_extended_data": True,
                "csv_30m": str(self.data_dir / "btc_30m.csv")
            },
            "feature_engineering": {
                "use_chunking": True,
                "chunk_size": 2000,
                "correlation_threshold": 0.9,
                "essential_features": [
                    "open", "high", "low", "close", "volume",
                    "ema_7", "ema_21", "ema_50", "ema_200",
                    "ema_cross", "hull_ma_16", "macd_histogram",
                    "rsi_14", "stoch_k", "stoch_d", "cci_20", "willr_14",
                    "rsi_divergence",
                    "atr_14", "bb_width", "bb_percent_b", "keltner_width", "natr",
                    "obv", "taker_buy_ratio", "cmf", "mfi", "volume_oscillator",
                    "net_taker_flow", "vwap_distance", "volume_roc",
                    "buy_sell_ratio", "cumulative_delta",
                    "market_regime", "volatility_regime",
                    "pattern_recognition"
                ],
                "indicators_to_compute": [
                    "ema_7", "ema_21", "ema_50", "ema_200",
                    "ema_cross", "hull_ma_16",
                    "macd", "macd_signal", "macd_histogram",
                    "parabolic_sar", "adx", "dmi_plus", "dmi_minus",
                    "rsi_14", "stoch_k", "stoch_d", "roc_10",
                    "cci_20", "willr_14", "mfi",
                    "rsi_divergence",
                    "atr_14", "bb_width", "bb_percent_b",
                    "keltner_width", "donchian_width",
                    "true_range", "natr",
                    "obv", "taker_buy_ratio", "cmf", "volume_oscillator",
                    "vwap", "force_index", "net_taker_flow",
                    "vwap_distance",
                    "volume_roc",
                    "buy_sell_ratio",
                    "volume_profile",
                    "cumulative_delta",
                    "liquidity_index"
                ],
                "ema_short_period": 7,
                "ema_medium_period": 21,
                "ema_long_period": 50,
                "ema_vlong_period": 200,
                "hull_ma_period": 16,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "adx_period": 14,
                "parabolic_sar_acceleration": 0.02,
                "parabolic_sar_max": 0.2,
                "rsi_period": 14,
                "rsi_divergence_period": 14,
                "stoch_period": 14,
                "stoch_smooth": 3,
                "roc_period": 10,
                "cci_period": 20,
                "willr_period": 14,
                "mfi_period": 14,
                "atr_period": 14,
                "bb_period": 20,
                "bb_stddev": 2,
                "keltner_period": 20,
                "keltner_atr_multiple": 2,
                "donchian_period": 20,
                "cmf_period": 14,
                "force_index_period": 13,
                "vwap_period": 14,
                "vwap_distance_period": 20,
                "volume_oscillator_short": 5,
                "volume_oscillator_long": 20,
                "volume_roc_period": 5,
                "cumulative_delta_period": 20
            },
            "model": {
                "sequence_length": 72,
                "horizon": 16,
                "train_ratio": 0.7,
                "normalize_method": "robust",
                "epochs": 25,
                "batch_size": 256,
                "use_mixed_precision": True,
                "early_stopping_patience": 7,
                "dropout_rate": 0.18,
                "l2_reg": 3e-6,
                "attention_enabled": True,
                "use_feature_importance": True,
                "cnn_filters": 72,
                "lstm_units": 108,
                "use_ensemble": True,
                "project_name": "btc_trading",
                "use_lr_schedule": True,
                "initial_learning_rate": 4e-4,
                "model_path": str(Path(__file__).resolve().parent / "results" / "models" / "best_model.keras")
            },
            "signal": {
                "confidence_threshold": 0.0008,
                "strong_signal_threshold": 0.08,
                "atr_multiplier_sl": 2.5,
                "use_regime_filter": True,
                "use_volatility_filter": True,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "return_threshold": 0.0001,
                "phase_return_thresholds": {
                    "neutral": 0.0001,
                    "uptrend": 0.00008,
                    "downtrend": 0.00012,
                    "ranging_at_support": 0.00012,
                    "ranging_at_resistance": 0.00015
                },
                "support_resistance_proximity_pct": 0.03,
                "multi_timeframe_lookbacks": {
                    "short": 5,
                    "medium": 20,
                    "long": 60
                },
                "price_action_patterns": {
                    "bullish_engulfing_weight": 0.8,
                    "bearish_engulfing_weight": 0.8,
                    "doji_weight": 0.4,
                    "hammer_weight": 0.7,
                    "shooting_star_weight": 0.7
                },
                "volume_confirmation_requirements": {
                    "neutral": 0.4,
                    "uptrend": 0.5,
                    "downtrend": 0.5,
                    "ranging_at_support": 0.6,
                    "ranging_at_resistance": 0.7
                }
            },
            "risk": {
                "initial_capital": 10000.0,
                "max_risk_per_trade": 0.025,
                "max_correlated_exposure": 0.12,
                "volatility_scaling": True,
                "max_drawdown_percent": 0.2,
                "max_trades_per_day": 20,
                "max_consecutive_losses": 4,
                "capital_floor_percent": 0.1,
                "min_trade_size_usd": 50.0,
                "min_trade_size_btc": 0.0005,
                "phase_sizing_factors": {
                    "neutral": 1.7,
                    "uptrend": 1.4,
                    "downtrend": 1.2,
                    "ranging_at_support": 1.1,
                    "ranging_at_resistance": 0.7
                },
                "max_risk_reward_ratio": 3.0,
                "base_atr_multiplier": 2.0
            },
            "time_management": {
                "min_profit_taking_hours": 1.8,
                "small_profit_exit_hours": 8.0,
                "stagnant_exit_hours": 10.0,
                "max_trade_duration_hours": 36.0,
                "short_term_lookback": 3,
                "medium_term_lookback": 6,
                "long_term_lookback": 12,
                "profit_targets": {
                    "micro": 0.004,
                    "quick": 0.008,
                    "small": 0.012,
                    "medium": 0.018,
                    "large": 0.03,
                    "extended": 0.045
                },
                "max_position_age": {
                    "neutral": 24.0,
                    "uptrend": 18.0,
                    "downtrend": 16.0,
                    "ranging_at_support": 12.0,
                    "ranging_at_resistance": 6.0,
                    "volatile": 10.0
                },
                "phase_exit_preferences": {
                    "neutral": {
                        "profit_factor": 1.0,
                        "duration_factor": 1.1
                    },
                    "ranging_at_resistance": {
                        "profit_factor": 0.6,
                        "duration_factor": 0.4
                    },
                    "uptrend": {
                        "profit_factor": 1.0,
                        "duration_factor": 1.0
                    },
                    "downtrend": {
                        "profit_factor": 0.8,
                        "duration_factor": 0.7
                    }
                },
                "time_based_risk_factors": {
                    "2": 1.4,
                    "4": 1.3,
                    "8": 1.1,
                    "12": 1.0,
                    "16": 0.9,
                    "24": 0.8,
                    "36": 0.6,
                    "48": 0.5,
                    "72": 0.4
                }
            },
            "backtest": {
                "train_window_size": 5000,
                "test_window_size": 1000,
                "walk_forward_steps": 30,
                "fixed_cost": 0.001,
                "variable_cost": 0.0005,
                "slippage": 0.0005,
                "min_hours_between_trades": 2,
                "track_indicator_metrics": True,
                "enhanced_exit_analysis": True,
                "export_phase_metrics": True,
                "track_exit_performance": True,
                "adaptive_trade_spacing": True
            }
        }

    def _validate_config(self) -> None:
        validators = {
            "data": self._validate_data_config,
            "feature_engineering": self._validate_feature_engineering_config,
            "model": self._validate_model_config,
            "signal": self._validate_signal_config,
            "risk": self._validate_risk_config,
            "time_management": self._validate_time_management_config,
            "backtest": self._validate_backtest_config
        }

        for section, validator in validators.items():
            try:
                if section in self.config:
                    validator(self.config[section])
            except Exception as e:
                self.logger.error(f"Error validating {section} configuration: {e}")
                self.logger.info(f"Using default values for invalid {section} settings")

    def _validate_data_config(self, config: Dict[str, Any]) -> None:
        self._validate_string(config, "symbol", "BTCUSDT")
        self._validate_string(config, "interval_30m", "30m")
        self._validate_positive_int(config, "min_candles", 1000)
        self._validate_boolean(config, "use_api")
        self._validate_boolean(config, "fetch_extended_data")
        self._validate_string(config, "csv_30m", str(self.data_dir / "btc_30m.csv"))

    def _validate_feature_engineering_config(self, config: Dict[str, Any]) -> None:
        self._validate_boolean(config, "use_chunking")
        self._validate_positive_int(config, "chunk_size", 500)
        self._validate_float_range(config, "correlation_threshold", 0.5, 1.0)

        indicator_params = {
            "ema_short_period": (7, self._validate_positive_int),
            "ema_medium_period": (21, self._validate_positive_int),
            "ema_long_period": (50, self._validate_positive_int),
            "ema_vlong_period": (200, self._validate_positive_int),
            "hull_ma_period": (16, self._validate_positive_int),
            "macd_fast": (12, self._validate_positive_int),
            "macd_slow": (26, self._validate_positive_int),
            "macd_signal": (9, self._validate_positive_int),
            "adx_period": (14, self._validate_positive_int),
            "parabolic_sar_acceleration": (0.02, lambda c, k, d: self._validate_float_range(c, k, 0.01, 0.1)),
            "parabolic_sar_max": (0.2, lambda c, k, d: self._validate_float_range(c, k, 0.1, 0.5)),
            "rsi_period": (14, self._validate_positive_int),
            "rsi_divergence_period": (14, self._validate_positive_int),
            "stoch_period": (14, self._validate_positive_int),
            "stoch_smooth": (3, self._validate_positive_int),
            "roc_period": (10, self._validate_positive_int),
            "cci_period": (20, self._validate_positive_int),
            "willr_period": (14, self._validate_positive_int),
            "mfi_period": (14, self._validate_positive_int),
            "atr_period": (14, self._validate_positive_int),
            "bb_period": (20, self._validate_positive_int),
            "bb_stddev": (2.0, self._validate_positive_float),
            "keltner_period": (20, self._validate_positive_int),
            "keltner_atr_multiple": (2.0, self._validate_positive_float),
            "donchian_period": (20, self._validate_positive_int),
            "cmf_period": (14, self._validate_positive_int),
            "force_index_period": (13, self._validate_positive_int),
            "vwap_period": (14, self._validate_positive_int),
            "vwap_distance_period": (20, self._validate_positive_int),
            "volume_oscillator_short": (5, self._validate_positive_int),
            "volume_oscillator_long": (20, self._validate_positive_int),
            "volume_roc_period": (5, self._validate_positive_int),
            "cumulative_delta_period": (20, self._validate_positive_int)
        }

        for param, (default, validator) in indicator_params.items():
            validator(config, param, default)

    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        self._validate_positive_int(config, "sequence_length", 10)
        self._validate_positive_int(config, "horizon", 1)
        self._validate_float_range(config, "train_ratio", 0.5, 0.95)
        self._validate_positive_int(config, "epochs", 1)
        self._validate_positive_int(config, "batch_size", 16)
        self._validate_boolean(config, "use_mixed_precision")
        self._validate_positive_int(config, "early_stopping_patience", 1)
        self._validate_float_range(config, "dropout_rate", 0.0, 0.5)
        self._validate_positive_float(config, "l2_reg", 1e-6)
        self._validate_boolean(config, "attention_enabled")
        self._validate_boolean(config, "use_feature_importance")
        self._validate_positive_int(config, "cnn_filters", 32)
        self._validate_positive_int(config, "lstm_units", 32)
        self._validate_boolean(config, "use_ensemble")
        self._validate_string(config, "project_name", "btc_trading")
        self._validate_boolean(config, "use_lr_schedule")
        self._validate_positive_float(config, "initial_learning_rate", 1e-4)
        self._validate_string(config, "model_path", str(self.results_dir / "models" / "best_model.keras"))

    def _validate_signal_config(self, config: Dict[str, Any]) -> None:
        self._validate_positive_float(config, "confidence_threshold", 0.0001)
        self._validate_positive_float(config, "strong_signal_threshold", 0.01)
        self._validate_positive_float(config, "atr_multiplier_sl", 0.5)
        self._validate_boolean(config, "use_regime_filter")
        self._validate_boolean(config, "use_volatility_filter")
        self._validate_float_range(config, "rsi_overbought", 60, 90)
        self._validate_float_range(config, "rsi_oversold", 10, 40)
        self._validate_positive_float(config, "return_threshold", 0.00001)
        self._validate_positive_float(config, "support_resistance_proximity_pct", 0.001)

        self._validate_nested_dict_float_range(config, "phase_return_thresholds", 0.00001, 0.01)
        self._validate_nested_dict_positive_int(config, "multi_timeframe_lookbacks", 1)
        self._validate_nested_dict_float_range(config, "price_action_patterns", 0.1, 2.0)
        self._validate_nested_dict_float_range(config, "volume_confirmation_requirements", 0.1, 1.0)

    def _validate_risk_config(self, config: Dict[str, Any]) -> None:
        self._validate_positive_float(config, "initial_capital", 100.0)
        self._validate_float_range(config, "max_risk_per_trade", 0.001, 0.05)
        self._validate_float_range(config, "max_correlated_exposure", 0.01, 0.2)
        self._validate_float_range(config, "max_drawdown_percent", 0.05, 0.5)
        self._validate_positive_int(config, "max_trades_per_day", 1)
        self._validate_positive_int(config, "max_consecutive_losses", 1)
        self._validate_float_range(config, "capital_floor_percent", 0.05, 0.5)
        self._validate_positive_float(config, "min_trade_size_usd", 1.0)
        self._validate_positive_float(config, "min_trade_size_btc", 0.0001)
        self._validate_boolean(config, "volatility_scaling")
        self._validate_float_range(config, "max_risk_reward_ratio", 1.0, 10.0)
        self._validate_float_range(config, "base_atr_multiplier", 0.5, 5.0)

        self._validate_nested_dict_float_range(config, "phase_sizing_factors", 0.1, 2.0)

    def _validate_time_management_config(self, config: Dict[str, Any]) -> None:
        self._validate_positive_float(config, "min_profit_taking_hours", 0.5)
        self._validate_positive_float(config, "small_profit_exit_hours", 1.0)
        self._validate_positive_float(config, "stagnant_exit_hours", 1.0)
        self._validate_positive_float(config, "max_trade_duration_hours", 10.0)
        self._validate_positive_int(config, "short_term_lookback", 1)
        self._validate_positive_int(config, "medium_term_lookback", 3)
        self._validate_positive_int(config, "long_term_lookback", 6)

        self._validate_nested_dict_float_range(config, "profit_targets", 0.001, 0.1)
        self._validate_nested_dict_float_range(config, "max_position_age", 1.0, 240.0)

        self._validate_nested_dict_with_keys(
            config, "phase_exit_preferences",
            expected_keys={"profit_factor", "duration_factor"},
            key_validators={
                "profit_factor": lambda v: isinstance(v, (int, float)) and v > 0,
                "duration_factor": lambda v: isinstance(v, (int, float)) and v > 0
            }
        )

        if "time_based_risk_factors" in config:
            for hours_str, factor in list(config["time_based_risk_factors"].items()):
                try:
                    hours = int(hours_str)
                    if not isinstance(factor, (int, float)) or factor <= 0:
                        config["time_based_risk_factors"][hours_str] = 1.0
                        self.logger.warning(f"Invalid risk factor {factor} for {hours} hours, using default")
                except ValueError:
                    self.logger.warning(f"Invalid hour key {hours_str} in time_based_risk_factors, removing")
                    config["time_based_risk_factors"].pop(hours_str, None)

    def _validate_backtest_config(self, config: Dict[str, Any]) -> None:
        self._validate_positive_int(config, "train_window_size", 1000)
        self._validate_positive_int(config, "test_window_size", 100)
        self._validate_positive_int(config, "walk_forward_steps", 1)
        self._validate_float_range(config, "slippage", 0.0, 0.01)
        self._validate_float_range(config, "fixed_cost", 0.0, 0.01)
        self._validate_float_range(config, "variable_cost", 0.0, 0.01)
        self._validate_positive_int(config, "min_hours_between_trades", 1)
        self._validate_boolean(config, "track_indicator_metrics")
        self._validate_boolean(config, "enhanced_exit_analysis")
        self._validate_boolean(config, "export_phase_metrics")
        self._validate_boolean(config, "track_exit_performance")
        self._validate_boolean(config, "adaptive_trade_spacing")

    def _validate_string(self, section: Dict[str, Any], key: str, default: str) -> None:
        if key not in section or not isinstance(section[key], str) or not section[key]:
            self.logger.warning(f"Invalid or missing {key} in configuration. Using default: {default}")
            section[key] = default

    def _validate_positive_int(self, section: Dict[str, Any], key: str, min_value: int) -> None:
        if key not in section or not isinstance(section[key], int) or section[key] < min_value:
            self.logger.warning(f"Invalid {key} in configuration. Must be an integer >= {min_value}")
            section[key] = min_value

    def _validate_positive_float(self, section: Dict[str, Any], key: str, min_value: float) -> None:
        if key not in section or not isinstance(section[key], (int, float)) or section[key] < min_value:
            self.logger.warning(f"Invalid {key} in configuration. Must be a number >= {min_value}")
            section[key] = min_value

    def _validate_float_range(self, section: Dict[str, Any], key: str, min_value: float, max_value: float) -> None:
        if key not in section or not isinstance(section[key], (int, float)) or section[key] < min_value or section[key] > max_value:
            self.logger.warning(f"Invalid {key} in configuration. Must be between {min_value} and {max_value}")
            if key in section:
                section[key] = min(max(section[key], min_value), max_value)
            else:
                section[key] = min_value

    def _validate_boolean(self, section: Dict[str, Any], key: str) -> None:
        if key not in section or not isinstance(section[key], bool):
            self.logger.warning(f"Invalid {key} in configuration. Must be a boolean.")
            section[key] = section.get(key, True) in (True, 1, "true", "True", "yes", "Yes", "1")

    def _validate_nested_dict_float_range(self, config: Dict[str, Any], key: str, min_val: float, max_val: float) -> None:
        if key in config and isinstance(config[key], dict):
            for subkey, value in list(config[key].items()):
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    self.logger.warning(f"Invalid {key}.{subkey} value: {value}. Must be between {min_val} and {max_val}")
                    config[key][subkey] = min(max(float(value) if isinstance(value, (int, float)) else min_val, min_val), max_val)

    def _validate_nested_dict_positive_int(self, config: Dict[str, Any], key: str, min_val: int) -> None:
        if key in config and isinstance(config[key], dict):
            for subkey, value in list(config[key].items()):
                if not isinstance(value, int) or value < min_val:
                    self.logger.warning(f"Invalid {key}.{subkey} value: {value}. Must be an integer >= {min_val}")
                    config[key][subkey] = min_val

    def _validate_nested_dict_with_keys(self, config: Dict[str, Any], key: str,
                                        expected_keys: Set[str], key_validators: Dict[str, callable]) -> None:
        if key in config and isinstance(config[key], dict):
            for subkey, subconfig in list(config[key].items()):
                if not isinstance(subconfig, dict):
                    self.logger.warning(f"Invalid {key}.{subkey}: must be a dictionary")
                    config[key][subkey] = {k: 1.0 for k in expected_keys}
                    continue

                for expected_key in expected_keys:
                    if expected_key not in subconfig:
                        self.logger.warning(f"Missing {expected_key} in {key}.{subkey}")
                        subconfig[expected_key] = 1.0
                    elif expected_key in key_validators and not key_validators[expected_key](subconfig[expected_key]):
                        self.logger.warning(f"Invalid {key}.{subkey}.{expected_key}: {subconfig[expected_key]}")
                        subconfig[expected_key] = 1.0

    def _load_from_file(self, config_path: str) -> None:
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)

            self._deep_update(self.config, user_config)
            self.logger.info(f"Loaded configuration from {config_path}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file {config_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

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
            except Exception as e:
                console.error(f"Error creating log file {log_path}: {e}")

        return logger

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
            self.logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        return self.config