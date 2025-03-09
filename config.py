import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TypeVar, cast

T = TypeVar('T')


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"

        self.data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "backtest").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)

        self.logger = self._setup_logger("Config", "config.log")

        self.config = self._get_default_config()

        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)

        self._validate_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "data": {
                "symbol": "BTCUSDT",
                "interval_30m": "30m",
                "min_candles": 25000,
                "use_api": True,
                "fetch_extended_data": True
            },
            "feature_engineering": {
                "use_feature_scaling": True,
                "use_chunking": True,
                "chunk_size": 2000,
                "correlation_threshold": 0.9,
                "keep_essential_features": [
                    "open", "high", "low", "close", "volume",
                    "market_regime", "volatility_regime", "obv",
                    "ema_20", "rsi_14", "macd", "macd_signal", "macd_histogram",
                    "cmf", "mfi", "vwap"
                ],
                "indicators": [
                    "ema_20", "rsi_14", "macd", "macd_signal", "macd_histogram",
                    "bb_width", "atr_14", "obv",
                    "cmf", "mfi", "vwap"
                ],
                "ema_period": 20,
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_stddev": 2,
                "atr_period": 14,
                "cmf_period": 20,
                "mfi_period": 14,
                "vwap_period": 14
            },
            "model": {
                "sequence_length": 72,
                "horizon": 16,
                "train_ratio": 0.7,
                "normalize_method": "zscore",
                "epochs": 20,
                "batch_size": 256,
                "use_mixed_precision": True,
                "early_stopping_patience": 5,
                "dropout_rate": 0.2
            },
            "signal": {
                "confidence_threshold": 0.5,
                "strong_signal_threshold": 0.7,
                "atr_multiplier_sl": 2.5,
                "use_regime_filter": True,
                "use_volatility_filter": True,
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "use_macd_filter": True
            },
            "risk": {
                "initial_capital": 10000.0,
                "max_risk_per_trade": 0.015,
                "max_correlated_exposure": 0.06,
                "volatility_scaling": True,
                "max_drawdown_percent": 0.2,
                "max_trades_per_day": 10,
                "max_consecutive_losses": 3,
                "capital_floor_percent": 0.1,
                "min_trade_size_usd": 50.0,
                "min_trade_size_btc": 0.0005,
                "use_rsi_scaling": True,
                "use_ema_based_exits": True,
                "partial_exit": {
                    "threshold": 0.025,
                    "portion": 0.5
                }
            },
            "backtest": {
                "train_window_size": 5000,
                "test_window_size": 1000,
                "walk_forward_steps": 30,
                "slippage": 0.0005,
                "fixed_cost": 0.001,
                "variable_cost": 0.0005,
                "min_hours_between_trades": 2,
                "track_indicator_metrics": True
            }
        }

    def _validate_config(self) -> None:
        self._validate_string(self.config["data"], "symbol", "BTCUSDT")
        self._validate_positive_int(self.config["data"], "min_candles", 1000)
        self._validate_boolean(self.config["data"], "use_api")
        self._validate_boolean(self.config["data"], "fetch_extended_data")

        self._validate_boolean(self.config["feature_engineering"], "use_feature_scaling")
        self._validate_boolean(self.config["feature_engineering"], "use_chunking")
        self._validate_positive_int(self.config["feature_engineering"], "chunk_size", 500)
        self._validate_float_range(self.config["feature_engineering"], "correlation_threshold", 0.5, 1.0)

        self._validate_positive_int(self.config["feature_engineering"], "ema_period", 10)
        self._validate_positive_int(self.config["feature_engineering"], "rsi_period", 2)
        self._validate_positive_int(self.config["feature_engineering"], "macd_fast", 8)
        self._validate_positive_int(self.config["feature_engineering"], "macd_slow", 12)
        self._validate_positive_int(self.config["feature_engineering"], "macd_signal", 3)

        self._validate_positive_int(self.config["model"], "sequence_length", 10)
        self._validate_positive_int(self.config["model"], "horizon", 1)
        self._validate_float_range(self.config["model"], "train_ratio", 0.5, 0.95)
        self._validate_positive_int(self.config["model"], "epochs", 1)
        self._validate_positive_int(self.config["model"], "batch_size", 16)
        self._validate_string(self.config["model"], "model_path", str(self.results_dir / "models" / "best_model.keras"))
        self._validate_boolean(self.config["model"], "use_mixed_precision")

        self._validate_float_range(self.config["model"], "dropout_rate", 0.0, 0.5)

        self._validate_float_range(self.config["signal"], "confidence_threshold", 0.1, 0.9)
        self._validate_float_range(self.config["signal"], "strong_signal_threshold", 0.5, 0.95)
        self._validate_float_range(self.config["signal"], "atr_multiplier_sl", 0.5, 5.0)
        self._validate_boolean(self.config["signal"], "use_regime_filter")
        self._validate_boolean(self.config["signal"], "use_volatility_filter")

        self._validate_float_range(self.config["signal"], "rsi_overbought", 60, 90)
        self._validate_float_range(self.config["signal"], "rsi_oversold", 10, 40)
        self._validate_boolean(self.config["signal"], "use_macd_filter")

        self._validate_positive_float(self.config["risk"], "initial_capital", 100.0)
        self._validate_float_range(self.config["risk"], "max_risk_per_trade", 0.001, 0.05)
        self._validate_float_range(self.config["risk"], "max_correlated_exposure", 0.01, 0.2)
        self._validate_float_range(self.config["risk"], "max_drawdown_percent", 0.05, 0.5)
        self._validate_positive_int(self.config["risk"], "max_trades_per_day", 1)
        self._validate_positive_int(self.config["risk"], "max_consecutive_losses", 1)
        self._validate_float_range(self.config["risk"], "capital_floor_percent", 0.05, 0.5)
        self._validate_positive_float(self.config["risk"], "min_trade_size_usd", 1.0)
        self._validate_positive_float(self.config["risk"], "min_trade_size_btc", 0.0001)

        self._validate_boolean(self.config["risk"], "use_rsi_scaling")
        self._validate_boolean(self.config["risk"], "use_ema_based_exits")

        if "partial_exit" in self.config["risk"]:
            self._validate_float_range(self.config["risk"]["partial_exit"], "threshold", 0.005, 0.1)
            self._validate_float_range(self.config["risk"]["partial_exit"], "portion", 0.1, 0.9)

        self._validate_positive_int(self.config["backtest"], "train_window_size", 1000)
        self._validate_positive_int(self.config["backtest"], "test_window_size", 100)
        self._validate_positive_int(self.config["backtest"], "walk_forward_steps", 1)
        self._validate_float_range(self.config["backtest"], "slippage", 0.0, 0.01)
        self._validate_float_range(self.config["backtest"], "fixed_cost", 0.0, 0.01)
        self._validate_float_range(self.config["backtest"], "variable_cost", 0.0, 0.01)
        self._validate_positive_int(self.config["backtest"], "min_hours_between_trades", 1)
        self._validate_boolean(self.config["backtest"], "track_indicator_metrics")

    def _load_from_file(self, config_path: str) -> None:
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)

            self._deep_update(self.config, user_config)
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

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
        if key not in section or not isinstance(section[key], (int, float)) or section[key] < min_value or section[
            key] > max_value:
            self.logger.warning(f"Invalid {key} in configuration. Must be between {min_value} and {max_value}")
            if key in section:
                section[key] = min(max(section[key], min_value), max_value)
            else:
                section[key] = min_value

    def _validate_boolean(self, section: Dict[str, Any], key: str) -> None:
        if key not in section or not isinstance(section[key], bool):
            self.logger.warning(f"Invalid {key} in configuration. Must be a boolean.")
            section[key] = section.get(key, True) in (True, 1, "true", "True", "yes", "Yes", "1")

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
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger