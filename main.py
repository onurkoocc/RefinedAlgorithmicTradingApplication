import os
import argparse
import logging
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import traceback
import gc

from config import Config
from data_manager import DataManager
from feature_engineering import FeatureEngineer
from data_preparer import DataPreparer
from model import TradingModel
from signal_processor import SignalGenerator
from risk_manager import RiskManager
from backtest_engine import BacktestEngine


def setup_logging(config_instance: Config, log_level=logging.INFO) -> logging.Logger:
    logs_dir = config_instance.results_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"bitcoin_trader_{timestamp}.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def configure_gpu_tf():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"TensorFlow: Enabled memory growth for {len(gpus)} GPU(s).")
            if tf.keras.mixed_precision.global_policy().name != 'mixed_float16':
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logging.info("TensorFlow: Mixed precision 'mixed_float16' set globally.")
            return True
        except RuntimeError as e:
            logging.error(f"TensorFlow: Error setting memory growth or mixed precision: {e}")
    else:
        logging.info("TensorFlow: No GPU found, using CPU.")
    return False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Bitcoin Algorithmic Trading System")
    parser.add_argument("--mode", "-m", type=str, choices=["backtest", "fetch-data"], default="backtest",
                        help="Operation mode")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to custom JSON config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory from config")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory from config")
    parser.add_argument("--use-api", action="store_true", help="Force use of API for data fetching (overrides config)")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Logging level")
    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        config_instance = Config(args.config)
    except Exception as e:
        print(f"FATAL: Error initializing configuration: {e}")
        return 1

    if args.data_dir:
        config_instance.data_dir = Path(args.data_dir).resolve()
        config_instance.set("data", "csv_30m", str(config_instance.data_dir / "btc_30m.csv"))
    if args.results_dir:
        config_instance.results_dir = Path(args.results_dir).resolve()
        config_instance.set("model", "model_path", str(config_instance.results_dir / "models" / "best_model.keras"))

    if args.use_api:
        config_instance.set("data", "use_api", True)

    logger = setup_logging(config_instance, getattr(logging, args.log_level.upper()))
    logger.info(f"Starting application in mode: {args.mode}")
    logger.info(f"Using data directory: {config_instance.data_dir}")
    logger.info(f"Using results directory: {config_instance.results_dir}")

    configure_gpu_tf()

    try:
        data_manager = DataManager(config_instance)

        if args.mode == "fetch-data":
            logger.info("Fetching data...")
            df_ohlcv = data_manager.fetch_all_data(live=True)
            if df_ohlcv.empty:
                logger.error("Data fetching failed.")
                return 1
            logger.info(f"Data fetched successfully. Shape: {df_ohlcv.shape}")
            return 0

        logger.info("Loading data...")
        df_ohlcv = data_manager.fetch_all_data(live=config_instance.get("data", "use_api", False))
        if df_ohlcv.empty:
            logger.error("Failed to load data. Exiting.")
            return 1
        logger.info(f"Data loaded. Shape: {df_ohlcv.shape}")

        logger.info("Engineering features...")
        feature_engineer = FeatureEngineer(config_instance)
        df_features = feature_engineer.process_features(df_ohlcv)
        del df_ohlcv
        gc.collect()
        if df_features.empty:
            logger.error("Feature engineering failed. Exiting.")
            return 1
        logger.info(f"Features engineered. Shape: {df_features.shape}")

        data_preparer = DataPreparer(config_instance)
        model_trainer = TradingModel(config_instance)
        signal_generator = SignalGenerator(config_instance)
        risk_manager = RiskManager(config_instance)

        logger.info("Initializing backtest engine...")
        backtest_engine = BacktestEngine(config_instance, data_preparer, model_trainer, signal_generator,
                                         risk_manager)

        logger.info("Running backtest with integrated training...")
        results_summary_df = backtest_engine.run_backtest(df_features)
        del df_features
        gc.collect()

        if results_summary_df is None or results_summary_df.empty:
            logger.error("Backtest execution failed or produced no results.")
            return 1

        results_file = config_instance.results_dir / "backtest" / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_summary_df.to_csv(results_file)
        logger.info(f"Backtest completed. Summary saved to {results_file}")
        logger.info("\n" + results_summary_df.to_string())

    except Exception as e:
        logger.error(f"An unhandled error occurred in main execution: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
        logging.info("Application finished.")

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)