import os
import argparse
import logging
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import traceback

from backtest_engine import BacktestEngine
from config import Config
from data_manager import DataManager
from data_preparer import DataPreparer
from feature_engineering import FeatureEngineer
from model import TradingModel
from risk_manager import RiskManager
from signal_processor import SignalGenerator
from adaptive_time_management import AdaptiveTimeManager
from optuna_feature_selector import OptunaFeatureSelector


def setup_logging(config, log_level=logging.INFO):
    logs_dir = config.results_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"enhanced_bitcoin_trading_{timestamp}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def configure_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        else:
            return False
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced Bitcoin Trading System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["backtest", "train", "fetch-data", "optimize-exits", "optimize-features"],
        default="backtest",
        help="Operation mode"
    )

    parser.add_argument(
        "--data-folder", "-d",
        type=str,
        default="data",
        help="Folder to store data files"
    )

    parser.add_argument(
        "--output-folder", "-o",
        type=str,
        default="results",
        help="Folder to store results"
    )

    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API to fetch data instead of cached data"
    )

    parser.add_argument(
        "--enhanced-exits",
        action="store_true",
        help="Enable enhanced exit strategy optimizations"
    )

    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for feature optimization"
    )

    return parser.parse_args()


def fetch_data(config, use_api=False):
    if use_api:
        config.set("data", "use_api", True)

    data_manager = DataManager(config)
    df_30m = data_manager.fetch_all_data(live=use_api)

    if df_30m is None or df_30m.empty:
        return None, None

    return df_30m


def create_features(config, df_30m):
    feature_engineer = FeatureEngineer(config)
    df_features = feature_engineer.process_features(df_30m)

    if df_features.empty:
        return None

    return df_features


def train_model(config, df_features):
    data_preparer = DataPreparer(config)
    X_train, y_train, X_val, y_val, df_val, fwd_returns_val = data_preparer.prepare_data(df_features)

    if len(X_train) == 0 or len(y_train) == 0:
        return None

    model = TradingModel(config)
    trained_model = model.train_model(
        X_train, y_train, X_val, y_val, df_val, fwd_returns_val
    )

    tf.keras.backend.clear_session()

    return model


def run_enhanced_backtest(config, df_features):
    data_preparer = DataPreparer(config)
    model = TradingModel(config)
    signal_processor = SignalGenerator(config)
    risk_manager = RiskManager(config)
    time_manager = AdaptiveTimeManager(config)

    backtest_engine = BacktestEngine(
        config,
        data_preparer,
        model,
        signal_processor,
        risk_manager
    )

    backtest_engine.time_manager = time_manager
    results = backtest_engine.run_backtest(df_features)

    if results.empty:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(config.results_dir) / "backtest"
    results_file = results_dir / f"enhanced_backtest_results_{timestamp}.csv"

    results.to_csv(results_file, index=False)

    return results


def optimize_exit_strategies(config, df_features):
    config.set("risk", "enhanced_exit_strategy", True)
    config.set("risk", "momentum_exit_enabled", True)
    config.set("risk", "dynamic_trailing_stop", True)
    config.set("backtest", "enhanced_exit_analysis", True)
    config.set("backtest", "track_exit_performance", True)

    data_preparer = DataPreparer(config)
    model = TradingModel(config)
    signal_processor = SignalGenerator(config)
    risk_manager = RiskManager(config)
    time_manager = AdaptiveTimeManager(config)

    backtest_engine = BacktestEngine(
        config,
        data_preparer,
        model,
        signal_processor,
        risk_manager
    )

    backtest_engine.time_manager = time_manager

    original_steps = config.get("backtest", "walk_forward_steps")
    config.set("backtest", "walk_forward_steps", min(5, original_steps))

    results = backtest_engine.run_backtest(df_features)
    config.set("backtest", "walk_forward_steps", original_steps)

    if results.empty:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(config.results_dir) / "backtest"
    results_file = results_dir / f"exit_strategy_optimization_{timestamp}.csv"
    results.to_csv(results_file, index=False)

    return results


def optimize_features(config, df_features):
    logger = logging.getLogger(__name__)
    logger.info("Starting feature optimization with Optuna")

    # Update Optuna parameters from command line if provided
    if hasattr(args, 'optuna_trials') and args.optuna_trials:
        config.set("feature_engineering", "optuna_n_trials", args.optuna_trials)

    # Create feature selector
    feature_selector = OptunaFeatureSelector(config)

    # Run optimization
    optimized_features = feature_selector.optimize_features(df_features)

    # Report results
    if optimized_features:
        logger.info(f"Feature optimization complete. Selected {len(optimized_features)} features")
        essential_features = config.get("feature_engineering", "essential_features", [])
        non_essential = [f for f in optimized_features if f not in essential_features]

        logger.info(f"Essential features: {len(essential_features)}")
        logger.info(f"Additional optimized features: {len(non_essential)}")

        # Create a DataPreparer and test with the optimized features
        data_preparer = DataPreparer(config)
        data_preparer.optimized_features = optimized_features

        # Run a short test with optimized features
        X_train, y_train, X_val, y_val, df_val, fwd_returns_val = data_preparer.prepare_data(df_features)

        if len(X_train) > 0 and len(y_train) > 0:
            logger.info(f"Test successful with optimized features: X_train shape: {X_train.shape}")
            return optimized_features
        else:
            logger.error("Test failed with optimized features")
            return None
    else:
        logger.error("Feature optimization failed")
        return None


def main():
    global args
    args = parse_args()

    try:
        config = Config(None)
    except Exception as e:
        print(f"Error initializing configuration: {e}")
        return 1

    logger = setup_logging(config)
    configure_gpu()

    if args.data_folder:
        data_path = Path(args.data_folder).resolve()
        config.data_dir = data_path
        config.set("data", "csv_30m", str(data_path / "btc_30m.csv"))

    if args.output_folder:
        results_path = Path(args.output_folder).resolve()
        config.results_dir = results_path
        config.set("model", "model_path", str(results_path / "models" / "best_model.keras"))

    if args.enhanced_exits:
        config.set("risk", "enhanced_exit_strategy", True)
        config.set("risk", "momentum_exit_enabled", True)
        config.set("risk", "dynamic_trailing_stop", True)
        config.set("backtest", "enhanced_exit_analysis", True)
        config.set("backtest", "track_exit_performance", True)

    if args.use_api:
        config.set("data", "use_api", True)

    try:
        if args.mode == "fetch-data":
            fetch_data(config, live=True)

        elif args.mode == "train":
            df_30m = fetch_data(config, use_api=args.use_api)
            if df_30m is None:
                return 1

            df_features = create_features(config, df_30m)
            if df_features is None:
                return 1

            model = train_model(config, df_features)
            if model is None:
                return 1

        elif args.mode == "optimize-exits":
            df_30m = fetch_data(config, use_api=args.use_api)
            if df_30m is None:
                return 1

            df_features = create_features(config, df_30m)
            if df_features is None:
                return 1

            results = optimize_exit_strategies(config, df_features)
            if results is None:
                return 1

        elif args.mode == "optimize-features":
            df_30m = fetch_data(config, use_api=args.use_api)
            if df_30m is None:
                return 1

            df_features = create_features(config, df_30m)
            if df_features is None:
                return 1

            optimized_features = optimize_features(config, df_features)
            if optimized_features is None:
                return 1

        elif args.mode == "backtest":
            df_30m = fetch_data(config, use_api=args.use_api)
            if df_30m is None:
                return 1

            df_features = create_features(config, df_30m)
            if df_features is None:
                return 1

            results = run_enhanced_backtest(config, df_features)
            if results is None:
                return 1

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        return 1

    finally:
        tf.keras.backend.clear_session()

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)