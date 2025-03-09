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
from signal_processor import SignalProcessor


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
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def configure_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            gpu_info = []
            for i, gpu in enumerate(gpus):
                gpu_info.append(f"GPU {i}: {gpu.name}")

            logging.info(f"GPU settings configured for {len(gpus)} GPU(s): {', '.join(gpu_info)}")
            return True
        else:
            logging.info("No GPU found, using CPU")
            return False
    except Exception as e:
        logging.warning(f"Error configuring GPU: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced Bitcoin Trading System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["backtest", "train", "fetch-data"],
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

    return parser.parse_args()


def fetch_data(config, use_api=False):
    logging.info("Fetching 30m data...")

    if use_api:
        config.set("data", "use_api", True)

    data_manager = DataManager(config)
    df_30m = data_manager.fetch_all_data(live=use_api)

    if df_30m is None or df_30m.empty:
        logging.error("Failed to fetch 30-minute data. Exiting.")
        return None, None

    logging.info(f"Data fetched: {len(df_30m)} 30m candles")

    return df_30m


def create_features(config, df_30m):
    logging.info("Creating features with advanced feature engineering...")

    feature_engineer = FeatureEngineer(config)
    df_features = feature_engineer.process_features(df_30m)

    if df_features.empty:
        logging.error("Feature engineering produced an empty DataFrame. Exiting.")
        return None

    feature_columns = list(df_features.columns)
    logging.info(f"Features created: {len(df_features)} rows, {len(feature_columns)} columns")

    return df_features


def train_model(config, df_features):
    logging.info("Preparing data for training...")

    data_preparer = DataPreparer(config)
    X_train, y_train, X_val, y_val, df_val, fwd_returns_val = data_preparer.prepare_data(df_features)

    if len(X_train) == 0 or len(y_train) == 0:
        logging.error("Data preparation failed. No training data available.")
        return None

    logging.info(f"Training data prepared: {X_train.shape}, validation data: {X_val.shape}")

    model = TradingModel(config)
    logging.info(f"Starting model training with {config.get('model', 'epochs')} epochs")

    trained_model = model.train_model(
        X_train, y_train, X_val, y_val, df_val, fwd_returns_val
    )

    tf.keras.backend.clear_session()

    return model


def run_enhanced_backtest(config, df_features):
    logging.info("Setting up enhanced backtesting...")

    data_preparer = DataPreparer(config)
    model = TradingModel(config)
    signal_processor = SignalProcessor(config)
    risk_manager = RiskManager(config)

    backtest_engine = BacktestEngine(
        config,
        data_preparer,
        model,
        signal_processor,
        risk_manager
    )

    logging.info("Running enhanced walk-forward backtest...")
    results = backtest_engine.run_walk_forward(df_features)

    if results.empty:
        logging.error("Backtesting failed or produced no results.")
        return None

    print_backtest_summary(results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(config.results_dir) / "backtest"
    results_file = results_dir / f"enhanced_backtest_results_{timestamp}.csv"

    results.to_csv(results_file, index=False)
    logging.info(f"Backtest results saved to {results_file}")

    return results


def print_backtest_summary(results):
    logging.info("\nEnhanced Backtest Results Summary:")

    consolidated_row = results[results['iteration'] == 999]
    if not consolidated_row.empty:
        final_equity = consolidated_row['final_equity'].iloc[0]
        total_trades = consolidated_row['trades'].iloc[0]
        win_rate = consolidated_row['win_rate'].iloc[0] * 100
        profit_factor = consolidated_row['profit_factor'].iloc[0]
        max_drawdown = consolidated_row['max_drawdown'].iloc[0] * 100
        sharpe = consolidated_row.get('sharpe_ratio', 0).iloc[0]
        return_pct = consolidated_row['return_pct'].iloc[0]

        logging.info(f"Final Equity: ${final_equity:.2f}")
        logging.info(f"Total Return: {return_pct:.2f}%")
        logging.info(f"Total Trades: {total_trades}")
        logging.info(f"Win Rate: {win_rate:.2f}%")
        logging.info(f"Profit Factor: {profit_factor:.2f}")
        logging.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logging.info(f"Sharpe Ratio: {sharpe:.2f}")


def main():
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

    logger.info(f"Enhanced Bitcoin Trading System")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data folder: {config.data_dir}")
    logger.info(f"Output folder: {config.results_dir}")

    if args.use_api:
        config.set("data", "use_api", True)

    try:
        if args.mode == "fetch-data":
            fetch_data(config, live=True)
            logger.info("Data fetching completed")

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

            logger.info("Model training completed")

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

            logger.info("Enhanced backtesting completed")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        return 1

    finally:
        tf.keras.backend.clear_session()

    logger.info("Process completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)