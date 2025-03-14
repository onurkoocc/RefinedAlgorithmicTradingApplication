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
from time_based_trade_management import TimeBasedTradeManager


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
        choices=["backtest", "train", "fetch-data", "optimize-exits"],
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

    # Initialize with enhanced exit strategy support
    risk_manager = RiskManager(config)

    # Set up the TimeBasedTradeManager with enhanced settings
    time_manager = TimeBasedTradeManager(config)

    backtest_engine = BacktestEngine(
        config,
        data_preparer,
        model,
        signal_processor,
        risk_manager
    )

    # Replace the time manager with our enhanced version
    backtest_engine.time_manager = time_manager

    logging.info("Running enhanced walk-forward backtest with optimized exit strategies...")
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


def optimize_exit_strategies(config, df_features):
    """Run optimization focused specifically on exit strategies"""
    logging.info("Setting up exit strategy optimization...")

    # Ensure enhanced exit strategy settings are enabled
    config.set("risk", "enhanced_exit_strategy", True)
    config.set("risk", "momentum_exit_enabled", True)
    config.set("risk", "dynamic_trailing_stop", True)
    config.set("backtest", "enhanced_exit_analysis", True)
    config.set("backtest", "track_exit_performance", True)

    # Set up components
    data_preparer = DataPreparer(config)
    model = TradingModel(config)
    signal_processor = SignalProcessor(config)
    risk_manager = RiskManager(config)
    time_manager = TimeBasedTradeManager(config)

    # Create backtest engine with optimized settings
    backtest_engine = BacktestEngine(
        config,
        data_preparer,
        model,
        signal_processor,
        risk_manager
    )

    # Set optimized time manager
    backtest_engine.time_manager = time_manager

    # Run a focused backtest with fewer iterations but more detailed exit analysis
    logging.info("Running exit strategy optimization backtest...")

    # Use a smaller number of iterations for faster optimization
    original_steps = config.get("backtest", "walk_forward_steps")
    config.set("backtest", "walk_forward_steps", min(10, original_steps))

    # Run the backtest
    results = backtest_engine.run_walk_forward(df_features)

    # Restore original settings
    config.set("backtest", "walk_forward_steps", original_steps)

    if results.empty:
        logging.error("Exit strategy optimization failed.")
        return None

    # Print optimization results
    print_exit_optimization_summary(backtest_engine, results)

    # Save optimization results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(config.results_dir) / "backtest"
    results_file = results_dir / f"exit_strategy_optimization_{timestamp}.csv"
    optimization_file = results_dir / f"exit_strategy_recommendations_{timestamp}.txt"

    # Save results and recommendations
    results.to_csv(results_file, index=False)

    with open(optimization_file, 'w') as f:
        f.write("Exit Strategy Optimization Recommendations\n")
        f.write("======================================\n\n")

        # Best exit types
        if hasattr(backtest_engine, 'exit_performance'):
            f.write("Best Performing Exit Types:\n")
            for exit_type, stats in sorted(
                    backtest_engine.exit_performance.items(),
                    key=lambda x: x[1]['avg_pnl'] if x[1]['count'] >= 5 else -9999,
                    reverse=True
            )[:5]:
                if stats['count'] >= 5:  # Only show with enough data
                    f.write(
                        f"- {exit_type}: ${stats['avg_pnl']:.2f} avg PnL, {stats['win_rate'] * 100:.1f}% win rate\n")

        # Best market phases
        if hasattr(backtest_engine, 'best_performing_phases'):
            f.write("\nBest Performing Market Phases:\n")
            for phase in backtest_engine.best_performing_phases[:3]:
                f.write(f"- {phase}\n")

        # Recommended settings
        f.write("\nRecommended Exit Strategy Settings:\n")
        f.write("- Use multi-stage partial exits with 5-6 levels\n")
        f.write(
            f"- Optimal hold time: {time_manager.calculate_optimal_trade_duration(backtest_engine.consolidated_trades).get('optimal_hold_time', 24):.1f} hours\n")
        f.write("- Prioritize QuickProfitTaken exits in neutral market phase\n")
        f.write("- Use tighter stops in ranging_at_resistance phase\n")
        f.write("- Implement momentum-based exits for positions with >1.5% profit\n")

    logging.info(f"Exit strategy optimization results saved to {results_file}")
    logging.info(f"Exit strategy recommendations saved to {optimization_file}")

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


def print_exit_optimization_summary(backtest_engine, results):
    """Print a summary focused on exit strategy performance"""
    logging.info("\nExit Strategy Optimization Results:")

    # Print consolidated results first
    consolidated_row = results[results['iteration'] == 999]
    if not consolidated_row.empty:
        final_equity = consolidated_row['final_equity'].iloc[0]
        win_rate = consolidated_row['win_rate'].iloc[0] * 100
        profit_factor = consolidated_row['profit_factor'].iloc[0]

        logging.info(
            f"Overall Performance: ${final_equity:.2f} final equity, {win_rate:.2f}% win rate, {profit_factor:.2f} profit factor")

    # Print exit type performance (if available)
    if hasattr(backtest_engine, 'exit_performance'):
        logging.info("\nExit Type Performance:")
        for exit_type, stats in sorted(
                backtest_engine.exit_performance.items(),
                key=lambda x: x[1]['avg_pnl'] if x[1]['count'] >= 5 else -9999,
                reverse=True
        )[:5]:
            if stats['count'] >= 5:  # Only show exit types with enough data
                logging.info(
                    f"- {exit_type}: ${stats['avg_pnl']:.2f} avg PnL, {stats['win_rate'] * 100:.1f}% win rate ({stats['count']} trades)")

    # Print market phase performance (if available)
    if hasattr(backtest_engine, 'best_performing_phases'):
        logging.info("\nBest Performing Market Phases:")
        for phase in backtest_engine.best_performing_phases[:3]:
            logging.info(f"- {phase}")

    # Print time-based recommendations
    if hasattr(backtest_engine, 'time_manager'):
        optimal_data = backtest_engine.time_manager.calculate_optimal_trade_duration(
            backtest_engine.consolidated_trades)
        optimal_hold_time = optimal_data.get('optimal_hold_time', 24)
        logging.info(f"\nOptimal hold time: {optimal_hold_time:.1f} hours")

        if 'phase_optimal_durations' in optimal_data:
            logging.info("Phase-specific optimal durations:")
            for phase, duration in optimal_data['phase_optimal_durations'].items():
                logging.info(f"- {phase}: {duration:.1f} hours")


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

    # Enable enhanced exit strategies if requested
    if args.enhanced_exits:
        config.set("risk", "enhanced_exit_strategy", True)
        config.set("risk", "momentum_exit_enabled", True)
        config.set("risk", "dynamic_trailing_stop", True)
        config.set("backtest", "enhanced_exit_analysis", True)
        config.set("backtest", "track_exit_performance", True)
        logger.info("Enhanced exit strategies enabled")

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

            logger.info("Exit strategy optimization completed")

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