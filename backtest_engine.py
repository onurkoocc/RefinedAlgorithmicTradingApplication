import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from uuid import uuid4
import tensorflow as tf
import gc
import math

# Import PerformanceTracker from risk_manager.py if it's defined there,
# or ensure it's defined/imported correctly if it's a standalone class.
# Assuming PerformanceTracker is defined in risk_manager.py as per your previous structure.
from risk_manager import PerformanceTracker


class PortfolioManager:
    def __init__(self, config, risk_manager_instance):
        self.initial_capital = config.get("risk", "initial_capital", 10000.0)
        self.current_capital = self.initial_capital
        self.risk_manager = risk_manager_instance

    def reset(self):
        self.current_capital = self.initial_capital
        self.risk_manager.current_capital = self.initial_capital
        self.risk_manager.peak_capital = self.initial_capital
        # Correct way to instantiate PerformanceTracker
        self.risk_manager.performance_tracker = PerformanceTracker()

    def update_capital_from_risk_manager(self):
        self.current_capital = self.risk_manager.current_capital


class WalkForwardManager:
    def __init__(self, config):
        self.config = config

    def create_windows(self, df_full: pd.DataFrame, train_size: int, test_size: int, num_target_windows: int) -> List[
        Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
        df_len = len(df_full)
        if df_len < train_size + test_size:
            logging.getLogger("WalkForwardManager").error("Not enough data for any window.")
            return []

        windows = []
        current_test_start_idx = train_size  # Start first test window after initial training window
        window_count = 0

        while current_test_start_idx + test_size <= df_len and (
                num_target_windows <= 0 or window_count < num_target_windows):
            train_start_idx = current_test_start_idx - train_size
            test_end_idx = current_test_start_idx + test_size

            train_df = df_full.iloc[train_start_idx:current_test_start_idx].copy()
            test_df = df_full.iloc[current_test_start_idx:test_end_idx].copy()

            if test_df.empty or train_df.empty or train_df.index[-1] >= test_df.index[0]:
                current_test_start_idx += 1  # Small adjustment if there's an overlap issue
                continue

            window_info = {
                "train_start_time": train_df.index[0], "train_end_time": train_df.index[-1],
                "test_start_time": test_df.index[0], "test_end_time": test_df.index[-1],
                "train_size": len(train_df), "test_size": len(test_df)
            }
            windows.append((train_df, test_df, window_info))

            # Next test window starts immediately after current test window ends
            current_test_start_idx = test_end_idx
            window_count += 1

        return windows


class MarketSimulator:
    def __init__(self, config, portfolio_manager):
        self.config = config
        self.logger = logging.getLogger("MarketSimulator")
        self.portfolio_manager = portfolio_manager

        self.slippage_pct = config.get("backtest", "slippage", 0.0005)
        self.transaction_cost_pct = config.get("backtest", "fixed_cost", 0.0010)

        self.trades_executed = []
        self.current_position = None

    def _apply_entry_costs(self, entry_price: float, quantity: float, direction: str) -> float:
        slipped_price = entry_price * (1 + self.slippage_pct if direction == 'long' else 1 - self.slippage_pct)
        cost = slipped_price * quantity * self.transaction_cost_pct
        return slipped_price, cost

    def _apply_exit_costs(self, exit_price: float, quantity: float, direction: str) -> float:
        slipped_price = exit_price * (1 - self.slippage_pct if direction == 'long' else 1 + self.slippage_pct)
        cost = slipped_price * quantity * self.transaction_cost_pct
        return slipped_price, cost

    def simulate_trading(self, df_test_features: pd.DataFrame, model_predictions: np.ndarray,
                         signal_generator, risk_manager_instance) -> List[Dict]:
        self.trades_executed = []
        self.current_position = None

        if len(df_test_features) != len(model_predictions):
            self.logger.warning(
                f"Test features ({len(df_test_features)}) and predictions ({len(model_predictions)}) length mismatch. Assuming df_test_features is aligned with prediction output.")

        for i in range(len(model_predictions)):
            current_data_point = df_test_features.iloc[[i]]
            current_time = df_test_features.index[i]
            current_actual_close = df_test_features['actual_close'].iloc[i]
            lookback_for_signal = self.config.get("market_regime", "lookback_period", 60)
            market_condition_df = df_test_features.iloc[max(0, i - lookback_for_signal + 1): i + 1]

            if self.current_position:
                market_conditions_for_exit = {
                    'market_phase': self.current_position.get('market_phase_at_entry', 'neutral'),
                    'volatility': market_condition_df['volatility_regime'].iloc[
                        -1] if 'volatility_regime' in market_condition_df else 0.5,
                    'momentum': market_condition_df['trend_strength'].iloc[
                        -1] if 'trend_strength' in market_condition_df else 0,
                    'rsi_14': market_condition_df['rsi_14'].iloc[-1] if 'rsi_14' in market_condition_df else 50,
                    'macd_histogram': market_condition_df[
                        f'macd_histogram_{self.config.get("feature_engineering", "macd_fast", 12)}_{self.config.get("feature_engineering", "macd_slow", 26)}_{self.config.get("feature_engineering", "macd_signal", 9)}'].iloc[
                        -1] if f'macd_histogram_{self.config.get("feature_engineering", "macd_fast", 12)}_{self.config.get("feature_engineering", "macd_slow", 26)}_{self.config.get("feature_engineering", "macd_signal", 9)}' in market_condition_df else 0,
                    'current_time': current_time
                }

                exit_decision = risk_manager_instance.handle_exit_decision(
                    self.current_position, current_actual_close, current_time, market_conditions_for_exit
                )

                if exit_decision.get("exit", False):
                    exit_price_slipped, exit_tx_cost = self._apply_exit_costs(
                        exit_decision.get("exit_price", current_actual_close),
                        self.current_position["quantity"],
                        self.current_position["direction"]
                    )

                    pnl = 0
                    if self.current_position["direction"] == "long":
                        pnl = (exit_price_slipped - self.current_position["entry_price_slipped"]) * \
                              self.current_position["quantity"]
                    else:
                        pnl = (self.current_position["entry_price_slipped"] - exit_price_slipped) * \
                              self.current_position["quantity"]

                    pnl -= (self.current_position["entry_tx_cost"] + exit_tx_cost)

                    trade_record = {**self.current_position,
                                    "exit_time": current_time,
                                    "exit_price_slipped": exit_price_slipped,
                                    "exit_reason": exit_decision["reason"],
                                    "pnl": pnl,
                                    "exit_tx_cost": exit_tx_cost
                                    }
                    self.trades_executed.append(trade_record)
                    risk_manager_instance.update_after_trade(trade_record)
                    self.portfolio_manager.update_capital_from_risk_manager()
                    self.current_position = None

                elif exit_decision.get("partial_exit", False):
                    partial_portion = exit_decision["portion"]
                    partial_quantity = self.current_position["quantity"] * partial_portion
                    remaining_quantity = self.current_position["quantity"] - partial_quantity

                    exit_price_slipped, partial_exit_tx_cost = self._apply_exit_costs(
                        exit_decision.get("price", current_actual_close),
                        partial_quantity,
                        self.current_position["direction"]
                    )

                    partial_pnl = 0
                    if self.current_position["direction"] == "long":
                        partial_pnl = (exit_price_slipped - self.current_position[
                            "entry_price_slipped"]) * partial_quantity
                    else:
                        partial_pnl = (self.current_position[
                                           "entry_price_slipped"] - exit_price_slipped) * partial_quantity

                    pro_rata_entry_cost = self.current_position["entry_tx_cost"] * partial_portion
                    partial_pnl -= (pro_rata_entry_cost + partial_exit_tx_cost)

                    partial_trade_record = {**self.current_position,
                                            "quantity": partial_quantity,
                                            "exit_time": current_time,
                                            "exit_price_slipped": exit_price_slipped,
                                            "exit_reason": exit_decision["reason"],
                                            "pnl": partial_pnl,
                                            "is_partial": True,
                                            "partial_id": exit_decision.get("id"),
                                            "exit_tx_cost": partial_exit_tx_cost
                                            }
                    self.trades_executed.append(partial_trade_record)
                    risk_manager_instance.update_after_trade(partial_trade_record)

                    self.current_position["quantity"] = remaining_quantity
                    self.current_position["entry_tx_cost"] -= pro_rata_entry_cost
                    self.current_position[exit_decision.get("update_position_flag", "partial_exit_taken")] = True
                    self.portfolio_manager.update_capital_from_risk_manager()


                elif exit_decision.get("update_stop", False):
                    self.current_position["current_stop_loss"] = exit_decision["new_stop"]

            if not self.current_position:
                signal_details = signal_generator.generate_signal(model_predictions[i], market_condition_df)

                if signal_details and signal_details.get("signal_type") not in ["NoTrade", None]:
                    entry_price_actual = df_test_features['actual_open'].iloc[i]
                    atr_at_entry = signal_details.get("atr_at_entry", entry_price_actual * 0.015)

                    initial_sl_price = risk_manager_instance.get_initial_stop_loss(signal_details,
                                                                                   entry_price_actual, atr_at_entry)

                    quantity = risk_manager_instance.calculate_position_size(signal_details, entry_price_actual,
                                                                             initial_sl_price)

                    if quantity > 0:
                        entry_price_slipped, entry_tx_cost = self._apply_entry_costs(entry_price_actual, quantity,
                                                                                     signal_details["direction"])
                        self.current_position = {
                            "id": str(uuid4()), "entry_time": current_time,
                            "direction": signal_details["direction"],
                            "entry_price_actual": entry_price_actual,
                            "entry_price_slipped": entry_price_slipped,
                            "quantity": quantity,
                            "initial_stop_loss": initial_sl_price,
                            "current_stop_loss": initial_sl_price,
                            "entry_signal_type": signal_details["signal_type"],
                            "market_phase_at_entry": signal_details.get("market_phase", "neutral"),
                            "ensemble_score": signal_details.get("ensemble_score", 0.5),
                            "atr_at_entry": atr_at_entry,
                            "entry_tx_cost": entry_tx_cost
                        }
                        for k, v in signal_details.items():
                            if k not in self.current_position: self.current_position[k] = v

        if self.current_position:
            last_close_price = df_test_features['actual_close'].iloc[-1]
            exit_price_slipped, exit_tx_cost = self._apply_exit_costs(last_close_price,
                                                                      self.current_position["quantity"],
                                                                      self.current_position["direction"])

            pnl = 0
            if self.current_position["direction"] == "long":
                pnl = (exit_price_slipped - self.current_position["entry_price_slipped"]) * self.current_position[
                    "quantity"]
            else:
                pnl = (self.current_position["entry_price_slipped"] - exit_price_slipped) * self.current_position[
                    "quantity"]
            pnl -= (self.current_position["entry_tx_cost"] + exit_tx_cost)

            trade_record = {**self.current_position,
                            "exit_time": df_test_features.index[-1],
                            "exit_price_slipped": exit_price_slipped,
                            "exit_reason": "EndOfBacktest",
                            "pnl": pnl,
                            "exit_tx_cost": exit_tx_cost
                            }
            self.trades_executed.append(trade_record)
            risk_manager_instance.update_after_trade(trade_record)
            self.portfolio_manager.update_capital_from_risk_manager()
            self.current_position = None

        return self.trades_executed


class BacktestEngine:
    def __init__(self, config, data_preparer, model_trainer, signal_generator, risk_manager):
        self.config = config
        self.logger = logging.getLogger("BacktestEngine")
        self.data_preparer = data_preparer
        self.model_trainer = model_trainer
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager

        from metric_calculator import MetricCalculator
        from exporter import Exporter
        self.metric_calculator = MetricCalculator(config)
        self.exporter = Exporter(config)

        self.train_window_size = config.get("backtest", "train_window_size", 3000)
        self.test_window_size = config.get("backtest", "test_window_size", 600)
        self.walk_forward_target_windows = config.get("backtest", "walk_forward_steps", 10)

        self.portfolio_manager = PortfolioManager(config, self.risk_manager)
        self.market_simulator = MarketSimulator(config, self.portfolio_manager)
        self.walk_forward_manager = WalkForwardManager(config)

        self.all_trades_across_windows = []

        self.use_optuna_features_config = config.get("feature_engineering", "use_optuna_features", False)
        if self.use_optuna_features_config:
            from optuna_feature_selector import OptunaFeatureSelector
            self.optuna_feature_selector = OptunaFeatureSelector(config, self.data_preparer)
        else:
            self.optuna_feature_selector = None
        self.current_feature_set = None

    def run_backtest(self, df_full_features: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting walk-forward backtest...")
        if len(df_full_features) < (self.train_window_size + self.test_window_size):
            self.logger.error("Insufficient data for backtest.")
            return pd.DataFrame()

        walk_forward_windows = self.walk_forward_manager.create_windows(
            df_full_features, self.train_window_size, self.test_window_size, self.walk_forward_target_windows
        )

        if not walk_forward_windows:
            self.logger.error("No walk-forward windows created.")
            return pd.DataFrame()

        self.portfolio_manager.reset()
        self.all_trades_across_windows = []
        window_results_summary = []

        if self.optuna_feature_selector and self.use_optuna_features_config:
            self.logger.info("Performing initial Optuna feature selection...")
            initial_optuna_df = walk_forward_windows[0][0] if walk_forward_windows else df_full_features
            self.current_feature_set = self.optuna_feature_selector.optimize_features(initial_optuna_df)
            if self.current_feature_set:
                self.data_preparer.selected_features = self.current_feature_set
            else:
                self.logger.warning("Optuna did not return features, DataPreparer will use its default logic.")
        elif self.optuna_feature_selector:
            self.current_feature_set = self.optuna_feature_selector.load_best_features()
            if self.current_feature_set:
                self.data_preparer.selected_features = self.current_feature_set

        for i, (train_df, test_df, window_info) in enumerate(walk_forward_windows):
            self.logger.info(
                f"Processing WF Window {i + 1}/{len(walk_forward_windows)}: Train {window_info['train_start_time']} to {window_info['train_end_time']}, Test {window_info['test_start_time']} to {window_info['test_end_time']}")

            optimize_this_iteration = (i + 1) % self.config.get("backtest", "optimize_every_n_iterations", 3) == 0

            if self.config.get("backtest", "adaptive_training", True) and optimize_this_iteration:
                if self.optuna_feature_selector and self.use_optuna_features_config:
                    self.logger.info(f"Re-optimizing features with Optuna for window {i + 1}...")
                    optimized_features = self.optuna_feature_selector.optimize_features(train_df)
                    if optimized_features:
                        self.current_feature_set = optimized_features
                        self.data_preparer.selected_features = self.current_feature_set
                        self.logger.info(
                            f"Using new feature set with {len(self.current_feature_set)} features for window {i + 1}.")
                    else:
                        self.logger.warning("Optuna re-optimization did not return features, using previous set.")

            X_train, y_train, X_val, y_val, df_val_for_callback, fwd_returns_val = \
                self.data_preparer.prepare_data(train_df)

            if X_train.size == 0 or X_val.size == 0:
                self.logger.warning(
                    f"Skipping window {i + 1} due to insufficient training/validation data after preparation.")
                continue

            self.model_trainer.main_model = None
            self.model_trainer.ensemble_models = []
            tf.keras.backend.clear_session()
            gc.collect()

            self.logger.info(f"Training model for window {i + 1}...")
            self.model_trainer.train_model(X_train, y_train, X_val, y_val, df_val_for_callback, fwd_returns_val)

            if not self.model_trainer.main_model and not (
                    self.model_trainer.use_ensemble and self.model_trainer.ensemble_models):
                self.logger.error(f"Model training failed for window {i + 1}. Skipping test.")
                continue

            X_test, _, df_test_actuals, fwd_returns_test_raw = \
                self.data_preparer.prepare_test_data(test_df)

            if X_test.size == 0:
                self.logger.warning(
                    f"Skipping test for window {i + 1} due to insufficient test data after preparation.")
                continue

            model_predictions_scaled = self.model_trainer.predict(X_test)

            window_trades = self.market_simulator.simulate_trading(
                df_test_actuals, model_predictions_scaled, self.signal_generator, self.risk_manager
            )
            self.all_trades_across_windows.extend(window_trades)

            window_pnl = sum(t['pnl'] for t in window_trades)
            self.logger.info(
                f"Window {i + 1} completed. Trades: {len(window_trades)}, PnL: {window_pnl:.2f}, Current Capital: {self.portfolio_manager.current_capital:.2f}")
            window_results_summary.append({
                "window": i + 1,
                "train_start": window_info['train_start_time'],
                "test_end": window_info['test_end_time'],
                "num_trades": len(window_trades),
                "pnl": window_pnl,
                "end_capital": self.risk_manager.current_capital
            })

            del X_train, y_train, X_val, y_val, df_val_for_callback, fwd_returns_val
            del X_test, df_test_actuals, fwd_returns_test_raw, model_predictions_scaled
            tf.keras.backend.clear_session()
            gc.collect()

        if not self.all_trades_across_windows:
            self.logger.warning("No trades were executed across all windows.")
            return pd.DataFrame(window_results_summary)

        final_metrics = self.metric_calculator.calculate_consolidated_metrics(
            self.all_trades_across_windows, self.risk_manager.current_capital, self.risk_manager.initial_capital
        )
        self.logger.info(f"Backtest finished. Final Capital: {self.risk_manager.current_capital:.2f}")
        self.logger.info(f"Overall Metrics: {final_metrics}")

        self.exporter.export_trade_details(self.all_trades_across_windows, self.risk_manager.current_capital,
                                           self.metric_calculator, self.risk_manager.initial_capital)
        self.exporter.export_time_analysis(self.all_trades_across_windows)

        summary_df = pd.DataFrame(window_results_summary)
        overall_metrics_df = pd.DataFrame([final_metrics])
        overall_metrics_df.index = ["Overall"]

        if summary_df.empty:
            summary_df = overall_metrics_df
        else:
            summary_df = pd.concat([summary_df, overall_metrics_df], ignore_index=False)

        return summary_df
