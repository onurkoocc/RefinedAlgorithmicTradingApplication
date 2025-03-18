import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.utils import compute_class_weight
from pathlib import Path
from uuid import uuid4
import itertools
from collections import deque

from feature_engineering import FeatureEngineer
from time_based_trade_management import TimeBasedTradeManager


class BacktestEngine:
    def __init__(self, config, data_preparer, model, signal_processor, risk_manager):
        self.config = config
        self.logger = logging.getLogger("BacktestEngine")

        self.data_preparer = data_preparer
        self.model = model
        self.signal_processor = signal_processor
        self.risk_manager = risk_manager

        self.time_manager = TimeBasedTradeManager(config)

        self.train_window_size = config.get("backtest", "train_window_size", 5000)
        self.test_window_size = config.get("backtest", "test_window_size", 1000)
        self.walk_forward_steps = config.get("backtest", "walk_forward_steps", 30)
        self.fixed_cost = config.get("backtest", "fixed_cost", 0.001)
        self.variable_cost = config.get("backtest", "variable_cost", 0.0005)
        self.slippage = config.get("backtest", "slippage", 0.0005)
        self.min_hours_between_trades = config.get("backtest", "min_hours_between_trades", 2)

        # Enhanced settings
        self.use_dynamic_slippage = config.get("backtest", "use_dynamic_slippage", True)
        self.adaptive_training = config.get("backtest", "adaptive_training", True)
        self.equity_weighted_metrics = config.get("backtest", "equity_weighted_metrics", True)
        self.use_early_validation = config.get("backtest", "use_early_validation", True)
        self.train_confidence_threshold = config.get("backtest", "train_confidence_threshold", 0.6)

        results_dir = config.results_dir
        self.output_dir = Path(results_dir) / "backtest"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.consolidated_trades = []
        self.feature_engineer = FeatureEngineer(config)

        self.exit_reason_stats = {}
        self.market_phase_stats = {}
        self.total_partial_exits = 0
        self.total_quick_profit_exits = 0

        self.best_exit_reasons = []
        self.best_performing_phases = []
        self.avg_trade_holding_time = 0

        self.indicator_stats = {}

        # Enhanced performance tracking
        self.daily_returns = []
        self.monthly_returns = {}
        self.drawdown_periods = []
        self.equity_curve_points = []
        self.equity_curve_timestamps = []
        self.max_drawdown = 0
        self.max_drawdown_duration = 0
        self.current_drawdown_start = None
        self.peak_equity = self.risk_manager.initial_capital

    def run_walk_forward(self, df_features: pd.DataFrame) -> pd.DataFrame:
        df_len = len(df_features)
        min_required = self.train_window_size + self.test_window_size

        if df_len < min_required:
            self.logger.warning(
                f"Not enough data for train+test. Need at least {min_required}, have {df_len}."
            )
            return pd.DataFrame()

        # Adjust step size based on data length
        step_size = max(self.test_window_size // 2, 200)

        available_steps = (df_len - self.train_window_size - self.test_window_size) // step_size + 1
        max_iterations = min(self.walk_forward_steps, available_steps)

        self.logger.info(f"Starting walk-forward backtest with {max_iterations} iteration(s)")

        all_results = []
        cumulative_capital = self.risk_manager.initial_capital
        self.equity_curve_points = [cumulative_capital]
        self.equity_curve_timestamps = [df_features.index[0]]

        for iteration in range(1, max_iterations + 1):
            start_idx = (iteration - 1) * step_size

            self.logger.info(f"Iteration {iteration}/{max_iterations} | StartIdx={start_idx}")

            result = self._run_iteration(iteration, start_idx, df_features, cumulative_capital)

            if result:
                all_results.append(result)

                if 'final_equity' in result:
                    cumulative_capital = result['final_equity']
                    self.equity_curve_points.append(cumulative_capital)
                    end_idx = min(start_idx + self.train_window_size + self.test_window_size, len(df_features) - 1)
                    self.equity_curve_timestamps.append(df_features.index[end_idx])

                    # Update drawdown statistics
                    self._update_drawdown_stats(cumulative_capital)

                if 'trades' in result and result['trades']:
                    self.consolidated_trades.extend(result['trades'])

                # Update daily and monthly returns
                if 'daily_returns' in result:
                    self.daily_returns.extend(result['daily_returns'])
                    self._update_monthly_returns(result['daily_returns'], df_features, start_idx)

            tf.keras.backend.clear_session()

            if iteration % 3 == 0 and self.consolidated_trades:
                self.time_manager.optimize_time_parameters()

                # Refresh risk manager parameters based on recent performance
                if hasattr(self.risk_manager, 'get_performance_metrics'):
                    metrics = self.risk_manager.get_performance_metrics()
                    self.logger.info(
                        f"Current metrics - Win Rate: {metrics['win_rate']:.2f}, Profit Factor: {metrics['profit_factor']:.2f}")

        if not all_results:
            self.logger.warning("No results generated from walk-forward")
            return pd.DataFrame()

        df_results = pd.DataFrame(all_results)

        if self.consolidated_trades:
            final_equity = df_results.iloc[-1].get('final_equity', self.risk_manager.initial_capital)

            consolidated_metrics = self._calculate_consolidated_metrics(final_equity)

            consolidated_row = {
                'iteration': 999,
                'start_idx': 0,
                'final_equity': final_equity,
                'trades': len(self.consolidated_trades),
                'win_rate': consolidated_metrics.get('win_rate', 0),
                'profit_factor': consolidated_metrics.get('profit_factor', 0),
                'max_drawdown': consolidated_metrics.get('max_drawdown', 0),
                'return_pct': consolidated_metrics.get('return', 0),
                'sharpe_ratio': consolidated_metrics.get('sharpe_ratio', 0),
                'sortino_ratio': consolidated_metrics.get('sortino_ratio', 0),
                'max_drawdown_duration': consolidated_metrics.get('max_drawdown_duration', 0)
            }
            df_results = pd.concat([df_results, pd.DataFrame([consolidated_row])], ignore_index=True)

            self._export_trade_details(final_equity)
            self._export_feature_impact_analysis()

            time_stats = self.time_manager.get_exit_performance_stats()
            self._export_time_analysis(time_stats)

            self._export_exit_strategy_analysis()

            # Export enhanced analytics
            self._export_drawdown_analysis()
            self._export_monthly_performance()

        self.logger.info(f"Walk-forward complete. Produced {len(df_results)} results")
        return df_results

    def _update_drawdown_stats(self, current_equity):
        """Update drawdown statistics based on current equity"""
        self.peak_equity = max(self.peak_equity, current_equity)

        current_drawdown = 0 if self.peak_equity == 0 else (self.peak_equity - current_equity) / self.peak_equity

        if current_drawdown > 0:
            if self.current_drawdown_start is None:
                self.current_drawdown_start = len(self.equity_curve_points) - 1

            # Update max drawdown if needed
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

            # Calculate drawdown duration in days (approximate from number of iterations)
            current_dd_duration = len(self.equity_curve_points) - 1 - self.current_drawdown_start
            self.max_drawdown_duration = max(self.max_drawdown_duration, current_dd_duration)

            # Record drawdown period if significant
            if current_drawdown > 0.05:  # Only track drawdowns > 5%
                self.drawdown_periods.append({
                    'start_idx': self.current_drawdown_start,
                    'current_idx': len(self.equity_curve_points) - 1,
                    'depth': current_drawdown,
                    'duration': current_dd_duration
                })
        elif self.current_drawdown_start is not None:
            # Drawdown ended, reset tracking
            self.current_drawdown_start = None

    def _update_monthly_returns(self, daily_returns, df_features, start_idx):
        """Update monthly returns statistics"""
        if not daily_returns:
            return

        # Get dates from the corresponding data slice
        end_idx = min(start_idx + self.train_window_size + self.test_window_size, len(df_features))
        date_slice = df_features.index[start_idx:end_idx]

        # Make sure we have dates for each return
        dates = date_slice[-len(daily_returns):]

        for i, ret in enumerate(daily_returns):
            if i < len(dates):
                date = dates[i]
                month_key = date.strftime('%Y-%m')

                if month_key not in self.monthly_returns:
                    self.monthly_returns[month_key] = []

                self.monthly_returns[month_key].append(ret)

    def _run_iteration(self, iteration: int, start_idx: int, df_features: pd.DataFrame, cumulative_capital: float) -> \
            Dict[str, Any]:
        if start_idx >= len(df_features):
            self.logger.warning(
                f"Iteration {iteration}: start_idx {start_idx} exceeds dataframe length {len(df_features)}")
            return None

        train_end = min(start_idx + self.train_window_size, len(df_features))
        test_end = min(train_end + self.test_window_size, len(df_features))

        df_train = df_features.iloc[start_idx:train_end].copy()
        df_test = df_features.iloc[train_end:test_end].copy()

        if df_train.empty or df_test.empty:
            self.logger.warning(f"Iteration {iteration}: empty train/test slices. Skipping.")
            return None

        df_train.columns = [col.lower() for col in df_train.columns]
        df_test.columns = [col.lower() for col in df_test.columns]

        try:
            df_train = self._ensure_required_features(df_train)
            df_test = self._ensure_required_features(df_test)

            df_train = self.feature_engineer.compute_advanced_features(df_train)
            df_test = self.feature_engineer.compute_advanced_features(df_test)

            # If using adaptive training, bias recent data more heavily
            if self.adaptive_training and len(self.consolidated_trades) > 0:
                train_data_result = self._prepare_adaptive_training_data(df_train)
            else:
                train_data_result = self.data_preparer.prepare_data(df_train)

            X_train, y_train, X_val, y_val, df_val, fwd_returns_val = train_data_result

            if len(X_train) == 0:
                self.logger.warning(f"Iteration {iteration}: no valid training data after preparation")
                return None

            class_weight_dict = self._compute_class_weights(y_train)

            # Early validation to decide if we should retrain
            if self.use_early_validation and iteration > 1 and hasattr(self.model,
                                                                       'model') and self.model.model is not None:
                validation_score = self._perform_early_validation(X_val, y_val)

                if validation_score >= self.train_confidence_threshold:
                    self.logger.info(
                        f"Iteration {iteration}: Skipping retraining as model validation score is good: {validation_score:.4f}")
                else:
                    self.logger.info(f"Iteration {iteration}: training model on {len(X_train)} sample(s)")
                    self.model.train_model(
                        X_train, y_train,
                        X_val, y_val,
                        df_val, fwd_returns_val,
                        class_weight=class_weight_dict
                    )
            else:
                self.logger.info(f"Iteration {iteration}: training model on {len(X_train)} sample(s)")
                self.model.train_model(
                    X_train, y_train,
                    X_val, y_val,
                    df_val, fwd_returns_val,
                    class_weight=class_weight_dict
                )

        except Exception as e:
            self.logger.error(f"Iteration {iteration}: error in training => {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

        local_risk_manager = type(self.risk_manager)(self.config)
        local_risk_manager.initial_capital = cumulative_capital
        local_risk_manager.current_capital = cumulative_capital
        local_risk_manager.peak_capital = cumulative_capital

        # Transfer relevant state from the global risk manager
        if hasattr(self.risk_manager, 'current_win_streak'):
            local_risk_manager.current_win_streak = self.risk_manager.current_win_streak
            local_risk_manager.current_loss_streak = self.risk_manager.current_loss_streak

        if hasattr(self.risk_manager, 'recent_trades') and self.risk_manager.recent_trades:
            local_risk_manager.recent_trades = self.risk_manager.recent_trades.copy()

        if hasattr(self.risk_manager, 'market_phase_performance'):
            local_risk_manager.market_phase_performance = self.risk_manager.market_phase_performance.copy()

        simulation_result = self._simulate_trading(
            iteration, df_test, local_risk_manager
        )

        final_equity = simulation_result.get("final_equity", local_risk_manager.initial_capital)
        trades = simulation_result.get("trades", [])
        metrics = simulation_result.get("metrics", {})
        daily_returns = simulation_result.get("daily_returns", [])

        result = {
            "iteration": iteration,
            "start_idx": start_idx,
            "final_equity": final_equity,
            "trades": len(trades),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "return_pct": metrics.get("return", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "sortino_ratio": metrics.get("sortino_ratio", 0),
            "max_drawdown_duration": metrics.get("max_drawdown_duration", 0),
            "trades": trades,
            "daily_returns": daily_returns
        }

        # Update the main risk manager with new phase performance data
        if hasattr(local_risk_manager, 'market_phase_performance'):
            self.risk_manager.market_phase_performance = local_risk_manager.market_phase_performance.copy()

        # Update win/loss streaks
        if hasattr(local_risk_manager, 'current_win_streak'):
            self.risk_manager.current_win_streak = local_risk_manager.current_win_streak
            self.risk_manager.current_loss_streak = local_risk_manager.current_loss_streak

        return result

    def _prepare_adaptive_training_data(self, df_train):
        """Prepare training data with recent trades weighted more heavily"""
        # First prepare regular training data
        X_train, y_train, X_val, y_val, df_val, fwd_returns_val = self.data_preparer.prepare_data(df_train)

        # If we don't have enough trades yet, return regular data
        if len(self.consolidated_trades) < 10:
            return X_train, y_train, X_val, y_val, df_val, fwd_returns_val

        # Analyze recent trade performance
        recent_trades = sorted(self.consolidated_trades[-20:], key=lambda t: t.get('exit_time', datetime.now()))
        profitable_conditions = {}

        # Extract market conditions from profitable trades
        for trade in recent_trades:
            if trade.get('pnl', 0) > 0:
                phase = trade.get('market_phase', 'neutral')
                if phase not in profitable_conditions:
                    profitable_conditions[phase] = 0
                profitable_conditions[phase] += 1

        # If we have profitable conditions, adjust validation split to focus on them
        if profitable_conditions:
            total_profitable = sum(profitable_conditions.values())
            # Calculate phase weights
            phase_weights = {phase: count / total_profitable for phase, count in profitable_conditions.items()}

            # Identify samples in validation set matching profitable conditions
            matching_indices = []

            # Only include indices that are within bounds of X_val
            max_valid_index = len(X_val) - 1

            for i, row in enumerate(df_val.itertuples(), 0):
                if i > max_valid_index:
                    # Skip indices that would be out of bounds for X_val
                    continue

                phase = getattr(row, 'market_phase', 'neutral')
                if phase in phase_weights:
                    matching_indices.append(i)

            # If we found matches, weight the validation set
            if matching_indices and len(matching_indices) >= 5:
                # Take more validation samples from phases that have been profitable
                val_size = min(len(matching_indices), len(X_val) // 3)
                sampled_indices = np.random.choice(matching_indices, size=val_size, replace=False)

                # Create weighted validation set
                weighted_X_val = np.vstack([X_val[sampled_indices], X_val])
                weighted_y_val = np.concatenate([y_val[sampled_indices], y_val])
                weighted_fwd_returns = np.concatenate([fwd_returns_val[sampled_indices], fwd_returns_val])

                self.logger.info(f"Using adaptive training with {len(weighted_X_val)} weighted validation samples")
                return X_train, y_train, weighted_X_val, weighted_y_val, df_val, weighted_fwd_returns

        # Return original data if no adaptation was possible
        return X_train, y_train, X_val, y_val, df_val, fwd_returns_val

    def _perform_early_validation(self, X_val, y_val):
        """Perform early validation on current model to see if retraining is needed"""
        if not hasattr(self.model, 'model') or self.model.model is None:
            return 0.0

        try:
            # Make predictions with current model
            predictions = self.model.predict(X_val)

            # Calculate simple directional accuracy
            correct_direction = np.sum(np.sign(predictions.flatten()) == np.sign(y_val))
            total_samples = len(y_val)

            if total_samples > 0:
                accuracy = correct_direction / total_samples
                return accuracy
            return 0.0
        except Exception as e:
            self.logger.warning(f"Error in early validation: {e}")
            return 0.0

    def _simulate_trading(self, iteration: int, df_test: pd.DataFrame, risk_manager) -> Dict[str, Any]:
        try:
            X_test, y_test, df_labeled, fwd_returns_test = self.data_preparer.prepare_test_data(df_test)
        except Exception as e:
            self.logger.error(f"Error in prepare_test_data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"final_equity": risk_manager.initial_capital, "trades": [], "metrics": {}}

        if len(X_test) == 0:
            self.logger.warning("No test sequences available after preparation")
            return {"final_equity": risk_manager.initial_capital, "trades": [], "metrics": {}}

        # Check for a valid model
        has_valid_model = False
        if self.model.model is not None:
            has_valid_model = True
        elif (hasattr(self.model, 'ensemble_models') and
              self.model.ensemble_models is not None and
              len(self.model.ensemble_models.models) > 0):
            has_valid_model = True

        # Attempt to load model if no valid model is found
        if not has_valid_model:
            self.model.load_model()
            has_valid_model = (self.model.model is not None) or (
                    hasattr(self.model, 'ensemble_models') and
                    self.model.ensemble_models is not None and
                    len(self.model.ensemble_models.models) > 0
            )

            if not has_valid_model:
                self.logger.error("No valid model found to predict with. Skipping test.")
                return {"final_equity": risk_manager.initial_capital, "trades": [], "metrics": {}}

        self.logger.info(f"Predicting on test data of length {len(X_test)}")
        predictions = self.model.predict(X_test, batch_size=256)
        if len(predictions) != len(X_test):
            self.logger.error(f"Prediction shape mismatch: got {len(predictions)}, expected {len(X_test)}")
            return {"final_equity": risk_manager.initial_capital, "trades": [], "metrics": {}}

        trades = []
        equity_curve = [risk_manager.current_capital]
        daily_returns = []
        last_day = None

        position = 0
        trade_entry = None
        last_signal_time = None

        seq_len = self.data_preparer.sequence_length

        signal_counts = {"Buy": 0, "Sell": 0, "NoTrade": 0}

        recent_outcomes = []
        consecutive_win_count = 0
        consecutive_loss_count = 0

        market_phase_counts = {}
        exit_reason_counts = {}
        partial_exit_counts = 0
        quick_profit_counts = 0

        # For drawdown tracking
        peak_capital = risk_manager.current_capital
        current_drawdown_start = None
        max_drawdown = 0
        max_drawdown_duration = 0
        drawdown_periods = []

        for i, model_probs in enumerate(predictions):
            if i % 100 == 0:
                self.logger.debug(f"Processing prediction {i}/{len(predictions)}")

            row_idx = i + seq_len
            if row_idx >= len(df_test):
                break

            current_time = df_labeled.index[row_idx]

            # Track daily returns
            current_day = current_time.date()
            if last_day is not None and current_day != last_day:
                # Calculate daily return
                if len(equity_curve) >= 2:
                    daily_return = (equity_curve[-1] / equity_curve[-2]) - 1
                    daily_returns.append(daily_return)
            last_day = current_day

            current_price = self._get_current_price(df_labeled, row_idx)

            if np.isnan(current_price) or current_price <= 0:
                continue

            # Update drawdown tracking
            if risk_manager.current_capital > peak_capital:
                peak_capital = risk_manager.current_capital
                # Drawdown ended
                if current_drawdown_start is not None:
                    drawdown_duration = i - current_drawdown_start
                    max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
                    current_drawdown_start = None
            elif risk_manager.current_capital < peak_capital:
                # In drawdown
                current_drawdown = (peak_capital - risk_manager.current_capital) / peak_capital
                max_drawdown = max(max_drawdown, current_drawdown)

                if current_drawdown_start is None:
                    current_drawdown_start = i

                # Record significant drawdowns
                if current_drawdown > 0.05:  # Only track drawdowns > 5%
                    drawdown_duration = i - current_drawdown_start
                    drawdown_periods.append({
                        'start_idx': current_drawdown_start,
                        'current_idx': i,
                        'depth': current_drawdown,
                        'duration': drawdown_duration
                    })

            try:
                adjusted_signal = {
                    "adaptive_mode": True,
                    "win_streak": consecutive_win_count,
                    "loss_streak": consecutive_loss_count
                }

                if row_idx > 0:
                    signal = self.signal_processor.generate_signal(
                        model_probs,
                        df_labeled.iloc[:row_idx],
                        **adjusted_signal
                    )
                else:
                    signal = {"signal_type": "NoTrade", "reason": "InsufficientHistory"}

                if signal.get("signal_type", "").endswith("Buy"):
                    signal_counts["Buy"] += 1
                elif signal.get("signal_type", "").endswith("Sell"):
                    signal_counts["Sell"] += 1
                else:
                    signal_counts["NoTrade"] += 1

                market_phase = signal.get("market_phase", "neutral")
                if market_phase not in market_phase_counts:
                    market_phase_counts[market_phase] = 0
                market_phase_counts[market_phase] += 1

            except Exception as e:
                self.logger.error(f"Signal error at {current_time}: {e}")
                signal = {"signal_type": "NoTrade", "reason": f"SignalError_{e}"}
                signal_counts["NoTrade"] += 1

            if position != 0 and trade_entry is not None:
                trade_duration = (current_time - trade_entry['entry_time']).total_seconds() / 3600

                atr = self._compute_atr(df_labeled, row_idx, current_price)

                ema_20 = self._get_indicator_value(df_labeled, row_idx, 'ema_20', 'm30_ema_20')
                rsi_14 = self._get_indicator_value(df_labeled, row_idx, 'rsi_14', 'm30_rsi_14', default=50)
                macd_values = self._get_macd_values(df_labeled, row_idx)

                market_conditions = {
                    "market_phase": signal.get("market_phase", "neutral"),
                    "volatility": float(signal.get("volatility", 0.5)),
                    "regime": float(signal.get("regime", 0)),
                    "atr": atr,
                    "momentum": float(signal.get("momentum", 0)),
                    "current_time": current_time
                }

                time_exit_decision = self.time_manager.evaluate_time_based_exit(
                    trade_entry, current_price, current_time, market_conditions
                )

                if time_exit_decision.get("exit", False):
                    exit_reason = time_exit_decision.get("reason", "TimeBasedExit")

                    if exit_reason not in exit_reason_counts:
                        exit_reason_counts[exit_reason] = 0
                    exit_reason_counts[exit_reason] += 1

                    if exit_reason == "QuickProfitTaken":
                        quick_profit_counts += 1

                    trades, position, trade_entry = self._finalize_exit(
                        iteration, risk_manager, trades, trade_entry,
                        current_time, current_price, time_exit_decision,
                        df_labeled, row_idx
                    )

                    trade_pnl = trades[-1].get('pnl', 0)
                    if trade_pnl > 0:
                        consecutive_win_count += 1
                        consecutive_loss_count = 0
                        recent_outcomes.append(1)
                    else:
                        consecutive_loss_count += 1
                        consecutive_win_count = 0
                        recent_outcomes.append(-1)

                    if len(recent_outcomes) > 5:
                        recent_outcomes.pop(0)

                    equity_curve.append(risk_manager.current_capital)
                    last_signal_time = current_time
                    continue

                exit_decision = risk_manager.handle_exit_decision(
                    trade_entry,
                    current_price,
                    atr,
                    trade_duration=trade_duration,
                    market_regime=float(signal.get('regime', 0)),
                    volatility=float(signal.get('volatility', 0.5)),
                    ema_20=ema_20,
                    rsi_14=rsi_14,
                    macd=macd_values.get('macd', 0),
                    macd_signal=macd_values.get('macd_signal', 0),
                    macd_histogram=macd_values.get('macd_histogram', 0),
                    current_time=current_time
                )

                if exit_decision.get("exit", False):
                    exit_reason = exit_decision.get("reason", "RiskExit")

                    if exit_reason not in exit_reason_counts:
                        exit_reason_counts[exit_reason] = 0
                    exit_reason_counts[exit_reason] += 1

                    trades, position, trade_entry = self._finalize_exit(
                        iteration, risk_manager, trades, trade_entry,
                        current_time, current_price, exit_decision,
                        df_labeled, row_idx
                    )

                    trade_pnl = trades[-1].get('pnl', 0)
                    if trade_pnl > 0:
                        consecutive_win_count += 1
                        consecutive_loss_count = 0
                        recent_outcomes.append(1)
                    else:
                        consecutive_loss_count += 1
                        consecutive_win_count = 0
                        recent_outcomes.append(-1)

                    if len(recent_outcomes) > 5:
                        recent_outcomes.pop(0)

                    equity_curve.append(risk_manager.current_capital)
                    last_signal_time = current_time
                else:
                    if exit_decision.get("update_stop", False):
                        new_stop = float(exit_decision.get("new_stop", 0))
                        if not np.isnan(new_stop) and new_stop > 0:
                            if (trade_entry['direction'] == 'long' and new_stop < current_price) or \
                                    (trade_entry['direction'] == 'short' and new_stop > current_price):
                                trade_entry['current_stop_loss'] = new_stop

                    if time_exit_decision.get("update_stop", False):
                        new_stop = float(time_exit_decision.get("new_stop", 0))
                        if not np.isnan(new_stop) and new_stop > 0:
                            if (trade_entry['direction'] == 'long' and new_stop > trade_entry[
                                'current_stop_loss'] and new_stop < current_price) or \
                                    (trade_entry['direction'] == 'short' and new_stop < trade_entry[
                                        'current_stop_loss'] and new_stop > current_price):
                                trade_entry['current_stop_loss'] = new_stop

                    partial_exit = risk_manager.get_partial_exit_level(
                        trade_entry['direction'],
                        trade_entry['entry_price'],
                        current_price
                    )

                    if partial_exit and not trade_entry.get('partial_exit_taken', False):
                        partial_exit_counts += 1

                        trades, position, trade_entry = self._execute_partial_exit(
                            iteration, risk_manager, trades, trade_entry,
                            current_time, current_price, partial_exit,
                            df_labeled, row_idx
                        )
                        equity_curve.append(risk_manager.current_capital)

            if position == 0:
                min_hours_adjusted = self.min_hours_between_trades

                # Adapt trade frequency based on recent performance
                if consecutive_win_count >= 3:
                    min_hours_adjusted = max(1, self.min_hours_between_trades * 0.7)
                elif consecutive_loss_count >= 2:
                    min_hours_adjusted = self.min_hours_between_trades * 1.5

                if last_signal_time is not None:
                    hours_since_last = (current_time - last_signal_time).total_seconds() / 3600
                    if hours_since_last < min_hours_adjusted:
                        continue

                sig_type = signal.get('signal_type', '')
                if sig_type.endswith('Buy') or sig_type.endswith('Sell'):
                    can_trade, max_risk = risk_manager.check_correlation_risk(signal)

                    if can_trade:
                        direction = 'long' if sig_type.endswith('Buy') else 'short'

                        if 'open' in df_labeled.columns:
                            entry_price = float(df_labeled['open'].iloc[row_idx])
                        else:
                            entry_price = current_price

                        # Enhanced slippage model that adapts to market conditions
                        volatility = float(signal.get('volatility', 0.5))
                        market_activity = float(signal.get('volume_roc', 0)) if 'volume_roc' in signal else 0
                        ensemble_score = float(signal.get('ensemble_score', 0.5))

                        # Calculate dynamic slippage based on market conditions
                        if self.use_dynamic_slippage:
                            # Higher volatility and higher volume increase slippage
                            dynamic_slippage = self.slippage * (
                                    0.8 + volatility * 0.6 +
                                    max(0, min(0.3, abs(market_activity) * 0.01)) -
                                    min(0.2, ensemble_score * 0.3)  # Higher confidence can reduce slippage
                            )
                        else:
                            dynamic_slippage = self.slippage

                        slip = entry_price * dynamic_slippage
                        entry_price = entry_price + slip if direction == 'long' else entry_price - slip

                        stop_loss = float(signal.get('stop_loss', 0))
                        if np.isnan(stop_loss) or stop_loss <= 0:
                            atr = self._compute_atr(df_labeled, row_idx - 1, current_price)
                            stop_loss = entry_price * 0.95 if direction == 'long' else entry_price * 1.05

                        if (direction == 'long' and stop_loss >= entry_price) or \
                                (direction == 'short' and stop_loss <= entry_price):
                            self.logger.warning(
                                f"Invalid stop loss {stop_loss} for {direction} at price {entry_price}. Adjusting.")
                            stop_loss = entry_price * 0.95 if direction == 'long' else entry_price * 1.05

                        quantity = risk_manager.calculate_position_size(signal, entry_price, stop_loss)

                        if quantity > 0:
                            trade_entry = {
                                'id': str(uuid4()),
                                'entry_time': current_time,
                                'entry_price': entry_price,
                                'direction': direction,
                                'entry_signal': sig_type,
                                'entry_confidence': float(signal.get('confidence', 0.5)),
                                'market_regime': float(signal.get('regime', 0)),
                                'volatility_regime': float(signal.get('volatility', 0.5)),
                                'initial_stop_loss': stop_loss,
                                'current_stop_loss': stop_loss,
                                'take_profit': float(signal.get('take_profit', 0)),
                                'quantity': quantity,
                                'total_entry_cost': self.fixed_cost + (entry_price * quantity * self.variable_cost),
                                'partial_exit_taken': False,
                                'ensemble_score': float(signal.get('ensemble_score', 0.5)),
                                'market_phase': signal.get('market_phase', 'neutral'),
                                'volume_confirmation': bool(signal.get('volume_confirmation', False)),
                                'trend_strength': float(signal.get('trend_strength', 0.5)),
                                'consecutive_wins': consecutive_win_count,
                                'consecutive_losses': consecutive_loss_count
                            }

                            trade_entry['entry_ema_20'] = self._get_indicator_value(df_labeled, row_idx, 'ema_20',
                                                                                    'm30_ema_20')
                            trade_entry['entry_rsi_14'] = self._get_indicator_value(df_labeled, row_idx, 'rsi_14',
                                                                                    'm30_rsi_14', default=50)
                            macd_values = self._get_macd_values(df_labeled, row_idx)
                            trade_entry['entry_macd'] = macd_values.get('macd', 0)
                            trade_entry['entry_macd_signal'] = macd_values.get('macd_signal', 0)
                            trade_entry['entry_macd_histogram'] = macd_values.get('macd_histogram', 0)

                            position = 1 if direction == 'long' else -1
                            last_signal_time = current_time

                            self.logger.debug(
                                f"Opened {direction} position at {current_time}: {quantity} @ ${entry_price}"
                            )

        self.logger.info(
            f"Signal statistics: Buy: {signal_counts['Buy']}, Sell: {signal_counts['Sell']}, NoTrade: {signal_counts['NoTrade']}")

        self.total_partial_exits += partial_exit_counts
        self.total_quick_profit_exits += quick_profit_counts

        for phase, count in market_phase_counts.items():
            if phase not in self.market_phase_stats:
                self.market_phase_stats[phase] = 0
            self.market_phase_stats[phase] += count

        for reason, count in exit_reason_counts.items():
            if reason not in self.exit_reason_stats:
                self.exit_reason_stats[reason] = 0
            self.exit_reason_stats[reason] += count

        if position != 0 and trade_entry is not None:
            trade_entry['exit_signal'] = "EndOfTest"

            final_exit_price = self._get_current_price(df_labeled, len(df_labeled) - 1)
            if np.isnan(final_exit_price) or final_exit_price <= 0:
                final_exit_price = float(trade_entry['entry_price'])

            exit_decision = {"exit": True, "reason": "EndOfTest", "exit_price": final_exit_price}
            trades, position, trade_entry = self._finalize_exit(
                iteration, risk_manager, trades, trade_entry,
                df_labeled.index[-1], final_exit_price, exit_decision,
                df_labeled, len(df_labeled) - 1
            )
            equity_curve.append(risk_manager.current_capital)

        metrics = self._calculate_performance_metrics(trades, equity_curve, risk_manager.current_capital,
                                                      drawdown_periods)

        for trade in trades:
            self.time_manager.update_duration_stats(trade)

        self.logger.info(f"Simulation completed with {len(trades)} trades")

        return {
            "final_equity": risk_manager.current_capital,
            "trades": trades,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "daily_returns": daily_returns,
            "drawdown_periods": drawdown_periods
        }

    def _get_current_price(self, df: pd.DataFrame, row_idx: int, price_type: str = 'close') -> float:
        actual_col = f'actual_{price_type}'
        if actual_col in df.columns:
            try:
                return float(df[actual_col].iloc[row_idx])
            except (IndexError, ValueError, TypeError):
                pass

        if price_type in df.columns:
            try:
                return float(df[price_type].iloc[row_idx])
            except (IndexError, ValueError, TypeError):
                pass

        if 'actual_close' in df.columns:
            try:
                return float(df['actual_close'].iloc[row_idx])
            except (IndexError, ValueError, TypeError):
                pass

        if 'close' in df.columns:
            try:
                return float(df['close'].iloc[row_idx])
            except (IndexError, ValueError, TypeError):
                return 0.0

        return 0.0

    def _get_indicator_value(self, df: pd.DataFrame, row_idx: int,
                             indicator: str, fallback_indicator: str = None,
                             default: float = 0.0) -> float:
        if indicator in df.columns:
            try:
                value = float(df[indicator].iloc[row_idx])
                if not np.isnan(value):
                    self._track_indicator_stat(indicator, value)
                    return value
            except (IndexError, ValueError, TypeError):
                pass

        if fallback_indicator is not None and fallback_indicator in df.columns:
            try:
                value = float(df[fallback_indicator].iloc[row_idx])
                if not np.isnan(value):
                    self._track_indicator_stat(fallback_indicator, value)
                    return value
            except (IndexError, ValueError, TypeError):
                pass

        return default

    def _track_indicator_stat(self, indicator: str, value: float) -> None:
        if indicator not in self.indicator_stats:
            self.indicator_stats[indicator] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf')
            }

        stats = self.indicator_stats[indicator]
        stats['count'] += 1
        stats['sum'] += value
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)

    def _get_macd_values(self, df: pd.DataFrame, row_idx: int) -> Dict[str, float]:
        result = {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}

        macd_columns = [
            ('macd', 'macd'),
            ('m30_macd', 'macd'),
            ('macd_signal', 'macd_signal'),
            ('m30_macd_signal', 'macd_signal'),
            ('macd_histogram', 'macd_histogram'),
            ('m30_macd_histogram', 'macd_histogram')
        ]

        for col, key in macd_columns:
            if col in df.columns:
                try:
                    value = float(df[col].iloc[row_idx])
                    if not np.isnan(value):
                        result[key] = value
                        self._track_indicator_stat(col, value)
                except (IndexError, ValueError, TypeError):
                    pass

        return result

    def _compute_atr(self, df: pd.DataFrame, row_idx: int, current_price: float) -> float:
        for col in ['atr_14', 'm30_atr_14']:
            if col in df.columns:
                try:
                    atr_val = float(df[col].iloc[row_idx])
                    if not np.isnan(atr_val) and atr_val > 0:
                        self._track_indicator_stat(col, atr_val)
                        return atr_val
                except (IndexError, ValueError, TypeError):
                    pass

        return current_price * 0.01

    def _finalize_exit(self, iteration: int, risk_manager, trades: List[Dict[str, Any]],
                       trade_entry: Dict[str, Any], current_time, current_price: float,
                       exit_decision: Dict[str, Any], df_test=None, row_idx=None) -> Tuple[
        List[Dict[str, Any]], int, Optional[Dict[str, Any]]]:
        try:
            if trade_entry is None:
                self.logger.error("Trade entry is None in _finalize_exit")
                return trades, 0, None

            if not isinstance(current_price, (int, float)) or current_price <= 0:
                self.logger.error(f"Invalid current price in _finalize_exit: {current_price}")
                current_price = float(trade_entry.get('entry_price', 1000.0))

            direction = trade_entry.get('direction')
            if direction not in ['long', 'short']:
                self.logger.error(f"Invalid direction in trade entry: {direction}")
                direction = 'long'

            exit_price = float(exit_decision.get('exit_price', current_price))
            if np.isnan(exit_price) or exit_price <= 0:
                self.logger.warning(f"Invalid exit price: {exit_price}, using current price")
                exit_price = current_price

            # Apply dynamic slippage
            volatility = float(trade_entry.get('volatility_regime', 0.5))
            if self.use_dynamic_slippage:
                # Scale slippage based on exit reason - higher for stop-loss exits
                is_forced_exit = exit_decision.get('reason') in ['StopLoss', 'MaxDurationReached', 'StagnantPosition']
                slippage_factor = 1.2 if is_forced_exit else 0.8
                # Adjust for market volatility
                slippage_factor *= (0.8 + volatility * 0.4)

                # Apply the slippage
                slip_amount = exit_price * self.slippage * slippage_factor
            else:
                slip_amount = exit_price * self.slippage

            # Apply slippage in the correct direction
            if direction == 'long':
                exit_price = max(0.01, exit_price - slip_amount)
            else:
                exit_price = exit_price + slip_amount

            qty_open = float(trade_entry.get('quantity', 0))
            if np.isnan(qty_open) or qty_open <= 0:
                self.logger.warning(f"Invalid quantity in trade entry: {qty_open}")
                qty_open = 0.01

            exit_cost = self.fixed_cost + (exit_price * qty_open * self.variable_cost)

            entry_px = float(trade_entry.get('entry_price', 0))
            if np.isnan(entry_px) or entry_px <= 0:
                self.logger.warning(f"Invalid entry price: {entry_px}")
                entry_px = exit_price

            if direction == 'long':
                close_pnl = qty_open * (exit_price - entry_px) - exit_cost - float(
                    trade_entry.get('total_entry_cost', 0))
            else:
                close_pnl = qty_open * (entry_px - exit_price) - exit_cost - float(
                    trade_entry.get('total_entry_cost', 0))

            if np.isnan(close_pnl):
                self.logger.warning("NaN PnL calculated, setting to zero")
                close_pnl = 0.0

            t_rec = trade_entry.copy()
            t_rec.update({
                'exit_time': current_time,
                'exit_price': exit_price,
                'exit_signal': exit_decision.get('reason', 'ExitSignal'),
                'pnl': close_pnl,
                'iteration': iteration
            })

            if df_test is not None and row_idx is not None and 0 <= row_idx < len(df_test):
                t_rec['exit_ema_20'] = self._get_indicator_value(df_test, row_idx, 'ema_20', 'm30_ema_20')
                t_rec['exit_rsi_14'] = self._get_indicator_value(df_test, row_idx, 'rsi_14', 'm30_rsi_14', default=50)
                exit_macd = self._get_macd_values(df_test, row_idx)
                t_rec['exit_macd'] = exit_macd.get('macd', 0)
                t_rec['exit_macd_signal'] = exit_macd.get('macd_signal', 0)
                t_rec['exit_macd_histogram'] = exit_macd.get('macd_histogram', 0)
            else:
                t_rec['exit_ema_20'] = 0.0
                t_rec['exit_rsi_14'] = 50.0
                t_rec['exit_macd'] = 0.0
                t_rec['exit_macd_signal'] = 0.0
                t_rec['exit_macd_histogram'] = 0.0

            entry_time = t_rec.get('entry_time')
            exit_time = t_rec.get('exit_time')
            if entry_time and exit_time and isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
                try:
                    if entry_time.tzinfo is not None and exit_time.tzinfo is None:
                        exit_time = exit_time.replace(tzinfo=entry_time.tzinfo)
                    elif entry_time.tzinfo is None and exit_time.tzinfo is not None:
                        entry_time = entry_time.replace(tzinfo=exit_time.tzinfo)

                    duration_hours = (exit_time - entry_time).total_seconds() / 3600
                    t_rec['duration_hours'] = duration_hours

                    if len(trades) > 0:
                        self.avg_trade_holding_time = ((self.avg_trade_holding_time * len(trades)) + duration_hours) / (
                                len(trades) + 1)
                    else:
                        self.avg_trade_holding_time = duration_hours
                except Exception as e:
                    self.logger.warning(f"Error calculating duration: {e}")
                    t_rec['duration_hours'] = 0.0

            trades.append(t_rec)

            exit_reason = exit_decision.get('reason', 'ExitSignal')
            self._track_exit_performance(exit_reason, close_pnl)

            try:
                risk_manager.update_after_trade(t_rec)
            except Exception as e:
                self.logger.error(f"Error updating risk manager: {e}")

            position = 0
            trade_entry = None

            return trades, position, trade_entry

        except Exception as e:
            self.logger.error(f"Error in _finalize_exit: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            try:
                emergency_trade = {
                    'entry_time': trade_entry.get('entry_time', current_time - timedelta(hours=1)),
                    'exit_time': current_time,
                    'entry_price': trade_entry.get('entry_price', current_price),
                    'exit_price': current_price,
                    'direction': trade_entry.get('direction', 'long'),
                    'quantity': trade_entry.get('quantity', 0.01),
                    'exit_signal': 'EmergencyExit',
                    'pnl': 0.0,
                    'iteration': iteration
                }
                trades.append(emergency_trade)

                try:
                    risk_manager.update_after_trade(emergency_trade)
                except:
                    pass

            except Exception:
                pass

            return trades, 0, None

    def _execute_partial_exit(self, iteration: int, risk_manager, trades: List[Dict[str, Any]],
                              trade_entry: Dict[str, Any], current_time, current_price: float,
                              partial_exit: Dict[str, Any], df_test=None, row_idx=None) -> Tuple[
        List[Dict[str, Any]], int, Dict[str, Any]]:
        try:
            if trade_entry is None:
                self.logger.error("Trade entry is None in _execute_partial_exit")
                return trades, 0, None

            direction = trade_entry.get('direction')
            if direction not in ['long', 'short']:
                self.logger.error(f"Invalid direction in trade entry: {direction}")
                direction = 'long'

            if not isinstance(current_price, (int, float)) or current_price <= 0:
                self.logger.error(f"Invalid current price in _execute_partial_exit: {current_price}")
                current_price = float(trade_entry.get('entry_price', 1000.0))

            if not partial_exit or 'portion' not in partial_exit:
                self.logger.error(f"Invalid partial exit data: {partial_exit}")
                return trades, 1 if direction == 'long' else -1, trade_entry

            exit_price = current_price

            # Apply slippage - less for partial exits than full exits
            slip_factor = 0.8  # Reduced slippage for partial exits
            slip_amount = exit_price * self.slippage * slip_factor
            if direction == 'long':
                exit_price = max(0.01, exit_price - slip_amount)
            else:
                exit_price = exit_price + slip_amount

            portion = float(partial_exit.get('portion', 0.2))
            portion = max(0.01, min(0.99, portion))

            original_qty = float(trade_entry.get('quantity', 0))
            if original_qty <= 0:
                self.logger.warning(f"Invalid quantity in trade_entry: {original_qty}")
                return trades, 1 if direction == 'long' else -1, trade_entry

            exit_qty = original_qty * portion
            remaining_qty = original_qty - exit_qty

            exit_cost = self.fixed_cost + (exit_price * exit_qty * self.variable_cost)

            entry_px = float(trade_entry.get('entry_price', 0))
            if entry_px <= 0:
                self.logger.warning(f"Invalid entry price: {entry_px}")
                entry_px = exit_price

            proportional_entry_cost = float(trade_entry.get('total_entry_cost', 0)) * portion

            if direction == 'long':
                partial_pnl = exit_qty * (exit_price - entry_px) - exit_cost - proportional_entry_cost
            else:
                partial_pnl = exit_qty * (entry_px - exit_price) - exit_cost - proportional_entry_cost

            if np.isnan(partial_pnl):
                self.logger.warning("NaN partial PnL calculated, setting to zero")
                partial_pnl = 0.0

            partial_rec = trade_entry.copy()
            partial_rec.update({
                'exit_time': current_time,
                'exit_price': exit_price,
                'exit_signal': f"PartialExit_{int(portion * 100)}pct",
                'pnl': partial_pnl,
                'quantity': exit_qty,
                'iteration': iteration,
                'is_partial': True,
                'partial_id': partial_exit.get('id', '')
            })

            if df_test is not None and row_idx is not None and 0 <= row_idx < len(df_test):
                partial_rec['exit_ema_20'] = self._get_indicator_value(df_test, row_idx, 'ema_20', 'm30_ema_20')
                partial_rec['exit_rsi_14'] = self._get_indicator_value(df_test, row_idx, 'rsi_14', 'm30_rsi_14',
                                                                       default=50)
                exit_macd = self._get_macd_values(df_test, row_idx)
                partial_rec['exit_macd'] = exit_macd.get('macd', 0)
                partial_rec['exit_macd_signal'] = exit_macd.get('macd_signal', 0)
                partial_rec['exit_macd_histogram'] = exit_macd.get('macd_histogram', 0)
            else:
                partial_rec['exit_ema_20'] = 0.0
                partial_rec['exit_rsi_14'] = 50.0
                partial_rec['exit_macd'] = 0.0
                partial_rec['exit_macd_signal'] = 0.0
                partial_rec['exit_macd_histogram'] = 0.0

            entry_time = partial_rec.get('entry_time')
            if entry_time and isinstance(entry_time, datetime) and isinstance(current_time, datetime):
                try:
                    if entry_time.tzinfo is not None and current_time.tzinfo is None:
                        current_time = current_time.replace(tzinfo=entry_time.tzinfo)
                    elif entry_time.tzinfo is None and current_time.tzinfo is not None:
                        entry_time = entry_time.replace(tzinfo=current_time.tzinfo)

                    duration_hours = (current_time - entry_time).total_seconds() / 3600
                    partial_rec['duration_hours'] = duration_hours
                except Exception as e:
                    self.logger.warning(f"Error calculating duration: {e}")
                    partial_rec['duration_hours'] = 0.0

            trades.append(partial_rec)

            exit_reason = f"PartialExit_{int(portion * 100)}pct"
            self._track_exit_performance(exit_reason, partial_pnl)

            risk_manager.current_capital += partial_pnl

            trade_entry['quantity'] = remaining_qty
            trade_entry['partial_exit_taken'] = True

            trade_entry['total_entry_cost'] = float(trade_entry.get('total_entry_cost', 0)) * (1 - portion)

            position = 1 if direction == 'long' else -1

            self.logger.debug(f"Partial exit at {current_time}: {exit_qty} @ ${exit_price}, PnL: ${partial_pnl:.2f}")

            return trades, position, trade_entry

        except Exception as e:
            self.logger.error(f"Error in _execute_partial_exit: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            position_dir = 1 if trade_entry.get('direction', 'long') == 'long' else -1
            return trades, position_dir, trade_entry

    def _track_exit_performance(self, exit_reason: str, pnl: float):
        if not hasattr(self, 'exit_performance'):
            self.exit_performance = {}

        if exit_reason not in self.exit_performance:
            self.exit_performance[exit_reason] = {
                'count': 0,
                'total_pnl': 0,
                'win_count': 0,
                'avg_pnl': 0,
                'win_rate': 0
            }

        perf = self.exit_performance[exit_reason]
        perf['count'] += 1
        perf['total_pnl'] += pnl

        if pnl > 0:
            perf['win_count'] += 1

        perf['avg_pnl'] = perf['total_pnl'] / perf['count']
        perf['win_rate'] = perf['win_count'] / perf['count']

        if perf['count'] >= 5 and perf['avg_pnl'] > 0:
            if exit_reason not in self.best_exit_reasons:
                self.best_exit_reasons.append(exit_reason)
                self.best_exit_reasons = sorted(
                    self.best_exit_reasons,
                    key=lambda x: self.exit_performance[x]['avg_pnl'] if x in self.exit_performance else 0,
                    reverse=True
                )

    def _compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        if len(y_train.shape) == 1 or y_train.shape[1] == 1:
            return {0: 1.0}

        try:
            y_train_flat = np.argmax(y_train, axis=1)
            unique_cls = np.unique(y_train_flat)

            cw = compute_class_weight('balanced', classes=unique_cls, y=y_train_flat)
            cw_dict = {cls: w for cls, w in zip(unique_cls, cw)}

            for c_ in range(5):
                if c_ not in cw_dict:
                    cw_dict[c_] = 1.0

            return cw_dict
        except Exception as e:
            self.logger.warning(f"Class weight error: {e}")
            return {i: 1.0 for i in range(5)}

    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]],
                                       equity_curve: List[float],
                                       final_equity: float,
                                       drawdown_periods: List[Dict[str, Any]] = None) -> Dict[str, float]:
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 0,
                'max_drawdown_duration': 0, 'avg_trade': 0, 'return': 0
            }

        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) <= 0]
        total_tr = len(trades)

        w_rate = len(wins) / total_tr if total_tr else 0

        prof_sum = sum(t.get('pnl', 0) for t in wins)
        loss_sum = abs(sum(t.get('pnl', 0) for t in losses))
        pf = prof_sum / max(loss_sum, 1e-10)

        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev_val = equity_curve[i - 1]
            if prev_val > 0:
                daily_returns.append((equity_curve[i] - prev_val) / prev_val)
            else:
                daily_returns.append(0)

        # Calculate Sharpe Ratio
        if len(daily_returns) > 1:
            avg_ret = np.mean(daily_returns)
            std_ret = max(np.std(daily_returns), 1e-10)
            sharpe = (avg_ret / std_ret) * np.sqrt(252)
        else:
            sharpe = 0

        # Calculate Sortino Ratio (downside risk only)
        if len(daily_returns) > 1:
            down_returns = [r for r in daily_returns if r < 0]
            if down_returns:
                downside_dev = max(np.std(down_returns), 1e-10)
                sortino = (avg_ret / downside_dev) * np.sqrt(252)
            else:
                sortino = sharpe * 1.5  # If no negative returns, set higher than sharpe
        else:
            sortino = 0

        # Calculate Drawdown Statistics
        max_dd = 0
        max_dd_duration = 0
        peak = equity_curve[0]
        current_dd_start = None

        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                # Drawdown ended
                if current_dd_start is not None:
                    dd_duration = i - current_dd_start
                    max_dd_duration = max(max_dd_duration, dd_duration)
                    current_dd_start = None
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                # Start tracking new drawdown
                if dd > 0 and current_dd_start is None:
                    current_dd_start = i
                max_dd = max(max_dd, dd)

        # If we're still in a drawdown at the end, update duration
        if current_dd_start is not None:
            dd_duration = len(equity_curve) - current_dd_start
            max_dd_duration = max(max_dd_duration, dd_duration)

        # Use provided drawdown periods if available
        if drawdown_periods:
            for period in drawdown_periods:
                max_dd = max(max_dd, period['depth'])
                max_dd_duration = max(max_dd_duration, period['duration'])

        initial_capital = equity_curve[0]
        total_ret = ((final_equity - initial_capital) / initial_capital) * 100

        # Calculate indicator statistics
        avg_entry_rsi = np.mean([t.get('entry_rsi_14', 50) for t in trades])
        avg_exit_rsi = np.mean([t.get('exit_rsi_14', 50) for t in trades])

        if wins:
            avg_win_entry_macd_hist = np.mean([w.get('entry_macd_histogram', 0) for w in wins])
            avg_win_exit_macd_hist = np.mean([w.get('exit_macd_histogram', 0) for w in wins])
        else:
            avg_win_entry_macd_hist = 0
            avg_win_exit_macd_hist = 0

        # Calculate market phase performance
        phase_metrics = {}
        for t in trades:
            phase = t.get('market_phase', 'neutral')
            pnl = t.get('pnl', 0)

            if phase not in phase_metrics:
                phase_metrics[phase] = {
                    'count': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'avg_pnl': 0
                }

            stats = phase_metrics[phase]
            stats['count'] += 1
            stats['total_pnl'] += pnl

            if pnl > 0:
                stats['wins'] += 1

            if stats['count'] > 0:
                stats['win_rate'] = stats['wins'] / stats['count']
                stats['avg_pnl'] = stats['total_pnl'] / stats['count']

            if stats['count'] >= 5 and stats['avg_pnl'] > 0:
                if phase not in self.best_performing_phases:
                    self.best_performing_phases.append(phase)
                    self.best_performing_phases = sorted(
                        self.best_performing_phases,
                        key=lambda x: phase_metrics[x]['avg_pnl'] if x in phase_metrics else 0,
                        reverse=True
                    )

        # Calculate average trade duration
        avg_hours_in_trade = 0
        trade_count_with_duration = 0
        for t in trades:
            if 'entry_time' in t and 'exit_time' in t:
                try:
                    duration = (t['exit_time'] - t['entry_time']).total_seconds() / 3600
                    avg_hours_in_trade += duration
                    trade_count_with_duration += 1
                except:
                    pass

        if trade_count_with_duration > 0:
            avg_hours_in_trade /= trade_count_with_duration

        # Calculate trade frequency (trades per day)
        if trades and len(trades) >= 2:
            try:
                first_trade_time = min(t.get('entry_time') for t in trades)
                last_trade_time = max(t.get('exit_time') for t in trades)
                trading_days = (last_trade_time - first_trade_time).total_seconds() / (24 * 3600)
                trades_per_day = len(trades) / max(1, trading_days)
            except:
                trades_per_day = 0
        else:
            trades_per_day = 0

        return {
            'total_trades': total_tr,
            'win_rate': w_rate,
            'profit_factor': pf,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_drawdown_duration': max_dd_duration,
            'return': total_ret,
            'avg_entry_rsi': avg_entry_rsi,
            'avg_exit_rsi': avg_exit_rsi,
            'avg_win_entry_macd_hist': avg_win_entry_macd_hist,
            'avg_win_exit_macd_hist': avg_win_exit_macd_hist,
            'avg_hours_in_trade': avg_hours_in_trade,
            'trades_per_day': trades_per_day,
            'phase_metrics': phase_metrics,
            'avg_win': prof_sum / len(wins) if wins else 0,
            'avg_loss': -loss_sum / len(losses) if losses else 0
        }

    def _calculate_consolidated_metrics(self, final_equity: float) -> Dict[str, float]:
        if not self.consolidated_trades:
            return {}

        eq_curve = [self.risk_manager.initial_capital]
        balance = self.risk_manager.initial_capital

        # Used for drawdown calculations
        peak_capital = balance
        current_drawdown_start = None
        max_drawdown = 0
        max_drawdown_duration = 0
        last_date = None

        # Daily and monthly returns tracking
        daily_returns = []
        monthly_returns = {}

        for tr in sorted(self.consolidated_trades, key=lambda x: x['exit_time']):
            # Calculate balance after trade
            pnl = tr.get('pnl', 0)
            balance += pnl
            eq_curve.append(balance)

            # Track daily returns
            exit_date = tr['exit_time'].date()
            if last_date is not None and exit_date != last_date:
                # New day - calculate daily return
                if len(eq_curve) >= 2:
                    daily_return = (eq_curve[-2] / eq_curve[-3]) - 1 if eq_curve[-3] > 0 else 0
                    daily_returns.append(daily_return)

                    # Track monthly returns
                    month_key = exit_date.strftime('%Y-%m')
                    if month_key not in monthly_returns:
                        monthly_returns[month_key] = []
                    monthly_returns[month_key].append(daily_return)
            last_date = exit_date

            # Update drawdown tracking
            if balance > peak_capital:
                peak_capital = balance
                if current_drawdown_start is not None:
                    # Drawdown ended - calculate duration
                    drawdown_duration = len(eq_curve) - 1 - current_drawdown_start
                    max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
                    current_drawdown_start = None
            elif balance < peak_capital:
                # In drawdown
                current_drawdown = (peak_capital - balance) / peak_capital
                max_drawdown = max(max_drawdown, current_drawdown)

                if current_drawdown_start is None:
                    current_drawdown_start = len(eq_curve) - 1

        ret_pct = ((final_equity / self.risk_manager.initial_capital) - 1) * 100

        wins = [t for t in self.consolidated_trades if t.get('pnl', 0) > 0]
        total_tr = len(self.consolidated_trades)
        win_rate = len(wins) / total_tr if total_tr else 0

        losses = [t for t in self.consolidated_trades if t.get('pnl', 0) <= 0]
        p_sum = sum(t.get('pnl', 0) for t in wins)
        n_sum = abs(sum(t.get('pnl', 0) for t in losses))
        pf = p_sum / max(n_sum, 1e-10)

        # Daily returns statistics
        if daily_returns:
            avg_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns) if len(daily_returns) > 1 else 1e-10
            sharpe = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0

            # Sortino ratio (downside deviation only)
            down_returns = [r for r in daily_returns if r < 0]
            if down_returns:
                downside_dev = np.std(down_returns)
                sortino = (avg_daily_return / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0
            else:
                sortino = sharpe * 1.5  # If no down days, set higher than sharpe
        else:
            sharpe = 0
            sortino = 0

        # Calculate best and worst trades
        best_trade = max(self.consolidated_trades, key=lambda x: x.get('pnl', 0)) if self.consolidated_trades else {}
        worst_trade = min(self.consolidated_trades, key=lambda x: x.get('pnl', 0)) if self.consolidated_trades else {}

        # Calculate win/loss streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for tr in sorted(self.consolidated_trades, key=lambda x: x['exit_time']):
            pnl = tr.get('pnl', 0)
            if pnl > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))

        # Calculate monthly performance statistics
        if monthly_returns:
            best_month = max(monthly_returns.items(), key=lambda x: np.mean(x[1]))
            worst_month = min(monthly_returns.items(), key=lambda x: np.mean(x[1]))
            monthly_sharpes = {
                month: np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
                for month, returns in monthly_returns.items()}
            best_sharpe_month = max(monthly_sharpes.items(), key=lambda x: x[1])
        else:
            best_month = ("", [])
            worst_month = ("", [])
            best_sharpe_month = ("", 0)

        return {
            'return': ret_pct,
            'win_rate': win_rate,
            'profit_factor': pf,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_duration': max_drawdown_duration,
            'best_trade_pnl': best_trade.get('pnl', 0),
            'worst_trade_pnl': worst_trade.get('pnl', 0),
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'best_month': best_month[0] if best_month[1] else "",
            'best_month_return': np.mean(best_month[1]) * 100 if best_month[1] else 0,
            'worst_month': worst_month[0] if worst_month[1] else "",
            'worst_month_return': np.mean(worst_month[1]) * 100 if worst_month[1] else 0
        }

    def _export_trade_details(self, final_equity: float) -> None:
        if not self.consolidated_trades:
            self.logger.warning("No trades to export")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        trade_records = []
        balance = self.risk_manager.initial_capital

        for tr in sorted(self.consolidated_trades, key=lambda x: x['exit_time']):
            pnl = float(tr.get('pnl', 0))
            balance += pnl

            record = {
                'iteration': int(tr.get('iteration', 0)),
                'date': tr.get('exit_time').strftime('%Y-%m-%d %H:%M:%S'),
                'direction': tr.get('direction', 'unknown'),
                'entry_price': round(float(tr.get('entry_price', 0)), 2),
                'exit_price': round(float(tr.get('exit_price', 0)), 2),
                'stop_loss': round(float(tr.get('initial_stop_loss', 0)), 2),
                'position_size': round(float(tr.get('quantity', 0)), 6),
                'pnl': round(pnl, 2),
                'balance': round(balance, 2),
                'signal': tr.get('entry_signal', 'unknown'),
                'exit_reason': tr.get('exit_signal', 'unknown'),
                'is_partial': tr.get('is_partial', False),
                'entry_rsi_14': round(float(tr.get('entry_rsi_14', 50)), 2),
                'exit_rsi_14': round(float(tr.get('exit_rsi_14', 50)), 2),
                'entry_macd_histogram': round(float(tr.get('entry_macd_histogram', 0)), 6),
                'exit_macd_histogram': round(float(tr.get('exit_macd_histogram', 0)), 6),
                'ensemble_score': round(float(tr.get('ensemble_score', 0)), 2),
                'market_phase': tr.get('market_phase', 'neutral'),
                'trend_strength': round(float(tr.get('trend_strength', 0)), 2),
                'duration_hours': round((tr.get('exit_time') - tr.get('entry_time')).total_seconds() / 3600, 1)
            }
            trade_records.append(record)

        df_trades = pd.DataFrame(trade_records)
        csv_path = self.output_dir / f'trade_details_{timestamp}.csv'
        df_trades.to_csv(csv_path, index=False)

        summary_path = self.output_dir / f'trade_summary_{timestamp}.txt'

        with open(summary_path, 'w') as f:
            metrics = self._calculate_consolidated_metrics(final_equity)

            f.write("Trading Results Summary\n")
            f.write("======================\n\n")
            f.write(f"Total Trades: {len(df_trades)}\n")
            f.write(f"Initial Balance: ${self.risk_manager.initial_capital:.2f}\n")
            f.write(f"Final Balance: ${final_equity:.2f}\n")
            f.write(f"Total Profit/Loss: ${final_equity - self.risk_manager.initial_capital:.2f}\n")
            f.write(f"Return: {((final_equity / self.risk_manager.initial_capital) - 1) * 100:.2f}%\n\n")

            win_trades = df_trades[df_trades['pnl'] > 0]
            loss_trades = df_trades[df_trades['pnl'] <= 0]
            win_rate = len(win_trades) / len(df_trades) if len(df_trades) else 0
            f.write(f"Win Rate: {win_rate * 100:.2f}%\n")

            if not win_trades.empty:
                f.write(f"Average Win: ${win_trades['pnl'].mean():.2f}\n")
                f.write(f"Best Trade: ${win_trades['pnl'].max():.2f}\n")
            if not loss_trades.empty:
                f.write(f"Average Loss: ${loss_trades['pnl'].mean():.2f}\n")
                f.write(f"Worst Trade: ${loss_trades['pnl'].min():.2f}\n")

            total_profit = win_trades['pnl'].sum() if not win_trades.empty else 0
            total_loss = abs(loss_trades['pnl'].sum()) if not loss_trades.empty else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            f.write(f"Profit Factor: {profit_factor:.2f}\n")
            f.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n")
            f.write(f"Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%\n")
            f.write(f"Max Drawdown Duration: {metrics.get('max_drawdown_duration', 0)} trades\n\n")

            f.write(f"Average Trade Duration: {df_trades['duration_hours'].mean():.1f} hours\n")
            f.write(f"Max Win Streak: {metrics.get('max_win_streak', 0)}\n")
            f.write(f"Max Loss Streak: {metrics.get('max_loss_streak', 0)}\n\n")

            if metrics.get('best_month'):
                f.write(f"Best Month: {metrics.get('best_month', '')} ({metrics.get('best_month_return', 0):.2f}%)\n")
            if metrics.get('worst_month'):
                f.write(
                    f"Worst Month: {metrics.get('worst_month', '')} ({metrics.get('worst_month_return', 0):.2f}%)\n\n")

            f.write("Indicator Statistics\n")
            f.write("-------------------\n")
            f.write(f"Average Entry RSI: {df_trades['entry_rsi_14'].mean():.2f}\n")
            f.write(f"Average Exit RSI: {df_trades['exit_rsi_14'].mean():.2f}\n")

            if not win_trades.empty:
                f.write(f"Winning Trades Avg Entry MACD Histogram: {win_trades['entry_macd_histogram'].mean():.6f}\n")
                f.write(f"Winning Trades Avg Exit MACD Histogram: {win_trades['exit_macd_histogram'].mean():.6f}\n")

            f.write("\nExit Reason Statistics\n")
            f.write("----------------------\n")
            reason_stats = df_trades.groupby('exit_reason').agg({
                'pnl': ['count', 'mean', 'sum'],
                'duration_hours': ['mean']
            })

            for reason, stats in reason_stats.iterrows():
                count = stats[('pnl', 'count')]
                avg_pnl = stats[('pnl', 'mean')]
                total_pnl = stats[('pnl', 'sum')]
                avg_duration = stats[('duration_hours', 'mean')]
                f.write(f"{reason}: {count} trades, Avg P&L: ${avg_pnl:.2f}, " +
                        f"Total P&L: ${total_pnl:.2f}, Avg Duration: {avg_duration:.1f}h\n")

            f.write("\nMarket Phase Statistics\n")
            f.write("----------------------\n")
            phase_stats = df_trades.groupby('market_phase').agg({
                'pnl': ['count', 'mean', 'sum'],
                'duration_hours': ['mean']
            })

            for phase, stats in phase_stats.iterrows():
                count = stats[('pnl', 'count')]
                avg_pnl = stats[('pnl', 'mean')]
                total_pnl = stats[('pnl', 'sum')]
                avg_duration = stats[('duration_hours', 'mean')]
                f.write(f"{phase}: {count} trades, Avg P&L: ${avg_pnl:.2f}, " +
                        f"Total P&L: ${total_pnl:.2f}, Avg Duration: {avg_duration:.1f}h\n")

        self.logger.info(f"Exported {len(df_trades)} trades to {csv_path}")
        self.logger.info(f"Exported summary to {summary_path}")

    def _export_drawdown_analysis(self) -> None:
        """Export detailed drawdown analysis"""
        if not self.drawdown_periods and not self.consolidated_trades:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        drawdown_path = self.output_dir / f'drawdown_analysis_{timestamp}.txt'

        with open(drawdown_path, 'w') as f:
            f.write("Drawdown Analysis\n")
            f.write("=================\n\n")

            f.write("Overall Drawdown Statistics\n")
            f.write("---------------------------\n")
            f.write(f"Maximum Drawdown: {self.max_drawdown * 100:.2f}%\n")
            f.write(f"Maximum Drawdown Duration: {self.max_drawdown_duration} iterations\n\n")

            # Analyze recovery periods
            if self.consolidated_trades and len(self.consolidated_trades) > 10:
                f.write("Recovery Analysis\n")
                f.write("-----------------\n")

                # Find significant drawdowns and subsequent recovery
                equity_curve = [self.risk_manager.initial_capital]
                peak = self.risk_manager.initial_capital
                drawdown_start = None
                drawdowns = []

                # Calculate equity curve
                for t in sorted(self.consolidated_trades, key=lambda x: x['exit_time']):
                    equity = equity_curve[-1] + t.get('pnl', 0)
                    equity_curve.append(equity)

                    if equity > peak:
                        peak = equity
                        if drawdown_start is not None:
                            # Record completed drawdown
                            drawdown_end = len(equity_curve) - 2  # Previous point
                            recovery_length = len(equity_curve) - 2 - drawdown_start
                            max_drawdown = min(equity_curve[drawdown_start:drawdown_end + 1])
                            drawdown_pct = (peak - max_drawdown) / peak

                            if drawdown_pct > 0.05:  # Only record significant drawdowns
                                drawdowns.append({
                                    'start': drawdown_start,
                                    'end': drawdown_end,
                                    'depth': drawdown_pct,
                                    'recovery_length': recovery_length,
                                    'peak': peak
                                })
                            drawdown_start = None
                    elif equity < peak and drawdown_start is None:
                        drawdown_start = len(equity_curve) - 2

                # Report on drawdowns and recoveries
                if drawdowns:
                    f.write(f"Found {len(drawdowns)} significant drawdown periods (>5%)\n\n")
                    for i, dd in enumerate(drawdowns, 1):
                        f.write(f"Drawdown #{i}:\n")
                        f.write(f"  Depth: {dd['depth'] * 100:.2f}%\n")
                        f.write(f"  Recovery Length: {dd['recovery_length']} trades\n")
                        f.write(f"  Peak Before: ${dd['peak']:.2f}\n\n")

                    # Calculate average recovery statistics
                    avg_recovery = sum(dd['recovery_length'] for dd in drawdowns) / len(drawdowns)
                    f.write(f"Average Recovery Time: {avg_recovery:.1f} trades\n")

                    # Find correlation between drawdown depth and recovery time
                    depths = [dd['depth'] for dd in drawdowns]
                    recovery_times = [dd['recovery_length'] for dd in drawdowns]
                    corr = np.corrcoef(depths, recovery_times)[0, 1] if len(drawdowns) > 1 else 0
                    f.write(f"Correlation between drawdown depth and recovery time: {corr:.2f}\n\n")
                else:
                    f.write("No significant drawdown periods found.\n\n")

                # Analyze trades during drawdowns vs. normal periods
                if drawdowns:
                    # Identify trades during drawdown periods
                    drawdown_trade_indices = set()
                    for dd in drawdowns:
                        for i in range(dd['start'], dd['end'] + 1):
                            drawdown_trade_indices.add(i)

                    # Separate trades
                    trades_during_drawdown = []
                    trades_during_normal = []

                    for i, t in enumerate(sorted(self.consolidated_trades, key=lambda x: x['exit_time'])):
                        if i in drawdown_trade_indices:
                            trades_during_drawdown.append(t)
                        else:
                            trades_during_normal.append(t)

                    # Calculate statistics
                    dd_win_rate = len([t for t in trades_during_drawdown if t.get('pnl', 0) > 0]) / len(
                        trades_during_drawdown) if trades_during_drawdown else 0
                    normal_win_rate = len([t for t in trades_during_normal if t.get('pnl', 0) > 0]) / len(
                        trades_during_normal) if trades_during_normal else 0

                    dd_avg_pnl = sum(t.get('pnl', 0) for t in trades_during_drawdown) / len(
                        trades_during_drawdown) if trades_during_drawdown else 0
                    normal_avg_pnl = sum(t.get('pnl', 0) for t in trades_during_normal) / len(
                        trades_during_normal) if trades_during_normal else 0

                    f.write("Trade Performance During Drawdowns vs Normal Periods\n")
                    f.write("-------------------------------------------------\n")
                    f.write(f"Trades during drawdowns: {len(trades_during_drawdown)}\n")
                    f.write(f"Trades during normal periods: {len(trades_during_normal)}\n\n")
                    f.write(f"Drawdown period win rate: {dd_win_rate * 100:.2f}%\n")
                    f.write(f"Normal period win rate: {normal_win_rate * 100:.2f}%\n\n")
                    f.write(f"Avg PnL during drawdowns: ${dd_avg_pnl:.2f}\n")
                    f.write(f"Avg PnL during normal periods: ${normal_avg_pnl:.2f}\n\n")

                    # Analyze what leads to recovery
                    if trades_during_drawdown:
                        winning_exit_types = {}
                        for t in trades_during_drawdown:
                            if t.get('pnl', 0) > 0:
                                exit_type = t.get('exit_signal', 'Unknown')
                                winning_exit_types[exit_type] = winning_exit_types.get(exit_type, 0) + 1

                        f.write("Winning Exit Types During Drawdowns:\n")
                        for exit_type, count in sorted(winning_exit_types.items(), key=lambda x: x[1], reverse=True):
                            f.write(f"- {exit_type}: {count} trades\n")

                        # Analyze market phases during drawdowns
                        winning_phases = {}
                        for t in trades_during_drawdown:
                            if t.get('pnl', 0) > 0:
                                phase = t.get('market_phase', 'neutral')
                                winning_phases[phase] = winning_phases.get(phase, 0) + 1

                        f.write("\nWinning Market Phases During Drawdowns:\n")
                        for phase, count in sorted(winning_phases.items(), key=lambda x: x[1], reverse=True):
                            f.write(f"- {phase}: {count} trades\n")

            f.write("\nDrawdown Recovery Recommendations:\n")
            f.write("-------------------------------\n")
            f.write("1. During drawdowns, consider these adjustments:\n")
            f.write("   - Reduce position size by 30-50%\n")
            f.write("   - Focus on shorter-duration trades\n")
            f.write("   - Take partial profits earlier\n")
            f.write("   - Avoid trading against the dominant trend\n\n")

            f.write("2. For faster recovery:\n")
            f.write("   - Look for high-probability setups with 2:1 or better risk-reward\n")
            f.write("   - Favor market phases that historically perform best during drawdowns\n")
            f.write("   - Consider using exit types that have shown the best performance during drawdowns\n")

        self.logger.info(f"Exported drawdown analysis to {drawdown_path}")

    def _export_monthly_performance(self) -> None:
        """Export monthly performance analysis"""
        if not self.monthly_returns:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        monthly_path = self.output_dir / f'monthly_performance_{timestamp}.txt'

        with open(monthly_path, 'w') as f:
            f.write("Monthly Performance Analysis\n")
            f.write("===========================\n\n")

            # Calculate monthly statistics
            monthly_stats = {}
            for month, returns in self.monthly_returns.items():
                if not returns:
                    continue

                monthly_return = sum(returns)  # Compounded return for the month
                avg_daily_return = np.mean(returns)
                volatility = np.std(returns) if len(returns) > 1 else 0
                sharpe = avg_daily_return / volatility if volatility > 0 else 0
                win_days = sum(1 for r in returns if r > 0)
                win_rate = win_days / len(returns) if returns else 0

                monthly_stats[month] = {
                    'return': monthly_return,
                    'avg_daily': avg_daily_return,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'win_rate': win_rate,
                    'days': len(returns)
                }

            # Sort months chronologically
            sorted_months = sorted(monthly_stats.keys())

            # Print monthly performance table
            f.write("Monthly Returns Summary\n")
            f.write("======================\n\n")
            f.write(f"{'Month':<10} {'Return':<10} {'Avg Daily':<10} {'Win Rate':<10} {'Sharpe':<10} {'Days':<10}\n")
            f.write("-" * 60 + "\n")

            for month in sorted_months:
                stats = monthly_stats[month]
                f.write(
                    f"{month:<10} {stats['return'] * 100:8.2f}% {stats['avg_daily'] * 100:8.2f}% {stats['win_rate'] * 100:8.2f}% {stats['sharpe']:8.2f} {stats['days']:8d}\n")

            # Calculate best and worst months
            if monthly_stats:
                best_month = max(monthly_stats.items(), key=lambda x: x[1]['return'])
                worst_month = min(monthly_stats.items(), key=lambda x: x[1]['return'])
                most_volatile = max(monthly_stats.items(), key=lambda x: x[1]['volatility'])
                best_sharpe = max(monthly_stats.items(), key=lambda x: x[1]['sharpe'])

                f.write("\nPerformance Highlights\n")
                f.write("=====================\n")
                f.write(f"Best Month: {best_month[0]} ({best_month[1]['return'] * 100:.2f}%)\n")
                f.write(f"Worst Month: {worst_month[0]} ({worst_month[1]['return'] * 100:.2f}%)\n")
                f.write(
                    f"Most Volatile Month: {most_volatile[0]} (Volatility: {most_volatile[1]['volatility'] * 100:.2f}%)\n")
                f.write(f"Best Risk-Adjusted Month: {best_sharpe[0]} (Sharpe: {best_sharpe[1]['sharpe']:.2f})\n\n")

                # Calculate consistency metrics
                winning_months = sum(1 for _, stats in monthly_stats.items() if stats['return'] > 0)
                total_months = len(monthly_stats)
                monthly_win_rate = winning_months / total_months if total_months > 0 else 0

                f.write(f"Monthly Win Rate: {monthly_win_rate * 100:.2f}% ({winning_months}/{total_months} months)\n")

                # Calculate average return and standard deviation
                avg_monthly_return = np.mean([stats['return'] for _, stats in monthly_stats.items()])
                std_monthly_return = np.std([stats['return'] for _, stats in monthly_stats.items()]) if len(
                    monthly_stats) > 1 else 0

                f.write(f"Average Monthly Return: {avg_monthly_return * 100:.2f}%\n")
                f.write(f"Monthly Return Standard Deviation: {std_monthly_return * 100:.2f}%\n")
                f.write(
                    f"Return/Risk Ratio: {(avg_monthly_return / std_monthly_return) if std_monthly_return > 0 else 0:.2f}\n\n")

                # Analyze month-to-month consistency
                if len(sorted_months) >= 2:
                    consecutive_wins = 0
                    max_consecutive_wins = 0
                    consecutive_losses = 0
                    max_consecutive_losses = 0

                    for i in range(len(sorted_months)):
                        month = sorted_months[i]
                        if monthly_stats[month]['return'] > 0:
                            consecutive_wins += 1
                            consecutive_losses = 0
                            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                        else:
                            consecutive_losses += 1
                            consecutive_wins = 0
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

                    f.write(f"Maximum Consecutive Winning Months: {max_consecutive_wins}\n")
                    f.write(f"Maximum Consecutive Losing Months: {max_consecutive_losses}\n\n")

            # Provide monthly performance recommendations
            f.write("Monthly Performance Recommendations\n")
            f.write("=================================\n")

            # Analyze monthly patterns if enough data
            if len(monthly_stats) >= 3:
                # Look for seasonal patterns
                month_numbers = [int(m.split('-')[1]) for m in monthly_stats.keys()]
                month_returns = {month_num: [] for month_num in range(1, 13)}

                for month in monthly_stats:
                    year, month_num = month.split('-')
                    month_returns[int(month_num)].append(monthly_stats[month]['return'])

                # Find best and worst months by average return
                avg_month_returns = {m: np.mean(returns) if returns else 0 for m, returns in month_returns.items()}
                best_month_num = max(avg_month_returns.items(), key=lambda x: x[1])
                worst_month_num = min(avg_month_returns.items(), key=lambda x: x[1])

                month_names = ["January", "February", "March", "April", "May", "June",
                               "July", "August", "September", "October", "November", "December"]

                if best_month_num[1] > 0 and len(month_returns[best_month_num[0]]) > 1:
                    f.write(
                        f"1. Historically strongest month: {month_names[best_month_num[0] - 1]} (avg: {best_month_num[1] * 100:.2f}%)\n")
                    f.write("   Consider increasing position sizes during this month.\n\n")

                if worst_month_num[1] < 0 and len(month_returns[worst_month_num[0]]) > 1:
                    f.write(
                        f"2. Historically weakest month: {month_names[worst_month_num[0] - 1]} (avg: {worst_month_num[1] * 100:.2f}%)\n")
                    f.write(
                        "   Consider reducing exposure or implementing tighter risk management during this month.\n\n")

                # Analyze consistency and recommend improvements
                if monthly_win_rate < 0.6:
                    f.write("3. Improve monthly consistency:\n")
                    f.write("   - Focus on capital preservation during challenging months\n")
                    f.write("   - Implement monthly drawdown limits (e.g., 5% monthly max drawdown)\n")
                    f.write("   - Consider monthly rebalancing of risk parameters\n\n")

                # Check for end-of-month effects
                if self.consolidated_trades and len(self.consolidated_trades) >= 20:
                    # Group trades by month and analyze EOM vs rest of month
                    eom_trades = []
                    other_trades = []

                    for trade in self.consolidated_trades:
                        exit_date = trade['exit_time'].date()
                        next_month = (exit_date.month % 12) + 1
                        next_year = exit_date.year + (1 if next_month == 1 else 0)
                        days_to_eom = (datetime(next_year, next_month, 1).date() - exit_date).days

                        if days_to_eom <= 3:  # Last 3 days of month
                            eom_trades.append(trade)
                        else:
                            other_trades.append(trade)

                    if eom_trades:
                        eom_win_rate = sum(1 for t in eom_trades if t.get('pnl', 0) > 0) / len(eom_trades)
                        other_win_rate = sum(1 for t in other_trades if t.get('pnl', 0) > 0) / len(
                            other_trades) if other_trades else 0

                        if abs(eom_win_rate - other_win_rate) > 0.1:  # Significant difference
                            f.write("4. End of Month Effect Detected:\n")
                            if eom_win_rate > other_win_rate:
                                f.write("   - End of month trading performs better (Win Rate: "
                                        f"{eom_win_rate * 100:.1f}% vs {other_win_rate * 100:.1f}%)\n")
                                f.write("   - Consider increasing exposure during the last 3 days of each month\n\n")
                            else:
                                f.write("   - End of month trading performs worse (Win Rate: "
                                        f"{eom_win_rate * 100:.1f}% vs {other_win_rate * 100:.1f}%)\n")
                                f.write("   - Consider reducing exposure during the last 3 days of each month\n\n")

            else:
                f.write("Insufficient data for detailed monthly analysis.\n")
                f.write("Continue collecting performance data for at least 3 months\n")
                f.write("to enable monthly pattern detection and optimization.\n")

        self.logger.info(f"Exported monthly performance analysis to {monthly_path}")

    def _export_feature_impact_analysis(self) -> None:
        """Calculate feature impact on trading results"""
        if not self.consolidated_trades:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_path = self.output_dir / f'feature_impact_analysis_{timestamp}.txt'

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(analysis_path), exist_ok=True)

            with open(analysis_path, 'w') as f:
                f.write("Feature Impact Analysis\n")
                f.write("======================\n\n")

                f.write("Feature Values at Entry\n")
                f.write("---------------------\n")

                entry_stats = {}
                for trade in self.consolidated_trades:
                    direction = trade.get('direction', '')

                    for key, value in trade.items():
                        if key.startswith('entry_') and key != 'entry_time' and key != 'entry_price':
                            feature_name = key[6:]

                            if feature_name not in entry_stats:
                                entry_stats[feature_name] = {
                                    'long': [],
                                    'short': []
                                }

                            try:
                                if direction == 'long':
                                    float_value = float(value)
                                    entry_stats[feature_name]['long'].append(float_value)
                                elif direction == 'short':
                                    float_value = float(value)
                                    entry_stats[feature_name]['short'].append(float_value)
                            except (ValueError, TypeError):
                                continue

                for feature, stats in sorted(entry_stats.items()):
                    f.write(f"{feature}:\n")

                    if stats['long']:
                        avg_long = sum(stats['long']) / len(stats['long'])
                        f.write(f"  Long entries avg: {avg_long:.4f}\n")

                    if stats['short']:
                        avg_short = sum(stats['short']) / len(stats['short'])
                        f.write(f"  Short entries avg: {avg_short:.4f}\n")

                    f.write("\n")

                f.write("\nFeature Values at Exit by Exit Reason\n")
                f.write("----------------------------------\n")

                exit_reason_stats = {}
                for trade in self.consolidated_trades:
                    exit_reason = trade.get('exit_signal', 'Unknown')

                    if exit_reason not in exit_reason_stats:
                        exit_reason_stats[exit_reason] = {}

                    for key, value in trade.items():
                        if key.startswith(
                                'exit_') and key != 'exit_time' and key != 'exit_price' and key != 'exit_signal':
                            feature_name = key[5:]

                            if feature_name not in exit_reason_stats[exit_reason]:
                                exit_reason_stats[exit_reason][feature_name] = []

                            try:
                                float_value = float(value)
                                exit_reason_stats[exit_reason][feature_name].append(float_value)
                            except (ValueError, TypeError):
                                continue

                for reason, features in sorted(exit_reason_stats.items()):
                    if len(features) == 0:
                        continue

                    f.write(f"{reason}:\n")

                    for feature, values in sorted(features.items()):
                        if values:
                            avg_value = sum(values) / len(values)
                            f.write(f"  {feature}: {avg_value:.4f}\n")

                    f.write("\n")

                f.write("\nFeature Impact Recommendations\n")
                f.write("----------------------------\n")

                entry_importance = {}
                exit_importance = {}

                winning_trades = [t for t in self.consolidated_trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in self.consolidated_trades if t.get('pnl', 0) <= 0]

                if len(winning_trades) < 5 or len(losing_trades) < 5:
                    f.write("Not enough trades for reliable feature impact analysis\n")
                    return

                for feature in entry_stats:
                    if len(entry_stats[feature]['long']) > 0:
                        winning_values = []
                        losing_values = []

                        for t in winning_trades:
                            if t.get('direction') == 'long' and f'entry_{feature}' in t:
                                try:
                                    val = float(t.get(f'entry_{feature}', 0))
                                    winning_values.append(val)
                                except (ValueError, TypeError):
                                    pass

                        for t in losing_trades:
                            if t.get('direction') == 'long' and f'entry_{feature}' in t:
                                try:
                                    val = float(t.get(f'entry_{feature}', 0))
                                    losing_values.append(val)
                                except (ValueError, TypeError):
                                    pass

                        if winning_values and losing_values:
                            avg_winning = sum(winning_values) / len(winning_values)
                            avg_losing = sum(losing_values) / len(losing_values)
                            entry_importance[feature] = abs(avg_winning - avg_losing) * max(avg_winning, avg_losing)

                top_entry_features = sorted(entry_importance.items(), key=lambda x: x[1], reverse=True)[:5]

                f.write("Key features for entry decisions:\n")
                for i, (feature, impact) in enumerate(top_entry_features, 1):
                    f.write(f"{i}. {feature} (impact: {impact:.4f})\n")

                f.write("\n")

                for reason, features in exit_reason_stats.items():
                    for feature, values in features.items():
                        trades_with_reason = [t for t in self.consolidated_trades if t.get('exit_signal') == reason]
                        if len(trades_with_reason) < 5:
                            continue

                        feature_values = []
                        pnl_values = []

                        for t in trades_with_reason:
                            if f'exit_{feature}' in t:
                                try:
                                    feature_val = float(t.get(f'exit_{feature}', 0))
                                    feature_values.append(feature_val)
                                    pnl_values.append(t.get('pnl', 0))
                                except (ValueError, TypeError):
                                    pass

                        if len(feature_values) != len(pnl_values) or not feature_values:
                            continue

                        avg_value = sum(feature_values) / len(feature_values)
                        avg_pnl = sum(pnl_values) / len(pnl_values)

                        if avg_pnl > 0:
                            importance = abs(avg_value) * avg_pnl
                        else:
                            importance = abs(avg_value) * 0.1

                        if feature not in exit_importance:
                            exit_importance[feature] = 0

                        exit_importance[feature] += importance

                top_exit_features = sorted(exit_importance.items(), key=lambda x: x[1], reverse=True)[:5]

                f.write("Key features for exit decisions:\n")
                for i, (feature, impact) in enumerate(top_exit_features, 1):
                    f.write(f"{i}. {feature} (impact: {impact:.4f})\n")

        except Exception as e:
            self.logger.error(f"Error in feature impact analysis: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _export_exit_strategy_analysis(self) -> None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_path = self.output_dir / f'exit_strategy_analysis_{timestamp}.txt'

        try:
            with open(analysis_path, 'w') as f:
                f.write("Enhanced Exit Strategy Analysis\n")
                f.write("==============================\n\n")

                f.write("Exit Strategy Performance\n")
                f.write("-----------------------\n")
                if hasattr(self, 'exit_performance'):
                    f.write(f"Total trades analyzed: {len(self.consolidated_trades)}\n")
                    f.write(f"Total partial exits executed: {self.total_partial_exits}\n")
                    f.write(f"Total quick profit exits executed: {self.total_quick_profit_exits}\n\n")

                    f.write("Exit Strategy Performance by Type:\n")
                    for reason, perf in sorted(self.exit_performance.items(),
                                               key=lambda x: x[1]['avg_pnl'] if x[1]['count'] > 5 else -9999,
                                               reverse=True):
                        if perf['count'] >= 5:
                            f.write(f"  {reason}:\n")
                            f.write(f"    Count: {perf['count']}\n")
                            f.write(f"    Win Rate: {perf['win_rate'] * 100:.1f}%\n")
                            f.write(f"    Avg P&L: ${perf['avg_pnl']:.2f}\n")
                            f.write(f"    Total P&L: ${perf['total_pnl']:.2f}\n\n")
                else:
                    f.write("No exit performance data available.\n\n")

                f.write("Top Performing Exit Strategies\n")
                f.write("----------------------------\n")
                if hasattr(self, 'best_exit_reasons') and self.best_exit_reasons:
                    for i, reason in enumerate(self.best_exit_reasons[:5], 1):
                        if reason in self.exit_performance:
                            perf = self.exit_performance[reason]
                            f.write(
                                f"{i}. {reason}: ${perf['avg_pnl']:.2f} avg P&L, {perf['win_rate'] * 100:.1f}% win rate\n")
                else:
                    f.write("No best exit strategies identified.\n")
                f.write("\n")

                f.write("Market Phase Performance\n")
                f.write("-----------------------\n")
                for phase, count in sorted(self.market_phase_stats.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{phase}: {count} occurrences\n")
                f.write("\n")

                f.write("Top Performing Market Phases\n")
                f.write("--------------------------\n")
                if hasattr(self, 'best_performing_phases') and self.best_performing_phases:
                    for i, phase in enumerate(self.best_performing_phases[:3], 1):
                        f.write(f"{i}. {phase}\n")
                else:
                    f.write("No best performing phases identified.\n")
                f.write("\n")

                f.write("Trading Timing Analysis\n")
                f.write("----------------------\n")
                f.write(f"Average trade holding time: {self.avg_trade_holding_time:.2f} hours\n")

                f.write("\nExit Strategy Recommendations\n")
                f.write("---------------------------\n")

                if hasattr(self, 'exit_performance') and self.exit_performance:
                    best_strategies = sorted(
                        [(k, v) for k, v in self.exit_performance.items() if v['count'] >= 5],
                        key=lambda x: x[1]['avg_pnl'],
                        reverse=True
                    )[:3]

                    worst_strategies = sorted(
                        [(k, v) for k, v in self.exit_performance.items() if v['count'] >= 5],
                        key=lambda x: x[1]['avg_pnl']
                    )[:3]

                    if best_strategies:
                        f.write("1. Prioritize these exit strategies:\n")
                        for i, (strategy, perf) in enumerate(best_strategies, 1):
                            f.write(f"   {i}. {strategy}: ${perf['avg_pnl']:.2f} avg P&L\n")

                    if worst_strategies:
                        f.write("\n2. Avoid or modify these exit strategies:\n")
                        for i, (strategy, perf) in enumerate(worst_strategies, 1):
                            f.write(f"   {i}. {strategy}: ${perf['avg_pnl']:.2f} avg P&L\n")

                    f.write("\n3. Time-based recommendations:\n")
                    if self.avg_trade_holding_time < 5:
                        f.write("   - Consider longer holding periods for winning trades\n")
                    elif self.avg_trade_holding_time > 24:
                        f.write("   - Consider taking profits earlier on winning trades\n")

                    if hasattr(self, 'best_performing_phases') and self.best_performing_phases:
                        f.write("\n4. Market phase strategy recommendations:\n")
                        for phase in self.best_performing_phases[:2]:
                            f.write(f"   - Increase position size during {phase} phase\n")

                    partial_exits = [k for k in self.exit_performance.keys() if "PartialExit" in k]
                    if partial_exits:
                        best_partial = max(partial_exits, key=lambda x: self.exit_performance[x][
                            'avg_pnl'] if x in self.exit_performance else 0)
                        f.write(f"\n5. Partial exit optimization:\n")
                        f.write(f"   - Prioritize {best_partial} partial exit strategy\n")

                    # Add enhanced recommendations for trade/exit optimization
                    f.write("\n6. Advanced exit optimization strategy:\n")
                    f.write("   - Implement dynamic partial exits based on market volatility\n")
                    f.write("   - Use trailing stops that adapt to price momentum\n")
                    f.write("   - Consider market regime when setting profit targets\n")

                else:
                    f.write("Insufficient data for exit strategy recommendations.\n")

                # Add section on multi-timeframe exit confirmation
                f.write("\nMulti-timeframe Exit Confirmation Strategy\n")
                f.write("---------------------------------------\n")
                f.write("1. Primary timeframe exit signals should be confirmed with:\n")
                f.write("   - Momentum indicators on higher timeframe (e.g., 4h MACD for 30m trades)\n")
                f.write("   - Support/resistance levels on lower timeframe (e.g., 5m price action)\n")
                f.write("2. Exit strategy matrix:\n")
                f.write("   - Strong trend: Use trailing stops only\n")
                f.write("   - Ranging market: Use fixed take-profit levels\n")
                f.write("   - Mixed signals: Use partial exits at key levels\n")

        except Exception as e:
            self.logger.error(f"Error in exit strategy analysis: {e}")

    def _export_time_analysis(self, time_stats: Dict[str, Any]) -> None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        time_path = self.output_dir / f'time_analysis_{timestamp}.txt'

        with open(time_path, 'w') as f:
            f.write("Time-Based Trading Analysis\n")
            f.write("==========================\n\n")

            f.write("Exit Type Performance\n")
            f.write("--------------------\n")
            for exit_type, stats in time_stats.get("exit_stats", {}).items():
                f.write(f"{exit_type}:\n")
                f.write(f"  Count: {stats.get('count', 0)}\n")
                f.write(f"  Win Rate: {stats.get('win_rate', 0) * 100:.1f}%\n")
                f.write(f"  Avg PnL: ${stats.get('avg_pnl', 0):.2f}\n")
                f.write(f"  Avg Duration: {stats.get('avg_duration', 0):.1f}h\n")
                f.write(f"  Total PnL: ${stats.get('total_pnl', 0):.2f}\n\n")

            f.write("Optimal Trade Duration\n")
            f.write("---------------------\n")
            optimal = time_stats.get("optimal_durations", {})
            f.write(f"Optimal Hold Time: {optimal.get('optimal_hold_time', 24):.1f}h\n")
            f.write(f"Confidence: {optimal.get('confidence', 'low')}\n")
            f.write(f"Data Points: {optimal.get('data_points', 0)}\n")
            f.write(f"Avg Trade Duration: {optimal.get('avg_trade_duration', 0):.1f}h\n")
            f.write(f"Avg Profitable Duration: {optimal.get('avg_profitable_duration', 0):.1f}h\n")

            if "percentiles" in optimal:
                percentiles = optimal["percentiles"]
                f.write(f"Profitable Duration Percentiles: 25%={percentiles.get('p25', 0):.1f}h, " +
                        f"50%={percentiles.get('p50', 0):.1f}h, 75%={percentiles.get('p75', 0):.1f}h\n")

            if "phase_optimal_durations" in optimal:
                f.write("\nOptimal Durations by Market Phase\n")
                f.write("-------------------------------\n")
                for phase, duration in optimal["phase_optimal_durations"].items():
                    f.write(f"{phase}: {duration:.1f}h\n")

            # Add intraday analysis if available
            if self.consolidated_trades and len(self.consolidated_trades) >= 20:
                f.write("\nIntraday Performance Analysis\n")
                f.write("---------------------------\n")

                # Group trades by hour of day
                hour_performance = {}

                for trade in self.consolidated_trades:
                    try:
                        hour = trade['entry_time'].hour
                        if hour not in hour_performance:
                            hour_performance[hour] = {
                                'count': 0,
                                'wins': 0,
                                'total_pnl': 0
                            }

                        perf = hour_performance[hour]
                        perf['count'] += 1
                        pnl = trade.get('pnl', 0)
                        perf['total_pnl'] += pnl

                        if pnl > 0:
                            perf['wins'] += 1
                    except (AttributeError, KeyError):
                        continue

                # Calculate performance metrics by hour
                for hour, perf in hour_performance.items():
                    if perf['count'] > 0:
                        perf['win_rate'] = perf['wins'] / perf['count']
                        perf['avg_pnl'] = perf['total_pnl'] / perf['count']

                # Display hourly performance
                f.write("Performance by Hour of Day:\n")
                f.write(f"{'Hour':<6} {'Count':<6} {'Win Rate':<10} {'Avg PnL':<10} {'Total PnL':<10}\n")
                f.write("-" * 50 + "\n")

                for hour in sorted(hour_performance.keys()):
                    perf = hour_performance[hour]
                    if perf['count'] >= 3:  # Only show hours with enough data
                        f.write(f"{hour:02d}:00  {perf['count']:<6} {perf['win_rate'] * 100:8.1f}%  "
                                f"${perf['avg_pnl']:8.2f}  ${perf['total_pnl']:8.2f}\n")

                # Identify best and worst hours
                if hour_performance:
                    best_hour = max(hour_performance.items(), key=lambda x: x[1]['avg_pnl']
                    if x[1]['count'] >= 5 else -1000)
                    worst_hour = min(hour_performance.items(), key=lambda x: x[1]['avg_pnl']
                    if x[1]['count'] >= 5 else 1000)

                    if best_hour[1]['count'] >= 5:
                        f.write(f"\nBest Hour: {best_hour[0]:02d}:00 (Avg PnL: ${best_hour[1]['avg_pnl']:.2f}, "
                                f"Win Rate: {best_hour[1]['win_rate'] * 100:.1f}%)\n")

                    if worst_hour[1]['count'] >= 5:
                        f.write(f"Worst Hour: {worst_hour[0]:02d}:00 (Avg PnL: ${worst_hour[1]['avg_pnl']:.2f}, "
                                f"Win Rate: {worst_hour[1]['win_rate'] * 100:.1f}%)\n")

                # Analyze day of week performance
                day_performance = {}
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

                for trade in self.consolidated_trades:
                    try:
                        day = trade['entry_time'].weekday()
                        if day not in day_performance:
                            day_performance[day] = {
                                'count': 0,
                                'wins': 0,
                                'total_pnl': 0
                            }

                        perf = day_performance[day]
                        perf['count'] += 1
                        pnl = trade.get('pnl', 0)
                        perf['total_pnl'] += pnl

                        if pnl > 0:
                            perf['wins'] += 1
                    except (AttributeError, KeyError):
                        continue

                # Calculate performance metrics by day
                for day, perf in day_performance.items():
                    if perf['count'] > 0:
                        perf['win_rate'] = perf['wins'] / perf['count']
                        perf['avg_pnl'] = perf['total_pnl'] / perf['count']

                # Display day of week performance
                f.write("\nPerformance by Day of Week:\n")
                f.write(f"{'Day':<10} {'Count':<6} {'Win Rate':<10} {'Avg PnL':<10} {'Total PnL':<10}\n")
                f.write("-" * 55 + "\n")

                for day in range(7):
                    if day in day_performance and day_performance[day]['count'] >= 3:
                        perf = day_performance[day]
                        f.write(f"{day_names[day]:<10} {perf['count']:<6} {perf['win_rate'] * 100:8.1f}%  "
                                f"${perf['avg_pnl']:8.2f}  ${perf['total_pnl']:8.2f}\n")

        self.logger.info(f"Exported time analysis to {time_path}")

    def _ensure_required_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df.columns = [col.lower() for col in df.columns]

        for feature in self.data_preparer.essential_features:
            if feature not in df.columns:
                prefixed = f'm30_{feature}'
                if prefixed in df.columns:
                    df[feature] = df[prefixed]
                else:
                    if feature in ['rsi_14']:
                        df[feature] = 50
                    elif feature in ['stoch_k', 'stoch_d']:
                        df[feature] = 50
                    elif feature in ['willr_14']:
                        df[feature] = -50
                    elif feature in ['macd', 'macd_signal', 'macd_histogram']:
                        df[feature] = 0
                    elif feature in ['market_regime']:
                        df[feature] = 0
                    elif feature in ['volatility_regime']:
                        df[feature] = 0.5
                    elif feature in ['taker_buy_ratio']:
                        df[feature] = 0.5

        return df