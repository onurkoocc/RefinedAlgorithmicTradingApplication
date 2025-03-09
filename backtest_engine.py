import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.utils import compute_class_weight
from pathlib import Path
from uuid import uuid4

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

        results_dir = config.results_dir
        self.output_dir = Path(results_dir) / "backtest"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.consolidated_trades = []
        self.feature_engineer = FeatureEngineer(config)

    def run_walk_forward(self, df_features: pd.DataFrame) -> pd.DataFrame:
        df_len = len(df_features)
        min_required = self.train_window_size + self.test_window_size

        if df_len < min_required:
            self.logger.warning(
                f"Not enough data for train+test. Need at least {min_required}, have {df_len}."
            )
            return pd.DataFrame()

        step_size = max(self.test_window_size // 2, 200)

        available_steps = (df_len - self.train_window_size - self.test_window_size) // step_size + 1
        max_iterations = min(self.walk_forward_steps, available_steps)

        self.logger.info(f"Starting walk-forward backtest with {max_iterations} iteration(s)")

        all_results = []
        cumulative_capital = self.risk_manager.initial_capital

        for iteration in range(1, max_iterations + 1):
            start_idx = (iteration - 1) * step_size

            self.logger.info(f"Iteration {iteration}/{max_iterations} | StartIdx={start_idx}")

            result = self._run_iteration(iteration, start_idx, df_features, cumulative_capital)

            if result:
                all_results.append(result)

                if 'final_equity' in result:
                    cumulative_capital = result['final_equity']

                if 'trades' in result and result['trades']:
                    self.consolidated_trades.extend(result['trades'])

            tf.keras.backend.clear_session()

            if iteration % 3 == 0 and self.consolidated_trades:
                self.time_manager.optimize_time_parameters()

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
                'sharpe_ratio': consolidated_metrics.get('sharpe_ratio', 0)
            }
            df_results = pd.concat([df_results, pd.DataFrame([consolidated_row])], ignore_index=True)

            self._export_trade_details(final_equity)

            time_stats = self.time_manager.get_exit_performance_stats()
            self._export_time_analysis(time_stats)

        self.logger.info(f"Walk-forward complete. Produced {len(df_results)} results")
        return df_results

    # These changes should be applied to the _run_iteration method in your BacktestEngine class

    def _run_iteration(self, iteration: int, start_idx: int, df_features: pd.DataFrame, cumulative_capital: float) -> \
    Dict[str, Any]:
        train_end = start_idx + self.train_window_size
        test_end = min(train_end + self.test_window_size, len(df_features))

        df_train = df_features.iloc[start_idx:train_end].copy()
        df_test = df_features.iloc[train_end:test_end].copy()

        # Add logging to debug the feature computation
        self.logger.info(f"Processing features for training set of size {len(df_train)}")
        df_train = self.feature_engineer.compute_advanced_features(df_train)
        self.logger.info(f"Processing features for test set of size {len(df_test)}")
        df_test = self.feature_engineer.compute_advanced_features(df_test)

        if df_train.empty or df_test.empty:
            self.logger.warning(f"Iteration {iteration}: empty train/test slices. Skipping.")
            return None

        try:
            X_train, y_train, X_val, y_val, df_val, fwd_returns_val = self.data_preparer.prepare_data(df_train)

            if len(X_train) == 0:
                self.logger.warning(f"Iteration {iteration}: no valid training data after preparation")
                return None

            class_weight_dict = self._compute_class_weights(y_train)

            self.logger.info(f"Iteration {iteration}: training model on {len(X_train)} sample(s)")
            self.model.train_model(
                X_train, y_train,
                X_val, y_val,
                df_val, fwd_returns_val,
                class_weight=class_weight_dict
            )

        except Exception as e:
            self.logger.error(f"Iteration {iteration}: error in training => {e}")
            return None

        local_risk_manager = type(self.risk_manager)(self.config)
        local_risk_manager.initial_capital = cumulative_capital
        local_risk_manager.current_capital = cumulative_capital
        local_risk_manager.peak_capital = cumulative_capital

        simulation_result = self._simulate_trading(
            iteration, df_test, local_risk_manager
        )

        final_equity = simulation_result.get("final_equity", local_risk_manager.initial_capital)
        trades = simulation_result.get("trades", [])
        metrics = simulation_result.get("metrics", {})

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
            "trades": trades
        }

        return result

    # And these changes should be applied to the _simulate_trading method in your BacktestEngine class

    # Modify in backtest_engine.py - _simulate_trading method

    def _simulate_trading(self, iteration: int, df_test: pd.DataFrame,
                          risk_manager) -> Dict[str, Any]:
        try:
            X_test, y_test, df_labeled, fwd_returns_test = self.data_preparer.prepare_test_data(df_test)
        except Exception as e:
            self.logger.error(f"Error in prepare_test_data: {e}")
            return {"final_equity": risk_manager.initial_capital, "trades": [], "metrics": {}}

        if len(X_test) == 0:
            self.logger.warning("No test sequences available after preparation")
            return {"final_equity": risk_manager.initial_capital, "trades": [], "metrics": {}}

        has_valid_model = False
        if self.model.model is not None:
            has_valid_model = True
        elif hasattr(self.model, 'ensemble_models') and len(self.model.ensemble_models) > 0:
            has_valid_model = True

        if not has_valid_model:
            self.model.load_model()
            has_valid_model = (self.model.model is not None) or (
                    hasattr(self.model, 'ensemble_models') and len(self.model.ensemble_models) > 0
            )

            if not has_valid_model:
                self.logger.error("No valid model found to predict with. Skipping test.")
                return {"final_equity": risk_manager.initial_capital, "trades": [], "metrics": {}}

        self.logger.info(f"Predicting on test data of length {len(X_test)}")
        predictions = self.model.predict(X_test, batch_size=512)
        if len(predictions) != len(X_test):
            self.logger.error("Prediction shape mismatch")
            return {"final_equity": risk_manager.initial_capital, "trades": [], "metrics": {}}

        trades = []
        equity_curve = [risk_manager.current_capital]

        position = 0
        trade_entry = None
        last_signal_time = None

        seq_len = self.data_preparer.sequence_length

        signal_counts = {"Buy": 0, "Sell": 0, "NoTrade": 0}

        # NEW: Track consecutive trade outcomes to adapt strategy
        recent_outcomes = []
        consecutive_win_count = 0
        consecutive_loss_count = 0

        for i, model_probs in enumerate(predictions):
            if i % 100 == 0:
                self.logger.debug(f"Processing prediction {i}/{len(predictions)}")

            row_idx = i + seq_len - 1
            if row_idx >= len(df_test):
                break

            current_time = df_test.index[row_idx]
            current_price = self._get_current_price(df_test, row_idx)

            if np.isnan(current_price) or current_price <= 0:
                continue

            try:
                # NEW: Adjust signal generation based on recent performance
                adjusted_signal = {
                    "adaptive_mode": False,
                    "win_streak": consecutive_win_count,
                    "loss_streak": consecutive_loss_count
                }

                signal = self.signal_processor.generate_signal(
                    model_probs,
                    df_test.iloc[:row_idx + 1],
                    **adjusted_signal  # Add this parameter to your generate_signal method
                )

                # Count signal types for debugging
                if signal.get("signal_type", "").endswith("Buy"):
                    signal_counts["Buy"] += 1
                elif signal.get("signal_type", "").endswith("Sell"):
                    signal_counts["Sell"] += 1
                else:
                    signal_counts["NoTrade"] += 1

            except Exception as e:
                self.logger.error(f"Signal error at {current_time}: {e}")
                signal = {"signal_type": "NoTrade", "reason": f"SignalError_{e}"}
                signal_counts["NoTrade"] += 1

            if position != 0 and trade_entry is not None:
                trade_duration = (current_time - trade_entry['entry_time']).total_seconds() / 3600

                atr = self._compute_atr(df_test, row_idx, current_price)

                ema_20 = self._get_ema_20(df_test, row_idx)
                rsi_14 = self._get_rsi_14(df_test, row_idx)
                macd_values = self._get_macd_values(df_test, row_idx)

                market_conditions = {
                    "market_phase": signal.get("market_phase", "neutral"),
                    "volatility": float(signal.get("volatility", 0.5)),
                    "regime": float(signal.get("regime", 0)),
                    "atr": atr,
                    "momentum": float(signal.get("momentum", 0))
                }

                time_exit_decision = self.time_manager.evaluate_time_based_exit(
                    trade_entry, current_price, current_time, market_conditions
                )

                if time_exit_decision.get("exit", False):
                    trades, position, trade_entry = self._finalize_exit(
                        iteration, risk_manager, trades, trade_entry,
                        current_time, current_price, time_exit_decision,
                        df_test, row_idx
                    )

                    # NEW: Track consecutive outcomes
                    trade_pnl = trades[-1].get('pnl', 0)
                    if trade_pnl > 0:
                        consecutive_win_count += 1
                        consecutive_loss_count = 0
                        recent_outcomes.append(1)  # Win
                    else:
                        consecutive_loss_count += 1
                        consecutive_win_count = 0
                        recent_outcomes.append(-1)  # Loss

                    # Keep only last 5 outcomes
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
                    macd_histogram=macd_values.get('macd_histogram', 0)
                )

                if exit_decision.get("exit", False):
                    trades, position, trade_entry = self._finalize_exit(
                        iteration, risk_manager, trades, trade_entry,
                        current_time, current_price, exit_decision,
                        df_test, row_idx
                    )

                    # NEW: Track consecutive outcomes
                    trade_pnl = trades[-1].get('pnl', 0)
                    if trade_pnl > 0:
                        consecutive_win_count += 1
                        consecutive_loss_count = 0
                        recent_outcomes.append(1)  # Win
                    else:
                        consecutive_loss_count += 1
                        consecutive_win_count = 0
                        recent_outcomes.append(-1)  # Loss

                    # Keep only last 5 outcomes
                    if len(recent_outcomes) > 5:
                        recent_outcomes.pop(0)

                    equity_curve.append(risk_manager.current_capital)
                    last_signal_time = current_time
                else:
                    if exit_decision.get("update_stop", False):
                        new_stop = float(exit_decision.get("new_stop", 0))
                        if not np.isnan(new_stop) and new_stop > 0:
                            trade_entry['current_stop_loss'] = new_stop

                    if time_exit_decision.get("update_stop", False):
                        new_stop = float(time_exit_decision.get("new_stop", 0))
                        if not np.isnan(new_stop) and new_stop > 0:
                            if (trade_entry['direction'] == 'long' and new_stop > trade_entry['current_stop_loss']) or \
                                    (trade_entry['direction'] == 'short' and new_stop < trade_entry[
                                        'current_stop_loss']):
                                trade_entry['current_stop_loss'] = new_stop

                    partial_exit = risk_manager.get_partial_exit_level(
                        trade_entry['direction'],
                        trade_entry['entry_price'],
                        current_price
                    )

                    if partial_exit and not trade_entry.get('partial_exit_taken', False):
                        trades, position, trade_entry = self._execute_partial_exit(
                            iteration, risk_manager, trades, trade_entry,
                            current_time, current_price, partial_exit,
                            df_test, row_idx
                        )
                        equity_curve.append(risk_manager.current_capital)

            if position == 0:
                # NEW: Adaptive min hours between trades based on recent performance
                min_hours_adjusted = self.min_hours_between_trades

                # If on a winning streak, be more aggressive
                if consecutive_win_count >= 3:
                    min_hours_adjusted = max(1, self.min_hours_between_trades * 0.7)
                # If on a losing streak, be more cautious
                elif consecutive_loss_count >= 2:
                    min_hours_adjusted = self.min_hours_between_trades * 1.5

                if last_signal_time is not None:
                    hours_since_last = (current_time - last_signal_time).total_seconds() / 3600
                    if hours_since_last < min_hours_adjusted:
                        continue

                sig_type = signal.get('signal_type', '')
                if sig_type.endswith('Buy') or sig_type.endswith('Sell'):
                    can_trade, _ = risk_manager.check_correlation_risk(signal)

                    if can_trade:
                        direction = 'long' if sig_type.endswith('Buy') else 'short'

                        entry_price = current_price
                        slip = entry_price * self.slippage
                        entry_price = entry_price + slip if direction == 'long' else entry_price - slip

                        stop_loss = float(signal.get('stop_loss', 0))
                        if np.isnan(stop_loss) or stop_loss <= 0:
                            atr = self._compute_atr(df_test, row_idx, current_price)
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
                                'consecutive_wins': consecutive_win_count,  # NEW
                                'consecutive_losses': consecutive_loss_count  # NEW
                            }

                            trade_entry['entry_ema_20'] = self._get_ema_20(df_test, row_idx)
                            trade_entry['entry_rsi_14'] = self._get_rsi_14(df_test, row_idx)
                            macd_values = self._get_macd_values(df_test, row_idx)
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

        if position != 0 and trade_entry is not None:
            trade_entry['exit_signal'] = "EndOfTest"

            final_exit_price = self._get_current_price(df_test, len(df_test) - 1)
            if np.isnan(final_exit_price) or final_exit_price <= 0:
                final_exit_price = float(trade_entry['entry_price'])

            exit_decision = {"exit": True, "reason": "EndOfTest", "exit_price": final_exit_price}
            trades, position, trade_entry = self._finalize_exit(
                iteration, risk_manager, trades, trade_entry,
                df_test.index[-1], final_exit_price, exit_decision,
                df_test, len(df_test) - 1
            )
            equity_curve.append(risk_manager.current_capital)

        metrics = self._calculate_performance_metrics(trades, equity_curve, risk_manager.current_capital)

        for trade in trades:
            self.time_manager.update_duration_stats(trade)

        self.logger.info(f"Simulation completed with {len(trades)} trades")

        return {
            "final_equity": risk_manager.current_capital,
            "trades": trades,
            "metrics": metrics,
            "equity_curve": equity_curve
        }

    def _get_current_price(self, df: pd.DataFrame, row_idx: int) -> float:
        if 'actual_close' in df.columns:
            return float(df['actual_close'].iloc[row_idx])
        else:
            return float(df['close'].iloc[row_idx])

    def _get_ema_20(self, df: Optional[pd.DataFrame], row_idx: Optional[int]) -> float:
        if df is None or row_idx is None:
            return 0.0

        if row_idx < 0 or row_idx >= len(df):
            return 0.0

        if 'm30_ema_20' in df.columns:
            try:
                value = float(df['m30_ema_20'].iloc[row_idx])
                if not np.isnan(value) and value > 0:
                    return value
            except:
                pass

        if 'ema_20' in df.columns:
            try:
                value = float(df['ema_20'].iloc[row_idx])
                if not np.isnan(value) and value > 0:
                    return value
            except:
                pass

        return 0.0

    def _get_rsi_14(self, df: Optional[pd.DataFrame], row_idx: Optional[int]) -> float:
        if df is None or row_idx is None:
            return 50.0

        if row_idx < 0 or row_idx >= len(df):
            return 50.0

        if 'm30_rsi_14' in df.columns:
            try:
                value = float(df['m30_rsi_14'].iloc[row_idx])
                if not np.isnan(value) and 0 <= value <= 100:
                    return value
            except:
                pass

        if 'rsi_14' in df.columns:
            try:
                value = float(df['rsi_14'].iloc[row_idx])
                if not np.isnan(value) and 0 <= value <= 100:
                    return value
            except:
                pass

        return 50.0

    def _get_macd_values(self, df: Optional[pd.DataFrame], row_idx: Optional[int]) -> Dict[str, float]:
        result = {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}

        if df is None or row_idx is None:
            return result

        if row_idx < 0 or row_idx >= len(df):
            return result

        macd_columns = [
            ('m30_macd', 'macd'),
            ('m30_macd_signal', 'macd_signal'),
            ('m30_macd_histogram', 'macd_histogram'),
            ('macd', 'macd'),
            ('macd_signal', 'macd_signal'),
            ('macd_histogram', 'macd_histogram')
        ]

        for col, key in macd_columns:
            if col in df.columns:
                try:
                    value = float(df[col].iloc[row_idx])
                    if not np.isnan(value):
                        result[key] = value
                except:
                    pass

        return result

    def _compute_atr(self, df: pd.DataFrame, row_idx: int, current_price: float) -> float:
        for col in ['atr_14', 'm30_atr_14']:
            if col in df.columns:
                try:
                    atr_val = float(df[col].iloc[row_idx])
                    if not np.isnan(atr_val) and atr_val > 0:
                        return atr_val
                except:
                    pass

        return current_price * 0.01

    def _finalize_exit(self, iteration: int, risk_manager, trades: List[Dict[str, Any]],
                       trade_entry: Dict[str, Any], current_time, current_price: float,
                       exit_decision: Dict[str, Any], df_test=None, row_idx=None) -> Tuple[
        List[Dict[str, Any]], int, Optional[Dict[str, Any]]]:
        direction = trade_entry['direction']

        exit_price = float(exit_decision.get('exit_price', current_price))
        if np.isnan(exit_price) or exit_price <= 0:
            exit_price = current_price

        slip_amount = exit_price * self.slippage
        exit_price = (exit_price - slip_amount) if direction == 'long' else (exit_price + slip_amount)

        qty_open = float(trade_entry['quantity'])
        if np.isnan(qty_open) or qty_open <= 0:
            qty_open = 0.01

        exit_cost = self.fixed_cost + (exit_price * qty_open * self.variable_cost)

        entry_px = float(trade_entry['entry_price'])
        if np.isnan(entry_px) or entry_px <= 0:
            entry_px = current_price

        if direction == 'long':
            close_pnl = qty_open * (exit_price - entry_px) - exit_cost
        else:
            close_pnl = qty_open * (entry_px - exit_price) - exit_cost

        if np.isnan(close_pnl):
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
            t_rec['exit_ema_20'] = self._get_ema_20(df_test, row_idx)
            t_rec['exit_rsi_14'] = self._get_rsi_14(df_test, row_idx)
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

        trades.append(t_rec)

        risk_manager.update_after_trade(t_rec)

        position = 0
        trade_entry = None

        return trades, position, trade_entry

    def _execute_partial_exit(self, iteration: int, risk_manager, trades: List[Dict[str, Any]],
                              trade_entry: Dict[str, Any], current_time, current_price: float,
                              partial_exit: Dict[str, Any], df_test=None, row_idx=None) -> Tuple[
        List[Dict[str, Any]], int, Dict[str, Any]]:
        direction = trade_entry['direction']
        exit_price = current_price

        slip_amount = exit_price * self.slippage
        exit_price = (exit_price - slip_amount) if direction == 'long' else (exit_price + slip_amount)

        portion = partial_exit['portion']
        original_qty = float(trade_entry['quantity'])
        exit_qty = original_qty * portion
        remaining_qty = original_qty - exit_qty

        exit_cost = self.fixed_cost + (exit_price * exit_qty * self.variable_cost)

        entry_px = float(trade_entry['entry_price'])

        if direction == 'long':
            partial_pnl = exit_qty * (exit_price - entry_px) - exit_cost
        else:
            partial_pnl = exit_qty * (entry_px - exit_price) - exit_cost

        if np.isnan(partial_pnl):
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
            partial_rec['exit_ema_20'] = self._get_ema_20(df_test, row_idx)
            partial_rec['exit_rsi_14'] = self._get_rsi_14(df_test, row_idx)
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

        trades.append(partial_rec)

        risk_manager.current_capital += partial_pnl

        trade_entry['quantity'] = remaining_qty
        trade_entry['partial_exit_taken'] = True

        position = 1 if direction == 'long' else -1

        self.logger.debug(f"Partial exit at {current_time}: {exit_qty} @ ${exit_price}, PnL: ${partial_pnl:.2f}")

        return trades, position, trade_entry

    def _compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
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
                                       final_equity: float) -> Dict[str, float]:
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0,
                'avg_trade': 0, 'return': 0
            }

        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) <= 0]
        total_tr = len(trades)

        w_rate = len(wins) / total_tr if total_tr else 0

        prof_sum = sum(t.get('pnl', 0) for t in wins)
        loss_sum = abs(sum(t.get('pnl', 0) for t in losses))
        pf = prof_sum / max(loss_sum, 1e-10)

        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev_val = equity_curve[i - 1]
            if prev_val > 0:
                daily_returns.append((equity_curve[i] - prev_val) / prev_val)
            else:
                daily_returns.append(0)

        if len(daily_returns) > 1:
            avg_ret = np.mean(daily_returns)
            std_ret = max(np.std(daily_returns), 1e-10)
            sharpe = (avg_ret / std_ret) * np.sqrt(252)
        else:
            sharpe = 0

        max_dd = 0
        peak = equity_curve[0]

        for value in equity_curve:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

        initial_capital = equity_curve[0]
        total_ret = ((final_equity - initial_capital) / initial_capital) * 100

        avg_entry_rsi = np.mean([t.get('entry_rsi_14', 50) for t in trades])
        avg_exit_rsi = np.mean([t.get('exit_rsi_14', 50) for t in trades])

        if wins:
            avg_win_entry_macd_hist = np.mean([w.get('entry_macd_histogram', 0) for w in wins])
            avg_win_exit_macd_hist = np.mean([w.get('exit_macd_histogram', 0) for w in wins])
        else:
            avg_win_entry_macd_hist = 0
            avg_win_exit_macd_hist = 0

        avg_hours_in_trade = 0
        for t in trades:
            if 'entry_time' in t and 'exit_time' in t:
                try:
                    duration = (t['exit_time'] - t['entry_time']).total_seconds() / 3600
                    avg_hours_in_trade += duration
                except:
                    pass

        if trades:
            avg_hours_in_trade /= len(trades)

        return {
            'total_trades': total_tr,
            'win_rate': w_rate,
            'profit_factor': pf,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'return': total_ret,
            'avg_entry_rsi': avg_entry_rsi,
            'avg_exit_rsi': avg_exit_rsi,
            'avg_win_entry_macd_hist': avg_win_entry_macd_hist,
            'avg_win_exit_macd_hist': avg_win_exit_macd_hist,
            'avg_hours_in_trade': avg_hours_in_trade
        }

    def _calculate_consolidated_metrics(self, final_equity: float) -> Dict[str, float]:
        if not self.consolidated_trades:
            return {}

        eq_curve = [self.risk_manager.initial_capital]
        balance = self.risk_manager.initial_capital

        for tr in sorted(self.consolidated_trades, key=lambda x: x['exit_time']):
            balance += tr.get('pnl', 0)
            eq_curve.append(balance)

        ret_pct = ((final_equity / self.risk_manager.initial_capital) - 1) * 100

        wins = [t for t in self.consolidated_trades if t.get('pnl', 0) > 0]
        total_tr = len(self.consolidated_trades)
        win_rate = len(wins) / total_tr if total_tr else 0

        losses = [t for t in self.consolidated_trades if t.get('pnl', 0) <= 0]
        p_sum = sum(t.get('pnl', 0) for t in wins)
        n_sum = abs(sum(t.get('pnl', 0) for t in losses))
        pf = p_sum / max(n_sum, 1e-10)

        max_dd = 0
        peak = eq_curve[0]

        for value in eq_curve:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

        daily_returns = []
        for i in range(1, len(eq_curve)):
            prev_val = eq_curve[i - 1]
            if prev_val > 0:
                daily_returns.append((eq_curve[i] - prev_val) / prev_val)
            else:
                daily_returns.append(0)

        if len(daily_returns) > 1:
            avg_ret = np.mean(daily_returns)
            std_ret = max(np.std(daily_returns), 1e-10)
            sharpe = (avg_ret / std_ret) * np.sqrt(252)
        else:
            sharpe = 0

        avg_entry_rsi = np.mean([t.get('entry_rsi_14', 50) for t in self.consolidated_trades])
        avg_exit_rsi = np.mean([t.get('exit_rsi_14', 50) for t in self.consolidated_trades])

        if wins:
            avg_win_entry_macd_hist = np.mean([w.get('entry_macd_histogram', 0) for w in wins])
            avg_win_exit_macd_hist = np.mean([w.get('exit_macd_histogram', 0) for w in wins])
        else:
            avg_win_entry_macd_hist = 0
            avg_win_exit_macd_hist = 0

        if wins:
            exit_reasons = {}
            for w in wins:
                reason = w.get('exit_signal', 'Unknown')
                if reason in exit_reasons:
                    exit_reasons[reason]['count'] += 1
                    exit_reasons[reason]['profit'] += w.get('pnl', 0)
                else:
                    exit_reasons[reason] = {'count': 1, 'profit': w.get('pnl', 0)}

            for reason in exit_reasons:
                exit_reasons[reason]['avg_profit'] = exit_reasons[reason]['profit'] / exit_reasons[reason]['count']

            best_exit_reason = max(exit_reasons.items(), key=lambda x: x[1]['avg_profit'])[0]
        else:
            best_exit_reason = "None"

        return {
            'return': ret_pct,
            'win_rate': win_rate,
            'profit_factor': pf,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'avg_entry_rsi': avg_entry_rsi,
            'avg_exit_rsi': avg_exit_rsi,
            'avg_win_entry_macd_hist': avg_win_entry_macd_hist,
            'avg_win_exit_macd_hist': avg_win_exit_macd_hist,
            'best_exit_reason': best_exit_reason
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
                'market_phase': tr.get('market_phase', 'unknown'),
                'trend_strength': round(float(tr.get('trend_strength', 0)), 2),
                'duration_hours': round((tr.get('exit_time') - tr.get('entry_time')).total_seconds() / 3600, 1)
            }
            trade_records.append(record)

        df_trades = pd.DataFrame(trade_records)
        csv_path = self.output_dir / f'trade_details_{timestamp}.csv'
        df_trades.to_csv(csv_path, index=False)

        summary_path = self.output_dir / f'trade_summary_{timestamp}.txt'

        with open(summary_path, 'w') as f:
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
            if not loss_trades.empty:
                f.write(f"Average Loss: ${loss_trades['pnl'].mean():.2f}\n")

            total_profit = win_trades['pnl'].sum() if not win_trades.empty else 0
            total_loss = abs(loss_trades['pnl'].sum()) if not loss_trades.empty else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            f.write(f"Profit Factor: {profit_factor:.2f}\n\n")

            f.write(f"Average Trade Duration: {df_trades['duration_hours'].mean():.1f} hours\n\n")

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

        self.logger.info(f"Exported time analysis to {time_path}")