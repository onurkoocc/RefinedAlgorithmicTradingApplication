import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from uuid import uuid4
from collections import deque
import tensorflow as tf

class BacktestEngine:
    def __init__(self, config, data_preparer, model, signal_generator, risk_manager):
        self.config = config
        self.logger = logging.getLogger("BacktestEngine")

        self.data_preparer = data_preparer
        self.model = model
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager

        from adaptive_time_management import AdaptiveTimeManager
        from metric_calculator import MetricCalculator
        from exporter import Exporter
        from indicator_util import IndicatorUtil

        self.time_manager = AdaptiveTimeManager(config)
        self.metric_calculator = MetricCalculator(config)
        self.exporter = Exporter(config)
        self.indicator_util = IndicatorUtil()

        self.train_window_size = config.get("backtest", "train_window_size", 4500)
        self.test_window_size = config.get("backtest", "test_window_size", 500)
        self.walk_forward_steps = config.get("backtest", "walk_forward_steps", 5)

        self.slippage = config.get("backtest", "slippage", 0.0004)
        self.fixed_cost = config.get("backtest", "fixed_cost", 0.0009)
        self.variable_cost = config.get("backtest", "variable_cost", 0.00045)

        self.results_dir = Path(config.results_dir) / "backtest"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self.consolidated_trades = []
        self.portfolio_manager = PortfolioManager(config)
        self.portfolio_manager.set_risk_manager(self.risk_manager)
        self.market_simulator = MarketSimulator(config, self.portfolio_manager)
        self.walk_forward_manager = WalkForwardManager(config)
        self.optimization_engine = OptimizationEngine(config)
        self.performance_analyzer = PerformanceAnalyzer(config)

        self.total_partial_exits = 0
        self.total_quick_profit_exits = 0
        self.avg_trade_holding_time = 0
        self.market_phase_stats = {}

    def run_backtest(self, df_features: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting backtest with advanced signal generation and exit strategies")

        if len(df_features) < (self.train_window_size + self.test_window_size):
            self.logger.error(
                f"Insufficient data for backtest. Need at least {self.train_window_size + self.test_window_size} samples.")
            return pd.DataFrame()

        walk_forward_windows = self.walk_forward_manager.create_windows(
            df_features,
            self.train_window_size,
            self.test_window_size,
            self.walk_forward_steps
        )

        all_results = []
        self.portfolio_manager.reset()

        for i, (train_df, test_df, window_info) in enumerate(walk_forward_windows):
            self.logger.info(f"Processing walk-forward window {i + 1}/{len(walk_forward_windows)}: {window_info}")

            iteration_result = self._run_window_backtest(i + 1, train_df, test_df, window_info)

            if iteration_result:
                all_results.append(iteration_result)
                self._update_optimization_parameters(iteration_result)

            tf.keras.backend.clear_session()

        self._process_consolidated_results(all_results)
        return self._create_results_dataframe(all_results)

    def _run_window_backtest(self, iteration: int, train_df: pd.DataFrame, test_df: pd.DataFrame,
                             window_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            model_trained = self._train_model_for_window(train_df)
            if not model_trained:
                return None

            simulation_result = self._simulate_trading(iteration, test_df)

            if not simulation_result:
                return None

            window_metrics = self.performance_analyzer.calculate_window_metrics(
                simulation_result["trades"],
                simulation_result["equity_curve"],
                window_info
            )

            result = {
                "iteration": iteration,
                "window_info": window_info,
                "final_equity": simulation_result["final_equity"],
                "trades": simulation_result["trades"],
                "metrics": window_metrics,
                "trade_count": len(simulation_result["trades"]),
                "daily_returns": simulation_result["daily_returns"]
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in window backtest iteration {iteration}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _train_model_for_window(self, train_df: pd.DataFrame) -> bool:
        try:
            prepared_data = self.data_preparer.prepare_data(train_df)

            if not prepared_data or len(prepared_data[0]) == 0:
                self.logger.warning("No valid training data after preparation")
                return False

            X_train, y_train, X_val, y_val, df_val, fwd_returns_val = prepared_data

            if hasattr(self.model, 'model') and self.model.model is not None:
                validation_score = self._validate_existing_model(X_val, y_val)
                if validation_score >= self.config.get("backtest", "train_confidence_threshold", 0.65):
                    self.logger.info(f"Skipping retraining as model validation score is good: {validation_score:.4f}")
                    return True

            class_weights = self._compute_class_weights(y_train)
            self.model.train_model(X_train, y_train, X_val, y_val, df_val, fwd_returns_val, class_weight=class_weights)
            return True

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _validate_existing_model(self, X_val, y_val) -> float:
        try:
            predictions = self.model.predict(X_val)
            correct_direction = np.sum(np.sign(predictions.flatten()) == np.sign(y_val))
            return correct_direction / len(y_val) if len(y_val) > 0 else 0.0
        except Exception as e:
            self.logger.warning(f"Error in model validation: {e}")
            return 0.0

    def _compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        from sklearn.utils import compute_class_weight

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

    def _simulate_trading(self, iteration: int, test_df: pd.DataFrame) -> Dict[str, Any]:
        try:
            prepared_test_data = self.data_preparer.prepare_test_data(test_df)

            if not prepared_test_data or len(prepared_test_data[0]) == 0:
                self.logger.warning("No valid test data after preparation")
                return None

            X_test, y_test, df_labeled, fwd_returns_test = prepared_test_data

            if not self._ensure_model_loaded():
                return None

            predictions = self.model.predict(X_test)

            simulation_result = self.market_simulator.simulate(
                iteration,
                predictions,
                df_labeled,
                fwd_returns_test,
                self.signal_generator,
                self.risk_manager,
                self.time_manager
            )

            if simulation_result and "trades" in simulation_result:
                self.consolidated_trades.extend(simulation_result["trades"])

                for trade in simulation_result["trades"]:
                    self.time_manager.update_duration_stats(trade)

            return simulation_result

        except Exception as e:
            self.logger.error(f"Error in trading simulation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _ensure_model_loaded(self) -> bool:
        has_valid_model = False
        if self.model.model is not None:
            has_valid_model = True
        elif (hasattr(self.model, 'ensemble_models') and
              self.model.ensemble_models is not None and
              len(self.model.ensemble_models.models) > 0):
            has_valid_model = True

        if not has_valid_model:
            self.model.load_model()
            has_valid_model = (self.model.model is not None) or (
                    hasattr(self.model, 'ensemble_models') and
                    self.model.ensemble_models is not None and
                    len(self.model.ensemble_models.models) > 0
            )

            if not has_valid_model:
                self.logger.error("No valid model found for prediction. Skipping test.")
                return False

        return True

    def _update_optimization_parameters(self, iteration_result: Dict[str, Any]) -> None:
        if not iteration_result or not iteration_result.get("trades"):
            return

        optimize_frequency = self.config.get("backtest", "optimize_every_n_iterations", 1)
        if iteration_result["iteration"] % optimize_frequency == 0:
            self.time_manager.optimize_time_parameters()
            self.optimization_engine.optimize_exit_strategies(self.consolidated_trades)
            self.optimization_engine.optimize_signal_thresholds(self.consolidated_trades)

            if hasattr(self.signal_generator, 'update_parameters'):
                optimized_params = self.optimization_engine.get_signal_parameters()
                self.signal_generator.update_parameters(optimized_params)

            if hasattr(self.risk_manager, 'update_parameters'):
                optimized_params = self.optimization_engine.get_risk_parameters()
                self.risk_manager.update_parameters(optimized_params)

    def _process_consolidated_results(self, all_results: List[Dict[str, Any]]) -> None:
        if not all_results or not self.consolidated_trades:
            self.logger.warning("No valid results to process")
            return

        final_equity = all_results[-1].get('final_equity', self.risk_manager.initial_capital)

        consolidated_metrics = self.metric_calculator.calculate_consolidated_metrics(
            self.consolidated_trades,
            final_equity
        )

        self.exporter.export_trade_details(self.consolidated_trades, final_equity,
                                           self.metric_calculator, self.risk_manager.initial_capital)
        self.exporter.export_feature_impact_analysis(self.consolidated_trades)

        time_stats = self.time_manager.get_exit_performance_stats()
        self.exporter.export_time_analysis(time_stats)

        self.total_partial_exits = sum(1 for t in self.consolidated_trades if t.get('is_partial', False))
        self.total_quick_profit_exits = sum(
            1 for t in self.consolidated_trades if t.get('exit_signal') == 'QuickProfitTaken')

        trade_durations = [t.get('duration_hours', 0) for t in self.consolidated_trades if 'duration_hours' in t]
        self.avg_trade_holding_time = sum(trade_durations) / len(trade_durations) if trade_durations else 0

        self.market_phase_stats = {}
        for trade in self.consolidated_trades:
            phase = trade.get('market_phase', 'neutral')
            self.market_phase_stats[phase] = self.market_phase_stats.get(phase, 0) + 1

        self.exporter.export_exit_strategy_analysis(self)
        self.exporter.export_drawdown_analysis(
            self.metric_calculator.drawdown_periods,
            self.consolidated_trades,
            self.portfolio_manager.current_capital,
            self.metric_calculator.peak_capital
        )

        self.exporter.export_monthly_performance(self.metric_calculator.monthly_returns)

        self.optimization_engine.export_optimization_results(self.results_dir)

    def _create_results_dataframe(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        if not all_results:
            return pd.DataFrame()

        df_results = pd.DataFrame(all_results)

        if self.consolidated_trades:
            final_equity = all_results[-1].get('final_equity', self.risk_manager.initial_capital)
            consolidated_metrics = self.metric_calculator.calculate_consolidated_metrics(
                self.consolidated_trades,
                final_equity
            )

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

        return df_results


class MarketSimulator:
    def __init__(self, config, portfolio_manager):
        self.config = config
        self.logger = logging.getLogger("MarketSimulator")
        self.portfolio_manager = portfolio_manager

        self.slippage = config.get("backtest", "slippage", 0.0004)
        self.fixed_cost = config.get("backtest", "fixed_cost", 0.0009)
        self.variable_cost = config.get("backtest", "variable_cost", 0.00045)
        self.use_dynamic_slippage = config.get("backtest", "use_dynamic_slippage", True)
        self.min_hours_between_trades = config.get("backtest", "min_hours_between_trades", 1)

        self.signal_stats = {"Buy": 0, "Sell": 0, "NoTrade": 0}
        self.no_trade_reasons = {}
        self.position = 0
        self.trade_entry = None
        self.last_signal_time = None
        self.equity_curve = []
        self.daily_returns = []
        self.trades = []
        self.peak_capital = 0
        self.drawdown_periods = []
        self.iteration = 0

    def simulate(self, iteration, predictions, df_labeled, fwd_returns,
                 signal_generator, risk_manager, time_manager) -> Dict[str, Any]:
        self._reset_simulation_state()
        self.iteration = iteration

        self.equity_curve = [self.portfolio_manager.current_capital]
        self.peak_capital = self.portfolio_manager.current_capital

        seq_len = self.config.get("model", "sequence_length", 72)
        last_day = None

        for i, model_probs in enumerate(predictions):
            if i % 100 == 0:
                self.logger.debug(f"Processing prediction {i}/{len(predictions)}")

            row_idx = i + seq_len
            if row_idx >= len(df_labeled):
                break

            current_time = df_labeled.index[row_idx]

            current_day = current_time.date()
            if last_day is not None and current_day != last_day:
                if len(self.equity_curve) >= 2:
                    daily_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
                    self.daily_returns.append(daily_return)
            last_day = current_day

            current_price = self._get_current_price(df_labeled, row_idx)
            if np.isnan(current_price) or current_price <= 0:
                continue

            self._update_drawdown_metrics(i, current_price)

            if self.position != 0 and self.trade_entry is not None:
                self._evaluate_existing_position(
                    row_idx, current_time, current_price, df_labeled,
                    risk_manager, time_manager, signal_generator
                )

            if self.position == 0:
                self._evaluate_new_entry(
                    i, row_idx, model_probs, current_time, current_price, df_labeled,
                    signal_generator, risk_manager
                )

        if self.position != 0 and self.trade_entry is not None:
            self._close_final_position(df_labeled)

        return {
            "final_equity": self.portfolio_manager.current_capital,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "daily_returns": self.daily_returns,
            "drawdown_periods": self.drawdown_periods,
            "no_trade_reasons": self.no_trade_reasons,
            "signal_stats": self.signal_stats
        }

    def _reset_simulation_state(self):
        self.position = 0
        self.trade_entry = None
        self.last_signal_time = None
        self.equity_curve = []
        self.daily_returns = []
        self.trades = []
        self.signal_stats = {"Buy": 0, "Sell": 0, "NoTrade": 0}
        self.no_trade_reasons = {}
        self.peak_capital = self.portfolio_manager.current_capital
        self.drawdown_periods = []

    def _update_drawdown_metrics(self, i, current_price):
        if self.portfolio_manager.current_capital > self.peak_capital:
            self.peak_capital = self.portfolio_manager.current_capital
            if self.portfolio_manager.current_drawdown_start is not None:
                drawdown_duration = i - self.portfolio_manager.current_drawdown_start
                if drawdown_duration > 0 and self.portfolio_manager.current_drawdown > 0.05:
                    self.drawdown_periods.append({
                        'start_idx': self.portfolio_manager.current_drawdown_start,
                        'end_idx': i,
                        'depth': self.portfolio_manager.current_drawdown,
                        'duration': drawdown_duration
                    })
                self.portfolio_manager.current_drawdown_start = None
        elif self.portfolio_manager.current_capital < self.peak_capital:
            current_drawdown = (self.peak_capital - self.portfolio_manager.current_capital) / self.peak_capital
            self.portfolio_manager.current_drawdown = current_drawdown

            if self.portfolio_manager.current_drawdown_start is None:
                self.portfolio_manager.current_drawdown_start = i

    def _evaluate_existing_position(self, row_idx, current_time, current_price, df_labeled,
                                    risk_manager, time_manager, signal_generator):
        trade_duration = (current_time - self.trade_entry['entry_time']).total_seconds() / 3600

        atr = self._compute_atr(df_labeled, row_idx, current_price)
        market_conditions = self._gather_market_conditions(df_labeled, row_idx, signal_generator)
        market_conditions["current_time"] = current_time

        time_exit_decision = time_manager.evaluate_time_based_exit(
            self.trade_entry, current_price, current_time, market_conditions
        )

        if time_exit_decision.get("exit", False):
            exit_reason = time_exit_decision.get("reason", "TimeBasedExit")

            self._finalize_exit(
                current_time, current_price, time_exit_decision,
                df_labeled, row_idx, risk_manager
            )

            self.equity_curve.append(self.portfolio_manager.current_capital)
            self.last_signal_time = current_time
            return

        exit_decision = risk_manager.handle_exit_decision(
            self.trade_entry,
            current_price,
            atr,
            trade_duration=trade_duration,
            **market_conditions
        )

        if exit_decision.get("exit", False):
            self._finalize_exit(
                current_time, current_price, exit_decision,
                df_labeled, row_idx, risk_manager
            )

            self.equity_curve.append(self.portfolio_manager.current_capital)
            self.last_signal_time = current_time
            return

        if exit_decision.get("update_stop", False):
            new_stop = float(exit_decision.get("new_stop", 0))
            if not np.isnan(new_stop) and new_stop > 0:
                if (self.trade_entry['direction'] == 'long' and new_stop < current_price) or \
                        (self.trade_entry['direction'] == 'short' and new_stop > current_price):
                    self.trade_entry['current_stop_loss'] = new_stop

        if time_exit_decision.get("update_stop", False):
            new_stop = float(time_exit_decision.get("new_stop", 0))
            if not np.isnan(new_stop) and new_stop > 0:
                if (self.trade_entry['direction'] == 'long' and new_stop > self.trade_entry[
                    'current_stop_loss'] and new_stop < current_price) or \
                        (self.trade_entry['direction'] == 'short' and new_stop < self.trade_entry[
                            'current_stop_loss'] and new_stop > current_price):
                    self.trade_entry['current_stop_loss'] = new_stop

        partial_exit = risk_manager.get_partial_exit_levels(
            self.trade_entry['direction'],
            self.trade_entry['entry_price'],
            current_price,
            **market_conditions
        )

        if partial_exit and not self.trade_entry.get('partial_exit_taken', False):
            self._execute_partial_exit(
                current_time, current_price, partial_exit,
                df_labeled, row_idx, risk_manager
            )
            self.equity_curve.append(self.portfolio_manager.current_capital)

    def should_skip_trade(self, timestamp):
        if not isinstance(timestamp, datetime):
            return False

        weekday = timestamp.weekday()
        hour = timestamp.hour

        if (weekday == 5 and hour >= 2) or weekday == 6 or (weekday == 0 and hour < 6):
            return True

        if 2 <= hour < 6:
            return True

        if 22 <= hour < 23:
            return True

        return False

    def _evaluate_new_entry(self, i, row_idx, model_probs, current_time, current_price, df_labeled,
                            signal_generator, risk_manager):
        if self.should_skip_trade(current_time):
            return

        min_hours_adjusted = self._get_adjusted_min_hours()
        if self.last_signal_time is not None:
            hours_since_last = (current_time - self.last_signal_time).total_seconds() / 3600
            if hours_since_last < min_hours_adjusted:
                return

        try:
            adjusted_signal = {
                "adaptive_mode": True,
                "win_streak": self.portfolio_manager.current_win_streak,
                "loss_streak": self.portfolio_manager.current_loss_streak
            }

            signal = signal_generator.generate_signal(
                model_probs,
                df_labeled.iloc[:row_idx],
                **adjusted_signal
            )
        except Exception as e:
            self.logger.error(f"Signal error at {current_time}: {e}")
            signal = {"signal_type": "NoTrade", "reason": f"SignalError_{e}"}

        self._update_signal_stats(signal)

        sig_type = signal.get('signal_type', '')
        if sig_type.endswith('Buy') or sig_type.endswith('Sell'):
            can_trade, max_risk = risk_manager.check_correlation_risk(signal)

            if can_trade:
                direction = 'long' if sig_type.endswith('Buy') else 'short'

                if 'open' in df_labeled.columns:
                    entry_price = float(df_labeled['open'].iloc[row_idx])
                else:
                    entry_price = current_price

                entry_price = self._apply_slippage(entry_price, direction, signal)
                stop_loss = self._get_stop_loss(entry_price, direction, signal, df_labeled, row_idx)
                quantity = risk_manager.calculate_position_size(signal, entry_price, stop_loss)

                if quantity > 0:
                    trade_entry = self._create_trade_entry(
                        current_time, entry_price, direction, sig_type, stop_loss, quantity, signal, df_labeled, row_idx
                    )

                    self.trade_entry = trade_entry
                    self.position = 1 if direction == 'long' else -1
                    self.last_signal_time = current_time

                    self.logger.debug(f"Opened {direction} position at {current_time}: {quantity} @ ${entry_price}")

    def _get_adjusted_min_hours(self):
        min_hours_adjusted = self.min_hours_between_trades

        if self.portfolio_manager.current_win_streak >= 3:
            min_hours_adjusted = max(1, self.min_hours_between_trades * 0.7)
        elif self.portfolio_manager.current_loss_streak >= 2:
            min_hours_adjusted = self.min_hours_between_trades * 1.5

        return min_hours_adjusted

    def _update_signal_stats(self, signal):
        signal_type = signal.get("signal_type", "")

        if signal_type.endswith("Buy"):
            self.signal_stats["Buy"] += 1
        elif signal_type.endswith("Sell"):
            self.signal_stats["Sell"] += 1
        else:
            self.signal_stats["NoTrade"] += 1
            reason = signal.get("reason", "Unknown")
            self.no_trade_reasons[reason] = self.no_trade_reasons.get(reason, 0) + 1

    def _apply_slippage(self, price, direction, signal):
        if not self.use_dynamic_slippage:
            slip = price * self.slippage
            return price + slip if direction == 'long' else price - slip

        volatility = float(signal.get('volatility', 0.5))
        market_activity = float(signal.get('volume_roc', 0)) if 'volume_roc' in signal else 0
        ensemble_score = float(signal.get('ensemble_score', 0.5))

        dynamic_slippage = self.slippage * (
                0.8 + volatility * 0.6 +
                max(0, min(0.3, abs(market_activity) * 0.01)) -
                min(0.2, ensemble_score * 0.3)
        )

        slip = price * dynamic_slippage
        return price + slip if direction == 'long' else price - slip

    def _get_stop_loss(self, entry_price, direction, signal, df, row_idx):
        stop_loss = float(signal.get('stop_loss', 0))

        if np.isnan(stop_loss) or stop_loss <= 0:
            atr = self._compute_atr(df, row_idx - 1, entry_price)

            market_phase = signal.get('market_phase', 'neutral')
            volatility = float(signal.get('volatility', 0.5))

            if hasattr(self.portfolio_manager.risk_manager, 'atr_multiplier_map'):
                atr_multiplier = self.portfolio_manager.risk_manager.atr_multiplier_map.get(
                    market_phase, {}).get(direction, 2.2)
            else:
                atr_multiplier = 2.2

            if volatility > 0.7:
                atr_multiplier *= 0.9
            elif volatility < 0.3:
                atr_multiplier *= 1.1

            if direction == 'long':
                stop_loss = entry_price - (atr * atr_multiplier)
            else:
                stop_loss = entry_price + (atr * atr_multiplier)

        if (direction == 'long' and stop_loss >= entry_price) or \
                (direction == 'short' and stop_loss <= entry_price):
            self.logger.warning(f"Invalid stop loss {stop_loss} for {direction} at price {entry_price}. Adjusting.")
            stop_loss = entry_price * 0.95 if direction == 'long' else entry_price * 1.05

        return stop_loss

    def _create_trade_entry(self, current_time, entry_price, direction, sig_type,
                            stop_loss, quantity, signal, df, row_idx):
        entry_cost = self.fixed_cost + (entry_price * quantity * self.variable_cost)

        trade_entry = {
            'id': str(uuid4()),
            'iteration': self.iteration,
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
            'total_entry_cost': entry_cost,
            'partial_exit_taken': False,
            'ensemble_score': float(signal.get('ensemble_score', 0.5)),
            'market_phase': signal.get('market_phase', 'neutral'),
            'volume_confirmation': bool(signal.get('volume_confirmation', False)),
            'trend_strength': float(signal.get('trend_strength', 0.5)),
            'consecutive_wins': self.portfolio_manager.current_win_streak,
            'consecutive_losses': self.portfolio_manager.current_loss_streak
        }

        trade_entry['entry_ema_20'] = self._get_indicator_value(df, row_idx, 'ema_20', 'm30_ema_20')
        trade_entry['entry_rsi_14'] = self._get_indicator_value(df, row_idx, 'rsi_14', 'm30_rsi_14', default=50)

        macd_values = self._get_macd_values(df, row_idx)
        trade_entry['entry_macd'] = macd_values.get('macd', 0)
        trade_entry['entry_macd_signal'] = macd_values.get('macd_signal', 0)
        trade_entry['entry_macd_histogram'] = macd_values.get('macd_histogram', 0)
        trade_entry['atr'] = self._compute_atr(df, row_idx, entry_price)

        return trade_entry

    def _finalize_exit(self, current_time, current_price, exit_decision, df, row_idx, risk_manager):
        try:
            if self.trade_entry is None:
                return

            exit_price = float(exit_decision.get('exit_price', current_price))
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = current_price

            direction = self.trade_entry.get('direction', 'long')
            volatility = float(self.trade_entry.get('volatility_regime', 0.5))

            if self.use_dynamic_slippage:
                is_forced_exit = exit_decision.get('reason') in ['StopLoss', 'MaxDurationReached', 'StagnantPosition']
                slippage_factor = 1.1 if is_forced_exit else 0.7
                slippage_factor *= (0.8 + volatility * 0.3)

                slip_amount = exit_price * self.slippage * slippage_factor
            else:
                slip_amount = exit_price * self.slippage

            if direction == 'long':
                exit_price = max(0.01, exit_price - slip_amount)
            else:
                exit_price = exit_price + slip_amount

            qty_open = float(self.trade_entry.get('quantity', 0))
            exit_cost = self.fixed_cost + (exit_price * qty_open * self.variable_cost * 0.9)

            entry_px = float(self.trade_entry.get('entry_price', 0))

            if direction == 'long':
                close_pnl = qty_open * (exit_price - entry_px) - exit_cost - float(
                    self.trade_entry.get('total_entry_cost', 0))
            else:
                close_pnl = qty_open * (entry_px - exit_price) - exit_cost - float(
                    self.trade_entry.get('total_entry_cost', 0))

            trade_record = self.trade_entry.copy()
            trade_record.update({
                'exit_time': current_time,
                'exit_price': exit_price,
                'exit_signal': exit_decision.get('reason', 'ExitSignal'),
                'pnl': close_pnl
            })

            if df is not None and row_idx is not None and 0 <= row_idx < len(df):
                trade_record['exit_ema_20'] = self._get_indicator_value(df, row_idx, 'ema_20', 'm30_ema_20')
                trade_record['exit_rsi_14'] = self._get_indicator_value(df, row_idx, 'rsi_14', 'm30_rsi_14', default=50)

                exit_macd = self._get_macd_values(df, row_idx)
                trade_record['exit_macd'] = exit_macd.get('macd', 0)
                trade_record['exit_macd_signal'] = exit_macd.get('macd_signal', 0)
                trade_record['exit_macd_histogram'] = exit_macd.get('macd_histogram', 0)

            entry_time = trade_record.get('entry_time')
            exit_time = trade_record.get('exit_time')

            if entry_time and exit_time:
                duration_hours = (exit_time - entry_time).total_seconds() / 3600
                trade_record['duration_hours'] = duration_hours

            self.trades.append(trade_record)

            self.portfolio_manager.update_after_trade(close_pnl)

            # Use the risk_manager from portfolio manager if the passed risk_manager is None
            valid_risk_manager = risk_manager
            if valid_risk_manager is None and hasattr(self.portfolio_manager, 'risk_manager'):
                valid_risk_manager = self.portfolio_manager.risk_manager

            if valid_risk_manager is not None:
                valid_risk_manager.update_after_trade(trade_record)
            else:
                self.logger.warning("No valid risk_manager available for update_after_trade")

            self.position = 0
            self.trade_entry = None

            return trade_record

        except Exception as e:
            self.logger.error(f"Error in finalizing exit: {e}")

            self.position = 0
            self.trade_entry = None
            return None

    def _execute_partial_exit(self, current_time, current_price, partial_exit,
                              df, row_idx, risk_manager):
        try:
            if self.trade_entry is None:
                return

            direction = self.trade_entry.get('direction', '')
            entry_price = float(self.trade_entry.get('entry_price', 0))

            exit_price = current_price
            slip_factor = 0.8
            slip_amount = exit_price * self.slippage * slip_factor

            if direction == 'long':
                exit_price = max(0.01, exit_price - slip_amount)
            else:
                exit_price = exit_price + slip_amount

            portion = float(partial_exit.get('portion', 0.2))
            original_qty = float(self.trade_entry.get('quantity', 0))
            exit_qty = original_qty * portion
            remaining_qty = original_qty - exit_qty

            exit_cost = self.fixed_cost + (exit_price * exit_qty * self.variable_cost)
            proportional_entry_cost = float(self.trade_entry.get('total_entry_cost', 0)) * portion

            if direction == 'long':
                partial_pnl = exit_qty * (exit_price - entry_price) - exit_cost - proportional_entry_cost
            else:
                partial_pnl = exit_qty * (entry_price - exit_price) - exit_cost - proportional_entry_cost

            partial_record = self.trade_entry.copy()
            partial_record.update({
                'exit_time': current_time,
                'exit_price': exit_price,
                'exit_signal': f"PartialExit_{int(portion * 100)}pct",
                'pnl': partial_pnl,
                'quantity': exit_qty,
                'is_partial': True,
                'partial_id': partial_exit.get('id', '')
            })

            if df is not None and row_idx is not None and 0 <= row_idx < len(df):
                partial_record['exit_ema_20'] = self._get_indicator_value(df, row_idx, 'ema_20', 'm30_ema_20')
                partial_record['exit_rsi_14'] = self._get_indicator_value(df, row_idx, 'rsi_14', 'm30_rsi_14',
                                                                          default=50)

                exit_macd = self._get_macd_values(df, row_idx)
                partial_record['exit_macd'] = exit_macd.get('macd', 0)
                partial_record['exit_macd_signal'] = exit_macd.get('macd_signal', 0)
                partial_record['exit_macd_histogram'] = exit_macd.get('macd_histogram', 0)

            entry_time = partial_record.get('entry_time')
            if entry_time and current_time:
                duration_hours = (current_time - entry_time).total_seconds() / 3600
                partial_record['duration_hours'] = duration_hours

            self.trades.append(partial_record)
            self.portfolio_manager.current_capital += partial_pnl

            self.trade_entry['quantity'] = remaining_qty
            self.trade_entry['partial_exit_taken'] = True

            if 'update_position_flag' in partial_exit:
                flag_name = partial_exit['update_position_flag']
                self.trade_entry[flag_name] = True

            self.trade_entry['total_entry_cost'] = float(self.trade_entry.get('total_entry_cost', 0)) * (1 - portion)

            self.logger.debug(f"Partial exit at {current_time}: {exit_qty} @ ${exit_price}, PnL: ${partial_pnl:.2f}")

            return partial_record

        except Exception as e:
            self.logger.error(f"Error in partial exit: {e}")
            return None

    def _close_final_position(self, df_labeled):
        if self.position == 0 or self.trade_entry is None:
            return

        self.trade_entry['exit_signal'] = "EndOfTest"

        final_exit_price = self._get_current_price(df_labeled, len(df_labeled) - 1)
        if np.isnan(final_exit_price) or final_exit_price <= 0:
            final_exit_price = float(self.trade_entry['entry_price'])

        exit_decision = {"exit": True, "reason": "EndOfTest", "exit_price": final_exit_price}

        self._finalize_exit(
            df_labeled.index[-1],
            final_exit_price,
            exit_decision,
            df_labeled,
            len(df_labeled) - 1,
            self.portfolio_manager.risk_manager
        )

        self.equity_curve.append(self.portfolio_manager.current_capital)

    def _gather_market_conditions(self, df, row_idx, signal_generator):
        market_phase = self._detect_market_phase(df, row_idx, signal_generator)
        volatility = self._get_indicator_value(df, row_idx, 'volatility_regime', default=0.5)
        rsi_14 = self._get_indicator_value(df, row_idx, 'rsi_14', 'm30_rsi_14', default=50)
        bb_width = self._get_indicator_value(df, row_idx, 'bb_width_20', 'm30_bb_width_20', default=0.02)

        macd_values = self._get_macd_values(df, row_idx)

        # Get order flow metrics for the most recent candles
        recent_data = df.iloc[max(0, row_idx - 20):row_idx + 1]
        order_flow = {}

        if hasattr(signal_generator, 'volume_analyzer') and hasattr(signal_generator.volume_analyzer,
                                                                    'calculate_order_flow_metrics'):
            order_flow = signal_generator.volume_analyzer.calculate_order_flow_metrics(recent_data)
        else:
            # Fallback if method is not available
            order_flow = {
                "volume_delta": 0,
                "volume_trend": 0,
                "buying_pressure": 0.5,
                "selling_pressure": 0.5
            }

        return {
            "market_phase": market_phase,
            "volatility": float(volatility),
            "regime": self._get_indicator_value(df, row_idx, 'market_regime', default=0),
            "momentum": self._calculate_momentum(df, row_idx),
            "rsi_14": rsi_14,
            "bb_width": bb_width,
            "macd": macd_values.get('macd', 0),
            "macd_signal": macd_values.get('macd_signal', 0),
            "macd_histogram": macd_values.get('macd_histogram', 0),
            "volume_delta": order_flow.get("volume_delta", 0),
            "volume_trend": order_flow.get("volume_trend", 0),
            "buying_pressure": order_flow.get("buying_pressure", 0.5),
            "selling_pressure": order_flow.get("selling_pressure", 0.5)
        }

    def _detect_market_phase(self, df, row_idx, signal_generator):
        if hasattr(signal_generator, 'indicator_util') and hasattr(signal_generator.indicator_util,
                                                                   'detect_market_phase'):
            row_df = df.iloc[[row_idx]]
            return signal_generator.indicator_util.detect_market_phase(row_df)

        if 'market_phase' in df.columns:
            return df['market_phase'].iloc[row_idx]

        if 'adx_14' in df.columns and 'ema_9' in df.columns and 'ema_50' in df.columns:
            adx = df['adx_14'].iloc[row_idx]
            ema9 = df['ema_9'].iloc[row_idx]
            ema50 = df['ema_50'].iloc[row_idx]
            close = df['close'].iloc[row_idx]

            if adx > 25:
                if close > ema9 > ema50:
                    return "uptrend"
                elif close < ema9 < ema50:
                    return "downtrend"

            if adx < 20:
                return "ranging"

        return "neutral"

    def _calculate_momentum(self, df, row_idx):
        if 'price_momentum' in df.columns:
            return df['price_momentum'].iloc[row_idx]

        if len(df) < row_idx + 1 or row_idx < 10:
            return 0

        if 'ema_9' in df.columns:
            ema_vals = df['ema_9'].values[row_idx - 10:row_idx + 1]
            if len(ema_vals) >= 5:
                return (ema_vals[-1] / ema_vals[-5] - 1)

        return (df['close'].iloc[row_idx] / df['close'].iloc[max(0, row_idx - 5)] - 1)

    def _get_current_price(self, df, row_idx, price_type='close'):
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

    def _get_indicator_value(self, df, row_idx, indicator, fallback=None, default=0.0):
        if indicator in df.columns:
            try:
                value = float(df[indicator].iloc[row_idx])
                if not np.isnan(value):
                    return value
            except (IndexError, ValueError, TypeError):
                pass

        if fallback and fallback in df.columns:
            try:
                value = float(df[fallback].iloc[row_idx])
                if not np.isnan(value):
                    return value
            except (IndexError, ValueError, TypeError):
                pass

        return default

    def _get_macd_values(self, df, row_idx):
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
                except (IndexError, ValueError, TypeError):
                    pass

        return result

    def _compute_atr(self, df, row_idx, current_price):
        for col in ['atr_14', 'm30_atr_14']:
            if col in df.columns:
                try:
                    atr_val = float(df[col].iloc[row_idx])
                    if not np.isnan(atr_val) and atr_val > 0:
                        return atr_val
                except (IndexError, ValueError, TypeError):
                    pass

        return current_price * 0.01


class PortfolioManager:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.get("risk", "initial_capital", 10000.0)
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.current_drawdown_start = None
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.risk_manager = None
        self.recent_trades = deque(maxlen=100)

    def set_risk_manager(self, risk_manager):
        self.risk_manager = risk_manager

    def reset(self):
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.current_drawdown_start = None
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.recent_trades.clear()

    def update_after_trade(self, pnl):
        self.current_capital += pnl

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            self.current_drawdown = 0.0
            self.current_drawdown_start = None
        else:
            self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        if pnl > 0:
            self.current_win_streak += 1
            self.current_loss_streak = 0
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
        else:
            self.current_loss_streak += 1
            self.current_win_streak = 0
            self.max_loss_streak = max(self.max_loss_streak, self.current_loss_streak)

        self.recent_trades.append({'pnl': pnl, 'is_win': pnl > 0})

    def get_risk_state(self):
        drawdown_risk_state = "normal"

        if self.current_drawdown > 0.15:
            drawdown_risk_state = "minimal"
        elif self.current_drawdown > 0.10:
            drawdown_risk_state = "reduced"
        elif self.current_drawdown > 0.05:
            drawdown_risk_state = "caution"

        recovery_mode = self.current_drawdown > 0.12

        return {
            "drawdown_state": drawdown_risk_state,
            "recovery_mode": recovery_mode,
            "win_streak": self.current_win_streak,
            "loss_streak": self.current_loss_streak
        }

    def get_performance_metrics(self):
        wins = sum(1 for trade in self.recent_trades if trade['is_win'])
        total = len(self.recent_trades)
        win_rate = wins / total if total > 0 else 0

        return {
            "win_rate": win_rate,
            "recent_trades": total,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "current_drawdown": self.current_drawdown,
            "win_streak": self.current_win_streak,
            "loss_streak": self.current_loss_streak
        }


class WalkForwardManager:
    def __init__(self, config):
        self.config = config

    def create_windows(self, df, train_size, test_size, num_windows):
        df_len = len(df)
        min_required = train_size + test_size

        if df_len < min_required:
            return []

        step_size =test_size
        available_steps = (df_len - min_required) // step_size + 1
        max_iterations = min(num_windows, available_steps)

        windows = []
        for i in range(max_iterations):
            start_idx = i * step_size
            train_end = min(start_idx + train_size, df_len)
            test_end = min(train_end + test_size, df_len)

            if test_end <= train_end:
                break

            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            window_info = {
                "start_idx": start_idx,
                "train_start": df.index[start_idx],
                "train_end": df.index[train_end - 1],
                "test_start": df.index[train_end],
                "test_end": df.index[test_end - 1],
                "train_size": len(train_df),
                "test_size": len(test_df)
            }

            windows.append((train_df, test_df, window_info))

        return windows


class OptimizationEngine:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("OptimizationEngine")
        self.optimized_signal_params = {}
        self.optimized_risk_params = {}
        self.optimization_history = []

    def optimize_exit_strategies(self, trades):
        if not trades or len(trades) < 20:
            return

        exit_performance = {}

        for trade in trades:
            exit_reason = trade.get('exit_signal', 'Unknown')
            pnl = trade.get('pnl', 0)

            if exit_reason not in exit_performance:
                exit_performance[exit_reason] = {
                    'count': 0,
                    'win_count': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'win_rate': 0
                }

            stats = exit_performance[exit_reason]
            stats['count'] += 1

            if pnl > 0:
                stats['win_count'] += 1

            stats['total_pnl'] += pnl

            if stats['count'] > 0:
                stats['avg_pnl'] = stats['total_pnl'] / stats['count']
                stats['win_rate'] = stats['win_count'] / stats['count']

        best_exits = sorted(
            [reason for reason, stats in exit_performance.items()
             if stats['count'] >= 5 and stats['avg_pnl'] > 0],
            key=lambda r: exit_performance[r]['avg_pnl'],
            reverse=True
        )[:5]

        optimized_params = {}

        if "ProfitTargetReached" in best_exits or "ProfitTarget" in best_exits:
            optimized_params["profit_target_factor"] = 0.95
        else:
            optimized_params["profit_target_factor"] = 1.05

        if "TrailingStopExit" in best_exits or "TrailingStop" in best_exits:
            optimized_params["trailing_activation_factor"] = 0.9
        else:
            optimized_params["trailing_activation_factor"] = 1.1

        if "PartialExit" in [reason for reason in best_exits if "PartialExit" in reason]:
            optimized_params["enable_partial_exits"] = True
            optimized_params["partial_exit_threshold_factor"] = 0.9
        else:
            optimized_params["enable_partial_exits"] = False

        if "QuickProfitTaken" in best_exits or "QuickProfit" in best_exits:
            optimized_params["quick_profit_threshold_factor"] = 0.9
        else:
            optimized_params["quick_profit_threshold_factor"] = 1.1

        self.optimized_risk_params.update(optimized_params)

        self.optimization_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "exit_strategies",
            "best_exits": best_exits,
            "parameters": optimized_params,
            "exit_performance": exit_performance
        })

        return optimized_params

    def optimize_signal_thresholds(self, trades):
        if not trades or len(trades) < 20:
            return

        phase_performance = {}

        for trade in trades:
            market_phase = trade.get('market_phase', 'neutral')
            direction = trade.get('direction', 'unknown')
            pnl = trade.get('pnl', 0)

            if market_phase not in phase_performance:
                phase_performance[market_phase] = {
                    'long': {'count': 0, 'win_count': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl': 0},
                    'short': {'count': 0, 'win_count': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl': 0}
                }

            if direction not in ['long', 'short']:
                continue

            stats = phase_performance[market_phase][direction]
            stats['count'] += 1

            if pnl > 0:
                stats['win_count'] += 1

            stats['total_pnl'] += pnl

            if stats['count'] > 0:
                stats['win_rate'] = stats['win_count'] / stats['count']
                stats['avg_pnl'] = stats['total_pnl'] / stats['count']

        best_phase_directions = []

        for phase, directions in phase_performance.items():
            for direction, stats in directions.items():
                if stats['count'] >= 5:
                    best_phase_directions.append({
                        'phase': phase,
                        'direction': direction,
                        'win_rate': stats['win_rate'],
                        'avg_pnl': stats['avg_pnl'],
                        'score': stats['win_rate'] * 0.4 + (stats['avg_pnl'] * 100) * 0.6
                    })

        best_phase_directions.sort(key=lambda x: x['score'], reverse=True)

        threshold_adjustments = {}

        for item in best_phase_directions[:5]:
            phase = item['phase']
            direction = item['direction']

            if phase not in threshold_adjustments:
                threshold_adjustments[phase] = {}

            threshold_adjustments[phase][direction] = 0.85

        for item in best_phase_directions[-5:]:
            phase = item['phase']
            direction = item['direction']

            if phase not in threshold_adjustments:
                threshold_adjustments[phase] = {}

            if direction not in threshold_adjustments[phase]:
                threshold_adjustments[phase][direction] = 1.2

        optimized_params = {
            "threshold_adjustments": threshold_adjustments,
            "use_adaptive_thresholds": True
        }

        self.optimized_signal_params.update(optimized_params)

        self.optimization_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "signal_thresholds",
            "parameters": optimized_params,
            "phase_performance": phase_performance
        })

        return optimized_params

    def get_signal_parameters(self):
        return self.optimized_signal_params

    def get_risk_parameters(self):
        return self.optimized_risk_params

    def export_optimization_results(self, output_dir):
        if not self.optimization_history:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"optimization_history_{timestamp}.json"

        try:
            import json
            with open(filename, 'w') as f:
                json.dump(self.optimization_history, f, indent=2)

            self.logger.info(f"Exported optimization history to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting optimization history: {e}")


class PerformanceAnalyzer:
    def __init__(self, config):
        self.config = config

    def calculate_window_metrics(self, trades, equity_curve, window_info):
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_duration': 0,
                'return': 0
            }

        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) <= 0]

        win_rate = len(wins) / len(trades) if trades else 0

        profit_sum = sum(t.get('pnl', 0) for t in wins)
        loss_sum = abs(sum(t.get('pnl', 0) for t in losses))
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                daily_returns.append((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])
            else:
                daily_returns.append(0)

        if daily_returns:
            avg_daily_return = np.mean(daily_returns)
            daily_std = np.std(daily_returns) if len(daily_returns) > 1 else 1e-10
            sharpe_ratio = (avg_daily_return / daily_std) * np.sqrt(252) if daily_std > 0 else 0

            downside_returns = [r for r in daily_returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns and len(downside_returns) > 1 else 1e-10
            sortino_ratio = (avg_daily_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        max_drawdown = 0
        max_drawdown_duration = 0
        peak = equity_curve[0]
        dd_start = None

        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                if dd_start is not None:
                    dd_duration = i - dd_start
                    max_drawdown_duration = max(max_drawdown_duration, dd_duration)
                    dd_start = None
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, dd)

                if dd > 0 and dd_start is None:
                    dd_start = i

        if len(equity_curve) >= 2:
            total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        else:
            total_return = 0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'return': total_return,
            'trade_count': len(trades),
            'avg_trade': sum(t.get('pnl', 0) for t in trades) / len(trades) if trades else 0
        }