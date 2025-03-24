import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class Exporter:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Exporter")
        self.output_dir = Path(config.results_dir) / "backtest"
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def export_trade_details(self, consolidated_trades, final_equity,
                             metric_calculator, initial_capital):
        if not consolidated_trades:
            self.logger.warning("No trades to export")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        trade_records = []
        balance = initial_capital

        for tr in sorted(consolidated_trades, key=lambda x: x['exit_time']):
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
            metrics = metric_calculator.calculate_consolidated_metrics(consolidated_trades, final_equity)

            f.write("Trading Results Summary\n")
            f.write("======================\n\n")
            f.write(f"Total Trades: {len(df_trades)}\n")
            f.write(f"Initial Balance: ${initial_capital:.2f}\n")
            f.write(f"Final Balance: ${final_equity:.2f}\n")
            f.write(f"Total Profit/Loss: ${final_equity - initial_capital:.2f}\n")
            f.write(f"Return: {((final_equity / initial_capital) - 1) * 100:.2f}%\n\n")

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

    def export_feature_impact_analysis(self, consolidated_trades):
        if not consolidated_trades:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_path = self.output_dir / f'feature_impact_analysis_{timestamp}.txt'

        try:
            os.makedirs(os.path.dirname(analysis_path), exist_ok=True)

            with open(analysis_path, 'w') as f:
                f.write("Feature Impact Analysis\n")
                f.write("======================\n\n")

                f.write("Feature Values at Entry\n")
                f.write("---------------------\n")

                entry_stats = {}
                for trade in consolidated_trades:
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
                for trade in consolidated_trades:
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

                winning_trades = [t for t in consolidated_trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in consolidated_trades if t.get('pnl', 0) <= 0]

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
                        trades_with_reason = [t for t in consolidated_trades if t.get('exit_signal') == reason]
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

    def export_exit_strategy_analysis(self, backtest_engine):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_path = self.output_dir / f'exit_strategy_analysis_{timestamp}.txt'

        try:
            with open(analysis_path, 'w') as f:
                f.write("Enhanced Exit Strategy Analysis\n")
                f.write("==============================\n\n")

                f.write("Exit Strategy Performance\n")
                f.write("-----------------------\n")
                if hasattr(backtest_engine.metric_calculator, 'exit_performance'):
                    f.write(f"Total trades analyzed: {len(backtest_engine.consolidated_trades)}\n")

                    # Add safety checks for the attributes
                    total_partial_exits = getattr(backtest_engine, 'total_partial_exits', 0)
                    total_quick_profit_exits = getattr(backtest_engine, 'total_quick_profit_exits', 0)

                    f.write(f"Total partial exits executed: {total_partial_exits}\n")
                    f.write(f"Total quick profit exits executed: {total_quick_profit_exits}\n\n")

                    f.write("Exit Strategy Performance by Type:\n")
                    for reason, perf in sorted(backtest_engine.metric_calculator.exit_performance.items(),
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
                if hasattr(backtest_engine.metric_calculator,
                           'best_exit_reasons') and backtest_engine.metric_calculator.best_exit_reasons:
                    for i, reason in enumerate(backtest_engine.metric_calculator.best_exit_reasons[:5], 1):
                        if reason in backtest_engine.metric_calculator.exit_performance:
                            perf = backtest_engine.metric_calculator.exit_performance[reason]
                            f.write(
                                f"{i}. {reason}: ${perf['avg_pnl']:.2f} avg P&L, {perf['win_rate'] * 100:.1f}% win rate\n")
                else:
                    f.write("No best exit strategies identified.\n")
                f.write("\n")

                f.write("Market Phase Performance\n")
                f.write("-----------------------\n")
                market_phase_stats = getattr(backtest_engine, 'market_phase_stats', {})
                for phase, count in sorted(market_phase_stats.items(), key=lambda x: x[1],
                                           reverse=True):
                    f.write(f"{phase}: {count} occurrences\n")
                f.write("\n")

                f.write("Top Performing Market Phases\n")
                f.write("--------------------------\n")
                if hasattr(backtest_engine.metric_calculator,
                           'best_performing_phases') and backtest_engine.metric_calculator.best_performing_phases:
                    for i, phase in enumerate(backtest_engine.metric_calculator.best_performing_phases[:3], 1):
                        f.write(f"{i}. {phase}\n")
                else:
                    f.write("No best performing phases identified.\n")
                f.write("\n")

                f.write("Trading Timing Analysis\n")
                f.write("----------------------\n")
                avg_trade_holding_time = getattr(backtest_engine, 'avg_trade_holding_time', 0)
                f.write(f"Average trade holding time: {avg_trade_holding_time:.2f} hours\n")

                f.write("\nExit Strategy Recommendations\n")
                f.write("---------------------------\n")

                if hasattr(backtest_engine.metric_calculator,
                           'exit_performance') and backtest_engine.metric_calculator.exit_performance:
                    best_strategies = sorted(
                        [(k, v) for k, v in backtest_engine.metric_calculator.exit_performance.items() if
                         v['count'] >= 5],
                        key=lambda x: x[1]['avg_pnl'],
                        reverse=True
                    )[:3]

                    worst_strategies = sorted(
                        [(k, v) for k, v in backtest_engine.metric_calculator.exit_performance.items() if
                         v['count'] >= 5],
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
                    if avg_trade_holding_time < 5:
                        f.write("   - Consider longer holding periods for winning trades\n")
                    elif avg_trade_holding_time > 24:
                        f.write("   - Consider taking profits earlier on winning trades\n")

                    if hasattr(backtest_engine.metric_calculator,
                               'best_performing_phases') and backtest_engine.metric_calculator.best_performing_phases:
                        f.write("\n4. Market phase strategy recommendations:\n")
                        for phase in backtest_engine.metric_calculator.best_performing_phases[:2]:
                            f.write(f"   - Increase position size during {phase} phase\n")

                    partial_exits = [k for k in backtest_engine.metric_calculator.exit_performance.keys() if
                                     "PartialExit" in k]
                    if partial_exits:
                        best_partial = max(partial_exits,
                                           key=lambda x: backtest_engine.metric_calculator.exit_performance[x][
                                               'avg_pnl'] if x in backtest_engine.metric_calculator.exit_performance else 0)
                        f.write(f"\n5. Partial exit optimization:\n")
                        f.write(f"   - Prioritize {best_partial} partial exit strategy\n")

                    f.write("\n6. Advanced exit optimization strategy:\n")
                    f.write("   - Implement dynamic partial exits based on market volatility\n")
                    f.write("   - Use trailing stops that adapt to price momentum\n")
                    f.write("   - Consider market regime when setting profit targets\n")

                else:
                    f.write("Insufficient data for exit strategy recommendations.\n")

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

    def export_time_analysis(self, time_stats: Dict[str, Any]):
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
            if hasattr(self, 'consolidated_trades') and len(self.consolidated_trades) >= 20:
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

    def export_drawdown_analysis(self, drawdown_periods, consolidated_trades, current_capital, peak_capital):
        if not drawdown_periods and not consolidated_trades:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        drawdown_path = self.output_dir / f'drawdown_analysis_{timestamp}.txt'

        with open(drawdown_path, 'w') as f:
            f.write("Drawdown Analysis\n")
            f.write("=================\n\n")

            f.write("Overall Drawdown Statistics\n")
            f.write("---------------------------\n")
            max_drawdown = max(period['depth'] for period in drawdown_periods) if drawdown_periods else 0
            max_drawdown_duration = max(period['duration'] for period in drawdown_periods) if drawdown_periods else 0

            f.write(f"Maximum Drawdown: {max_drawdown * 100:.2f}%\n")
            f.write(f"Maximum Drawdown Duration: {max_drawdown_duration} iterations\n\n")

            # Analyze recovery periods
            if consolidated_trades and len(consolidated_trades) > 10:
                f.write("Recovery Analysis\n")
                f.write("-----------------\n")

                # Find significant drawdowns and subsequent recovery
                equity_curve = [self.config.get("risk", "initial_capital")]
                peak = self.config.get("risk", "initial_capital")
                drawdown_start = None
                drawdowns = []

                # Calculate equity curve
                for t in sorted(consolidated_trades, key=lambda x: x['exit_time']):
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

                    for i, t in enumerate(sorted(consolidated_trades, key=lambda x: x['exit_time'])):
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

    def export_monthly_performance(self, monthly_returns):
        if not monthly_returns:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        monthly_path = self.output_dir / f'monthly_performance_{timestamp}.txt'

        with open(monthly_path, 'w') as f:
            f.write("Monthly Performance Analysis\n")
            f.write("===========================\n\n")

            # Calculate monthly statistics
            monthly_stats = {}
            for month, returns in monthly_returns.items():
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
                if hasattr(self, 'consolidated_trades') and len(self.consolidated_trades) >= 20:
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