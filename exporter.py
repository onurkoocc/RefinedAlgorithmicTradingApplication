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

    def export_trade_details(self, consolidated_trades: List[Dict[str, Any]], final_equity: float,
                             metric_calculator, initial_capital: float):
        if not consolidated_trades:
            self.logger.warning("No trades to export.")
            return

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        df_trades = pd.DataFrame(consolidated_trades)
        if df_trades.empty:
            self.logger.warning("Trade list resulted in an empty DataFrame.")
            return

        essential_cols = ['entry_time', 'exit_time', 'direction', 'entry_price_actual', 'exit_price_slipped',
                          'quantity', 'pnl', 'entry_signal_type', 'exit_reason', 'market_phase_at_entry',
                          'ensemble_score']
        for col in essential_cols:
            if col not in df_trades.columns:
                df_trades[col] = None

        if 'duration_hours' not in df_trades.columns and 'entry_time' in df_trades.columns and 'exit_time' in df_trades.columns:
            df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
            df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
            df_trades['duration_hours'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 3600
        elif 'duration_hours' not in df_trades.columns:
            df_trades['duration_hours'] = 0

        df_trades_csv = df_trades.copy()
        if 'entry_time' in df_trades_csv.columns:
            df_trades_csv['entry_time'] = pd.to_datetime(df_trades_csv['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'exit_time' in df_trades_csv.columns:
            df_trades_csv['exit_time'] = pd.to_datetime(df_trades_csv['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

        csv_cols = ['id', 'entry_time', 'exit_time', 'direction', 'entry_price_actual', 'exit_price_slipped',
                    'quantity', 'pnl', 'duration_hours', 'entry_signal_type', 'exit_reason',
                    'market_phase_at_entry', 'ensemble_score', 'is_partial', 'partial_id']
        df_trades_csv = df_trades_csv[[col for col in csv_cols if col in df_trades_csv.columns]]

        csv_path = self.output_dir / f'trade_details_{timestamp_str}.csv'
        df_trades_csv.to_csv(csv_path, index=False, float_format='%.6f')
        self.logger.info(f"Exported {len(df_trades_csv)} trade details to {csv_path}")

        summary_path = self.output_dir / f'trade_summary_{timestamp_str}.txt'
        metrics = metric_calculator.calculate_consolidated_metrics(consolidated_trades, final_equity, initial_capital)

        with open(summary_path, 'w') as f:
            f.write(f"Trading Performance Summary - {timestamp_str}\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Initial Capital: ${metrics.get('initial_capital', 0):,.2f}\n")
            f.write(f"Final Equity:    ${metrics.get('final_equity', 0):,.2f}\n")
            total_pnl_usd = metrics.get('final_equity', 0) - metrics.get('initial_capital', 0)
            f.write(f"Total PnL (USD): ${total_pnl_usd:,.2f}\n")
            f.write(f"Total Return:    {metrics.get('total_return_pct', 0):.2f}%\n\n")

            f.write("Key Performance Indicators:\n")
            f.write(f"  Number of Trades: {metrics.get('num_trades', 0)}\n")
            f.write(f"  Win Rate:         {metrics.get('win_rate_pct', 0):.2f}%\n")
            f.write(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}\n")
            f.write(f"  Avg Win (USD):    ${metrics.get('avg_win_usd', 0):,.2f}\n")
            f.write(f"  Avg Loss (USD):   ${metrics.get('avg_loss_usd', 0):,.2f}\n")
            f.write(f"  Reward/Risk Ratio:{metrics.get('reward_risk_ratio', 0):.2f}\n")
            f.write(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%\n")
            f.write(f"  Max DD Duration (Trades): {metrics.get('max_drawdown_duration_trades', 0)}\n")
            f.write(f"  Sharpe Ratio (Approx): {metrics.get('sharpe_ratio_approx', 0):.2f}\n")
            f.write(f"  Sortino Ratio (Approx):{metrics.get('sortino_ratio_approx', 0):.2f}\n")
            f.write(f"  Max Win Streak:   {metrics.get('max_win_streak', 0)}\n")
            f.write(f"  Max Loss Streak:  {metrics.get('max_loss_streak', 0)}\n\n")

            avg_duration = df_trades['duration_hours'].mean() if 'duration_hours' in df_trades and not df_trades[
                'duration_hours'].empty else 0
            f.write(f"Average Trade Duration: {avg_duration:.2f} hours\n\n")

            f.write("Monthly Performance:\n")
            f.write(f"  Avg Monthly Growth: {metrics.get('avg_monthly_growth_pct', 0):.2f}%\n")
            f.write(f"  Growth Target Consistency: {metrics.get('growth_target_consistency_pct', 0):.2f}%\n")
            monthly_details = metrics.get('monthly_returns_detailed', {})
            if monthly_details:
                f.write("  Monthly Returns Breakdown:\n")
                for month_year, ret_pct in sorted(monthly_details.items()):
                    f.write(f"    {month_year}: {ret_pct * 100:.2f}%\n")
            f.write("\n")

            if 'exit_reason' in df_trades.columns:
                f.write("Exit Reason Analysis:\n")
                exit_summary = df_trades.groupby('exit_reason')['pnl'].agg(['count', 'sum', 'mean'])
                exit_summary.columns = ['Trade Count', 'Total PnL', 'Avg PnL']
                exit_summary = exit_summary.sort_values(by='Trade Count', ascending=False)
                f.write(exit_summary.to_string(float_format="%.2f"))
                f.write("\n\n")

            if 'market_phase_at_entry' in df_trades.columns:
                f.write("Market Phase at Entry Analysis:\n")
                phase_summary = df_trades.groupby('market_phase_at_entry')['pnl'].agg(['count', 'sum', 'mean'])
                phase_summary.columns = ['Trade Count', 'Total PnL', 'Avg PnL']
                phase_summary = phase_summary.sort_values(by='Trade Count', ascending=False)
                f.write(phase_summary.to_string(float_format="%.2f"))
                f.write("\n\n")

        self.logger.info(f"Exported trade summary to {summary_path}")

    def export_time_analysis(self, trades_list: List[Dict[str, Any]]):
        if not trades_list: return
        df_trades = pd.DataFrame(trades_list)
        if df_trades.empty or 'duration_hours' not in df_trades.columns: return

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        time_path = self.output_dir / f'time_analysis_{timestamp_str}.txt'

        with open(time_path, 'w') as f:
            f.write("Trade Duration Analysis\n")
            f.write("=" * 30 + "\n\n")

            f.write(f"Overall Avg Duration: {df_trades['duration_hours'].mean():.2f} hours\n")
            f.write(f"Median Duration:      {df_trades['duration_hours'].median():.2f} hours\n")
            f.write(f"Min Duration:         {df_trades['duration_hours'].min():.2f} hours\n")
            f.write(f"Max Duration:         {df_trades['duration_hours'].max():.2f} hours\n\n")

            f.write("Performance by Duration Buckets:\n")
            bins = [0, 1, 2, 4, 8, 16, 24, 48, np.inf]
            labels = ["<1h", "1-2h", "2-4h", "4-8h", "8-16h", "16-24h", "24-48h", ">48h"]
            df_trades['duration_bucket'] = pd.cut(df_trades['duration_hours'], bins=bins, labels=labels, right=False)

            duration_summary = df_trades.groupby('duration_bucket')['pnl'].agg(['count', 'sum', 'mean'])
            duration_summary.columns = ['Trade Count', 'Total PnL', 'Avg PnL']
            f.write(duration_summary.to_string(float_format="%.2f"))
            f.write("\n\n")

        self.logger.info(f"Exported time analysis to {time_path}")