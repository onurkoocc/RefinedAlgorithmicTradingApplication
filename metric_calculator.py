import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple


class MetricCalculator:
    def __init__(self, config):
        self.config = config

    def calculate_consolidated_metrics(self, trades_list: List[Dict[str, Any]], final_equity: float,
                                       initial_capital: float) -> Dict[str, Any]:
        if not trades_list:
            return self._empty_metrics(initial_capital)

        df_trades = pd.DataFrame(trades_list)
        if df_trades.empty:
            return self._empty_metrics(initial_capital)

        if 'pnl' not in df_trades.columns or 'exit_time' not in df_trades.columns:
            self.logger.error("Missing 'pnl' or 'exit_time' in trade data for metrics.")
            return self._empty_metrics(initial_capital)
        df_trades['pnl'] = pd.to_numeric(df_trades['pnl'], errors='coerce').fillna(0)
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'], errors='raise')
        df_trades.sort_values(by='exit_time', inplace=True)

        total_return_pct = ((final_equity / initial_capital) - 1) * 100 if initial_capital > 0 else 0
        num_trades = len(df_trades)

        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]

        win_rate = len(wins) / num_trades if num_trades > 0 else 0

        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 1.0

        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0
        reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf if avg_win > 0 else 1.0

        equity_curve = [initial_capital] + (initial_capital + df_trades['pnl'].cumsum()).tolist()
        peak_equity = np.maximum.accumulate(equity_curve)
        drawdowns = (peak_equity - equity_curve) / peak_equity
        max_drawdown_pct = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        in_dd = False
        current_dd_length = 0
        max_dd_length_trades = 0
        for dd_val in drawdowns:
            if dd_val > 0.001:
                if not in_dd:
                    in_dd = True
                    current_dd_length = 1
                else:
                    current_dd_length += 1
            else:
                if in_dd:
                    max_dd_length_trades = max(max_dd_length_trades, current_dd_length)
                    in_dd = False
                    current_dd_length = 0
        if in_dd: max_dd_length_trades = max(max_dd_length_trades, current_dd_length)

        avg_trade_return_pct = df_trades['pnl'].mean() / initial_capital if initial_capital > 0 else 0
        std_trade_return_pct = df_trades['pnl'].std() / initial_capital if initial_capital > 0 and num_trades > 1 else 0

        trading_days = (df_trades['exit_time'].max() - df_trades['exit_time'].min()).days if num_trades > 1 else 1
        trades_per_year_approx = (num_trades / trading_days) * 252 if trading_days > 0 else num_trades * 252

        sharpe_ratio = (avg_trade_return_pct / (std_trade_return_pct + 1e-9)) * np.sqrt(
            trades_per_year_approx) if std_trade_return_pct > 0 else 0

        negative_returns_pct = df_trades[df_trades['pnl'] < 0][
                                   'pnl'] / initial_capital if initial_capital > 0 else pd.Series(dtype=float)
        downside_std_dev = negative_returns_pct.std() if not negative_returns_pct.empty and len(
            negative_returns_pct) > 1 else std_trade_return_pct
        sortino_ratio = (avg_trade_return_pct / (downside_std_dev + 1e-9)) * np.sqrt(
            trades_per_year_approx) if downside_std_dev > 0 else 0

        max_win_s = 0
        current_win_s = 0
        max_loss_s = 0
        current_loss_s = 0
        for pnl_val in df_trades['pnl']:
            if pnl_val > 0:
                current_win_s += 1
                current_loss_s = 0
            else:
                current_loss_s += 1
                current_win_s = 0
            max_win_s = max(max_win_s, current_win_s)
            max_loss_s = max(max_loss_s, current_loss_s)

        monthly_returns_agg = {}
        if 'exit_time' in df_trades.columns:
            df_trades['month_year'] = df_trades['exit_time'].dt.to_period('M')
            monthly_pnl = df_trades.groupby('month_year')['pnl'].sum()

            current_balance_month_start = initial_capital
            for period, pnl_sum in monthly_pnl.items():
                monthly_return_val = pnl_sum / current_balance_month_start if current_balance_month_start > 0 else 0
                monthly_returns_agg[str(period)] = monthly_return_val
                current_balance_month_start += pnl_sum

        avg_monthly_growth_achieved = np.mean(list(monthly_returns_agg.values())) if monthly_returns_agg else 0
        target_monthly_min = self.config.get("model", "growth_metric", {}).get("min_target", 0.07)
        target_monthly_max = self.config.get("model", "growth_metric", {}).get("max_target", 0.13)

        growth_target_met_consistency = 0
        if monthly_returns_agg:
            met_months = sum(1 for r in monthly_returns_agg.values() if target_monthly_min <= r <= target_monthly_max)
            growth_target_met_consistency = met_months / len(monthly_returns_agg)

        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win_usd': avg_win,
            'avg_loss_usd': avg_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'max_drawdown_pct': max_drawdown_pct * 100,
            'max_drawdown_duration_trades': max_dd_length_trades,
            'sharpe_ratio_approx': sharpe_ratio,
            'sortino_ratio_approx': sortino_ratio,
            'max_win_streak': max_win_s,
            'max_loss_streak': max_loss_s,
            'avg_monthly_growth_pct': avg_monthly_growth_achieved * 100,
            'growth_target_consistency_pct': growth_target_met_consistency * 100,
            'monthly_returns_detailed': monthly_returns_agg
        }

    def _empty_metrics(self, initial_capital: float) -> Dict[str, Any]:
        return {
            'initial_capital': initial_capital, 'final_equity': initial_capital,
            'total_return_pct': 0.0, 'num_trades': 0, 'win_rate_pct': 0.0, 'profit_factor': 0.0,
            'avg_win_usd': 0.0, 'avg_loss_usd': 0.0, 'reward_risk_ratio': 0.0,
            'max_drawdown_pct': 0.0, 'max_drawdown_duration_trades': 0,
            'sharpe_ratio_approx': 0.0, 'sortino_ratio_approx': 0.0,
            'max_win_streak': 0, 'max_loss_streak': 0,
            'avg_monthly_growth_pct': 0.0, 'growth_target_consistency_pct': 0.0,
            'monthly_returns_detailed': {}
        }