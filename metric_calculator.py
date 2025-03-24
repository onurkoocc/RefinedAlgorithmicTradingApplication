import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple


class MetricCalculator:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.get("risk", "initial_capital", 10000.0)
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0
        self.max_drawdown_duration = 0
        self.current_drawdown_start = None
        self.equity_curve_points = []
        self.equity_curve_timestamps = []
        self.daily_returns = []
        self.monthly_returns = {}
        self.drawdown_periods = []
        self.exit_performance = {}
        self.best_exit_reasons = []
        self.best_performing_phases = []

    def update_drawdown_stats(self, current_equity):
        self.peak_capital = max(self.peak_capital, current_equity)

        current_drawdown = 0 if self.peak_capital == 0 else (self.peak_capital - current_equity) / self.peak_capital

        if current_drawdown > 0:
            if self.current_drawdown_start is None:
                self.current_drawdown_start = len(self.equity_curve_points) - 1

            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

            current_dd_duration = len(self.equity_curve_points) - 1 - self.current_drawdown_start
            self.max_drawdown_duration = max(self.max_drawdown_duration, current_dd_duration)

            if current_drawdown > 0.05:
                self.drawdown_periods.append({
                    'start_idx': self.current_drawdown_start,
                    'current_idx': len(self.equity_curve_points) - 1,
                    'depth': current_drawdown,
                    'duration': current_dd_duration
                })
        elif self.current_drawdown_start is not None:
            self.current_drawdown_start = None

    def update_monthly_returns(self, daily_returns, df_features, start_idx):
        if not daily_returns:
            return

        end_idx = min(start_idx + self.config.get("backtest", "train_window_size") +
                      self.config.get("backtest", "test_window_size"), len(df_features))
        date_slice = df_features.index[start_idx:end_idx]

        dates = date_slice[-len(daily_returns):]

        for i, ret in enumerate(daily_returns):
            if i < len(dates):
                date = dates[i]
                month_key = date.strftime('%Y-%m')

                if month_key not in self.monthly_returns:
                    self.monthly_returns[month_key] = []

                self.monthly_returns[month_key].append(ret)

    def track_exit_performance(self, exit_reason, pnl):
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

    def calculate_performance_metrics(self, trades, equity_curve, final_equity, drawdown_periods=None):
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

        if len(daily_returns) > 1:
            down_returns = [r for r in daily_returns if r < 0]
            if down_returns:
                downside_dev = max(np.std(down_returns), 1e-10)
                sortino = (avg_ret / downside_dev) * np.sqrt(252)
            else:
                sortino = sharpe * 1.5
        else:
            sortino = 0

        max_dd = 0
        max_dd_duration = 0
        peak = equity_curve[0]
        current_dd_start = None

        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                if current_dd_start is not None:
                    dd_duration = i - current_dd_start
                    max_dd_duration = max(max_dd_duration, dd_duration)
                    current_dd_start = None
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                if dd > 0 and current_dd_start is None:
                    current_dd_start = i
                max_dd = max(max_dd, dd)

        if current_dd_start is not None:
            dd_duration = len(equity_curve) - current_dd_start
            max_dd_duration = max(max_dd_duration, dd_duration)

        if drawdown_periods:
            for period in drawdown_periods:
                max_dd = max(max_dd, period['depth'])
                max_dd_duration = max(max_dd_duration, period['duration'])

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

    def calculate_consolidated_metrics(self, consolidated_trades, final_equity):
        if not consolidated_trades:
            return {}

        eq_curve = [self.initial_capital]
        balance = self.initial_capital

        peak_capital = balance
        current_drawdown_start = None
        max_drawdown = 0
        max_drawdown_duration = 0
        last_date = None

        daily_returns = []
        monthly_returns = {}

        for tr in sorted(consolidated_trades, key=lambda x: x['exit_time']):
            pnl = tr.get('pnl', 0)
            balance += pnl
            eq_curve.append(balance)

            exit_date = tr['exit_time'].date()
            if last_date is not None and exit_date != last_date:
                if len(eq_curve) >= 2:
                    daily_return = (eq_curve[-2] / eq_curve[-3]) - 1 if eq_curve[-3] > 0 else 0
                    daily_returns.append(daily_return)

                    month_key = exit_date.strftime('%Y-%m')
                    if month_key not in monthly_returns:
                        monthly_returns[month_key] = []
                    monthly_returns[month_key].append(daily_return)
            last_date = exit_date

            if balance > peak_capital:
                peak_capital = balance
                if current_drawdown_start is not None:
                    drawdown_duration = len(eq_curve) - 1 - current_drawdown_start
                    max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
                    current_drawdown_start = None
            elif balance < peak_capital:
                current_drawdown = (peak_capital - balance) / peak_capital
                max_drawdown = max(max_drawdown, current_drawdown)

                if current_drawdown_start is None:
                    current_drawdown_start = len(eq_curve) - 1

        ret_pct = ((final_equity / self.initial_capital) - 1) * 100

        wins = [t for t in consolidated_trades if t.get('pnl', 0) > 0]
        total_tr = len(consolidated_trades)
        win_rate = len(wins) / total_tr if total_tr else 0

        losses = [t for t in consolidated_trades if t.get('pnl', 0) <= 0]
        p_sum = sum(t.get('pnl', 0) for t in wins)
        n_sum = abs(sum(t.get('pnl', 0) for t in losses))
        pf = p_sum / max(n_sum, 1e-10)

        if daily_returns:
            avg_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns) if len(daily_returns) > 1 else 1e-10
            sharpe = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0

            down_returns = [r for r in daily_returns if r < 0]
            if down_returns:
                downside_dev = np.std(down_returns)
                sortino = (avg_daily_return / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0
            else:
                sortino = sharpe * 1.5
        else:
            sharpe = 0
            sortino = 0

        best_trade = max(consolidated_trades, key=lambda x: x.get('pnl', 0)) if consolidated_trades else {}
        worst_trade = min(consolidated_trades, key=lambda x: x.get('pnl', 0)) if consolidated_trades else {}

        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for tr in sorted(consolidated_trades, key=lambda x: x['exit_time']):
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