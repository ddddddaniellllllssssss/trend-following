import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime

class PerformanceMetrics:
    """Calculate comprehensive performance metrics for backtesting results"""

    @staticmethod
    def calculate_all_metrics(results: Dict) -> Dict:
        """
        Calculate all performance metrics from backtest results

        Args:
            results: Dictionary containing backtest results

        Returns:
            Dictionary with comprehensive performance metrics
        """
        trades = results.get('trades', [])
        equity_curve = results.get('equity_curve', pd.DataFrame())

        if not trades:
            return PerformanceMetrics._empty_metrics()

        trades_df = pd.DataFrame(trades)

        # Basic metrics
        basic_metrics = PerformanceMetrics._calculate_basic_metrics(trades_df, results)

        # Risk metrics
        risk_metrics = PerformanceMetrics._calculate_risk_metrics(equity_curve, trades_df)

        # Trade analysis
        trade_metrics = PerformanceMetrics._calculate_trade_metrics(trades_df)

        # Time-based metrics
        time_metrics = PerformanceMetrics._calculate_time_metrics(trades_df, equity_curve)

        # Combine all metrics
        all_metrics = {
            **basic_metrics,
            **risk_metrics,
            **trade_metrics,
            **time_metrics
        }

        return all_metrics

    @staticmethod
    def _empty_metrics() -> Dict:
        """Return empty metrics when no trades exist"""
        return {
            'total_return': 0,
            'total_pnl': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_rr_ratio': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_trade_duration': 0,
            'total_days': 0
        }

    @staticmethod
    def _calculate_basic_metrics(trades_df: pd.DataFrame, results: Dict) -> Dict:
        """Calculate basic performance metrics"""
        total_pnl = trades_df['net_pnl'].sum()
        initial_capital = 100000  # From config
        total_return = total_pnl / initial_capital

        return {
            'total_return': total_return,
            'total_pnl': total_pnl,
            'num_trades': len(trades_df),
            'avg_rr_ratio': trades_df['rr_ratio'].mean() if len(trades_df) > 0 else 0
        }

    @staticmethod
    def _calculate_risk_metrics(equity_curve: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """Calculate risk-adjusted metrics"""
        if equity_curve.empty:
            return {'max_drawdown': 0, 'sharpe_ratio': 0}

        # Maximum drawdown
        equity_values = equity_curve['portfolio_value'].values
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = (equity_values - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))

        # Sharpe ratio (simplified - using trade returns)
        if len(trades_df) > 1:
            trade_returns = trades_df['net_pnl'] / 100000  # Normalize by initial capital
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252) if np.std(trade_returns) > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

    @staticmethod
    def _calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict:
        """Calculate trade-specific metrics"""
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] < 0]

        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

        # Profit factor
        gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Average wins and losses
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0

        # Largest wins and losses
        largest_win = winning_trades['net_pnl'].max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades['net_pnl'].min() if len(losing_trades) > 0 else 0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades)
        }

    @staticmethod
    def _calculate_time_metrics(trades_df: pd.DataFrame, equity_curve: pd.DataFrame) -> Dict:
        """Calculate time-based metrics"""
        # Average trade duration
        avg_trade_duration = trades_df['duration_days'].mean() if len(trades_df) > 0 else 0

        # Total backtest period
        if not equity_curve.empty:
            start_date = equity_curve['date'].iloc[0]
            end_date = equity_curve['date'].iloc[-1]
            total_days = (end_date - start_date).days
        else:
            total_days = 0

        return {
            'avg_trade_duration': avg_trade_duration,
            'total_days': total_days
        }

    @staticmethod
    def print_performance_report(metrics: Dict):
        """Print a formatted performance report"""
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)

        print(f"\nOverall Performance:")
        print(f"  Total Return:        {metrics['total_return']:.2%}")
        print(f"  Total P&L:           ${metrics['total_pnl']:,.2f}")
        print(f"  Number of Trades:    {metrics['num_trades']}")

        print(f"\nTrade Analysis:")
        print(f"  Win Rate:           {metrics['win_rate']:.2%}")
        print(f"  Average R:R Ratio:  {metrics['avg_rr_ratio']:.2f}")
        print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")

        print(f"\nWin/Loss Breakdown:")
        print(f"  Winning Trades:     {metrics.get('num_winning_trades', 0)}")
        print(f"  Losing Trades:      {metrics.get('num_losing_trades', 0)}")
        print(f"  Average Win:        ${metrics['avg_win']:,.2f}")
        print(f"  Average Loss:       ${metrics['avg_loss']:,.2f}")
        print(f"  Largest Win:        ${metrics['largest_win']:,.2f}")
        print(f"  Largest Loss:       ${metrics['largest_loss']:,.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Maximum Drawdown:   {metrics['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")

        print(f"\nTime Analysis:")
        print(f"  Avg Trade Duration: {metrics['avg_trade_duration']:.1f} days")
        print(f"  Total Backtest:     {metrics['total_days']} days")

        print("="*50)