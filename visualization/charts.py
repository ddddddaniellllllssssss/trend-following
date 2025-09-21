import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from matplotlib.patches import Rectangle

class TradingCharts:
    """Create charts for trading strategy visualization"""

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('default')

    def _plot_candlesticks(self, ax, data: pd.DataFrame, width_factor: float = 0.6):
        """
        Plot candlesticks on the given axes

        Args:
            ax: matplotlib axes object
            data: DataFrame with OHLC data
            width_factor: Width of candlesticks relative to time spacing
        """
        # Calculate width for candlesticks
        if len(data) > 1:
            time_diff = (data.index[1] - data.index[0]).total_seconds() / (24 * 3600)  # days
            width = time_diff * width_factor
        else:
            width = 1 * width_factor

        for i, (date, row) in enumerate(data.iterrows()):
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']

            # Determine color (green for bullish, red for bearish)
            color = 'green' if close_price >= open_price else 'red'

            # Plot the high-low line (wick)
            ax.plot([date, date], [low_price, high_price], color='black', linewidth=1, alpha=0.8)

            # Plot the open-close rectangle (body)
            body_low = min(open_price, close_price)
            body_high = max(open_price, close_price)
            body_height = body_high - body_low

            if body_height > 0:  # Avoid zero height rectangles
                rect = Rectangle((date - pd.Timedelta(hours=width*12), body_low),
                               pd.Timedelta(hours=width*24), body_height,
                               facecolor=color, edgecolor='black', alpha=0.8, linewidth=0.5)
                ax.add_patch(rect)
            else:
                # Doji - just a horizontal line
                ax.plot([date - pd.Timedelta(hours=width*12), date + pd.Timedelta(hours=width*12)],
                       [open_price, close_price], color='black', linewidth=2)

    def plot_price_with_ranges(self, data: pd.DataFrame, ranges: List, trades: List,
                             title: str = "Price Chart with Trading Ranges") -> plt.Figure:
        """
        Plot price chart with trading ranges and entry/exit points

        Args:
            data: DataFrame with OHLC data
            ranges: List of trading ranges
            trades: List of executed trades
            title: Chart title

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot candlesticks
        self._plot_candlesticks(ax, data)

        # Add candlestick legend entry (invisible line just for legend)
        ax.plot([], [], color='green', linewidth=3, label='Candlesticks (Green=Bullish, Red=Bearish)')

        # Plot trading ranges
        self._plot_ranges(ax, data, ranges)

        # Plot trades
        self._plot_trades(ax, data, trades)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_ranges(self, ax, data, ranges):
        """Plot trading ranges on the chart"""
        for i, trading_range in enumerate(ranges):
            if not trading_range.is_valid:
                continue

            # Get start and end dates
            start_date = data.index[trading_range.start_index]
            end_date = data.index[trading_range.end_index] if trading_range.end_index < len(data) else data.index[-1]

            # Determine color based on direction
            color = 'green' if trading_range.direction == 'bullish' else 'red'
            alpha = 0.2

            # Plot range rectangle
            ax.axhspan(trading_range.low_price, trading_range.high_price,
                      xmin=(start_date - data.index[0]).days / (data.index[-1] - data.index[0]).days,
                      xmax=(end_date - data.index[0]).days / (data.index[-1] - data.index[0]).days,
                      alpha=alpha, color=color, label=f'{trading_range.direction.title()} Range' if i == 0 else "")

            # Plot order block if exists
            if trading_range.order_block:
                ob = trading_range.order_block
                ob_date = data.index[ob['index']]

                ax.axhspan(ob['low'], ob['high'],
                          xmin=(ob_date - data.index[0]).days / (data.index[-1] - data.index[0]).days,
                          xmax=(ob_date - data.index[0]).days / (data.index[-1] - data.index[0]).days + 0.01,
                          alpha=0.6, color='orange', label='Order Block' if i == 0 else "")

    def _plot_trades(self, ax, data, trades):
        """Plot trade entry and exit points including partial exits"""
        # Group trades by entry point to handle partial exits
        trade_groups = {}
        for trade in trades:
            key = (trade['entry_date'], trade['entry_price'])
            if key not in trade_groups:
                trade_groups[key] = []
            trade_groups[key].append(trade)

        for (entry_date, entry_price), trade_list in trade_groups.items():
            # Determine trade type from first trade
            trade_type = trade_list[0]['type']

            # Entry point (plot once per group)
            color = 'green' if trade_type == 'long' else 'red'
            ax.scatter(entry_date, entry_price, color=color,
                      marker='^' if trade_type == 'long' else 'v',
                      s=100, alpha=0.8, zorder=5,
                      label=f"{trade_type.upper()} Entry" if len([t for g in trade_groups.values() for t in g if t['type'] == trade_type]) == len(trade_list) else "")

            # Plot each exit
            for i, trade in enumerate(trade_list):
                exit_date = trade['exit_date']
                exit_price = trade['exit_price']

                # Exit point
                exit_color = 'green' if trade['net_pnl'] > 0 else 'red'

                # Different markers for different exit types
                if trade.get('is_partial', False):
                    marker = 'o'  # Circle for partial exits
                    label_text = 'Partial Exit (2:1 R:R)'
                elif trade.get('is_remaining', False):
                    marker = 's'  # Square for remaining position exits
                    label_text = f"Remaining Exit ({trade['exit_reason']})"
                else:
                    marker = 'x'  # X for full exits
                    label_text = f"Exit ({trade['exit_reason']})"

                ax.scatter(exit_date, exit_price, color=exit_color, marker=marker,
                          s=100, alpha=0.8, zorder=5)

                # Connect entry and exit
                linestyle = '-' if trade.get('is_partial', False) else '--'
                ax.plot([entry_date, exit_date], [entry_price, exit_price],
                       color=exit_color, linestyle=linestyle, alpha=0.6)

    def plot_equity_curve(self, equity_curve: pd.DataFrame,
                         title: str = "Equity Curve") -> plt.Figure:
        """
        Plot portfolio equity curve

        Args:
            equity_curve: DataFrame with portfolio values over time
            title: Chart title

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])

        # Equity curve
        ax1.plot(equity_curve['date'], equity_curve['portfolio_value'], 'b-', linewidth=2)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Drawdown
        if 'drawdown' in equity_curve.columns:
            ax2.fill_between(equity_curve['date'], equity_curve['drawdown'], 0,
                           color='red', alpha=0.3, label='Drawdown')
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_trade_analysis(self, trades: List[Dict]) -> plt.Figure:
        """
        Plot trade analysis charts

        Args:
            trades: List of trade dictionaries

        Returns:
            matplotlib Figure object
        """
        if not trades:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No trades to analyze', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            return fig

        trades_df = pd.DataFrame(trades)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)

        # P&L distribution
        ax1.hist(trades_df['net_pnl'], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('P&L Distribution')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Trade duration
        ax2.hist(trades_df['duration_days'], bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_title('Trade Duration Distribution')
        ax2.set_xlabel('Duration (days)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # R:R ratio distribution
        ax3.hist(trades_df['rr_ratio'], bins=20, alpha=0.7, edgecolor='black', color='green')
        ax3.set_title('Risk:Reward Ratio Distribution')
        ax3.set_xlabel('R:R Ratio')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # Cumulative P&L
        cumulative_pnl = trades_df['net_pnl'].cumsum()
        ax4.plot(range(len(cumulative_pnl)), cumulative_pnl, 'b-', linewidth=2)
        ax4.set_title('Cumulative P&L by Trade')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative P&L ($)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_all_charts(self, results: Dict, output_dir: str = 'charts'):
        """
        Save all charts to files

        Args:
            results: Backtest results dictionary
            output_dir: Directory to save charts
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Need to reconstruct data for charts - this would need actual implementation
        # based on the data structure from the backtest engine
        print(f"Charts would be saved to {output_dir}/")
        print("Chart saving functionality would need actual data reconstruction")

    def create_summary_report(self, metrics: Dict) -> plt.Figure:
        """
        Create a visual summary report of key metrics

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)

        # Key metrics table
        metrics_text = f"""
        Total Return: {metrics.get('total_return', 0):.2%}
        Total Trades: {metrics.get('num_trades', 0)}
        Win Rate: {metrics.get('win_rate', 0):.2%}
        Avg R:R Ratio: {metrics.get('avg_rr_ratio', 0):.2f}
        Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
        Profit Factor: {metrics.get('profit_factor', 0):.2f}
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        """

        ax1.text(0.1, 0.5, metrics_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace')
        ax1.set_title('Key Performance Metrics', fontweight='bold')
        ax1.axis('off')

        # Win/Loss pie chart
        wins = metrics.get('num_winning_trades', 0)
        losses = metrics.get('num_losing_trades', 0)
        if wins + losses > 0:
            ax2.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
                   colors=['green', 'red'])
            ax2.set_title('Win/Loss Ratio')

        # Monthly returns would go here (ax3)
        ax3.text(0.5, 0.5, 'Monthly Returns\n(Not implemented)', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('Monthly Returns')

        # Risk metrics
        ax4.text(0.5, 0.5, 'Risk Analysis\n(Not implemented)', ha='center', va='center',
                transform=ax4.transAxes)
        ax4.set_title('Risk Analysis')

        plt.tight_layout()
        return fig

    def create_trade_chart(self, trade: Dict, data: pd.DataFrame, ranges: List,
                          trade_num: int, context_days: int = 90) -> plt.Figure:
        """
        Create individual trade chart showing price, ranges, and entry/exit

        Args:
            trade: Trade dictionary with entry/exit information
            data: Full OHLC data
            ranges: List of trading ranges active during trade
            trade_num: Trade number for title
            context_days: Number of days before/after trade to show

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(15, 8))

        # Convert numpy datetime64 to pandas timestamps for comparison
        entry_date = pd.Timestamp(trade['entry_date'])
        exit_date = pd.Timestamp(trade['exit_date'])

        # Find the date range to display - include full range formation
        entry_idx = data.index.get_indexer([entry_date], method='nearest')[0]
        exit_idx = data.index.get_indexer([exit_date], method='nearest')[0]

        # Get swing points from trade to determine range formation period
        swing_high = trade.get('swing_high')
        swing_low = trade.get('swing_low')

        # Find the earliest point in the range formation
        range_start_idx = entry_idx
        if swing_high and swing_low:
            swing_high_idx = swing_high[0] if isinstance(swing_high, tuple) else swing_high.get('index', entry_idx)
            swing_low_idx = swing_low[0] if isinstance(swing_low, tuple) else swing_low.get('index', entry_idx)
            range_start_idx = min(swing_high_idx, swing_low_idx)

        # Set chart boundaries to show full range plus some context
        start_idx = max(0, range_start_idx - context_days)
        end_idx = min(len(data) - 1, exit_idx + context_days)

        chart_data = data.iloc[start_idx:end_idx + 1]

        # Plot candlesticks
        self._plot_candlesticks(ax, chart_data)

        # Plot only the specific range and order block used for this trade
        trade_range = trade.get('range')
        trade_order_block = trade.get('order_block')

        if trade_range:
            range_start_date = data.index[trade_range.start_index] if trade_range.start_index < len(data) else data.index[-1]
            range_end_date = data.index[trade_range.end_index] if trade_range.end_index < len(data) else data.index[-1]

            # Determine color based on direction
            color = 'green' if trade_range.direction == 'bullish' else 'red'
            alpha = 0.2

            # Calculate x positions for the range rectangle
            chart_start = chart_data.index[0]
            chart_end = chart_data.index[-1]
            chart_duration = (chart_end - chart_start).total_seconds()

            range_start_rel = max(0, (range_start_date - chart_start).total_seconds() / chart_duration)
            range_end_rel = min(1, (range_end_date - chart_start).total_seconds() / chart_duration)

            # Plot range rectangle
            ax.axhspan(trade_range.low_price, trade_range.high_price,
                      xmin=range_start_rel, xmax=range_end_rel,
                      alpha=alpha, color=color, label=f'{trade_range.direction.title()} Range')

        # Plot only the specific order block used for this trade
        if trade_order_block and trade_order_block['index'] < len(data):
            ob_date = data.index[trade_order_block['index']]
            if chart_data.index[0] <= ob_date <= chart_data.index[-1]:
                chart_start = chart_data.index[0]
                chart_end = chart_data.index[-1]
                chart_duration = (chart_end - chart_start).total_seconds()
                ob_rel_pos = (ob_date - chart_start).total_seconds() / chart_duration
                ax.axhspan(trade_order_block['low'], trade_order_block['high'],
                          xmin=ob_rel_pos, xmax=min(1, ob_rel_pos + 0.02),
                          alpha=0.8, color='orange', label='Order Block')

        # Plot swing points that created the range
        if swing_high and swing_low:
            # Extract swing point data
            if isinstance(swing_high, tuple):
                swing_high_idx, _, swing_high_price = swing_high
                swing_high_date = data.index[swing_high_idx] if swing_high_idx < len(data) else entry_date
            else:
                swing_high_price = swing_high.get('price', 0)
                swing_high_date = swing_high.get('date', entry_date)

            if isinstance(swing_low, tuple):
                swing_low_idx, _, swing_low_price = swing_low
                swing_low_date = data.index[swing_low_idx] if swing_low_idx < len(data) else entry_date
            else:
                swing_low_price = swing_low.get('price', 0)
                swing_low_date = swing_low.get('date', entry_date)

            # Plot swing high and low
            ax.scatter(swing_high_date, swing_high_price, color='red', marker='v',
                      s=200, alpha=0.8, zorder=6, label='Swing High')
            ax.scatter(swing_low_date, swing_low_price, color='green', marker='^',
                      s=200, alpha=0.8, zorder=6, label='Swing Low')

            # Draw line connecting the range
            ax.plot([swing_low_date, swing_high_date], [swing_low_price, swing_high_price],
                   color='gray', linestyle='-', alpha=0.5, linewidth=2, label='Range Formation')

        # Plot trade entry
        entry_color = 'blue'
        entry_marker = '^' if trade['type'] == 'long' else 'v'
        ax.scatter(entry_date, trade['entry_price'], color=entry_color,
                  marker=entry_marker, s=150, alpha=1.0, zorder=5,
                  label=f"{trade['type'].upper()} Entry")

        # Plot trade exit
        exit_color = 'purple'
        ax.scatter(exit_date, trade['exit_price'], color=exit_color,
                  marker='x', s=150, alpha=1.0, zorder=5,
                  label=f"Exit")

        # Add stop loss and take profit levels only
        if 'stop_loss' in trade and 'take_profit' in trade:
            ax.axhline(y=trade['stop_loss'], color='red', linestyle='-',
                      alpha=0.8, linewidth=2, label='Stop Loss')
            ax.axhline(y=trade['take_profit'], color='green', linestyle='-',
                      alpha=0.8, linewidth=2, label='Take Profit')

        # Formatting
        title = f"Trade #{trade_num} - {trade['type'].upper()} | " \
                f"P&L: ${trade['net_pnl']:.2f} | R:R: {trade['rr_ratio']:.2f}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format dates on x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, context_days // 5)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig