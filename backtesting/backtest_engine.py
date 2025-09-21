import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
import os

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processor import DataProcessor
from strategy.swing_detector import SwingDetector
from strategy.range_manager import RangeManager
from strategy.order_blocks import OrderBlockDetector
from strategy.signal_generator import SignalGenerator
from backtesting.portfolio import Portfolio
import config

class BacktestEngine:
    """Event-driven backtesting engine for the range trading strategy"""

    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL

        # Initialize components
        self.data_processor = DataProcessor()
        self.swing_detector = SwingDetector()
        self.range_manager = RangeManager()
        self.order_block_detector = OrderBlockDetector()
        self.signal_generator = SignalGenerator()
        self.portfolio = Portfolio(self.initial_capital)

        # Data storage
        self.data = None
        self.ohlc = None
        self.results = []
        self.trades = []

        # Current state
        self.current_position = None
        self.current_swings = []

    def load_data(self):
        """Load and validate market data"""
        print("Loading market data...")
        self.data = self.data_processor.load_data()

        if not self.data_processor.validate_data():
            raise ValueError("Data validation failed")

        self.ohlc = self.data_processor.get_ohlc()
        print(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")

    def run_backtest(self) -> Dict:
        """
        Run the complete backtest

        Returns:
            Dictionary containing backtest results
        """
        if self.data is None:
            self.load_data()

        print("Starting backtest...")

        # Get OHLC arrays
        opens = self.ohlc['open']
        highs = self.ohlc['high']
        lows = self.ohlc['low']
        closes = self.ohlc['close']
        dates = self.ohlc['datetime']

        # Initialize tracking variables
        previous_price = None

        # Process each bar
        for i in range(len(closes)):
            current_date = dates[i]
            current_open = opens[i]
            current_high = highs[i]
            current_low = lows[i]
            current_close = closes[i]

            # Update portfolio value
            self.portfolio.update_value(current_close, i)

            # Process existing position exits first
            if self.current_position:
                exit_signal = self.signal_generator.check_exit_signals(
                    self.current_position, current_high, current_low, current_close
                )

                if exit_signal:
                    self._execute_exit(i, exit_signal, current_date, current_close)

            # Only look for new entries if we don't have a position
            if not self.current_position:
                # Update swing points with data up to current bar
                if i >= 2:  # Need at least 3 bars for swing detection
                    highs_subset = highs[:i+1]
                    lows_subset = lows[:i+1]

                    # Get filtered swing points (ensures alternating pattern)
                    all_swings = self.swing_detector.get_combined_swings(
                        highs_subset, lows_subset
                    )
                    self.current_swings = all_swings

                    # Update ranges
                    self.range_manager.update_ranges(
                        all_swings, i, current_high, current_low, self.ohlc
                    )

                    # Add order blocks to valid ranges
                    valid_ranges = self.range_manager.get_valid_ranges()
                    self.order_block_detector.add_order_blocks_to_ranges(
                        valid_ranges, opens, highs, lows, closes
                    )

                    # Get largest valid range
                    largest_range = self.range_manager.get_largest_valid_range()

                    # Check for entry signals
                    if previous_price is not None and len(all_swings) >= 2:
                        entry_signal = self.signal_generator.check_entry_signals(
                            all_swings, current_close, current_high, current_low, self.ohlc, i
                        )

                        if entry_signal:
                            self._execute_entry(i, entry_signal, current_date, current_close)

            # Store bar results
            self.results.append({
                'date': current_date,
                'index': i,
                'close': current_close,
                'portfolio_value': self.portfolio.get_value(),
                'position': self.current_position is not None,
                'num_valid_ranges': len(self.range_manager.get_valid_ranges()),
                'largest_range_size': self.range_manager.get_largest_valid_range().range_size
                                    if self.range_manager.get_largest_valid_range() else 0
            })

            previous_price = current_close

        print(f"Backtest completed. Processed {len(closes)} bars")
        print(f"Total trades: {len(self.trades)}")
        print(f"Final portfolio value: ${self.portfolio.get_value():,.2f}")

        return self._compile_results()

    def _execute_entry(self, index: int, signal: Dict, date: datetime, price: float):
        """Execute entry order"""
        # Calculate position size based on risk
        shares = self.signal_generator.calculate_position_size(
            signal['entry_price'], signal['stop_loss'], self.portfolio.get_value()
        )

        if shares <= 0:
            return

        # Execute trade
        cost = shares * signal['entry_price']

        # Apply commission
        commission = cost * config.COMMISSION
        self.portfolio.cash -= (cost + commission)

        # Store position
        self.current_position = {
            'type': signal['type'],
            'entry_index': index,
            'entry_date': date,
            'entry_price': signal['entry_price'],
            'shares': shares,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'rr_ratio': signal['rr_ratio'],
            'cost': cost,
            'commission': commission,
            'swing_high': signal.get('swing_high'),  # Store the swing high used
            'swing_low': signal.get('swing_low'),  # Store the swing low used
            'order_block': signal.get('order_block')  # Store the specific order block used
        }

        # Log entry with swing information
        swing_info = ""
        if signal.get('swing_high') and signal.get('swing_low'):
            swing_high = signal['swing_high']
            swing_low = signal['swing_low']
            swing_info = f" | Swing High: ${swing_high[2]:.2f} at index {swing_high[0]}, Swing Low: ${swing_low[2]:.2f} at index {swing_low[0]}"

        print(f"Entry: {signal['type']} at ${signal['entry_price']:.2f} on {pd.Timestamp(date).strftime('%Y-%m-%d')}{swing_info}")

    def _execute_partial_exit(self, index: int, date: datetime, price: float):
        """Execute partial exit at configurable R:R (close half position)"""
        if not self.current_position:
            return

        position = self.current_position

        # Calculate exit price at configured R:R level
        risk = abs(position['stop_loss'] - position['entry_price'])
        if position['type'] == 'long':
            exit_price = position['entry_price'] + (config.PARTIAL_EXIT_RR * risk)
        else:  # short
            exit_price = position['entry_price'] - (config.PARTIAL_EXIT_RR * risk)

        # Close half the position
        shares_to_close = position['shares'] / 2
        remaining_shares = position['shares'] - shares_to_close

        # Calculate P&L for closed portion
        if position['type'] == 'long':
            pnl = shares_to_close * (exit_price - position['entry_price'])
        else:  # short
            pnl = shares_to_close * (position['entry_price'] - exit_price)

        # Apply commission
        commission = shares_to_close * exit_price * config.COMMISSION
        net_pnl = pnl - commission

        # Update portfolio
        self.portfolio.cash += shares_to_close * exit_price - commission

        # Record partial trade
        partial_trade = {
            'entry_index': position['entry_index'],
            'exit_index': index,
            'entry_date': position['entry_date'],
            'exit_date': date,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'shares': shares_to_close,
            'gross_pnl': pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'exit_reason': f'partial_exit_{config.PARTIAL_EXIT_RR:.1f}_1',
            'rr_ratio': config.PARTIAL_EXIT_RR,
            'duration_days': (pd.Timestamp(date) - pd.Timestamp(position['entry_date'])).days,
            'is_partial': True,
            'swing_high': position.get('swing_high'),
            'swing_low': position.get('swing_low'),
            'order_block': position.get('order_block'),
            'swing_high_price': position.get('swing_high')[2] if position.get('swing_high') else None,
            'swing_low_price': position.get('swing_low')[2] if position.get('swing_low') else None,
            'swing_high_index': position.get('swing_high')[0] if position.get('swing_high') else None,
            'swing_low_index': position.get('swing_low')[0] if position.get('swing_low') else None
        }

        self.trades.append(partial_trade)

        # Update position
        position['shares'] = remaining_shares
        position['partial_taken'] = True

        # Move stop loss to breakeven for remaining position
        position['stop_loss'] = position['entry_price']

        print(f"Partial Exit: {position['type']} at ${exit_price:.2f} ({config.PARTIAL_EXIT_RR:.1f}:1 R:R) on {pd.Timestamp(date).strftime('%Y-%m-%d')}, P&L: ${net_pnl:.2f}")
        print(f"Remaining shares: {remaining_shares:.2f}, Stop moved to breakeven: ${position['entry_price']:.2f}")

    def _execute_exit(self, index: int, exit_type: str, date: datetime, price: float):
        """Execute exit order"""
        if not self.current_position:
            return

        position = self.current_position

        # Handle partial exit differently
        if exit_type == 'partial_exit':
            self._execute_partial_exit(index, date, price)
            return

        # Determine exit price based on exit type
        if exit_type == 'stop_loss':
            exit_price = position['stop_loss']
        elif exit_type == 'take_profit':
            exit_price = position['take_profit']
        else:
            exit_price = price

        # Calculate P&L
        shares = position['shares']
        if position['type'] == 'long':
            pnl = shares * (exit_price - position['entry_price'])
        else:  # short
            pnl = shares * (position['entry_price'] - exit_price)

        # Apply commission
        commission = shares * exit_price * config.COMMISSION
        net_pnl = pnl - commission - position['commission']

        # Update portfolio
        self.portfolio.cash += shares * exit_price - commission

        # Record trade
        trade = {
            'entry_index': position['entry_index'],
            'exit_index': index,
            'entry_date': position['entry_date'],
            'exit_date': date,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'shares': shares,
            'gross_pnl': pnl,
            'commission': position['commission'] + commission,
            'net_pnl': net_pnl,
            'exit_reason': exit_type,
            'rr_ratio': position['rr_ratio'],
            'duration_days': (pd.Timestamp(date) - pd.Timestamp(position['entry_date'])).days,
            'swing_high': position.get('swing_high'),  # Store the swing high used for this trade
            'swing_low': position.get('swing_low'),  # Store the swing low used for this trade
            'order_block': position.get('order_block'),  # Store the specific order block used for this trade
            'swing_high_price': position.get('swing_high')[2] if position.get('swing_high') else None,
            'swing_low_price': position.get('swing_low')[2] if position.get('swing_low') else None,
            'swing_high_index': position.get('swing_high')[0] if position.get('swing_high') else None,
            'swing_low_index': position.get('swing_low')[0] if position.get('swing_low') else None
        }

        self.trades.append(trade)

        print(f"Exit: {exit_type} at ${exit_price:.2f} on {pd.Timestamp(date).strftime('%Y-%m-%d')}, P&L: ${net_pnl:.2f}")

        # Clear position
        self.current_position = None

    def _compile_results(self) -> Dict:
        """Compile final backtest results"""
        if not self.trades:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_rr_ratio': 0,
                'max_drawdown': 0,
                'trades': [],
                'equity_curve': pd.DataFrame(self.results)
            }

        trades_df = pd.DataFrame(self.trades)

        # Calculate metrics
        total_pnl = trades_df['net_pnl'].sum()
        total_return = (self.portfolio.get_value() - self.initial_capital) / self.initial_capital

        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

        avg_rr_ratio = trades_df['rr_ratio'].mean()

        # Calculate drawdown
        equity_curve = pd.DataFrame(self.results)
        equity_curve['running_max'] = equity_curve['portfolio_value'].cummax()
        equity_curve['drawdown'] = (equity_curve['portfolio_value'] - equity_curve['running_max']) / equity_curve['running_max']
        max_drawdown = equity_curve['drawdown'].min()

        return {
            'total_return': total_return,
            'total_pnl': total_pnl,
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_rr_ratio': avg_rr_ratio,
            'max_drawdown': abs(max_drawdown),
            'avg_trade_duration': trades_df['duration_days'].mean(),
            'trades': self.trades,
            'equity_curve': equity_curve,
            'range_stats': self.range_manager.get_range_stats()
        }

    def save_trade_log(self, filename: str = 'trade_log.csv'):
        """Save detailed trade log with range timestamps"""
        if not self.trades:
            print("No trades to save")
            return

        import pandas as pd

        # Create a simplified trade log for CSV export
        trade_records = []
        for trade in self.trades:
            record = {
                'entry_date': trade['entry_date'],
                'exit_date': trade['exit_date'],
                'type': trade['type'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'net_pnl': trade['net_pnl'],
                'exit_reason': trade['exit_reason'],
                'rr_ratio': trade['rr_ratio'],
                'duration_days': trade['duration_days'],
                'swing_high_price': trade.get('swing_high_price'),
                'swing_low_price': trade.get('swing_low_price'),
                'swing_high_index': trade.get('swing_high_index'),
                'swing_low_index': trade.get('swing_low_index')
            }
            trade_records.append(record)

        df = pd.DataFrame(trade_records)
        df.to_csv(filename, index=False)
        print(f"Trade log saved to {filename}")