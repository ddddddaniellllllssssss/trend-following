#!/usr/bin/env python3
"""
Main entry point for the Range-Based Swing Trading Strategy

This script runs the complete backtesting pipeline:
1. Load and validate market data
2. Run the strategy backtest
3. Calculate performance metrics
4. Generate visualizations
5. Print results summary
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics import PerformanceMetrics
from visualization.charts import TradingCharts
import config

def main():
    """Run the complete backtesting pipeline"""
    print("Range-Based Swing Trading Strategy Backtest")
    print("=" * 50)
    print(f"Data: {config.DATA_PATH}")
    print(f"Period: {config.START_DATE} to {config.END_DATE}")
    print(f"Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"Risk Per Trade: {config.RISK_PERCENT:.1%} of account")
    print(f"Min R:R Ratio: {config.MIN_RR_RATIO}")
    print("=" * 50)

    try:
        # Initialize backtest engine
        engine = BacktestEngine(config.INITIAL_CAPITAL)

        # Run backtest
        print("\nRunning backtest...")
        results = engine.run_backtest()

        # Calculate comprehensive metrics
        print("\nCalculating performance metrics...")
        metrics = PerformanceMetrics.calculate_all_metrics(results)

        # Print performance report
        PerformanceMetrics.print_performance_report(metrics)

        # Save detailed trade log
        print("\nSaving trade log...")
        engine.save_trade_log('trade_log.csv')

        # Create visualizations
        print("\nGenerating charts...")
        charts = TradingCharts()

        try:
            # Summary report
            summary_fig = charts.create_summary_report(metrics)
            summary_fig.savefig('strategy_summary.png', dpi=300, bbox_inches='tight')
            print("Saved: strategy_summary.png")

            # Equity curve
            if 'equity_curve' in results and not results['equity_curve'].empty:
                equity_fig = charts.plot_equity_curve(results['equity_curve'])
                equity_fig.savefig('equity_curve.png', dpi=300, bbox_inches='tight')
                print("Saved: equity_curve.png")

            # Trade analysis
            if results['trades']:
                trade_fig = charts.plot_trade_analysis(results['trades'])
                trade_fig.savefig('trade_analysis.png', dpi=300, bbox_inches='tight')
                print("Saved: trade_analysis.png")

            # Individual trade charts
            if results['trades'] and hasattr(engine, 'data'):
                print("\nGenerating individual trade charts...")
                for i, trade in enumerate(results['trades']):
                    try:
                        trade_chart = charts.create_trade_chart(
                            trade, engine.data, [], i + 1, context_days=90
                        )
                        filename = f'trade_{i+1}_chart.png'
                        trade_chart.savefig(filename, dpi=300, bbox_inches='tight')
                        print(f"Saved: {filename}")
                    except Exception as e:
                        print(f"Warning: Could not generate chart for trade {i+1}: {e}")

        except Exception as e:
            print(f"Warning: Could not generate some charts: {e}")

        # Print trade details if any trades were made
        if results['trades']:
            print(f"\nFirst 5 trades:")
            print("-" * 80)
            for i, trade in enumerate(results['trades'][:5]):
                print(f"Trade {i+1}: {trade['type'].upper()} | "
                      f"Entry: ${trade['entry_price']:.2f} ({pd.Timestamp(trade['entry_date']).strftime('%Y-%m-%d')}) | "
                      f"Exit: ${trade['exit_price']:.2f} ({pd.Timestamp(trade['exit_date']).strftime('%Y-%m-%d')}) | "
                      f"P&L: ${trade['net_pnl']:.2f} | "
                      f"R:R: {trade['rr_ratio']:.2f}")

        # Save detailed results
        print(f"\nBacktest completed successfully!")
        print(f"Check the generated PNG files for detailed analysis.")

        return results, metrics

    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, metrics = main()