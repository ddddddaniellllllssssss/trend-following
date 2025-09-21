#!/usr/bin/env python3
"""
Basic test of the strategy components without external dependencies
"""

import sys
import os
import csv
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_csv_data(file_path):
    """Load CSV data manually"""
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'time': int(row['time']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
    return data

def test_swing_detection():
    """Test swing point detection"""
    print("Testing swing point detection...")

    from strategy.swing_detector import SwingDetector

    # Create test data
    highs = [100, 105, 102, 108, 103, 110, 105]  # Swing high at index 3 and 5
    lows = [95, 98, 92, 100, 96, 105, 98]       # Swing low at index 2

    detector = SwingDetector()
    swing_highs, swing_lows = detector.find_swing_points(highs, lows)

    print(f"Swing highs found at indices: {swing_highs}")
    print(f"Swing lows found at indices: {swing_lows}")

    combined = detector.get_combined_swings(highs, lows)
    print(f"Combined swings: {combined}")

    return swing_highs, swing_lows

def test_range_creation():
    """Test range creation"""
    print("\nTesting range creation...")

    from strategy.range_manager import RangeManager

    # Test swings: [(index, type, price)]
    swings = [(1, 'high', 105), (2, 'low', 92), (3, 'high', 108), (4, 'low', 96)]

    manager = RangeManager()
    ranges = manager.create_ranges(swings)

    print(f"Created {len(ranges)} ranges:")
    for i, r in enumerate(ranges):
        print(f"  Range {i+1}: {r.direction} from {r.low_price} to {r.high_price}")

    return ranges

def test_basic_strategy():
    """Test basic strategy functionality"""
    print("="*50)
    print("BASIC STRATEGY TEST")
    print("="*50)

    try:
        # Test swing detection
        swing_highs, swing_lows = test_swing_detection()

        # Test range creation
        ranges = test_range_creation()

        # Test order block detection
        print("\nTesting order block detection...")
        from strategy.order_blocks import OrderBlockDetector

        detector = OrderBlockDetector()

        # Test data for order block detection
        opens = [100, 102, 101, 105, 104, 108, 106]
        highs = [101, 105, 102, 108, 105, 110, 107]
        lows = [99, 101, 99, 104, 102, 107, 105]
        closes = [102, 104, 100, 107, 103, 109, 106]

        if ranges:
            order_block = detector.find_order_block(ranges[0], opens, highs, lows, closes)
            print(f"Order block found: {order_block}")

        print("\nBasic strategy test completed successfully!")
        return True

    except Exception as e:
        print(f"Error in basic test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test loading the actual BTC data"""
    print("\nTesting data loading...")

    try:
        data_path = "data/btc-usdt-binance-daily.csv"
        data = load_csv_data(data_path)

        print(f"Loaded {len(data)} data points")
        print(f"First row: {data[0]}")
        print(f"Last row: {data[-1]}")

        # Filter for 2022-2024 (timestamps)
        start_timestamp = 1640995200  # 2022-01-01
        end_timestamp = 1735689600    # 2024-12-31

        filtered_data = [d for d in data if start_timestamp <= d['time'] <= end_timestamp]
        print(f"Filtered to {len(filtered_data)} data points for 2022-2024")

        return filtered_data

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Test basic strategy components
    success = test_basic_strategy()

    if success:
        # Test data loading
        data = test_data_loading()

        if data:
            print(f"\nStrategy implementation is ready!")
            print(f"Data range: {len(data)} bars")
            print(f"To run full backtest, install: pandas, numpy, matplotlib")
        else:
            print(f"\nCould not load data, but core strategy is implemented")
    else:
        print(f"\nBasic tests failed - check implementation")