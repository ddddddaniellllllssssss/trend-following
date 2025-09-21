import numpy as np
from typing import Dict, Optional
from .range_manager import TradingRange

class OrderBlockDetector:
    """Detects order blocks within trading ranges"""

    def find_order_block(self, trading_range: TradingRange,
                        opens: np.ndarray, highs: np.ndarray,
                        lows: np.ndarray, closes: np.ndarray) -> Optional[Dict]:
        """
        Find order block within a trading range

        For bearish ranges: Find last green (bullish) candle before the move down
        For bullish ranges: Find last red (bearish) candle before the move up

        Args:
            trading_range: The trading range to analyze
            opens, highs, lows, closes: OHLC price arrays

        Returns:
            Dictionary with order block info or None if not found
        """
        start_idx = trading_range.start_index
        end_idx = trading_range.end_index

        # Ensure we have valid indices
        if start_idx >= end_idx or end_idx >= len(closes):
            return None

        if trading_range.direction == 'bearish':
            # For bearish range (high to low), find last green candle before the low
            return self._find_last_bullish_candle(start_idx, end_idx, opens, highs, lows, closes)

        elif trading_range.direction == 'bullish':
            # For bullish range (low to high), find last red candle before the high
            return self._find_last_bearish_candle(start_idx, end_idx, opens, highs, lows, closes)

        return None

    def _find_last_bullish_candle(self, start_idx: int, end_idx: int,
                                 opens: np.ndarray, highs: np.ndarray,
                                 lows: np.ndarray, closes: np.ndarray) -> Optional[Dict]:
        """
        Find the last bullish (green) candle in the range
        This represents the order block for bearish ranges
        """
        # Search from end backwards to start
        for i in range(end_idx, start_idx - 1, -1):
            if i >= len(closes):
                continue

            # Check if candle is bullish (close > open)
            if closes[i] > opens[i]:
                return {
                    'index': i,
                    'high': highs[i],
                    'low': lows[i],
                    'open': opens[i],
                    'close': closes[i],
                    'type': 'bullish'
                }

        return None

    def _find_last_bearish_candle(self, start_idx: int, end_idx: int,
                                 opens: np.ndarray, highs: np.ndarray,
                                 lows: np.ndarray, closes: np.ndarray) -> Optional[Dict]:
        """
        Find the last bearish (red) candle in the range
        This represents the order block for bullish ranges
        """
        # Search from end backwards to start
        for i in range(end_idx, start_idx - 1, -1):
            if i >= len(closes):
                continue

            # Check if candle is bearish (close < open)
            if closes[i] < opens[i]:
                return {
                    'index': i,
                    'high': highs[i],
                    'low': lows[i],
                    'open': opens[i],
                    'close': closes[i],
                    'type': 'bearish'
                }

        return None

    def add_order_blocks_to_ranges(self, ranges: list,
                                  opens: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray):
        """
        Add order block information to all ranges

        Args:
            ranges: List of TradingRange objects
            opens, highs, lows, closes: OHLC price arrays
        """
        for trading_range in ranges:
            order_block = self.find_order_block(trading_range, opens, highs, lows, closes)
            trading_range.order_block = order_block

    def validate_order_block(self, order_block: Dict, trading_range: TradingRange) -> bool:
        """
        Validate that the order block makes sense within the range

        Args:
            order_block: Order block dictionary
            trading_range: The trading range

        Returns:
            True if order block is valid
        """
        if not order_block:
            return False

        # Order block should be within the range price boundaries
        ob_high = order_block['high']
        ob_low = order_block['low']

        # Check if order block is within range boundaries
        if ob_high > trading_range.high_price or ob_low < trading_range.low_price:
            return False

        # Order block should be between start and end indices
        if not (trading_range.start_index <= order_block['index'] <= trading_range.end_index):
            return False

        return True