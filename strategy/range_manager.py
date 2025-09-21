import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TradingRange:
    """Represents a trading range with all necessary information"""
    start_index: int
    end_index: int
    high_price: float
    low_price: float
    high_index: int
    low_index: int
    direction: str  # 'bullish' or 'bearish'
    is_valid: bool = True
    order_block: Optional[Dict] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    high_date: Optional[str] = None
    low_date: Optional[str] = None

    @property
    def range_size(self) -> float:
        """Calculate the size of the range"""
        return self.high_price - self.low_price

class RangeManager:
    """Manages trading ranges creation, validation and invalidation"""

    def __init__(self):
        self.ranges = []

    def create_ranges(self, swings: List[Tuple[int, str, float]], ohlc_data: Dict = None) -> List[TradingRange]:
        """
        Create ranges from consecutive swing pairs

        Args:
            swings: List of (index, type, price) tuples sorted chronologically

        Returns:
            List of TradingRange objects
        """
        ranges = []

        # Need at least 2 swings to create a range
        if len(swings) < 2:
            return ranges

        # Create ranges from consecutive swing pairs
        for i in range(len(swings) - 1):
            swing1 = swings[i]
            swing2 = swings[i + 1]

            # Determine range characteristics
            start_index = swing1[0]
            end_index = swing2[0]

            if swing1[1] == 'high' and swing2[1] == 'low':
                # High to low = bearish range
                high_price = swing1[2]
                low_price = swing2[2]
                high_index = swing1[0]
                low_index = swing2[0]
                direction = 'bearish'
            elif swing1[1] == 'low' and swing2[1] == 'high':
                # Low to high = bullish range
                high_price = swing2[2]
                low_price = swing1[2]
                high_index = swing2[0]
                low_index = swing1[0]
                direction = 'bullish'
            else:
                # Skip if consecutive highs or lows
                continue

            # Get timestamps if OHLC data is available
            start_date = None
            end_date = None
            high_date = None
            low_date = None

            if ohlc_data and 'datetime' in ohlc_data:
                dates = ohlc_data['datetime']
                if start_index < len(dates):
                    start_date = str(dates[start_index])
                if end_index < len(dates):
                    end_date = str(dates[end_index])
                if high_index < len(dates):
                    high_date = str(dates[high_index])
                if low_index < len(dates):
                    low_date = str(dates[low_index])

            # Create range object
            trading_range = TradingRange(
                start_index=start_index,
                end_index=end_index,
                high_price=high_price,
                low_price=low_price,
                high_index=high_index,
                low_index=low_index,
                direction=direction,
                start_date=start_date,
                end_date=end_date,
                high_date=high_date,
                low_date=low_date
            )

            ranges.append(trading_range)

        self.ranges = ranges
        return ranges

    def invalidate_ranges(self, current_index: int, current_high: float, current_low: float):
        """
        Mark ranges as invalid if price breaks above high or below low

        Args:
            current_index: Current bar index
            current_high: Current bar high price
            current_low: Current bar low price
        """
        for trading_range in self.ranges:
            if not trading_range.is_valid:
                continue

            # Only check ranges that have been formed (current index > end_index)
            if current_index <= trading_range.end_index:
                continue

            # Check if price broke above range high or below range low
            if current_high > trading_range.high_price or current_low < trading_range.low_price:
                trading_range.is_valid = False

    def get_valid_ranges(self) -> List[TradingRange]:
        """Get all currently valid ranges"""
        return [r for r in self.ranges if r.is_valid]

    def get_largest_valid_range(self) -> Optional[TradingRange]:
        """Get the largest valid range by price span"""
        valid_ranges = self.get_valid_ranges()

        if not valid_ranges:
            return None

        # Return range with largest price span
        return max(valid_ranges, key=lambda r: r.range_size)

    def update_ranges(self, swings: List[Tuple[int, str, float]], current_index: int,
                     current_high: float, current_low: float, ohlc_data: Dict = None):
        """
        Update ranges: create new ones from swings and invalidate existing ones

        Args:
            swings: Current list of all swing points
            current_index: Current bar index
            current_high: Current bar high
            current_low: Current bar low
            ohlc_data: OHLC data for validation
        """
        # Recreate ranges from all swings
        self.create_ranges(swings, ohlc_data)

        # Validate ranges for monotonic progression if OHLC data available
        if ohlc_data:
            self.validate_monotonic_ranges(ohlc_data)

        # Invalidate ranges based on current price action
        self.invalidate_ranges(current_index, current_high, current_low)

    def get_range_stats(self) -> Dict:
        """Get statistics about ranges"""
        valid_ranges = self.get_valid_ranges()
        all_ranges = self.ranges

        return {
            'total_ranges': len(all_ranges),
            'valid_ranges': len(valid_ranges),
            'invalidated_ranges': len(all_ranges) - len(valid_ranges),
            'avg_range_size': np.mean([r.range_size for r in all_ranges]) if all_ranges else 0,
            'largest_range_size': max([r.range_size for r in valid_ranges]) if valid_ranges else 0
        }

    def validate_monotonic_ranges(self, ohlc_data: Dict):
        """
        Validate that ranges have monotonic price progression

        For bullish ranges (low → high): All intermediate lows must be higher than start low
        For bearish ranges (high → low): All intermediate highs must be lower than start high
        """
        highs = ohlc_data['high']
        lows = ohlc_data['low']

        for trading_range in self.ranges:
            if not trading_range.is_valid:
                continue

            start_idx = trading_range.start_index
            end_idx = trading_range.end_index

            if trading_range.direction == 'bullish':
                # Check that all lows AND highs between start and end are higher than start values
                start_low = trading_range.low_price
                start_high = highs[start_idx] if start_idx < len(highs) else 0

                for i in range(start_idx + 1, end_idx):
                    if i < len(lows) and i < len(highs):
                        # Both lows and highs must be higher for true bullish progression
                        if lows[i] <= start_low or highs[i] <= start_high:
                            trading_range.is_valid = False
                            break

            elif trading_range.direction == 'bearish':
                # Check that all highs AND lows between start and end are lower than start values
                start_high = trading_range.high_price
                start_low = lows[start_idx] if start_idx < len(lows) else float('inf')

                for i in range(start_idx + 1, end_idx):
                    if i < len(highs) and i < len(lows):
                        # Both highs and lows must be lower for true bearish progression
                        if highs[i] >= start_high or lows[i] >= start_low:
                            trading_range.is_valid = False
                            break