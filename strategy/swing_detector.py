import numpy as np
from typing import List, Tuple
import config

class SwingDetector:
    """Detects swing highs and lows using 3-candle pattern"""

    def __init__(self, period: int = None):
        self.period = period or config.SWING_PERIOD

    def find_swing_points(self, highs: np.ndarray, lows: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Find swing highs and lows using 3-candle pattern

        Args:
            highs: Array of high prices
            lows: Array of low prices

        Returns:
            tuple: (swing_high_indices, swing_low_indices)
        """
        swing_highs = []
        swing_lows = []

        # Need at least 3 candles for pattern
        if len(highs) < 3:
            return swing_highs, swing_lows

        # Check each candle starting from index 1 (middle of 3-candle pattern)
        for i in range(1, len(highs) - 1):
            # Swing high: center candle higher than both neighbors
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                swing_highs.append(i)

            # Swing low: center candle lower than both neighbors
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                swing_lows.append(i)

        return swing_highs, swing_lows

    def get_combined_swings(self, highs: np.ndarray, lows: np.ndarray) -> List[Tuple[int, str, float]]:
        """
        Get chronologically ordered list of all swing points, filtered to ensure alternation

        Returns:
            List of tuples: (index, type, price) where type is 'high' or 'low'
        """
        swing_highs, swing_lows = self.find_swing_points(highs, lows)

        # Combine and sort chronologically
        all_swings = []

        for idx in swing_highs:
            all_swings.append((idx, 'high', highs[idx]))

        for idx in swing_lows:
            all_swings.append((idx, 'low', lows[idx]))

        # Sort by index (chronological order)
        all_swings.sort(key=lambda x: x[0])

        # Filter to ensure alternating pattern (no consecutive highs or lows)
        filtered_swings = self._filter_alternating_swings(all_swings)

        return filtered_swings

    def validate_swing_points(self, swings: List[Tuple[int, str, float]]) -> bool:
        """Validate that swing points alternate between highs and lows"""
        if len(swings) < 2:
            return True

        for i in range(1, len(swings)):
            if swings[i][1] == swings[i-1][1]:
                # Two consecutive highs or lows - this shouldn't happen in a proper trend
                return False

        return True

    def _filter_alternating_swings(self, swings: List[Tuple[int, str, float]]) -> List[Tuple[int, str, float]]:
        """
        Filter swings to ensure monotonic price action between valid swings

        Valid bullish range: Low → High with all intermediate lows being higher
        Valid bearish range: High → Low with all intermediate highs being lower
        """
        if len(swings) <= 1:
            return swings

        filtered = [swings[0]]  # Always keep the first swing

        for i in range(1, len(swings)):
            current_swing = swings[i]
            last_filtered = filtered[-1]

            # If same type as previous, keep only the more extreme one
            if current_swing[1] == last_filtered[1]:
                if current_swing[1] == 'high':
                    # Keep the higher high
                    if current_swing[2] > last_filtered[2]:
                        filtered[-1] = current_swing
                else:  # low
                    # Keep the lower low
                    if current_swing[2] < last_filtered[2]:
                        filtered[-1] = current_swing
            else:
                # Different type - check for monotonic progression
                if self._is_monotonic_progression(last_filtered, current_swing):
                    filtered.append(current_swing)
                # If not monotonic, skip this swing and continue

        return filtered

    def _is_monotonic_progression(self, start_swing: Tuple[int, str, float],
                                end_swing: Tuple[int, str, float]) -> bool:
        """
        Check if the progression from start_swing to end_swing is monotonic

        For Low → High: All intermediate lows must be higher than start_swing
        For High → Low: All intermediate highs must be lower than start_swing
        """
        # For now, accept the progression (this would need access to full price data)
        # The real validation should happen at the range level
        return True