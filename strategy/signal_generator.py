import numpy as np
from typing import Optional, Dict, Tuple, List
from .range_manager import TradingRange
import config

class SignalGenerator:
    """Generates entry and exit signals based on range and order block analysis"""

    def __init__(self, min_rr_ratio: float = None):
        self.min_rr_ratio = min_rr_ratio or config.MIN_RR_RATIO

    def check_entry_signals(self, current_swings: List, current_price: float,
                           current_high: float, current_low: float,
                           ohlc_data: Dict = None, current_index: int = None) -> Optional[Dict]:
        """
        Check for retracement-based entry signals

        At swing highs: Look for LONG entries on pullbacks to order blocks
        At swing lows: Look for SHORT entries on pullbacks to previous swing high area

        Args:
            current_swings: List of all swing points
            current_price: Current close price
            current_high: Current bar high
            current_low: Current bar low
            ohlc_data: OHLC data for order block detection

        Returns:
            Dictionary with signal info or None
        """
        if len(current_swings) < 2 or not ohlc_data:
            return None

        # Check for LONG entries (at any swing high with pullback to order block)
        long_signal = self._check_long_entry_at_swing_high(
            current_swings, current_price, current_high, current_low, ohlc_data, current_index
        )
        if long_signal:
            return long_signal

        # Check for SHORT entries (at any swing low with pullback to order block)
        short_signal = self._check_short_entry_at_swing_low(
            current_swings, current_price, current_high, current_low, ohlc_data, current_index
        )
        if short_signal:
            return short_signal

        return None

    def _check_long_entry_at_swing_high(self, current_swings: List, current_price: float,
                                       current_high: float, current_low: float,
                                       ohlc_data: Dict, current_index: int = None) -> Optional[Dict]:
        """
        At every swing high: Look for LONG entry on pullback to order block

        Logic:
        1. For each swing high, find the most recent swing low before it
        2. This creates a range (swing low → swing high)
        3. Wait for price to retrace back to order block (candle before swing low)
        4. Enter LONG when price touches order block area
        """
        # Check every swing high from oldest to newest to catch earliest entries
        for i in range(len(current_swings)):
            swing_index, swing_type, swing_price = current_swings[i]

            if swing_type != 'high':
                continue

            # Find the most recent swing low before this swing high
            previous_swing_low = None
            for j in range(i - 1, -1, -1):
                if current_swings[j][1] == 'low':
                    previous_swing_low = current_swings[j]
                    break

            if not previous_swing_low:
                continue

            swing_low_index = previous_swing_low[0]
            swing_low_price = previous_swing_low[2]

            # Check if this range has been invalidated by swing low being taken out
            range_invalidated = False

            # First check: Has current price broken below the swing low?
            if current_low < swing_low_price:
                print(f"LONG range INVALIDATED: Current price ${current_low:.2f} broke below swing low ${swing_low_price:.2f}")
                range_invalidated = True

            # Second check: Look for any swing low after our swing low that broke below it
            if not range_invalidated:
                for later_swing in current_swings:
                    later_index, later_type, later_price = later_swing
                    if (later_type == 'low' and
                        later_index > swing_low_index and  # Came after our swing low
                        later_price < swing_low_price):  # Broke below our swing low
                        print(f"LONG range INVALIDATED: Swing low ${swing_low_price:.2f} taken out by later low ${later_price:.2f} at index {later_index}")
                        range_invalidated = True
                        break

            if range_invalidated:
                continue

            # Check if current swing high broke above a previous swing high (showing strength)
            range_validated = False
            if swing_index > swing_low_index:  # Current swing high came after the swing low
                # Look for any previous swing high that this swing broke above
                for prev_swing in current_swings:
                    prev_index, prev_type, prev_price = prev_swing
                    if (prev_type == 'high' and
                        prev_index < swing_low_index and  # Previous high before our swing low
                        swing_price > prev_price):  # Current swing high broke above it
                        print(f"LONG range VALIDATED: Current swing high ${swing_price:.2f} broke above previous high ${prev_price:.2f}")
                        range_validated = True
                        break

            if not range_validated:
                print(f"LONG range REJECTED: Current swing high ${swing_price:.2f} did not break any previous highs")
                continue

            # Check minimum range width - calculate as percentage move from low to high
            range_width_pct = (swing_price - swing_low_price) / swing_low_price
            print(f"LONG range check: High ${swing_price:.2f} -> Low ${swing_low_price:.2f} = {range_width_pct:.1%} (min: {config.MIN_RANGE_WIDTH:.1%})")
            if range_width_pct < config.MIN_RANGE_WIDTH:
                print(f"  -> REJECTED: Range too small")
                continue

            # Get order block (candle before the swing low)
            order_block_index = swing_low_index - 1
            if order_block_index < 0:
                continue

            order_block_high = ohlc_data['high'][order_block_index]
            order_block_low = ohlc_data['low'][order_block_index]

            # Validate order block constraints
            range_size = swing_price - swing_low_price
            order_block_size = order_block_high - order_block_low
            max_order_block_size = range_size / 3  # Maximum 1/3 of range

            print(f"    Order block: ${order_block_low:.2f}-${order_block_high:.2f} (size: ${order_block_size:.2f})")
            print(f"    Range: ${swing_low_price:.2f}-${swing_price:.2f} (size: ${range_size:.2f})")
            print(f"    Max order block size (1/3): ${max_order_block_size:.2f}")

            # Check if order block is completely contained within range
            if order_block_low < swing_low_price or order_block_high > swing_price:
                print(f"    -> REJECTED: Order block not contained within range")
                continue

            # Check if order block is too large (more than 1/3 of range)
            if order_block_size > max_order_block_size:
                print(f"    -> REJECTED: Order block too large ({order_block_size:.2f} > {max_order_block_size:.2f})")
                continue

            # Debug: Always log when we're evaluating this range
            print(f"    LONG range evaluation: Swing high ${swing_price:.2f} at {swing_index}, Order block ${order_block_low:.2f}-${order_block_high:.2f} at {order_block_index}")
            print(f"    Current index {current_index}, Current low ${current_low:.2f}, Target: <= ${order_block_high:.2f}")

            # Enter LONG when price retraces within the order block (but not below swing low)
            if (current_low <= order_block_high and  # Price touches order block high
                current_low >= order_block_low and   # Price stays within order block (not below)
                current_low > swing_low_price):      # Price hasn't broken swing low
                print(f"    LONG ENTRY TRIGGER: Current low ${current_low:.2f} within order block ${order_block_low:.2f}-${order_block_high:.2f}, swing low ${swing_low_price:.2f} unbreached (index {current_index})")
                entry_price = order_block_high  # Enter at order block high
                stop_loss = order_block_low  # Stop at order block low
                take_profit = swing_price + (swing_price - swing_low_price) * 0.1  # Target above swing high

                risk = abs(entry_price - stop_loss)  # Distance from entry to stop loss
                reward = abs(take_profit - entry_price)  # Distance from entry to take profit

                if risk <= 0:
                    print(f"    -> REJECTED: Risk is zero or negative ({risk})")
                    continue

                rr_ratio = reward / risk  # R:R = Reward : Risk

                if rr_ratio >= self.min_rr_ratio:
                    print(f"  -> LONG TRADE TRIGGERED! Entry at index {current_index}, Range: {range_width_pct:.1%}, R:R: {rr_ratio:.1f}")
                    return {
                        'type': 'long',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk': risk,
                        'reward': reward,
                        'rr_ratio': rr_ratio,
                        'swing_high': (swing_index, swing_type, swing_price),
                        'swing_low': previous_swing_low,
                        'order_block': {'high': order_block_high, 'low': order_block_low, 'index': order_block_index}
                    }
                else:
                    print(f"    -> REJECTED: R:R too low ({rr_ratio:.2f} < {self.min_rr_ratio})")

        return None

    def _check_short_entry_at_swing_low(self, current_swings: List, current_price: float,
                                      current_high: float, current_low: float,
                                      ohlc_data: Dict, current_index: int = None) -> Optional[Dict]:
        """
        At every swing low: Look for SHORT entry on pullback to order block

        Logic:
        1. For each swing low, find the most recent swing high before it
        2. This creates a range (swing high → swing low)
        3. Wait for price to retrace back to order block (candle before swing high)
        4. Enter SHORT when price touches order block area
        """
        # Check every swing low from oldest to newest to catch earliest entries
        for i in range(len(current_swings)):
            swing_index, swing_type, swing_price = current_swings[i]

            if swing_type != 'low':
                continue

            # Find the most recent swing high before this swing low
            previous_swing_high = None
            for j in range(i - 1, -1, -1):
                if current_swings[j][1] == 'high':
                    previous_swing_high = current_swings[j]
                    break

            if not previous_swing_high:
                continue

            swing_high_index = previous_swing_high[0]
            swing_high_price = previous_swing_high[2]

            # Check if this range has been invalidated by swing high being taken out
            range_invalidated = False

            # First check: Has current price broken above the swing high?
            if current_high > swing_high_price:
                print(f"SHORT range INVALIDATED: Current price ${current_high:.2f} broke above swing high ${swing_high_price:.2f}")
                range_invalidated = True

            # Second check: Look for any swing high after our swing high that broke above it
            if not range_invalidated:
                for later_swing in current_swings:
                    later_index, later_type, later_price = later_swing
                    if (later_type == 'high' and
                        later_index > swing_high_index and  # Came after our swing high
                        later_price > swing_high_price):  # Broke above our swing high
                        print(f"SHORT range INVALIDATED: Swing high ${swing_high_price:.2f} taken out by later high ${later_price:.2f} at index {later_index}")
                        range_invalidated = True
                        break

            if range_invalidated:
                continue

            # Check if current swing low broke below a previous swing low (showing weakness)
            range_validated = False
            if swing_index > swing_high_index:  # Current swing low came after the swing high
                # Look for any previous swing low that this swing broke below
                for prev_swing in current_swings:
                    prev_index, prev_type, prev_price = prev_swing
                    if (prev_type == 'low' and
                        prev_index < swing_high_index and  # Previous low before our swing high
                        swing_price < prev_price):  # Current swing low broke below it
                        print(f"SHORT range VALIDATED: Current swing low ${swing_price:.2f} broke below previous low ${prev_price:.2f}")
                        range_validated = True
                        break

            if not range_validated:
                print(f"SHORT range REJECTED: Current swing low ${swing_price:.2f} did not break any previous lows")
                continue

            # Check minimum range width - calculate as percentage move from high to low
            range_width_pct = (swing_high_price - swing_price) / swing_high_price
            print(f"SHORT range check: High ${swing_high_price:.2f} -> Low ${swing_price:.2f} = {range_width_pct:.1%} (min: {config.MIN_RANGE_WIDTH:.1%})")
            if range_width_pct < config.MIN_RANGE_WIDTH:
                print(f"  -> REJECTED: Range too small")
                continue

            # Get order block (candle before the swing high)
            order_block_index = swing_high_index - 1
            if order_block_index < 0:
                continue

            order_block_high = ohlc_data['high'][order_block_index]
            order_block_low = ohlc_data['low'][order_block_index]

            # Validate order block constraints
            range_size = swing_high_price - swing_price
            order_block_size = order_block_high - order_block_low
            max_order_block_size = range_size / 3  # Maximum 1/3 of range

            print(f"    Order block: ${order_block_low:.2f}-${order_block_high:.2f} (size: ${order_block_size:.2f})")
            print(f"    Range: ${swing_high_price:.2f}-${swing_price:.2f} (size: ${range_size:.2f})")
            print(f"    Max order block size (1/3): ${max_order_block_size:.2f}")

            # Check if order block is completely contained within range
            if order_block_low < swing_price or order_block_high > swing_high_price:
                print(f"    -> REJECTED: Order block not contained within range")
                continue

            # Check if order block is too large (more than 1/3 of range)
            if order_block_size > max_order_block_size:
                print(f"    -> REJECTED: Order block too large ({order_block_size:.2f} > {max_order_block_size:.2f})")
                continue

            # Enter SHORT when price retraces within the order block (but not above swing high)
            if (current_high >= order_block_low and   # Price touches order block low
                current_high <= order_block_high and  # Price stays within order block (not above)
                current_high < swing_high_price):     # Price hasn't broken swing high
                print(f"    SHORT ENTRY TRIGGER: Current high ${current_high:.2f} within order block ${order_block_low:.2f}-${order_block_high:.2f}, swing high ${swing_high_price:.2f} unbreached (index {current_index})")
                entry_price = order_block_low  # Enter at order block low
                stop_loss = order_block_high  # Stop at order block high
                take_profit = swing_price - (swing_high_price - swing_price) * 0.1  # Target below swing low

                risk = abs(stop_loss - entry_price)  # Distance from entry to stop loss
                reward = abs(entry_price - take_profit)  # Distance from entry to take profit

                if risk <= 0:
                    continue

                rr_ratio = reward / risk  # R:R = Reward : Risk

                if rr_ratio >= self.min_rr_ratio:
                    print(f"  -> SHORT TRADE TRIGGERED! Entry at index {current_index}, Range: {range_width_pct:.1%}, R:R: {rr_ratio:.1f}")
                    return {
                        'type': 'short',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk': risk,
                        'reward': reward,
                        'rr_ratio': rr_ratio,
                        'swing_high': previous_swing_high,
                        'swing_low': (swing_index, swing_type, swing_price),
                        'order_block': {'high': order_block_high, 'low': order_block_low, 'index': order_block_index}
                    }

        return None

    def check_exit_signals(self, position: Dict, current_high: float,
                          current_low: float, current_price: float) -> Optional[str]:
        """
        Check for exit signals (stop loss, partial exit at configurable R:R, or take profit hit)

        Args:
            position: Current position dictionary
            current_high: Current bar high
            current_low: Current bar low
            current_price: Current close price

        Returns:
            'stop_loss', 'take_profit', 'partial_exit', or None
        """
        if not position:
            return None

        position_type = position['type']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        entry_price = position['entry_price']

        # Check for partial exit first (if enabled and not already taken)
        if (config.PARTIAL_EXIT_RR and
            not position.get('partial_taken', False)):

            risk = abs(entry_price - stop_loss)
            partial_target_distance = risk * config.PARTIAL_EXIT_RR

            if position_type == 'long':
                partial_target = entry_price + partial_target_distance
                if current_high >= partial_target:
                    return 'partial_exit'
            elif position_type == 'short':
                partial_target = entry_price - partial_target_distance
                if current_low <= partial_target:
                    return 'partial_exit'

        if position_type == 'long':
            # Long position exits
            if current_low <= stop_loss:
                return 'stop_loss'
            elif current_high >= take_profit:
                return 'take_profit'

        elif position_type == 'short':
            # Short position exits
            if current_high >= stop_loss:
                return 'stop_loss'
            elif current_low <= take_profit:
                return 'take_profit'

        return None

    def calculate_position_size(self, entry_price: float, stop_loss: float,
                              portfolio_value: float, risk_percent: float = None) -> float:
        """
        Calculate position size based on risk per trade (risk-based position sizing)

        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price for the trade
            portfolio_value: Current portfolio value
            risk_percent: Percentage of portfolio to risk (default from config)

        Returns:
            Number of shares/units to trade
        """
        risk_pct = risk_percent or config.RISK_PERCENT

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0 or entry_price <= 0 or portfolio_value <= 0:
            return 0

        # Calculate maximum dollar amount to risk
        max_risk_dollars = portfolio_value * risk_pct

        # Calculate position size based on risk
        shares = max_risk_dollars / risk_per_share

        return shares