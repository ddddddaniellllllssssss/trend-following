import numpy as np
from typing import List, Dict
import config

class Portfolio:
    """Manages portfolio state, positions, and cash"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # Currently not used as we only hold one position at a time
        self.equity_history = []

    def get_value(self) -> float:
        """Get current total portfolio value"""
        # Since we only hold one position at a time and it's managed externally,
        # portfolio value is just the cash
        return self.cash

    def update_value(self, current_price: float, index: int):
        """Update portfolio value and store in history"""
        current_value = self.get_value()
        self.equity_history.append({
            'index': index,
            'value': current_value,
            'cash': self.cash
        })

    def get_available_capital(self) -> float:
        """Get available capital for new positions"""
        return self.cash

    def can_afford_position(self, shares: float, price: float) -> bool:
        """Check if portfolio can afford a position"""
        cost = shares * price
        commission = cost * config.COMMISSION
        total_cost = cost + commission

        return self.cash >= total_cost

    def calculate_max_position_size(self, price: float, position_size_pct: float = None) -> float:
        """Calculate maximum position size based on available capital"""
        pct = position_size_pct or config.POSITION_SIZE
        available_capital = self.cash * pct

        if price <= 0:
            return 0

        # Account for commission
        max_shares = available_capital / (price * (1 + config.COMMISSION))
        return max_shares

    def get_performance_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        if not self.equity_history:
            return {}

        values = [entry['value'] for entry in self.equity_history]

        total_return = (values[-1] - self.initial_capital) / self.initial_capital
        peak_value = max(values)
        current_drawdown = (values[-1] - peak_value) / peak_value if peak_value > 0 else 0

        # Calculate maximum drawdown
        running_max = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        return {
            'total_return': total_return,
            'current_value': values[-1],
            'peak_value': peak_value,
            'current_drawdown': current_drawdown,
            'max_drawdown': abs(max_drawdown),
            'initial_capital': self.initial_capital
        }

    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.equity_history = []