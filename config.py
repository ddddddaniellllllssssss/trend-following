# Strategy Configuration
import os

# Data Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'xrp-usdt-binance-daily.csv')
START_DATE = '2022-01-01'
END_DATE = '2025-09-30'

# Strategy Parameters
SWING_PERIOD = 3  # 3-candle pattern for swing detection
RISK_PERCENT = 0.01  # Risk 1% of account per trade
MIN_RR_RATIO = 2.0  # Minimum risk-reward ratio
MIN_RANGE_WIDTH = 0.05  # Minimum 5% range width
PARTIAL_EXIT_RR = 3.0  # R:R ratio at which to take partial profits (None to disable)

# Initial Portfolio
INITIAL_CAPITAL = 100000  # Starting with $100k

# Trading Parameters
COMMISSION = 0.001  # 0.1% per trade
SLIPPAGE = 0.0005  # 0.05% slippage

# Visualization Parameters
GENERATE_CHARTS = False  # Set to False to skip chart generation