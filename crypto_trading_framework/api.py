"""
Quantuis compatibility shim for tests and legacy import paths.
"""

from crypto_trading_framework.core.api import BotController, create_app
from crypto_trading_framework.core.bot import AutomatedTradingBot

__all__ = ["BotController", "create_app", "AutomatedTradingBot"]