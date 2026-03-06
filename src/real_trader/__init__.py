"""
Polymarket Real Trading Module

Complete suite for trading on Polymarket CLOB.
Includes authentication, order management, position tracking, and balance management.
"""

from .auth import PolyAuth
from .order_manager import OrderManager
from .position_tracker import PositionTracker
from .balance_manager import BalanceManager
from .unified_trader import UnifiedTrader
from .models import (
    Side,
    OrderType,
    OrderRequest,
    OrderResult,
    TrackedOrder,
    Position,
    BalanceInfo,
    MarketData
)

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "PolyAuth",
    "OrderManager",
    "PositionTracker",
    "BalanceManager",
    "UnifiedTrader",

    # Models
    "Side",
    "OrderType",
    "OrderRequest",
    "OrderResult",
    "TrackedOrder",
    "Position",
    "BalanceInfo",
    "MarketData",
]
