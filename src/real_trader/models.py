"""
Data models for Polymarket trading.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Side(str, Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type"""
    GTC = "GTC"  # Good Till Canceled
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


@dataclass
class OrderRequest:
    """Request to place an order"""
    token_id: str
    price: float
    size: float
    side: Side
    order_type: OrderType = OrderType.FOK
    event_slug: str = ""
    range_label: str = ""
    market_title: str = ""
    token_side: str = ""  # "YES" or "NO"
    is_stop_loss: bool = False


@dataclass
class OrderResult:
    """Result of placing an order"""
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TrackedOrder:
    """Order being tracked"""
    order_id: str
    token_id: str
    event_slug: str
    range_label: str
    side: Side
    order_type: OrderType
    price: float
    size: float
    timestamp: int
    market_title: str = ""
    token_side: str = ""


@dataclass
class Position:
    """Trading position"""
    token_id: str
    event_slug: str
    range_label: str
    side: Side
    size: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: int

    # Stop loss / Take profit
    peak_price: Optional[float] = None
    use_trailing_stop: bool = False
    trailing_stop_percent: Optional[float] = None
    fixed_stop_price: Optional[float] = None
    fixed_stop_loss_percent: Optional[float] = None
    take_profit_price: Optional[float] = None
    take_profit_percent: Optional[float] = None

    # Metadata
    market_title: str = ""
    token_side: str = ""  # "YES" or "NO"
    is_lottery_ticket: bool = False
    partial_take_profit_executed: bool = False


@dataclass
class BalanceInfo:
    """Wallet balance information"""
    usdc_balance: float
    usdc_allowance: float
    pol_balance: float
    in_positions: float
    available: float
    unrealized_pnl: float
    total_exposure: float


@dataclass
class MarketData:
    """Market data from Polymarket"""
    condition_id: str
    question: str
    end_date: str
    tokens: list  # List of token info
    active: bool = True
