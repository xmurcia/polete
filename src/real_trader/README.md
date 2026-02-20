# Polymarket Real Trading Module

Complete Python suite for real trading on Polymarket CLOB.

## Overview

This module provides everything needed for live trading on Polymarket:
- **Authentication** (L1 → L2 credential flow with caching)
- **Order Management** (place, cancel, track orders)
- **Position Tracking** (sync positions, calculate P&L, manage stops)
- **Balance Management** (track capital, exposure, risk limits)

## Installation

```bash
cd real_trader
./install.sh
```

Configure `.env`:
```bash
PRIVATE_KEY=your_ethereum_private_key
WALLET_ADDRESS=your_polymarket_proxy_wallet
MAX_POSITIONS_PER_EVENT=6
MAX_EXPOSURE=0.99
MAX_DAILY_LOSS=30
```

## Quick Start

```python
from real_trader import (
    PolyAuth,
    OrderManager,
    PositionTracker,
    BalanceManager,
    OrderRequest,
    Side,
    OrderType
)

# Initialize
auth = PolyAuth()
balance_mgr = BalanceManager(auth)
position_tracker = PositionTracker(auth)
order_mgr = OrderManager(auth, balance_mgr)

await balance_mgr.initialize()
await position_tracker.sync_positions()

# Place order
order = OrderRequest(
    token_id="0x123...",
    price=0.50,
    size=10,
    side=Side.BUY,
    order_type=OrderType.FOK,
    event_slug="elon-tweets",
    range_label="10-15"
)

result = await order_mgr.place_order(order)
```

## Core Classes

### PolyAuth
Handles authentication with Polymarket CLOB.
- L1 → L2 credential derivation
- Automatic credential caching (24h)
- Returns authenticated `ClobClient`

```python
auth = PolyAuth()
client = auth.get_client()
wallet = auth.get_wallet_address()
```

### OrderManager
Manages order placement and tracking.
- Place orders (GTC, FOK, GTD)
- Auto-validates against position limits
- Tracks orders by event
- Syncs with blockchain

```python
order_mgr = OrderManager(auth, balance_mgr)
result = await order_mgr.place_order(order_request)
orders = order_mgr.get_open_orders()
await order_mgr.cancel_order(order_id)
```

### PositionTracker
Tracks positions and calculates P&L.
- Syncs positions from Polymarket API
- Updates current prices
- Manages stop loss / take profit
- Tracks entered ranges

```python
tracker = PositionTracker(auth)
await tracker.sync_positions()
await tracker.update_current_prices()

positions = tracker.get_positions()
pnl = tracker.get_total_unrealized_pnl()
```

### BalanceManager
Manages capital and risk limits.
- Fetches USDC balance
- Validates order capacity
- Tracks exposure and P&L
- Enforces daily loss limits

```python
balance_mgr = BalanceManager(auth)
await balance_mgr.initialize()

available = await balance_mgr.get_available_balance()
can_trade = await balance_mgr.can_place_order(order_value)
```

## Models

### OrderRequest
```python
@dataclass
class OrderRequest:
    token_id: str
    price: float
    size: float
    side: Side  # BUY or SELL
    order_type: OrderType  # GTC, FOK, GTD
    event_slug: str
    range_label: str
    market_title: str
    token_side: str  # "YES" or "NO"
```

### Position
```python
@dataclass
class Position:
    token_id: str
    event_slug: str
    range_label: str
    side: Side
    size: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    # Stop loss / Take profit fields
    peak_price: Optional[float]
    use_trailing_stop: bool
    trailing_stop_percent: Optional[float]
    fixed_stop_price: Optional[float]
    take_profit_price: Optional[float]
```

## Complete Example

See `example_bot.py` for a full trading bot implementation.

```bash
./venv/bin/python example_bot.py
```

## Integration with Paper Trading

This module is **100% real trading**. For paper trading, wrap calls in your main bot:

```python
if PAPER_MODE:
    # Use your PaperTrader class
    paper_trader.place_order(...)
else:
    # Use real_trader module
    result = await order_manager.place_order(...)
```

## Requirements

- Python 3.8+
- POL for gas (~0.01 POL minimum)
- USDC in proxy wallet
- Approved allowance (run `approve_usdc_blockchain.py` once)

## Utilities

### Check Wallet Status
```bash
./venv/bin/python check_both_wallets.py
```

Shows USDC, POL, and allowance for both EOA and Proxy wallets.

### Approve Allowance (One-time)
```bash
./venv/bin/python approve_usdc_blockchain.py
```

Approves unlimited USDC spending for Polymarket CLOB.

## Architecture

```
Bot (elon_auto_bot_threads.py)
  ├─ if PAPER_MODE:
  │    └─ PaperTrader (existing class)
  └─ else:
       └─ real_trader module
            ├─ PolyAuth (authentication)
            ├─ OrderManager (orders)
            ├─ PositionTracker (positions)
            └─ BalanceManager (capital)
```

## Error Handling

All methods handle errors gracefully:
- OrderManager returns `OrderResult(success=False, error="...")`
- Position/Balance methods log errors and continue
- Auth failures raise clear exceptions

## Security

- Private key only used for L1 signing
- L2 credentials cached (24h rotation)
- All sensitive files gitignored
- No paper mode = no simulated balance confusion

## API Reference

Full documentation in code docstrings. Key methods:

**OrderManager:**
- `place_order(request) -> OrderResult`
- `cancel_order(order_id) -> bool`
- `get_open_orders() -> List[TrackedOrder]`
- `sync_open_orders() -> None`

**PositionTracker:**
- `sync_positions() -> None`
- `update_current_prices() -> None`
- `get_positions() -> List[Position]`
- `get_total_unrealized_pnl() -> float`

**BalanceManager:**
- `initialize() -> None`
- `get_available_balance() -> float`
- `can_place_order(amount) -> bool`
- `log_status(...) -> None`
