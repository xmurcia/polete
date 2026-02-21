#!/usr/bin/env python3
"""
Manual SELL test script for real trading.
Executes a SELL order for a specific token_id to test decimal precision.
"""

import asyncio
import sys
from src.real_trader.auth import PolyAuth
from src.real_trader.balance_manager import BalanceManager
from src.real_trader.order_manager import OrderManager
from src.real_trader.position_tracker import PositionTracker
from src.real_trader.models import OrderRequest, Side, OrderType


async def execute_test_sell(token_id: str, price: float):
    """
    Execute a test SELL order for the given token_id.

    Args:
        token_id: The token ID to sell
        price: The bid price to sell at
    """
    print("\n" + "="*60)
    print("REAL SELL ORDER TEST")
    print("="*60)
    print(f"Token ID: {token_id}")
    print(f"Price: {price}")
    print("="*60 + "\n")

    # Initialize trading components
    print("🔧 Initializing trading components...")
    auth = PolyAuth()
    balance_mgr = BalanceManager(auth)
    order_mgr = OrderManager(auth, balance_mgr)
    position_tracker = PositionTracker(auth)

    await balance_mgr.initialize()
    await position_tracker.sync_positions()

    # Find the position
    print(f"\n🔍 Looking for position with token_id: {token_id[:20]}...")
    positions = position_tracker.get_positions()

    target_position = None
    for pos in positions:
        if pos.token_id == token_id:
            target_position = pos
            break

    if not target_position:
        print(f"\n❌ Position not found!")
        print(f"\nAvailable positions:")
        for pos in positions:
            print(f"  - Token: {pos.token_id[:20]}... | Size: {pos.size} | Price: {pos.avg_entry_price}")
        return False

    print(f"\n✅ Position found:")
    print(f"  Event: {target_position.event_slug}")
    print(f"  Range: {target_position.range_label}")
    print(f"  Size: {target_position.size} shares")
    print(f"  Entry Price: ${target_position.avg_entry_price:.3f}")
    print(f"  Current Price: ${target_position.current_price:.3f}")
    print(f"  Unrealized P&L: ${target_position.unrealized_pnl:.2f}")

    # Confirm
    print(f"\n⚠️  WARNING: This will execute a REAL SELL order on Polymarket!")
    print(f"  Selling: {target_position.size} shares @ ${price:.3f}")
    print(f"  Expected proceeds: ${target_position.size * price:.2f}")

    confirm = input("\nType 'YES' to confirm: ")
    if confirm != "YES":
        print("\n❌ Test cancelled.")
        return False

    # Create SELL order
    print(f"\n📤 Creating SELL order...")
    order_request = OrderRequest(
        token_id=token_id,
        price=price,
        size=target_position.size,
        side=Side.SELL,
        order_type=OrderType.FOK,
        event_slug=target_position.event_slug,
        range_label=target_position.range_label,
        market_title=target_position.event_slug,
        token_side=target_position.token_side
    )

    # Execute order
    result = await order_mgr.place_order(order_request)

    if result.success:
        print(f"\n✅ SELL ORDER SUCCESSFUL!")
        print(f"  Order ID: {result.order_id}")

        # Calculate P&L
        revenue = target_position.size * price
        cost = target_position.size * target_position.avg_entry_price
        profit = revenue - cost

        print(f"\n💰 Trade Summary:")
        print(f"  Revenue: ${revenue:.2f}")
        print(f"  Cost: ${cost:.2f}")
        print(f"  Profit: ${profit:.2f} ({(profit/cost)*100:.2f}%)")

        return True
    else:
        print(f"\n❌ SELL ORDER FAILED!")
        print(f"  Error: {result.error}")
        return False


async def main():
    if len(sys.argv) < 3:
        print("\nUsage: python test_real_sell.py <token_id> <price>")
        print("\nExample:")
        print("  python test_real_sell.py 20238471253322408675... 0.158")
        print("\nTo find your token_id, check the logs or portfolio.")
        sys.exit(1)

    token_id = sys.argv[1]
    price = float(sys.argv[2])

    success = await execute_test_sell(token_id, price)

    if success:
        print("\n" + "="*60)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
