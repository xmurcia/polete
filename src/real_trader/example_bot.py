#!/usr/bin/env python3
"""
Example trading bot using the real_trader module.
Demonstrates full trading lifecycle: auth, orders, positions, balance.
"""

import asyncio
from src.real_trader import (
    PolyAuth,
    OrderManager,
    PositionTracker,
    BalanceManager,
    OrderRequest,
    Side,
    OrderType
)


async def main():
    print("=" * 70)
    print("Polymarket Trading Bot Example")
    print("=" * 70)

    # 1. Initialize authentication
    print("\n[1] Initializing authentication...")
    auth = PolyAuth()
    client = auth.get_client()
    print(f"✅ Authenticated: {auth.get_wallet_address()}")

    # 2. Initialize managers
    print("\n[2] Initializing managers...")

    balance_manager = BalanceManager(auth)
    await balance_manager.initialize()

    position_tracker = PositionTracker(auth)
    await position_tracker.sync_positions()

    order_manager = OrderManager(auth, balance_manager)
    await order_manager.sync_open_orders()

    print("✅ All managers initialized")

    # 3. Check current status
    print("\n[3] Current Status")
    print("=" * 70)

    available = await balance_manager.get_available_balance()
    positions = position_tracker.get_positions()
    orders = order_manager.get_open_orders()

    print(f"💰 Available Balance: ${available:.2f}")
    print(f"📊 Open Positions: {len(positions)}")
    print(f"📋 Open Orders: {len(orders)}")

    if positions:
        print("\nPositions:")
        for pos in positions[:5]:  # Show first 5
            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            print(f"  • {pos.range_label}: {pos.size:.2f} shares @ "
                  f"{pos.avg_entry_price*100:.2f}¢ "
                  f"(P&L: {pnl_sign}${pos.unrealized_pnl:.2f})")

    # 4. Example: Place an order (commented out - uncomment to test)
    """
    print("\n[4] Placing example order...")

    order_request = OrderRequest(
        token_id="your_token_id_here",
        price=0.50,  # 50 cents
        size=10,     # 10 shares
        side=Side.BUY,
        order_type=OrderType.FOK,
        event_slug="elon-musk-tweets",
        range_label="10-15 tweets",
        market_title="Elon Musk Tweet Count",
        token_side="YES"
    )

    result = await order_manager.place_order(order_request)

    if result.success:
        print(f"✅ Order placed: {result.order_id}")
    else:
        print(f"❌ Order failed: {result.error}")
    """

    # 5. Update prices and calculate P&L
    print("\n[5] Updating current prices...")
    await position_tracker.update_current_prices()

    total_pnl = position_tracker.get_total_unrealized_pnl()
    print(f"✅ Total Unrealized P&L: ${total_pnl:.2f}")

    # 6. Log final status
    print("\n[6] Final Status")
    print("=" * 70)

    total_position_value = sum(
        p.avg_entry_price * p.size
        for p in positions
    )

    await balance_manager.log_status(
        total_position_value=total_position_value,
        unrealized_pnl=total_pnl,
        position_count=len(positions)
    )

    print("\n✅ Example completed!")
    print("\nNext steps:")
    print("  1. Modify this script for your trading strategy")
    print("  2. Add your order placement logic")
    print("  3. Implement stop loss / take profit")
    print("  4. Run in production with PAPER_MODE=false")


if __name__ == "__main__":
    asyncio.run(main())
