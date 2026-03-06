#!/usr/bin/env python3
"""
Test real trade execution on Polymarket.
This script will execute a REAL trade with REAL money.
"""

import asyncio
import sys
from auth import PolyAuth
from order_manager import OrderManager
from position_tracker import PositionTracker
from balance_manager import BalanceManager
from models import OrderRequest, Side, OrderType


async def main():
    print("=" * 70)
    print("POLYMARKET REAL TRADE TEST")
    print("⚠️  WARNING: This will execute a REAL trade with REAL money!")
    print("=" * 70)

    # Initialize
    print("\n[1] Initializing authentication...")
    auth = PolyAuth()
    print(f"✅ Connected: {auth.get_wallet_address()}")

    print("\n[2] Initializing managers...")
    balance_mgr = BalanceManager(auth)
    await balance_mgr.initialize()

    position_tracker = PositionTracker(auth)
    await position_tracker.sync_positions()

    order_mgr = OrderManager(auth, balance_mgr)
    await order_mgr.sync_open_orders()

    print("✅ All managers initialized")

    # Show current status
    print("\n[3] Current Status")
    print("=" * 70)

    available = await balance_mgr.get_available_balance()
    positions = position_tracker.get_positions()
    orders = order_mgr.get_open_orders()

    print(f"💰 Available Balance: ${available:.2f} USDC")
    print(f"📊 Open Positions: {len(positions)}")
    print(f"📋 Open Orders: {len(orders)}")

    if positions:
        print("\n📊 Current Positions:")
        for pos in positions[:5]:
            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            print(f"  • {pos.range_label}: {pos.size:.2f} shares @ "
                  f"{pos.avg_entry_price*100:.2f}¢ "
                  f"(P&L: {pnl_sign}${pos.unrealized_pnl:.2f})")

    # Get user confirmation
    print("\n" + "=" * 70)
    print("TRADE PARAMETERS")
    print("=" * 70)
    print("\n⚠️  We need market data to place a real order.")
    print("Please provide the following information:\n")

    # Ask for trade details
    token_id = input("Token ID (0x...): ").strip()
    if not token_id:
        print("❌ Token ID required")
        return

    price_str = input("Price (e.g., 0.50 for 50¢): ").strip()
    try:
        price = float(price_str)
        if price <= 0 or price >= 1:
            print("❌ Price must be between 0 and 1")
            return
    except ValueError:
        print("❌ Invalid price")
        return

    size_str = input("Size (number of shares, e.g., 10): ").strip()
    try:
        size = float(size_str)
        if size <= 0:
            print("❌ Size must be positive")
            return
    except ValueError:
        print("❌ Invalid size")
        return

    side_str = input("Side (BUY or SELL): ").strip().upper()
    if side_str not in ["BUY", "SELL"]:
        print("❌ Side must be BUY or SELL")
        return
    side = Side.BUY if side_str == "BUY" else Side.SELL

    token_side_str = input("Token side (YES or NO): ").strip().upper()
    if token_side_str not in ["YES", "NO"]:
        print("❌ Token side must be YES or NO")
        return

    event_slug = input("Event slug (e.g., elon-tweets): ").strip() or "test-event"
    range_label = input("Range label (e.g., 10-15): ").strip() or "test-range"

    # Ask for order type
    print("\nOrder types:")
    print("  FOK (Fill or Kill) - Execute immediately or cancel")
    print("  GTC (Good Till Cancelled) - Leave order in book until filled")
    order_type_str = input("Order type (FOK or GTC, default GTC): ").strip().upper() or "GTC"

    if order_type_str == "FOK":
        order_type = OrderType.FOK
    else:
        order_type = OrderType.GTC

    # Calculate order value
    order_value = price * size

    print("\n" + "=" * 70)
    print("ORDER SUMMARY")
    print("=" * 70)
    print(f"Token ID: {token_id[:20]}...")
    print(f"Side: {side} {token_side_str}")
    print(f"Price: {price*100:.2f}¢ (${price:.2f})")
    print(f"Size: {size:.2f} shares")
    print(f"Total Value: ${order_value:.2f} USDC")
    print(f"Event: {event_slug}")
    print(f"Range: {range_label}")
    print(f"Order Type: {order_type.value}")
    print(f"\n💡 You are {side_str}ing the {token_side_str} outcome")

    print("\n⚠️  This will execute a REAL trade on Polymarket blockchain!")
    confirm = input("\nType 'YES' to confirm and place order: ").strip()

    if confirm != "YES":
        print("\n❌ Trade cancelled by user")
        return

    # Place order
    print("\n[4] Placing order...")
    print("=" * 70)

    order_request = OrderRequest(
        token_id=token_id,
        price=price,
        size=size,
        side=side,
        order_type=order_type,
        event_slug=event_slug,
        range_label=range_label,
        market_title="Test Trade",
        token_side=token_side_str
    )

    result = await order_mgr.place_order(order_request)

    print("\n" + "=" * 70)
    if result.success:
        print("✅ ORDER EXECUTED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Order ID: {result.order_id}")
        print(f"Side: {side}")
        print(f"Price: {price*100:.2f}¢")
        print(f"Size: {size:.2f} shares")
        print(f"Value: ${order_value:.2f} USDC")

        print("\n📊 Syncing positions...")
        await position_tracker.sync_positions()

        new_positions = position_tracker.get_positions()
        print(f"✅ New position count: {len(new_positions)}")

        if new_positions:
            print("\nLatest positions:")
            for pos in new_positions[:3]:
                print(f"  • {pos.range_label}: {pos.size:.2f} @ {pos.avg_entry_price*100:.2f}¢")
    else:
        print("❌ ORDER FAILED!")
        print("=" * 70)
        print(f"Error: {result.error}")
        print("\nPossible reasons:")
        print("  - Insufficient balance")
        print("  - Invalid token ID")
        print("  - Market closed")
        print("  - Price out of range")

    # Final status
    print("\n[5] Final Status")
    print("=" * 70)

    final_balance = await balance_mgr.get_available_balance()
    final_positions = position_tracker.get_positions()

    await balance_mgr.log_status(
        total_position_value=sum(p.avg_entry_price * p.size for p in final_positions),
        unrealized_pnl=position_tracker.get_total_unrealized_pnl(),
        position_count=len(final_positions)
    )

    print("\n✅ Test completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
