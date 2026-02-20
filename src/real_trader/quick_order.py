#!/usr/bin/env python3
"""Quick order execution with known parameters."""

import asyncio
from auth import PolyAuth
from order_manager import OrderManager
from balance_manager import BalanceManager
from models import OrderRequest, Side, OrderType


async def main():
    print("=" * 70)
    print("QUICK ORDER EXECUTION")
    print("=" * 70)

    # Known parameters from screenshot
    token_id = "0x37fae7dcd8854b7f1b813463b0038e3aa42134840184209ed1d3e931f654686e"
    price = 0.20  # 20¢
    target_value = 1.00  # $1
    size = target_value / price  # 5 shares

    print(f"\nOrder details:")
    print(f"  Event: Elon Musk # tweets February 13 - February 20, 2026")
    print(f"  Range: 300-319")
    print(f"  Token: YES")
    print(f"  Token ID: {token_id}")
    print(f"  Price: {price*100:.0f}¢ (${price:.2f})")
    print(f"  Size: {size:.0f} shares")
    print(f"  Total: ${size * price:.2f}")

    # Initialize
    print("\n[1] Initializing...")
    auth = PolyAuth()
    balance_mgr = BalanceManager(auth)
    await balance_mgr.initialize()
    order_mgr = OrderManager(auth, balance_mgr)

    # Check balance
    available = await balance_mgr.get_available_balance()
    print(f"\n[2] Balance: ${available:.2f} USDC")

    if size * price > available:
        print("❌ Insufficient balance")
        return

    # Confirm
    print("\n" + "=" * 70)
    print("CONFIRM ORDER")
    print("=" * 70)
    print(f"BUY YES 300-319 @ {price*100:.0f}¢")
    print(f"Size: {size:.0f} shares = ${size * price:.2f} USDC")
    print(f"Order type: FOK (Fill or Kill)")
    print("=" * 70)

    confirm = input("\nType 'YES' to execute: ").strip()
    if confirm != "YES":
        print("\n❌ Cancelled")
        return

    # Execute
    print("\n[3] Placing order...")

    order_request = OrderRequest(
        token_id=token_id,
        price=price,
        size=size,
        side=Side.BUY,
        order_type=OrderType.FOK,
        event_slug="elon-musk-of-tweets-february-13-february-20",
        range_label="300-319",
        market_title="Elon Musk # tweets February 13 - February 20, 2026",
        token_side="YES"
    )

    result = await order_mgr.place_order(order_request)

    print("\n" + "=" * 70)
    if result.success:
        print("✅ ORDER EXECUTED!")
        print("=" * 70)
        print(f"Order ID: {result.order_id}")
        print(f"Filled: {size:.0f} shares @ ~{price*100:.0f}¢")
        print(f"Total: ${size * price:.2f} USDC")
    else:
        print("❌ ORDER FAILED")
        print("=" * 70)
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
