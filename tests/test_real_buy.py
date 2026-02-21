#!/usr/bin/env python3
"""
Manual BUY test script for real trading.
Executes a BUY order to test decimal precision fix.
"""

import asyncio
import sys
from src.real_trader.auth import PolyAuth
from src.real_trader.balance_manager import BalanceManager
from src.real_trader.order_manager import OrderManager
from src.real_trader.models import OrderRequest, Side, OrderType


async def execute_test_buy(token_id: str, price: float, market_title: str, bucket: str, amount_usd: float = 5.0):
    """
    Execute a test BUY order for the given token_id.

    Args:
        token_id: The token ID to buy
        price: The ask price to buy at
        market_title: Market title for display
        bucket: Bucket label for display
        amount_usd: Amount in USD to spend (default: $5)
    """
    print("\n" + "="*60)
    print("REAL BUY ORDER TEST")
    print("="*60)
    print(f"Token ID: {token_id[:20]}...")
    print(f"Market: {market_title}")
    print(f"Bucket: {bucket}")
    print(f"Price: ${price:.3f}")
    print(f"Amount: ${amount_usd:.2f}")
    print("="*60 + "\n")

    # Initialize trading components
    print("🔧 Initializing trading components...")
    auth = PolyAuth()
    balance_mgr = BalanceManager(auth)
    order_mgr = OrderManager(auth, balance_mgr)

    await balance_mgr.initialize()

    # Check balance
    available = await balance_mgr.get_available_balance()
    print(f"\n💰 Available balance: ${available:.2f}")

    if amount_usd > available:
        print(f"\n❌ Insufficient balance! Need ${amount_usd:.2f}, have ${available:.2f}")
        return False

    # Calculate shares and round to reasonable precision
    shares = round(amount_usd / price, 2)  # Round to 2 decimals for simplicity
    print(f"\n📊 Order details:")
    print(f"  Amount: ${amount_usd:.2f}")
    print(f"  Price: ${price:.3f}")
    print(f"  Shares: {shares}")
    print(f"  Price × Shares: ${price * shares:.6f}")

    # Confirm
    print(f"\n⚠️  WARNING: This will execute a REAL BUY order on Polymarket!")
    print(f"  Buying: ~{shares:.2f} shares @ ${price:.3f}")
    print(f"  Total cost: ${amount_usd:.2f}")

    confirm = input("\nType 'YES' to confirm: ")
    if confirm != "YES":
        print("\n❌ Test cancelled.")
        return False

    # Create BUY order
    print(f"\n📤 Creating BUY order...")
    order_request = OrderRequest(
        token_id=token_id,
        price=price,
        size=shares,
        side=Side.BUY,
        order_type=OrderType.FOK,
        event_slug=market_title,
        range_label=bucket,
        market_title=market_title,
        token_side="YES"
    )

    # Execute order
    result = await order_mgr.place_order(order_request)

    if result.success:
        print(f"\n✅ BUY ORDER SUCCESSFUL!")
        print(f"  Order ID: {result.order_id}")
        print(f"\n💰 Trade Summary:")
        print(f"  Spent: ${amount_usd:.2f}")
        print(f"  Shares acquired: ~{shares:.2f}")
        print(f"  Entry price: ${price:.3f}")

        return True
    else:
        print(f"\n❌ BUY ORDER FAILED!")
        print(f"  Error: {result.error}")
        return False


async def main():
    if len(sys.argv) < 3:
        print("\nUsage: python test_real_buy.py <token_id> <price> [amount_usd]")
        print("\nExample:")
        print("  python test_real_buy.py 20514581474220126032... 0.159 5.0")
        print("\nOptional:")
        print("  amount_usd: Amount in USD to spend (default: $5.00)")
        sys.exit(1)

    token_id = sys.argv[1]
    price = float(sys.argv[2])
    amount_usd = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    # Fixed market details (for Feb 17-24 event, 380-399 bucket)
    market_title = "Elon Musk # tweets February 17 - February 24, 2026?"
    bucket = "380-399"

    success = await execute_test_buy(token_id, price, market_title, bucket, amount_usd)

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
