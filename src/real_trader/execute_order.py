#!/usr/bin/env python3
"""
Execute a specific order on Polymarket.
"""

import asyncio
import requests
import json
from auth import PolyAuth
from order_manager import OrderManager
from balance_manager import BalanceManager
from models import OrderRequest, Side, OrderType


async def main():
    print("=" * 70)
    print("EXECUTING POLYMARKET ORDER")
    print("=" * 70)

    # Initialize
    print("\n[1] Initializing...")
    auth = PolyAuth()
    balance_mgr = BalanceManager(auth)
    await balance_mgr.initialize()
    order_mgr = OrderManager(auth, balance_mgr)

    # Target parameters
    event_slug = "elon-musk-of-tweets-february-13-february-20"
    range_label = "300-319"
    target_value = 1.00  # $1

    print(f"\n[2] Finding market: {event_slug}")
    print(f"    Range: {range_label}")

    # Get event details
    try:
        response = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={
                "limit": 100,
                "active": "true",
                "closed": "false",
                "archived": "false",
                "order": "volume24hr",
                "ascending": "false"
            },
            timeout=10
        )

        if response.status_code != 200:
            print(f"❌ Failed to fetch events: {response.status_code}")
            return

        events = response.json()
        target_event = None

        for event in events:
            if event.get("slug") == event_slug:
                target_event = event
                break

        if not target_event:
            print(f"❌ Event not found: {event_slug}")
            return

        print(f"✅ Found event: {target_event['title']}")

        # Find the specific market (300-319 range)
        markets = target_event.get("markets", [])
        target_market = None

        for market in markets:
            question = market.get("question", "")
            if "300" in question and "319" in question:
                target_market = market
                break

        if not target_market:
            print(f"❌ Market not found for range: {range_label}")
            print(f"Available markets: {len(markets)}")
            for m in markets[:5]:
                print(f"  • {m.get('question', 'N/A')}")
            return

        print(f"✅ Found market: {target_market['question']}")

        # Get token ID for YES
        clob_token_ids = target_market.get("clobTokenIds", [])

        # Parse if it's a JSON string
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except:
                print(f"❌ Failed to parse clobTokenIds")
                return

        if not clob_token_ids or len(clob_token_ids) < 1:
            print("❌ No token IDs found")
            return

        # In Polymarket, first token is typically YES
        token_id_decimal = clob_token_ids[0]

        # Convert decimal to hex format
        try:
            token_id_int = int(token_id_decimal)
            token_id = f"0x{hex(token_id_int)[2:].zfill(64)}"
            print(f"✅ Token ID (YES): {token_id}")
        except:
            print(f"❌ Failed to convert token ID: {token_id_decimal}")
            return

        # Get current price using ClobClient
        print("\n[3] Getting current market price...")
        try:
            client = auth.get_client()
            price_data = client.get_price(token_id, "BUY")  # BUY side = asking price

            ask = float(price_data.get("price", 0))

            # Get bid price (SELL side)
            bid_data = client.get_price(token_id, "SELL")
            bid = float(bid_data.get("price", 0))
        except Exception as e:
            print(f"❌ Failed to get price: {e}")
            return

        if ask == 0:
            print(f"❌ Market not active (ask = 0)")
            return

        print(f"✅ Current prices:")
        print(f"   Bid: {bid*100:.2f}¢ (${bid:.4f})")
        print(f"   Ask: {ask*100:.2f}¢ (${ask:.4f})")

        # Calculate size for ~$1
        size = target_value / ask
        actual_value = size * ask

        print(f"\n[4] Order calculation:")
        print(f"   Target: ${target_value:.2f}")
        print(f"   Price: {ask*100:.2f}¢ (${ask:.4f})")
        print(f"   Size: {size:.2f} shares")
        print(f"   Actual value: ${actual_value:.2f}")

        # Check balance
        available = await balance_mgr.get_available_balance()
        print(f"\n[5] Balance check:")
        print(f"   Available: ${available:.2f}")
        print(f"   Required: ${actual_value:.2f}")

        if actual_value > available:
            print(f"❌ Insufficient balance")
            return

        # Confirm
        print("\n" + "=" * 70)
        print("ORDER SUMMARY")
        print("=" * 70)
        print(f"Event: {target_event['title']}")
        print(f"Market: {target_market['question']}")
        print(f"Token: YES (BUY)")
        print(f"Range: {range_label}")
        print(f"Price: {ask*100:.2f}¢ (with 2% FOK slippage buffer)")
        print(f"Size: {size:.2f} shares")
        print(f"Total: ${actual_value:.2f} USDC")
        print("=" * 70)

        confirm = input("\nType 'YES' to execute order: ").strip()
        if confirm != "YES":
            print("\n❌ Order cancelled")
            return

        # Execute order
        print("\n[6] Placing order...")

        order_request = OrderRequest(
            token_id=token_id,
            price=ask,
            size=size,
            side=Side.BUY,
            order_type=OrderType.FOK,
            event_slug=event_slug,
            range_label=range_label,
            market_title=target_event['title'],
            token_side="YES"
        )

        result = await order_mgr.place_order(order_request)

        print("\n" + "=" * 70)
        if result.success:
            print("✅ ORDER EXECUTED SUCCESSFULLY!")
            print("=" * 70)
            print(f"Order ID: {result.order_id}")
            print(f"Side: BUY YES")
            print(f"Price: {ask*100:.2f}¢ (+ 2% slippage)")
            print(f"Size: {size:.2f} shares")
            print(f"Value: ${actual_value:.2f} USDC")
        else:
            print("❌ ORDER FAILED!")
            print("=" * 70)
            print(f"Error: {result.error}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
