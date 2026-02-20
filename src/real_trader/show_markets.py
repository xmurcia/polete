#!/usr/bin/env python3
"""
Show available Elon Musk markets and their prices.
"""

import asyncio
import requests
from auth import PolyAuth


def get_elon_markets():
    """Fetch Elon Musk tweet markets"""
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
            print(f"❌ API returned {response.status_code}")
            return []

        events = response.json()

        # Filter for Elon tweet markets
        elon_events = [
            e for e in events
            if "elon" in e.get("title", "").lower()
        ]

        # If no Elon markets, show any active markets
        if not elon_events:
            print("⚠️  No Elon markets found, showing other active markets...")
            return events[:10]  # Return first 10 active markets

        return elon_events

    except Exception as e:
        print(f"❌ Error fetching markets: {e}")
        return []


def get_market_prices(token_ids):
    """Get prices for multiple tokens"""
    try:
        response = requests.post(
            "https://clob.polymarket.com/prices",
            json={"token_ids": token_ids},
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


async def main():
    print("=" * 70)
    print("POLYMARKET - ELON MUSK TWEET MARKETS")
    print("=" * 70)

    # Connect
    print("\n[1] Connecting to Polymarket...")
    auth = PolyAuth()
    print(f"✅ Connected: {auth.get_wallet_address()}")

    # Get markets
    print("\n[2] Fetching Elon Musk tweet markets...")
    events = get_elon_markets()

    if not events:
        print("❌ No markets found")
        return

    print(f"✅ Found {len(events)} Elon tweet markets")

    # Show markets
    print("\n" + "=" * 70)
    print("AVAILABLE MARKETS")
    print("=" * 70)

    for i, event in enumerate(events[:5], 1):  # Show first 5
        print(f"\n{i}. {event['title']}")
        print(f"   Slug: {event['slug']}")
        print(f"   End: {event.get('endDate', 'N/A')}")

        # Get tokens/markets
        markets = event.get("markets", [])
        if markets:
            print(f"   Markets ({len(markets)}):")

            # Get prices for all tokens
            token_ids = [m.get("clobTokenIds", [""])[0] for m in markets if m.get("clobTokenIds")]
            prices = get_market_prices(token_ids)

            for market in markets[:3]:  # Show first 3 markets
                question = market.get("question", "N/A")
                token_id = market.get("clobTokenIds", [""])[0] if market.get("clobTokenIds") else None

                if token_id and token_id in prices:
                    price_data = prices[token_id]
                    bid = float(price_data.get("bid", 0))
                    ask = float(price_data.get("ask", 0))

                    print(f"      • {question}")
                    print(f"        Token: {token_id[:20]}...")
                    print(f"        Bid: {bid*100:.2f}¢ | Ask: {ask*100:.2f}¢")
                else:
                    print(f"      • {question}")
                    if token_id:
                        print(f"        Token: {token_id[:20]}...")

    # Interactive selection
    print("\n" + "=" * 70)
    print("SELECT A MARKET TO TRADE")
    print("=" * 70)

    try:
        event_num = int(input(f"\nSelect event (1-{min(len(events), 5)}): ").strip())
        if event_num < 1 or event_num > min(len(events), 5):
            print("❌ Invalid selection")
            return

        selected_event = events[event_num - 1]
        markets = selected_event.get("markets", [])

        if not markets:
            print("❌ No markets in this event")
            return

        print(f"\nMarkets in '{selected_event['title']}':")
        for i, market in enumerate(markets, 1):
            print(f"{i}. {market.get('question', 'N/A')}")

        market_num = int(input(f"\nSelect market (1-{len(markets)}): ").strip())
        if market_num < 1 or market_num > len(markets):
            print("❌ Invalid selection")
            return

        selected_market = markets[market_num - 1]
        token_id = selected_market.get("clobTokenIds", [""])[0] if selected_market.get("clobTokenIds") else None

        if not token_id:
            print("❌ No token ID available")
            return

        # Get current price
        prices = get_market_prices([token_id])
        if token_id in prices:
            price_data = prices[token_id]
            bid = float(price_data.get("bid", 0))
            ask = float(price_data.get("ask", 0))

            print("\n" + "=" * 70)
            print("SELECTED MARKET")
            print("=" * 70)
            print(f"Event: {selected_event['title']}")
            print(f"Market: {selected_market.get('question')}")
            print(f"Token ID: {token_id}")
            print(f"Current Bid: {bid*100:.2f}¢ (${bid:.2f})")
            print(f"Current Ask: {ask*100:.2f}¢ (${ask:.2f})")

            print("\n💡 To place an order, run:")
            print(f"   ./venv/bin/python test_real_trade.py")
            print(f"\n   Use these parameters:")
            print(f"   Token ID: {token_id}")
            print(f"   Price: {ask:.3f} (to BUY) or {bid:.3f} (to SELL)")
            print(f"   Event: {selected_event['slug']}")

    except (ValueError, KeyboardInterrupt):
        print("\n❌ Cancelled")


if __name__ == "__main__":
    asyncio.run(main())
