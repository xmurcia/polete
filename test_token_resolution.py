"""
Test script to verify token_id resolution is working correctly
Tests that bucket 65-89 doesn't match 165-189 (and vice versa)
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.real_trader import UnifiedTrader

async def test_token_resolution():
    """Test token resolution for buckets that could have false matches"""

    print("\n🧪 Testing Token Resolution Logic")
    print("=" * 80)

    trader = UnifiedTrader(use_real=True)

    # Test cases: buckets that could have false positive matches
    test_cases = [
        # (market_title, bucket, description)
        ("Elon Musk # tweets February 21 - February 23, 2026?", "65-89", "Should NOT match 165-189"),
        ("Elon Musk # tweets February 21 - February 23, 2026?", "40-64", "Should NOT match 140-164"),
        ("Elon Musk # tweets February 17 - February 24, 2026?", "280-299", "Should NOT match 80-299 or 280-99"),
        ("Elon Musk # tweets February 17 - February 24, 2026?", "300-319", "Should match exactly"),
    ]

    print("\n📋 Test Cases:")
    for i, (market, bucket, desc) in enumerate(test_cases, 1):
        print(f"  {i}. {bucket:15} - {desc}")

    print("\n🔍 Resolving tokens...\n")

    for market_title, bucket, description in test_cases:
        print(f"Testing: {bucket}")
        print(f"  Market: {market_title}")
        print(f"  Note: {description}")

        try:
            token_id = await trader._resolve_token_id(market_title, bucket, side="YES")

            if token_id:
                print(f"  ✅ Resolved: {token_id[:30]}...")
                print(f"  🔍 Verifying it's the correct token...")

                # Verify by checking the reverse lookup
                import requests
                import json

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

                if response.status_code == 200:
                    events = response.json()

                    # Find the market that contains this token
                    found = False
                    for event in events:
                        if "elon" not in event.get("title", "").lower():
                            continue

                        for market in event.get("markets", []):
                            clob_token_ids = market.get("clobTokenIds", [])
                            if isinstance(clob_token_ids, str):
                                clob_token_ids = json.loads(clob_token_ids)

                            if str(token_id) in [str(t) for t in clob_token_ids]:
                                question = market.get("question", "")
                                print(f"  ✅ Verified: {question[:80]}")

                                # Check if bucket is actually in the question
                                if bucket.lower() in question.lower():
                                    print(f"  ✅ CORRECT MATCH: '{bucket}' found in question")
                                else:
                                    print(f"  ❌ WRONG MATCH: '{bucket}' NOT in question!")

                                found = True
                                break

                        if found:
                            break

                    if not found:
                        print(f"  ⚠️  Could not verify token in API response")

            else:
                print(f"  ❌ Failed to resolve token")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()

    print("=" * 80)
    print("🏁 Test Complete\n")

if __name__ == "__main__":
    asyncio.run(test_token_resolution())
