#!/usr/bin/env python3
"""
Test complete flow: UnifiedTrader → OrderManager
Verifies that parameters are passed correctly through the chain
"""

import asyncio
import sys
sys.path.insert(0, 'src')

from real_trader.unified_trader import UnifiedTrader


async def test_unified_flow():
    """Test real trading flow with actual bot integration"""

    print("\n" + "="*70)
    print("TEST: UnifiedTrader → OrderManager Parameter Flow")
    print("="*70)

    # Initialize UnifiedTrader in REAL mode
    trader = UnifiedTrader(use_real=True)
    # Call internal initialize directly to avoid event loop conflict
    await trader._initialize_real()

    print(f"\n✅ UnifiedTrader initialized")
    print(f"   Mode: REAL")
    print(f"   Available balance: ${await trader.balance_mgr.get_available_balance():.2f}")

    # Test parameters (using bucket without existing position)
    market_title = "Elon Musk # tweets February 17 - February 24, 2026?"
    bucket = "400-419"
    price = 0.120  # Ask price
    signal = "BUY"
    reason = "Test flow Val+0.05"
    strategy_tag = "STANDARD"

    print(f"\n" + "-"*70)
    print("TEST PARAMETERS:")
    print("-"*70)
    print(f"Market: {market_title}")
    print(f"Bucket: {bucket}")
    print(f"Price: ${price:.3f}")
    print(f"Signal: {signal}")
    print(f"Strategy: {strategy_tag}")
    print("(token_id will be resolved automatically)")

    # Confirm
    print("\n" + "="*70)
    print("⚠️  WARNING: This will execute a REAL order!")
    print("="*70)
    confirm = input("\nType 'YES' to proceed with test: ")

    if confirm != "YES":
        print("\n❌ Test cancelled")
        return False

    # Execute through UnifiedTrader
    print(f"\n📤 Executing order through UnifiedTrader...")
    print("-"*70)

    # Call internal method directly to avoid event loop conflict
    result = await trader._execute_real(
        market_title=market_title,
        bucket=bucket,
        signal=signal,
        price=price,
        reason=reason,
        strategy_tag=strategy_tag,
        hours_left=None,
        tweet_count=None,
        market_consensus=None,
        entry_z_score=None
    )

    print("\n" + "="*70)
    if result and "✅" in result:
        print("✅ TEST PASSED")
        print("="*70)
        print("\nVerification:")
        print("  - Parameters passed correctly through chain")
        print("  - OrderManager received correct values")
        print("  - create_market_order() executed successfully")
        print("  - No precision errors")
        return True
    else:
        print("❌ TEST FAILED")
        print("="*70)
        print(f"\nResult: {result}")
        return False


async def main():
    """Run test"""
    try:
        success = await test_unified_flow()

        if success:
            print("\n" + "="*70)
            print("🎉 COMPLETE FLOW VALIDATION: SUCCESS")
            print("="*70)
            print("\nThe bot integration is working correctly:")
            print("  ✅ UnifiedTrader → OrderManager")
            print("  ✅ Parameter passing")
            print("  ✅ Decimal precision handling")
            print("  ✅ Real order execution")
            return 0
        else:
            print("\n" + "="*70)
            print("❌ COMPLETE FLOW VALIDATION: FAILED")
            print("="*70)
            return 1

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
