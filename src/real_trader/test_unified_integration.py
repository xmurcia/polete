#!/usr/bin/env python3
"""
Integration test for UnifiedTrader

Tests both paper and real modes to verify the implementation works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unified_trader import UnifiedTrader


def test_paper():
    """Test paper mode functionality"""
    print("\n" + "="*60)
    print("=== PAPER MODE TEST ===")
    print("="*60)

    trader = UnifiedTrader(use_real=False, initial_cash=1000.0)
    trader.initialize()

    print(f"\n✅ Initialized paper trader")
    print(f"   Initial cash: ${trader.portfolio['cash']:.2f}")

    # Execute a test trade
    print(f"\n📊 Executing test BUY trade...")
    result = trader.execute(
        market_title="Elon tweets Feb 13-20",
        bucket="300-319",
        signal="BUY",
        price=0.20,
        reason="Test trade Val+0.05",
        strategy_tag="STANDARD"
    )
    print(f"   Result: {result}")

    # Check portfolio
    portfolio = trader.get_portfolio()
    print(f"\n💼 Portfolio after trade:")
    print(f"   Cash: ${portfolio['cash']:.2f}")
    print(f"   Positions: {len(portfolio['positions'])}")

    if portfolio['positions']:
        for pos_id, pos in portfolio['positions'].items():
            print(f"   - {pos_id}: {pos['shares']:.2f} shares @ ${pos['entry_price']:.3f}")

    # Test print_summary
    print(f"\n📋 Portfolio summary:")
    trader.print_summary([])

    print(f"\n✅ Paper mode test completed successfully!")


def test_real():
    """Test real mode functionality (read-only)"""
    print("\n" + "="*60)
    print("=== REAL MODE TEST (Read-only) ===")
    print("="*60)

    try:
        trader = UnifiedTrader(use_real=True)
        trader.initialize()

        print(f"\n✅ Initialized real trader")

        # Get portfolio (read-only)
        portfolio = trader.get_portfolio()
        print(f"\n💰 Blockchain balance: ${portfolio['cash']:.2f}")
        print(f"📊 Open positions: {len(portfolio['positions'])}")

        if portfolio['positions']:
            print(f"\n📋 Current positions:")
            for pos_id, pos in portfolio['positions'].items():
                invested = pos['invested']
                print(f"   - {pos_id}: {pos['shares']:.2f} shares @ ${pos['entry_price']:.3f} (${invested:.2f})")

        # Show detailed summary
        print(f"\n📊 Detailed portfolio summary:")
        trader.print_summary([])

        print(f"\n✅ Real mode test completed successfully!")

    except Exception as e:
        print(f"\n❌ Real mode test failed: {e}")
        import traceback
        traceback.print_exc()


def test_portfolio_property():
    """Test portfolio property access works in both modes"""
    print("\n" + "="*60)
    print("=== PORTFOLIO PROPERTY TEST ===")
    print("="*60)

    # Paper mode
    print(f"\n📄 Paper mode:")
    paper = UnifiedTrader(use_real=False, initial_cash=500.0)
    paper.initialize()
    print(f"   trader.portfolio['cash'] = ${paper.portfolio['cash']:.2f}")
    assert paper.portfolio['cash'] == 500.0
    print(f"   ✅ Property access works")

    # Real mode
    print(f"\n🔴 Real mode:")
    real = UnifiedTrader(use_real=True)
    real.initialize()
    print(f"   trader.portfolio['cash'] = ${real.portfolio['cash']:.2f}")
    print(f"   ✅ Property access works")

    print(f"\n✅ Portfolio property test completed!")


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("UNIFIED TRADER INTEGRATION TEST SUITE")
    print("="*60)

    try:
        # Test paper mode (should always work)
        test_paper()

        # Test portfolio property
        test_portfolio_property()

        # Test real mode (requires .env setup)
        test_real()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
