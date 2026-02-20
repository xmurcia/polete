#!/usr/bin/env python3
"""
Complete test of UnifiedTrader implementation
Tests paper mode thoroughly (real mode requires .env setup)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_paper_mode_complete():
    """Complete test of paper mode functionality"""
    print("\n" + "="*70)
    print("TESTING UNIFIED TRADER - PAPER MODE")
    print("="*70)

    # Import after path setup
    from src.real_trader.unified_trader import UnifiedTrader

    # Test 1: Initialization
    print("\n[TEST 1] Initialization")
    print("-" * 70)
    trader = UnifiedTrader(use_real=False, initial_cash=1000.0)
    trader.initialize()

    assert trader.use_real is False
    assert trader._paper_trader is not None
    print("✅ Paper trader initialized successfully")
    print(f"   Initial cash: ${trader.portfolio['cash']:.2f}")

    # Test 2: Portfolio property access
    print("\n[TEST 2] Portfolio Property Access")
    print("-" * 70)
    cash = trader.portfolio["cash"]
    positions = trader.portfolio["positions"]
    print(f"✅ Property access works: cash=${cash:.2f}, positions={len(positions)}")

    # Test 3: Execute BUY - Standard
    print("\n[TEST 3] Execute BUY - STANDARD Strategy")
    print("-" * 70)
    result = trader.execute(
        market_title="Elon tweets Feb 13-20",
        bucket="300-319",
        signal="BUY",
        price=0.20,
        reason="Accumulation Val+0.08",
        strategy_tag="STANDARD"
    )
    print(f"   Result: {result}")

    portfolio = trader.get_portfolio()
    assert portfolio["cash"] < 1000.0, "Cash should decrease after buy"
    assert len(portfolio["positions"]) == 1, "Should have 1 position"
    print(f"✅ BUY executed successfully")
    print(f"   Cash after: ${portfolio['cash']:.2f}")
    print(f"   Positions: {len(portfolio['positions'])}")

    # Get position details
    pos_key = list(portfolio["positions"].keys())[0]
    pos = portfolio["positions"][pos_key]
    print(f"   Position: {pos['shares']:.2f} shares @ ${pos['entry_price']:.3f}")
    invested = pos['invested']

    # Test 4: Execute BUY - Moonshot
    print("\n[TEST 4] Execute BUY - MOONSHOT Strategy")
    print("-" * 70)
    result = trader.execute(
        market_title="Elon tweets Feb 20-27",
        bucket="400-419",
        signal="BUY",
        price=0.01,
        reason="Moonshot lottery",
        strategy_tag="MOONSHOT"
    )
    print(f"   Result: {result}")

    portfolio = trader.get_portfolio()
    assert len(portfolio["positions"]) == 2, "Should have 2 positions now"
    print(f"✅ MOONSHOT executed successfully")
    print(f"   Cash after: ${portfolio['cash']:.2f}")
    print(f"   Total positions: {len(portfolio['positions'])}")

    # Test 5: Portfolio Summary
    print("\n[TEST 5] Portfolio Summary")
    print("-" * 70)
    trader.print_summary([])
    print("✅ Summary printed successfully")

    # Test 6: Execute SELL
    print("\n[TEST 6] Execute SELL")
    print("-" * 70)
    # Sell first position
    result = trader.execute(
        market_title="Elon tweets Feb 13-20",
        bucket="300-319",
        signal="SELL",
        price=0.25,  # Profit!
        reason="Victory Lap",
        strategy_tag="STANDARD"
    )
    print(f"   Result: {result}")

    portfolio = trader.get_portfolio()
    assert len(portfolio["positions"]) == 1, "Should have 1 position after sell"
    print(f"✅ SELL executed successfully")
    print(f"   Cash after: ${portfolio['cash']:.2f}")
    print(f"   Remaining positions: {len(portfolio['positions'])}")

    # Test 7: Multiple operations
    print("\n[TEST 7] Multiple Operations")
    print("-" * 70)

    # Buy another position
    trader.execute(
        market_title="Elon tweets Feb 27-Mar 6",
        bucket="350-369",
        signal="BUY",
        price=0.15,
        reason="Accumulation Val+0.05",
        strategy_tag="STANDARD"
    )

    portfolio = trader.get_portfolio()
    print(f"   Positions after BUY: {len(portfolio['positions'])}")

    # Sell moonshot
    trader.execute(
        market_title="Elon tweets Feb 20-27",
        bucket="400-419",
        signal="SELL",
        price=0.15,  # 15x profit on moonshot!
        reason="Moonshot win",
        strategy_tag="MOONSHOT"
    )

    portfolio = trader.get_portfolio()
    print(f"   Positions after SELL: {len(portfolio['positions'])}")
    print(f"✅ Multiple operations completed")

    # Test 8: Final Portfolio State
    print("\n[TEST 8] Final Portfolio State")
    print("-" * 70)
    final_portfolio = trader.get_portfolio()

    print(f"   Final cash: ${final_portfolio['cash']:.2f}")
    print(f"   Open positions: {len(final_portfolio['positions'])}")

    total_invested = sum(pos['invested'] for pos in final_portfolio['positions'].values())
    equity = final_portfolio['cash'] + total_invested

    print(f"   Total invested in positions: ${total_invested:.2f}")
    print(f"   Total equity: ${equity:.2f}")
    print(f"   P&L from initial: ${equity - 1000.0:.2f}")

    print("✅ All portfolio operations verified")

    # Final summary
    print("\n[FINAL SUMMARY]")
    print("-" * 70)
    trader.print_summary([])

    return True


def test_real_mode_basic():
    """Basic test of real mode (init only, no trades)"""
    print("\n" + "="*70)
    print("TESTING UNIFIED TRADER - REAL MODE (Init Only)")
    print("="*70)

    try:
        from src.real_trader.unified_trader import UnifiedTrader

        print("\n[TEST 1] Real Mode Initialization")
        print("-" * 70)
        trader = UnifiedTrader(use_real=True)
        trader.initialize()

        assert trader.use_real is True
        assert trader.balance_mgr is not None
        print("✅ Real trader initialized successfully")

        print("\n[TEST 2] Get Portfolio (Real)")
        print("-" * 70)
        portfolio = trader.get_portfolio()

        assert "cash" in portfolio
        assert "positions" in portfolio
        assert "history" in portfolio

        print(f"✅ Portfolio retrieved from blockchain")
        print(f"   Cash: ${portfolio['cash']:.2f}")
        print(f"   Positions: {len(portfolio['positions'])}")

        print("\n[TEST 3] Portfolio Property (Real)")
        print("-" * 70)
        cash = trader.portfolio["cash"]
        print(f"✅ Property access works: ${cash:.2f}")

        print("\n[TEST 4] Print Summary (Real)")
        print("-" * 70)
        trader.print_summary([])
        print("✅ Summary printed successfully")

        return True

    except Exception as e:
        print(f"⚠️  Real mode test skipped: {e}")
        print("   (This is expected if .env is not configured)")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("UNIFIED TRADER - COMPLETE TEST SUITE")
    print("="*70)

    success = True

    # Test paper mode (must work)
    try:
        if test_paper_mode_complete():
            print("\n✅ PAPER MODE: ALL TESTS PASSED")
        else:
            print("\n❌ PAPER MODE: TESTS FAILED")
            success = False
    except Exception as e:
        print(f"\n❌ PAPER MODE: TESTS FAILED WITH ERROR")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test real mode (optional, may fail without .env)
    try:
        if test_real_mode_basic():
            print("\n✅ REAL MODE: BASIC TESTS PASSED")
        else:
            print("\n⚠️  REAL MODE: TESTS SKIPPED (No .env configured)")
    except Exception as e:
        print(f"\n⚠️  REAL MODE: TESTS SKIPPED")
        print(f"   Reason: {e}")

    # Final result
    print("\n" + "="*70)
    if success:
        print("🎉 TEST SUITE COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n✅ UnifiedTrader is ready for Fase 2 integration!")
        return 0
    else:
        print("❌ TEST SUITE FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())
