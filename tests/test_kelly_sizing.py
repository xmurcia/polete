#!/usr/bin/env python3
"""
Test Kelly sizing calculation without executing trades
"""

import asyncio
import sys
import os

sys.path.insert(0, 'real_trader')

async def test_kelly_calculation():
    """Test the bet sizing calculation"""
    print("\n" + "="*70)
    print("KELLY SIZING TEST")
    print("="*70)

    from balance_manager import BalanceManager
    from auth import PolyAuth

    # Initialize
    auth = PolyAuth()
    balance_mgr = BalanceManager(auth)
    await balance_mgr.initialize()

    # Test parameters from the actual trade
    available_balance = 92.90
    price = 0.27
    edge_value = 0.42
    strategy_tag = "STANDARD"

    print(f"\n📊 Test Parameters:")
    print(f"  Available Balance: ${available_balance:.2f}")
    print(f"  Price: ${price:.2f}")
    print(f"  Edge Value: {edge_value} (42%)")
    print(f"  Strategy: {strategy_tag}")

    # Override available balance for testing
    original_get_balance = balance_mgr.get_available_balance

    async def mock_balance():
        return available_balance

    balance_mgr.get_available_balance = mock_balance

    # Calculate bet size
    bet_amount, shares = await balance_mgr.calculate_bet_size(
        price=price,
        strategy_tag=strategy_tag,
        edge_value=edge_value,
        is_hedge=False
    )

    print(f"\n💰 Calculation Results:")
    print(f"  Base Risk %: {balance_mgr.risk_pct_normal*100:.1f}%")
    print(f"  Edge >= 0.40: Multiplier 2x")
    print(f"  Final %: {balance_mgr.risk_pct_normal * 2 * 100:.1f}% (capped at 10%)")
    print(f"  Expected Bet: ${available_balance * 0.08:.2f} (8% of $92.90)")
    print(f"  Expected Shares: {(available_balance * 0.08) / price:.2f}")

    print(f"\n🔍 Actual Calculation:")
    print(f"  Bet Amount: ${bet_amount:.2f}")
    print(f"  Shares: {shares:.2f}")
    print(f"  Total Cost: ${shares * price:.2f}")
    print(f"  Percentage of Capital: {(bet_amount / available_balance) * 100:.1f}%")

    # Verify
    expected_bet = available_balance * 0.08  # 8% with edge 42%
    expected_shares = expected_bet / price

    print(f"\n✅ Verification:")
    if abs(bet_amount - expected_bet) < 0.5:
        print(f"  ✅ Bet amount correct: ${bet_amount:.2f} ≈ ${expected_bet:.2f}")
    else:
        print(f"  ❌ Bet amount WRONG:")
        print(f"     Expected: ${expected_bet:.2f}")
        print(f"     Got: ${bet_amount:.2f}")
        print(f"     Difference: ${bet_amount - expected_bet:.2f}")

    if abs(shares - expected_shares) < 1:
        print(f"  ✅ Shares correct: {shares:.2f} ≈ {expected_shares:.2f}")
    else:
        print(f"  ❌ Shares WRONG:")
        print(f"     Expected: {expected_shares:.2f}")
        print(f"     Got: {shares:.2f}")
        print(f"     Difference: {shares - expected_shares:.2f}")

    # Restore
    balance_mgr.get_available_balance = original_get_balance

    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(test_kelly_calculation())
