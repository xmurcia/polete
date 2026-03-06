"""
Tests for balance manager position sizing logic.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import PolyAuth
from balance_manager import BalanceManager


@pytest.fixture
def auth():
    """Fixture for PolyAuth"""
    return PolyAuth()


@pytest.fixture
def balance_mgr(auth):
    """Fixture for BalanceManager"""
    return BalanceManager(auth)


@pytest.mark.asyncio
async def test_calculate_bet_size_standard(balance_mgr):
    """Test standard position sizing"""
    await balance_mgr.initialize()

    bet_amount, shares = await balance_mgr.calculate_bet_size(
        price=0.20,
        strategy_tag="STANDARD"
    )

    # Should be ~4% of capital or min_bet (whichever is higher)
    expected_pct = balance_mgr.risk_pct_normal
    expected_amount = max(balance_mgr.initial_capital * expected_pct, balance_mgr.min_bet)

    assert bet_amount >= balance_mgr.min_bet
    assert bet_amount <= balance_mgr.initial_capital
    assert shares == bet_amount / 0.20
    assert bet_amount == expected_amount


@pytest.mark.asyncio
async def test_calculate_bet_size_with_edge(balance_mgr):
    """Test Kelly multiplier with edge"""
    await balance_mgr.initialize()

    # No edge
    bet_no_edge, _ = await balance_mgr.calculate_bet_size(
        price=0.20,
        strategy_tag="STANDARD"
    )

    # 25% edge (should be 1.5x)
    bet_medium_edge, _ = await balance_mgr.calculate_bet_size(
        price=0.20,
        strategy_tag="STANDARD",
        edge_value=0.25
    )

    # 45% edge (should be 2.0x)
    bet_high_edge, _ = await balance_mgr.calculate_bet_size(
        price=0.20,
        strategy_tag="STANDARD",
        edge_value=0.45
    )

    # Verify multipliers
    assert bet_medium_edge > bet_no_edge
    assert bet_high_edge > bet_medium_edge
    assert bet_medium_edge / bet_no_edge <= 1.5
    assert bet_high_edge / bet_no_edge <= 2.0


@pytest.mark.asyncio
async def test_calculate_bet_size_moonshot(balance_mgr):
    """Test moonshot sizing (capped at $10)"""
    await balance_mgr.initialize()

    bet_amount, shares = await balance_mgr.calculate_bet_size(
        price=0.01,
        strategy_tag="MOONSHOT"
    )

    # Should be capped at max_moonshot_bet
    assert bet_amount <= balance_mgr.max_moonshot_bet
    assert bet_amount >= balance_mgr.min_bet
    assert shares == bet_amount / 0.01


@pytest.mark.asyncio
async def test_calculate_bet_size_lotto(balance_mgr):
    """Test lottery ticket sizing (1%)"""
    await balance_mgr.initialize()

    bet_amount, shares = await balance_mgr.calculate_bet_size(
        price=0.05,
        strategy_tag="LOTTO"
    )

    # Should be ~1% of capital or min_bet (whichever is higher)
    expected_pct = balance_mgr.risk_pct_lotto
    expected_amount = max(balance_mgr.initial_capital * expected_pct, balance_mgr.min_bet)

    assert bet_amount >= balance_mgr.min_bet
    assert bet_amount == expected_amount


@pytest.mark.asyncio
async def test_calculate_bet_size_hedge(balance_mgr):
    """Test hedge sizing (2.5%)"""
    await balance_mgr.initialize()

    bet_amount, shares = await balance_mgr.calculate_bet_size(
        price=0.15,
        is_hedge=True
    )

    # Should be ~2.5% of capital or min_bet (whichever is higher)
    expected_pct = 0.025
    expected_amount = max(balance_mgr.initial_capital * expected_pct, balance_mgr.min_bet)

    assert bet_amount >= balance_mgr.min_bet
    assert bet_amount == expected_amount


@pytest.mark.asyncio
async def test_calculate_bet_size_safety_cap(balance_mgr):
    """Test that sizing is capped at 10% of capital"""
    await balance_mgr.initialize()

    # Try with very high edge (would be 2x multiplier)
    bet_amount, _ = await balance_mgr.calculate_bet_size(
        price=0.20,
        strategy_tag="STANDARD",
        edge_value=0.50  # 50% edge
    )

    # Should be capped at 10% max
    max_allowed = balance_mgr.initial_capital * 0.10
    assert bet_amount <= max_allowed


@pytest.mark.asyncio
async def test_calculate_bet_size_min_bet(balance_mgr):
    """Test minimum bet enforcement"""
    await balance_mgr.initialize()

    # Even with small capital percentage, should enforce min_bet
    bet_amount, _ = await balance_mgr.calculate_bet_size(
        price=0.50,
        strategy_tag="LOTTO"  # 1% of capital
    )

    assert bet_amount >= balance_mgr.min_bet


@pytest.mark.asyncio
async def test_moonshot_no_kelly_boost(balance_mgr):
    """Test that moonshots don't get Kelly multiplier"""
    await balance_mgr.initialize()

    # Moonshot with edge shouldn't get boosted
    bet_no_edge, _ = await balance_mgr.calculate_bet_size(
        price=0.01,
        strategy_tag="MOONSHOT"
    )

    bet_with_edge, _ = await balance_mgr.calculate_bet_size(
        price=0.01,
        strategy_tag="MOONSHOT",
        edge_value=0.50  # 50% edge
    )

    # Should be the same (no Kelly boost for moonshots)
    assert bet_no_edge == bet_with_edge


@pytest.mark.asyncio
async def test_hedge_no_kelly_boost(balance_mgr):
    """Test that hedges don't get Kelly multiplier"""
    await balance_mgr.initialize()

    # Hedge with edge shouldn't get boosted
    bet_no_edge, _ = await balance_mgr.calculate_bet_size(
        price=0.15,
        is_hedge=True
    )

    bet_with_edge, _ = await balance_mgr.calculate_bet_size(
        price=0.15,
        is_hedge=True,
        edge_value=0.50
    )

    # Should be the same (no Kelly boost for hedges)
    assert bet_no_edge == bet_with_edge


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
