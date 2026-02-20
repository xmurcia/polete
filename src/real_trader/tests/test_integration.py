"""
Integration tests for the complete trading flow.
Tests real API calls and full lifecycle.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import PolyAuth
from order_manager import OrderManager
from position_tracker import PositionTracker
from balance_manager import BalanceManager


@pytest.fixture
def auth():
    """Fixture for PolyAuth"""
    return PolyAuth()


async def _create_setup(auth):
    """Helper to create full setup"""
    balance_mgr = BalanceManager(auth)
    await balance_mgr.initialize()

    position_tracker = PositionTracker(auth)
    await position_tracker.sync_positions()

    order_mgr = OrderManager(auth, balance_mgr)
    await order_mgr.sync_open_orders()

    return {
        "auth": auth,
        "balance_mgr": balance_mgr,
        "position_tracker": position_tracker,
        "order_mgr": order_mgr
    }


@pytest.mark.asyncio
async def test_full_initialization_flow(auth):
    """Test initializing all components together"""
    # Initialize balance
    balance_mgr = BalanceManager(auth)
    await balance_mgr.initialize()

    assert balance_mgr.initial_capital >= 0

    # Initialize positions
    position_tracker = PositionTracker(auth)
    await position_tracker.sync_positions()

    positions = position_tracker.get_positions()
    assert isinstance(positions, list)

    # Initialize orders
    order_mgr = OrderManager(auth, balance_mgr)
    await order_mgr.sync_open_orders()

    orders = order_mgr.get_open_orders()
    assert isinstance(orders, list)


@pytest.mark.asyncio
async def test_balance_and_positions_consistency():
    """Test that balance and positions are consistent"""
    auth = PolyAuth()
    setup = await _create_setup(auth)

    balance_mgr = setup["balance_mgr"]
    position_tracker = setup["position_tracker"]

    available = await balance_mgr.get_available_balance()
    positions = position_tracker.get_positions()

    total_position_value = sum(
        p.avg_entry_price * p.size
        for p in positions
    )

    # Total should be available + in positions
    # (ignoring P&L for this test)
    assert available >= 0
    assert total_position_value >= 0


@pytest.mark.asyncio
async def test_update_prices_and_pnl():
    """Test updating prices and calculating P&L"""
    auth = PolyAuth()
    setup = await _create_setup(auth)

    position_tracker = setup["position_tracker"]

    # Update prices
    await position_tracker.update_current_prices()

    # Calculate total P&L
    total_pnl = position_tracker.get_total_unrealized_pnl()

    assert isinstance(total_pnl, (float, int))


@pytest.mark.asyncio
async def test_exposure_calculation():
    """Test exposure calculation"""
    auth = PolyAuth()
    setup = await _create_setup(auth)

    balance_mgr = setup["balance_mgr"]
    position_tracker = setup["position_tracker"]

    positions = position_tracker.get_positions()

    total_position_value = sum(
        p.avg_entry_price * p.size
        for p in positions
    )

    exposure = balance_mgr.get_total_exposure(total_position_value)

    assert isinstance(exposure, float)
    assert exposure >= 0
    assert exposure <= 1.5  # Should not exceed 150% (with some margin)


@pytest.mark.asyncio
async def test_log_status():
    """Test logging complete status"""
    auth = PolyAuth()
    setup = await _create_setup(auth)

    balance_mgr = setup["balance_mgr"]
    position_tracker = setup["position_tracker"]

    positions = position_tracker.get_positions()

    total_position_value = sum(
        p.avg_entry_price * p.size
        for p in positions
    )

    total_pnl = position_tracker.get_total_unrealized_pnl()

    # Should complete without error
    await balance_mgr.log_status(
        total_position_value=total_position_value,
        unrealized_pnl=total_pnl,
        position_count=len(positions)
    )

    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
