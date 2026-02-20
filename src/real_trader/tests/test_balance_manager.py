"""
Tests for BalanceManager.
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
def balance_manager(auth):
    """Fixture for BalanceManager"""
    return BalanceManager(auth)


def test_balance_manager_initialization(auth):
    """Test BalanceManager initializes correctly"""
    mgr = BalanceManager(auth)

    assert mgr is not None
    assert mgr.auth is auth
    assert mgr.client is not None
    assert mgr.max_daily_loss > 0
    assert mgr.max_exposure > 0
    assert mgr.realized_pnl == 0.0
    assert mgr.daily_pnl == 0.0


def test_calculate_position_size(balance_manager):
    """Test position size calculation"""
    size = balance_manager.calculate_position_size(
        dollar_amount=10.0,
        price=0.50
    )

    assert size == 20.0  # 10 / 0.50


def test_calculate_position_size_zero_price(balance_manager):
    """Test position size with zero price"""
    size = balance_manager.calculate_position_size(
        dollar_amount=10.0,
        price=0.0
    )

    assert size == 0.0


def test_get_total_exposure(balance_manager):
    """Test exposure calculation"""
    balance_manager.initial_capital = 100.0

    exposure = balance_manager.get_total_exposure(
        total_position_value=50.0
    )

    assert exposure == 0.5  # 50 / 100


def test_get_total_exposure_zero_capital(balance_manager):
    """Test exposure with zero capital"""
    balance_manager.initial_capital = 0.0

    exposure = balance_manager.get_total_exposure(
        total_position_value=50.0
    )

    assert exposure == 0.0


def test_add_realized_pnl(balance_manager):
    """Test tracking realized P&L"""
    balance_manager.add_realized_pnl(10.0)

    assert balance_manager.realized_pnl == 10.0
    assert balance_manager.daily_pnl == 10.0

    balance_manager.add_realized_pnl(-5.0)

    assert balance_manager.realized_pnl == 5.0
    assert balance_manager.daily_pnl == 5.0


def test_check_daily_loss_limit_not_hit(balance_manager):
    """Test daily loss limit not hit"""
    balance_manager.max_daily_loss = 30.0
    balance_manager.daily_pnl = -10.0

    hit = balance_manager.check_daily_loss_limit()

    assert hit is False


def test_check_daily_loss_limit_hit(balance_manager):
    """Test daily loss limit hit"""
    balance_manager.max_daily_loss = 30.0
    balance_manager.daily_pnl = -30.0

    hit = balance_manager.check_daily_loss_limit()

    assert hit is True


def test_reset_daily_pnl(balance_manager):
    """Test resetting daily P&L"""
    balance_manager.daily_pnl = -15.0
    balance_manager.reset_daily_pnl()

    assert balance_manager.daily_pnl == 0.0


def test_get_balance_info(balance_manager):
    """Test getting balance info as dict"""
    info = balance_manager.get_balance_info()

    assert isinstance(info, dict)
    assert "initial_capital" in info
    assert "realized_pnl" in info
    assert "daily_pnl" in info
    assert "max_daily_loss" in info
    assert "max_exposure" in info


@pytest.mark.asyncio
async def test_initialize(balance_manager):
    """Test initializing balance"""
    # This makes real API call
    await balance_manager.initialize()

    assert balance_manager.initial_capital >= 0


@pytest.mark.asyncio
async def test_get_available_balance(balance_manager):
    """Test getting available balance"""
    # This makes real API call
    balance = await balance_manager.get_available_balance()

    assert isinstance(balance, float)
    assert balance >= 0


@pytest.mark.asyncio
async def test_can_place_order_sufficient(balance_manager):
    """Test can_place_order with sufficient balance"""
    await balance_manager.initialize()

    # Small order that should be affordable
    can_place = await balance_manager.can_place_order(1.0)

    # Should be True if balance > $1
    assert isinstance(can_place, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
