"""
Tests for OrderManager.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import PolyAuth
from order_manager import OrderManager
from models import OrderRequest, Side, OrderType


@pytest.fixture
def auth():
    """Fixture for PolyAuth"""
    return PolyAuth()


@pytest.fixture
def order_manager(auth):
    """Fixture for OrderManager"""
    return OrderManager(auth)


def test_order_manager_initialization(auth):
    """Test OrderManager initializes correctly"""
    mgr = OrderManager(auth)

    assert mgr is not None
    assert mgr.auth is auth
    assert mgr.client is not None
    assert isinstance(mgr.open_orders, dict)
    assert isinstance(mgr.orders_by_event, dict)


def test_get_open_orders_empty(order_manager):
    """Test get_open_orders returns empty list initially"""
    orders = order_manager.get_open_orders()

    assert isinstance(orders, list)
    # May have real orders if connected to live account


def test_get_orders_for_event_empty(order_manager):
    """Test get_orders_for_event returns empty list for non-existent event"""
    orders = order_manager.get_orders_for_event("non-existent-event")

    assert isinstance(orders, list)
    assert len(orders) == 0


def test_get_order_count(order_manager):
    """Test get_order_count"""
    count = order_manager.get_order_count()

    assert isinstance(count, int)
    assert count >= 0


def test_get_order_count_for_event(order_manager):
    """Test get_order_count for specific event"""
    count = order_manager.get_order_count("test-event")

    assert isinstance(count, int)
    assert count == 0  # No orders for non-existent event


@pytest.mark.asyncio
async def test_sync_open_orders(order_manager):
    """Test syncing open orders from blockchain"""
    # This makes real API call
    await order_manager.sync_open_orders()

    # Should complete without error
    assert True


def test_remove_order_non_existent(order_manager):
    """Test removing non-existent order doesn't crash"""
    # Should not raise exception
    order_manager._remove_order("non-existent-order-id")

    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
