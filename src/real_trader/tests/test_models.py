"""
Tests for data models.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import (
    Side, OrderType, OrderRequest, OrderResult,
    TrackedOrder, Position
)


def test_side_enum():
    """Test Side enum"""
    assert Side.BUY == "BUY"
    assert Side.SELL == "SELL"


def test_order_type_enum():
    """Test OrderType enum"""
    assert OrderType.GTC == "GTC"
    assert OrderType.FOK == "FOK"
    assert OrderType.GTD == "GTD"


def test_order_request_creation():
    """Test OrderRequest dataclass"""
    order = OrderRequest(
        token_id="0x123",
        price=0.50,
        size=10.0,
        side=Side.BUY,
        order_type=OrderType.FOK,
        event_slug="test-event",
        range_label="10-15"
    )

    assert order.token_id == "0x123"
    assert order.price == 0.50
    assert order.size == 10.0
    assert order.side == Side.BUY
    assert order.order_type == OrderType.FOK
    assert order.is_stop_loss is False


def test_order_result_success():
    """Test OrderResult for successful order"""
    result = OrderResult(
        success=True,
        order_id="ORDER_123"
    )

    assert result.success is True
    assert result.order_id == "ORDER_123"
    assert result.error is None


def test_order_result_failure():
    """Test OrderResult for failed order"""
    result = OrderResult(
        success=False,
        error="Insufficient balance"
    )

    assert result.success is False
    assert result.order_id is None
    assert result.error == "Insufficient balance"


def test_tracked_order_creation():
    """Test TrackedOrder dataclass"""
    order = TrackedOrder(
        order_id="ORDER_123",
        token_id="0x123",
        event_slug="test-event",
        range_label="10-15",
        side=Side.BUY,
        order_type=OrderType.FOK,
        price=0.50,
        size=10.0,
        timestamp=1234567890
    )

    assert order.order_id == "ORDER_123"
    assert order.side == Side.BUY
    assert order.price == 0.50


def test_position_creation():
    """Test Position dataclass"""
    position = Position(
        token_id="0x123",
        event_slug="test-event",
        range_label="10-15",
        side=Side.BUY,
        size=10.0,
        avg_entry_price=0.50,
        current_price=0.55,
        unrealized_pnl=0.50,
        timestamp=1234567890
    )

    assert position.token_id == "0x123"
    assert position.size == 10.0
    assert position.unrealized_pnl == 0.50
    assert position.use_trailing_stop is False


def test_position_with_stops():
    """Test Position with stop loss/take profit"""
    position = Position(
        token_id="0x123",
        event_slug="test-event",
        range_label="10-15",
        side=Side.BUY,
        size=10.0,
        avg_entry_price=0.50,
        current_price=0.55,
        unrealized_pnl=0.50,
        timestamp=1234567890,
        use_trailing_stop=True,
        trailing_stop_percent=0.30,
        fixed_stop_price=0.35,
        take_profit_price=0.75
    )

    assert position.use_trailing_stop is True
    assert position.trailing_stop_percent == 0.30
    assert position.fixed_stop_price == 0.35
    assert position.take_profit_price == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
