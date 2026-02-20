"""
Tests for PositionTracker.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import PolyAuth
from position_tracker import PositionTracker
from models import Side


@pytest.fixture
def auth():
    """Fixture for PolyAuth"""
    return PolyAuth()


@pytest.fixture
def tracker(auth):
    """Fixture for PositionTracker"""
    return PositionTracker(auth)


def test_tracker_initialization(auth):
    """Test PositionTracker initializes correctly"""
    tracker = PositionTracker(auth)

    assert tracker is not None
    assert tracker.auth is auth
    assert tracker.client is not None
    assert isinstance(tracker.positions, dict)
    assert isinstance(tracker.entered_ranges, dict)


def test_get_positions_empty(tracker):
    """Test get_positions returns list"""
    positions = tracker.get_positions()

    assert isinstance(positions, list)


def test_get_positions_for_event_empty(tracker):
    """Test get_positions_for_event returns empty list"""
    positions = tracker.get_positions_for_event("non-existent-event")

    assert isinstance(positions, list)
    assert len(positions) == 0


def test_get_position_non_existent(tracker):
    """Test get_position returns None for non-existent token"""
    position = tracker.get_position("0xnonexistent")

    assert position is None


def test_get_total_unrealized_pnl_empty(tracker):
    """Test get_total_unrealized_pnl with no positions"""
    pnl = tracker.get_total_unrealized_pnl()

    assert isinstance(pnl, (float, int))
    assert pnl == 0.0


def test_get_position_count_empty(tracker):
    """Test get_position_count with no positions"""
    count = tracker.get_position_count()

    assert isinstance(count, int)
    assert count == 0


def test_has_entered_range_false(tracker):
    """Test has_entered_range returns False for non-entered range"""
    result = tracker.has_entered_range("test-event", "10-15")

    assert result is False


def test_mark_range_entered(tracker):
    """Test marking range as entered"""
    tracker.mark_range_entered("test-event", "10-15")

    assert tracker.has_entered_range("test-event", "10-15") is True


def test_get_entered_ranges_empty(tracker):
    """Test get_entered_ranges for event with no entries"""
    ranges = tracker.get_entered_ranges("non-existent-event")

    assert isinstance(ranges, set)
    assert len(ranges) == 0


def test_add_position(tracker):
    """Test adding a position"""
    tracker.add_position(
        token_id="0x123",
        event_slug="test-event",
        range_label="10-15",
        side=Side.BUY,
        size=10.0,
        entry_price=0.50
    )

    position = tracker.get_position("0x123")

    assert position is not None
    assert position.token_id == "0x123"
    assert position.size == 10.0
    assert position.avg_entry_price == 0.50


def test_add_position_average_down(tracker):
    """Test averaging down a position"""
    # First position
    tracker.add_position(
        token_id="0x123",
        event_slug="test-event",
        range_label="10-15",
        side=Side.BUY,
        size=10.0,
        entry_price=0.50
    )

    # Add more at different price
    tracker.add_position(
        token_id="0x123",
        event_slug="test-event",
        range_label="10-15",
        side=Side.BUY,
        size=10.0,
        entry_price=0.40
    )

    position = tracker.get_position("0x123")

    assert position.size == 20.0
    assert position.avg_entry_price == 0.45  # (10*0.50 + 10*0.40) / 20


def test_remove_position_full(tracker):
    """Test removing full position"""
    tracker.add_position(
        token_id="0x123",
        event_slug="test-event",
        range_label="10-15",
        side=Side.BUY,
        size=10.0,
        entry_price=0.50
    )

    tracker.remove_position("0x123", 10.0)

    position = tracker.get_position("0x123")
    assert position is None


def test_remove_position_partial(tracker):
    """Test partially removing position"""
    tracker.add_position(
        token_id="0x123",
        event_slug="test-event",
        range_label="10-15",
        side=Side.BUY,
        size=10.0,
        entry_price=0.50
    )

    tracker.remove_position("0x123", 5.0)

    position = tracker.get_position("0x123")
    assert position is not None
    assert position.size == 5.0


def test_clear_all_stops(tracker):
    """Test clearing all stops from positions"""
    tracker.add_position(
        token_id="0x123",
        event_slug="test-event",
        range_label="10-15",
        side=Side.BUY,
        size=10.0,
        entry_price=0.50,
        use_trailing_stop=True,
        trailing_stop_percent=0.30,
        fixed_stop_loss_percent=0.35
    )

    tracker.clear_all_stops()

    position = tracker.get_position("0x123")
    assert position.use_trailing_stop is False
    assert position.trailing_stop_percent is None
    assert position.fixed_stop_price is None


@pytest.mark.asyncio
async def test_sync_positions(tracker):
    """Test syncing positions from API"""
    # This makes real API call
    await tracker.sync_positions()

    # Should complete without error
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
