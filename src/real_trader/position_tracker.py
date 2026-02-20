"""
Position Tracker for Polymarket trading.
Tracks open positions, P&L, and manages stop loss / take profit.
"""

import os
import time
import requests
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone
from dotenv import load_dotenv

try:
    from .models import Position, Side
    from .auth import PolyAuth
except ImportError:
    from models import Position, Side
    from auth import PolyAuth

load_dotenv()


class PositionTracker:
    """Tracks trading positions and P&L"""

    def __init__(self, auth: PolyAuth):
        self.auth = auth
        self.client = auth.get_client()
        self.wallet = auth.get_wallet_address()

        # Track positions
        self.positions: Dict[str, Position] = {}
        self.entered_ranges: Dict[str, Set[str]] = {}

        print(f"[PositionTracker] Initialized")

    async def sync_positions(self):
        """Sync positions from Polymarket API"""
        try:
            print("[PositionTracker] 🔄 Syncing positions from API...")

            # Fetch positions from data API
            response = requests.get(
                f"https://data-api.polymarket.com/positions",
                params={
                    "user": self.wallet,
                    "sizeThreshold": 0,
                    "limit": 100,
                    "sortBy": "CASHPNL",
                    "sortDirection": "DESC"
                },
                timeout=10
            )

            if response.status_code != 200:
                print(f"[PositionTracker] ⚠️  API returned {response.status_code}")
                return

            api_positions = response.json()
            print(f"[PositionTracker] Received {len(api_positions)} positions from API")

            self.positions.clear()
            self.entered_ranges.clear()

            filtered_count = 0
            for api_pos in api_positions:
                size = float(api_pos.get("size", 0))

                # Filter: only positions with size > 1
                if size < 1:
                    filtered_count += 1
                    print(f"[PositionTracker] 🔍 Filtered small position: {api_pos.get('outcome', 'N/A')} - {size:.4f} shares @ {api_pos.get('avgPrice', 0)}¢")
                    continue

                # Filter: only Elon Musk events
                event_slug = api_pos.get("eventSlug", "")
                if "elon" not in event_slug.lower():
                    filtered_count += 1
                    continue

                # Filter: exclude expired markets (current price = 0)
                current_price = float(api_pos.get("curPrice", 0))
                if current_price == 0:
                    filtered_count += 1
                    print(f"[PositionTracker] ⏳ Skipping expired: {api_pos.get('outcome', 'N/A')} (curPrice=0)")
                    continue

                token_id = api_pos.get("asset", "")
                avg_entry_price = float(api_pos.get("avgPrice", 0))
                unrealized_pnl = float(api_pos.get("cashPnl", 0))

                # Default stop loss and take profit
                fixed_stop_loss_percent = 0.35
                take_profit_percent = 0.80
                use_trailing_stop = True
                trailing_stop_percent = 0.30

                position = Position(
                    token_id=token_id,
                    event_slug=event_slug,
                    range_label=api_pos.get("outcome", ""),
                    side=Side.BUY,
                    size=size,
                    avg_entry_price=avg_entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    timestamp=int(time.time()),
                    fixed_stop_price=avg_entry_price * (1 - fixed_stop_loss_percent),
                    fixed_stop_loss_percent=fixed_stop_loss_percent,
                    take_profit_price=min(avg_entry_price * (1 + take_profit_percent), 0.999),
                    take_profit_percent=take_profit_percent,
                    use_trailing_stop=use_trailing_stop,
                    trailing_stop_percent=trailing_stop_percent,
                    peak_price=max(current_price, avg_entry_price),
                    market_title=api_pos.get("title", ""),
                    token_side=api_pos.get("outcome", "")
                )

                self.positions[token_id] = position

                # Mark range as entered
                if position.range_label:
                    self.mark_range_entered(event_slug, position.range_label)

                print(f"[PositionTracker] ✅ Synced: {position.range_label} - "
                      f"{size:.2f} @ {avg_entry_price*100:.2f}¢")

            print(f"[PositionTracker] Synced {len(self.positions)} positions "
                  f"({filtered_count} filtered out)")

        except Exception as e:
            print(f"[PositionTracker] ❌ Sync failed: {e}")

    def add_position(
        self,
        token_id: str,
        event_slug: str,
        range_label: str,
        side: Side,
        size: float,
        entry_price: float,
        use_trailing_stop: bool = False,
        trailing_stop_percent: Optional[float] = None,
        fixed_stop_loss_percent: Optional[float] = None,
        take_profit_percent: Optional[float] = None,
        is_lottery_ticket: bool = False,
        market_title: str = "",
        token_side: str = ""
    ):
        """Add or update a position"""

        existing = self.positions.get(token_id)

        if existing:
            # Average down/up
            total_size = existing.size + size
            total_value = (existing.size * existing.avg_entry_price) + (size * entry_price)
            new_avg_price = total_value / total_size

            existing.size = total_size
            existing.avg_entry_price = new_avg_price
            existing.timestamp = int(time.time())

            print(f"[PositionTracker] Updated: {range_label} - "
                  f"{total_size:.2f} @ {new_avg_price*100:.2f}¢")
        else:
            # New position
            fixed_stop_price = (entry_price * (1 - fixed_stop_loss_percent)
                                if fixed_stop_loss_percent else None)
            take_profit_price = (min(entry_price * (1 + take_profit_percent), 0.999)
                                 if take_profit_percent else None)

            position = Position(
                token_id=token_id,
                event_slug=event_slug,
                range_label=range_label,
                side=side,
                size=size,
                avg_entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0,
                timestamp=int(time.time()),
                peak_price=entry_price,
                use_trailing_stop=use_trailing_stop,
                trailing_stop_percent=trailing_stop_percent,
                fixed_stop_price=fixed_stop_price,
                fixed_stop_loss_percent=fixed_stop_loss_percent,
                take_profit_price=take_profit_price,
                take_profit_percent=take_profit_percent,
                is_lottery_ticket=is_lottery_ticket,
                market_title=market_title,
                token_side=token_side
            )

            self.positions[token_id] = position

            print(f"[PositionTracker] New: {range_label} - {size:.2f} @ {entry_price*100:.2f}¢")

    def remove_position(self, token_id: str, size: float):
        """Remove or reduce a position"""
        position = self.positions.get(token_id)

        if not position:
            print(f"[PositionTracker] ⚠️  Position not found: {token_id[:10]}...")
            return

        if size >= position.size:
            del self.positions[token_id]
            print(f"[PositionTracker] Closed: {position.range_label}")
        else:
            position.size -= size
            print(f"[PositionTracker] Reduced: {position.range_label} - "
                  f"{position.size:.2f} remaining")

    async def update_current_prices(self):
        """
        Update current prices for all positions.

        NOTE: We do NOT recalculate P&L here because sync_positions() already
        provides the correct cashPnl from Polymarket API (which includes fees).
        We only update current_price for display and peak tracking.
        """
        for token_id, position in self.positions.items():
            try:
                # Get current price from CLOB
                sell_side = "SELL" if position.side == Side.BUY else "BUY"
                price_data = self.client.get_price(token_id, sell_side)
                current_price = float(price_data.get("price", 0))

                if current_price > 0:
                    position.current_price = current_price

                    # Update peak price for trailing stops
                    if not position.peak_price or current_price > position.peak_price:
                        position.peak_price = current_price

                    # NOTE: P&L is NOT updated here - it comes from sync_positions()
                    # which fetches the correct cashPnl from Polymarket API (includes fees)

            except Exception as e:
                print(f"[PositionTracker] ⚠️  Price update failed for {position.range_label}: {e}")

    def get_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())

    def get_positions_for_event(self, event_slug: str) -> List[Position]:
        """Get positions for specific event"""
        return [p for p in self.positions.values() if p.event_slug == event_slug]

    def get_position(self, token_id: str) -> Optional[Position]:
        """Get position by token ID"""
        return self.positions.get(token_id)

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions"""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_position_count(self, event_slug: Optional[str] = None) -> int:
        """Get position count (total or for event)"""
        if event_slug:
            return len(self.get_positions_for_event(event_slug))
        return len(self.positions)

    def has_entered_range(self, event_slug: str, range_label: str) -> bool:
        """Check if range was already entered"""
        ranges = self.entered_ranges.get(event_slug, set())
        return range_label in ranges

    def mark_range_entered(self, event_slug: str, range_label: str):
        """Mark range as entered"""
        if event_slug not in self.entered_ranges:
            self.entered_ranges[event_slug] = set()
        self.entered_ranges[event_slug].add(range_label)

    def get_entered_ranges(self, event_slug: str) -> Set[str]:
        """Get all entered ranges for an event"""
        return self.entered_ranges.get(event_slug, set())

    def clear_all_stops(self):
        """Clear all stop loss / take profit from positions"""
        print("[PositionTracker] 🛑 Clearing all stops...")
        cleared = 0

        for position in self.positions.values():
            if (position.use_trailing_stop or position.fixed_stop_price or
                position.take_profit_price):
                position.use_trailing_stop = False
                position.trailing_stop_percent = None
                position.fixed_stop_price = None
                position.fixed_stop_loss_percent = None
                position.take_profit_price = None
                position.take_profit_percent = None
                cleared += 1
                print(f"  ✅ Cleared: {position.range_label}")

        print(f"[PositionTracker] Cleared stops from {cleared} positions")
