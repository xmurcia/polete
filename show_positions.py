#!/usr/bin/env python3
"""
Show all current positions with their token IDs.
"""

import asyncio
from src.real_trader.auth import PolyAuth
from src.real_trader.position_tracker import PositionTracker


async def show_positions():
    print("\n" + "="*80)
    print("CURRENT POSITIONS")
    print("="*80 + "\n")

    auth = PolyAuth()
    tracker = PositionTracker(auth)

    await tracker.sync_positions()
    positions = tracker.get_positions()

    if not positions:
        print("No positions found.\n")
        return

    for i, pos in enumerate(positions, 1):
        print(f"{i}. Event: {pos.event_slug}")
        print(f"   Range: {pos.range_label}")
        print(f"   Token ID: {pos.token_id}")
        print(f"   Size: {pos.size} shares @ ${pos.avg_entry_price:.3f}")
        print(f"   Current Price: ${pos.current_price:.3f}")
        print(f"   Unrealized P&L: ${pos.unrealized_pnl:.2f}")
        print()

    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(show_positions())
