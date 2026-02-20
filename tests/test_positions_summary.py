#!/usr/bin/env python3
"""Test positions summary notification"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.notifications.telegram_notifier import TelegramNotifier

def test_positions_summary():
    """Test the positions summary notification"""
    notifier = TelegramNotifier()

    if not notifier.enabled:
        print("❌ Telegram notifications not enabled")
        print("Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return

    # Mock positions data
    positions = [
        {
            'event_slug': 'Elon tweets Feb 13-20',
            'range_label': '300-319',
            'avg_entry_price': 0.270,
            'current_price': 0.310,
            'size': 27.5,
            'unrealized_pnl': 1.10
        },
        {
            'event_slug': 'Elon tweets Feb 20-27',
            'range_label': '280-299',
            'avg_entry_price': 0.180,
            'current_price': 0.165,
            'size': 35.2,
            'unrealized_pnl': -0.53
        }
    ]

    balance = 84.14

    print("\n🧪 Sending test positions summary...")
    notifier.notify_positions_summary(
        positions=positions,
        balance=balance,
        mode="REAL"
    )

    print("✅ Test sent! Check Telegram for the message.\n")

if __name__ == "__main__":
    test_positions_summary()
