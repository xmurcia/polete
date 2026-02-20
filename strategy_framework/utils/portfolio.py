import json
import os
from typing import Dict, Optional
from datetime import datetime

class Portfolio:
    """
    Manages portfolio state: cash, positions, trade history.

    Simplified version of PaperTrader from elon_auto_bot_threads.py
    Separated from trading logic for testability.
    """

    def __init__(self, initial_cash: float = 1000.0, file_path: Optional[str] = None):
        self.file_path = file_path
        self.data = {
            "cash": initial_cash,
            "positions": {},
            "history": [],
            "peak_value": initial_cash
        }

        if file_path and os.path.exists(file_path):
            self._load()

    def _load(self):
        """Load portfolio from JSON file"""
        try:
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading portfolio: {e}")

    def _save(self):
        """Save portfolio to JSON file"""
        if self.file_path:
            try:
                with open(self.file_path, 'w') as f:
                    json.dump(self.data, f, indent=2)
            except Exception as e:
                print(f"⚠️ Error saving portfolio: {e}")

    def get_cash(self) -> float:
        return self.data["cash"]

    def get_positions(self) -> Dict:
        return self.data["positions"]

    def get_position(self, market: str, bucket: str) -> Optional[Dict]:
        """Get specific position or None"""
        pos_id = f"{market}|{bucket}"
        return self.data["positions"].get(pos_id)

    def add_position(
        self,
        market: str,
        bucket: str,
        shares: float,
        entry_price: float,
        invested: float,
        strategy_tag: str = "STANDARD"  # DNA Tag
    ) -> bool:
        """
        Add new position to portfolio.

        Returns:
            True if successful, False if insufficient cash
        """
        if self.data["cash"] < invested:
            return False

        pos_id = f"{market}|{bucket}"
        self.data["cash"] -= invested
        self.data["positions"][pos_id] = {
            "shares": shares,
            "entry_price": entry_price,
            "market": market,
            "bucket": bucket,
            "timestamp": datetime.now().timestamp(),
            "invested": invested,
            "max_price_seen": entry_price,
            "price_history": [],
            "strategy_tag": strategy_tag  # Store DNA tag
        }

        self._save()
        return True

    def close_position(
        self,
        market: str,
        bucket: str,
        exit_price: float
    ) -> Optional[Dict]:
        """
        Close position and return P&L info.

        Returns:
            Dict with profit, roi, etc. or None if position doesn't exist
        """
        pos_id = f"{market}|{bucket}"
        pos = self.data["positions"].get(pos_id)

        if not pos:
            return None

        revenue = pos["shares"] * exit_price
        cost_basis = pos.get("invested", pos["shares"] * pos["entry_price"])
        profit = revenue - cost_basis
        roi = (profit / cost_basis) * 100 if cost_basis > 0 else 0

        self.data["cash"] += revenue
        del self.data["positions"][pos_id]

        trade_record = {
            "market": market,
            "bucket": bucket,
            "profit": profit,
            "roi": roi,
            "exit_price": exit_price,
            "exit_time": datetime.now().timestamp()
        }
        self.data["history"].append(trade_record)

        self._save()
        return trade_record

    def update_position_metadata(self, market: str, bucket: str, updates: Dict):
        """Update position metadata (e.g., max_price_seen, price_history)"""
        pos_id = f"{market}|{bucket}"
        if pos_id in self.data["positions"]:
            self.data["positions"][pos_id].update(updates)
            self._save()

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.

        Args:
            current_prices: Dict mapping position_id -> current_bid_price
        """
        invested_value = 0.0
        for pos_id, pos in self.data["positions"].items():
            current_price = current_prices.get(pos_id, pos["entry_price"])
            invested_value += pos["shares"] * current_price

        return self.data["cash"] + invested_value

    def get_statistics(self) -> Dict:
        """Get portfolio statistics"""
        total_trades = len(self.data["history"])
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_profit": 0
            }

        winning_trades = [t for t in self.data["history"] if t["profit"] > 0]
        total_pnl = sum(t["profit"] for t in self.data["history"])

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": total_trades - len(winning_trades),
            "win_rate": len(winning_trades) / total_trades,
            "total_pnl": total_pnl,
            "avg_profit": total_pnl / total_trades,
            "best_trade": max(self.data["history"], key=lambda x: x["profit"])["profit"],
            "worst_trade": min(self.data["history"], key=lambda x: x["profit"])["profit"]
        }
