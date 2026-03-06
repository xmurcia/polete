"""
Balance Manager for Polymarket trading.
Tracks balance, allowance, exposure, and P&L.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

try:
    from .auth import PolyAuth
except ImportError:
    from auth import PolyAuth

load_dotenv()


class BalanceManager:
    """Manages trading balance and risk limits"""

    def __init__(self, auth: PolyAuth):
        self.auth = auth
        self.client = auth.get_client()

        # Risk Configuration (from PaperTrader)
        self.risk_pct_normal = float(os.getenv("RISK_PCT_NORMAL", "0.04"))      # 4% per trade
        self.risk_pct_lotto = float(os.getenv("RISK_PCT_LOTTO", "0.01"))        # 1% lottery tickets
        self.risk_pct_moonshot = float(os.getenv("RISK_PCT_MOONSHOT", "0.01"))  # 1% moonshots
        self.max_moonshot_bet = float(os.getenv("MAX_MOONSHOT_BET", "10.0"))    # $10 max per moonshot
        self.min_bet = float(os.getenv("MIN_BET", "5.0"))                       # $5 minimum bet

        # Limits Configuration
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "200"))  # Increased from 30 to 200
        self.max_exposure = float(os.getenv("MAX_EXPOSURE", "0.99"))

        # Tracking
        self.initial_capital = 0.0
        self.realized_pnl = 0.0
        self.daily_pnl = 0.0

        print(f"[BalanceManager] Initialized")
        print(f"  Risk per trade (normal): {self.risk_pct_normal*100:.1f}%")
        print(f"  Risk per trade (lotto): {self.risk_pct_lotto*100:.1f}%")
        print(f"  Risk per trade (moonshot): {self.risk_pct_moonshot*100:.1f}%")
        print(f"  Max moonshot bet: ${self.max_moonshot_bet:.2f}")
        print(f"  Min bet: ${self.min_bet:.2f}")
        print(f"  Max Daily Loss: ${self.max_daily_loss:.2f}")
        print(f"  Max Exposure: {self.max_exposure*100:.0f}%")

    async def initialize(self):
        """Initialize balance from blockchain"""
        try:
            balance = await self.get_available_balance()
            self.initial_capital = balance
            print(f"[BalanceManager] Initial capital: ${balance:.2f}")
        except Exception as e:
            print(f"[BalanceManager] ⚠️  Failed to fetch balance: {e}")

    async def get_available_balance(self) -> float:
        """Get available USDC balance"""
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            response = self.client.get_balance_allowance(params)
            balance = float(response.get("balance", 0)) / 1e6
            return balance
        except Exception as e:
            print(f"[BalanceManager] ⚠️  Balance fetch failed: {e}")
            return 0.0

    async def get_conditional_balance(self, token_id: str) -> float:
        """Get CTF conditional token balance for a specific token"""
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
            response = self.client.get_balance_allowance(params)
            # CTF balance is returned in human-readable format (already divided by 1e6)
            balance = float(response.get("balance", 0))
            return balance
        except Exception as e:
            print(f"[BalanceManager] ⚠️  CTF balance fetch failed: {e}")
            return 0.0

    async def can_place_order(self, dollar_amount: float) -> bool:
        """Check if order can be placed (sufficient balance)"""
        available = await self.get_available_balance()

        if dollar_amount > available:
            print(f"[BalanceManager] ❌ Insufficient: need ${dollar_amount:.2f}, "
                  f"have ${available:.2f}")
            return False

        return True

    def calculate_position_size(self, dollar_amount: float, price: float) -> float:
        """Calculate number of shares for given dollar amount"""
        if price <= 0:
            return 0
        return dollar_amount / price

    async def calculate_bet_size(
        self,
        price: float,
        strategy_tag: str = "STANDARD",
        edge_value: Optional[float] = None,
        is_hedge: bool = False
    ) -> tuple[float, float]:
        """
        Calculate bet size using PaperTrader logic.
        Returns (bet_amount, shares).

        Args:
            price: Entry price
            strategy_tag: "STANDARD", "MOONSHOT", or "LOTTO"
            edge_value: Optional edge value for Kelly criterion (e.g., 0.20 = 20% edge)
            is_hedge: Whether this is a hedge trade
        """
        available = await self.get_available_balance()

        # 1. Define base percentage
        if strategy_tag == "MOONSHOT":
            base_pct = self.risk_pct_moonshot
        elif strategy_tag == "LOTTO":
            base_pct = self.risk_pct_lotto
        elif is_hedge:
            base_pct = 0.025  # 2.5% for hedges
        else:
            base_pct = self.risk_pct_normal

        # 2. Apply Kelly criterion multiplier (Sniper Mode)
        multiplier = 1.0
        if edge_value is not None and edge_value > 0 and not is_hedge:
            if edge_value >= 0.40:
                multiplier = 2.0    # 2x for 40%+ edge
            elif edge_value >= 0.20:
                multiplier = 1.5    # 1.5x for 20%+ edge

        # 3. Calculate final amount with safety belt
        final_pct = base_pct * multiplier
        final_pct = min(final_pct, 0.10)  # Cap at 10% of capital
        bet_amount = max(available * final_pct, self.min_bet)

        # 4. Apply moonshot cap
        if strategy_tag == "MOONSHOT":
            bet_amount = min(bet_amount, self.max_moonshot_bet)

        # 5. Ensure we have enough balance
        bet_amount = min(bet_amount, available)

        # 6. Calculate shares
        shares = bet_amount / price if price > 0 else 0

        return (bet_amount, shares)

    def get_total_exposure(self, total_position_value: float) -> float:
        """Calculate exposure as ratio of positions to capital"""
        if self.initial_capital <= 0:
            return 0.0
        return total_position_value / self.initial_capital

    def add_realized_pnl(self, amount: float):
        """Track realized P&L"""
        self.realized_pnl += amount
        self.daily_pnl += amount
        print(f"[BalanceManager] Realized P&L: "
              f"{'+ ' if amount >= 0 else ''} ${amount:.2f} "
              f"(Total: ${self.realized_pnl:.2f})")

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit was hit"""
        if self.daily_pnl <= -self.max_daily_loss:
            print(f"[BalanceManager] ⚠️  Daily loss limit reached: "
                  f"-${abs(self.daily_pnl):.2f}")
            return True
        return False

    def reset_daily_pnl(self):
        """Reset daily P&L counter (call at start of new day)"""
        print(f"[BalanceManager] Resetting daily P&L (was: ${self.daily_pnl:.2f})")
        self.daily_pnl = 0.0

    async def log_status(
        self,
        total_position_value: float,
        unrealized_pnl: float,
        position_count: int
    ):
        """Log current balance status"""
        available = await self.get_available_balance()
        exposure = self.get_total_exposure(total_position_value)

        print(f"\n[BalanceManager] === Balance Status ===")
        print(f"  Available: ${available:.2f}")
        print(f"  In Positions: ${total_position_value:.2f} ({position_count} positions)")
        print(f"  Exposure: {exposure*100:.1f}% / {self.max_exposure*100:.0f}%")
        print(f"  Unrealized P&L: ${unrealized_pnl:.2f}")
        print(f"  Realized P&L: ${self.realized_pnl:.2f}")
        print(f"  Daily P&L: ${self.daily_pnl:.2f}")
        print("=" * 50)

    def get_balance_info(self) -> dict:
        """Get balance information as dict"""
        return {
            "initial_capital": self.initial_capital,
            "realized_pnl": self.realized_pnl,
            "daily_pnl": self.daily_pnl,
            "max_daily_loss": self.max_daily_loss,
            "max_exposure": self.max_exposure
        }
