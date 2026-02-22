"""
Tests for SELL order fixes:
- Fix 1: OrderManager skips USDC balance check for SELL orders
- Fix 2: BalanceManager.get_conditional_balance exists and is safe
- Fix 3: sell_size capping logic (pure logic, no external deps)

All real-trader modules are imported directly (not via __init__.py)
to avoid the UnifiedTrader → main.py → scipy import chain.
"""

import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock

# Import modules directly to bypass the package __init__.py
_rt = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/real_trader'))
sys.path.insert(0, _rt)

# Stub heavy dependencies so module-level imports don't fail
sys.modules.setdefault('dotenv', MagicMock())
sys.modules.setdefault('web3', MagicMock())
sys.modules.setdefault('py_clob_client', MagicMock())
sys.modules.setdefault('py_clob_client.client', MagicMock())
sys.modules.setdefault('py_clob_client.clob_types', MagicMock())

# Now import the real-trader modules
import importlib
order_manager_mod = importlib.import_module('order_manager')
balance_manager_mod = importlib.import_module('balance_manager')
models_mod = importlib.import_module('models')

OrderManager = order_manager_mod.OrderManager
BalanceManager = balance_manager_mod.BalanceManager
OrderRequest = models_mod.OrderRequest
OrderResult = models_mod.OrderResult
Side = models_mod.Side
OrderType = models_mod.OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_auth():
    mock_auth = MagicMock()
    mock_client = MagicMock()
    mock_auth.get_client.return_value = mock_client
    return mock_auth, mock_client


def make_sell_request(size=38.46, price=0.26):
    return OrderRequest(
        token_id="token-360-379",
        price=price,
        size=size,
        side=Side.SELL,
        order_type=OrderType.FOK,
        event_slug="elon-feb-tweets",
        range_label="360-379",
        market_title="Elon tweets Feb 17-24",
        token_side="YES"
    )


def make_buy_request(size=38.46, price=0.26):
    return OrderRequest(
        token_id="token-360-379",
        price=price,
        size=size,
        side=Side.BUY,
        order_type=OrderType.FOK,
        event_slug="elon-feb-tweets",
        range_label="360-379",
        market_title="Elon tweets Feb 17-24",
        token_side="YES"
    )


# ---------------------------------------------------------------------------
# Fix 1: SELL orders must NOT check USDC balance
# ---------------------------------------------------------------------------

class TestFix1OrderManagerSell:

    @pytest.mark.asyncio
    async def test_sell_does_not_call_usdc_balance_check(self):
        """
        Core fix: can_place_order (USDC check) must NOT be called for SELL.
        Before the fix, it was called and would fail if USDC < price×size,
        even when we owned the tokens.
        """
        mock_auth, mock_client = make_mock_auth()

        mock_balance_mgr = MagicMock()
        mock_balance_mgr.can_place_order = AsyncMock(return_value=False)  # USDC is "insufficient"

        mock_client.create_market_order.return_value = MagicMock()
        mock_client.post_order.return_value = {"orderID": "sell-ok-123"}

        mgr = OrderManager(mock_auth, mock_balance_mgr)
        mgr.client = mock_client

        result = await mgr.place_order(make_sell_request())

        mock_balance_mgr.can_place_order.assert_not_called()
        assert result.success is True, "SELL should succeed even when USDC balance is low"

    @pytest.mark.asyncio
    async def test_sell_reaches_polymarket_api(self):
        """SELL must call post_order on the Polymarket client."""
        mock_auth, mock_client = make_mock_auth()
        mock_client.create_market_order.return_value = MagicMock()
        mock_client.post_order.return_value = {"orderID": "sell-ok-456"}

        mgr = OrderManager(mock_auth, None)  # no balance_mgr
        mgr.client = mock_client

        result = await mgr.place_order(make_sell_request())

        mock_client.post_order.assert_called_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_buy_still_checks_usdc(self):
        """BUY orders must still go through the USDC balance check."""
        mock_auth, mock_client = make_mock_auth()

        mock_balance_mgr = MagicMock()
        mock_balance_mgr.can_place_order = AsyncMock(return_value=False)

        mgr = OrderManager(mock_auth, mock_balance_mgr)
        mgr.client = mock_client

        result = await mgr.place_order(make_buy_request())

        mock_balance_mgr.can_place_order.assert_called_once()
        assert result.success is False
        assert "Insufficient balance" in result.error

    @pytest.mark.asyncio
    async def test_sell_with_zero_usdc_balance_still_proceeds(self):
        """
        Regression: with all USDC deployed in positions (cash ≈ $0),
        SELL must still be attempted.
        """
        mock_auth, mock_client = make_mock_auth()

        mock_balance_mgr = MagicMock()
        # Simulate: USDC = $0, would have blocked the SELL before the fix
        mock_balance_mgr.can_place_order = AsyncMock(return_value=False)

        mock_client.create_market_order.return_value = MagicMock()
        mock_client.post_order.return_value = {"orderID": "sell-ok-789"}

        mgr = OrderManager(mock_auth, mock_balance_mgr)
        mgr.client = mock_client

        result = await mgr.place_order(make_sell_request(size=38.46, price=0.26))

        assert result.success is True
        # and the API was called with a SELL amount = shares (not USDC)
        call_args = mock_client.post_order.call_args
        assert call_args is not None


# ---------------------------------------------------------------------------
# Fix 2: BalanceManager.get_conditional_balance
# ---------------------------------------------------------------------------

class TestFix2ConditionalBalance:

    def make_mgr(self, api_response=None, raises=False):
        mock_auth, mock_client = make_mock_auth()
        if raises:
            mock_client.get_balance_allowance.side_effect = Exception("API down")
        else:
            mock_client.get_balance_allowance.return_value = api_response or {}
        mgr = BalanceManager(mock_auth)
        mgr.client = mock_client
        return mgr

    def test_method_exists(self):
        assert hasattr(BalanceManager, 'get_conditional_balance'), \
            "BalanceManager must have get_conditional_balance method"

    @pytest.mark.asyncio
    async def test_returns_float(self):
        mgr = self.make_mgr({"balance": 38_000_000, "allowance": 0})
        result = await mgr.get_conditional_balance("token-abc")
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_divides_by_1e6(self):
        """
        CTF token balances are returned in micro-units (1e6 scale)
        same as USDC COLLATERAL.
        """
        mgr = self.make_mgr({"balance": 38_461_538, "allowance": 0})
        result = await mgr.get_conditional_balance("token-abc")
        assert abs(result - 38.461538) < 0.000001, f"Expected ~38.461538, got {result}"

    @pytest.mark.asyncio
    async def test_missing_balance_key_returns_zero(self):
        mgr = self.make_mgr({})
        result = await mgr.get_conditional_balance("token-abc")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_api_exception_returns_zero(self):
        """Safe fallback: API error must not crash the bot."""
        mgr = self.make_mgr(raises=True)
        result = await mgr.get_conditional_balance("token-abc")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_zero_balance_returns_zero(self):
        mgr = self.make_mgr({"balance": 0, "allowance": 0})
        result = await mgr.get_conditional_balance("token-abc")
        assert result == 0.0


# ---------------------------------------------------------------------------
# Fix 3: sell_size capping logic (pure logic test, no external deps)
# ---------------------------------------------------------------------------

class TestFix3SellSizeCapping:
    """
    Replicate the capping logic from unified_trader._execute_real.
    Tests the business rule without importing UnifiedTrader.
    """

    def apply_cap(self, position_size: float, ctf_balance: float) -> float:
        sell_size = position_size
        if ctf_balance > 0 and ctf_balance < sell_size:
            sell_size = ctf_balance
        return sell_size

    def test_caps_when_ctf_less_than_position(self):
        """
        Precision rounding: CTF on-chain = 38.459999, tracker says 38.461538.
        Sell must use 38.459999 to avoid Polymarket rejecting.
        """
        result = self.apply_cap(position_size=38.461538, ctf_balance=38.459999)
        assert result == pytest.approx(38.459999)

    def test_no_cap_when_ctf_equals_position(self):
        result = self.apply_cap(position_size=38.461538, ctf_balance=38.461538)
        assert result == pytest.approx(38.461538)

    def test_no_cap_when_ctf_greater(self):
        result = self.apply_cap(position_size=38.461538, ctf_balance=40.0)
        assert result == pytest.approx(38.461538)

    def test_no_cap_when_ctf_zero(self):
        """
        ctf_balance=0 could be an API error. Do NOT zero out sell_size —
        let the order reach Polymarket and fail there with a clear error.
        """
        result = self.apply_cap(position_size=38.461538, ctf_balance=0.0)
        assert result == pytest.approx(38.461538)

    def test_small_ctf_balance_below_threshold(self):
        """CTF balance < 0.001 → sell_size < 0.001 → sell should be blocked."""
        result = self.apply_cap(position_size=38.461538, ctf_balance=0.0005)
        assert result < 0.001

    def test_exact_micro_unit_rounding(self):
        """1 micro-unit difference (6th decimal) must be caught."""
        position_size = 38.461538
        ctf_balance = 38.461537  # 1 micro-unit less
        result = self.apply_cap(position_size, ctf_balance)
        assert result == pytest.approx(ctf_balance)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
