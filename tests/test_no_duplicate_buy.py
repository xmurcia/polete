"""
Test: prevent duplicate BUY in real trading mode.

Scenario reproduced from production logs:
  17:00:03  Cycle 1 → BUY executed (bucket 360-379)
  17:00:10  Cycle 2 → SAME BUY fired again (bug: double spend)

Root cause: after a successful BUY the Polymarket data API does not
immediately reflect the new position.  When the next cycle calls
get_portfolio() it triggers sync_positions() which clears the
in-memory position dict, so the bucket appears un-owned and a second
BUY is placed.

The fix must block the duplicate even when:
  1. sync_positions() is called between cycles (wipes add_position())
  2. The API still returns an empty position list on the second sync
"""

import sys
import os
import types
import importlib.util
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Helpers to add stub modules BEFORE any project code is imported
# ---------------------------------------------------------------------------

def _stub(name, **attrs):
    """Register a stub module in sys.modules, setting attrs on it."""
    mod = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_stub("py_clob_client")
_stub("py_clob_client.client",
      ClobClient=MagicMock)
_stub("py_clob_client.clob_types",
      ApiCreds=MagicMock,
      OrderArgs=MagicMock,
      MarketOrderArgs=MagicMock,
      PartialCreateOrderOptions=MagicMock,
      BalanceAllowanceParams=MagicMock,
      AssetType=MagicMock)
_stub("eth_account")
_stub("eth_account.messages", encode_defunct=MagicMock())
_stub("web3",    Web3=MagicMock)
_stub("web3.middleware")
_stub("telegram")
_stub("telegram.ext")
_stub("dotenv",  load_dotenv=lambda: None)
_stub("scipy")
_stub("scipy.stats", norm=MagicMock())
_stub("numpy")
_stub("pandas")
_stub("main",  PaperTrader=MagicMock)

# ---------------------------------------------------------------------------
# Load modules directly from their .py paths (bypass __init__ chain)
# ---------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load(module_name, rel_path):
    """Import a single .py file without going through the package __init__."""
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(ROOT, rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load models first (no external deps other than our stubs)
models_mod = _load("real_trader.models", "src/real_trader/models.py")
Position  = models_mod.Position
Side      = models_mod.Side
OrderType = models_mod.OrderType
OrderResult = models_mod.OrderResult

# Now load unified_trader (it imports from .models, .auth, etc. — all stubbed)
# First stub the sub-modules it references so relative imports resolve
_stub("real_trader",
      models=models_mod,
      Position=Position,
      Side=Side,
      OrderType=OrderType,
      OrderResult=OrderResult)

# Stub the sub-module imports unified_trader does
_stub("real_trader.auth",            PolyAuth=MagicMock)
_stub("real_trader.balance_manager", BalanceManager=MagicMock)
_stub("real_trader.order_manager",   OrderManager=MagicMock)
_stub("real_trader.position_tracker",PositionTracker=MagicMock)

# Stub database
_stub("database", is_db_available=lambda: False, shadow_write=MagicMock())

# Stub src.notifications.telegram_notifier
_stub("src")
_stub("src.notifications")
_stub("src.notifications.telegram_notifier", TelegramNotifier=None)
_stub("src.utils.error_logger", get_error_logger=None)
_stub("src.utils")

# Load unified_trader.py directly
ut_mod = _load("real_trader.unified_trader",
               "src/real_trader/unified_trader.py")
UnifiedTrader = ut_mod.UnifiedTrader

# Force these module-level flags off so DB/Telegram branches are skipped
ut_mod.DB_AVAILABLE      = False
ut_mod.TelegramNotifier  = None
ut_mod.get_error_logger  = None

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
TOKEN_ID  = "9990000000000000000000000000000000000000000000000000000000000001"
MARKET    = "Elon Musk # tweets February 17 - February 24, 2026?"
BUCKET    = "360-379"
ASK_PRICE = 0.31
SHARES    = 16.1
BET_AMT   = 5.0


# ---------------------------------------------------------------------------
# Build a minimal fake UnifiedTrader (bypasses __init__)
# ---------------------------------------------------------------------------

def _make_trader():
    trader = object.__new__(UnifiedTrader)
    trader.use_real = True
    trader.logs_dir = "/tmp"
    trader.trade_log_path = "/tmp/test_trade_history.csv"

    trader._token_metadata            = {}
    trader._token_id_forward_cache    = {f"{MARKET}|{BUCKET}": TOKEN_ID}
    trader._last_position_sync        = 0
    trader._position_sync_interval    = 5
    trader._pending_buy_token_ids     = {}   # guard that survives sync_positions() clears

    trader.balance_mgr = MagicMock()
    trader.balance_mgr.calculate_bet_size    = AsyncMock(return_value=(BET_AMT, SHARES))
    trader.balance_mgr.get_available_balance = AsyncMock(return_value=48.36)
    trader.balance_mgr.add_realized_pnl      = MagicMock()

    ok = MagicMock()
    ok.success  = True
    ok.order_id = "order-abc-123"
    ok.error    = None
    trader.order_mgr = MagicMock()
    trader.order_mgr.place_order = AsyncMock(return_value=ok)

    trader.telegram     = None
    trader.error_logger = None
    return trader


class _FakePositionTracker:
    """
    PositionTracker whose sync_positions() empties the dict — simulating
    Polymarket's data API not yet propagating the new position.
    """

    def __init__(self):
        self.positions: dict = {}
        self.entered_ranges: dict = {}

    async def sync_positions(self):
        self.positions.clear()
        self.entered_ranges.clear()

    def get_positions(self):
        return list(self.positions.values())

    def get_position(self, token_id):
        return self.positions.get(token_id)

    def add_position(self, token_id, event_slug, range_label, side, size,
                     entry_price, **kwargs):
        if token_id in self.positions:
            pos = self.positions[token_id]
            total = pos.size + size
            pos.avg_entry_price = (
                pos.size * pos.avg_entry_price + size * entry_price
            ) / total
            pos.size = total
        else:
            self.positions[token_id] = Position(
                token_id=token_id,
                event_slug=event_slug,
                range_label=range_label,
                side=Side.BUY,
                size=size,
                avg_entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0,
                timestamp=0,
                market_title=kwargs.get("market_title", ""),
                token_side=kwargs.get("token_side", "YES"),
            )

    def mark_range_entered(self, event_slug, range_label):
        self.entered_ranges.setdefault(event_slug, set()).add(range_label)


# ===========================================================================
# Test 1 — critical: duplicate BUY must be blocked (main bug)
# ===========================================================================

@pytest.mark.asyncio
async def test_no_duplicate_buy_after_api_sync():
    """
    Reproduce the exact race condition from the production Telegram logs.

    Sequence:
      1. Cycle 1 — first BUY executes
      2. Inter-cycle — sync_positions() clears in-memory positions
         (Polymarket API hasn't propagated the new position yet)
      3. Cycle 2 — same BUY attempted again → MUST be blocked
    """
    trader  = _make_trader()
    tracker = _FakePositionTracker()
    trader.position_tracker = tracker

    # Patch _sync_positions_cached to just call tracker.sync_positions()
    # so the test controls exactly when a sync wipes the state
    async def _fake_sync_cached(force=False):
        await tracker.sync_positions()
    trader._sync_positions_cached = _fake_sync_cached

    # ---------- Cycle 1: first BUY ----------
    result1 = await trader._execute_real(
        market_title=MARKET,
        bucket=BUCKET,
        signal="BUY",
        price=ASK_PRICE,
        reason="Val+0.16",
        strategy_tag="STANDARD",
    )

    assert result1 is not None, "First BUY should succeed"
    assert trader.order_mgr.place_order.call_count == 1, \
        "First BUY: place_order must be called exactly once"

    # ---------- Inter-cycle: get_portfolio() triggers API sync ----------
    # sync_positions() clears in-memory state; API hasn't propagated yet
    await tracker.sync_positions()

    assert len(tracker.positions) == 0, (
        "Simulated API sync must wipe in-memory positions "
        "(Polymarket data API lag)"
    )

    # ---------- Cycle 2: duplicate BUY attempt ----------
    result2 = await trader._execute_real(
        market_title=MARKET,
        bucket=BUCKET,
        signal="BUY",
        price=ASK_PRICE,
        reason="Val+0.16",
        strategy_tag="STANDARD",
    )

    assert result2 is None, (
        "Second BUY for the same bucket MUST be blocked even after the "
        "API sync wiped the in-memory position"
    )
    assert trader.order_mgr.place_order.call_count == 1, (
        f"place_order called {trader.order_mgr.place_order.call_count}x "
        "(expected 1) — duplicate BUY was NOT prevented!"
    )


# ===========================================================================
# Test 2 — sanity: first BUY always succeeds
# ===========================================================================

@pytest.mark.asyncio
async def test_first_buy_always_succeeds():
    trader  = _make_trader()
    tracker = _FakePositionTracker()
    trader.position_tracker = tracker

    async def _noop_sync(force=False):
        pass
    trader._sync_positions_cached = _noop_sync

    result = await trader._execute_real(
        market_title=MARKET,
        bucket=BUCKET,
        signal="BUY",
        price=ASK_PRICE,
        reason="Val+0.16",
        strategy_tag="STANDARD",
    )

    assert result is not None, "First BUY should always succeed"
    assert trader.order_mgr.place_order.call_count == 1


# ===========================================================================
# Test 3 — sanity: different buckets both succeed
# ===========================================================================

@pytest.mark.asyncio
async def test_different_buckets_both_succeed():
    TOKEN_B2 = TOKEN_ID[:-3] + "002"
    BUCKET_2 = "380-399"

    trader  = _make_trader()
    tracker = _FakePositionTracker()
    trader.position_tracker = tracker
    trader._token_id_forward_cache[f"{MARKET}|{BUCKET_2}"] = TOKEN_B2

    async def _noop_sync(force=False):
        pass
    trader._sync_positions_cached = _noop_sync

    r1 = await trader._execute_real(
        market_title=MARKET, bucket=BUCKET,
        signal="BUY", price=ASK_PRICE, reason="Val+0.16", strategy_tag="STANDARD",
    )
    r2 = await trader._execute_real(
        market_title=MARKET, bucket=BUCKET_2,
        signal="BUY", price=ASK_PRICE, reason="Val+0.12", strategy_tag="STANDARD",
    )

    assert r1 is not None, "First bucket BUY should succeed"
    assert r2 is not None, "Second (different) bucket BUY should succeed"
    assert trader.order_mgr.place_order.call_count == 2
