"""
Unified Trader - Wrapper que unifica PaperTrader y RealTrader.
Permite cambiar entre paper y real con un simple bool.
"""

import os
import sys
from typing import Optional, Dict, Any

# Add parent directory to path for PaperTrader import (need to go up 2 levels: src/real_trader -> src -> root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import database for dual-write
try:
    import database as db
    DB_AVAILABLE = db.is_db_available()
except ImportError:
    DB_AVAILABLE = False

# Import Telegram notifier
try:
    from src.notifications.telegram_notifier import TelegramNotifier
except ImportError:
    # Fallback if not found
    TelegramNotifier = None

try:
    from .auth import PolyAuth
    from .balance_manager import BalanceManager
    from .order_manager import OrderManager
    from .position_tracker import PositionTracker
    from .models import OrderRequest, Side, OrderType
except ImportError:
    from auth import PolyAuth
    from balance_manager import BalanceManager
    from order_manager import OrderManager
    from position_tracker import PositionTracker
    from models import OrderRequest, Side, OrderType

# Import PaperTrader from main bot (now we can access it from root)
from main import PaperTrader


class UnifiedTrader:
    """
    Wrapper unificado para Paper y Real trading.

    Usage:
        # Paper trading
        trader = UnifiedTrader(use_real=False, initial_cash=1000.0)

        # Real trading
        trader = UnifiedTrader(use_real=True)
    """

    def __init__(self, use_real: bool = False, initial_cash: float = 1000.0):
        self.use_real = use_real

        # Setup logging paths (same as PaperTrader)
        self.logs_dir = "logs"
        self.trade_log_path = os.path.join(self.logs_dir, "trade_history.csv")
        self._ensure_log_header()

        # Token metadata cache: token_id → (market_title, bucket)
        self._token_metadata = {}

        # Position sync cache to avoid excessive API calls
        self._last_position_sync = 0  # timestamp
        self._position_sync_interval = 5  # seconds

        if use_real:
            print("[UnifiedTrader] 🔴 REAL TRADING MODE - Using real money!")
            self.auth = PolyAuth()
            self.balance_mgr = BalanceManager(self.auth)
            self.order_mgr = OrderManager(self.auth, self.balance_mgr)
            self.position_tracker = PositionTracker(self.auth)
            # Setup Telegram notifications (ONLY in real mode)
            self.telegram = TelegramNotifier() if TelegramNotifier else None
            self._paper_trader = None
        else:
            print("[UnifiedTrader] 📄 PAPER TRADING MODE - Simulation only")
            self._paper_trader = PaperTrader(initial_cash=initial_cash)
            self.auth = None
            self.balance_mgr = None
            self.order_mgr = None
            self.position_tracker = None
            self.telegram = None  # No Telegram in paper mode

    def initialize(self):
        """Initialize trader (sync wrapper)"""
        if self.use_real:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                raise RuntimeError("Cannot call initialize() from async context")
            except RuntimeError:
                asyncio.run(self._initialize_real())
        else:
            print("[UnifiedTrader] ✅ Paper trader initialized")

    async def _sync_positions_cached(self, force: bool = False):
        """Sync positions with cache to avoid excessive API calls"""
        import time
        current_time = time.time()

        # Only sync if cache expired or forced
        if force or (current_time - self._last_position_sync) > self._position_sync_interval:
            await self.position_tracker.sync_positions()
            self._last_position_sync = current_time

    async def _initialize_real(self):
        """Initialize real trader components"""
        await self.balance_mgr.initialize()
        await self._sync_positions_cached(force=True)  # Force on init
        await self.order_mgr.sync_open_orders()

        # Get initial balance
        balance = await self.balance_mgr.get_available_balance()
        positions = self.position_tracker.get_positions()

        # Send startup notification
        if self.telegram:
            self.telegram.notify_startup(
                mode="REAL",
                balance=balance,
                positions=len(positions)
            )

        print("[UnifiedTrader] ✅ Real trader initialized")

    def execute(
        self,
        market_title: str,
        bucket: str,
        signal: str,
        price: float,
        reason: str = "Manual",
        strategy_tag: str = "STANDARD",
        hours_left: Optional[float] = None,
        tweet_count: Optional[int] = None,
        market_consensus: Optional[float] = None,
        entry_z_score: Optional[float] = None
    ) -> Optional[str]:
        """
        Execute a trade (paper or real).

        Args:
            market_title: Market title/event slug
            bucket: Bucket identifier (e.g., "300-319")
            signal: "BUY", "SELL", "ROTATE", etc.
            price: Price to execute at
            reason: Reason for the trade
            strategy_tag: "STANDARD", "MOONSHOT", "LOTTO", "HEDGE"

        Returns:
            Result message or None
        """
        if self.use_real:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                raise RuntimeError("Cannot call execute() from async context")
            except RuntimeError:
                return asyncio.run(self._execute_real(
                    market_title, bucket, signal, price, reason, strategy_tag,
                    hours_left, tweet_count, market_consensus, entry_z_score
                ))
        else:
            # Paper mode - PaperTrader handles its own logging
            return self._paper_trader.execute(
                market_title, bucket, signal, price, reason, strategy_tag,
                hours_left, tweet_count, market_consensus, entry_z_score
            )

    async def _execute_real(
        self,
        market_title: str,
        bucket: str,
        signal: str,
        price: float,
        reason: str,
        strategy_tag: str,
        hours_left: Optional[float] = None,
        tweet_count: Optional[int] = None,
        market_consensus: Optional[float] = None,
        entry_z_score: Optional[float] = None
    ) -> Optional[str]:
        """Execute real trade on blockchain"""

        pos_id = f"{market_title}|{bucket}"

        # --- BUY Logic ---
        if "BUY" in signal or "HEDGE" in signal:
            # 0. Get token_id FIRST (before position check)
            token_id = await self._resolve_token_id(market_title, bucket, side="YES")
            if not token_id:
                print(f"[UnifiedTrader] ❌ Could not resolve token_id for {market_title} {bucket}")
                return None

            # 1. Sync positions and check if position already exists BY TOKEN_ID
            if self.use_real:
                # Sync from Polymarket API (cached)
                await self._sync_positions_cached()

                # Check if position already exists for this token_id
                positions = self.position_tracker.get_positions()
                for pos in positions:
                    # Match by token_id (most reliable)
                    if pos.token_id == token_id:
                        print(f"[UnifiedTrader] ⚠️  Position already exists: {bucket} (token: {token_id[:20]}...)")
                        return None
            else:
                # Paper mode: check local portfolio
                if pos_id in self._paper_trader.portfolio["positions"]:
                    print(f"[UnifiedTrader] ⚠️  Position already exists: {pos_id}")
                    return None

            # 2. Parse edge value from reason (Kelly criterion)
            edge_value = None
            if "Val+" in reason:
                try:
                    edge_value = float(reason.split("Val+")[1].split()[0])
                except:
                    pass

            # 3. Calculate bet size using balance_mgr
            is_hedge = "HEDGE" in signal
            bet_amount, shares = await self.balance_mgr.calculate_bet_size(
                price=price,
                strategy_tag=strategy_tag,
                edge_value=edge_value,
                is_hedge=is_hedge
            )

            # 4. Check if we have enough balance
            available = await self.balance_mgr.get_available_balance()
            if bet_amount > available:
                print(f"[UnifiedTrader] ❌ Insufficient balance: need ${bet_amount:.2f}, have ${available:.2f}")
                return None

            # 5. Create and place order (FOK = Fill or Kill, market order)
            order_request = OrderRequest(
                token_id=token_id,
                price=price,  # Ask price from bot
                size=shares,
                side=Side.BUY,
                order_type=OrderType.FOK,  # Market order - execute immediately or cancel
                event_slug=market_title,
                range_label=bucket,
                market_title=market_title,
                token_side="YES"
            )

            result = await self.order_mgr.place_order(order_request)

            if result.success:
                # Cache token metadata for display purposes
                self._token_metadata[token_id] = {
                    'market_title': market_title,
                    'bucket': bucket
                }

                # Get cash after trade
                cash_after = await self.balance_mgr.get_available_balance()

                # Log trade (with context for dual-write)
                self._log_trade(
                    action="BUY",
                    market=market_title,
                    bucket=bucket,
                    price=price,
                    shares=shares,
                    reason=reason,
                    pnl=0.0,
                    cash_after=cash_after,
                    strategy=strategy_tag,
                    hours_left=hours_left,
                    tweet_count=tweet_count,
                    market_consensus=market_consensus
                )

                # Write position to DB (shadow mode)
                if DB_AVAILABLE:
                    pos_data = {
                        "shares": shares,
                        "entry_price": price,
                        "market": market_title,
                        "bucket": bucket,
                        "timestamp": result.created_at if hasattr(result, 'created_at') else None,
                        "invested": bet_amount,
                        "strategy_tag": strategy_tag,
                        "token_id": token_id,
                        "entry_z_score": entry_z_score
                    }
                    pos_id = f"{market_title}|{bucket}"
                    db.shadow_write(db.upsert_position, pos_id, pos_data, mode="REAL")

                # Send Telegram notification
                if self.telegram:
                    # Calculate total invested after this trade
                    positions = self.position_tracker.get_positions()
                    total_invested = sum(p.size * p.avg_entry_price for p in positions) + bet_amount

                    self.telegram.notify_trade_buy(
                        market=market_title,
                        bucket=bucket,
                        price=price,
                        shares=shares,
                        amount=bet_amount,
                        reason=reason,
                        balance=cash_after,
                        invested=total_invested,
                        strategy=strategy_tag,
                        mode="REAL"
                    )

                print(f"[UnifiedTrader] ✅ BUY: ${bet_amount:.2f} ({shares:.2f} shares) - Order {result.order_id}")
                return f"✅ BUY: ${bet_amount:.2f}"
            else:
                print(f"[UnifiedTrader] ❌ Order failed: {result.error}")
                return None

        # --- SELL/ROTATE Logic ---
        elif "SELL" in signal or "ROTATE" in signal or "DUMP" in signal:
            # 1. Find position in position_tracker
            positions = self.position_tracker.get_positions()
            position = None

            for pos in positions:
                if bucket in pos.range_label and market_title in pos.event_slug:
                    position = pos
                    break

            if not position:
                print(f"[UnifiedTrader] ⚠️  No position found for {bucket}")
                return None

            # 2. Create sell order (FOK = Fill or Kill, market order)
            order_request = OrderRequest(
                token_id=position.token_id,
                price=price,
                size=position.size,
                side=Side.SELL,
                order_type=OrderType.FOK,  # Market order - execute immediately or cancel
                event_slug=position.event_slug,
                range_label=position.range_label,
                market_title=market_title,
                token_side=position.token_side
            )

            result = await self.order_mgr.place_order(order_request)

            if result.success:
                # Calculate P&L
                revenue = position.size * price
                cost = position.size * position.avg_entry_price
                profit = revenue - cost

                # Track realized P&L
                self.balance_mgr.add_realized_pnl(profit)

                # Get cash after trade
                cash_after = await self.balance_mgr.get_available_balance()

                # Log trade (with context for dual-write)
                self._log_trade(
                    action="SELL",
                    market=market_title,
                    bucket=bucket,
                    price=price,
                    shares=position.size,
                    reason=reason,
                    pnl=profit,
                    cash_after=cash_after,
                    strategy=strategy_tag,
                    hours_left=hours_left,
                    tweet_count=tweet_count,
                    market_consensus=market_consensus
                )

                # Delete position from DB (shadow mode)
                if DB_AVAILABLE:
                    pos_id = f"{market_title}|{bucket}"
                    db.shadow_write(db.delete_position, pos_id, mode="REAL")

                # Send Telegram notification
                if self.telegram:
                    self.telegram.notify_trade_sell(
                        market=market_title,
                        bucket=bucket,
                        price=price,
                        shares=position.size,
                        pnl=profit,
                        pnl_pct=(profit / cost) * 100 if cost > 0 else 0.0,
                        balance=cash_after,
                        reason=reason,
                        mode="REAL"
                    )

                print(f"[UnifiedTrader] 💰 SELL: P&L ${profit:.2f} - Order {result.order_id}")
                return f"💰 SELL: P&L ${profit:.2f}"
            else:
                print(f"[UnifiedTrader] ❌ Order failed: {result.error}")
                return None

        return None

    async def _resolve_token_id(
        self,
        market_title: str,
        bucket: str,
        side: str = "YES"
    ) -> Optional[str]:
        """
        Resolve bucket identifier to token_id.

        Uses lazy resolution: queries Gamma API on-demand.

        Args:
            market_title: Event title (may need slug conversion)
            bucket: Bucket label like "300-319"
            side: "YES" or "NO"

        Returns:
            Token ID (decimal string) or None if not found
        """
        import requests
        import json

        try:
            # 1. Search for event in Gamma API
            response = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={
                    "limit": 100,
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "order": "volume24hr",
                    "ascending": "false"
                },
                timeout=10
            )

            if response.status_code != 200:
                print(f"[UnifiedTrader] ⚠️  Gamma API error: {response.status_code}")
                return None

            events = response.json()

            # 2. Find matching event using same logic as elon_auto_bot_threads.py
            import re

            def titles_match_paranoid(t1, t2):
                """Same logic as paper trader - extract numbers and compare"""
                t1 = t1.lower()
                t2 = t2.lower()

                # Simple containment check
                if t1 in t2 or t2 in t1:
                    return True

                # Extract numbers (excluding years)
                def get_nums(txt):
                    return {n for n in re.findall(r'\d+', txt) if n not in ['2024', '2025', '2026']}

                # Need at least 2 numbers in common (e.g., Feb 19-21 vs Feb 13-20)
                return len(get_nums(t1).intersection(get_nums(t2))) >= 2

            target_event = None

            for event in events:
                event_title = event.get("title", "")

                # Filter: must contain "elon" and "tweets"
                if "elon" not in event_title.lower() or "tweets" not in event_title.lower():
                    continue

                # Use paranoid matching to ensure correct event
                if titles_match_paranoid(market_title, event_title):
                    target_event = event
                    break

            if not target_event:
                print(f"[UnifiedTrader] ⚠️  Event not found: {market_title}")
                print(f"[UnifiedTrader] 📋 Available Elon events:")
                for i, event in enumerate(events[:5], 1):
                    title = event.get("title", "")
                    if "elon" in title.lower() and "tweets" in title.lower():
                        print(f"     {i}. {title[:80]}...")
                return None

            print(f"[UnifiedTrader] 🔍 Found event: {target_event.get('title', 'Unknown')}")
            print(f"[UnifiedTrader] 🔍 Looking for bucket: {bucket}")

            # 3. Find market for this bucket
            markets = target_event.get("markets", [])

            # Extract bucket numbers for flexible matching
            bucket_parts = bucket.split("-")
            bucket_min = bucket_parts[0] if len(bucket_parts) > 0 else ""
            bucket_max = bucket_parts[1] if len(bucket_parts) > 1 else ""

            for market in markets:
                question = market.get("question", "").lower()

                # Multiple matching strategies
                match_found = False

                # Strategy 1: Exact bucket match "300-319"
                if bucket.lower() in question:
                    match_found = True

                # Strategy 2: With spaces "300 - 319"
                elif bucket.replace("-", " - ").lower() in question:
                    match_found = True

                # Strategy 3: With "to" "300 to 319"
                elif f"{bucket_min} to {bucket_max}".lower() in question:
                    match_found = True

                # Strategy 4: Both numbers present in question
                elif bucket_min in question and bucket_max in question:
                    match_found = True

                if match_found:
                    # Found the market!
                    clob_token_ids = market.get("clobTokenIds", [])

                    # Parse if JSON string
                    if isinstance(clob_token_ids, str):
                        clob_token_ids = json.loads(clob_token_ids)

                    if clob_token_ids and len(clob_token_ids) >= 1:
                        # First token is YES, second is NO
                        token_index = 0 if side == "YES" else 1
                        token_id = clob_token_ids[token_index] if len(clob_token_ids) > token_index else clob_token_ids[0]

                        print(f"[UnifiedTrader] ✅ Resolved token: {bucket} → {token_id[:20]}...")
                        return str(token_id)

            # Debug: show available markets
            print(f"[UnifiedTrader] ⚠️  Bucket not found: {bucket}")
            print(f"[UnifiedTrader] 📋 Available markets in event:")
            for i, market in enumerate(markets[:5]):  # Show first 5
                q = market.get("question", "N/A")
                print(f"     {i+1}. {q[:80]}...")
            return None

        except Exception as e:
            print(f"[UnifiedTrader] ❌ Token resolution failed: {e}")
            return None

    def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio information (sync wrapper)"""
        if self.use_real:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                raise RuntimeError("Cannot call get_portfolio() from async context")
            except RuntimeError:
                return asyncio.run(self._get_portfolio_real())
        else:
            return self._paper_trader.portfolio

    async def _get_portfolio_real(self) -> Dict[str, Any]:
        """Get real portfolio from blockchain"""
        # 1. Get cash balance
        cash = await self.balance_mgr.get_available_balance()

        # 2. Sync positions and update prices (cached)
        await self._sync_positions_cached()
        await self.position_tracker.update_current_prices()

        # 3. Get positions
        positions = self.position_tracker.get_positions()

        # 4. Convert to PaperTrader format
        portfolio_positions = {}

        for pos in positions:
            pos_id = f"{pos.event_slug}|{pos.range_label}"
            portfolio_positions[pos_id] = {
                "shares": pos.size,
                "entry_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "market": pos.event_slug,
                "bucket": pos.range_label,
                "timestamp": pos.timestamp,
                "invested": pos.size * pos.avg_entry_price,
                "strategy_tag": "STANDARD"  # TODO: track this in Position model
            }

        return {
            "cash": cash,
            "positions": portfolio_positions,
            "history": []  # Not tracking history in real mode yet
        }

    def print_summary(self, current_prices_data):
        """Print portfolio summary (sync wrapper)"""
        if self.use_real:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                raise RuntimeError("Cannot call print_summary() from async context")
            except RuntimeError:
                asyncio.run(self._print_summary_real(current_prices_data))
        else:
            self._paper_trader.print_summary(current_prices_data)

    async def _print_summary_real(self, current_prices_data):
        """Print real portfolio summary"""
        # Sync positions from API (cached, max once per 5 seconds)
        # This fetches: size, avgPrice, curPrice, cashPnl (all correct from Polymarket)
        await self._sync_positions_cached()

        # Get data
        cash = await self.balance_mgr.get_available_balance()
        positions = self.position_tracker.get_positions()
        total_invested = sum(p.size * p.avg_entry_price for p in positions)
        total_pnl = self.position_tracker.get_total_unrealized_pnl()

        print("\n💼 --- REAL PORTFOLIO ---")
        print(f"   {'EVENT':<20} | {'BUCKET':<10} | {'SHARES':>7} | {'AVG':>4} | {'NOW':>4} | {'VALUE':>8} | {'P&L $':>7} | {'P&L %':>7}")
        print("   " + "-" * 100)

        for pos in positions:
            # Try to get metadata from cache (market_title + bucket)
            metadata = self._token_metadata.get(pos.token_id)

            if metadata:
                # Use cached data
                event_label = self._extract_date_label(metadata['market_title'])
                bucket = metadata['bucket']
            else:
                # Fallback: resolve from API (for positions that existed before this session)
                # This is a sync method calling sync helper
                resolved = self._resolve_position_display_sync(pos.token_id, pos.event_slug)
                event_label = resolved['event_label']
                bucket = resolved['bucket']

                # Cache it for next time
                self._token_metadata[pos.token_id] = {
                    'market_title': resolved['market_title'],
                    'bucket': bucket
                }

            # Calculate position value
            # VALUE = shares × current_price (what you'd get if you sold now)
            position_value = pos.size * pos.current_price

            # COST = VALUE - P&L (reverse calculation from API's cashPnl which includes fees)
            # This gives us the actual cost including fees
            cost_with_fees = position_value - pos.unrealized_pnl

            # P&L percentage based on actual cost with fees
            pnl_pct = (pos.unrealized_pnl / cost_with_fees * 100) if cost_with_fees > 0 else 0.0

            # Convert prices to cents for display (0.46 -> 46¢)
            avg_cents = pos.avg_entry_price * 100
            now_cents = pos.current_price * 100

            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            print(f"   {event_label:<20} | {bucket:<10} | {pos.size:>7.1f} | {avg_cents:>3.0f}¢ | {now_cents:>3.0f}¢ | ${position_value:>7.2f} | {pnl_sign}${pos.unrealized_pnl:>6.2f} | {pnl_sign}{pnl_pct:>6.2f}%")

        print("   " + "-" * 100)
        pnl_sign = "+" if total_pnl >= 0 else ""
        print(f"   💵 Cash: ${cash:.2f}  |  📈 Equity: ${cash + total_invested:.2f}  |  💰 Total P&L: {pnl_sign}${total_pnl:.2f}")

    def _resolve_position_display_sync(self, token_id: str, event_slug: str) -> dict:
        """
        Resolve position display info from token_id by querying Gamma API.
        Returns dict with: market_title, event_label, bucket
        """
        import requests
        import json

        try:
            print(f"[UnifiedTrader] 🔍 Resolving display for token: {token_id[:20]}...")

            # Query Gamma API for events (including closed ones)
            response = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={
                    "limit": 200,  # Increased limit
                    "closed": "false",  # Try without archived filter
                    "order": "volume24hr",
                    "ascending": "false"
                },
                timeout=10
            )

            if response.status_code != 200:
                print(f"[UnifiedTrader] ❌ API error: {response.status_code}")
                raise Exception(f"API error: {response.status_code}")

            events = response.json()
            print(f"[UnifiedTrader] 📊 Checking {len(events)} events...")

            # Find event that contains this token_id
            for event in events:
                event_title = event.get("title", "")
                markets = event.get("markets", [])

                for market in markets:
                    clob_token_ids = market.get("clobTokenIds", [])
                    if isinstance(clob_token_ids, str):
                        clob_token_ids = json.loads(clob_token_ids)

                    # Convert to strings for comparison
                    clob_token_ids = [str(t) for t in clob_token_ids]

                    # Check if our token_id is in this market
                    if str(token_id) in clob_token_ids:
                        # Found it!
                        market_title = event.get("title", "Unknown")
                        question = market.get("question", "")

                        print(f"[UnifiedTrader] ✅ Found: {market_title}")
                        print(f"[UnifiedTrader] 📝 Question: {question}")

                        # Extract bucket from question (e.g., "280-299 tweets")
                        import re
                        bucket_match = re.search(r'(\d+)-(\d+)', question)
                        bucket = f"{bucket_match.group(1)}-{bucket_match.group(2)}" if bucket_match else question[:10]

                        print(f"[UnifiedTrader] 🎯 Bucket: {bucket}")

                        return {
                            'market_title': market_title,
                            'event_label': self._extract_date_label(market_title),
                            'bucket': bucket
                        }

            # Not found in active events
            print(f"[UnifiedTrader] ⚠️  Token not found in {len(events)} events")
            return {
                'market_title': event_slug,
                'event_label': event_slug[:20],
                'bucket': 'N/A'
            }

        except Exception as e:
            print(f"[UnifiedTrader] ❌ Failed to resolve position display: {e}")
            import traceback
            traceback.print_exc()
            return {
                'market_title': event_slug,
                'event_label': event_slug[:20],
                'bucket': 'N/A'
            }

    def _extract_date_label(self, market_title: str) -> str:
        """Extract date range from market title (e.g., 'Feb13-Feb20')"""
        import re

        # Try to extract date pattern like "February 13 - February 20"
        match = re.search(r'(\w+)\s+(\d+)\s*-\s*(\w+)\s+(\d+)', market_title)
        if match:
            month1, day1, month2, day2 = match.groups()
            # Shorten month names and remove spaces
            month1_short = month1[:3]
            month2_short = month2[:3]
            return f"{month1_short}{day1}-{month2_short}{day2}"

        # Fallback: return first 12 chars
        return market_title[:12] if len(market_title) > 12 else market_title

    def _ensure_log_header(self):
        """Ensure trade log CSV has header"""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        if not os.path.exists(self.trade_log_path):
            with open(self.trade_log_path, "w", encoding='utf-8') as f:
                f.write("Timestamp,Action,Market,Bucket,Price,Shares,Reason,PnL,Cash_After,Mode\n")

    def _log_trade(self, action: str, market: str, bucket: str, price: float,
                   shares: float, reason: str, pnl: float = 0.0, cash_after: float = 0.0,
                   strategy: str = "STANDARD", hours_left: Optional[float] = None,
                   tweet_count: Optional[int] = None, market_consensus: Optional[float] = None):
        """Log trade to CSV and DB (dual-write)"""
        from datetime import datetime

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_clean = market.replace(",", "")
        reason_clean = reason.replace(",", ".")
        mode = "REAL" if self.use_real else "PAPER"

        # 1. Escribir a CSV (OBLIGATORIO - fuente de verdad)
        row = f"{ts},{action},{market_clean},{bucket},{price:.3f},{shares:.1f},{reason_clean},{pnl:.2f},{cash_after:.2f},{mode}\n"

        with open(self.trade_log_path, "a", encoding='utf-8') as f:
            f.write(row)

        # 2. Escribir a DB en shadow mode (OPCIONAL - no bloquea si falla)
        if DB_AVAILABLE:
            db.shadow_write(
                db.log_trade,
                action=action,
                market=market,
                bucket=bucket,
                price=price,
                shares=shares,
                reason=reason,
                pnl=pnl,
                cash_after=cash_after,
                mode=mode,
                strategy=strategy,
                hours_left=hours_left,
                tweet_count=tweet_count,
                market_consensus=market_consensus
            )

        print(f"[UnifiedTrader] 📝 Logged: {action} {bucket} @ ${price:.3f}")

    @property
    def portfolio(self):
        """Access to portfolio (for compatibility)"""
        return self.get_portfolio()


# Example usage
if __name__ == "__main__":
    # Test paper mode (now fully sync)
    print("\n=== Testing Paper Mode ===")
    paper_trader = UnifiedTrader(use_real=False, initial_cash=1000.0)
    paper_trader.initialize()

    result = paper_trader.execute(
        market_title="Elon tweets Feb 13-20",
        bucket="300-319",
        signal="BUY",
        price=0.20,
        reason="Test trade",
        strategy_tag="STANDARD"
    )
    print(result)

    # Test real mode (but don't execute)
    print("\n=== Testing Real Mode (Init Only) ===")
    real_trader = UnifiedTrader(use_real=True)
    real_trader.initialize()
    print("Real trader initialized successfully")
