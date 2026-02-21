"""
Order Manager for Polymarket trading.
Handles order placement, cancellation, and tracking.
"""

import os
import time
from typing import Dict, Set, List, Optional
from dotenv import load_dotenv
from py_clob_client.clob_types import OrderArgs

try:
    from .models import OrderRequest, OrderResult, TrackedOrder, Side, OrderType
    from .auth import PolyAuth
except ImportError:
    from models import OrderRequest, OrderResult, TrackedOrder, Side, OrderType
    from auth import PolyAuth

load_dotenv()


class OrderManager:
    """Manages orders for Polymarket trading"""

    def __init__(self, auth: PolyAuth, balance_manager=None):
        self.auth = auth
        self.client = auth.get_client()
        self.balance_manager = balance_manager

        # Track orders in memory
        self.open_orders: Dict[str, TrackedOrder] = {}
        self.orders_by_event: Dict[str, Set[str]] = {}

        # Configuration
        self.max_positions_per_event = int(os.getenv("MAX_POSITIONS_PER_EVENT", "6"))
        self.max_exposure = float(os.getenv("MAX_EXPOSURE", "0.99"))

        print(f"[OrderManager] Initialized")

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order on Polymarket"""

        # DISABLED: Max positions per event check (to match PaperTrader behavior)
        # PaperTrader doesn't have this limit, so we remove it for consistency
        # if not request.is_stop_loss:
        #     event_orders = self.orders_by_event.get(request.event_slug, set())
        #     if len(event_orders) >= self.max_positions_per_event:
        #         print(f"[OrderManager] ⚠️  Max positions reached for {request.event_slug}")
        #         return OrderResult(
        #             success=False,
        #             error=f"Max positions per event: {self.max_positions_per_event}"
        #         )

        # DISABLED: Exposure limit check (to match PaperTrader behavior)
        # PaperTrader doesn't have this limit, will use Telegram alerts instead
        # if self.balance_manager and hasattr(request, 'current_total_position_value'):
        #     order_value = request.price * request.size
        #     projected_value = getattr(request, 'current_total_position_value', 0) + order_value
        #     exposure = self.balance_manager.get_total_exposure(projected_value)
        #
        #     if exposure > self.max_exposure:
        #         print(f"[OrderManager] ⚠️  Exposure limit exceeded: {exposure*100:.1f}%")
        #         return OrderResult(
        #             success=False,
        #             error=f"Exposure limit exceeded: {exposure*100:.1f}%"
        #         )

        # Check balance
        if self.balance_manager:
            order_value = request.price * request.size
            can_place = await self.balance_manager.can_place_order(order_value)
            if not can_place:
                print(f"[OrderManager] ⚠️  Insufficient balance")
                return OrderResult(success=False, error="Insufficient balance")

        # Place order
        try:
            print(f"[OrderManager] Placing {request.order_type} {request.side} order: {request.range_label}")

            # Adjust price for FOK BUY orders (add slippage buffer)
            price = request.price
            if request.order_type == OrderType.FOK and request.side == Side.BUY:
                slippage_buffer = price * 0.02
                price = price + slippage_buffer
                print(f"[OrderManager] FOK BUY: Added +2% slippage → {price*100:.2f}¢")

            # Enforce minimum price
            price = max(price, 0.001)

            # Round price to 2 decimals (maker amount requirement)
            price = round(price, 2)

            # Round size according to Polymarket requirements:
            # - BUY orders (taker amount): max 5 decimals
            # - SELL orders (maker amount): max 2 decimals
            if request.side == Side.BUY:
                size = round(request.size, 5)
            else:  # SELL
                size = round(request.size, 2)

            print(f"[OrderManager] Final values: price={price:.3f} (2 dec), size={size:.5f}")

            # Create and post order
            order_args = OrderArgs(
                token_id=request.token_id,
                price=price,
                size=size,
                side=request.side.value,
                fee_rate_bps=0
            )
            signed_order = self.client.create_order(order_args)

            result = self.client.post_order(signed_order, request.order_type.value)

            if not result or not result.get("orderID"):
                print(f"[OrderManager] ❌ Order failed: No orderID returned")
                return OrderResult(success=False, error="No orderID returned")

            order_id = result["orderID"]

            # Track order
            tracked = TrackedOrder(
                order_id=order_id,
                token_id=request.token_id,
                event_slug=request.event_slug,
                range_label=request.range_label,
                side=request.side,
                order_type=request.order_type,
                price=price,
                size=size,
                timestamp=int(time.time()),
                market_title=request.market_title,
                token_side=request.token_side
            )

            self.open_orders[order_id] = tracked

            if request.event_slug not in self.orders_by_event:
                self.orders_by_event[request.event_slug] = set()
            self.orders_by_event[request.event_slug].add(order_id)

            print(f"[OrderManager] ✅ Order placed: {order_id}")

            return OrderResult(success=True, order_id=order_id)

        except Exception as e:
            print(f"[OrderManager] ❌ Order failed: {e}")
            return OrderResult(success=False, error=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""

        try:
            await self.client.cancel_order({"orderID": order_id})
            self._remove_order(order_id)
            print(f"[OrderManager] ✅ Canceled: {order_id}")
            return True
        except Exception as e:
            print(f"[OrderManager] ❌ Cancel failed: {e}")
            return False

    def _remove_order(self, order_id: str):
        """Remove order from tracking"""
        order = self.open_orders.get(order_id)
        if order:
            del self.open_orders[order_id]

            event_orders = self.orders_by_event.get(order.event_slug)
            if event_orders:
                event_orders.discard(order_id)
                if not event_orders:
                    del self.orders_by_event[order.event_slug]

    async def sync_open_orders(self):
        """Sync orders from blockchain"""
        try:
            orders = self.client.get_orders()

            self.open_orders.clear()
            self.orders_by_event.clear()

            for order in orders:
                tracked = TrackedOrder(
                    order_id=order.get("id", order.get("orderID")),
                    token_id=order.get("asset_id", order.get("tokenID")),
                    event_slug="",
                    range_label="",
                    side=Side(order["side"]),
                    order_type=OrderType(order["type"]),
                    price=float(order["price"]),
                    size=float(order["size"]),
                    timestamp=int(time.time())
                )

                self.open_orders[tracked.order_id] = tracked

            print(f"[OrderManager] Synced {len(self.open_orders)} open orders")
        except Exception as e:
            print(f"[OrderManager] Sync failed: {e}")

    def get_open_orders(self) -> List[TrackedOrder]:
        """Get all open orders"""
        return list(self.open_orders.values())

    def get_orders_for_event(self, event_slug: str) -> List[TrackedOrder]:
        """Get orders for specific event"""
        order_ids = self.orders_by_event.get(event_slug, set())
        return [self.open_orders[oid] for oid in order_ids if oid in self.open_orders]

    def get_order_count(self, event_slug: Optional[str] = None) -> int:
        """Get order count (total or for event)"""
        if event_slug:
            return len(self.orders_by_event.get(event_slug, set()))
        return len(self.open_orders)
