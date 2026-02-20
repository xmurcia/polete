"""
Tape-Based Backtesting Engine

Uses real historical price snapshots from market_tape/*.json
Combined with actual tweet counts to simulate realistic trading.
"""

import json
import glob
import os
from datetime import datetime
from typing import List, Dict, Tuple
from ..strategies.base_strategy import BaseStrategy, MarketState, Position
from ..utils.portfolio import Portfolio
from .metrics import calculate_metrics, PerformanceMetrics


class TapeBacktest:
    """
    Backtests using real historical tape data.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 1000.0,
        tape_dir: str = "/Users/xavi.murcia/Desktop/poly-gemini/polete-volume-2/market_tape"
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.tape_dir = tape_dir
        self.portfolio = Portfolio(initial_cash=initial_capital)
        self.equity_curve = []
        self.trade_log = []

    def load_all_tapes(self) -> List[Dict]:
        """
        Load all tape files sorted chronologically.

        Returns:
            List of tape dicts with timestamp, meta, order_book
        """
        print(f"📂 Loading tapes from: {self.tape_dir}")

        tape_files = sorted(glob.glob(os.path.join(self.tape_dir, "tape_*.json")))

        if not tape_files:
            raise ValueError(f"No tape files found in {self.tape_dir}")

        tapes = []
        for tape_file in tape_files:
            try:
                with open(tape_file, 'r') as f:
                    tape = json.load(f)
                    tapes.append(tape)
            except Exception as e:
                print(f"⚠️ Error loading {os.path.basename(tape_file)}: {e}")

        print(f"✅ Loaded {len(tapes)} tape snapshots")

        if tapes:
            first_time = datetime.fromtimestamp(tapes[0]['timestamp'])
            last_time = datetime.fromtimestamp(tapes[-1]['timestamp'])
            print(f"   Period: {first_time} → {last_time}")

        return tapes

    def extract_market_state(self, tape: Dict, market_title: str) -> MarketState:
        """
        Extract MarketState for a specific market from tape.

        Args:
            tape: Tape snapshot dict
            market_title: Market title to extract

        Returns:
            MarketState or None if market not found
        """
        # Find market in meta
        market_meta = None
        for meta in tape.get('meta', []):
            if market_title.lower() in meta['title'].lower():
                market_meta = meta
                break

        if not market_meta:
            return None

        # Find order book for this market
        market_orderbook = None
        for ob in tape.get('order_book', []):
            if market_title.lower() in ob['title'].lower():
                market_orderbook = ob
                break

        if not market_orderbook:
            return None

        # Convert buckets format
        buckets = []
        for bucket in market_orderbook.get('buckets', []):
            buckets.append({
                'bucket': bucket['bucket'],
                'min': bucket['min'],
                'max': bucket['max'],
                'ask': bucket['ask'],
                'bid': bucket['bid']
            })

        return MarketState(
            title=market_meta['title'],
            count=market_meta['count'],
            hours_left=market_meta['hours'],
            daily_avg=market_meta['daily_avg'],
            buckets=buckets,
            timestamp=tape['timestamp']
        )

    def run(
        self,
        market_filter: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> PerformanceMetrics:
        """
        Run backtest on tape data.

        Args:
            market_filter: Only trade markets matching this string (e.g., "January 23")
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            PerformanceMetrics with results
        """
        print("\n🚀 TAPE-BASED BACKTEST")
        print("="*70)

        # Load all tapes
        tapes = self.load_all_tapes()

        # Filter by date if specified
        if start_date:
            start_ts = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
            tapes = [t for t in tapes if t['timestamp'] >= start_ts]

        if end_date:
            end_ts = datetime.strptime(end_date, "%Y-%m-%d").timestamp()
            tapes = [t for t in tapes if t['timestamp'] <= end_ts]

        if not tapes:
            raise ValueError("No tapes in specified date range")

        print(f"\n💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Strategy: {self.strategy.name}")
        print(f"📊 Snapshots to process: {len(tapes)}")
        if market_filter:
            print(f"🔍 Market filter: {market_filter}")
        print("="*70 + "\n")

        # Track which markets we've seen
        seen_markets = set()

        # Process each tape snapshot
        for i, tape in enumerate(tapes):
            timestamp = tape['timestamp']
            readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')

            # Get all active markets from this tape
            active_markets = []
            for meta in tape.get('meta', []):
                if not meta.get('active', True):
                    continue

                title = meta['title']

                # Apply market filter if specified
                if market_filter and market_filter.lower() not in title.lower():
                    continue

                active_markets.append(title)
                seen_markets.add(title)

            if not active_markets:
                continue

            # Process each market
            for market_title in active_markets:
                market_state = self.extract_market_state(tape, market_title)

                if not market_state or not market_state.buckets:
                    continue

                # Get current positions for this market
                current_positions = self._get_current_positions_for_market(market_title)

                # Update max_price_seen for all positions (for trailing stop)
                for pos in current_positions:
                    # Find current price for this position
                    for bucket in market_state.buckets:
                        if bucket['bucket'] == pos.bucket:
                            current_bid = bucket.get('bid', 0)
                            if current_bid > pos.max_price_seen:
                                pos.max_price_seen = current_bid
                                # Update in portfolio
                                self.portfolio.update_position_metadata(
                                    pos.market, 
                                    pos.bucket, 
                                    {'max_price_seen': current_bid}
                                )
                            break

                # Get signals from strategy
                signals = self.strategy.analyze(market_state, current_positions)

                # Debug: Log signal count
                if signals:
                    print(f"  🎯 [{readable_time}] {market_title[:50]} → {len(signals)} signal(s)")

                # Execute signals
                for signal in signals:
                    self._execute_signal(signal, market_state, readable_time)

            # Update equity curve (every 10 snapshots to reduce noise)
            if i % 10 == 0:
                current_prices = self._get_all_current_prices(tape)
                equity = self.portfolio.get_total_value(current_prices)

                self.equity_curve.append({
                    'timestamp': timestamp,
                    'date': readable_time,
                    'equity': equity,
                    'cash': self.portfolio.get_cash(),
                    'num_positions': len(self.portfolio.get_positions())
                })

                # Progress indicator
                progress = ((i + 1) / len(tapes)) * 100
                if i % 50 == 0:
                    print(f"Progress: {progress:.1f}% | {readable_time} | Equity: ${equity:,.2f} | Positions: {len(self.portfolio.get_positions())}")

        print(f"\n✅ Processed {len(tapes)} snapshots")
        print(f"📈 Markets tracked: {len(seen_markets)}")
        print(f"   {', '.join(sorted(seen_markets)[:5])}{'...' if len(seen_markets) > 5 else ''}")

        # Calculate metrics
        if not self.equity_curve:
            # Add final equity point
            final_equity = self.portfolio.get_total_value({})
            self.equity_curve.append({
                'timestamp': tapes[-1]['timestamp'],
                'date': datetime.fromtimestamp(tapes[-1]['timestamp']).strftime('%Y-%m-%d %H:%M'),
                'equity': final_equity,
                'cash': self.portfolio.get_cash(),
                'num_positions': len(self.portfolio.get_positions())
            })

        metrics = calculate_metrics(
            strategy_name=self.strategy.name,
            equity_curve=self.equity_curve,
            trades=self.portfolio.data['history'],
            initial_capital=self.initial_capital
        )

        print("\n" + "="*70)
        metrics.print_summary()

        return metrics

    def _get_current_positions_for_market(self, market_title: str) -> List[Position]:
        """Get positions for a specific market"""
        positions = []
        for pos_id, pos_data in self.portfolio.get_positions().items():
            if market_title.lower() in pos_data['market'].lower():
                positions.append(Position(
                    market=pos_data['market'],
                    bucket=pos_data['bucket'],
                    shares=pos_data['shares'],
                    entry_price=pos_data['entry_price'],
                    timestamp=pos_data['timestamp'],
                    invested=pos_data['invested'],
                    max_price_seen=pos_data.get('max_price_seen', pos_data['entry_price']),
                    price_history=pos_data.get('price_history', []),
                    strategy_tag=pos_data.get('strategy_tag', 'STANDARD')  # Load DNA tag
                ))
        return positions

    def _execute_signal(self, signal, market_state: MarketState, readable_time: str):
        """Execute trading signal"""
        if signal.type.value == 'BUY':
            # Calculate position size
            bet_amount = self.strategy.calculate_position_size(
                signal,
                self.portfolio.get_cash(),
                self.portfolio.get_total_value({})
            )

            if bet_amount < 5.0:
                print(f"  ⚠️ SKIP BUY (bet ${bet_amount:.2f} < $5 min) [{readable_time}] {signal.bucket}")
                return

            shares = bet_amount / signal.price
            success = self.portfolio.add_position(
                market=signal.market_title,
                bucket=signal.bucket,
                shares=shares,
                entry_price=signal.price,
                invested=bet_amount,
                strategy_tag=signal.strategy_tag  # Pass DNA tag
            )

            if success:
                tag_emoji = "🛰️" if signal.strategy_tag == "MOONSHOT" else "📈"
                print(f"  {tag_emoji} BUY  [{readable_time}] {signal.bucket:<15} {shares:>6.1f} @ ${signal.price:.3f} = ${bet_amount:>6.2f} | {signal.reason}")
            else:
                print(f"  ❌ FAILED BUY [{readable_time}] {signal.bucket} (portfolio.add_position returned False)")

        elif signal.type.value in ['SELL', 'ROTATE']:
            trade_result = self.portfolio.close_position(
                market=signal.market_title,
                bucket=signal.bucket,
                exit_price=signal.price
            )

            if trade_result:
                pnl = trade_result.get('pnl', 0)
                invested = trade_result.get('invested', 1)
                pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
                print(f"  📉 SELL [{readable_time}] {signal.bucket:<15} @ ${signal.price:.3f} → ${pnl:>+7.2f} ({pnl_pct:>+6.1f}%) | {signal.reason}")

    def _get_all_current_prices(self, tape: Dict) -> Dict:
        """Get current bid prices for all buckets from tape"""
        prices = {}
        for ob in tape.get('order_book', []):
            market_title = ob['title']
            for bucket in ob.get('buckets', []):
                key = f"{market_title}_{bucket['bucket']}"
                prices[key] = bucket['bid']
        return prices
