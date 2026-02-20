# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated trading bot for Polymarket that predicts Elon Musk's tweet volume and trades on prediction markets. The system uses a **Hawkes Process** model combined with market consensus to generate predictions and execute paper trades.

## Core Architecture

### Main Entry Point

- **`main.py`**: Production version (v12.16 + Moonshot V33) - Run this for live operations
  - Formerly `elon_auto_bot_threads.py` (renamed for clarity)
- **`utils/elon_auto_bot_threads_v10.py`**: Previous stable version (archived)

### Key Components (Modular Architecture)

All components are now organized under `src/` directory:

1. **ClobMarketScanner** (`src/clob_scanner.py`)
   - Fetches live order book prices from Polymarket CLOB API
   - Filters for Elon tweet markets ("elon" + "tweets" in title)
   - Uses bulk pricing endpoint for efficiency

2. **PolymarketSensor** (`src/polymarket_sensor.py`)
   - Monitors tweet counts via xtracker.polymarket.com API
   - Detects new tweets by comparing counts between iterations
   - Uses ThreadPoolExecutor for parallel market fetching
   - Fixes end dates to 17:00 UTC for consistency

3. **PaperTrader** (`src/paper_trader.py`)
   - Manages virtual portfolio (default: $1000 starting cash)
   - Position sizing: 4% risk for normal, 1% for lottery tickets
   - Logs all trades to `logs/trade_history.csv`
   - Saves snapshots to `logs/snapshots/` on each trade

4. **MarketPanicSensor** (`src/market_panic_sensor.py`)
   - Detects abnormal price movements (PUMP/DUMP)
   - Uses rolling 5-window deque for historical comparison
   - Triggers alerts when price exceeds sensitivity threshold (default: 1.5x)

5. **Moonshot Module V33** (`src/moonshot.py`)
   - Satellite strategy for extreme upside bets on distant buckets
   - Entry: $0.005-$0.009 price (100x+ payoff), ≥72h events, 60%+ time remaining
   - Position sizing: Max $10 per moonshot (1% portfolio), limit 2 simultaneous
   - DNA tagging: `strategy_tag='MOONSHOT'` tracks positions through lifecycle
   - Exit rules:
     - ≤$0.01 bid: Forced expiration liquidation
     - ≥$0.99 bid: Victory lap (lock massive gains)
     - ≥$0.35 peak → -$0.15 drawdown: Trailing stop (protects 50x+ gains)
     - <$0.35: Full immunity from all other exit rules

6. **Auto-Hedge Module** (`src/auto_hedge.py`)
   - Automatic risk management for end-game scenarios
   - Floor hedge: Protection against undershooting
   - Ceiling hedge: Protection against overshooting

7. **Real Trader** (`src/real_trader/`)
   - UnifiedTrader: Wrapper for paper/real trading modes
   - Authentication, order management, position tracking
   - Balance management and blockchain integration

8. **Notifications** (`src/notifications/`)
   - Telegram notifications for trades and portfolio updates
   - Position summaries and daily digests

### Trading Logic (lines 461-850)

**Prediction Model:**

- Hybrid: `(1 - MARKET_WEIGHT) * Hawkes + MARKET_WEIGHT * Market_Consensus`
- Applies bio-rhythm multipliers (hourly & daily patterns)
- Uses normal distribution for probability calculations

**Entry Conditions (Standard Strategy):**

- Z-score ≤ 0.85 (MAX_Z_SCORE_ENTRY)
- Ask price ≥ $0.02 (MIN_PRICE_ENTRY)
- Fair value > Ask + 0.05 edge (Accumulation strategy)
- Clustering: permissive (within 40 units)

**Entry Conditions (Moonshot Strategy):**

- Event duration ≥72h (only long events)
- Time remaining ≥60% of total duration
- Price range: $0.005-$0.009 (ultra-distant buckets, 100x+ payoff)
- Direction: upward (bucket.min > current_count)
- Realism filter: max 3x daily average distance
- Inventory limit: 2 moonshots maximum

**Exit Conditions (Standard):**

- Proximity Danger: < safety threshold tweets to bucket max
- Victory Lap: bid > $0.95 in final 48h
- Paranoid Treasure: +150% profit AND Z > 0.9
- Protect Profit: +5% profit AND Z > 2.4 (mid-game)
- Stop Loss: Adaptive based on entry price (-40% to -75%)
- Extreme Panic: Z > 8.0

**Exit Conditions (Moonshot):**

- Expired: bid ≤ $0.01 → realize loss
- Victory Lap: bid ≥ $0.99 → lock 1000%+ gains
- Trailing Stop: peak ≥ $0.35 AND drawdown ≥ $0.15 (captures 35x-50x gains)
- Immunity: bid < $0.35 → HOLD (ignore all other rules)

## Data Persistence

### Critical Files

- `logs/live_history.json`: Real-time tweet events (24h rolling window)
- `logs/portfolio.json`: Current positions and cash balance
- `logs/trade_history.csv`: Complete trade log with P&L
- `logs/market_tape/*.json`: Order book snapshots every 30 min
- `daily_metrics_three_weeks/*.csv`: Historical tweet data for training

### State Management

- Bot maintains global events list in memory, synced to live_history.json
- Hawkes model retrains on startup by loading all CSVs + live events
- Portfolio persists across restarts via JSON serialization

## Running the Bot

```bash
# Paper trading mode (default)
python main.py --initial-cash 1000

# Real trading mode (requires blockchain setup)
python main.py --real-trading
```

Command-line arguments:
- `--initial-cash`: Initial cash for paper mode (default: 1000.0)
- `--real-trading`: Enable real trading mode (uses blockchain)

Loop runs every 8 seconds. Press Ctrl+C to stop.

## Cleaning State

Use `clean_bot.py` to reset the system:

```bash
python clean_bot.py
```

This wipes all logs, resets portfolio to $1000, and clears Hawkes memory.

## Configuration Constants

Key tunable parameters (lines 34-52):

- `MAX_Z_SCORE_ENTRY = 0.85`: Maximum statistical distance for entry
- `MIN_PRICE_ENTRY = 0.02`: Minimum price filter
- `ENABLE_CLUSTERING = True`: Require positions to cluster
- `CLUSTER_RANGE = 40`: Maximum distance between positions
- `MARKET_WEIGHT = 0.70`: Weight of market consensus in hybrid model

Position sizing:

- Standard trades: 4% portfolio (`risk_pct_normal`)
- Moonshot trades: 1% portfolio or $10 max (`risk_pct_moonshot`, `max_moonshot_bet`)

## API Endpoints

- **Gamma API**: `https://gamma-api.polymarket.com/events` (market discovery)
- **CLOB API**: `https://clob.polymarket.com/prices` (bulk pricing)
- **Xtracker API**: `https://xtracker.polymarket.com/api` (tweet counts)

## Dependencies

See `requirements.txt`. Core libraries:

- `requests`: API calls
- `numpy`, `pandas`: Data processing
- `scipy`: Hawkes optimization & probability calculations
- `websockets`: (unused in current version)

## Code Conventions

- No docstrings, inline comments minimal
- JSON files indented with 2 spaces
- CSV logs append-only with UTF-8 encoding
- Datetime strings: "%Y-%m-%d %H:%M:%S" or ISO 8601
- Prices stored as floats with 3 decimal precision
