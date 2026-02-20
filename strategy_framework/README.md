# Strategy Framework

Modular trading strategy framework for Polymarket prediction markets.

## Directory Structure

```
strategy_framework/
├── strategies/           # Trading strategy implementations
│   ├── base_strategy.py     # Abstract base class
│   ├── elon_hawkes_strategy.py  # V12.16 logic extracted
│   └── __init__.py
├── backtesting/         # Historical simulation tools
│   ├── backtest_runner.py   # Main backtesting engine
│   ├── metrics.py           # Performance calculations
│   └── __init__.py
├── utils/               # Shared utilities
│   ├── market_scanner.py    # CLOB price fetcher
│   ├── portfolio.py         # Position management
│   └── __init__.py
├── logs/                # Backtest results and reports
└── main.py              # Entry point (backtest or live)
```

## Phase 1: Foundation

1. Extract V12.16 components into modular classes
2. Keep original `elon_auto_bot_threads.py` untouched
3. Create parallel backtesting capability

## Setup

Install dependencies from the root `requirements.txt`:

```bash
cd ..  # Go to project root
pip install -r requirements.txt
```

## Usage

```bash
# Run backtest
python strategy_framework/main.py --mode backtest --strategy elon_hawkes --days 30

# Run live (future)
python strategy_framework/main.py --mode live --strategy elon_hawkes
```
