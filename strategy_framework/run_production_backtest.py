#!/usr/bin/env python3
"""
Production Backtest Runner

Uses EXACT production bot logic to validate performance on historical data.
Results should match what the bot would have done in production.
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategy_framework.strategies.production_strategy import ProductionStrategy
from strategy_framework.backtesting.tape_backtest import TapeBacktest
from strategy_framework.production_config import (
    MAX_Z_SCORE_ENTRY, MIN_PRICE_ENTRY, MIN_EDGE,
    ENABLE_CLUSTERING, CLUSTER_RANGE, RISK_PCT_NORMAL, MIN_BET
)

print("🤖 PRODUCTION BOT BACKTEST")
print("="*70)
print("Using EXACT production logic from elon_auto_bot_threads.py")
print("="*70)
print("\n📋 Configuration (matching production):")
print(f"  max_z_score_entry: {MAX_Z_SCORE_ENTRY}")
print(f"  min_price_entry: ${MIN_PRICE_ENTRY}")
print(f"  min_edge: {MIN_EDGE} ({MIN_EDGE*100:.0f}%)")
print(f"  risk_pct: {RISK_PCT_NORMAL} ({RISK_PCT_NORMAL*100:.0f}%)")
print(f"  min_bet: ${MIN_BET}")
print(f"  enable_clustering: {ENABLE_CLUSTERING}")
print(f"  cluster_range: {CLUSTER_RANGE}")
print("="*70)

# Initialize production strategy (no config override = pure production)
strategy = ProductionStrategy()

# Initialize backtester
backtester = TapeBacktest(
    strategy=strategy,
    initial_capital=1000.0,
    tape_dir="../polete-volume-2/market_tape"
)

# Run backtest
print("\n🚀 Running backtest with production logic...")
metrics = backtester.run(
    market_filter=None,  # All markets
    start_date=None,  # All available data
    end_date=None
)

# Save results
output_dir = "logs/backtest_results"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(output_dir, f"production_backtest_{timestamp}.json")

# Add configuration to results
results_dict = metrics.to_dict()
results_dict['configuration'] = {
    'max_z_score_entry': MAX_Z_SCORE_ENTRY,
    'min_price_entry': MIN_PRICE_ENTRY,
    'min_edge': MIN_EDGE,
    'risk_pct_normal': RISK_PCT_NORMAL,
    'min_bet': MIN_BET,
    'enable_clustering': ENABLE_CLUSTERING,
    'cluster_range': CLUSTER_RANGE
}

with open(results_file, 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\n💾 Results saved to: {results_file}")

# Save equity curve
import pandas as pd
if backtester.equity_curve:
    equity_df = pd.DataFrame(backtester.equity_curve)
    equity_file = os.path.join(output_dir, f"production_equity_{timestamp}.csv")
    equity_df.to_csv(equity_file, index=False)
    print(f"📈 Equity curve saved to: {equity_file}")

# Save trade log
if backtester.portfolio.data['history']:
    trades_df = pd.DataFrame(backtester.portfolio.data['history'])
    trades_file = os.path.join(output_dir, f"production_trades_{timestamp}.csv")
    trades_df.to_csv(trades_file, index=False)
    print(f"📊 Trade history saved to: {trades_file}")
    print(f"   Total trades: {len(trades_df)}")

# Hawkes activation report
if hasattr(strategy, 'hawkes_activations'):
    print(f"\n🔥 HAWKES BRAIN REPORT:")
    print(f"   Activations: {strategy.hawkes_activations} times")
    print(f"   Hawkes > Gaussian: {strategy.hawkes_wins} times")
    if strategy.hawkes_activations > 0:
        print(f"   Activation rate: {(strategy.hawkes_activations/420*100):.1f}% of snapshots")

print("\n" + "="*70)
print("✅ Production backtest complete!")
print("="*70)
print("\nThese results represent what the production bot would have done")
print("on this historical data using the EXACT same logic and parameters.")
print("\nTo test configuration changes:")
print("  1. Edit production_config.py")
print("  2. Run this script again")
print("  3. Compare results before deploying to production")
