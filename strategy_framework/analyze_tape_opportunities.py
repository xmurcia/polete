#!/usr/bin/env python3
"""
Analyze tape data to find realistic edge opportunities
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategy_framework.strategies.elon_hawkes_strategy import ElonHawkesStrategy
from strategy_framework.backtesting.tape_backtest import TapeBacktest
from scipy.stats import norm

config = {
    "max_z_score_entry": 1.6,
    "min_price_entry": 0.02,
    "min_edge": 0.01,  # Very low to see all opportunities
    "risk_pct_normal": 0.04,
    "min_bet": 5.0,
    "enable_clustering": False,
    "cluster_range": 40
}

strategy = ElonHawkesStrategy(config)
backtester = TapeBacktest(
    strategy=strategy,
    initial_capital=1000.0,
    tape_dir="/Users/xavi.murcia/Desktop/poly-gemini/polete-volume-2/market_tape"
)

# Load tapes
tapes = backtester.load_all_tapes()

print("\n🔍 Analyzing edge opportunities in tape data...")
print("="*80)

# Sample 10 random tapes
import random
sample_tapes = random.sample(tapes, min(10, len(tapes)))

all_edges = []

for tape in sample_tapes:
    timestamp = tape['timestamp']
    from datetime import datetime
    readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')

    print(f"\n📊 Snapshot: {readable_time}")
    print("-"*80)

    # Check each market
    for meta in tape.get('meta', []):
        if not meta.get('active', True):
            continue

        market_title = meta['title']
        market_state = backtester.extract_market_state(tape, market_title)

        if not market_state or not market_state.buckets:
            continue

        # Get prediction
        pred_mean, pred_std = strategy._calculate_prediction(market_state)

        print(f"\n  Market: {market_title[:60]}")
        print(f"  Count: {market_state.count} | Hours left: {market_state.hours_left:.1f}")
        print(f"  Prediction: {pred_mean:.1f} ± {pred_std:.1f}")

        # Analyze buckets
        opportunities = []
        for bucket in market_state.buckets[:20]:  # First 20 buckets
            if bucket['max'] < market_state.count:
                continue

            # Calculate mid
            if bucket['max'] >= 99999:
                mid = bucket['min'] + 20
            else:
                mid = (bucket['min'] + bucket['max']) / 2

            # Z-score
            z_score = abs(mid - pred_mean) / pred_std if pred_std > 0 else 999

            # Fair value
            p_min = norm.cdf(bucket['min'], pred_mean, pred_std)
            if bucket['max'] >= 99999:
                fair = 1.0 - p_min
            else:
                fair = norm.cdf(bucket['max'] + 1, pred_mean, pred_std) - p_min

            edge = fair - bucket['ask']

            # Check conditions
            z_ok = z_score <= 1.6
            price_ok = bucket['ask'] >= 0.02

            if z_ok and price_ok and edge > 0:
                opportunities.append({
                    'bucket': bucket['bucket'],
                    'ask': bucket['ask'],
                    'z_score': z_score,
                    'fair': fair,
                    'edge': edge
                })
                all_edges.append(edge)

        if opportunities:
            # Sort by edge
            opportunities.sort(key=lambda x: x['edge'], reverse=True)

            print(f"\n  Top opportunities:")
            print(f"  {'Bucket':<15} {'Ask':>6} {'Z-Score':>8} {'Fair':>8} {'Edge':>8} {'Edge%':>7}")
            for opp in opportunities[:5]:
                edge_pct = (opp['edge'] / opp['ask']) * 100 if opp['ask'] > 0 else 0
                print(f"  {opp['bucket']:<15} {opp['ask']:>6.3f} {opp['z_score']:>8.2f} {opp['fair']:>8.3f} {opp['edge']:>8.3f} {edge_pct:>6.1f}%")
        else:
            print(f"  ❌ No opportunities found")

# Summary statistics
if all_edges:
    print(f"\n" + "="*80)
    print(f"📈 EDGE STATISTICS (across all analyzed snapshots)")
    print("="*80)
    import numpy as np
    all_edges = np.array(all_edges)

    print(f"Total opportunities found: {len(all_edges)}")
    print(f"Average edge: {np.mean(all_edges):.4f} ({np.mean(all_edges)*100:.2f}%)")
    print(f"Median edge: {np.median(all_edges):.4f} ({np.median(all_edges)*100:.2f}%)")
    print(f"Max edge: {np.max(all_edges):.4f} ({np.max(all_edges)*100:.2f}%)")
    print(f"Min edge: {np.min(all_edges):.4f} ({np.min(all_edges)*100:.2f}%)")

    print(f"\nEdge percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(all_edges, p)
        print(f"  {p}th percentile: {val:.4f} ({val*100:.2f}%)")

    print(f"\nRecommended min_edge values:")
    print(f"  Aggressive (top 50%): {np.percentile(all_edges, 50):.4f}")
    print(f"  Moderate (top 25%): {np.percentile(all_edges, 75):.4f}")
    print(f"  Conservative (top 10%): {np.percentile(all_edges, 90):.4f}")
else:
    print("\n❌ No opportunities found in sampled data")
