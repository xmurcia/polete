"""
Production Configuration
Shared between production bot and backtest to ensure 100% consistency
"""

# Entry/Exit Thresholds
MAX_Z_SCORE_ENTRY = 0.85  # V12.29 - Maximum Z-score for entry
MIN_PRICE_ENTRY = 0.02
MIN_EDGE = 0.05  # 5% minimum edge (line 866)

# Position Sizing
RISK_PCT_NORMAL = 0.04  # 4% of portfolio per trade
RISK_PCT_MOONSHOT = 0.01  # 1% for moonshots (high risk)
MAX_MOONSHOT_BET = 10.0  # Hard limit: $10 per moonshot
MIN_BET = 5.0

# Clustering
ENABLE_CLUSTERING = True
CLUSTER_RANGE = 40

# Market Consensus Weight (NOT USED IN CURRENT PRODUCTION - line 605 uses 0.2)
MARKET_WEIGHT = 0.70  # Defined but not used in final mean calculation
MARKET_CONSENSUS_WEIGHT = 0.2  # Actual weight used in line 605

# Trading Fees
TRADING_FEE_PCT = 0.0  # Polymarket has no fees currently

# Slippage Simulation
SLIPPAGE_BPS = 10  # 0.1% average slippage (bid-ask spread)

# Safety Margins (lines 835-839)
def get_safety_margin(hours_left: float) -> int:
    """Get safety margin based on hours left (anti-kamikaze filter)"""
    if hours_left > 24.0:
        return 20
    elif hours_left > 12.0:
        return 15
    elif hours_left > 6.0:
        return 10
    elif hours_left > 1.0:
        return 5
    else:
        return 2

# Warmup Check (line 681)
def is_warmup(current_count: int, hours_left: float) -> bool:
    """CAMBIO #17: Warmup reducido para capturar oportunidades tempranas"""
    min_req = 20 if hours_left > 72.0 else 8
    return current_count < min_req

# Reality Check (lines 847-851)
def is_impossible_to_reach(current_count: int, bucket_min: int, hours_left: float) -> bool:
    """Check if bucket is mathematically impossible to reach"""
    if hours_left < 1.0:
        tweets_needed = bucket_min - current_count
        if tweets_needed > (hours_left * 15):
            return True
    return False

# ==============================================================================
# EXIT CONDITIONS - STOP LOSS
# ==============================================================================

# Stop loss for normal trades (entry price >= $0.12)
STOP_LOSS_NORMAL = -0.40  # -40% loss

# Stop loss for cheap entries (entry price < $0.12)
# "Diamond hands" mode - allows deep drawdowns on lottery tickets
STOP_LOSS_CHEAP_ENTRY = -2.0  # -200% (basically never sell on loss)

# Price threshold to classify as "cheap entry"
STOP_LOSS_CHEAP_THRESHOLD = 0.12  # $0.12

# Stop loss disabled in late game (< 48h remaining)
STOP_LOSS_LATE_GAME = -2.0  # No stop loss

# Minimum Z-score to trigger stop loss (prevents panic selling)
STOP_LOSS_Z_MIN = 1.3

# Victory lap time threshold
VICTORY_LAP_TIME_HOURS = 48.0
