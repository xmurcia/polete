"""
Configuration file for Elon Musk Tweet Trading Bot
All tunable parameters centralized with clear explanations

This file contains ALL magic numbers and thresholds used by the bot.
Every value has a comment explaining what it controls.

Last updated: 2026-02-20
Bot Version: V12.16 + Moonshot V33 + Auto-Hedge V2
"""

import os
from datetime import datetime

# ==============================================================================
# FILE PATHS & DIRECTORIES
# ==============================================================================

# Logs directory (all bot activity logs stored here)
LOGS_DIR = 'logs'

# Portfolio state file (JSON with positions and cash balance)
PORTFOLIO_PAPER_TRADER = "portfolio.json"

# Live events log (24h rolling window of detected tweets)
LIVE_LOG = "live_history.json"

# Trade history CSV (permanent record of all trades)
TRADE_LOG = "trade_history.csv"

# Snapshot directory (saves full context on each trade)
SNAPSHOTS_DIR = os.path.join(LOGS_DIR, "snapshots")

# Market tape directory (order book snapshots every 30 min)
MARKET_TAPE_DIR = os.path.join(LOGS_DIR, "market_tape")

# ==============================================================================
# API CONFIGURATION
# ==============================================================================

API_CONFIG = {
    'base_url': "https://xtracker.polymarket.com/api",      # Tweet tracking API
    'gamma_url': "https://gamma-api.polymarket.com/events",  # Market discovery
    'clob_url': "https://clob.polymarket.com/prices",        # Order book pricing
    'user': "elonmusk"  # Twitter username to track
}

# API timeout in seconds (how long to wait for responses)
API_TIMEOUT_SECONDS = 5

# Number of markets to fetch per request
API_MARKET_LIMIT = 100

# User agent for HTTP requests (pretend to be Chrome browser)
USER_AGENT_CHROME = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
USER_AGENT_SIMPLE = "Mozilla/5.0"

# ==============================================================================
# PORTFOLIO & POSITION SIZING
# ==============================================================================

# Starting cash for paper trading mode
PORTFOLIO_INITIAL_CASH = 1000.0

# Minimum bet size (prevents tiny positions)
PORTFOLIO_MIN_BET = 5.0

# Risk percentages per strategy type
RISK_PCT_NORMAL = 0.04      # 4% of portfolio per standard trade
RISK_PCT_LOTTO = 0.01       # 1% for lottery ticket (far bucket)
RISK_PCT_MOONSHOT = 0.01    # 1% for moonshot trades (extreme upside bets)
RISK_PCT_HEDGE = 0.025      # 2.5% for hedge positions (insurance)

# Maximum dollar amount per moonshot trade
MAX_MOONSHOT_BET = 10.0

# Maximum percentage of portfolio per single trade (safety cap)
MAX_POSITION_SIZE_PCT = 0.10  # 10% hard limit

# ==============================================================================
# KELLY CRITERION (Position Sizing When Edge Detected)
# ==============================================================================

# When bot detects high "edge" (Val+ in reason), multiply position size
KELLY_MULTIPLIER_HIGH_EDGE = 2.0    # 2x size when edge >= 40%
KELLY_MULTIPLIER_MED_EDGE = 1.5     # 1.5x size when edge >= 20%

# Edge thresholds to trigger Kelly multipliers
KELLY_EDGE_THRESHOLD_HIGH = 0.40    # 40% edge = high confidence
KELLY_EDGE_THRESHOLD_MED = 0.20     # 20% edge = medium confidence

# ==============================================================================
# ENTRY CONDITIONS (Standard Strategy)
# ==============================================================================

# Maximum Z-score to enter a trade (statistical distance from prediction)
# Lower = closer to prediction, safer
MAX_Z_SCORE_ENTRY = 0.75

# Minimum ask price to consider (filters out nearly-expired buckets)
MIN_PRICE_ENTRY = 0.02  # $0.02 minimum

# Minimum edge required to enter (fair value - ask price)
MIN_EDGE_BASE = 0.05  # 5% base edge required

# Dynamic edge adjustment based on volatility
EDGE_STD_MULTIPLIER = 0.01  # Add 1% per unit of std deviation (reduced from 0.01 to allow earlier entries in long events)

# ==============================================================================
# CLUSTERING (Position Concentration Strategy)
# ==============================================================================

# Whether to require positions to be clustered together
ENABLE_CLUSTERING = True

# Maximum distance between positions in same market (tweet count units)
# For long events (≥72h): Use fixed value
CLUSTER_RANGE = 40  # Positions must be within 40 tweets of each other

# For short events (<72h): Use dynamic clustering based on bucket size and time remaining
# Bucket sizes: ~24 tweets for short events, ~19 for long events
# Multipliers by time remaining (short events only):
CLUSTER_MULTIPLIER_SHORT_EARLY = 1.5   # >24h remaining: Allow 1.5 buckets apart (36 tweets)
CLUSTER_MULTIPLIER_SHORT_LATE = 1.0    # ≤24h remaining: Allow 1.0 buckets apart (24 tweets, neighbors only)

# ==============================================================================
# WARMUP CONDITIONS (Avoid Trading with Insufficient Data)
# ==============================================================================

# Minimum tweet count before allowing trades on long events (>72h)
WARMUP_MIN_TWEETS_LONG = 20

# Minimum tweet count before allowing trades on short events (<72h)
WARMUP_MIN_TWEETS_SHORT = 35

# Minimum historical events needed for valid Hawkes predictions
WARMUP_MIN_HISTORY_COUNT = 5

# ==============================================================================
# EXIT CONDITIONS - VICTORY LAP
# ==============================================================================

# Sell when price reaches this level in final hours (lock in profit)
VICTORY_LAP_PRICE = 0.95  # $0.95 or higher

# Only trigger victory lap in final hours of event
VICTORY_LAP_TIME_HOURS = 48.0  # Last 48 hours

# ==============================================================================
# EXIT CONDITIONS - PROFIT PROTECTION
# ==============================================================================

# Minimum profit to trigger profit protection rules
PROFIT_PROTECT_MIN_PCT = 0.05  # 5% profit minimum

# Z-score thresholds for profit protection (vary by time remaining)
PROFIT_PROTECT_Z_LONG = 2.4    # For events > 48h remaining
PROFIT_PROTECT_Z_MID = 1.8     # For events 24-48h remaining

# "Paranoid Treasure" - exit at massive profit even if Z-score low
PARANOID_TREASURE_MIN_PCT = 1.5  # 150% profit
PARANOID_TREASURE_Z = 0.9        # Z-score threshold

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

# Mid-game emergency stop (24-48h): Cubre el gap donde STOP_LOSS_LATE_GAME anula el sl_limit normal
# Solo se activa con pérdidas extremas Y Z-score alto (estadísticamente imposible recuperar)
STOP_LOSS_MID_GAME_EMERGENCY = -0.60  # -60% pérdida en franja 24-48h
STOP_LOSS_MID_GAME_Z_MIN = 4.0        # Z > 4.0 = muy lejos de la predicción

# Minimum Z-score to trigger stop loss (prevents panic selling)
STOP_LOSS_Z_MIN = 1.3

# ==============================================================================
# EXIT CONDITIONS - CATASTROPHIC LOSS (Final Phase Only)
# ==============================================================================

# Catastrophic stop loss for final phase (< 24h remaining)
# Only triggers when loss is extreme AND outcome is mathematically impossible
STOP_LOSS_CATASTROPHIC = -0.75  # -75% loss (worse than normal -40%)
STOP_LOSS_CATASTROPHIC_Z_MIN = 5.0  # Z > 5.0 = 5+ std deviations (99.9999% impossible)

# ==============================================================================
# EXIT CONDITIONS - EXTREME PANIC
# ==============================================================================

# Z-score threshold for "extreme panic" exit
EXTREME_PANIC_Z = 8.0  # Statistical outlier, model says impossible

# Maximum profit to still exit on extreme panic
EXTREME_PANIC_MAX_PROFIT = 0.10  # Only if profit < 10%

# ==============================================================================
# EXIT CONDITIONS - PROXIMITY DANGER
# ==============================================================================

# How many tweets of "headroom" needed before bucket max
# Varies by time remaining (less time = less buffer needed)

PROXIMITY_BASE_THRESHOLD_LONG = 15   # > 24h remaining
PROXIMITY_BASE_THRESHOLD_MID = 12    # 12-24h remaining
PROXIMITY_BASE_THRESHOLD_SHORT = 10  # 6-12h remaining
PROXIMITY_BASE_THRESHOLD_FINAL = 8   # < 6h remaining

# Volatility buffer added to base threshold
PROXIMITY_VOLATILITY_MULTIPLIER = 1.5  # Add 1.5x std deviation

# ==============================================================================
# TIME THRESHOLDS
# ==============================================================================

# Event duration classifications (determines strategy mode)
EVENT_DURATION_HOURS_LONG = 168.0     # 7 days = MARATHON mode
EVENT_DURATION_HOURS_MID = 72.0       # 3 days
EVENT_DURATION_HOURS_SHORT = 48.0     # 2 days
EVENT_DURATION_HOURS_VERY_SHORT = 40.0

# Time remaining classifications (affects urgency)
TIME_REMAINING_HOURS_SPRINT = 6.0     # < 6h = SPRINT mode (urgent)
TIME_REMAINING_HOURS_RUN = 24.0       # < 24h = RUN mode (active)
TIME_REMAINING_HOURS_MARATHON = 72.0  # > 72h = MARATHON mode (patient)

# Minimum time remaining to consider trading
TIME_REMAINING_HOURS_FINAL = 0.5      # 30 minutes minimum

# Time boundary for "impossible trade" filter
TIME_REMAINING_HOURS_VERY_LATE = 1.0  # < 1h = check tweet velocity

# ==============================================================================
# COOLDOWN PERIODS (Prevent Re-entering Bad Trades)
# ==============================================================================

# Hours to wait before re-entering bucket that hit stop loss
COOLDOWN_STOP_LOSS_HOURS = 24

# Hours to wait before re-entering expired moonshot
COOLDOWN_MOONSHOT_EXPIRED_HOURS = 24

# Hours to wait before re-entering sold moonshot
COOLDOWN_MOONSHOT_EXIT_HOURS = 48

# ==============================================================================
# MARKET EXPIRATION & TOLERANCE
# ==============================================================================

# Market end date fixed time (all markets end at 17:00 UTC)
END_DATE_FIXED_HOUR = 17
END_DATE_FIXED_MINUTE = 0
END_DATE_FIXED_SECOND = 0

# Grace period after official end time (allow late tweets)
MARKET_EXPIRATION_TOLERANCE_HOURS = 12.0

# ==============================================================================
# MOONSHOT STRATEGY (Extreme Upside Bets on Distant Buckets)
# ==============================================================================

# Entry price range for moonshots (ultra-cheap = huge upside)
MOONSHOT_MIN_PRICE = 0.005  # $0.005 minimum (200x potential)
MOONSHOT_MAX_PRICE = 0.011  # $0.011 maximum (90x potential)

# Only trade moonshots on long events (need time for extreme scenarios)
MOONSHOT_MIN_EVENT_DURATION = 72.0  # 72 hours minimum

# Only enter if plenty of time remaining (need runway)
MOONSHOT_MIN_TIME_REMAINING_PCT = 0.60  # 60% of event duration left

# Minimum tweet count to start moonshot hunting
MOONSHOT_MIN_COUNT_THRESHOLD = 35

# Realism filter - don't buy buckets that are TOO far away
MOONSHOT_MAX_DISTANCE_MULTIPLIER = 2.0  # Max distance = daily_avg * 2

# Rage target calculation (how far up could Elon go in rage mode?)
MOONSHOT_RAGE_TARGET_BUFFER = 120.0  # Base projection + 120 tweets

# Default mid-point offset for buckets with max = 99999
MOONSHOT_DEFAULT_MID_OFFSET = 20

# Maximum concurrent moonshot positions
MAX_MOONSHOTS_CONCURRENT = 1

# Clustering for moonshots (similar to standard strategy)
MOONSHOT_CLUSTER_DISTANCE = 40

# Exit conditions for moonshots
MOONSHOT_EXIT_EXPIRED_PRICE = 0.01  # ≤ $0.01 = expired, realize loss
MOONSHOT_EXIT_VICTORY_PRICE = 0.99  # ≥ $0.99 = victory lap, lock 100x gain

# Partial exit for moonshots (lock some profit early)
MOONSHOT_EXIT_PARTIAL_MIN_PRICE = 0.20   # Partial exit range $0.20-$0.30
MOONSHOT_EXIT_PARTIAL_MAX_PRICE = 0.30
MOONSHOT_EXIT_PARTIAL_MIN_PROFIT = 3.0   # 300% profit required

# Trailing stop for moonshots (protect massive gains)
MOONSHOT_TRAILING_PEAK_THRESHOLD = 0.50  # Start tracking when bid ≥ $0.50
MOONSHOT_TRAILING_DRAWDOWN = 0.15        # Exit if drops $0.15 from peak

# ==============================================================================
# AUTO-HEDGE STRATEGY (Insurance for End-Game)
# ==============================================================================

# Only activate hedge in final hours
HEDGE_MAX_TIME_HOURS = 12.0  # Last 12 hours (for long events ≥72h)
HEDGE_MIN_TIME_HOURS = 0.5   # But not in final 30 minutes

# For short events (<72h): More conservative activation
HEDGE_MAX_TIME_HOURS_SHORT = 6.0  # Last 6 hours only (not 12h - too early for short events)

# Price filters for hedge positions (must be cheap insurance)
HEDGE_MIN_PRICE = 0.005
HEDGE_MAX_PRICE = 0.12  # Don't pay more than $0.30 for insurance

# Projection rates for hedge scenarios (LONG EVENTS ≥72h ONLY)
HEDGE_FLOOR_RATE = 1.0    # Pessimistic: 1 tweet/hour (slow day)
HEDGE_CEILING_RATE = 3.5  # Optimistic: 3.5 tweets/hour (rage mode)

# For short events (<72h): Use adaptive rates based on current rhythm
# These are multipliers applied to current rate, with caps
HEDGE_CEILING_MULTIPLIER_SHORT = 1.3  # Max 30% faster than current rate
HEDGE_FLOOR_MULTIPLIER_SHORT = 0.7    # Max 30% slower than current rate
HEDGE_CEILING_CAP_SHORT = 2.5         # Absolute max: 2.5 tweets/hour
HEDGE_FLOOR_CAP_SHORT = 0.5           # Absolute min: 0.5 tweets/hour

# ==============================================================================
# STATISTICAL PARAMETERS (Sigma/Volatility Calculations)
# ==============================================================================

# Base sigma calculation: sigma = sqrt(mean) * multiplier
SIGMA_BASE_MULTIPLIER = 1.5

# Minimum effective standard deviation
SIGMA_MIN_VALUE = 3.0

# Time decay factors for sigma (longer events = more uncertainty)
SIGMA_TIME_FACTOR_MIN = 0.3
SIGMA_TIME_FACTOR_MAX = 1.0
SIGMA_TIME_DECAY_BASE = 168.0  # 7 days for full sigma

# Decay factor for sigma in proximity calculations
SIGMA_DECAY_FACTOR_MIN = 0.25
SIGMA_DECAY_FACTOR_MAX = 1.0
SIGMA_DECAY_BASE_HOURS = 72.0

# ==============================================================================
# MARKET CONSENSUS (Hybrid Model with Market Prices)
# ==============================================================================

# Weight given to market consensus vs Hawkes model prediction
MARKET_WEIGHT = 0.70  # 70% market, 30% Hawkes (market is wise)

# Only consider buckets with meaningful bids
CONSENSUS_MIN_BID = 0.05

# Deviation threshold to trigger consensus adjustment
CONSENSUS_DEVIATION_THRESHOLD = 0.15  # 15% difference

# Weights when model diverges from market
CONSENSUS_WEIGHT_WHEN_DIVERGENT = 0.4   # 40% market
CONSENSUS_MODEL_WEIGHT_WHEN_DIVERGENT = 0.6  # 60% Hawkes

# Only apply consensus if enough time remaining
CONSENSUS_TIME_MIN_HOURS = 12.0

# ==============================================================================
# LARGE BUCKET HANDLING
# ==============================================================================

# How to detect "large" buckets (e.g., "100+" instead of "100-109")
LARGE_BUCKET_SIZE_MULTIPLIER = 5.0  # Bucket > avg_step * 5 = large

# How to calculate mid-point for large buckets
LARGE_BUCKET_MID_OFFSET = None  # Use avg_step if available

# Bucket size upper limit (for "99999" open-ended buckets)
BUCKET_MAX_OPEN_ENDED = 99999

# Default mid-point offset for open-ended buckets
BUCKET_OPEN_ENDED_MID_OFFSET = 20

# ==============================================================================
# FATIGUE WEIGHTS (How Much to Trust Current Rate vs Historical)
# ==============================================================================

# As event progresses, weight current rate more heavily
FATIGUE_WEIGHT_SPRINT = 0.95   # < 6h: 95% current, 5% historical
FATIGUE_WEIGHT_RUN = 0.75      # < 24h: 75% current, 25% historical
FATIGUE_WEIGHT_MARATHON = 0.35 # > 24h: 35% current, 65% historical

# ==============================================================================
# RATE LIMITING (Prevent Absurd Predictions in Marathon Mode)
# ==============================================================================

# Maximum rate multiplier in marathon mode
MAX_RATE_MULTIPLIER_MARATHON = 3.0  # Max rate = historical_avg * 3

# ==============================================================================
# IMPOSSIBLE TRADE FILTER
# ==============================================================================

# Maximum tweets per hour assumption (prevents trading impossible buckets)
IMPOSSIBLE_TWEETS_PER_HOUR = 15

# ==============================================================================
# BID/ASK FILTERING
# ==============================================================================

# Minimum bid to process a bucket (skip near-zero bids)
MIN_BID_FOR_PROCESSING = 0.001

# ==============================================================================
# HAWKES PROCESS MODEL PARAMETERS
# ==============================================================================

# Initial parameter values (before optimization)
HAWKES_INITIAL_MU = 0.4     # Baseline tweet rate
HAWKES_INITIAL_ALPHA = 3.0  # Self-excitation strength
HAWKES_INITIAL_BETA = 4.0   # Decay rate

# Optimization constraints
HAWKES_MIN_TIMESTAMPS = 50       # Need 50+ events to optimize
HAWKES_OPT_MIN_PARAM = 1e-4      # Minimum parameter value
HAWKES_OPT_METHOD = 'L-BFGS-B'   # Optimization algorithm
HAWKES_NLL_PENALTY_INVALID = 1e10  # Penalty for invalid params

# Simulation parameters
HAWKES_NUM_SIMULATIONS = 1000  # Monte Carlo runs per prediction

# Cutoff for historical event influence
HAWKES_CUTOFF_MULTIPLIER = 10.0  # cutoff = 10 / beta

# Minimum lambda value in simulations
HAWKES_MIN_LAMBDA = 0.001

# Random seed for reproducible timestamp spreading
HAWKES_RANDOM_SEED = 42

# Spread daily events across hour (avoids all at midnight)
HAWKES_INTRADAY_SPREAD_SECONDS = 3600

# ==============================================================================
# BIO-RHYTHM MULTIPLIERS (Currently Unused, Kept for Future)
# ==============================================================================

# Hourly activity multipliers (Elon tweets more at certain hours)
HOURLY_MULTIPLIERS = {
    0: 0.97,  1: 0.80,  2: 0.42,  3: 0.20,
    4: 0.39,  5: 0.48,  6: 2.11,  7: 1.41,
    8: 1.46,  9: 1.58, 10: 0.44, 11: 0.21,
    12: 0.35, 13: 0.49, 14: 1.72, 15: 1.71,
    16: 1.37, 17: 2.03, 18: 1.34, 19: 1.24,
    20: 1.01, 21: 0.89, 22: 0.82, 23: 0.61
}

# Daily activity multipliers (Elon tweets more on certain days)
DAILY_MULTIPLIERS = {
    0: 0.90,  # Monday
    1: 0.75,  # Tuesday
    2: 1.25,  # Wednesday
    3: 0.95,  # Thursday
    4: 0.95,  # Friday
    5: 1.15,  # Saturday
    6: 1.10   # Sunday
}

# ==============================================================================
# LOOP & TIMING
# ==============================================================================

# Main loop delay between iterations
LOOP_DELAY_SECONDS = 5

# Delay after error before retry
LOOP_ERROR_RETRY_SECONDS = 5

# Delay when no data available
LOOP_NO_DATA_WAIT_SECONDS = 3

# How often to save market tape snapshots
MARKET_TAPE_SAVE_INTERVAL_SECONDS = 1800  # 30 minutes

# ==============================================================================
# PANIC SENSOR CONFIGURATION
# ==============================================================================

# Price movement sensitivity (1.5 = 50% above average triggers PUMP alert)
PANIC_SENSITIVITY = 1.5

# Rolling window size for price history
PANIC_WINDOW_SIZE = 5

# Minimum price thresholds to trigger alerts
PANIC_MIN_ASKS_FOR_PUMP = 0.01  # Ignore pumps below $0.01
PANIC_MIN_BIDS_FOR_DUMP = 0.05  # Only alert dumps above $0.05

# Minimum history length before analyzing
PANIC_MIN_HISTORY_LENGTH = 3

# ==============================================================================
# MARKET FILTERING
# ==============================================================================

# Title keywords to filter markets (must contain both)
MARKET_FILTER_KEYWORD_1 = "elon"
MARKET_FILTER_KEYWORD_2 = "tweets"

# Years to exclude from date matching (avoid false positives)
# Dynamically generate around current year
_CURRENT_YEAR = datetime.now().year
DATE_FILTER_YEARS = [str(y) for y in range(_CURRENT_YEAR - 2, _CURRENT_YEAR + 5)]

# Minimum date matches required for title matching
DATE_MIN_MATCHES_REQUIRED = 2

# Market status filters for API
MARKET_STATUS_ACTIVE = "true"
MARKET_STATUS_CLOSED = "false"
MARKET_STATUS_ARCHIVED = "false"

# Sorting preferences
MARKET_SORT_BY = "volume24hr"
MARKET_SORT_ASCENDING = "false"

# Thread pool for parallel fetching
THREAD_POOL_MAX_WORKERS = 5

# ==============================================================================
# DISPLAY & FORMATTING
# ==============================================================================

# Activar/desactivar el formato enriquecido de mensajes de Telegram.
# True  → mensajes agrupados por evento con P&L en vivo, emojis y tabla monoespaciada.
# False → formato compacto original (tabla simple en una sola línea).
RICH_NOTIFICATIONS = True

# Decimal precision for different value types
PRICE_DECIMAL_PLACES = 3   # $0.123
PNL_DECIMAL_PLACES = 2     # $12.34
STAT_DECIMAL_PLACES = 1    # 45.6
SHARES_DECIMAL_PLACES = 1  # 123.4

# JSON formatting
JSON_INDENT = 2

# Console table widths
TABLE_WIDTH = 65
TABLE_SEPARATOR_WIDTH = 85

# ==============================================================================
# VERSION TAGS
# ==============================================================================

BOT_VERSION = "V12.16"
MOONSHOT_VERSION = "V33"
HEDGE_VERSION = "V2"

# Mode display tags
MODE_TAG_SPRINT = "🚀 SPRINT"
MODE_TAG_RUN = "🏃 RUN"
MODE_TAG_MARATHON = "⚓ MARATHON"
MODE_TAG_MARKET_CONSENSUS = "+MKT"
MODE_TAG_ERROR = "⚠️ ERROR"

# ==============================================================================
# TIME CONSTANTS (Conversions)
# ==============================================================================

SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
MS_PER_DAY = 86_400_000  # 24 * 60 * 60 * 1000
MS_PER_SECOND = 1000
