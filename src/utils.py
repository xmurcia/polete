import os
import json
import re
import time
from datetime import datetime
from config import *

try:
    import database as db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

def save_market_tape(clob_data, markets_meta):
    """Save market tape snapshot to disk (and optionally to DB)"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_unix = time.time()

    # 1. Escribir a archivo (OBLIGATORIO - fuente de verdad)
    with open(os.path.join(MARKET_TAPE_DIR, f"tape_{ts}.json"), "w") as f:
        json.dump({"timestamp": ts_unix, "meta": markets_meta, "order_book": clob_data}, f)

    # 2. Escribir a DB en shadow mode (OPCIONAL - no bloquea si falla)
    if DB_AVAILABLE:
        db.shadow_write(
            db.save_tape,
            ts_unix=ts_unix,
            meta=markets_meta,
            order_book=clob_data
        )

def save_trade_snapshot(action, m_title, bucket, price, reason, ctx, hours_left=None, tweet_count=None, mode="PAPER"):
    """Save trade snapshot with context to disk (and optionally to DB)"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"snap_{action}_{ts}.json"

    # 1. Escribir a archivo (OBLIGATORIO - fuente de verdad)
    with open(os.path.join(SNAPSHOTS_DIR, fname), "w") as f:
        json.dump({"action": action, "market": m_title, "bucket": bucket, "price": price, "reason": reason, "context": ctx}, f, indent=2)

    # 2. Escribir a DB en shadow mode (OPCIONAL - no bloquea si falla)
    if DB_AVAILABLE:
        db.shadow_write(
            db.log_snapshot,
            action=action,
            market=m_title,
            bucket=bucket,
            price=price,
            reason=reason,
            context=ctx,
            mode=mode,
            hours_left=hours_left,
            tweet_count=tweet_count
        )

def titles_match_paranoid(t1, t2):
    """Check if two market titles match using paranoid comparison"""
    t1 = t1.lower(); t2 = t2.lower()
    if t1 in t2 or t2 in t1: return True
    def get_nums(txt): return {n for n in re.findall(r'\d+', txt) if n not in DATE_FILTER_YEARS}
    return len(get_nums(t1).intersection(get_nums(t2))) >= 2

def get_bio_multiplier():
    """Get biological rhythm multiplier based on hour and day"""
    n = datetime.now()
    return HOURLY_MULTIPLIERS.get(n.hour, 1.0) * DAILY_MULTIPLIERS.get(n.weekday(), 1.0)

def detect_event_type(m_poly, clob_buckets):
    """
    Detect if event is short (<72h) or long (≥72h)

    Uses hybrid approach:
    1. Primary: Analyze bucket sizes (short=~24 tweets, long=~19 tweets)
    2. Fallback: Heuristic from count/hours ratio
    3. Default: 'long' (conservative)

    Returns: ('short' or 'long', bucket_size or None)
    """
    # Method 1: Bucket size analysis (most reliable)
    bucket_sizes = []
    for b in clob_buckets:
        if 'max' in b and 'min' in b and b['max'] < 99999:
            size = b['max'] - b['min'] + 1
            bucket_sizes.append(size)

    if bucket_sizes:
        avg_size = sum(bucket_sizes) / len(bucket_sizes)
        # Threshold: 21.5 (midpoint between 19 and 24)
        event_type = 'short' if avg_size > 21.5 else 'long'
        return event_type, avg_size

    # Method 2: Heuristic from count/hours (fallback)
    count = m_poly.get('count', 0)
    hours_left = m_poly.get('hours', 0)

    if count > 50 and hours_left > 0:
        # If we have significant tweets, we can estimate
        ratio = count / hours_left
        # Short events typically have denser activity
        if ratio > 2.5 and hours_left < 48:
            return 'short', None

    # Fallback: assume long (conservative, won't apply aggressive fixes)
    return 'long', None
