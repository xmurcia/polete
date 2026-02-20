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
