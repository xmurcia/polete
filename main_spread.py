"""
main_spread.py
==============
Ejecutor de señales D3 que reutiliza la infraestructura del bot existente.

- Lee señales de signals/pending_signals.json (generadas por orchestrator.py)
- Ejecuta BUY en modo paper o real usando UnifiedTrader
- Reutiliza TODA la lógica de EXIT del bot (Victory Lap, Stop Loss, etc.)
- Corre en paralelo con main.py sin ningún conflicto

Uso:
  python main_spread.py               # paper mode
  python main_spread.py --real-trading  # real mode (cuando el modelo esté listo)
"""

import time
import json
import os
import sys
import argparse
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from config import *
from src.clob_scanner import ClobMarketScanner
from src.polymarket_sensor import PolymarketSensor
from src.utils import save_trade_snapshot, titles_match_paranoid

SIGNALS_FILE  = "signals/pending_signals.json"
EXECUTED_FILE = "signals/executed_signals.json"
LOOP_DELAY    = 10

os.makedirs("signals", exist_ok=True)
if not os.path.exists(SNAPSHOTS_DIR):
    os.makedirs(SNAPSHOTS_DIR)


def load_pending_signals() -> list:
    if not os.path.exists(SIGNALS_FILE):
        return []
    try:
        with open(SIGNALS_FILE) as f:
            signals = json.load(f)
        now = datetime.now()
        return [s for s in signals if _not_expired(s, now)]
    except:
        return []


def _not_expired(s, now) -> bool:
    try:
        exp = datetime.fromisoformat(s.get("expires_at", "2000-01-01"))
        return exp > now
    except:
        return True


def load_executed() -> set:
    if not os.path.exists(EXECUTED_FILE):
        return set()
    try:
        with open(EXECUTED_FILE) as f:
            return set(json.load(f))
    except:
        return set()


def mark_executed(sig_id: str):
    ex = load_executed()
    ex.add(sig_id)
    with open(EXECUTED_FILE, "w") as f:
        json.dump(list(ex), f)


def sig_id(s: dict) -> str:
    return f"{s.get('slug','')}_{s.get('type','')}_{s.get('generated_at','')}"


def execute_spread_yes(trader, signal, clob_data, m_poly, done_this_cycle):
    """Compra cada rango del spread usando UnifiedTrader."""
    m_clob = next((c for c in clob_data
                   if titles_match_paranoid(m_poly["title"], c["title"])), None)
    if not m_clob:
        return 0

    bought = 0
    for rng in signal.get("ranges", []):
        bd = next((b for b in m_clob["buckets"] if b["bucket"] == rng), None)
        if not bd:
            continue

        ask = bd.get("ask", 0)
        bid = bd.get("bid", 0)
        if ask <= 0 or ask > 0.80 or bid <= 0.001:
            continue

        trade_key = (m_poly["title"], rng, "BUY_SPREAD")
        if trade_key in done_this_cycle:
            continue

        conf      = signal.get("confidence", "MEDIA")
        risk_pct  = RISK_PCT_SPREAD * (KELLY_MULTIPLIER_MED_EDGE
                                       if conf == "ALTA" else 1.0)
        reason    = (f"SPREAD_YES D{signal.get('source','?')} "
                     f"EV{signal.get('ev',0)*100:+.0f}% {conf}")

        res = trader.execute(
            m_poly["title"], rng, "BUY", ask, reason,
            strategy_tag="SPREAD_YES",
            hours_left=m_poly.get("hours", 100),
            tweet_count=m_poly.get("count", 0),
            market_consensus=signal.get("projection"),
            entry_z_score=0.0,
        )
        if res:
            save_trade_snapshot(
                "BUY", m_poly["title"], rng, ask, reason,
                {"ev": signal.get("ev", 0), "confidence": conf,
                 "projection": signal.get("projection")},
                hours_left=m_poly.get("hours", 100),
                tweet_count=m_poly.get("count", 0),
            )
            done_this_cycle.add(trade_key)
            bought += 1
            print(f"[SPREAD] ✅ BUY {rng} @ {ask:.3f} | {reason}")
        else:
            print(f"[SPREAD] ❌ BUY falló: {rng}")

    return bought


def check_spread_exits(trader, m_poly, m_clob_obj, portfolio_cache,
                       done_this_cycle, sl_cooldowns):
    """EXIT para posiciones SPREAD_YES — reutiliza lógica del bot."""
    my_pos = {
        pid: pos for pid, pos in portfolio_cache["positions"].items()
        if (titles_match_paranoid(m_poly["title"], pos["market"])
            and pos.get("strategy_tag") == "SPREAD_YES")
    }
    if not my_pos:
        return

    bkts = {b["bucket"]: b for b in m_clob_obj.get("buckets", [])}

    for pos_id, pos in my_pos.items():
        bucket = pos["bucket"]
        bd     = bkts.get(bucket)
        if not bd:
            continue

        bid        = bd.get("bid", 0)
        entry      = pos.get("entry_price", bid)
        profit_pct = (bid - entry) / entry if entry > 0 else 0
        hours_left = m_poly.get("hours", 100)
        sell       = False
        reason     = ""

        if hours_left <= VICTORY_LAP_TIME_HOURS and bid >= VICTORY_LAP_PRICE:
            sell = True
            reason = f"Victory Lap ({bid:.2f})"
        elif hours_left <= TIME_REMAINING_HOURS_RUN and profit_pct < STOP_LOSS_CATASTROPHIC:
            sell = True
            reason = f"Catastrophic ({profit_pct*100:.0f}%)"
        elif (TIME_REMAINING_HOURS_RUN < hours_left <= VICTORY_LAP_TIME_HOURS
              and profit_pct < STOP_LOSS_MID_GAME_EMERGENCY):
            sell = True
            reason = f"Mid-Game Emergency ({profit_pct*100:.0f}%)"
        elif hours_left > VICTORY_LAP_TIME_HOURS:
            sl = (STOP_LOSS_CHEAP_ENTRY if entry < STOP_LOSS_CHEAP_THRESHOLD
                  else STOP_LOSS_NORMAL)
            if profit_pct < sl:
                sell = True
                reason = f"Stop Loss ({profit_pct*100:.0f}%)"
        if (not sell and hours_left <= SPREAD_LOSER_PRUNE_HOURS
                and bid < SPREAD_LOSER_PRUNE_BID):
            sell = True
            reason = f"Spread Prune (bid={bid:.3f})"

        if sell:
            tk = (m_poly["title"], bucket, "ROTATE")
            if tk not in done_this_cycle:
                res = trader.execute(
                    m_poly["title"], bucket, "ROTATE", bid, reason,
                    strategy_tag="SPREAD_YES",
                    hours_left=hours_left,
                    tweet_count=m_poly.get("count", 0),
                )
                if res:
                    save_trade_snapshot(
                        "SMART_ROTATE", m_poly["title"], bucket, bid, reason,
                        {"pnl": profit_pct},
                        hours_left=hours_left,
                        tweet_count=m_poly.get("count", 0),
                    )
                    print(f"[EXIT] 🔴 {bucket} @ {bid:.3f} | {reason}")
                    if any(x in reason for x in
                           ["Stop Loss", "Emergency", "Catastrophic"]):
                        sl_cooldowns[bucket] = (
                            datetime.now()
                            + timedelta(hours=COOLDOWN_STOP_LOSS_HOURS))
                    done_this_cycle.add(tk)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-trading", action="store_true")
    parser.add_argument("--initial-cash", type=float, default=500.0)
    args = parser.parse_args()

    print("\n📊 SPREAD-YES BOT — D3 Signal Executor")
    print(f"   Mode:    {'🔴 REAL' if args.real_trading else '📄 PAPER'}")
    print(f"   Capital: ${args.initial_cash:.0f}")
    print(f"   Signals: {SIGNALS_FILE}\n")

    from src.real_trader import UnifiedTrader
    trader = UnifiedTrader(use_real=args.real_trading,
                           initial_cash=args.initial_cash)
    trader.initialize()

    sensor  = ClobMarketScanner()
    polysen = PolymarketSensor()

    sl_cooldowns    = {}
    done_this_cycle = set()
    executed        = load_executed()

    while True:
        try:
            done_this_cycle.clear()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ─── ciclo ───")

            clob_data = sensor.get_market_prices()
            markets   = polysen.get_active_counts()
            if not clob_data or not markets:
                print("  💤 Sin datos")
                time.sleep(LOOP_DELAY)
                continue

            portfolio_cache = (trader.get_portfolio()
                               if hasattr(trader, "get_portfolio")
                               else trader.portfolio)

            signals = load_pending_signals()

            for m_poly in markets:
                m_clob_obj = next(
                    (c for c in clob_data
                     if titles_match_paranoid(m_poly["title"], c["title"])), None)
                if not m_clob_obj:
                    continue

                # Exits (siempre)
                check_spread_exits(trader, m_poly, m_clob_obj,
                                   portfolio_cache, done_this_cycle, sl_cooldowns)

                # Entries (si hay señal)
                for sig in signals:
                    sid  = sig_id(sig)
                    if sid in executed:
                        continue

                    slug = sig.get("slug", "")
                    if (slug.lower() not in m_poly["title"].lower()
                            and not any(p in m_poly["title"].lower()
                                        for p in slug.split("-") if len(p) > 3)):
                        continue

                    stype = sig.get("type", "")

                    if stype == "SPREAD_YES":
                        n = execute_spread_yes(trader, sig, clob_data,
                                               m_poly, done_this_cycle)
                        if n > 0:
                            mark_executed(sid)
                            executed.add(sid)
                            print(f"  ✅ {n} rangos comprados")

                    elif stype == "BURST_EXPAND":
                        print(f"\n  🔥 BURST: {sig.get('burst_topic')}  "
                              f"pace={sig.get('burst_pace'):.0f}/h  "
                              f"new_proj={sig.get('new_proj')}")
                        print(f"     Rangos: {sig.get('new_ranges')}")
                        print(f"     ⏱ <{sig.get('urgency_min', 15)} min")
                        mark_executed(sid)
                        executed.add(sid)

            trader.print_summary(clob_data)
            time.sleep(LOOP_DELAY)

        except KeyboardInterrupt:
            print("\n[SPREAD BOT] Detenido.")
            break
        except Exception as e:
            print(f"[SPREAD BOT] Error: {e}")
            time.sleep(LOOP_DELAY)


if __name__ == "__main__":
    run()
