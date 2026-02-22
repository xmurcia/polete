import time
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from scipy.stats import norm

# ==========================================
# LOAD ENVIRONMENT VARIABLES (for dual-write)
# ==========================================
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# IMPORT CONFIGURATION FROM CENTRALIZED FILE
# ==========================================
from config import *
from src.clob_scanner import ClobMarketScanner
from src.polymarket_sensor import PolymarketSensor
from src.paper_trader import PaperTrader
from src.market_panic_sensor import MarketPanicSensor
from src.moonshot import ejecutar_moonshot_satelite
from src.auto_hedge import gestionar_cobertura_final
from src.utils import save_market_tape, save_trade_snapshot, titles_match_paranoid, get_bio_multiplier, detect_event_type

# Asegurar que existen los directorios necesarios
if not os.path.exists(SNAPSHOTS_DIR): os.makedirs(SNAPSHOTS_DIR)
if not os.path.exists(MARKET_TAPE_DIR): os.makedirs(MARKET_TAPE_DIR)

def run():
    print("\n🤖 ELON-BOT V12.16 (ACCUMULATION STRATEGY RESTORED + MOONSHOT V33 + AUTO-HEDGE)")

    # ==========================================
    # ARGUMENT PARSER
    # ==========================================
    def parse_args():
        """Parse command-line arguments"""
        parser = argparse.ArgumentParser(description='Polymarket Elon Tweet Trading Bot')

        parser.add_argument(
            '--real-trading',
            action='store_true',
            help='Enable REAL trading mode (uses blockchain)'
        )

        parser.add_argument(
            '--initial-cash',
            type=float,
            default=1000.0,
            help='Initial cash for paper trading mode (default: 1000.0)'
        )

        return parser.parse_args()

    # ==========================================
    # IMPORT UNIFIED TRADER
    # ==========================================
    try:
        from src.real_trader import UnifiedTrader
        UNIFIED_TRADER_AVAILABLE = True
    except ImportError:
        print("⚠️  Warning: UnifiedTrader not available. Using PaperTrader only.")
        UNIFIED_TRADER_AVAILABLE = False

    # ==========================================
    # INITIALIZE TRADER BASED ON MODE
    # ==========================================
    args = parse_args()
    
    sensor = PolymarketSensor()
    pricer = ClobMarketScanner()

    # Initialize trader based on mode
    if args.real_trading:
        if not UNIFIED_TRADER_AVAILABLE:
            print("❌ Error: --real-trading requires UnifiedTrader module")
            print("   Install: pip install -r requirements.txt")
            sys.exit(1)

        print("🔴 REAL TRADING MODE")
        trader = UnifiedTrader(use_real=True)
        trader.initialize()

        portfolio = trader.get_portfolio()
        print(f"💰 Blockchain balance: ${portfolio['cash']:.2f}")
        print(f"📊 Open positions: {len(portfolio['positions'])}")

    else:
        # Paper mode (default)
        if UNIFIED_TRADER_AVAILABLE:
            print("📄 PAPER MODE (using UnifiedTrader)")
            trader = UnifiedTrader(use_real=False, initial_cash=args.initial_cash)
            trader.initialize()
        else:
            print("📄 PAPER MODE (using legacy PaperTrader)")
            trader = PaperTrader(initial_cash=args.initial_cash)

        print(f"💵 Initial cash: ${trader.portfolio['cash']:.2f}")

    panic_sensor = MarketPanicSensor()

    last_counts = {}
    last_tape = 0
    # Set to 0 to send portfolio summary on first iteration (for testing)
    # Change to time.time() to wait 2 hours
    last_positions_summary = 0  # Track last positions summary (every 2 hours)
    last_daily_digest_date = datetime.now().date()  # Track last daily digest date

    global_events = []
    if os.path.exists(os.path.join(LOGS_DIR, LIVE_LOG)):
        try:
            with open(os.path.join(LOGS_DIR, LIVE_LOG)) as f:
                d = json.load(f)
                global_events = [e for e in d if (time.time()*MS_PER_SECOND - e['timestamp']) < MS_PER_DAY]
        except: pass

    # ==============================================================================
    # ❄️ INIT COOLDOWNS: Memoria de castigos
    # ==============================================================================
    stop_loss_cooldowns = {}
    moonshot_cooldowns = {}

    # ==============================================================================
    # 🔒 TRADE LOCK: Evitar trades duplicados en el mismo ciclo
    # ==============================================================================
    executed_trades_this_cycle = set()  # Set of (market_title, bucket, signal) tuples 

    while True:
        try:
            # Clear trade lock at the start of each cycle
            executed_trades_this_cycle.clear()

            # Show mode indicator
            mode_indicator = "🔴 REAL" if (hasattr(trader, 'use_real') and trader.use_real) else "📄 PAPER"
            print(f"\n{'='*80}")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | {mode_indicator}")
            print(f"{'='*80}")

            # 1. Obtener Mercados
            markets = sensor.get_active_counts()
            if not markets:
                print("💤 Waiting for data...", end="\r"); time.sleep(LOOP_NO_DATA_WAIT_SECONDS); continue

            # 2. Tweets
            ts_now = time.time() * 1000
            for m in markets:
                curr = m['count']
                prev = last_counts.get(m['id'])
                if prev is not None and curr > prev:
                    diff = curr - prev
                    print(f"\n🚨 TWEET DETECTADO! (+{diff})")
                    for _ in range(diff): global_events.append({'timestamp': ts_now})
                    with open(os.path.join(LOGS_DIR, LIVE_LOG), 'w') as f: json.dump(global_events, f)
                last_counts[m['id']] = curr
            
            global_events = [e for e in global_events if (ts_now - e['timestamp']) < MS_PER_DAY]
            ts_list = [e['timestamp'] for e in global_events]
            IS_WARMUP = len(ts_list) < 5

            # 3. Pre-cargar portfolio (CRITICAL for REAL mode exit logic)
            portfolio_cache = trader.get_portfolio() if hasattr(trader, 'get_portfolio') else trader.portfolio

            # DEBUG: Print entire portfolio cache to diagnose matching issues
            print(f"\n[DEBUG PORTFOLIO] Total positions: {len(portfolio_cache['positions'])}")
            for pos_id, pos in portfolio_cache['positions'].items():
                print(f"[DEBUG PORTFOLIO] {pos_id} → Market: '{pos['market']}', Bucket: '{pos['bucket']}', Shares: {pos.get('shares', 0)}")
            print("")

            # 4. Precios y Analisis
            clob_data = pricer.get_market_prices()
            if clob_data:
                # Tape
                if time.time() - last_tape > MARKET_TAPE_SAVE_INTERVAL_SECONDS:
                    save_market_tape(clob_data, markets); last_tape = time.time()

                # Prefetch token_ids for real mode (eliminates per-trade Gamma API latency)
                if hasattr(trader, 'use_real') and trader.use_real:
                    buckets_by_market = {}
                    for m_poly in markets:
                        m_clob = next((c for c in clob_data if titles_match_paranoid(m_poly['title'], c['title'])), None)
                        if m_clob:
                            buckets_by_market[m_poly['title']] = [b['bucket'] for b in m_clob['buckets']]
                    if buckets_by_market:
                        trader.prefetch_token_ids(buckets_by_market)

                # Panic
                alerts = panic_sensor.analyze(clob_data)
                for a in alerts:
                    print(f"⚠️ PÁNICO V10: {a['type']} en {a['bucket']} (Price: {a['price']})")

                for m_poly in markets:
                    m_clob = next((c for c in clob_data if titles_match_paranoid(m_poly['title'], c['title'])), None)
                    if not m_clob: continue

                    # ==============================================================
                    # 🧠 CEREBRO V22: UNIVERSAL SCALE (AGNOSTIC)
                    # ==============================================================
                    try:
                        p_count = m_poly.get('count', 0)
                        p_hours_left = m_poly.get('hours', HEDGE_MAX_TIME_HOURS)
                        p_avg_hist = m_poly.get('daily_avg', 45.0)

                        if p_count > 130 or p_hours_left > MOONSHOT_MIN_EVENT_DURATION: total_duration = EVENT_DURATION_HOURS_LONG
                        elif p_hours_left < EVENT_DURATION_HOURS_VERY_SHORT: total_duration = EVENT_DURATION_HOURS_SHORT
                        else: total_duration = EVENT_DURATION_HOURS_MID

                        hours_elapsed = max(1.0, total_duration - p_hours_left)
                        rate_actual_diario = (p_count / hours_elapsed) * HOURS_PER_DAY

                        if p_hours_left < TIME_REMAINING_HOURS_SPRINT:
                            fatigue_weight = FATIGUE_WEIGHT_SPRINT; mode_tag = MODE_TAG_SPRINT
                        elif p_hours_left < TIME_REMAINING_HOURS_RUN:
                            fatigue_weight = FATIGUE_WEIGHT_RUN; mode_tag = MODE_TAG_RUN
                        else:
                            fatigue_weight = FATIGUE_WEIGHT_MARATHON; mode_tag = MODE_TAG_MARATHON

                        projected_rate = (rate_actual_diario * fatigue_weight) + (p_avg_hist * (1 - fatigue_weight))

                        if p_hours_left > TIME_REMAINING_HOURS_RUN:
                            max_rate_allowed = p_avg_hist * MAX_RATE_MULTIPLIER_MARATHON
                            projected_rate = min(projected_rate, max_rate_allowed)

                        mean_prediction = p_count + (projected_rate / HOURS_PER_DAY * p_hours_left)
                        final_mean = mean_prediction 
                        
                        try:
                            all_buckets = m_clob.get('buckets', [])
                            market_buckets = [b for b in all_buckets if b['bid'] > CONSENSUS_MIN_BID]

                            if market_buckets and p_hours_left > CONSENSUS_TIME_MIN_HOURS:
                                bucket_sizes = []
                                for b in all_buckets:
                                    size = b['max'] - b['min']
                                    if size < 100000 and size > 0:
                                        bucket_sizes.append(size)
                                if bucket_sizes:
                                    bucket_sizes.sort()
                                    avg_step = bucket_sizes[len(bucket_sizes)//2]
                                else: avg_step = 10.0

                                w_sum = 0; w_vol = 0
                                for b in market_buckets:
                                    range_size = b['max'] - b['min']
                                    if range_size > (avg_step * LARGE_BUCKET_SIZE_MULTIPLIER):
                                        mid_val = b['min'] + avg_step
                                    else: mid_val = (b['min'] + b['max']) / 2
                                    w_sum += mid_val * b['bid']
                                    w_vol += b['bid']

                                if w_vol > 0:
                                    consensus = w_sum / w_vol
                                    if abs(final_mean - consensus) > (consensus * CONSENSUS_DEVIATION_THRESHOLD):
                                        final_mean = (final_mean * CONSENSUS_MODEL_WEIGHT_WHEN_DIVERGENT) + (consensus * CONSENSUS_WEIGHT_WHEN_DIVERGENT)
                                        mode_tag += MODE_TAG_MARKET_CONSENSUS
                        except Exception: pass 

                        raw_sigma = (final_mean ** 0.5) * SIGMA_BASE_MULTIPLIER
                        time_factor = (p_hours_left / SIGMA_TIME_DECAY_BASE) ** 0.5
                        eff_std = max(raw_sigma * max(SIGMA_TIME_FACTOR_MIN, time_factor), SIGMA_MIN_VALUE)
                        brain_mode = f"🧠 {mode_tag}"

                    except Exception as e:
                        print(f"Brain Error: {e}")
                        final_mean = p_count + (p_avg_hist / HOURS_PER_DAY * p_hours_left)
                        eff_std = 5.0
                        brain_mode = MODE_TAG_ERROR

                    print("-" * TABLE_WIDTH)
                    print(f">>> {m_poly['title']}")
                    time_str = f"{p_hours_left/HOURS_PER_DAY:.1f}d" if p_hours_left > HOURS_PER_DAY else f"{p_hours_left:.1f}h"
                    print(f"    Tweets: {m_poly['count']} | ⏳ {time_str} | {brain_mode}: {final_mean:.1f} (σ={eff_std:.1f})")
                    print("-" * TABLE_WIDTH)
                    print(f"{'BUCKET':<10} | {'BID':<6} | {'ASK':<6} | {'FAIR':<6} | {'Z-SCR':<6} | {'ACTION'}")

                    my_buckets_ids = []
                    for pos in portfolio_cache['positions'].values():
                        if titles_match_paranoid(m_poly['title'], pos['market']):
                            my_buckets_ids.append(pos['bucket'])

                    # DEBUG: Print ALL markets being evaluated with owned buckets
                    if my_buckets_ids:
                        print(f"[DEBUG] Market: {m_poly['title']}, Owned: {my_buckets_ids}, Hours: {m_poly.get('hours', 'N/A')}")

                    for b in m_clob['buckets']:
                        if b['max'] < m_poly['count']: continue

                        min_req = WARMUP_MIN_TWEETS_LONG if m_poly['hours'] > MOONSHOT_MIN_EVENT_DURATION else WARMUP_MIN_TWEETS_SHORT
                        IS_WARMUP = m_poly['count'] < min_req

                        if b['bucket'] in stop_loss_cooldowns:
                            if datetime.now() < stop_loss_cooldowns[b['bucket']]: continue
                            else: del stop_loss_cooldowns[b['bucket']]

                        bid, ask = b.get('bid',0), b.get('ask',0)

                        if bid <= MIN_BID_FOR_PROCESSING and p_hours_left > 2.0: continue

                        if b['max'] >= BUCKET_MAX_OPEN_ENDED: mid = b['min'] + BUCKET_OPEN_ENDED_MID_OFFSET
                        else: mid = (b['min'] + b['max']) / 2

                        h_left = m_poly.get('hours', HEDGE_MAX_TIME_HOURS)
                        decay_factor = (h_left / SIGMA_DECAY_BASE_HOURS) ** 0.5
                        decay_factor = max(SIGMA_DECAY_FACTOR_MIN, min(SIGMA_DECAY_FACTOR_MAX, decay_factor)) 
                        decayed_std = eff_std * decay_factor
                        
                        z_score = abs(mid - final_mean) / decayed_std
                        p_min = norm.cdf(b['min'], final_mean, decayed_std)
                        if b['max'] >= BUCKET_MAX_OPEN_ENDED: fair = 1.0 - p_min
                        else: fair = norm.cdf(b['max']+1, final_mean, decayed_std) - p_min
                        
                        action = "-"
                        reason = ""
                        owned = b['bucket'] in my_buckets_ids
                        
                        if owned:
                            pos_data = next((v for k,v in portfolio_cache['positions'].items()
                                           if v['bucket'] == b['bucket'] and titles_match_paranoid(m_poly['title'], v['market'])), None)
                            if pos_data:
                                entry = pos_data['entry_price']
                                profit_pct = (bid - entry) / entry if entry > 0 else 0
                                
                                if pos_data.get('strategy_tag') == 'MOONSHOT':
                                    trade_key = (m_poly['title'], b['bucket'], "ROTATE")
                                    if trade_key not in executed_trades_this_cycle:
                                        if bid <= MOONSHOT_EXIT_EXPIRED_PRICE:
                                            action = "SMART_ROTATE"; reason = f"Moonshot Expired (Total Loss) Z{z_score:.1f}"
                                            res = trader.execute(m_poly['title'], b['bucket'], "ROTATE", bid, reason)
                                            if res:
                                                save_trade_snapshot("SMART_ROTATE", m_poly['title'], b['bucket'], bid, reason, {"z": z_score, "pnl": profit_pct})
                                                moonshot_cooldowns[b['bucket']] = datetime.now() + timedelta(hours=COOLDOWN_MOONSHOT_EXPIRED_HOURS)
                                                executed_trades_this_cycle.add(trade_key)
                                            continue

                                        if bid >= MOONSHOT_EXIT_VICTORY_PRICE:
                                            action = "SMART_ROTATE"; reason = f"Moonshot Victory Lap (${bid:.2f}) Z{z_score:.1f}"
                                            res = trader.execute(m_poly['title'], b['bucket'], "ROTATE", bid, reason)
                                            if res:
                                                save_trade_snapshot("SMART_ROTATE", m_poly['title'], b['bucket'], bid, reason, {"z": z_score, "pnl": profit_pct})
                                                executed_trades_this_cycle.add(trade_key)
                                            continue

                                        if MOONSHOT_EXIT_PARTIAL_MIN_PRICE <= bid <= MOONSHOT_EXIT_PARTIAL_MAX_PRICE and profit_pct >= MOONSHOT_EXIT_PARTIAL_MIN_PROFIT:
                                            action = "SMART_ROTATE"; reason = f"Moonshot Partial Exit (Lock {profit_pct*100:.0f}%, ${bid:.2f})"
                                            res = trader.execute(m_poly['title'], b['bucket'], "ROTATE", bid, reason)
                                            if res:
                                                save_trade_snapshot("SMART_ROTATE", m_poly['title'], b['bucket'], bid, reason, {"z": z_score, "pnl": profit_pct})
                                                executed_trades_this_cycle.add(trade_key)
                                            continue

                                    current_max = pos_data.get('max_price_seen', entry)
                                    if bid > current_max:
                                        pos_data['max_price_seen'] = bid
                                        current_max = bid
                                        trader._save()

                                    if current_max >= MOONSHOT_TRAILING_PEAK_THRESHOLD:
                                        drawdown_from_peak = current_max - bid
                                        if drawdown_from_peak >= MOONSHOT_TRAILING_DRAWDOWN:
                                            trade_key = (m_poly['title'], b['bucket'], "ROTATE")
                                            if trade_key not in executed_trades_this_cycle:
                                                action = "SMART_ROTATE"; reason = f"Moonshot Trailing Stop (Peak ${current_max:.2f} -> ${bid:.2f}) Z{z_score:.1f}"
                                                res = trader.execute(m_poly['title'], b['bucket'], "ROTATE", bid, reason)
                                                if res:
                                                    save_trade_snapshot("SMART_ROTATE", m_poly['title'], b['bucket'], bid, reason, {"z": z_score, "pnl": profit_pct})
                                                    moonshot_cooldowns[b['bucket']] = datetime.now() + timedelta(hours=COOLDOWN_MOONSHOT_EXIT_HOURS)
                                                    executed_trades_this_cycle.add(trade_key)
                                            continue
                                    continue
                                
                                should_sell = False; sell_reason = ""
                                bucket_headroom = b['max'] - m_poly['count']
                                hours_left = m_poly['hours']

                                if hours_left > TIME_REMAINING_HOURS_RUN: base_threshold = PROXIMITY_BASE_THRESHOLD_LONG
                                elif hours_left > 12.0: base_threshold = PROXIMITY_BASE_THRESHOLD_MID
                                elif hours_left > TIME_REMAINING_HOURS_SPRINT: base_threshold = PROXIMITY_BASE_THRESHOLD_SHORT
                                else: base_threshold = PROXIMITY_BASE_THRESHOLD_FINAL

                                volatility_buffer = int(decayed_std * PROXIMITY_VOLATILITY_MULTIPLIER)
                                safety_threshold = base_threshold + volatility_buffer

                                # --- INICIO MODIFICACIÓN SMART PROXIMITY ---
                                # 1. Calcular si estamos cubiertos arriba
                                is_covered_above = False
                                next_neighbor_min = b['max'] + 1
                                for owned_b in my_buckets_ids:
                                    try:
                                        if "+" in owned_b: o_min = int(owned_b.replace("+",""))
                                        else: o_min = int(owned_b.split("-")[0])
                                        
                                        if o_min == next_neighbor_min:
                                            is_covered_above = True
                                            break
                                    except: pass

                                # 2. Evaluar Proximity Danger
                                if bucket_headroom < safety_threshold and bucket_headroom >= 0:
                                    if is_covered_above:
                                        # Si estamos cubiertos, ignoramos el peligro de proximidad
                                        pass 
                                    else:
                                        # Si NO estamos cubiertos, vendemos por pánico
                                        should_sell = True; sell_reason = f"Proximity Danger ({bucket_headroom} left)" 
                                # --- FIN MODIFICACIÓN ---

                                # DEBUG: Print Victory Lap evaluation for ALL owned positions
                                print(f"[DEBUG VL] {b['bucket']}: hrs={hours_left:.2f}, bid={bid:.3f}, should_VL={(hours_left <= VICTORY_LAP_TIME_HOURS and bid > VICTORY_LAP_PRICE)}")

                                if hours_left <= VICTORY_LAP_TIME_HOURS and bid > VICTORY_LAP_PRICE:
                                    should_sell = True; sell_reason = f"Victory Lap (Price {bid:.2f} > {VICTORY_LAP_PRICE})"

                                # CATASTROPHIC STOP LOSS: Final phase only (< 24h)
                                # Triggers when loss is extreme AND mathematically impossible to recover
                                elif hours_left <= TIME_REMAINING_HOURS_RUN and profit_pct < STOP_LOSS_CATASTROPHIC and z_score > STOP_LOSS_CATASTROPHIC_Z_MIN:
                                    should_sell = True; sell_reason = f"Catastrophic Loss ({profit_pct*100:.0f}%, Z{z_score:.1f} impossible)"

                                elif hours_left > TIME_REMAINING_HOURS_RUN:
                                    profit_threshold = PROFIT_PROTECT_Z_MID if hours_left > VICTORY_LAP_TIME_HOURS else PROFIT_PROTECT_Z_LONG
                                    if profit_pct > PARANOID_TREASURE_MIN_PCT and z_score > PARANOID_TREASURE_Z: should_sell = True; sell_reason = "Paranoid Treasure (Secured)"
                                    elif profit_pct > PROFIT_PROTECT_MIN_PCT and z_score > profit_threshold: should_sell = True; sell_reason = f"Protect Profit (Mid-Game Z{profit_threshold})"

                                    avg_entry = bid / (1 + profit_pct) if (1 + profit_pct) != 0 else bid

                                    # Si entramos barato, NUNCA vendemos por pérdidas (Spread Filter)
                                    if avg_entry < STOP_LOSS_CHEAP_THRESHOLD:
                                        sl_limit = STOP_LOSS_CHEAP_ENTRY
                                    else:
                                        sl_limit = STOP_LOSS_NORMAL

                                    if hours_left < VICTORY_LAP_TIME_HOURS: sl_limit = STOP_LOSS_LATE_GAME

                                    if profit_pct < sl_limit and z_score > STOP_LOSS_Z_MIN: should_sell = True; sell_reason = f"Stop Loss Adaptativo (Hit {profit_pct*100:.1f}%)"
                                    elif z_score > EXTREME_PANIC_Z and profit_pct < EXTREME_PANIC_MAX_PROFIT: should_sell = True; sell_reason = f"Extreme Panic (Z>{EXTREME_PANIC_Z})"

                                if should_sell:
                                    # Check if trade already executed this cycle
                                    trade_key = (m_poly['title'], b['bucket'], "ROTATE")
                                    if trade_key not in executed_trades_this_cycle:
                                        action = "SMART_ROTATE"; reason = f"{sell_reason} Z{z_score:.1f}"
                                        res = trader.execute(m_poly['title'], b['bucket'], "ROTATE", bid, reason)
                                        if res:
                                            save_trade_snapshot("SMART_ROTATE", m_poly['title'], b['bucket'], bid, reason, {"z": z_score, "pnl": profit_pct})
                                            executed_trades_this_cycle.add(trade_key)  # Mark as executed

                        elif not owned and not IS_WARMUP:
                            bucket_headroom = b['max'] - m_poly['count']
                            hours_left = m_poly['hours']

                            if hours_left > TIME_REMAINING_HOURS_RUN: base_threshold = PROXIMITY_BASE_THRESHOLD_LONG
                            elif hours_left > 12.0: base_threshold = PROXIMITY_BASE_THRESHOLD_MID
                            elif hours_left > TIME_REMAINING_HOURS_SPRINT: base_threshold = PROXIMITY_BASE_THRESHOLD_SHORT
                            elif hours_left > TIME_REMAINING_HOURS_VERY_LATE: base_threshold = PROXIMITY_BASE_THRESHOLD_FINAL
                            else: base_threshold = 2

                            volatility_buffer = int(decayed_std * PROXIMITY_VOLATILITY_MULTIPLIER)
                            buy_safety = base_threshold + volatility_buffer

                            if bucket_headroom < buy_safety: continue

                            is_impossible = False
                            if m_poly['hours'] < TIME_REMAINING_HOURS_VERY_LATE:
                                tweets_needed = b['min'] - m_poly['count']
                                if tweets_needed > (m_poly['hours'] * IMPOSSIBLE_TWEETS_PER_HOUR): is_impossible = True

                            if not is_impossible:
                                if z_score <= MAX_Z_SCORE_ENTRY and ask >= MIN_PRICE_ENTRY:
                                    edge = fair - ask
                                    dynamic_min_edge = MIN_EDGE_BASE + (decayed_std * EDGE_STD_MULTIPLIER)
                                    if edge > dynamic_min_edge:
                                        # FIX 2: Dynamic clustering for short events
                                        passes_clustering = True
                                        if ENABLE_CLUSTERING and my_buckets_ids:
                                            # Detect event type and calculate dynamic cluster range
                                            event_type, bucket_size_detected = detect_event_type(m_poly, m_clob['buckets'])

                                            if event_type == 'short':
                                                # SHORT EVENTS: Use dynamic clustering based on time
                                                if hours_left > 24:
                                                    multiplier = CLUSTER_MULTIPLIER_SHORT_EARLY  # 1.5x buckets
                                                else:
                                                    multiplier = CLUSTER_MULTIPLIER_SHORT_LATE   # 1.0x buckets

                                                # Use detected bucket size or default
                                                avg_bucket_size = bucket_size_detected if bucket_size_detected else 24
                                                cluster_range = avg_bucket_size * multiplier
                                            else:
                                                # LONG EVENTS: Use fixed cluster range (already works well)
                                                cluster_range = CLUSTER_RANGE

                                            # Check distance to existing positions
                                            try:
                                                if "+" in b['bucket']:
                                                    new_min = int(b['bucket'].replace("+", ""))
                                                else:
                                                    new_min = int(b['bucket'].split("-")[0])

                                                for owned_bucket in my_buckets_ids:
                                                    try:
                                                        if "+" in owned_bucket:
                                                            owned_min = int(owned_bucket.replace("+", ""))
                                                        else:
                                                            owned_min = int(owned_bucket.split("-")[0])

                                                        distance = abs(new_min - owned_min)
                                                        if distance > cluster_range:
                                                            passes_clustering = False
                                                            break
                                                    except:
                                                        pass
                                            except:
                                                pass

                                        if passes_clustering:
                                            # Check if trade already executed this cycle
                                            trade_key = (m_poly['title'], b['bucket'], "BUY")
                                            if trade_key not in executed_trades_this_cycle:
                                                action = "BUY"; reason = f"Val+{edge:.2f}"
                                                res = trader.execute(m_poly['title'], b['bucket'], "BUY", ask, reason)
                                                if res:
                                                    save_trade_snapshot("BUY", m_poly['title'], b['bucket'], ask, reason, {"z": z_score, "fair": fair})
                                                    executed_trades_this_cycle.add(trade_key)  # Mark as executed
                                            else:
                                                action = "-"; reason = "Already executed this cycle"

                        color_act = f"🟢 {action}" if "BUY" in action else (f"🔴 {action}" if "ROTATE" in action or "PROFIT" in action else "-")
                        bucket_display = f"*{b['bucket']}" if owned else f"{b['bucket']}"
                        print(f"{bucket_display:<10} | {bid:.3f}  | {ask:.3f}  | {fair:.3f}  | {z_score:.1f}   | {color_act} {reason}")

            # ==============================================================================
            # 🧼 FIX FINAL: HIGIENE TOTAL DEL PORTFOLIO (ONLY PAPER MODE)
            # ==============================================================================
            # In real mode, positions are synced from blockchain and filtered automatically
            # This cleanup is only needed for paper mode where positions persist in memory
            if not (hasattr(trader, 'use_real') and trader.use_real):
                live_markets_map = {}
                for m in markets:
                    clean_title = ''.join(filter(str.isalnum, m['title'].lower()))
                    live_markets_map[clean_title] = m

                    for symbol in list(trader.portfolio['positions'].keys()):
                        pos = trader.portfolio['positions'][symbol]
                        pos_fingerprint = ''.join(filter(str.isalnum, pos['market'].lower()))

                        found_market = None
                        for live_fp, m_obj in live_markets_map.items():
                            if pos_fingerprint in live_fp or live_fp in pos_fingerprint:
                                found_market = m_obj
                                break

                        if not found_market:
                            print(f"🧹 LIMPIEZA: Mercado expirado. Eliminando posición {pos['bucket']}.")
                            del trader.portfolio['positions'][symbol]
                            continue

                        try:
                            bucket_str = pos['bucket']
                            if "+" not in bucket_str:
                                max_val = int(bucket_str.split('-')[1])
                                current_count = found_market['count']
                                if current_count > max_val:
                                    pos['current_price'] = 0.0
                                    pos['market_value'] = 0.0
                                    clob_data[pos['bucket']] = 0.0
                        except: pass

            # ==================================================================
            # 🌑 LLAMADA AL SATÉLITE MOONSHOT (AL FINAL DEL CICLO)
            # ==================================================================
            try:
                ejecutar_moonshot_satelite(trader, m_poly, m_clob, p_count, p_avg_hist, p_hours_left, moonshot_cooldowns)
            except: pass
            
            # ==================================================================
            # 🛡️ LLAMADA AL AUTO-HEDGE (NUEVO)
            # ==================================================================
            try:
                if m_clob and 'buckets' in m_clob:
                    gestionar_cobertura_final(trader, m_poly, m_clob['buckets'])
            except Exception as e: pass
            # ==================================================================

            trader.print_summary(clob_data)

            # ==================================================================
            # 📊 TELEGRAM NOTIFICATIONS - PORTFOLIO SUMMARY (Every 2 hours)
            # ==================================================================
            if hasattr(trader, 'telegram') and trader.telegram and trader.telegram.enabled:
                current_time = time.time()

                # Portfolio summary every 2 hours
                if current_time - last_positions_summary >= 7200:  # 2 hours
                    try:
                        mode = "REAL" if (hasattr(trader, 'use_real') and trader.use_real) else "PAPER"

                        if mode == "REAL" and hasattr(trader, 'position_tracker'):
                            # Real mode: Get positions from position_tracker with proper bucket labels
                            positions_raw = trader.position_tracker.get_positions()
                            positions_list = []

                            print(f"[Telegram] 🔍 Procesando {len(positions_raw)} posiciones...")

                            for pos in positions_raw:
                                # ALWAYS resolve from API to ensure correct data
                                print(f"[Telegram] 🔍 Resolviendo posición token {pos.token_id[:20]}...")

                                try:
                                    resolved = trader._resolve_position_display_sync(pos.token_id, pos.event_slug)
                                    event_label = resolved['event_label']
                                    bucket = resolved['bucket']

                                    # Cache it for print_summary
                                    trader._token_metadata[pos.token_id] = {
                                        'market_title': resolved['market_title'],
                                        'bucket': bucket
                                    }

                                    print(f"[Telegram] ✅ {event_label} | {bucket}")

                                    positions_list.append({
                                        'event_slug': event_label,  # Use date label like "Feb 13 - Feb 20"
                                        'range_label': bucket,  # Use actual bucket like "280-299"
                                        'size': pos.size,
                                        'avg_entry_price': pos.avg_entry_price,
                                        'current_price': pos.current_price,
                                        'unrealized_pnl': pos.unrealized_pnl
                                    })
                                except Exception as e:
                                    print(f"[Telegram] ❌ Error resolviendo posición: {e}")
                                    # Fallback to raw data
                                    positions_list.append({
                                        'event_slug': pos.event_slug[:12],
                                        'range_label': pos.range_label,
                                        'size': pos.size,
                                        'avg_entry_price': pos.avg_entry_price,
                                        'current_price': pos.current_price,
                                        'unrealized_pnl': pos.unrealized_pnl
                                    })

                            # Get balance (sync wrapper)
                            import asyncio
                            try:
                                asyncio.get_running_loop()
                                balance = 0  # Can't call async from async context
                            except RuntimeError:
                                balance = asyncio.run(trader.balance_mgr.get_available_balance())
                        else:
                            # Paper mode: Use portfolio dict + live prices from clob_data
                            portfolio = trader.get_portfolio()
                            positions_list = []

                            for pos_id, pos_data in portfolio.get('positions', {}).items():
                                entry_price = pos_data.get('entry_price', 0)
                                bucket_label = pos_data.get('bucket', '')
                                market_title = pos_data.get('market', '')
                                shares = pos_data.get('shares', 0)

                                # Intentar resolver precio actual desde clob_data (fuente: clob)
                                live_price = 0.0
                                price_source = 'caché'  # etiqueta de fuente por defecto

                                if clob_data:
                                    for clob_mkt in clob_data:
                                        if titles_match_paranoid(market_title, clob_mkt['title']):
                                            for b in clob_mkt.get('buckets', []):
                                                if str(b['bucket']) == str(bucket_label):
                                                    # bid = mejor precio de compra (lo que el mercado paga)
                                                    live_price = b.get('bid', 0.0)
                                                    price_source = 'clob'
                                                    break
                                            break

                                # Fallback a precio de entrada si la API no devolvió dato válido
                                if live_price == 0.0:
                                    live_price = entry_price
                                    price_source = 'caché'

                                # P&L no realizado calculado con precio en vivo [FUENTE: calculado]
                                invested = shares * entry_price
                                unrealized_pnl = (live_price - entry_price) * shares

                                # Log de trazabilidad con etiqueta de fuente
                                print(
                                    f"[PAPER][FUENTE: {price_source}] "
                                    f"{market_title[:25]} | {bucket_label} "
                                    f"| Entrada={entry_price:.3f} "
                                    f"| Actual={live_price:.3f} [FUENTE: {price_source}] "
                                    f"| Valor=${invested:.2f} [FUENTE: calculado] "
                                    f"| P&L=${unrealized_pnl:+.2f} [FUENTE: calculado]"
                                )

                                positions_list.append({
                                    'event_slug': market_title,
                                    'range_label': bucket_label,
                                    'size': shares,
                                    'avg_entry_price': entry_price,   # [FUENTE: caché portfolio.json]
                                    'current_price': live_price,      # [FUENTE: clob | caché]
                                    'unrealized_pnl': unrealized_pnl, # [FUENTE: calculado]
                                    'price_source': price_source       # etiqueta para la notificación
                                })

                            balance = portfolio.get('cash', 0)

                        trader.telegram.notify_positions_summary(
                            positions=positions_list,
                            balance=balance,
                            mode=mode
                        )

                        last_positions_summary = current_time
                        print(f"[Telegram] 📊 Portfolio summary enviado")
                    except Exception as e:
                        print(f"[Telegram] ❌ Error enviando portfolio summary: {e}")

                # Daily digest at midnight
                current_date = datetime.now().date()
                if current_date != last_daily_digest_date:
                    try:
                        # TODO: Implement daily stats tracking
                        # For now, skip daily digest until we have stats
                        last_daily_digest_date = current_date
                        print(f"[Telegram] 📅 Nuevo día detectado: {current_date}")
                    except Exception as e:
                        print(f"[Telegram] ❌ Error con daily digest: {e}")

            time.sleep(LOOP_DELAY_SECONDS) 

        except KeyboardInterrupt: break
        except Exception as e:
            error_msg = f"Loop Error: {e}"
            print(error_msg)

            # Send Telegram notification for critical errors (only in real mode)
            if hasattr(trader, 'telegram') and trader.telegram and hasattr(trader, 'use_real') and trader.use_real:
                import traceback
                full_error = f"{error_msg}\n\n{traceback.format_exc()}"
                trader.telegram.notify_error(full_error, context="Main loop exception")

            time.sleep(LOOP_ERROR_RETRY_SECONDS)

if __name__ == "__main__":
    run()