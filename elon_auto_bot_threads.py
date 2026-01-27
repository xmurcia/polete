import time
import json
import os
import glob
import requests
import numpy as np
import pandas as pd
import re
from datetime import datetime, timezone
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
import dateutil.parser
from collections import deque
from scipy.stats import norm 

# ==========================================
# CONFIGURACIÓN (V10 ORIGINAL)
# ==========================================
DAILY_METRICS_THREE_WEEKS_DIR = "daily_metrics_three_weeks"
LOGS_DIR = 'logs'

PORTFOLIO_PAPER_TRADER = "portfolio.json"
LIVE_LOG = "live_history.json"
TRADE_LOG = "trade_history.csv"
MONITOR_LOG = "bot_monitor.log"
SNAPSHOTS_DIR = os.path.join(LOGS_DIR, "snapshots")
MARKET_TAPE_DIR = os.path.join(LOGS_DIR, "market_tape")

# Aseguramos directorios
if not os.path.exists(DAILY_METRICS_THREE_WEEKS_DIR): os.makedirs(DAILY_METRICS_THREE_WEEKS_DIR)
if not os.path.exists(SNAPSHOTS_DIR): os.makedirs(SNAPSHOTS_DIR)
if not os.path.exists(MARKET_TAPE_DIR): os.makedirs(MARKET_TAPE_DIR)

# --- AJUSTES V11 (SOLO PARAMETROS) ---
MAX_Z_SCORE_ENTRY = 1.6
MIN_PRICE_ENTRY = 0.02
ENABLE_CLUSTERING = True
CLUSTER_RANGE = 40
MARKET_WEIGHT = 0.30

# Bio-Ritmos
HOURLY_MULTIPLIERS = {
    0: 0.97,  1: 0.80,  2: 0.42,  3: 0.20,  
    4: 0.39,  5: 0.48,  6: 2.11,  7: 1.41, 
    8: 1.46,  9: 1.58, 10: 0.44, 11: 0.21, 
    12: 0.35, 13: 0.49, 14: 1.72, 15: 1.71, 
    16: 1.37, 17: 2.03, 18: 1.34, 19: 1.24, 
    20: 1.01, 21: 0.89, 22: 0.82, 23: 0.61  
}
DAILY_MULTIPLIERS = {
    0: 0.90, 1: 0.75, 2: 1.25, 3: 0.95, 4: 0.95, 5: 1.15, 6: 1.10
}

API_CONFIG = {
    'base_url': "https://xtracker.polymarket.com/api",
    'gamma_url': "https://gamma-api.polymarket.com/events",
    'clob_url': "https://clob.polymarket.com/prices",
    'user': "elonmusk"
}

# ==========================================
# 1. SCANNER DE PRECIOS (V10 INTACTO)
# ==========================================
class ClobMarketScanner:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Referer": "https://polymarket.com/"
        })
        self.bulk_prices_url = "https://clob.polymarket.com/prices"

    def get_market_prices(self):
        print("   🔎 Escaneando Order Book (Modo Bulk V9)...", end=" ")
        t_start = time.time()
        try:
            params = {
                "limit": 100, "active": "true", "closed": "false",
                "archived": "false", "order": "volume24hr", "ascending": "false"
            }
            resp = self.session.get(API_CONFIG['gamma_url'], params=params, timeout=5)
            data = resp.json()
            
            market_structure = []
            tokens_to_fetch = []

            for event in data:
                title = event.get('title', '').lower()
                if "elon" not in title or "tweets" not in title: continue
                if not event.get('markets'): continue
                
                buckets_list = []
                for m in event['markets']:
                    if m.get('closed') is True: continue
                    q = m.get('question', '')
                    r_match = re.search(r'(\d+)-(\d+)', q)
                    o_match = re.search(r'(\d+)\+', q)
                    
                    min_v, max_v, b_name = 0, 99999, "Unknown"
                    if r_match:
                        min_v, max_v = int(r_match.group(1)), int(r_match.group(2))
                        b_name = f"{min_v}-{max_v}"
                    elif o_match:
                        min_v = int(o_match.group(1))
                        b_name = f"{min_v}+"
                    else: continue

                    try:
                        t_ids = json.loads(m['clobTokenIds'])
                        yes_token = t_ids[0]
                        buckets_list.append({'bucket': b_name, 'min': min_v, 'max': max_v, 'token': yes_token})
                        tokens_to_fetch.append({"token_id": yes_token, "side": "BUY"})
                        tokens_to_fetch.append({"token_id": yes_token, "side": "SELL"})
                    except: continue
                
                if buckets_list:
                    buckets_list.sort(key=lambda x: x['min'])
                    market_structure.append({'title': event['title'], 'buckets': buckets_list})

            price_map = {} 
            if tokens_to_fetch:
                try:
                    bulk_resp = self.session.post(self.bulk_prices_url, json=tokens_to_fetch, timeout=5)
                    bulk_data = bulk_resp.json()
                    for token_id, prices in bulk_data.items():
                        price_map[token_id] = {
                            "buy": float(prices.get("BUY", 0) or 0),
                            "sell": float(prices.get("SELL", 0) or 0)
                        }
                except Exception: pass

            final_data = []
            for mkt in market_structure:
                clean_buckets = []
                for b in mkt['buckets']:
                    precios = price_map.get(b['token'], {})
                    clean_buckets.append({
                        'bucket': b['bucket'], 'min': b['min'], 'max': b['max'],
                        'ask': precios.get('sell', 0.0), 'bid': precios.get('buy', 0.0)
                    })
                final_data.append({'title': mkt['title'], 'buckets': clean_buckets})

            elapsed = time.time() - t_start
            print(f"✅ ({elapsed:.2f}s)")
            return final_data
        except Exception as e:
            print(f"❌ Error Scanner: {e}")
            return []

# ==========================================
# 2. CEREBRO MATEMÁTICO (V10 INTACTO - CARGA CSV)
# ==========================================
class HawkesBrain:
    def __init__(self):
        self.params = {'mu': 0.4, 'alpha': 3.0, 'beta': 4.0} 
        self.timestamps = [] 
        self.history_df = None
        self._ensure_directories()
        self.load_and_train() # V10: Carga CSVs

    def _ensure_directories(self):
        if not os.path.exists(DAILY_METRICS_THREE_WEEKS_DIR): os.makedirs(DAILY_METRICS_THREE_WEEKS_DIR)

    def load_and_train(self):
        print("🧠 Cargando y limpiando datos (V10 Logic)...")
        all_timestamps = []
        csv_files = glob.glob(os.path.join(DAILY_METRICS_THREE_WEEKS_DIR, "*.csv"))
        df_list = []
        
        for f in csv_files:
            if "dataset" in f or "trade" in f: continue
            try:
                df_temp = pd.read_csv(f)
                df_temp.columns = [c.strip() for c in df_temp.columns]
                if 'Date/Time' in df_temp.columns:
                    df_temp['Date_Clean'] = pd.to_datetime(df_temp['Date/Time'], errors='coerce')
                    df_temp = df_temp.dropna(subset=['Date_Clean'])
                    if not df_temp.empty: df_list.append(df_temp)
            except: pass

        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            clean_df = full_df.drop_duplicates(subset='Date_Clean', keep='first')
            self.history_df = clean_df.copy()
            
            np.random.seed(42)
            for _, row in clean_df.iterrows():
                posts_count = int(row['Posts'])
                if posts_count > 0:
                    ts = row['Date_Clean'].timestamp()
                    all_timestamps.extend(ts + np.random.uniform(0, 3600, posts_count))
        
        # Mezcla con Live Log
        live_path = os.path.join(LOGS_DIR, LIVE_LOG)
        if os.path.exists(live_path):
            try:
                with open(live_path, 'r') as f:
                    live_events = json.load(f)
                    now_ms = time.time() * 1000
                    live_events = [e for e in live_events if (now_ms - e['timestamp']) < 86400000]
                    all_timestamps.extend([e['timestamp']/1000.0 for e in live_events])
            except: pass

        if all_timestamps:
            self.timestamps = np.array(sorted(all_timestamps))
            self._optimize_params()

    def _optimize_params(self):
        if len(self.timestamps) < 50: return
        ts_h = (self.timestamps - self.timestamps[0]) / 3600.0
        T_max = ts_h[-1]
        def nll(p):
            mu, a, b = p
            if mu <= 0.001 or a <= 0.001 or b <= 0.001: return 1e10
            if a >= b: return 1e10
            n = len(ts_h)
            log_mu = -np.log(mu); term_sum = np.log(mu); A_prev = 0
            for i in range(1, n):
                dt = ts_h[i] - ts_h[i-1]
                A_curr = np.exp(-b * dt) * (A_prev + 1)
                lam = mu + a * A_curr
                term_sum += np.log(lam); A_prev = A_curr
            term_integral = np.sum((a/b) * (1 - np.exp(-b * (T_max - ts_h))))
            return -(term_sum - (mu * T_max + term_integral))

        try:
            res = minimize(nll, [self.params['mu'], self.params['alpha'], self.params['beta']], 
                           method='L-BFGS-B', bounds=[(1e-4, None)]*3)
            if res.success: self.params = dict(zip(['mu','alpha','beta'], res.x))
        except: pass

    def predict(self, history_ms, hours):
        mu, a, b = self.params.values()
        boost = 0
        if history_ms:
            current_time_sec = time.time()
            cutoff = 10.0 / b
            for t_ms in reversed(history_ms):
                t_sec = t_ms / 1000.0
                dt = current_time_sec - t_sec
                if dt > cutoff * 3600: break
                if dt < 0: dt = 0
                boost += a * np.exp(-b * (dt / 3600.0))
        sims = []
        for _ in range(1000):
            t, l_boost, ev = 0, boost, 0
            while t < hours:
                l_max = mu + l_boost
                if l_max <= 0: l_max = 0.001
                w = -np.log(np.random.uniform()) / l_max
                t += w
                if t >= hours: break
                l_boost *= np.exp(-b * w)
                if np.random.uniform() < (mu + l_boost)/l_max:
                    ev += 1; l_boost += a
            sims.append(ev)
        return sims

# ==========================================
# 3. SENSOR DE TWEETS (V10 INTACTO + OPTIMIZACION FECHA)
# ==========================================
class PolymarketSensor:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Mozilla/5.0"})

    def _fetch_tracking_detail(self, t, now):
        try:
            # 1. Petición V10
            response = self.s.get(f"{API_CONFIG['base_url']}/trackings/{t['id']}?includeStats=true", timeout=5).json()
            d = response.get('data', {})
            end_date_str = d.get('endDate') or t.get('endDate')
            hours = 0.0

            if end_date_str:
                try:
                    original_dt = dateutil.parser.isoparse(end_date_str)
                    # Martillazo Horario (Lo que te gustaba de la V10)
                    fixed_end_date = original_dt.replace(hour=17, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
                    hours = (fixed_end_date - now).total_seconds() / 3600.0
                except: pass

            count = d.get('stats', {}).get('total', 0)
            print(count)
            days_elapsed = d.get('stats', {}).get('daysElapsed', 0)
            daily_avg = 0.0
            if days_elapsed > 0: daily_avg = count / days_elapsed

            if hours > -2.0:
                return {
                    'id': t['id'], 'title': t['title'], 'count': count, 'hours': hours,
                    'daily_avg': daily_avg, 'active': True 
                }
        except: pass
        return None

    def get_active_counts(self):
        try:
            r = self.s.get(f"{API_CONFIG['base_url']}/users/{API_CONFIG['user']}", timeout=5).json()
            trackings = r.get('data', {}).get('trackings', [])
            res = []
            now = datetime.now(timezone.utc)
            
            # --- PEQUEÑA OPTIMIZACIÓN (NO CAMBIA LÓGICA, SOLO VELOCIDAD) ---
            candidates = []
            for t in trackings:
                start_str = t.get('startDate'); end_str = t.get('endDate')
                if start_str and end_str:
                    try:
                        end = dateutil.parser.isoparse(end_str)
                        if now <= (end + pd.Timedelta(hours=12)):
                            candidates.append(t)
                            continue
                    except: pass
                if t.get('isActive'): candidates.append(t)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self._fetch_tracking_detail, t, now): t for t in candidates}
                for f in as_completed(futures):
                    result = f.result()
                    if result: res.append(result)
            return res
        except: return []

# ==========================================
# 4. PAPER TRADER (V10 INTACTO)
# ==========================================
class PaperTrader:
    def __init__(self, initial_cash=1000.0):
        self.file_path = os.path.join(LOGS_DIR, PORTFOLIO_PAPER_TRADER)
        self.log_path = os.path.join(LOGS_DIR, TRADE_LOG)
        self.risk_pct_normal = 0.04
        self.risk_pct_lotto = 0.01
        self.min_bet = 5.0
        self.portfolio = self._load()
        self._ensure_log_header()
        if not self.portfolio: self.portfolio = {"cash": initial_cash, "positions": {}, "history": []}

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f: return json.load(f)
            except: return None
        return None

    def _save(self):
        with open(self.file_path, 'w') as f: json.dump(self.portfolio, f, indent=2)

    def _ensure_log_header(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding='utf-8') as f:
                f.write("Timestamp,Action,Market,Bucket,Price,Shares,Reason,PnL,Cash_After\n")

    def _clean_market_name(self, full_title):
        month_map = {"january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr", "may": "May", "june": "Jun", "july": "Jul", "august": "Aug", "september": "Sep", "october": "Oct", "november": "Nov", "december": "Dec"}
        pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d+)'
        matches = re.findall(pattern, full_title, re.IGNORECASE)
        if len(matches) >= 2:
            m1, d1 = matches[0]; m2, d2 = matches[1]
            return f"{month_map.get(m1.lower(), m1[:3])} {d1} - {month_map.get(m2.lower(), m2[:3])} {d2}"
        return "Evento Global"

    def _log_trade(self, action, market, bucket, price, shares, reason, pnl=0.0):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_clean = market.replace(",", "")
        reason_clean = reason.replace(",", ".")
        row = f"{ts},{action},{market_clean},{bucket},{price:.3f},{shares:.1f},{reason_clean},{pnl:.2f},{self.portfolio['cash']:.2f}\n"
        with open(self.log_path, "a", encoding='utf-8') as f: f.write(row)

    # Helper V11 para saber qué buckets tenemos
    def get_owned_buckets_val(self, market_title):
        vals = []
        for k, v in self.portfolio['positions'].items():
            if self._clean_market_name(v['market']) == self._clean_market_name(market_title):
                try:
                    if "+" in v['bucket']: mid = int(re.search(r'\d+', v['bucket']).group()) + 20
                    else: 
                        nums = [int(n) for n in re.findall(r'\d+', v['bucket'])]
                        mid = sum(nums)/2
                    vals.append(mid)
                except: pass
        return vals

    def execute(self, market_title, bucket, signal, price, reason="Manual"):
        pos_id = f"{market_title}|{bucket}"
        
        # BUY
        if "BUY" in signal or "FISH" in signal:
            if pos_id not in self.portfolio["positions"]:
                pct = self.risk_pct_lotto if "FISH" in signal else self.risk_pct_normal
                bet_amount = max(self.portfolio["cash"] * pct, self.min_bet)
                if self.portfolio["cash"] >= bet_amount:
                    shares = bet_amount / price
                    self.portfolio["cash"] -= bet_amount
                    self.portfolio["positions"][pos_id] = {
                        "shares": shares, "entry_price": price, "market": market_title,
                        "bucket": bucket, "timestamp": time.time(), "invested": bet_amount
                    }
                    self._save()
                    self._log_trade(signal, market_title, bucket, price, shares, reason)
                    return f"✅ BUY: ${bet_amount:.2f}"
        
        # SELL (Incluye TAKE PROFIT y ROTATE)
        elif "SELL" in signal or "DUMP" in signal or "ROTATE" in signal or "TAKE_PROFIT" in signal:
            if pos_id in self.portfolio["positions"]:
                pos = self.portfolio["positions"].pop(pos_id)
                revenue = pos["shares"] * price
                profit = revenue - pos.get("invested", pos["shares"] * pos["entry_price"])
                self.portfolio["cash"] += revenue
                self.portfolio["history"].append({"market": market_title, "profit": profit})
                self._save()
                self._log_trade(signal, market_title, bucket, price, pos['shares'], reason, pnl=profit)
                return f"💰 SELL: P&L ${profit:.2f}"
        return None

    def print_summary(self, current_prices_data):
        cash = self.portfolio["cash"]; invested = 0.0
        print("\n💼 --- PORTFOLIO ---")
        print(f"   🔹 {'FECHAS EVENTO':<20} | {'BUCKET':<10} | {'PRECIO':<8} | {'ACTUAL':<8} | {'P&L ($)':<8}")
        print("   " + "-"*85)
        for pid, pos in self.portfolio["positions"].items():
            curr_p = pos['entry_price']
            lbl = self._clean_market_name(pos.get('market', ''))
            for m in current_prices_data:
                if self._clean_market_name(m['title']) == lbl:
                    for b in m['buckets']:
                        if str(b['bucket']) == str(pos['bucket']): curr_p = b.get('bid', 0)
            val = pos['shares'] * curr_p
            pnl = val - (pos['shares'] * pos['entry_price'])
            invested += val
            print(f"   🔹 {lbl:<20} | {pos['bucket']:<10} | ${pos['entry_price']:.3f}  | ${curr_p:.3f}  | {pnl:+6.2f}")
        print("   " + "-"*85)
        print(f"   💵 Cash: ${cash:.2f} | 📈 Equity: ${cash+invested:.2f}")

# ==========================================
# 5. SENSOR DE PÁNICO (V10 INTACTO)
# ==========================================
class MarketPanicSensor:
    def __init__(self, sensitivity=1.5):
        self.history = {}; self.sensitivity = sensitivity; self.window_size = 5
    def analyze(self, market_data):
        alerts = []
        for m in market_data:
            for b in m['buckets']:
                key = f"{m['title']}|{b['bucket']}"
                if key not in self.history: self.history[key] = {'asks': deque(maxlen=5), 'bids': deque(maxlen=5)}
                h = self.history[key]
                h['asks'].append(b['ask']); h['bids'].append(b['bid'])
                if len(h['asks']) >= 3:
                    avg_a = sum(h['asks'])/len(h['asks']); avg_b = sum(h['bids'])/len(h['bids'])
                    if avg_a > 0.01 and b['ask'] > (avg_a * self.sensitivity):
                        alerts.append({'type': 'PUMP', 'market_title': m['title'], 'bucket': b['bucket'], 'price': b['ask'], 'min': b.get('min',0)})
                    if avg_b > 0.05 and b['bid'] < (avg_b / self.sensitivity):
                        alerts.append({'type': 'DUMP', 'market_title': m['title'], 'bucket': b['bucket'], 'price': b['bid'], 'min': b.get('min',0)})
        return alerts

# ==========================================
# 6. DIRECTOR (V12.15 - ACTIVE BUCKET SAFETY)
# ==========================================
def run():
    print("\n🤖 ELON-BOT V12.15 (ACTIVE BUCKET SAFETY MODE)")
    
    # Utiles V10
    def log_monitor(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(LOGS_DIR, MONITOR_LOG), "a") as f: f.write(f"[{ts}] {msg}\n")

    def save_market_tape(clob_data, markets_meta):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(MARKET_TAPE_DIR, f"tape_{ts}.json"), "w") as f:
            json.dump({"timestamp": time.time(), "meta": markets_meta, "order_book": clob_data}, f)

    def save_trade_snapshot(action, m_title, bucket, price, reason, ctx):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"snap_{action}_{ts}.json"
        with open(os.path.join(SNAPSHOTS_DIR, fname), "w") as f:
            json.dump({"action": action, "market": m_title, "bucket": bucket, "price": price, "reason": reason, "context": ctx}, f, indent=2)

    def titles_match_paranoid(t1, t2):
        t1 = t1.lower(); t2 = t2.lower()
        if t1 in t2 or t2 in t1: return True
        def get_nums(txt): return {n for n in re.findall(r'\d+', txt) if n not in ['2024', '2025', '2026']}
        return len(get_nums(t1).intersection(get_nums(t2))) >= 2

    def get_bio_multiplier():
        n = datetime.now(); return HOURLY_MULTIPLIERS.get(n.hour, 1.0) * DAILY_MULTIPLIERS.get(n.weekday(), 1.0)

    # Instanciamos clases V10
    brain = HawkesBrain()
    sensor = PolymarketSensor()
    pricer = ClobMarketScanner()
    trader = PaperTrader()
    panic_sensor = MarketPanicSensor()
    
    last_counts = {}
    last_tape = 0
    
    global_events = []
    if os.path.exists(os.path.join(LOGS_DIR, LIVE_LOG)):
        try:
            with open(os.path.join(LOGS_DIR, LIVE_LOG)) as f: 
                d = json.load(f)
                global_events = [e for e in d if (time.time()*1000 - e['timestamp']) < 86400000]
        except: pass

    while True:
        try:
            # 1. Obtener Mercados
            markets = sensor.get_active_counts()
            if not markets:
                print("💤 Waiting for data...", end="\r"); time.sleep(3); continue

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
            
            global_events = [e for e in global_events if (ts_now - e['timestamp']) < 86400000]
            ts_list = [e['timestamp'] for e in global_events]
            IS_WARMUP = len(ts_list) < 5

            # 3. Precios y Analisis
            clob_data = pricer.get_market_prices()
            if clob_data:
                # Tape
                if time.time() - last_tape > 1800:
                    save_market_tape(clob_data, markets); last_tape = time.time()
                
                # Panic
                alerts = panic_sensor.analyze(clob_data)
                for a in alerts:
                    print(f"⚠️ PÁNICO V10: {a['type']} en {a['bucket']} (Price: {a['price']})")
                
                bio_mult = get_bio_multiplier()
                
                for m_poly in markets:
                    m_clob = next((c for c in clob_data if titles_match_paranoid(m_poly['title'], c['title'])), None)
                    if not m_clob: continue

                    # PREDICCIÓN
                    base_sims = brain.predict(ts_list, m_poly['hours'])
                    pred_mean_hawkes = m_poly['count'] + np.mean(base_sims) * bio_mult
                    
                    market_buckets = [b for b in m_clob['buckets'] if b.get('bid',0) > 0.01]
                    consensus = 0
                    if market_buckets:
                        consensus = sum([(b['min']+b['max'])/2 * b['bid'] for b in market_buckets]) / sum([b['bid'] for b in market_buckets])
                    
                    final_mean = pred_mean_hawkes
                    if consensus > 0:
                        final_mean = (pred_mean_hawkes * (1-MARKET_WEIGHT)) + (consensus * MARKET_WEIGHT)
                    
                    # FIX 1: SIGMA DINÁMICA
                    if m_poly['hours'] < 2.0: std_floor = 0.5 
                    else: std_floor = 5.0
                    eff_std = max(np.std(base_sims), std_floor)

                    d_avg = m_poly.get('daily_avg', 0)
                    hours_left = m_poly.get('hours', 0)
                    days = int(hours_left // 24)
                    rem_hours = hours_left % 24
                    time_str = f"{days}d {rem_hours:.1f}h" if days > 0 else f"{hours_left:.1f}h"

                    print("-" * 65)
                    print(f">>> {m_poly['title']}")
                    print(f"    Tweets: {m_poly['count']} (Avg: {d_avg:.1f}/d) | ⏳ Quedan: {time_str} | 🧠 Híbrido: {final_mean:.1f} (σ={eff_std:.1f})")
                    print("-" * 65)
                    print(f"{'BUCKET':<10} | {'BID':<6} | {'ASK':<6} | {'FAIR':<6} | {'Z-SCR':<6} | {'ACTION'}")

                    my_buckets = trader.get_owned_buckets_val(m_poly['title'])

                    for b in m_clob['buckets']:
                        if b['max'] < m_poly['count']: continue 
                        
                        bid, ask = b.get('bid',0), b.get('ask',0)
                        
                        if b['max'] >= 99999: mid = b['min'] + 20
                        else: mid = (b['min'] + b['max']) / 2
                        
                        z_score = abs(mid - final_mean) / eff_std
                        p_min = norm.cdf(b['min'], final_mean, eff_std)
                        
                        if b['max'] >= 99999: fair = 1.0 - p_min
                        else: fair = norm.cdf(b['max']+1, final_mean, eff_std) - p_min

                        action = "-"
                        reason = ""
                        
                        # --- MOTOR ---
                        owned = any([x for x in trader.portfolio['positions'].values() 
                                     if x['bucket'] == b['bucket'] and titles_match_paranoid(x['market'], m_poly['title'])])
                        
                        if owned:
                            pos_data = next((v for k,v in trader.portfolio['positions'].items() if v['bucket'] == b['bucket']), None)
                            if pos_data:
                                entry = pos_data['entry_price']
                                profit_pct = (bid - entry) / entry if entry > 0 else 0
                                
                                # --- ESTRATEGIA DEFENSIVA V12.15 (SAFE ZONE) ---
                                
                                should_sell = False
                                sell_reason = ""
                                
                                # Calcular cuántos tweets quedan de margen
                                bucket_headroom = b['max'] - m_poly['count']
                                
                                # Detectar si estamos en el BUCKET ACTIVO (Zona de Muerte)
                                is_active_bucket = (m_poly['count'] >= b['min'])
                                
                                # Definir margen de seguridad dinámico
                                # Si es el bucket activo, queremos 12 tweets de margen.
                                # Si es futuro, 5 tweets.
                                safety_threshold = 12 if is_active_bucket else 5

                                # 1. REGLA DE PROXIMIDAD DINÁMICA
                                # Si nos acercamos al techo y vamos ganando -> VENDER
                                if bucket_headroom < safety_threshold and bucket_headroom >= 0 and profit_pct > 0:
                                    should_sell = True; sell_reason = f"Proximity Danger ({bucket_headroom} left)"

                                # 2. TESORO PARANOICO
                                # Ajusta el umbral de ganancia (profit_pct) y la tolerancia (z_score)
                                # OPCIÓN CONSERVADORA (La actual): > 1.0 (100%)
                                # OPCIÓN EQUILIBRADA (Recomendada): > 1.5 (150%)
                                # OPCIÓN AGRESIVA: > 2.0 (200%)
                                # Y recuerda el Z-Score: 0.9 es el equilibrio.
                                elif profit_pct > 1.5 and z_score > 0.9:  # <--- Aquí he puesto 1.5 (150%)
                                    should_sell = True; sell_reason = "Paranoid Treasure (>150%)"

                                # 3. PROTECCIÓN DE BENEFICIO NORMAL (>0% Ganancia)
                                elif profit_pct > 0.0 and z_score > 1.3:
                                    should_sell = True; sell_reason = "Protect Profit"

                                # 4. STOP LOSS (< -15%)
                                elif profit_pct < -0.15 and z_score > 1.3:
                                    should_sell = True; sell_reason = "Stop Loss"
                                
                                # 5. PÁNICO GLOBAL
                                elif z_score > 2.0:
                                    should_sell = True; sell_reason = "Panic Exit"

                                if should_sell:
                                    action = "SMART_ROTATE"; reason = f"{sell_reason} Z{z_score:.1f}"
                                    res = trader.execute(m_poly['title'], b['bucket'], "ROTATE", bid, reason)
                                    if res: save_trade_snapshot("SMART_ROTATE", m_poly['title'], b['bucket'], bid, reason, {"z": z_score, "pnl": profit_pct})

                        elif not owned and not IS_WARMUP:
                            # --- FIX 2: REALITY CHECK ---
                            is_impossible = False
                            if m_poly['hours'] < 1.0:
                                tweets_needed = b['min'] - m_poly['count']
                                if tweets_needed > (m_poly['hours'] * 15): is_impossible = True

                            if not is_impossible:
                                # === FIX CRÍTICO V12.16 (ANTI-CHURN) ===
                                # Antes: if z_score <= MAX_Z_SCORE_ENTRY
                                # Ahora: if z_score <= 0.8
                                # Explicación: Si nuestro Stop Loss salta en 1.3, NO PODEMOS comprar en 1.5.
                                # Ponemos el techo de compra en 0.8 para dejar un margen de seguridad.
                                
                                if z_score <= 0.8 and ask >= MIN_PRICE_ENTRY:
                                    is_neighbor = True
                                    if ENABLE_CLUSTERING and my_buckets:
                                        is_neighbor = any(abs(mid - ov) <= CLUSTER_RANGE for ov in my_buckets)
                                    
                                    edge = fair - ask
                                    if is_neighbor and edge > 0.05:
                                        action = "BUY"; reason = f"Val+{edge:.2f}"
                                        res = trader.execute(m_poly['title'], b['bucket'], "BUY", ask, reason)
                                        if res: save_trade_snapshot("BUY", m_poly['title'], b['bucket'], ask, reason, {"z": z_score, "fair": fair})

                        # PRINT
                        color_act = f"🟢 {action}" if "BUY" in action else (f"🔴 {action}" if "ROTATE" in action or "PROFIT" in action else "-")
                        bucket_display = f"*{b['bucket']}" if owned else f"{b['bucket']}"
                        print(f"{bucket_display:<10} | {bid:.3f}  | {ask:.3f}  | {fair:.3f}  | {z_score:.1f}   | {color_act} {reason}")

            # ==============================================================================
            # 💀 FIX REALIDAD V3: COMPARACIÓN NUCLEAR (Solo letras y números)
            # ==============================================================================
            for symbol, pos in trader.portfolio['positions'].items():
                # Función auxiliar para "desnudar" el string de todo lo que no sea texto
                def normalize(s): 
                    return ''.join(filter(str.isalnum, str(s).lower()))
                
                pos_clean = normalize(pos['market'])
                
                m_curr = None
                for m in markets:
                    m_clean = normalize(m['title'])
                    
                    # Comparamos las cadenas "desnudas"
                    if pos_clean in m_clean or m_clean in pos_clean:
                        m_curr = m
                        break
                
                if m_curr:
                    try:
                        bucket_str = pos['bucket']
                        if "+" in bucket_str: continue 
                        
                        max_val = int(bucket_str.split('-')[1])
                        
                        # VEREDICTO FINAL
                        if m_curr['count'] > max_val:
                            pos['current_price'] = 0.0
                            pos['market_value'] = 0.0 # Forzamos valor 0
                    except:
                        continue
            # ==============================================================================

            trader.print_summary(clob_data)
            time.sleep(8) 

        except KeyboardInterrupt: break
        except Exception as e: 
            print(f"Loop Error: {e}"); time.sleep(5)

if __name__ == "__main__":
    run()