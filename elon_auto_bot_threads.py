import time
import json
import os
import requests
import numpy as np
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import dateutil.parser
from scipy.stats import norm

# ==============================================================================
# CONFIGURACIÓN (V11.7 - BACK TO BASICS)
# ==============================================================================
LOGS_DIR = 'logs'
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)

FILES = {
    'portfolio': os.path.join(LOGS_DIR, "portfolio.json"),
    'history': os.path.join(LOGS_DIR, "live_history.json"),
    'trades': os.path.join(LOGS_DIR, "trade_history.csv"),
    'snapshots': os.path.join(LOGS_DIR, "snapshots_merged.json"),
    'market_tape': os.path.join(LOGS_DIR, "market_tape_merged.json")
}

API_CONFIG = {
    'base_url': "https://xtracker.polymarket.com/api",
    'gamma_url': "https://gamma-api.polymarket.com/events",
    'clob_url': "https://clob.polymarket.com/prices",
    'user': "elonmusk"
}

# PARAMETROS ESTRATEGIA
MAX_Z_SCORE_ENTRY = 1.6
MIN_PRICE_ENTRY = 0.02
ENABLE_CLUSTERING = True
CLUSTER_RANGE = 40
MARKET_WEIGHT = 0.30

# ==============================================================================
# 🛠️ UTILIDADES
# ==============================================================================
def append_to_json_file(filename, new_record):
    data_list = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                content = f.read().strip()
                if content:
                    data_list = json.loads(content)
                    if not isinstance(data_list, list): data_list = [data_list]
        except: pass
    data_list.append(new_record)
    temp = filename + ".tmp"
    try:
        with open(temp, 'w') as f: json.dump(data_list, f, indent=2)
        os.replace(temp, filename)
    except Exception: pass

def titles_match_paranoid(tracker_title, market_title):
    t1 = tracker_title.lower(); t2 = market_title.lower()
    if t1 in t2 or t2 in t1: return True
    def get_nums(txt): return {n for n in re.findall(r'\d+', txt) if n not in ['2024', '2025', '2026']}
    return len(get_nums(t1).intersection(get_nums(t2))) >= 2

# ==============================================================================
# 🧠 CEREBRO V11
# ==============================================================================
class HawkesBrain:
    def __init__(self):
        self.params = {'mu': 0.4, 'alpha': 3.0, 'beta': 4.0} 

    def get_market_consensus(self, m_poly, clob_buckets):
        sum_prod = 0; sum_w = 0
        for b in clob_buckets:
            try:
                if "+" in b['bucket']: mid = int(re.search(r'\d+', b['bucket']).group()) + 20
                else:
                    nums = [int(n) for n in re.findall(r'\d+', b['bucket'])]
                    if len(nums) == 2: mid = sum(nums) / 2
                    else: continue
                price = (b['bid'] + b['ask']) / 2
                if price > 0.01:
                    sum_prod += mid * price; sum_w += price
            except: continue
        return (sum_prod / sum_w) if sum_w > 0 else None

    def predict(self, history_ts, hours_left):
        mu, a, b = self.params.values()
        boost = 0
        if history_ts:
            last_ts = history_ts[-1]; now = time.time()
            cutoff = 10.0 / b
            for t_ts in reversed(history_ts):
                age_sec = now - t_ts
                if age_sec > cutoff * 3600: break
                if age_sec < 0: age_sec = 0
                boost += a * np.exp(-b * (age_sec / 3600.0))
        
        sims = []
        for _ in range(500):
            t, l_boost, ev = 0, boost, 0
            while t < hours_left:
                l_max = mu + l_boost
                if l_max <= 0: l_max = 0.001
                w = -np.log(np.random.uniform()) / l_max
                t += w
                if t >= hours_left: break
                l_boost *= np.exp(-b * w)
                if np.random.uniform() < (mu + l_boost)/l_max:
                    ev += 1; l_boost += a
            sims.append(ev)
        return np.mean(sims), np.std(sims)

# ==============================================================================
# 📡 SENSOR V10 ORIGINAL (EL QUE FUNCIONABA)
# ==============================================================================
class PolymarketSensor:
    def __init__(self):
        self.s = requests.Session()
        # Header simple V10
        self.s.headers.update({"User-Agent": "Mozilla/5.0"})

    def _fetch_tracking_detail(self, t, now):
        try:
            # 1. Petición segura a la API
            url = f"{API_CONFIG['base_url']}/trackings/{t['id']}?includeStats=true"
            # Timeout simple de 5s como en V10
            response = self.s.get(url, timeout=5).json()
            d = response.get('data', {})
            
            end_date_str = d.get('endDate') or t.get('endDate')
            hours = 0.0

            if end_date_str:
                try:
                    # A. Parseamos la fecha original
                    original_dt = dateutil.parser.isoparse(end_date_str)
                    
                    # B. 🔨 MARTILLAZO HORARIO: FORZAMOS LAS 17:00 UTC
                    fixed_end_date = original_dt.replace(
                        hour=17, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                    )
                    
                    # C. Calculamos las horas restantes
                    hours = (fixed_end_date - now).total_seconds() / 3600.0
                    
                except Exception:
                    pass

            # 2. Obtención del Conteo
            count = d.get('stats', {}).get('total', 0)
            days_elapsed = d.get('stats', {}).get('daysElapsed', 0)

            # 3. FILTRO DE VISIBILIDAD GENÉRICO
            if hours > -2.0:
                return {
                    'id': t['id'], 
                    'title': t['title'], 
                    'count': count, 
                    'hours': hours,
                    'daily_avg': days_elapsed > 0 and count/days_elapsed or 0,
                    'active': True 
                }

        except Exception:
            pass
        return None 

    def get_active_counts(self):
        url = f"{API_CONFIG['base_url']}/users/{API_CONFIG['user']}"
        try:
            r = self.s.get(url, timeout=5)
            # Si falla, devolvemos vacio sin bloquear ni imprimir errores raros
            if r.status_code != 200: return []
            
            trackings = r.json().get('data', {}).get('trackings', [])
            res = []
            now = datetime.now(timezone.utc)
            
            # Usamos ThreadPool como en V10
            with ThreadPoolExecutor(max_workers=5) as ex:
                futures = [ex.submit(self._fetch_tracking_detail, t, now) for t in trackings]
                for f in as_completed(futures):
                    if f.result(): res.append(f.result())
            return res
        except Exception:
            return []

# ==============================================================================
# 🔎 CLOB SCANNER
# ==============================================================================
class ClobMarketScanner:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_market_prices(self):
        try:
            params = {"limit": 100, "active": "true", "closed": "false", "archived": "false", "order": "volume24hr", "ascending": "false"}
            r = self.s.get(API_CONFIG['gamma_url'], params=params, timeout=5)
            if r.status_code != 200: return []
            data = r.json()
            
            structure = []; tokens = []
            for e in data:
                if "elon" not in e.get('title','').lower() or "tweets" not in e.get('title','').lower(): continue
                buckets = []
                for m in e.get('markets', []):
                    q = m.get('question', '')
                    r_match = re.search(r'(\d+)-(\d+)', q); o_match = re.search(r'(\d+)\+', q)
                    if r_match: b_name, min_v, max_v = f"{r_match.group(1)}-{r_match.group(2)}", int(r_match.group(1)), int(r_match.group(2))
                    elif o_match: b_name, min_v, max_v = f"{o_match.group(1)}+", int(o_match.group(1)), 99999
                    else: continue
                    try:
                        tid = json.loads(m['clobTokenIds'])[0]
                        buckets.append({'bucket': b_name, 'min': min_v, 'max': max_v, 'token': tid})
                        tokens.append({"token_id": tid, "side": "BUY"}); tokens.append({"token_id": tid, "side": "SELL"})
                    except: continue
                if buckets:
                    buckets.sort(key=lambda x: x['min'])
                    structure.append({'title': e['title'], 'buckets': buckets})

            price_map = {}
            if tokens:
                bulk = self.s.post(API_CONFIG['clob_url'], json=tokens, timeout=5).json()
                for tid, p in bulk.items():
                    price_map[tid] = {'bid': float(p.get('BUY', 0) or 0), 'ask': float(p.get('SELL', 0) or 0)}

            final = []
            for s in structure:
                clean_b = []
                for b in s['buckets']:
                    p = price_map.get(b['token'], {'bid':0, 'ask':0})
                    clean_b.append({**b, **p})
                final.append({'title': s['title'], 'buckets': clean_b})
            return final
        except Exception: return []

# ==============================================================================
# 💼 PAPER TRADER
# ==============================================================================
class PaperTrader:
    def __init__(self):
        self.portfolio = {'cash': 1000.0, 'positions': {}}
        self.load()
    
    def load(self):
        if os.path.exists(FILES['portfolio']):
            try:
                with open(FILES['portfolio'], 'r') as f: self.portfolio = json.load(f)
            except: pass

    def save(self):
        with open(FILES['portfolio'], 'w') as f: json.dump(self.portfolio, f, indent=4)

    def get_owned_buckets_val(self, market_title):
        vals = []
        for k, v in self.portfolio['positions'].items():
            if v['market'] == market_title:
                try:
                    if "+" in v['bucket']: mid = int(re.search(r'\d+', v['bucket']).group()) + 20
                    else: 
                        nums = [int(n) for n in re.findall(r'\d+', v['bucket'])]
                        mid = sum(nums)/2
                    vals.append(mid)
                except: pass
        return vals

    def execute(self, action, market, bucket, price, shares, reason, context):
        pos_key = f"{market} | {bucket}"
        if action == "BUY":
            cost = shares * price
            if self.portfolio['cash'] < cost: return None
            self.portfolio['cash'] -= cost
            self.portfolio['positions'][pos_key] = {
                'market': market, 'bucket': bucket, 'shares': shares, 'entry_price': price
            }
        elif action in ["SELL", "ROTATE", "TAKE_PROFIT"]:
            if pos_key not in self.portfolio['positions']: return None
            pos = self.portfolio['positions'][pos_key]
            revenue = pos['shares'] * price
            self.portfolio['cash'] += revenue
            del self.portfolio['positions'][pos_key]
            pnl = revenue - (pos['shares'] * pos['entry_price'])
            self._log_csv(action, market, bucket, price, pos['shares'], reason, pnl)
        
        if action == "BUY": self._log_csv(action, market, bucket, price, shares, reason, 0)
        self.save()
        snap = {'timestamp': time.time(), 'readable_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': action, 'market': market, 'bucket': bucket, 'price': price, 'reason': reason, 'context': context}
        append_to_json_file(FILES['snapshots'], snap)
        return f"{action} OK"

    def _log_csv(self, action, market, bucket, price, shares, reason, pnl):
        header = not os.path.exists(FILES['trades'])
        with open(FILES['trades'], 'a') as f:
            if header: f.write("Timestamp,Action,Market,Bucket,Price,Shares,Reason,PnL,Cash_After\n")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts},{action},{market},{bucket},{price:.3f},{shares:.1f},{reason},{pnl:.2f},{self.portfolio['cash']:.2f}\n")

    def print_market_summary(self, market_title, stats_ctx, buckets_data, decisions):
        print("-" * 75)
        print(f">>> {market_title}")
        print(f"    📊 Act: {stats_ctx['count']} | 🧠 Gauss: μ={stats_ctx['mean']:.1f} σ={stats_ctx['std']:.1f} | ⏳ Quedan: {stats_ctx['hours']:.1f}h")
        print("-" * 75)
        print(f"{'BUCKET':<10} | {'BID':<6} | {'ASK':<6} | {'FAIR':<6} | {'Z-SCR':<5} | {'ACCIÓN':<10} | {'MOTIVO'}")
        for b in buckets_data:
            act = "-"; reason = "_"
            for d in decisions:
                if d['bucket'] == b['bucket']:
                    act = d['action']; reason = d['reason']
                    if "BUY" in act: act = f"🟢 {act}"
                    if "SELL" in act or "ROTATE" in act: act = f"🔴 {act}"
            try:
                if "+" in b['bucket']: mid = int(re.search(r'\d+', b['bucket']).group()) + 20
                else: 
                    nums = [int(n) for n in re.findall(r'\d+', b['bucket'])]
                    mid = sum(nums)/2
                z = abs(mid - stats_ctx['mean']) / stats_ctx['std']
            except: z = 0.0
            fair = b.get('fair', 0.0)
            print(f"{b['bucket']:<10} | {b['bid']:.3f}  | {b['ask']:.3f}  | {fair:.3f}  | {z:.1f}   | {act:<10} | {reason}")
        print("-" * 75)

    def print_portfolio_summary(self):
        print("\n💼 --- PORTFOLIO ---")
        print(f"   {'FECHAS EVENTO':<30} | {'BUCKET':<10} | {'ENTRADA':<8} | {'SHARES':<8}")
        print("   " + "-"*65)
        total_equity = self.portfolio['cash']
        for k, v in self.portfolio['positions'].items():
            short_m = v['market'].replace("Elon Musk # tweets ", "")[:30]
            print(f"   🔹 {short_m:<30} | {v['bucket']:<10} | ${v['entry_price']:.3f}   | {v['shares']:.1f}")
            total_equity += (v['shares'] * v['entry_price'])
        print("   " + "-"*65)
        print(f"   💵 Cash: ${self.portfolio['cash']:.2f} | 📈 Equity Est: ${total_equity:.2f}")
        print("-" * 75)

# ==============================================================================
# 🚀 EJECUCIÓN PRINCIPAL
# ==============================================================================
def run():
    print("🤖 ELON BOT V11.7 [V10 SENSOR + V11 BRAIN]")
    
    brain = HawkesBrain()
    sensor = PolymarketSensor()
    pricer = ClobMarketScanner()
    trader = PaperTrader()
    
    last_tweets = []
    if os.path.exists(FILES['history']):
        try:
            with open(FILES['history']) as f: 
                data = json.load(f)
                if data:
                    if isinstance(data[0], dict): last_tweets = [e['timestamp'] for e in data]
                    else: last_tweets = data
            print(f"📂 Historial cargado: {len(last_tweets)} tweets.")
        except: pass

    REFRESH_RATE = 15 # Si ves bloqueos, súbelo a 10
    last_known_counts = {}

    while True:
        try:
            start_time = time.time()
            
            # 1. Obtener Datos
            markets_xtracker = sensor.get_active_counts()
            clob_data = pricer.get_market_prices()
            
            if not markets_xtracker:
                print(f"💤 Esperando datos API...", end="\r")
                time.sleep(REFRESH_RATE)
                continue

            # 2. Detector de Tweets
            markets_xtracker.sort(key=lambda x: x['count'], reverse=True)
            master_m = markets_xtracker[0]
            master_id = master_m['id']
            current_count = master_m['count']
            
            if master_id not in last_known_counts: last_known_counts[master_id] = current_count
            delta = current_count - last_known_counts[master_id]
            
            if delta > 0:
                print(f"\n🐦 ¡NUEVO TWEET! (+{delta})")
                now_ts = time.time()
                for _ in range(int(delta)): last_tweets.append(now_ts)
                try:
                    with open(FILES['history'], 'w') as f: json.dump([{'timestamp': t} for t in last_tweets], f, indent=2)
                except: pass
                last_known_counts[master_id] = current_count

            IS_WARMUP = len(last_tweets) < 5
            
            # 3. Análisis
            for m_poly in markets_xtracker:
                m_title = m_poly['title']
                m_clob = next((c for c in clob_data if titles_match_paranoid(m_title, c['title'])), None)
                if not m_clob: continue
                
                tape_rec = {'timestamp': time.time(), 'market': m_title, 'buckets': m_clob['buckets']}
                append_to_json_file(FILES['market_tape'], tape_rec)

                # Predicción
                pred_val, pred_std = brain.predict(last_tweets, m_poly['hours'])
                final_sim_mean = m_poly['count'] + pred_val
                if pred_std < 5: pred_std = 5.0
                
                mkt_consensus = brain.get_market_consensus(m_poly, m_clob['buckets'])
                
                if mkt_consensus:
                    combined_mean = (final_sim_mean * (1 - MARKET_WEIGHT)) + (mkt_consensus * MARKET_WEIGHT)
                    effective_std = max(pred_std + (abs(final_sim_mean - mkt_consensus)/4), 5.0)
                else:
                    combined_mean = final_sim_mean
                    effective_std = pred_std

                # Decisiones
                stats_ctx = {"count": m_poly['count'], "mean": combined_mean, "std": effective_std, "hours": m_poly['hours']}
                my_buckets = trader.get_owned_buckets_val(m_title)
                decisions_log = []
                
                for b in m_clob['buckets']:
                    p_min = norm.cdf(b['min'], combined_mean, effective_std)
                    p_max = norm.cdf(b['max'], combined_mean, effective_std)
                    fair = p_max - p_min
                    if "+" in b['bucket']: fair = 1.0 - p_min
                    b['fair'] = fair

                for b in m_clob['buckets']:
                    bid = b['bid']; ask = b['ask']; fair = b['fair']
                    b_mid = (b['min'] + b['max']) / 2
                    z_score = abs(b_mid - combined_mean) / effective_std
                    
                    pos_key = f"{m_title} | {b['bucket']}"
                    has_pos = pos_key in trader.portfolio['positions']
                    context = {"combined_mean": combined_mean, "z_score": z_score, "fair": fair}

                    if not has_pos and not IS_WARMUP:
                        if z_score <= MAX_Z_SCORE_ENTRY and ask >= MIN_PRICE_ENTRY:
                            is_cluster_ok = True
                            if ENABLE_CLUSTERING and my_buckets:
                                is_cluster_ok = any(abs(b_mid - ov) <= CLUSTER_RANGE for ov in my_buckets)
                            
                            if is_cluster_ok and fair > (ask + 0.15):
                                shares = min(trader.portfolio['cash'] / ask, 500)
                                if shares > 10:
                                    reason = f"Val+{fair-ask:.2f}"
                                    trader.execute("BUY", m_title, b['bucket'], ask, shares, reason, context)
                                    decisions_log.append({'bucket': b['bucket'], 'action': 'BUY', 'reason': reason})

                    elif has_pos:
                        pos = trader.portfolio['positions'][pos_key]
                        entry = pos['entry_price']
                        profit = (bid - entry) / entry if entry > 0 else 0
                        
                        if z_score > 2.0:
                            reason = f"Rot(Z={z_score:.1f})"
                            trader.execute("ROTATE", m_title, b['bucket'], bid, pos['shares'], reason, context)
                            decisions_log.append({'bucket': b['bucket'], 'action': 'ROTATE', 'reason': reason})
                        elif profit > 0.30 and z_score > 1.8:
                            reason = f"Pft+{profit*100:.0f}%"
                            trader.execute("TAKE_PROFIT", m_title, b['bucket'], bid, pos['shares'], reason, context)
                            decisions_log.append({'bucket': b['bucket'], 'action': 'PROFIT', 'reason': reason})
                
                trader.print_market_summary(m_title, stats_ctx, m_clob['buckets'], decisions_log)

            trader.print_portfolio_summary()
            
            elapsed = time.time() - start_time
            sleep_time = max(0.1, REFRESH_RATE - elapsed)
            print(f"⏱️ Scan took {elapsed:.2f}s | Sleeping {sleep_time:.1f}s... (Tweets: {len(last_tweets)})", end="\r")
            time.sleep(sleep_time)

        except KeyboardInterrupt: break
        except Exception:
            time.sleep(5)

if __name__ == "__main__":
    run()