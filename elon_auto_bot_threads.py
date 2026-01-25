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

# ==============================================================================
# CONFIGURACIÓN V11.1 (INFRA V10 + LOGIC V11)
# ==============================================================================
LOGS_DIR = 'logs'
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)

FILES = {
    'portfolio': os.path.join(LOGS_DIR, "portfolio.json"),
    'history': os.path.join(LOGS_DIR, "live_history.json"),
    'trades': os.path.join(LOGS_DIR, "trade_history.csv"),
    'monitor': os.path.join(LOGS_DIR, "bot_monitor.log"),
    'snapshots': os.path.join(LOGS_DIR, "snapshots_merged.json"),  # V11 Unificado
    'market_tape': os.path.join(LOGS_DIR, "market_tape_merged.json") # V11 Unificado
}

# Bio-Ritmos (V10)
HOURLY_MULTIPLIERS = {
    0: 0.97, 1: 0.80, 2: 0.42, 3: 0.20, 4: 0.39, 5: 0.48, 6: 2.11, 7: 1.41,
    8: 1.46, 9: 1.58, 10: 0.44, 11: 0.21, 12: 0.35, 13: 0.49, 14: 1.72, 15: 1.71,
    16: 1.37, 17: 2.03, 18: 1.34, 19: 1.24, 20: 1.01, 21: 0.89, 22: 0.82, 23: 0.61
}
DAILY_MULTIPLIERS = {0: 0.90, 1: 0.75, 2: 1.25, 3: 0.95, 4: 0.95, 5: 1.15, 6: 1.10}

API_CONFIG = {
    'base_url': "https://xtracker.polymarket.com/api",         # Para Tweets (Xtracker)
    'gamma_url': "https://gamma-api.polymarket.com/events",    # Para Mercados (Gamma)
    'clob_url': "https://clob.polymarket.com/prices",          # Para Precios (Bulk)
    'user': "elonmusk"
}

# --- PARÁMETROS V11 ---
MAX_Z_SCORE_ENTRY = 1.6
MIN_PRICE_ENTRY = 0.02
ENABLE_CLUSTERING = True
CLUSTER_RANGE = 40
MARKET_WEIGHT = 0.30

# ==============================================================================
# 🛠️ UTILIDADES V11
# ==============================================================================
def append_to_json_file(filename, new_record):
    """Guarda en JSON Array unificado (V11 Requirement)"""
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
    
    # Guardado atómico seguro
    temp_file = filename + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump(data_list, f, indent=2)
    os.replace(temp_file, filename)

def titles_match_paranoid(tracker_title, market_title):
    """V11: Comparación de títulos anti-bug (Ignora años 2025/2026)"""
    t1 = tracker_title.lower()
    t2 = market_title.lower()
    if t1 in t2 or t2 in t1: return True
    
    def get_nums(txt):
        return {n for n in re.findall(r'\d+', txt) if n not in ['2024', '2025', '2026']}
    
    nums1 = get_nums(t1)
    nums2 = get_nums(t2)
    return len(nums1.intersection(nums2)) >= 2

def log_monitor(message, force_print=False):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    try:
        with open(FILES['monitor'], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except: pass
    if force_print: print(line)

# ==============================================================================
# 🧠 CEREBRO HAWKES (V11 UPGRADED)
# ==============================================================================
class HawkesBrain:
    def __init__(self):
        self.params = {'mu': 0.4, 'alpha': 3.0, 'beta': 4.0}
        self.timestamps = []
        # Aquí podrías cargar datos históricos si tienes CSVs, 
        # para simplificar usamos parámetros por defecto robustos.

    def get_market_consensus(self, m_poly, clob_buckets):
        """V11: Calcula la media que el mercado está pagando"""
        sum_prod = 0; sum_w = 0
        for b in clob_buckets:
            try:
                # Parsear bucket
                if "+" in b['bucket']:
                    mid = int(re.search(r'\d+', b['bucket']).group()) + 20
                else:
                    nums = [int(n) for n in re.findall(r'\d+', b['bucket'])]
                    if len(nums) == 2: mid = sum(nums) / 2
                    else: continue
                
                price = (b['bid'] + b['ask']) / 2
                if price > 0.01:
                    sum_prod += mid * price
                    sum_w += price
            except: continue
        
        return (sum_prod / sum_w) if sum_w > 0 else None

    def predict(self, history_ms, hours):
        mu, a, b = self.params.values()
        boost = 0
        
        # Excitación reciente
        if history_ms:
            last_ts_sec = history_ms[-1] / 1000.0
            current_time_sec = time.time()
            cutoff = 10.0 / b
            
            for t_ms in reversed(history_ms):
                t_sec = t_ms / 1000.0
                dt = current_time_sec - t_sec
                if dt > cutoff * 3600: break
                if dt < 0: dt = 0
                boost += a * np.exp(-b * (dt / 3600.0))
        
        sims = []
        # Monte Carlo rápido
        for _ in range(500):
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

# ==============================================================================
# 📡 SENSOR POLYMARKET (V10: XTRACKER + GAMMA)
# ==============================================================================
class PolymarketSensor: # XTRACKER (Tweets Count)
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_active_counts(self):
        # Usamos el endpoint de USUARIO que sí funciona
        url = f"{API_CONFIG['base_url']}/users/{API_CONFIG['user']}"
        try:
            r = self.s.get(url, timeout=10).json()
            trackings = r.get('data', {}).get('trackings', [])
            
            res = []
            now = datetime.now(timezone.utc)
            
            # Procesamos en paralelo para velocidad
            def process_tracking(t):
                if not t.get('startDate') or not t.get('endDate'): return None
                try:
                    end = dateutil.parser.isoparse(t['endDate'])
                    # Filtro de fecha
                    if now.timestamp() > (end.timestamp() + 43200): return None # +12h margen
                    
                    # Detalle para obtener stats reales
                    det = self.s.get(f"{API_CONFIG['base_url']}/trackings/{t['id']}?includeStats=true", timeout=5).json()
                    d = det.get('data', {})
                    count = d.get('stats', {}).get('total', 0)
                    days = d.get('stats', {}).get('daysElapsed', 1)
                    
                    # Fix horario final
                    fixed_end = end.replace(hour=17, minute=0, second=0, tzinfo=timezone.utc)
                    hours = (fixed_end - now).total_seconds() / 3600.0
                    
                    return {
                        'id': t['id'], 'title': t['title'], 
                        'count': count, 'hours': hours,
                        'daily_avg': count/days if days > 0 else 0
                    }
                except: return None

            with ThreadPoolExecutor(max_workers=5) as ex:
                futures = [ex.submit(process_tracking, t) for t in trackings]
                for f in as_completed(futures):
                    if f.result(): res.append(f.result())
            return res
        except Exception as e:
            print(f"❌ Error Xtracker: {e}")
            return []

class ClobMarketScanner: # GAMMA + BULK (Precios)
    def __init__(self):
        self.s = requests.Session()
        self.bulk_url = API_CONFIG['clob_url']

    def get_market_prices(self):
        print("   🔎 Escaneando Precios...", end=" ")
        try:
            # 1. Traer Mercados de Gamma
            params = {"limit": 100, "active": "true", "closed": "false", "archived": "false", "order": "volume24hr", "ascending": "false"}
            data = self.s.get(API_CONFIG['gamma_url'], params=params, timeout=5).json()
            
            structure = []
            tokens = []
            
            for e in data:
                if "elon" not in e.get('title','').lower() or "tweets" not in e.get('title','').lower(): continue
                buckets = []
                for m in e.get('markets', []):
                    # Filtro Fecha Gamma (Doble check)
                    if m.get('endDate'):
                        try:
                            ed = dateutil.parser.isoparse(m['endDate']).replace(tzinfo=timezone.utc)
                            if datetime.now(timezone.utc) > ed: continue
                        except: pass
                    
                    # Parsear Bucket
                    q = m.get('question', '')
                    r = re.search(r'(\d+)-(\d+)', q); o = re.search(r'(\d+)\+', q)
                    if r: b_name, min_v, max_v = f"{r.group(1)}-{r.group(2)}", int(r.group(1)), int(r.group(2))
                    elif o: b_name, min_v, max_v = f"{o.group(1)}+", int(o.group(1)), 99999
                    else: continue
                    
                    try:
                        tid = json.loads(m['clobTokenIds'])[0]
                        buckets.append({'bucket': b_name, 'min': min_v, 'max': max_v, 'token': tid})
                        tokens.append({"token_id": tid, "side": "BUY"})
                        tokens.append({"token_id": tid, "side": "SELL"})
                    except: continue
                
                if buckets:
                    buckets.sort(key=lambda x: x['min'])
                    structure.append({'title': e['title'], 'buckets': buckets})

            # 2. Bulk Price Fetch
            price_map = {}
            if tokens:
                bulk = self.s.post(self.bulk_url, json=tokens, timeout=5).json()
                for tid, p in bulk.items():
                    price_map[tid] = {'bid': float(p.get('BUY',0) or 0), 'ask': float(p.get('SELL',0) or 0)}

            # 3. Ensamblar
            final = []
            for s in structure:
                clean_b = []
                for b in s['buckets']:
                    p = price_map.get(b['token'], {'bid':0, 'ask':0})
                    clean_b.append({**b, **p})
                final.append({'title': s['title'], 'buckets': clean_b})
            
            print("✅")
            return final
        except Exception as e:
            print(f"❌ Error Scanner: {e}")
            return []

# ==============================================================================
# 💼 PAPER TRADER (V11 COMPATIBLE)
# ==============================================================================
# ==============================================================================
# 💼 PAPER TRADER (V11.1 + VISUAL LOGS V10)
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
                'market': market, 'bucket': bucket, 
                'shares': shares, 'entry_price': price, 'avg_price': price
            }
        
        elif action in ["SELL", "ROTATE", "TAKE_PROFIT"]:
            if pos_key not in self.portfolio['positions']: return None
            pos = self.portfolio['positions'][pos_key]
            revenue = pos['shares'] * price
            self.portfolio['cash'] += revenue
            del self.portfolio['positions'][pos_key]
            
            pnl = revenue - (pos['shares'] * pos['entry_price'])
            self._log_csv(action, market, bucket, price, pos['shares'], reason, pnl)
        
        if action == "BUY":
            self._log_csv(action, market, bucket, price, shares, reason, 0)

        self.save()
        
        # Log Snapshot V11
        snap = {
            'timestamp': time.time(),
            'readable_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'action': action, 'market': market, 'bucket': bucket,
            'price': price, 'reason': reason, 'context': context
        }
        append_to_json_file(FILES['snapshots'], snap)
        return f"{action} OK"

    def _log_csv(self, action, market, bucket, price, shares, reason, pnl):
        header = not os.path.exists(FILES['trades'])
        with open(FILES['trades'], 'a') as f:
            if header: f.write("Timestamp,Action,Market,Bucket,Price,Shares,Reason,PnL,Cash_After\n")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts},{action},{market},{bucket},{price:.3f},{shares:.1f},{reason},{pnl:.2f},{self.portfolio['cash']:.2f}\n")

    # --- AQUÍ ESTÁ LA MAGIA VISUAL RECUPERADA ---
    def print_market_summary(self, market_title, stats_ctx, buckets_data, decisions):
        """Imprime la tabla bonita de la V10"""
        print("-" * 75)
        print(f">>> {market_title}")
        print(f"    📊 Act: {stats_ctx['count']} | 🧠 Gauss: μ={stats_ctx['mean']:.1f} σ={stats_ctx['std']:.1f} | ⏳ Quedan: {stats_ctx['hours']:.1f}h")
        print("-" * 75)
        print(f"{'BUCKET':<10} | {'BID':<6} | {'ASK':<6} | {'FAIR':<6} | {'Z-SCR':<5} | {'ACCIÓN':<10} | {'MOTIVO'}")
        
        for b in buckets_data:
            # Buscar decisión si existe
            act = "-"
            reason = "_"
            for d in decisions:
                if d['bucket'] == b['bucket']:
                    act = d['action']
                    reason = d['reason']
                    # Colores simples para consola
                    if "BUY" in act: act = f"🟢 {act}"
                    if "SELL" in act or "ROTATE" in act: act = f"🔴 {act}"
            
            # Calcular Z-Score para visualización
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
        print("\n💼 --- PORTFOLIO (SIMULADO) ---")
        print(f"   {'FECHAS EVENTO':<30} | {'BUCKET':<10} | {'ENTRADA':<8} | {'SHARES':<8}")
        print("   " + "-"*65)
        
        total_equity = self.portfolio['cash']
        for k, v in self.portfolio['positions'].items():
            short_market = v['market'].replace("Elon Musk # tweets ", "")[:30]
            print(f"   🔹 {short_market:<30} | {v['bucket']:<10} | ${v['entry_price']:.3f}   | {v['shares']:.1f}")
            # Estimación simple de valor actual (requeriría precio real)
            total_equity += (v['shares'] * v['entry_price']) 

        print("   " + "-"*65)
        print(f"   💵 Cash: ${self.portfolio['cash']:.2f} | 📈 Equity Est: ${total_equity:.2f}")
        print("-" * 75)

# ==============================================================================
# 🚀 EJECUCIÓN V11.1
# ==============================================================================
def run():
    print("🤖 ELON BOT V11.1 [HYBRID REAL + FAST SCAN]")
    
    # Inicialización de componentes
    brain = HawkesBrain()
    sensor = PolymarketSensor() # Xtracker (Tweets)
    pricer = ClobMarketScanner() # Gamma (Precios)
    trader = PaperTrader()
    
    # Cargar historial para Warmup
    last_tweets = []
    if os.path.exists(FILES['history']):
        try:
            with open(FILES['history']) as f: last_tweets = [e['timestamp'] for e in json.load(f)]
        except: pass

    # VELOCIDAD DE ESCANEO (Antes 60s, Ahora 6s)
    REFRESH_RATE = 6 

    while True:
        try:
            start_time = time.time()
            
            # 1. Obtener Datos (Infra V10)
            markets_xtracker = sensor.get_active_counts()
            clob_data = pricer.get_market_prices()
            
            if not markets_xtracker:
                print(f"💤 Sin mercados activos (Xtracker)... Reintentando en {REFRESH_RATE}s", end="\r")
                time.sleep(REFRESH_RATE)
                continue

            # Warmup Check
            IS_WARMUP = len(last_tweets) < 5
            
            # 2. Bucle de Mercados
            for m_poly in markets_xtracker:
                m_title = m_poly['title']
                
                # Buscar precios correspondientes (Matching Paranoico)
                m_clob = next((c for c in clob_data if titles_match_paranoid(m_title, c['title'])), None)
                if not m_clob: continue
                
                # Grabar Tape (V11 Feature)
                tape_rec = {'timestamp': time.time(), 'market': m_title, 'buckets': m_clob['buckets']}
                append_to_json_file(FILES['market_tape'], tape_rec)

                # 3. Predicciones V11 (Hybrid Brain)
                pred_sims = brain.predict(last_tweets, m_poly['hours'])
                # Factor biológico simplificado para la simulación media
                pred_sims_val = [s * 1.0 for s in pred_sims] 
                final_sims = np.array([m_poly['count'] + s for s in pred_sims_val])
                
                pred_mean = np.mean(final_sims)
                pred_std = np.std(final_sims)
                if pred_std < 5: pred_std = 5.0 # Suelo de seguridad
                
                # Hybrid Consensus (Fusión con Mercado)
                mkt_consensus = brain.get_market_consensus(m_poly, m_clob['buckets'])
                if mkt_consensus:
                    combined_mean = (pred_mean * (1 - MARKET_WEIGHT)) + (mkt_consensus * MARKET_WEIGHT)
                    conflict = abs(pred_mean - mkt_consensus)
                    # Si hay conflicto, aumentamos la desviación estándar (más prudencia)
                    effective_std = max(pred_std + (conflict/4), 5.0)
                else:
                    combined_mean = pred_mean
                    effective_std = pred_std

                # Contexto para la Tabla Visual
                stats_ctx = {
                    "count": m_poly['count'], 
                    "mean": combined_mean, 
                    "std": effective_std, 
                    "hours": m_poly['hours']
                }
                
                # 4. MOTOR DE DECISIÓN V11
                my_buckets = trader.get_owned_buckets_val(m_title)
                decisions_log = [] # Aquí guardamos las acciones para pintarlas en la tabla
                
                # Pre-cálculo de Fair Value para enriquecer la visualización
                for b in m_clob['buckets']:
                    p_min = norm.cdf(b['min'], combined_mean, effective_std)
                    p_max = norm.cdf(b['max'], combined_mean, effective_std)
                    fair = p_max - p_min
                    if "+" in b['bucket']: fair = 1.0 - p_min
                    b['fair'] = fair # Inyectamos 'fair' en el objeto bucket para usarlo abajo

                # Bucle de Buckets
                for b in m_clob['buckets']:
                    bid = b['bid']; ask = b['ask']; fair = b['fair']
                    
                    # Z-Score
                    b_mid = (b['min'] + b['max']) / 2
                    z_score = abs(b_mid - combined_mean) / effective_std
                    
                    # Estado Posición
                    pos_key = f"{m_title} | {b['bucket']}"
                    has_pos = pos_key in trader.portfolio['positions']
                    
                    context = {
                        "combined_mean": combined_mean, "z_score": z_score, 
                        "fair": fair, "hours": m_poly['hours']
                    }

                    # --- A. COMPRA (Sniper + Cluster) ---
                    if not has_pos and not IS_WARMUP:
                        # Filtros
                        if z_score > MAX_Z_SCORE_ENTRY: continue
                        if ask < MIN_PRICE_ENTRY: continue
                        
                        # Cluster check
                        if ENABLE_CLUSTERING and my_buckets:
                            if not any(abs(b_mid - ov) <= CLUSTER_RANGE for ov in my_buckets): continue
                        
                        # Value Logic
                        if fair > (ask + 0.15):
                            shares = min(trader.portfolio['cash'] / ask, 500)
                            if shares > 10:
                                reason = f"Value +{fair-ask:.2f}"
                                trader.execute("BUY", m_title, b['bucket'], ask, shares, reason, context)
                                decisions_log.append({'bucket': b['bucket'], 'action': 'BUY', 'reason': reason})

                    # --- B. VENTA (Rotación + Profit) ---
                    elif has_pos:
                        pos = trader.portfolio['positions'][pos_key]
                        entry = pos['entry_price']
                        profit_pct = (bid - entry) / entry if entry > 0 else 0
                        
                        # Rotación (Stop Loss Táctico)
                        if z_score > 2.0:
                            reason = f"Rot(Z={z_score:.1f})"
                            trader.execute("ROTATE", m_title, b['bucket'], bid, pos['shares'], reason, context)
                            decisions_log.append({'bucket': b['bucket'], 'action': 'ROTATE', 'reason': reason})
                        
                        # Take Profit
                        elif profit_pct > 0.30 and z_score > 1.8:
                            reason = f"Pft+{profit_pct*100:.0f}%"
                            trader.execute("TAKE_PROFIT", m_title, b['bucket'], bid, pos['shares'], reason, context)
                            decisions_log.append({'bucket': b['bucket'], 'action': 'PROFIT', 'reason': reason})
                
                # IMPRIMIR TABLA VISUAL (V10 Style)
                trader.print_market_summary(m_title, stats_ctx, m_clob['buckets'], decisions_log)

            # Resumen Global
            trader.print_portfolio_summary()
            
            # Control de Velocidad
            elapsed = time.time() - start_time
            sleep_time = max(0.5, REFRESH_RATE - elapsed)
            print(f"⏱️ Scan took {elapsed:.2f}s | Sleeping {sleep_time:.1f}s...", end="\r")
            time.sleep(sleep_time)

        except KeyboardInterrupt: break
        except Exception as e:
            print(f"🔥 Error Loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run()