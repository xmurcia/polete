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

# ==========================================
# CONFIGURACIÓN
# ==========================================
DATA_DIR = "brain_data"
PARAMS_FILE = "model_params.json"
LIVE_LOG = "live_history.json"

API_CONFIG = {
    'base_url': "https://xtracker.polymarket.com/api",
    'gamma_url': "https://gamma-api.polymarket.com/events",
    'clob_url': "https://clob.polymarket.com/price",
    'user': "elonmusk"
}

# ==========================================
# 1. SCANNER DE PRECIOS (CLOB + GAMMA)
# ==========================================
class ClobMarketScanner:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    # def _get_clob_price(self, token_id, side="sell"):
    #     """Consulta el precio 'Ask' (Venta) al CLOB."""
    #     try:
    #         params = {"token_id": token_id, "side": side}
    #         resp = self.session.get(API_CONFIG['clob_url'], params=params, timeout=2)
    #         data = resp.json()
    #         if 'price' in data and data['price']:
    #             return float(data['price'])
    #         return 0.0
    #     except: return 0.0

    def _get_clob_price(self, token_id, side):
        """
        side='sell' -> ASK (Precio para COMPRAR)
        side='buy'  -> BID (Precio para VENDER)
        """
        try:
            params = {"token_id": token_id, "side": side}
            resp = self.session.get(API_CONFIG['clob_url'], params=params, timeout=2)
            data = resp.json()
            if 'price' in data and data['price']:
                return float(data['price'])
            return 0.0 # Si no hay liquidez
        except: return 0.0

    def get_market_prices(self):
        """Devuelve precios reales de buckets de Elon."""
        print("   🔎 Escaneando precios en Order Book...", end=" ")
        try:
            params = {
                "limit": 50, "active": "true", "closed": "false",
                "archived": "false", "order": "volume24hr", "ascending": "false"
            }
            resp = self.session.get(API_CONFIG['gamma_url'], params=params, timeout=5)
            data = resp.json()
            
            valid_events = []
            for event in data:
                title = event.get('title', '').lower()
                if "elon" not in title or "tweet" not in title: continue
                if not event.get('markets'): continue
                
                markets_data = []
                for m in event['markets']:
                    if m['closed'] or not m['acceptingOrders']: continue
                    
                    # Regex para extraer buckets (100-109, 120+, etc)
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
                        
                        # OBTENEMOS LOS DOS PRECIOS
                        ask = self._get_clob_price(yes_token, "sell") # Para entrar (Buy)
                        bid = self._get_clob_price(yes_token, "buy")  # Para salir (Sell)
                        
                        markets_data.append({
                            'bucket': b_name, 
                            'ask': ask, 
                            'bid': bid, 
                            'min': min_v, 
                            'max': max_v
                        })
                    except: continue
                
                markets_data.sort(key=lambda x: x['min'])
                if markets_data:
                    valid_events.append({'title': event['title'], 'buckets': markets_data})
            
            print(f"✅")
            return valid_events
        except Exception as e:
            print(f"❌ Error Scanner: {e}")
            return []

# ==========================================
# 2. CEREBRO MATEMÁTICO (HAWKES)
# ==========================================
class HawkesBrain:
    def __init__(self):
        self.params = {'mu': 0.34, 'alpha': 3.16, 'beta': 3.71} 
        self.timestamps = [] 
        self._ensure_directories()
        self.load_and_train()

    def _ensure_directories(self):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

    def load_and_train(self):
        print("🧠 Cargando memoria histórica...")
        all_timestamps = []
        # Cargar CSVs
        for f in glob.glob(os.path.join(DATA_DIR, "*.csv")):
            try:
                df = pd.read_csv(f)
                df['Date/Time'] = pd.to_datetime(df['Date/Time'])
                df = df.drop_duplicates(subset=['Date/Time'])
                np.random.seed(42)
                for _, row in df.iterrows():
                    if row['Posts'] > 0:
                        ts = row['Date/Time'].timestamp()
                        all_timestamps.extend(ts + np.random.uniform(0, 3600, row['Posts']))
            except: pass
        
        # Cargar JSON reciente
        live_path = os.path.join(DATA_DIR, LIVE_LOG)
        if os.path.exists(live_path):
            try:
                with open(live_path, 'r') as f:
                    all_timestamps.extend([e['timestamp']/1000.0 for e in json.load(f)])
            except: pass

        if all_timestamps:
            self.timestamps = np.array(sorted(all_timestamps))
            self._optimize_params()

    def _optimize_params(self):
        if len(self.timestamps) < 50: return
        print("💪 Optimizando parámetros...")
        ts_h = (self.timestamps - self.timestamps[0]) / 3600.0
        T_max = ts_h[-1]
        
        def nll(p):
            mu, a, b = p
            if mu<=0 or a<=0 or b<=0: return 1e10
            n = len(ts_h)
            ll = -np.log(mu)
            r_prev, integral, term_sum = 0, mu*T_max, 0
            for i in range(1, n):
                dt = ts_h[i] - ts_h[i-1]
                r_curr = np.exp(-b*dt)*(r_prev + a)
                lam = mu + r_curr
                term_sum += np.log(lam)
                integral += (a/b)*(1 - np.exp(-b*(T_max - ts_h[i])))
                r_prev = r_curr
            return -(term_sum - integral)

        try:
            res = minimize(nll, [self.params['mu'], self.params['alpha'], self.params['beta']], 
                           method='L-BFGS-B', bounds=[(1e-4,None)]*3)
            if res.success: 
                self.params = dict(zip(['mu','alpha','beta'], res.x))
                print(f"✨ Params optimizados: {self.params}")
        except: pass

    def predict(self, history_ms, hours):
        mu, a, b = self.params.values()
        boost = 0
        if history_ms:
            t0 = history_ms[0]
            h_h = [(t-t0)/3600000.0 for t in history_ms]
            last = h_h[-1]
            for t in [x for x in h_h if (last-x) < 10.0/b]:
                boost += a * np.exp(-b*(last-t))
        
        sims = []
        for _ in range(1000):
            t, l_boost, ev = 0, boost, 0
            while t < hours:
                l_max = mu + l_boost
                if l_max <= 0: l_max = 0.001
                w = -np.log(np.random.uniform()) / l_max
                t += w
                if t >= hours: break
                l_boost *= np.exp(-b*w)
                if np.random.uniform() < (mu + l_boost)/l_max:
                    ev += 1; l_boost += a
            sims.append(ev)
        return sims

# ==========================================
# 3. SENSOR DE TWEETS (X-TRACKER)
# ==========================================
class PolymarketSensor:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_active_counts(self):
        try:
            r = self.s.get(f"{API_CONFIG['base_url']}/users/{API_CONFIG['user']}", timeout=5).json()
            res = []
            now = datetime.now(timezone.utc)
            for t in r.get('data', {}).get('trackings', []):
                # Verificar actividad de forma robusta
                is_active = t.get('isActive')
                if not is_active:
                    try:
                        start = datetime.fromisoformat(t['startDate'].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(t['endDate'].replace('Z', '+00:00'))
                        if start <= now <= end: is_active = True
                    except: pass
                
                if is_active:
                    try:
                        d = self.s.get(f"{API_CONFIG['base_url']}/trackings/{t['id']}?includeStats=true", timeout=5).json()['data']
                        end_d = datetime.fromisoformat(d['endDate'].replace('Z', '+00:00'))
                        hours = (end_d - now).total_seconds() / 3600.0
                        if hours > 0:
                            res.append({'id': t['id'], 'title': t['title'], 'count': d['stats']['total'], 'hours': hours})
                    except: pass
            return res
        except: return []

# ==========================================
# 4. SIMULADOR DE TRADING (PAPER HANDS)
# ==========================================
class PaperTrader:
    def __init__(self, initial_cash=1000.0):
        self.file_path = os.path.join(DATA_DIR, "portfolio.json")
        self.bet_size = 50.0  # Cuánto dinero apostar por trade
        self.portfolio = self._load()
        if not self.portfolio:
            self.portfolio = {"cash": initial_cash, "positions": {}, "history": []}

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f: return json.load(f)
            except: return None
        return None

    def _save(self):
        with open(self.file_path, 'w') as f: json.dump(self.portfolio, f, indent=2)

    def execute(self, market_title, bucket, signal, price):
        # Generar ID único para la posición (Ej: "Elon Jan 6|580+")
        pos_id = f"{market_title}|{bucket}"
        
        # --- LÓGICA DE COMPRA ---
        if "BUY" in signal:
            # Solo compramos si NO tenemos posición ya (para no duplicar infinito)
            # y si tenemos dinero suficiente
            if pos_id not in self.portfolio["positions"] and self.portfolio["cash"] >= self.bet_size:
                shares = self.bet_size / price
                self.portfolio["cash"] -= self.bet_size
                self.portfolio["positions"][pos_id] = {
                    "shares": shares,
                    "entry_price": price,
                    "market": market_title,
                    "bucket": bucket,
                    "timestamp": time.time()
                }
                self._save()
                return f"✅ ORDEN EJECUTADA: Compradas {shares:.1f} acciones a ${price:.3f}"

        # --- LÓGICA DE VENTA ---
        elif ("SELL" in signal or "DUMP" in signal):
            # Solo vendemos si tenemos la posición
            if pos_id in self.portfolio["positions"]:
                pos = self.portfolio["positions"].pop(pos_id)
                revenue = pos["shares"] * price
                profit = revenue - self.bet_size
                roi = (profit / self.bet_size) * 100
                
                self.portfolio["cash"] += revenue
                
                # Registrar en historial
                trade_record = {
                    "market": market_title,
                    "bucket": bucket,
                    "profit": profit,
                    "roi": roi,
                    "exit_time": time.time()
                }
                self.portfolio["history"].append(trade_record)
                self._save()
                
                color = "💰" if profit > 0 else "💸"
                return f"{color} ORDEN EJECUTADA: Venta cerrada. P&L: ${profit:.2f} ({roi:+.1f}%)"
        
        return None

    def print_summary(self, current_prices_data):
        """Calcula el valor actual de la cartera"""
        cash = self.portfolio["cash"]
        invested_value = 0.0
        
        print("\n💼 --- PORTFOLIO (SIMULADO) ---")
        
        # Calcular valor posiciones abiertas
        for pid, pos in self.portfolio["positions"].items():
            # Buscar precio actual
            current_price = pos['entry_price'] # Fallback
            
            # Intentar encontrar precio real actual en los datos del scanner
            found = False
            for m in current_prices_data:
                if m['title'] == pos['market']:
                    for b in m['buckets']:
                        if b['bucket'] == pos['bucket']:
                            # Valoramos a precio de BID (lo que sacaríamos si vendemos ya)
                            current_price = b.get('bid', 0)
                            found = True
                            break
                if found: break
            
            val = pos['shares'] * current_price
            pnl = val - (pos['shares'] * pos['entry_price'])
            invested_value += val
            
            print(f"   🔹 {pos['bucket']:<10} | Ent: {pos['entry_price']:.2f} | Act: {current_price:.2f} | P&L: {pnl:+.2f}")

        total_equity = cash + invested_value
        total_pnl = total_equity - 1000
        
        print(f"   💵 Cash: ${cash:.2f} | 📈 Equity: ${total_equity:.2f} | 🚀 Total P&L: {total_pnl:+.2f}")
        print("-----------------------------------")

# ==========================================
# 5. DIRECTOR DE ORQUESTA (CON TRADING SIM)
# ==========================================
def run():
    print("\n🤖 ELON-BOT: MARKET MAKER + PAPER TRADING")
    print("=========================================")
    
    brain = HawkesBrain()
    sensor = PolymarketSensor()
    pricer = ClobMarketScanner()
    trader = PaperTrader(initial_cash=1000.0) # <--- NUEVO
    
    last_counts = {}
    last_retrain_time = time.time()
    RETRAIN_INTERVAL = 21600 

    global_events = []
    log_path = os.path.join(DATA_DIR, LIVE_LOG)
    if os.path.exists(log_path):
        try:
            with open(log_path) as f: 
                d = json.load(f)
                global_events = [e for e in d if (time.time()*1000 - e['timestamp']) < 86400000]
        except: pass

    print(f"\n📡 Escuchando... (Mu={brain.params['mu']:.2f}, Alpha={brain.params['alpha']:.2f})")

    while True:
        try:
            # 1. MANTENIMIENTO
            if time.time() - last_retrain_time > RETRAIN_INTERVAL:
                brain.load_and_train()
                last_retrain_time = time.time()

            # 2. SENSOR
            markets = sensor.get_active_counts()
            if not markets: time.sleep(30); continue

            tweet_detected = False
            changes = []
            max_diff = 0
            for m in markets:
                curr = m['count']
                prev = last_counts.get(m['id'])
                if prev is None: last_counts[m['id']] = curr; continue
                if curr > prev:
                    diff = curr - prev
                    if diff > max_diff: max_diff = diff
                    changes.append(f"{m['title']}: +{diff}")
                    tweet_detected = True
                last_counts[m['id']] = curr

            now_ms = time.time() * 1000
            if tweet_detected:
                print(f"\n🚨 TWEET DETECTADO! {changes}")
                for _ in range(max_diff): global_events.append({'timestamp': now_ms})
                with open(log_path, 'w') as f: json.dump(global_events, f)
            
            # 3. ANÁLISIS CONTINUO
            global_events = [e for e in global_events if (now_ms - e['timestamp']) < 86400000]
            clob_data = pricer.get_market_prices()
            
            if clob_data:
                print(f"\n⏱️ {datetime.now().strftime('%H:%M:%S')} | Análisis de Mercado")

            ts_list = [e['timestamp'] for e in global_events]
            
            for m_poly in markets:
                relevant_prices = next((p for p in clob_data if m_poly['title'] in p['title']), None)
                if not relevant_prices: continue

                # SIMULACIÓN
                sims = brain.predict(ts_list, m_poly['hours'])
                final_sims = np.array([m_poly['count'] + s for s in sims])
                total_s = len(final_sims)

                print("-" * 75)
                print("-" * 75)
                print("-" * 75)
                print("-" * 75)
                
                # Visualización (Opcional: puedes comentar esto si ocupa mucho espacio)
                print(f"\n>>> {m_poly['title']} [Actual: {m_poly['count']} tweets] ({m_poly['hours']:.1f}h left)")

                print("-" * 75)

                # --- VISUALIZACIÓN ASCII (HISTOGRAMA) ---
                # --- VISUALIZACIÓN POR BUCKETS REALES ---
                print("    Distribución por Buckets (Mercado):")
                
                # 1. Pre-calcular probabilidades para encontrar el máximo (para escalar la barra)
                bucket_stats = []
                max_prob = 0
                
                for b in relevant_prices['buckets']:
                    # Contamos cuántas simulaciones caen exactamente en este bucket
                    count = sum(1 for x in final_sims if b['min'] <= x <= b['max'])
                    prob = count / len(final_sims)
                    if prob > max_prob: max_prob = prob
                    bucket_stats.append({'label': b['bucket'], 'prob': prob})

                # 2. Dibujar las barras alineadas con los buckets
                for item in bucket_stats:
                    # Filtro: Solo mostramos buckets con probabilidad relevante (>0.5%)
                    # para no llenar la pantalla de líneas vacías si hay 50 buckets.
                    if item['prob'] > 0.005: 
                        # Escalar barra a 30 caracteres
                        bar_len = int((item['prob'] / max_prob) * 30) if max_prob > 0 else 0
                        bar = "█" * bar_len
                        
                        # Si es el bucket favorito (el más alto), ponle un icono
                        icon = "⭐" if item['prob'] == max_prob else ""
                        
                        print(f"    {item['label']:<10} | {bar} ({item['prob']*100:.1f}%) {icon}")
                
                print("-" * 75)

                
                
                print(f"{'BUCKET':<10} | {'BID':<8} | {'ASK':<8} | {'FAIR':<8} | {'ACCIÓN':<10} | {'MOTIVO'}")

                for b in relevant_prices['buckets']:
                    matches = sum(1 for x in final_sims if b['min'] <= x <= b['max'])
                    fair_val = matches / total_s
                    ask = b.get('ask', 0)
                    bid = b.get('bid', 0)
                    
                    action = "-"
                    reason = "_"
                    
                    # --- REGLAS DE TRADING ---
                    # 1. Comprar si hay mucho Edge
                    if ask > 0 and fair_val > (ask + 0.10): 
                        action = f"🟢 BUY"
                        diff = fair_val - ask
                        reason = f"Edge +{diff:.2f} (Barato)"
                    
                    # 2. Vender si está caro
                    elif bid > 0 and fair_val < (bid - 0.05): 
                        action = f"🔴 SELL"
                        diff = bid - fair_val
                        reason = f"Sobreprecio +{diff:.2f}"
                    
                    # 3. Pánico (Time Decay)
                    if m_poly['hours'] < 4 and fair_val < 0.01 and bid > 0.05:
                         action = "💀 DUMP"
                         reason = "Zombie (Time Decay)"
                    
                    # Mostrar tabla visual
                    if ask > 0.01 or fair_val > 0.01:
                        print(f"{b['bucket']:<10} | {bid:.3f}    | {ask:.3f}    | {fair_val:.3f}    | {action:<10} | {reason}")

                    # --- EJECUCIÓN SIMULADA ---
                    if action != "-" and action != "💀 DUMP": # Ejecutamos BUY o SELL normales
                        # Pasamos solo la acción limpia ("🟢 BUY") al trader
                        clean_action = action.split()[1] if " " in action else action
                        trade_res = trader.execute(m_poly['title'], b['bucket'], action, ask if "BUY" in action else bid)
                        if trade_res:
                            print(f"   👉 {trade_res}")

            # 4. RESUMEN DE CARTERA AL FINAL DEL CICLO
            if clob_data:
                trader.print_summary(clob_data)
                
            if not clob_data: print(".", end="", flush=True)
            time.sleep(0.2)

        except KeyboardInterrupt: break
        except Exception as e: 
            print(f"Error Loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run()