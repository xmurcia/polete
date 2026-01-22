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
from scipy.stats import norm # Necesario para la Campana de Gauss

# ==========================================
# CONFIGURACIÓN
# ==========================================
DAILY_METRICS_THREE_WEEKS_DIR = "daily_metrics_three_weeks"
LOGS_DIR = 'logs'

PORTFOLIO_PAPER_TRADER = "portfolio.json"
LIVE_LOG = "live_history.json"
TRADE_LOG = "trade_history.csv"
MONITOR_LOG = "bot_monitor.log"      # <--- NUEVO: Diario del Capitán Automático
SNAPSHOTS_DIR = os.path.join(LOGS_DIR, "snapshots") # <--- NUEVO: Fotos Forenses
MARKET_TAPE_DIR = os.path.join(LOGS_DIR, "market_tape") # <--- NUEVO: Grabación continua

# Aseguramos que existan los directorios
if not os.path.exists(DAILY_METRICS_THREE_WEEKS_DIR): os.makedirs(DAILY_METRICS_THREE_WEEKS_DIR)
if not os.path.exists(SNAPSHOTS_DIR): os.makedirs(SNAPSHOTS_DIR)
if not os.path.exists(MARKET_TAPE_DIR): os.makedirs(MARKET_TAPE_DIR)

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
    'clob_url': "https://clob.polymarket.com/price",
    'user': "elonmusk"
}

def is_market_tradable(market_json):
    """
    Filtro Maestro V9: Decide si un mercado merece ser procesado.
    """
    # 1. Si está cerrado definitivamente (Settled), basura.
    if market_json.get('closed') is True:
        return False

    # 2. Filtro de Fecha (El Juez Supremo)
    # Si la fecha ya pasó, no nos interesa (evita procesar eventos de 2024).
    end_date_str = market_json.get('endDate') # "2026-01-20T17:00:00Z"
    if end_date_str:
        try:
            # Convertimos a objeto fecha consciente de zona horaria
            dt_end = datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            # Damos 2 horas de margen post-cierre para ver el settlement
            margin = 2 * 3600 
            if datetime.now(timezone.utc).timestamp() > (dt_end.timestamp() + margin):
                return False
        except:
            pass # Si falla la fecha, ante la duda lo dejamos pasar (Fail Open)

    # 3. NOTA: NO filtramos 'acceptingOrders'. 
    # Queremos ver el precio aunque el trading esté pausado.
    
    return True

# --- Ejemplo de uso con tu lista de mercados ---
# active_markets = [m for m in all_markets if is_market_tradable(m)]


# ==========================================
# 1. SCANNER DE PRECIOS (OPTIMIZADO V9 - BULK REQUEST)
# ==========================================
class ClobMarketScanner:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Referer": "https://polymarket.com/"
        })
        # Endpoint para precios en lote (mucho más rápido que hilos)
        self.bulk_prices_url = "https://clob.polymarket.com/prices"

    def get_market_prices(self):
        print("   🔎 Escaneando Order Book (Modo Bulk V9)...", end=" ")
        t_start = time.time()
        
        try:
            # --- 🛑 CORRECCIÓN CRÍTICA DE PARÁMETROS ---
            # Quitamos 'active' y 'closed' para que la API no oculte eventos en settlement.
            # Traemos TODO lo que tenga volumen.
            params = {
                "limit": 100, 
                "active": "true", 
                "closed": "false",
                "archived": "false", 
                "order": "volume24hr", 
                "ascending": "false"
            }
            resp = self.session.get(API_CONFIG['gamma_url'], params=params, timeout=5)
            data = resp.json()
            
            market_structure = []
            tokens_to_fetch = [] # Lista para la petición masiva (Bulk)

            # Fecha actual para calculos internos si hicieran falta
            now_utc = datetime.now(timezone.utc)
            
            for event in data:
                title = event.get('title', '').lower()

                # Filtro básico por nombre
                if "elon" not in title or "tweets" not in title: continue
                if not event.get('markets'): continue
                
                buckets_list = []
                for m in event['markets']:
                    
                    # --- 🛡️ FILTRO MAESTRO ---
                    # Delegamos la decisión a la función lógica.
                    # Esto permite pasar eventos "zombies" pero visibles.
                    if not is_market_tradable(m):
                        continue
                    # -------------------------
                           
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
                        buckets_list.append({
                            'bucket': b_name, 'min': min_v, 'max': max_v, 'token': yes_token
                        })
                        
                        # --- PREPARACIÓN BULK ---
                        # En lugar de crear tareas, añadimos a la lista de petición
                        tokens_to_fetch.append({"token_id": yes_token, "side": "BUY"})  # Bid
                        tokens_to_fetch.append({"token_id": yes_token, "side": "SELL"}) # Ask
                        
                    except: continue
                
                if buckets_list:
                    buckets_list.sort(key=lambda x: x['min'])
                    market_structure.append({'title': event['title'], 'buckets': buckets_list})

            # --- EJECUCIÓN BULK (SÚPER RÁPIDA) ---
            price_map = {} 
            if tokens_to_fetch:
                try:
                    # Una sola llamada HTTP para todos los precios
                    bulk_resp = self.session.post(
                        self.bulk_prices_url, 
                        json=tokens_to_fetch, 
                        timeout=5
                    )
                    bulk_data = bulk_resp.json()
                    
                    # Mapeamos la respuesta al formato que usa tu bot
                    # La API devuelve: { "token_id": { "BUY": "0.45", "SELL": "0.46" } }
                    for token_id, prices in bulk_data.items():
                        price_map[token_id] = {
                            "buy": float(prices.get("BUY", 0) or 0),   # Bid
                            "sell": float(prices.get("SELL", 0) or 0)  # Ask
                        }
                except Exception as e:
                    print(f"⚠️ Fallo en Bulk Request: {e}")

            # Construcción de datos finales (Igual que antes)
            final_data = []
            for mkt in market_structure:
                clean_buckets = []
                for b in mkt['buckets']:
                    precios = price_map.get(b['token'], {})
                    clean_buckets.append({
                        'bucket': b['bucket'],
                        'min': b['min'],
                        'max': b['max'],
                        'ask': precios.get('sell', 0.0),
                        'bid': precios.get('buy', 0.0)
                    })
                final_data.append({'title': mkt['title'], 'buckets': clean_buckets})

            elapsed = time.time() - t_start
            print(f"✅ ({elapsed:.2f}s)")
            return final_data
        except Exception as e:
            print(f"❌ Error Scanner: {e}")
            return []

# ==========================================
# 2. CEREBRO MATEMÁTICO (HAWKES - CONFIG NATURAL)
# ==========================================
class HawkesBrain:
    def __init__(self):
        # Parámetros iniciales por defecto (se optimizarán solos)
        self.params = {'mu': 0.4, 'alpha': 3.0, 'beta': 4.0} 
        self.timestamps = [] 
        self.history_df = None # Para consultas de eventos
        self._ensure_directories()
        self.load_and_train()

    def _ensure_directories(self):
        if not os.path.exists(DAILY_METRICS_THREE_WEEKS_DIR): os.makedirs(DAILY_METRICS_THREE_WEEKS_DIR)

    def load_and_train(self):
        print("🧠 Cargando y limpiando datos (Modo NATURAL: Sin frenos ni adrenalina)...")
        
        all_timestamps = []
        csv_files = glob.glob(os.path.join(DAILY_METRICS_THREE_WEEKS_DIR, "*.csv"))
        df_list = []
        
        print(f"   - Procesando {len(csv_files)} archivos CSV...")
        
        for f in csv_files:
            nombre = os.path.basename(f)
            # Evitamos leer archivos generados por nosotros
            if "dataset" in nombre or "trade" in nombre or "portfolio" in nombre: continue

            try:
                # 1. Leer CSV
                df_temp = pd.read_csv(f)
                df_temp.columns = [c.strip() for c in df_temp.columns] # Limpiar nombres col
                
                if 'Date/Time' in df_temp.columns:
                    # 2. Conversión de fecha BLINDADA (ISO + Texto)
                    df_temp['Date_Clean'] = pd.to_datetime(df_temp['Date/Time'], errors='coerce')
                    
                    # Si falló la conversión rápida, intentamos la lenta (para "Dec 9")
                    if df_temp['Date_Clean'].isna().any():
                        def parsear_fecha(x):
                            try: return dateutil.parser.parse(str(x))
                            except: return pd.NaT
                        mask_nulos = df_temp['Date_Clean'].isna()
                        df_temp.loc[mask_nulos, 'Date_Clean'] = df_temp.loc[mask_nulos, 'Date/Time'].apply(parsear_fecha)

                    df_temp['Date/Time'] = df_temp['Date_Clean']
                    df_temp = df_temp.dropna(subset=['Date/Time'])
                    
                    if not df_temp.empty:
                        df_list.append(df_temp)
                
            except Exception as e: 
                pass # Ignoramos archivos corruptos silenciosamente para no ensuciar el log

        # 3. FUSIONAR Y LIMPIAR
        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            
            # --- LA CLAVE DEL ÉXITO: ELIMINAR DUPLICADOS ---
            # Esto evita que el modelo vea 110 tweets/día cuando solo hubo 55.
            clean_df = full_df.drop_duplicates(subset='Date/Time', keep='first')
            
            # Guardamos copia para consultas del portfolio
            self.history_df = clean_df.copy()
            self.history_df.set_index('Date/Time', inplace=True)
            
            print(f"   - Datos Históricos: {len(clean_df)} horas únicas recuperadas.")

            np.random.seed(42)
            
            # 4. GENERAR TIMESTAMPS (SIN FILTROS)
            for _, row in clean_df.iterrows():
                posts_real = int(row['Posts'])
                fecha = row['Date/Time']
                
                # --- AQUÍ ESTABA EL FRENO, YA NO ESTÁ ---
                # Usamos los datos crudos. Si Elon hizo 70 tweets, usamos 70.
                posts_count = posts_real 
                # ----------------------------------------

                if posts_count > 0:
                    ts = fecha.timestamp()
                    # Distribuir aleatoriamente dentro de la hora
                    all_timestamps.extend(ts + np.random.uniform(0, 3600, posts_count))
        
        # 5. CARGAR LIVE DATA
        live_path = os.path.join(LOGS_DIR, LIVE_LOG)
        if os.path.exists(live_path):
            try:
                with open(live_path, 'r') as f:
                    live_events = json.load(f)
                    # Filtramos eventos de las últimas 24h
                    now_ms = time.time() * 1000
                    live_events = [e for e in live_events if (now_ms - e['timestamp']) < 86400000]
                    all_timestamps.extend([e['timestamp']/1000.0 for e in live_events])
            except: pass

        # 6. ENTRENAR
        if all_timestamps:
            self.timestamps = np.array(sorted(all_timestamps))
            print(f"✅ Entrenando con {len(self.timestamps)} eventos totales.")
            self._optimize_params()
        else:
            print("⚠️ No hay datos suficientes para entrenar.")

    def _optimize_params(self):
        if len(self.timestamps) < 50: return
        print("💪 Optimizando parámetros (Matemática Pura)...")
        
        # Normalizamos tiempos para que el algoritmo converja mejor
        ts_h = (self.timestamps - self.timestamps[0]) / 3600.0
        T_max = ts_h[-1]
        
        def nll(p):
            mu, a, b = p
            # Restricciones de estabilidad básicas
            if mu <= 0.001 or a <= 0.001 or b <= 0.001: return 1e10
            if a >= b: return 1e10 # Evita explosiones infinitas
            
            n = len(ts_h)
            log_mu = -np.log(mu)
            
            term_sum = 0
            integral = mu * T_max
            A_prev = 0
            
            term_sum += np.log(mu)
            
            # Bucle optimizado de Log-Likelihood
            for i in range(1, n):
                dt = ts_h[i] - ts_h[i-1]
                A_curr = np.exp(-b * dt) * (A_prev + 1)
                lam = mu + a * A_curr
                term_sum += np.log(lam)
                A_prev = A_curr
            
            term_integral_exc = np.sum((a/b) * (1 - np.exp(-b * (T_max - ts_h))))
            return -(term_sum - (integral + term_integral_exc))

        try:
            # Optimizamos sin forzar nada manualmente
            res = minimize(nll, [self.params['mu'], self.params['alpha'], self.params['beta']], 
                           method='L-BFGS-B', bounds=[(1e-4, None)]*3)
            
            if res.success: 
                self.params = dict(zip(['mu','alpha','beta'], res.x))
                print(f"✨ Params optimizados: {self.params}")
                # Resultado esperado: Mu ~0.4, Beta ~4.0 -> Media ~55 tweets/día
            else:
                print("⚠️ Optimización no convergió, usando anteriores.")
        except Exception as e:
            print(f"⚠️ Error optimizando: {e}")

    def predict(self, history_ms, hours):
        # Simulación de Monte Carlo estándar
        mu, a, b = self.params.values()
        boost = 0
        
        # Calculamos la excitación actual basada en los tweets recientes (Live)
        if history_ms:
            # Usamos el último timestamp conocido como referencia t0 local
            last_ts_sec = history_ms[-1] / 1000.0
            
            # Recorremos hacia atrás para sumar excitación
            # Solo importan los eventos recientes (ej: últimas 4 horas)
            cutoff = 10.0 / b # Más allá de esto la exponencial es 0
            
            # Convertimos history_ms a segundos relativos al 'ahora' virtual
            # Nota: Esta es una aproximación rápida para el boost
            current_time_sec = time.time()
            
            for t_ms in reversed(history_ms):
                t_sec = t_ms / 1000.0
                dt = current_time_sec - t_sec
                if dt > cutoff * 3600: break # Demasiado viejo
                if dt < 0: dt = 0
                
                # Sumamos la excitación remanente
                boost += a * np.exp(-b * (dt / 3600.0))
        
        sims = []
        # Realizamos 1000 futuros posibles
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
# 3. SENSOR DE TWEETS (OPTIMIZADO)
# ==========================================
class PolymarketSensor:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Mozilla/5.0"})

    def _fetch_tracking_detail(self, t, now):
        try:
            # 1. Petición segura a la API
            response = self.s.get(f"{API_CONFIG['base_url']}/trackings/{t['id']}?includeStats=true", timeout=5).json()
            d = response.get('data', {})
            
            end_date_str = d.get('endDate') or t.get('endDate')
            hours = 0.0

            if end_date_str:
                try:
                    # A. Parseamos la fecha original (que viene mal, ej: 05:00:59Z)
                    original_dt = dateutil.parser.isoparse(end_date_str)
                    
                    # B. 🔨 MARTILLAZO HORARIO: FORZAMOS LAS 17:00 UTC (18:00 ESPAÑA)
                    # Mantenemos Año, Mes y Día originales, pero reescribimos la hora.
                    fixed_end_date = original_dt.replace(
                        hour=17, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                    )
                    
                    # C. Calculamos las horas restantes usando la FECHA CORREGIDA
                    hours = (fixed_end_date - now).total_seconds() / 3600.0
                    
                except Exception as e:
                    # Si falla el parseo, hours se queda en 0.0
                    pass

            # 2. Obtención del Conteo
            count = d.get('stats', {}).get('total', 0)

            # 3. FILTRO DE VISIBILIDAD GENÉRICO
            # Si le quedan horas (o acaba de terminar hace menos de 2h), lo mostramos.
            if hours > -2.0:
                return {
                    'id': t['id'], 
                    'title': t['title'], 
                    'count': count, 
                    'hours': hours, # Esta hora ahora es CORRECTA (hasta las 18:00)
                    'active': True 
                }

        except Exception:
            pass
        
        return None

    def get_active_counts(self):
        try:
            # Petición a la API de trackings
            r = self.s.get(f"{API_CONFIG['base_url']}/users/{API_CONFIG['user']}", timeout=5).json()
            trackings = r.get('data', {}).get('trackings', [])
           
            res = []
            now = datetime.now(timezone.utc)
            
            candidates = []
            for t in trackings:
                # --- 🧠 LÓGICA V9: DATE-FIRST ---
                # Ignoramos si la API dice que está inactivo.
                # Si la fecha actual está dentro del rango (con margen), LO QUEREMOS.
                
                start_str = t.get('startDate')
                end_str = t.get('endDate')
                
                if start_str and end_str:
                    try:
                        # Parseamos fechas (soportando formato ISO con Z)
                        start = dateutil.parser.isoparse(start_str)
                        end = dateutil.parser.isoparse(end_str)
                        
                        # Margen de 12 horas extra tras el cierre para ver la resolución
                        margin = pd.Timedelta(hours=12)
                        
                        # Si HOY es menor que (Fin + 12h), el evento es relevante
                        if now <= (end + margin):
                            candidates.append(t)
                            continue # Ya lo tenemos, pasamos al siguiente
                    except:
                        pass # Si falla fecha, pasamos al fallback
                
                # Fallback: Si no pudimos leer fechas, usamos el flag isActive por defecto
                if t.get('isActive'): 
                    candidates.append(t)

            # Procesamiento en paralelo para detalles (Count, Hours...)
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self._fetch_tracking_detail, t, now): t for t in candidates}
                for f in as_completed(futures):
                    try:
                        result = f.result()
                        if result: res.append(result)
                    except: pass

            return res
        except Exception as e:
            print(f"Error en get_active_counts: {e}")
            return []

# ==========================================
# 4. PAPER TRADER (CON LOGGING CSV)
# ==========================================
class PaperTrader:
    def __init__(self, initial_cash=1000.0):
        # Asegúrate de que estas rutas coincidan con tu configuración
        self.file_path = os.path.join(LOGS_DIR, PORTFOLIO_PAPER_TRADER)
        self.log_path = os.path.join(LOGS_DIR, TRADE_LOG)
        
        # --- GESTIÓN DE RIESGO ---
        self.risk_pct_normal = 0.04  # 4% del capital en jugadas seguras
        self.risk_pct_lotto = 0.01   # 1% del capital en loterías (FISH)
        self.min_bet = 5.0           # Apuesta mínima en dólares
        
        self.portfolio = self._load()
        self._ensure_log_header()

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

    def _ensure_log_header(self):
        """Crea el archivo CSV con cabeceras si no existe"""
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding='utf-8') as f:
                f.write("Timestamp,Action,Market,Bucket,Price,Shares,Reason,PnL,Cash_After\n")

    def _clean_market_name(self, full_title):
        """Limpia el nombre del mercado para el log"""
        import re
        month_map = {
            "january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr",
            "may": "May", "june": "Jun", "july": "Jul", "august": "Aug",
            "september": "Sep", "october": "Oct", "november": "Nov", "december": "Dec"
        }
        pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d+)'
        matches = re.findall(pattern, full_title, re.IGNORECASE)
        
        if len(matches) >= 2:
            m1, d1 = matches[0]; m2, d2 = matches[1]
            m1_short = month_map.get(m1.lower(), m1[:3].title())
            m2_short = month_map.get(m2.lower(), m2[:3].title())
            return f"{m1_short} {d1} - {m2_short} {d2}"
        elif len(matches) == 1:
            m1, d1 = matches[0]
            m1_short = month_map.get(m1.lower(), m1[:3].title())
            return f"Event {m1_short} {d1}"
        return "Evento Global"

    def _log_trade(self, action, market, bucket, price, shares, reason, pnl=0.0):
        """Escribe una línea en el CSV"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_clean = market.replace(",", "")
        reason_clean = reason.replace(",", ".")
        
        row = f"{ts},{action},{market_clean},{bucket},{price:.3f},{shares:.1f},{reason_clean},{pnl:.2f},{self.portfolio['cash']:.2f}\n"
        
        with open(self.log_path, "a", encoding='utf-8') as f:
            f.write(row)

    def _calculate_position_size(self, signal, reason, available_cash):
        """
        Calcula cuánto dinero apostar basado en el tipo de jugada.
        """
        # 1. Determinar el porcentaje de riesgo
        if "FISH" in signal or "Lotto" in reason:
            pct = self.risk_pct_lotto # 1% para Lotería
        else:
            pct = self.risk_pct_normal # 4% para Normal
            
        # 2. Calcular monto
        bet_amount = available_cash * pct
        
        # 3. Aplicar suelo mínimo (para no hacer trades de 50 céntimos)
        bet_amount = max(bet_amount, self.min_bet)
        
        # 4. Cap final: No podemos apostar más de lo que tenemos
        if bet_amount > available_cash:
            bet_amount = available_cash
            
        return bet_amount

    def execute(self, market_title, bucket, signal, price, reason="Manual"):
        pos_id = f"{market_title}|{bucket}"
        
        # --- COMPRA ---
        if "BUY" in signal or "FISH" in signal: # FISH también es una compra
            if pos_id not in self.portfolio["positions"]:
                
                # CÁLCULO DINÁMICO DEL TAMAÑO
                bet_amount = self._calculate_position_size(signal, reason, self.portfolio["cash"])
                
                # Solo ejecutamos si tenemos cash suficiente y el monto merece la pena
                if self.portfolio["cash"] >= bet_amount and bet_amount >= self.min_bet:
                    shares = bet_amount / price
                    self.portfolio["cash"] -= bet_amount
                    
                    self.portfolio["positions"][pos_id] = {
                        "shares": shares,
                        "entry_price": price,
                        "market": market_title,
                        "bucket": bucket,
                        "timestamp": time.time(),
                        "invested": bet_amount # Guardamos cuánto invertimos
                    }
                    self._save()
                    
                    # LOGGING
                    self._log_trade(signal, market_title, bucket, price, shares, reason)
                    return f"✅ BUY ({reason}): ${bet_amount:.2f} ({shares:.1f} shares @ {price:.3f})"
                else:
                    return None # No hay cash suficiente

        # --- VENTA ---
        elif ("SELL" in signal or "DUMP" in signal):
            if pos_id in self.portfolio["positions"]:
                pos = self.portfolio["positions"].pop(pos_id)
                revenue = pos["shares"] * price
                
                # Recuperamos el coste original (si no lo guardamos antes, lo estimamos)
                cost_basis = pos.get("invested", pos["shares"] * pos["entry_price"])
                
                profit = revenue - cost_basis
                roi = (profit / cost_basis) * 100 if cost_basis > 0 else 0
                
                self.portfolio["cash"] += revenue
                
                trade_record = {
                    "market": market_title,
                    "bucket": bucket,
                    "profit": profit,
                    "roi": roi,
                    "exit_time": time.time()
                }
                self.portfolio["history"].append(trade_record)
                self._save()
                
                # LOGGING
                self._log_trade(signal, market_title, bucket, price, pos['shares'], reason, pnl=profit)
                
                color = "💰" if profit > 0 else "💸"
                return f"{color} SELL: P&L ${profit:.2f} ({roi:+.1f}%)"
        
        return None

    # -----------------------------------------------------------
    # 1. Añade este método auxiliar a tu clase (o fuera de ella)
    # -----------------------------------------------------------
    def _get_event_label(self, start_ts, duration_days=7):
        """Convierte un timestamp en un texto: 'Tweets del 12 Dic al 19 Dic'"""
        import datetime
        
        # Diccionario de meses
        meses = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
        
        try:
            # Asumimos que el bucket ID es el timestamp de inicio
            ts = float(start_ts)
            dt_start = datetime.datetime.fromtimestamp(ts)
            dt_end = dt_start + datetime.timedelta(days=duration_days)
            
            m_start = meses.get(dt_start.month, "")
            m_end = meses.get(dt_end.month, "")
            
            if dt_start.month == dt_end.month:
                return f"Tweets {dt_start.day}-{dt_end.day} {m_start}"
            else:
                return f"Tweets {dt_start.day} {m_start} - {dt_end.day} {m_end}"
        except:
            return f"Evento {start_ts}" # Fallback si falla la fecha

    # -----------------------------------------------------------
    # 2. Tu función print_summary modificada
    # -----------------------------------------------------------
    def print_summary(self, current_prices_data):
        cash = self.portfolio["cash"]
        invested_value = 0.0
        
        # Lista temporal para almacenar filas antes de imprimir
        rows_to_print = []

        print("\n💼 --- PORTFOLIO (SIMULADO) ---")
        print(f"   🔹 {'FECHAS EVENTO':<20} | {'BUCKET':<10} | {'PRECIO ENT.':<12} | {'PRECIO ACT.':<12} | {'P&L ($)':<10}")
        print("   " + "-"*85)

        for pid, pos in self.portfolio["positions"].items():
            current_price = pos['entry_price']
            
            # 1. Obtener nombre limpio del evento
            full_title = pos.get('market', 'Unknown')
            event_date_label = self._clean_market_name(full_title)
            
            # 2. Buscar precio actual en los datos del scanner
            found = False
            for m in current_prices_data:
                # Usamos una comparación laxa (si el título limpio coincide)
                if self._clean_market_name(m['title']) == event_date_label: 
                    for b in m['buckets']:
                        if str(b['bucket']) == str(pos['bucket']):
                            current_price = b.get('bid', 0)
                            found = True
                            break
                if found: break
            
            # --- 🧹 FILTRO DE LIMPIEZA (NUEVO) ---
            # Si 'found' sigue siendo False, significa que el mercado ya no está en el escáner
            # (porque expiró o se filtró por tiempo). Lo saltamos para que no salga en pantalla.
            if not found:
                continue
            # -------------------------------------
            
            # Cálculos financieros
            val = pos['shares'] * current_price
            pnl = val - (pos['shares'] * pos['entry_price'])
            invested_value += val
            
            # 3. Preparar datos para ordenación
            # Extraemos el valor numérico del bucket para ordenar (ej: "280-299" -> 280)
            try:
                bucket_str = str(pos['bucket'])
                # Tomamos lo que hay antes del guion o el más
                bucket_val = int(bucket_str.split('-')[0].replace('+', ''))
            except:
                bucket_val = 99999 # Si falla, lo mandamos al final

            # Guardamos en la lista en lugar de imprimir directamente
            rows_to_print.append({
                'label': event_date_label,
                'bucket_str': str(pos['bucket']),
                'bucket_val': bucket_val,
                'entry': pos['entry_price'],
                'current': current_price,
                'pnl': pnl
            })

        # 4. ORDENACIÓN MÁGICA
        # Primero por Nombre de Evento (label), luego por Bucket (bucket_val)
        rows_to_print.sort(key=lambda x: (x['label'], x['bucket_val']))

        # 5. IMPRESIÓN LIMPIA
        last_label = None
        for row in rows_to_print:
            # Si cambiamos de evento, ponemos una línea separadora
            if last_label and row['label'] != last_label:
                print("   " + "-"*85)
            
            print(f"   🔹 {row['label']:<20} | {row['bucket_str']:<10} | ${row['entry']:<11.2f} | ${row['current']:<11.2f} | {row['pnl']:+6.2f}")
            last_label = row['label']

        total_equity = cash + invested_value
        total_pnl = total_equity - 1000
        
        print("   " + "-"*85)
        print(f"   💵 Cash: ${cash:.2f} | 📈 Equity: ${total_equity:.2f} | 🚀 Total P&L: {total_pnl:+.2f}")
        print("-------------------------------------------------------------------------------------")

# ==========================================
# 5. SENSOR DE PÁNICO (DETECTOR DE MOMENTUM V9)
# ==========================================
class MarketPanicSensor:
    def __init__(self, sensitivity=1.5):
        self.history = {} 
        self.sensitivity = sensitivity
        self.window_size = 5

    def analyze(self, market_data):
        alerts = []
        for m in market_data:
            for b in m['buckets']:
                key = f"{m['title']}|{b['bucket']}"
                if key not in self.history:
                    self.history[key] = {'asks': deque(maxlen=self.window_size), 'bids': deque(maxlen=self.window_size)}
                
                hist = self.history[key]
                ask = b['ask'] # Precio de compra (para detectar subidas)
                bid = b['bid'] # Precio de venta (para detectar bajadas/dumps)
                
                hist['asks'].append(ask)
                hist['bids'].append(bid)
                
                if len(hist['asks']) >= 3:
                    avg_ask = sum(hist['asks']) / len(hist['asks'])
                    avg_bid = sum(hist['bids']) / len(hist['bids'])
                    
                    # 1. DETECTOR DE PUMP (Ola subiendo - FOMO)
                    # Si el precio sube violentamente -> Posible entrada en bucket siguiente
                    if avg_ask > 0.01 and ask > (avg_ask * self.sensitivity):
                        alerts.append({
                            'type': 'PUMP',
                            'market_title': m['title'],
                            'bucket': b['bucket'],
                            'min': b.get('min', 0),
                            'price': ask,
                            'change': (ask/avg_ask) - 1.0
                        })

                    # 2. DETECTOR DE DUMP (Ola rompiendo - PÁNICO)
                    # Si el BID cae violentamente (ej: de 0.50 a 0.30) -> Oportunidad de Recompra o Stop Loss
                    # Usamos la inversa de la sensibilidad (ej: precio < media / 1.5)
                    if avg_bid > 0.05 and bid < (avg_bid / self.sensitivity):
                        alerts.append({
                            'type': 'DUMP',
                            'market_title': m['title'],
                            'bucket': b['bucket'],
                            'min': b.get('min', 0), 
                            'price': bid,
                            'change': (bid/avg_bid) - 1.0
                        })
                        
        return alerts

# ==========================================
# 6. DIRECTOR DE ORQUESTA V9.3 (NUCLEAR SAFETY FINAL)
# ==========================================
def run():
    print("\n🤖 ELON-BOT: V9.3 (HAWKES + GAUSSIAN + SAFETY LOCK)")
    print("======================================================")
    
    # --- SISTEMA DE LOGGING AUTOMÁTICO ---
    def log_monitor(message, force_print=False):
        """Escribe en el diario automático y opcionalmente en pantalla"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        try:
            with open(os.path.join(LOGS_DIR, MONITOR_LOG), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except: pass
        if force_print: print(line)

    def save_trade_snapshot(action, market, bucket, price, reason, context_data):
        """Guarda una 'foto' completa de la decisión para auditoría"""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts_str}_{action}_{bucket}.json"
        filepath = os.path.join(SNAPSHOTS_DIR, filename)
        
        # Sanitizar datos para JSON
        if 'simulations_sample' in context_data:
            try: context_data['simulations_sample'] = [float(x) for x in context_data['simulations_sample']]
            except: pass
            
        snapshot = {
            "timestamp": time.time(),
            "readable_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action, "market": market, "bucket": bucket,
            "price": price, "reason": reason, "context": context_data
        }
        with open(filepath, "w", encoding="utf-8") as f: json.dump(snapshot, f, indent=2)
        log_monitor(f"📸 SNAPSHOT: {filename}")

    def save_market_tape(clob_data, markets_meta):
        """Graba el estado COMPLETO del mercado para análisis futuro"""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tape_{ts_str}.json"
        filepath = os.path.join(MARKET_TAPE_DIR, filename)
        tape_record = {
            "timestamp": time.time(),
            "readable_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "meta": markets_meta, "order_book": clob_data
        }
        with open(filepath, "w", encoding="utf-8") as f: json.dump(tape_record, f)

    # --- INICIALIZACIÓN ---
    def get_bio_multiplier():
        now = datetime.now()
        h_mult = HOURLY_MULTIPLIERS.get(now.hour, 1.0)
        d_mult = DAILY_MULTIPLIERS.get(now.weekday(), 1.0)
        return h_mult * d_mult, h_mult, d_mult

    brain = HawkesBrain()
    sensor = PolymarketSensor()
    pricer = ClobMarketScanner()
    trader = PaperTrader(initial_cash=1000.0)
    panic_sensor = MarketPanicSensor(sensitivity=1.4) 
    
    last_counts = {}
    last_retrain_time = time.time()
    last_monitor_log_time = 0
    last_tape_recording_time = 0
    
    RETRAIN_INTERVAL = 21600
    tape_interval_dynamic = 1800 

    global_events = []
    log_path = os.path.join(LOGS_DIR, LIVE_LOG)
    
    # Cargar historial
    if os.path.exists(log_path):
        try:
            with open(log_path) as f: 
                d = json.load(f)
                global_events = [e for e in d if (time.time()*1000 - e['timestamp']) < 86400000]
        except: pass

    log_monitor("🚀 INICIO V9.3 (NUCLEAR SAFETY)", force_print=True)

    while True:
        try:
            # 1. MANTENIMIENTO
            if time.time() - last_retrain_time > RETRAIN_INTERVAL:
                brain.load_and_train()
                last_retrain_time = time.time()
                log_monitor("🧠 Cerebro Re-entrenado")

            # 2. ESCANEO (XTRACKER)
            markets = sensor.get_active_counts()
            markets_map = {m['title'].lower(): m for m in markets}

            if not markets: 
                print("💤 No active markets...", end="\r")
                time.sleep(5) 
                continue
            
            # MODO ADRENALINA vs CRUCERO
            min_hours_left = min([m['hours'] for m in markets]) if markets else 99.0
            
            if min_hours_left < 1.0: 
                current_mode = "🩸 ADRENALINA"
                tape_interval_dynamic = 10     
                panic_sens = 1.15              
                refresh_sleep = 3.0 #0.5 --> sustituimos para mayor velocidad  
            else:
                current_mode = "🚢 CRUCERO"
                tape_interval_dynamic = 1800   
                panic_sens = 1.40              
                refresh_sleep =  15 #2.0        --> sustituimos para mayor velocidad    

            panic_sensor.sensitivity = panic_sens

            # 3. DETECCIÓN DE TWEETS
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
                msg = f"🚨 TWEET DETECTADO: {changes}"
                print(f"\n{msg}")
                log_monitor(msg)
                for _ in range(max_diff): global_events.append({'timestamp': now_ms})
                with open(log_path, 'w') as f: json.dump(global_events, f)
            
            global_events = [e for e in global_events if (now_ms - e['timestamp']) < 86400000]
            
            # =========================================================
            # 🛡️ ZONA DE SEGURIDAD MAESTRA (DEFINICIÓN)
            # =========================================================
            events_24h = [e for e in global_events if (now_ms - e['timestamp']) < 24*3600*1000]
            ts_list = [e['timestamp'] for e in global_events]
            
            # REGLA: Necesitamos al menos 5 tweets para operar.
            IS_WARMUP = len(ts_list) < 5
            
            # 4. CÁLCULO DE TENDENCIA
            if len(global_events) < 5: 
                pace_24h = 4.0; pace_status = "🔰 WARMUP"
            else:
                pace_24h = len(events_24h) / 24.0
                pace_status = "🔥" if pace_24h > 3.5 else ("❄️" if pace_24h < 2.0 else "😐")
            # =========================================================

            # 5. ORDER BOOK & TRADING
            clob_data = pricer.get_market_prices()
            
            if clob_data:
                bio_mult, h_m, d_m = get_bio_multiplier()

                # --- A. ALERTAS DE PÁNICO (AHORA CON CANDADO) ---
                raw_alerts = panic_sensor.analyze(clob_data)
                
                # 🔒 CANDADO: Si estamos en WARMUP, el pánico está desactivado
                if not IS_WARMUP:
                    for alert in raw_alerts:
                        m_info = next((m for t, m in markets_map.items() if t in alert['market_title'].lower()), None)
                        if not m_info: continue

                        distancia = alert['min'] - m_info['count']
                        
                        if alert['type'] == 'PUMP':
                            if 0 < distancia < 10: 
                                trader.execute(alert['market_title'], alert['bucket'], "🏄 SURF BUY", alert['price'], 
                                            reason=f"Pump Momentum (+{alert['change']:.1%})")

                        elif alert['type'] == 'DUMP':
                            sims = brain.predict(ts_list, m_info['hours'])
                            mean_prediction = m_info['count'] + np.mean(sims) * bio_mult
                            
                            if mean_prediction >= alert['min']:
                                trader.execute(alert['market_title'], alert['bucket'], "💎 DIP BUY", alert['price'], 
                                            reason=f"Pánico Injustificado (-{alert['change']:.1%})")
                
                # --- B. LOGGING & TAPE ---
                if time.time() - last_monitor_log_time > 900: 
                    status_pace = "WARMUP" if IS_WARMUP else f"{pace_24h:.1f}"
                    log_monitor(f"STATUS | Mode: {current_mode} | Pace: {status_pace} t/h | Bio: x{bio_mult:.2f}")
                    last_monitor_log_time = time.time()

                if time.time() - last_tape_recording_time > tape_interval_dynamic:
                    save_market_tape(clob_data, markets)
                    last_tape_recording_time = time.time()
                    if current_mode == "🩸 ADRENALINA":
                        print("   📼 Tape High-Res Grabado")
                
                warn_txt = "⚠️ WARMUP (NO TRADING)" if IS_WARMUP else f"Pace: {pace_24h:.1f} t/h"
                print(f"\n⏱️ {datetime.now().strftime('%H:%M:%S')} | {current_mode} | {warn_txt}")

            # Func. Auxiliar
            def titles_match(tracker_title, market_title):
                t1 = tracker_title.lower(); t2 = market_title.lower()
                if t1 in t2 or t2 in t1: return True
                nums1 = set(re.findall(r'\d+', t1)); nums2 = set(re.findall(r'\d+', t2))
                return len(nums1.intersection(nums2)) >= 2
            
            # --- 6. GESTIÓN ACTIVA DE PORTAFOLIO (ANNICA) ---
            if clob_data and trader.portfolio:
                for pos_key, pos_data in list(trader.portfolio.items()):
                    # Anti-Crash Fix
                    parts = pos_key.split(" | ")
                    if len(parts) != 2: continue
                    m_title, b_name = parts

                    # Buscar precio
                    current_price = 0.0
                    for m_clob in clob_data:
                        if titles_match(m_title, m_clob['title']):
                            for b_clob in m_clob['buckets']:
                                if b_clob['bucket'] == b_name:
                                    current_price = b_clob['bid']
                                    break
                    
                    if current_price > 0:
                        roi = (current_price - pos_data['avg_price']) / pos_data['avg_price']
                        if roi > 0.20 and pos_data['shares'] > 500:
                            trader.execute(m_title, b_name, "💰 SCALP SELL", current_price, 
                                         reason=f"Annica: Take Profit (+{roi*100:.1f}%)")

            # --- 7. ANÁLISIS DE MERCADO (GAUSSIAN ENGINE) ---
            for m_poly in markets:
                if m_poly['hours'] > 0.1: m_poly['active'] = True
                if m_poly['hours'] < 0.25: continue

                relevant_prices = next((p for p in clob_data if titles_match(m_poly['title'], p['title'])), None)
                if not relevant_prices: continue

                # PREDICCIÓN (CON PROTECCIÓN VISUAL)
                hours_to_predict = m_poly['hours']
                if hours_to_predict > 14.0: hours_to_predict -= 12.0
                else: hours_to_predict *= 0.90

                if IS_WARMUP:
                    pred_mean = float(m_poly['count'])
                    pred_std = 1.0 # Dummy
                    print(f"   ⚠️ WARMUP: Esperando {5-len(ts_list)} tweets más...", end="\r")
                else:
                    base_sims = brain.predict(ts_list, hours_to_predict)
                    sims = [s * bio_mult for s in base_sims] 
                    final_sims = np.array([m_poly['count'] + s for s in sims])
                    
                    pred_mean = np.mean(final_sims)
                    pred_std = np.std(final_sims)
                    if pred_std < 0.01: pred_std = 0.01

                # VISUALIZACIÓN
                dias = int(m_poly['hours'] // 24)
                horas_rest = int(m_poly['hours'] % 24)
                time_str = f"{dias}d {horas_rest}h" if dias > 0 else f"{m_poly['hours']:.1f}h"
                
                blind_tag = " (👁️ CIEGO - NO TRADING)" if IS_WARMUP else ""

                print("-" * 75)
                print(f"\n>>> {m_poly['title']}{blind_tag}")
                print(f"    📊 Act: {m_poly['count']} | 🧠 Gauss: μ={pred_mean:.1f} σ={pred_std:.1f} | ⏳ Quedan: {time_str}")
                print("-" * 75)
                print(f"{'BUCKET':<10} | {'BID':<8} | {'ASK':<8} | {'FAIR':<8} | {'ACCIÓN':<10} | {'MOTIVO'}")
                
                min_feasible_total = m_poly['count'] + (m_poly['hours'] * 0.9)
                is_long_term = (m_poly['hours'] > 72.0)

                for b in relevant_prices['buckets']:
                    if 'min' not in b: continue 
                    if b['max'] < m_poly['count']: continue
                    if b['max'] < min_feasible_total: continue 

                    # ============================================
                    # ⛔ ZONA DE SEGURIDAD: CANDADO NUCLEAR
                    # ============================================
                    if IS_WARMUP:
                        # Si estamos en warmup, imprimimos y SALTAMOS (continue)
                        # No calculamos probabilidades ni lógica de compra.
                        ask = b.get('ask', 0); bid = b.get('bid', 0)
                        print(f"{b['bucket']:<10} | {bid:.3f}    | {ask:.3f}    | {'-':<8} | {'-':<10} | 🔒 WARMUP")
                        continue
                    # ============================================

                    # CÁLCULO DE PROBABILIDAD (Solo si NO es warmup)
                    prob_min = norm.cdf(b['min'], loc=pred_mean, scale=pred_std)
                    prob_max = norm.cdf(b['max'] + 1, loc=pred_mean, scale=pred_std)
                    fair_val = prob_max - prob_min
                    
                    if "+" in b['bucket']: fair_val = 1.0 - prob_min
                    
                    ask = b.get('ask', 0); bid = b.get('bid', 0)

                    # 🛑 FIX ANTI-ESPEJISMOS (GHOST PRICES)
                    # Si el precio es menor a 0.2 céntimos, es un error de liquidez o falta de datos.
                    # Saltamos para evitar comprar "infinito" a precio 0.00.
                    if ask < 0.001: 
                        continue 

                    action = "-"; reason = "_"; special_tag = ""

                    spread = ask - bid
                    spread_pct = (spread / ask) if ask > 0 else 1.0
                    
                    # --- LÓGICA DE TRADING ---
                    
                    bad_spread = spread > 0.05 and spread_pct > 0.20
                    
                    # A. VALUE
                    if action == "-":
                        is_cheap = (ask < 0.05)
                        ok_spread = (not bad_spread) or is_cheap
                        if ask > 0 and fair_val > (ask + 0.10) and ok_spread: 
                            action = f"🟢 BUY"; diff = fair_val - ask; reason = f"Value +{diff:.2f}"
                        elif bid > 0.10 and fair_val < (bid - 0.10): 
                            action = f"🔴 SELL"; diff = bid - fair_val; reason = f"Sobreprecio +{diff:.2f}"

                    # B. LOTTO (CORREGIDO)
                    # Solo buscamos "pelotazos" si el mercado está muy activo (Pace alto)
                    # y vemos un bucket muy alto barato.
                    if action == "-" and is_long_term and pace_status != "🔰 WARMUP":
                        dist_from_mean = pred_mean - b['max'] 
                        
                        # ELIMINADO EL BLOQUE "SLOWDOWN" QUE COMPRABA BUCKETS BAJOS
                        # Nos quedamos solo con la apuesta "Active" (Breakout alcista)
                        
                        if dist_from_mean < -120 and pace_24h > 3.2 and ask <= 0.005 and ask > 0:
                                action = "🎣 FISH"; reason = "Lotto (Active)"; special_tag = "BUY"

                    # C. SNIPER
                    if action == "-" and m_poly['hours'] < 1.0:
                        if 0.75 < ask < 0.95 and b['min'] <= pred_mean <= b['max']:
                             action = "🔫 SNIPER"; reason = "Breakout >0.75 Last Hour"; special_tag = "BUY"

                    # ZOMBIE
                    if m_poly['hours'] < 6.0 and fair_val < 0.01 and bid > 0.02 and action == "-":
                         action = "💀 DUMP"; reason = "Exit"
                    
                    # EXECUTE
                    if ask > 0.01 or fair_val > 0.01 or special_tag:
                        print(f"{b['bucket']:<10} | {bid:.3f}    | {ask:.3f}    | {fair_val:.3f}    | {action:<10} | {reason}")

                    if action != "-": 
                        raw_act = action.split()[1] if " " in action else action
                        trade_act = special_tag if special_tag else ("SELL" if "DUMP" in action else raw_act)
                        price = ask if "BUY" in trade_act else bid
                        
                        trade_res = trader.execute(m_poly['title'], b['bucket'], trade_act, price, reason=reason)
                        
                        if trade_res: 
                            print(f"   👉 {trade_res}")
                            context = {
                                "predicted_mean": pred_mean,
                                "predicted_std": pred_std,
                                "fair_value": fair_val,
                                "pace_24h": pace_24h,
                                "hours_left": m_poly['hours'],
                                "bucket_info": b
                            }
                            save_trade_snapshot(trade_act, m_poly['title'], b['bucket'], price, reason, context)

            if clob_data: trader.print_summary(clob_data)
            if not clob_data: print(".", end="", flush=True)
            time.sleep(refresh_sleep)

        except KeyboardInterrupt: break
        except Exception as e: 
            print(f"Error Loop: {e}"); 
            log_monitor(f"ERROR: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run()
