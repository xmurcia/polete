import asyncio
import websockets
import json
import requests
import re
import os
import glob
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import deque

# Librerías Visuales
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console, Group
from rich import box
from rich.text import Text
from rich.align import Align

# ==========================================
# AUXILIAR: HISTOGRAMA COMPACTO
# ==========================================
def render_mini_histogram(buckets, sims, total_sims):
    if not sims: return ""
    lines = []
    max_prob = 0
    bucket_probs = []
    
    # Calcular probabilidades
    for b in buckets:
        hits = sum(1 for x in sims if b['min'] <= x <= b['max'])
        prob = hits / total_sims
        if prob > 0: # Solo mostramos lo que tiene probabilidad
            bucket_probs.append({'label': b['bucket'], 'prob': prob})
            if prob > max_prob: max_prob = prob
            
    # Renderizar (Solo las barras más relevantes para ahorrar espacio)
    # Ordenamos por bucket numérico
    for item in bucket_probs:
        if item['prob'] > 0.05: # Solo mostramos si > 5%
            bar_len = int((item['prob'] / max_prob) * 10) # Barra corta (max 10 chars)
            bar = "█" * bar_len
            color = "yellow" if item['prob'] == max_prob else "dim white"
            # Formato: 240+ █ 40%
            lines.append(f"[{color}]{item['label']:<5} {bar} {item['prob']:.0%}[/]")
            
    return "\n".join(lines)

# ==========================================
# CONFIGURACIÓN
# ==========================================
print("⚙️ Iniciando ELON-BOT V7.2 (Complete)...", flush=True)

DATA_DIR = "brain_data"
LIVE_LOG = "live_history.json"
PORTFOLIO_FILE = "portfolio_v2.json" # <--- CAMBIA ESTO A "portfolio_v3.json" PARA EMPEZAR DE CERO

API_CONFIG = {
    'base_url': "https://xtracker.polymarket.com/api",
    'gamma_url': "https://gamma-api.polymarket.com/events",
    'user': "elonmusk"
}

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# ==========================================
# ESTADO DEL BOT
# ==========================================
class BotState:
    def __init__(self):
        self.market_map = {}      
        self.prices = {}          
        self.tweet_data = {} 
        self.logs = deque(maxlen=12) 
        self.portfolio = self._load_portfolio()
        self.brain = None         

    def _load_portfolio(self):
        path = os.path.join(DATA_DIR, PORTFOLIO_FILE)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: 
                    p = json.load(f)
                    if "positions" not in p: p["positions"] = {}
                    if "cash" not in p: p["cash"] = 1000.0
                    return p
            except: pass
        # Si no existe, crea uno nuevo con $1000
        return {"cash": 1000.0, "positions": {}, "history": [], "equity": 1000.0}

    def save_portfolio(self):
        path = os.path.join(DATA_DIR, PORTFOLIO_FILE)
        with open(path, 'w') as f: json.dump(self.portfolio, f, indent=2)

    def log(self, msg, style="white"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.appendleft(f"[{style}][{timestamp}] {msg}[/]")

    def update_price(self, token_id, price, side):
        if token_id not in self.prices:
            self.prices[token_id] = {'bid': 0.0, 'ask': 0.0}
        if side == "BUY": self.prices[token_id]['bid'] = float(price)
        else:             self.prices[token_id]['ask'] = float(price)

STATE = BotState()

# ==========================================
# 1. CEREBRO IA
# ==========================================
class HawkesBrain:
    def __init__(self):
        self.params = {'mu': 0.34, 'alpha': 3.16, 'beta': 3.71} 

    def predict(self, hours):
        mu, a, b = self.params.values()
        sims = []
        # Monte Carlo simplificado
        for _ in range(200):
            t, ev = 0, 0
            while t < hours:
                step = -np.log(np.random.uniform()) / (mu * 4.0) 
                t += step
                if t < hours: ev += 1
            sims.append(ev)
        return sims

# ==========================================
# 2. SCANNER (Mapeo Robusto)
# ==========================================
class ClobMarketScanner:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def normalize_key(self, title):
        # Usamos la nueva lógica universal
        return normalize_key_universal(title)

    def map_market_ids(self):
        STATE.log("🔎 Indexando mercados...", "cyan")
        try:
            params = {"limit": 150, "active": "true", "closed": "false", "order": "volume24hr", "ascending": "false"}
            resp = self.session.get(API_CONFIG['gamma_url'], params=params, timeout=5)
            data = resp.json()
            
            count = 0
            for event in data:
                title = event.get('title', '')
                if "elon" not in title.lower() or "tweet" not in title.lower(): continue
                if not event.get('markets'): continue
                
                group_key = self.normalize_key(title)
                
                for m in event['markets']:
                    if m['closed']: continue
                    
                    q = m.get('question', '')
                    r_match = re.search(r'(\d+)-(\d+)', q)
                    o_match = re.search(r'(\d+)\+', q)
                    
                    if r_match: 
                        bucket_val = f"{r_match.group(1)}-{r_match.group(2)}"
                        min_v, max_v = int(r_match.group(1)), int(r_match.group(2))
                    elif o_match: 
                        bucket_val = f"{o_match.group(1)}+"
                        min_v, max_v = int(o_match.group(1)), 9999
                    else: continue

                    try:
                        t_ids = json.loads(m['clobTokenIds'])
                        yes_token = t_ids[0]
                        STATE.market_map[yes_token] = {
                            "original_title": title,
                            "group_key": group_key, 
                            "bucket": bucket_val,
                            "min": min_v,
                            "max": max_v
                        }
                        count += 1
                    except: continue
            STATE.log(f"✅ {count} mercados listos", "green")
        except Exception as e:
            STATE.log(f"❌ Scanner: {e}", "red")

# ==========================================
# 3. WEBSOCKET
# ==========================================
async def websocket_loop():
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    while True:
        ids = list(STATE.market_map.keys())
        if not ids: await asyncio.sleep(2); continue

        try:
            async with websockets.connect(uri, ping_interval=None) as ws:
                STATE.log("🔌 Socket Online", "green")
                chunk_size = 20
                for i in range(0, len(ids), chunk_size):
                    chunk = ids[i:i + chunk_size]
                    await ws.send(json.dumps({"assets_ids": chunk, "type": "market"}))
                    await asyncio.sleep(0.1)
                
                while True:
                    msg = await ws.recv()
                    if not msg: continue
                    try: raw_data = json.loads(msg)
                    except: continue

                    batch = raw_data if isinstance(raw_data, list) else [raw_data]
                    for data in batch:
                        if not isinstance(data, dict): continue
                        if data.get("event_type") == "price_change":
                            for c in data.get("price_changes", []):
                                tid = c.get("asset_id") or data.get("asset_id")
                                if tid in STATE.market_map:
                                    STATE.update_price(tid, c['price'], c['side'])
        except: await asyncio.sleep(5)

# ==========================================
# 4. SENSOR TWEETS (Grabación y Sync)
# ==========================================
async def tweet_sensor_loop():
    sensor_session = requests.Session()
    hist_path = os.path.join(DATA_DIR, LIVE_LOG)
    local_history = []
    
    # Cargar historial previo si existe
    if os.path.exists(hist_path):
        try:
            with open(hist_path, 'r') as f: local_history = json.load(f)
        except: pass

    while True:
        try:
            loop = asyncio.get_event_loop()
            r = await loop.run_in_executor(None, lambda: sensor_session.get(f"{API_CONFIG['base_url']}/users/{API_CONFIG['user']}"))
            trackings = r.json().get('data', {}).get('trackings', [])
            now = datetime.now(timezone.utc)
            
            for t in trackings:
                try:
                    end_str = t['endDate'].replace('Z', '+00:00')
                    end_d = datetime.fromisoformat(end_str)
                    hours = (end_d - now).total_seconds() / 3600.0
                    
                    if hours > 0:
                        det = await loop.run_in_executor(None, lambda: sensor_session.get(f"{API_CONFIG['base_url']}/trackings/{t['id']}?includeStats=true"))
                        count = det.json()['data']['stats']['total']
                        
                        # Normalización idéntica al Scanner
                        norm_key = normalize_key_universal(t['title'])
                        
                        # Detectar nuevo tweet y grabar
                        prev = STATE.tweet_data.get(norm_key)
                        if prev and count > prev['count']:
                            diff = count - prev['count']
                            STATE.log(f"🚨 TWEET DETECTADO (+{diff})", "bold yellow")
                            # GRABAR EN DISCO
                            local_history.append({'timestamp': time.time()*1000, 'count_state': count})
                            with open(hist_path, 'w') as f: json.dump(local_history, f)

                        STATE.tweet_data[norm_key] = {"count": count, "hours": hours, "title": t['title']}
                except: pass
        except: pass
        await asyncio.sleep(5)

# ==========================================
# 5. RENDERIZADOR HISTOGRAMA
# ==========================================
def render_ascii_histogram(buckets, sims, total_sims):
    lines = []
    bucket_probs = []
    max_prob = 0
    
    # Calcular
    for b in buckets:
        hits = sum(1 for x in sims if b['min'] <= x <= b['max'])
        prob = hits / total_sims
        bucket_probs.append({'label': b['bucket'], 'prob': prob})
        if prob > max_prob: max_prob = prob
        
    # Dibujar
    for item in bucket_probs:
        if item['prob'] > 0.01 or item['prob'] == max_prob:
            bar_len = int((item['prob'] / max_prob) * 25) if max_prob > 0 else 0
            bar_char = "█" * bar_len
            percent = f"{item['prob']*100:.0f}%"
            
            color = "white"
            icon = ""
            if item['prob'] == max_prob: 
                color = "yellow"
                icon = "⭐"
            
            lines.append(f"[{color}]{item['label']:<8} | {bar_char} {percent} {icon}[/]")
    return "\n".join(lines)

# ==========================================
# DASHBOARD TIPO GRID (3x3)
# ==========================================
def generate_dashboard():
    # --- HEADER (Igual que antes) ---
    current_equity = STATE.portfolio['cash']
    invested_val = 0.0
    active_positions_by_event = {} 
    
    for pos_id, pos in list(STATE.portfolio['positions'].items()):
        # ... (Cálculo de valor idéntico al anterior) ...
        entry = pos.get('entry', pos.get('entry_price', 0.0))
        price = entry 
        if pos_id in STATE.prices: price = STATE.prices[pos_id]['bid']
        val = pos['shares'] * price
        invested_val += val
        b_clean = pos.get('bucket','?').replace('[','').replace(']','')
        pnl_pct = ((price - entry)/entry)*100 if entry > 0 else 0
        color = "green" if pnl_pct >= 0 else "red"
        # Icono pequeño para la posición
        active_positions_by_event[b_clean] = f"[{color}]●{b_clean}[/]"

    current_equity += invested_val
    total_pnl = current_equity - 1000

    header = Panel(
        f"💵 ${STATE.portfolio['cash']:.0f} | 🏛️ ${current_equity:.0f} | 🚀 [bold {'green' if total_pnl>0 else 'red'}]{total_pnl:+.1f}[/]",
        style="white on blue", box=box.SIMPLE, padding=(0,1)
    )

    # --- PREPARACIÓN DE PANELES ---
    groups = {} 
    for tid, info in STATE.market_map.items():
        key = info['group_key']
        if key not in groups: groups[key] = []
        groups[key].append(tid)

    # Ordenar por urgencia
    def sort_key(k):
        if k in STATE.tweet_data: return STATE.tweet_data[k]['hours']
        return 9999
        
    sorted_keys = sorted(groups.keys(), key=sort_key)
    active_keys = sorted_keys[:9] # LÍMITE 9 PARA EL GRID 3x3

    panels_list = []
    
    for key in active_keys:
        t_data = STATE.tweet_data.get(key)
        if t_data:
            # Título recortado
            clean_title = t_data['title'].replace("Elon Musk", "").replace("Tweets", "").strip()
            title_txt = f"[b]{clean_title[:15]}..[/]\n🐦[yellow]{t_data['count']}[/] ⏳{t_data['hours']:.1f}h"
            hours, count = t_data['hours'], t_data['count']
        else:
            title_txt = f"[b]{key[:15]}..[/]\n[dim]Waiting...[/]"
            hours, count = 24, 0

        # Tabla ULTRA ESTRECHA
        table = Table(box=None, padding=(0,0), show_header=True, expand=True)
        table.add_column("Bkt", style="cyan", width=6)
        table.add_column("Bid", justify="right", width=5)
        table.add_column("Ask", justify="right", width=5)
        table.add_column("Fair", style="magenta", justify="right", width=5)
        table.add_column("Edg", justify="right", width=5)

        mkts = sorted(groups[key], key=lambda k: STATE.market_map[k]['min'])
        
        # Predicción
        sims = STATE.brain.predict(hours) if STATE.brain else []
        final_vals = [count + s for s in sims]
        total_sims = len(sims) if sims else 1
        
        buckets_data = [] # Para el histograma
        has_edge = False

        for tid in mkts:
            info = STATE.market_map[tid]
            pr = STATE.prices.get(tid, {'bid':0, 'ask':0})
            
            fair = 0.0
            if sims:
                hits = sum(1 for x in final_vals if info['min'] <= x <= info['max'])
                fair = hits / total_sims
            
            buckets_data.append({'bucket': info['bucket'], 'min': info['min'], 'max': info['max']})

            edge_str = ""
            if pr['ask'] > 0 and fair > pr['ask'] + 0.10: 
                edge_str = f"[green]+{fair-pr['ask']:.1f}[/]"
                has_edge = True
            elif pr['bid'] > 0 and fair < pr['bid'] - 0.10:
                edge_str = f"[red]{pr['bid']-fair:.1f}[/]"
            
            # Marcador de posición
            bucket_label = info['bucket']
            if bucket_label in active_positions_by_event:
                bucket_label = f"[u]{bucket_label}[/]"

            table.add_row(
                bucket_label, 
                f"{pr['bid']:.2f}", 
                f"{pr['ask']:.2f}", 
                f"{fair:.2f}", 
                edge_str
            )
        
        # Histograma Mini debajo de la tabla
        hist_str = render_mini_histogram(buckets_data, final_vals, total_sims)
        
        # Ensamblar contenido del cuadrado
        content = Group(
            Align.center(title_txt),
            Text("─"*20, style="dim"), # Separador fino
            table,
            Text("─"*20, style="dim") if hist_str else Text(""),
            Align.left(hist_str)
        )
        
        border = "green" if has_edge else "dim"
        panels_list.append(Panel(content, border_style=border, expand=True))

    # --- CONSTRUCCIÓN DEL GRID 3x3 ---
    # Rellenamos hasta 9 con paneles vacíos si faltan eventos
    while len(panels_list) < 9:
        panels_list.append(Panel("", border_style="dim"))

    layout = Layout()
    layout.split_column(
        Layout(header, size=3),
        Layout(name="grid_area"),
        Layout(name="footer", size=3)
    )

    # Dividir el área central en 3 filas
    rows = [Layout(name=f"row{i}") for i in range(3)]
    layout["grid_area"].split_column(*rows)

    # Dividir cada fila en 3 columnas y asignar paneles
    panel_idx = 0
    for r in range(3):
        cols = [Layout(name=f"c{r}{c}") for c in range(3)]
        rows[r].split_row(*cols)
        for c in range(3):
            cols[c].update(panels_list[panel_idx])
            panel_idx += 1

    # Logs footer
    footer_txt = " | ".join([log.replace("[white]","").replace("[/]","") for log in list(STATE.logs)[:3]])
    layout["footer"].update(Panel(footer_txt, title="Logs", box=box.SIMPLE))

    return layout

###
# UTILS
###
def normalize_key_universal(title):
    # 1. Todo a minúsculas
    t = title.lower()
    
    # 2. Diccionario de reemplazos (Meses largos -> cortos, y eliminar ruido)
    replacements = {
        "january": "jan", "february": "feb", "march": "mar", "april": "apr",
        "june": "jun", "july": "jul", "august": "aug", "september": "sep",
        "october": "oct", "november": "nov", "december": "dec",
        # Palabras de relleno que estorban
        "from": "", "to": "", "in": "", "until": "", "between": "", 
        "tweets": "", "tweet": "", "#": "",
        # Años (a veces uno lo tiene y el otro no, mejor quitarlos)
        "2025": "", "2026": ""
    }
    
    for old, new in replacements.items():
        t = t.replace(old, new)

    # 3. Eliminar todo lo que no sea letra o número (espacios, guiones, comas)
    clean = re.sub(r'[^a-z0-9]', '', t)
    
    return clean

# ==========================================
# MAIN
# ==========================================
async def main():
    STATE.brain = HawkesBrain()
    scanner = ClobMarketScanner()
    scanner.map_market_ids()
    
    asyncio.create_task(websocket_loop())
    asyncio.create_task(tweet_sensor_loop())
    
    with Live(generate_dashboard(), refresh_per_second=10, screen=True) as live:
      while True:
        live.update(generate_dashboard())
        await asyncio.sleep(0.1) # Bajamos el sueño a 100ms

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass