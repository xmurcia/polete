import asyncio
import aiohttp
import json
import os
import time
import re
import numpy as np
from datetime import datetime, timezone
from collections import deque

# Librerías visuales
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich import box
from rich.align import Align
from rich.text import Text

# ==========================================
# CONFIGURACIÓN
# ==========================================
DATA_DIR = "brain_data"
PORTFOLIO_FILE = "portfolio_turbo.json"

API_CONFIG = {
    'base_url': "https://xtracker.polymarket.com/api",
    'gamma_url': "https://gamma-api.polymarket.com/events",
    'clob_url': "https://clob.polymarket.com/price",
    'user': "elonmusk"
}

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# ==========================================
# 1. CEREBRO IA (Hawkes Process)
# ==========================================
class HawkesBrain:
    def __init__(self):
        # Parámetros optimizados previamente
        self.params = {'mu': 0.34, 'alpha': 3.16, 'beta': 3.71}

    def predict(self, current_count, hours_left):
        """Genera 500 simulaciones de cuántos tweets FALTAN por llegar"""
        mu, a, b = self.params.values()
        sims = []
        
        # Como no tenemos el historial completo en memoria en esta versión asíncrona,
        # usamos una simulación simplificada basada en la intensidad base (mu)
        # y un factor de aleatoriedad para velocidad.
        
        for _ in range(500):
            t = 0
            new_tweets = 0
            # Simulación Monte Carlo simple
            while t < hours_left:
                # Intensidad base
                step = -np.log(np.random.uniform()) / (mu * 1.5) # Factor 1.5 por actividad reciente
                t += step
                if t < hours_left:
                    new_tweets += 1
            sims.append(current_count + new_tweets)
            
        return sims

# ==========================================
# 2. ESTADO GLOBAL
# ==========================================
class BotState:
    def __init__(self):
        self.active_event = None    # {title, count, hours}
        self.market_ids = []        # Lista de buckets
        self.prices = {}            # {token_id: {'bid': 0, 'ask': 0}}
        self.logs = deque(maxlen=6)
        self.portfolio = {"cash": 1000.0, "positions": {}}
        self.brain = HawkesBrain()
        self.last_update = time.time()

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.appendleft(f"[{ts}] {msg}")

STATE = BotState()

# ==========================================
# 3. VISUALIZACIÓN (Histograma ASCII)
# ==========================================
def render_mini_histogram(buckets, sims, total_sims):
    if not sims: return "[dim]Calculando...[/]"
    lines = []
    max_prob = 0
    bucket_probs = []
    
    # Calcular probabilidades
    for b in buckets:
        hits = sum(1 for x in sims if b['min'] <= x <= b['max'])
        prob = hits / total_sims
        if prob > 0.005: # Filtro de ruido
            bucket_probs.append({'label': b['bucket'], 'prob': prob})
            if prob > max_prob: max_prob = prob
            
    # Renderizar barras
    for item in bucket_probs:
        bar_len = int((item['prob'] / max_prob) * 18) if max_prob > 0 else 0
        bar = "█" * bar_len
        color = "yellow" if item['prob'] == max_prob else "white"
        pct = f"{item['prob']:.1%}"
        lines.append(f"[{color}]{item['label']:<6} {bar} {pct}[/]")
            
    return "\n".join(lines)

# ==========================================
# 4. SCANNER ASÍNCRONO (Network)
# ==========================================
async def fetch_price_robust(session, token_id, side):
    """Consulta segura al CLOB. Devuelve (id, side, price)"""
    try:
        # side='buy' -> devuelve el BID más alto (Precio de venta inmediata)
        # side='sell' -> devuelve el ASK más bajo (Precio de compra inmediata)
        params = {"token_id": token_id, "side": side}
        async with session.get(API_CONFIG['clob_url'], params=params) as r:
            data = await r.json()
            price = float(data['price']) if 'price' in data else 0.0
            return token_id, side, price
    except:
        return token_id, side, 0.0

async def update_prices_loop():
    """Bucle ultra-rápido de actualización de precios"""
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            if STATE.market_ids:
                tasks = []
                # Lanzar peticiones dobles (Bid y Ask) para cada mercado
                for m in STATE.market_ids:
                    tid = m['yes_token']
                    tasks.append(fetch_price_robust(session, tid, "buy"))  # Get BID
                    tasks.append(fetch_price_robust(session, tid, "sell")) # Get ASK
                
                # FUEGO PARALELO
                results = await asyncio.gather(*tasks)
                
                # Procesar resultados
                for tid, side, price in results:
                    if tid not in STATE.prices: STATE.prices[tid] = {'bid': 0.0, 'ask': 0.0}
                    key = 'bid' if side == 'buy' else 'ask'
                    STATE.prices[tid][key] = price
                
                STATE.last_update = time.time()
                await asyncio.sleep(0.2) # Pausa mínima técnica
            else:
                await asyncio.sleep(1)

async def structure_scanner_loop():
    """Bucle lento: Busca NUEVOS mercados o eventos"""
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            try:
                # 1. Obtener Eventos de Elon
                params = {"limit": 20, "active": "true", "closed": "false", "order": "volume24hr", "ascending": "false"}
                async with session.get(API_CONFIG['gamma_url'], params=params) as r:
                    data = await r.json()
                
                current_count = STATE.active_event['count'] if STATE.active_event else 0
                temp_markets = []

                for event in data:
                    title = event.get('title', '').lower()
                    if "elon" not in title or "tweet" not in title: continue

                    for m in event['markets']:
                        if m['closed']: continue
                        
                        # Regex para buckets
                        q = m.get('question', '')
                        r_match = re.search(r'(\d+)-(\d+)', q)
                        o_match = re.search(r'(\d+)\+', q)
                        
                        min_v, max_v, label = 0, 99999, ""
                        if r_match:
                            min_v, max_v = int(r_match.group(1)), int(r_match.group(2))
                            label = f"{min_v}-{max_v}"
                        elif o_match:
                            min_v = int(o_match.group(1))
                            label = f"{min_v}+"
                        else: continue

                        # Solo buckets relevantes (futuros)
                        if max_v < current_count: continue

                        try:
                            t_ids = json.loads(m['clobTokenIds'])
                            yes_token = t_ids[0]
                            temp_markets.append({
                                'bucket': label, 'min': min_v, 'max': max_v, 
                                'yes_token': yes_token, 'title': event['title']
                            })
                        except: continue
                
                temp_markets.sort(key=lambda x: x['min'])
                STATE.market_ids = temp_markets # Actualización atómica
                
            except Exception as e:
                STATE.log(f"Error Gamma: {str(e)[:20]}")
            
            await asyncio.sleep(5)

# ==========================================
# 5. SENSOR DE TWEETS (X-Tracker)
# ==========================================
async def tweet_sensor_loop():
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            try:
                # Petición con límite alto para ver todo
                url = f"{API_CONFIG['base_url']}/users/{API_CONFIG['user']}"
                async with session.get(url, params={'limit': 50}) as r:
                    data = await r.json()
                
                trackings = data.get('data', {}).get('trackings', [])
                now = datetime.now(timezone.utc)
                best_evt = None
                min_h = 9999

                for t in trackings:
                    end_str = t['endDate'].replace('Z', '+00:00')
                    hours = (datetime.fromisoformat(end_str) - now).total_seconds() / 3600.0
                    
                    if hours > -24:
                        # Petición detalle conteo
                        async with session.get(f"{API_CONFIG['base_url']}/trackings/{t['id']}?includeStats=true") as rd:
                            d = await rd.json()
                            count = d['data']['stats']['total']
                            
                            # Priorizar el evento que acaba antes
                            if hours > 0 and hours < min_h:
                                min_h = hours
                                best_evt = {'title': t['title'], 'count': count, 'hours': hours}

                if best_evt:
                    if STATE.active_event and best_evt['count'] > STATE.active_event['count']:
                         STATE.log(f"[bold yellow]🚨 TWEET! {STATE.active_event['count']} -> {best_evt['count']}[/]")
                    STATE.active_event = best_evt

            except: pass
            await asyncio.sleep(3)

# ==========================================
# 6. DASHBOARD GENERATOR (Limpio y Completo)
# ==========================================
def generate_dashboard():
    # --- HEADER ---
    latencia = (time.time() - STATE.last_update) * 1000
    color_lat = "green" if latencia < 600 else "red"
    
    evt_text = "Buscando datos..."
    if STATE.active_event:
        evt_text = f"{STATE.active_event['title'][:30]}.. | 🐦 [yellow]{STATE.active_event['count']}[/] | ⏳ {STATE.active_event['hours']:.1f}h"

    header = Panel(
        f"⚡ Ping: [{color_lat}]{latencia:.0f}ms[/] | 💵 Cash: ${STATE.portfolio['cash']:.0f} | {evt_text}",
        style="white on blue", box=box.SIMPLE, padding=(0,1)
    )

    # --- BODY (Split Layout) ---
    # 1. Tabla de Precios (Izquierda)
    table = Table(box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Bucket", style="cyan")
    table.add_column("Bid", justify="right", style="green")
    table.add_column("Ask", justify="right", style="red")
    table.add_column("Spread", justify="right", style="dim")
    table.add_column("Signal", justify="center")

    hist_str = ""

    if STATE.active_event and STATE.market_ids:
        # A. Ejecutar Predicción IA
        sims = STATE.brain.predict(STATE.active_event['count'], STATE.active_event['hours'])
        
        # B. Generar filas
        for m in STATE.market_ids:
            # Obtener precio limpio
            prices = STATE.prices.get(m['yes_token'], {'bid': 0.0, 'ask': 0.0})
            bid, ask = prices['bid'], prices['ask']
            spread = ask - bid
            
            # Cálculo de Fair Value (Probabilidad simple)
            hits = sum(1 for x in sims if m['min'] <= x <= m['max'])
            prob = hits / 500.0
            
            signal = "-"
            # Señal de compra agresiva
            if ask > 0 and prob > (ask + 0.15): 
                signal = "[bold green]BUY[/]"
            
            # Solo mostrar si hay liquidez o probabilidad relevante
            if ask > 0 or bid > 0 or prob > 0.01:
                table.add_row(m['bucket'], f"{bid:.2f}", f"{ask:.2f}", f"{spread:.2f}", signal)
        
        # C. Generar Histograma
        hist_str = render_mini_histogram(STATE.market_ids, sims, 500)

    # 2. Histograma Panel (Derecha)
    hist_panel = Panel(hist_str, title="Probabilidades (IA)", border_style="yellow")

    body = Layout()
    body.split_row(
        Layout(table, ratio=2),
        Layout(hist_panel, ratio=1)
    )

    # --- FOOTER (Logs) ---
    logs = "\n".join(STATE.logs)
    
    # --- ENSAMBLAJE FINAL ---
    layout = Layout()
    layout.split_column(
        Layout(header, size=3),
        Layout(body),
        Layout(Panel(logs, title="Logs", border_style="dim"), size=6)
    )
    return layout

# ==========================================
# MAIN LOOP
# ==========================================
async def main():
    print("🚀 Iniciando ELON-BOT V3 (Async Turbo)...")
    
    # Lanzar trabajadores en segundo plano
    asyncio.create_task(tweet_sensor_loop())
    asyncio.create_task(structure_scanner_loop())
    asyncio.create_task(update_prices_loop())
    
    # Bucle de Interfaz Gráfica
    with Live(generate_dashboard(), refresh_per_second=10, screen=True) as live:
        while True:
            live.update(generate_dashboard())
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass