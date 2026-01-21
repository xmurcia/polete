import asyncio
import websockets
import json
import requests
import re

class ClobMarketScanner:
    def __init__(self):
        self.gamma_url = "https://gamma-api.polymarket.com/events"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_market_map(self):
        """
        Genera un mapa completo:
        TOKEN_ID -> { 'evento': 'Nombre Evento', 'bucket': 'Nombre Bucket', 'event_id': '123' }
        """
        print("🔎 [SCANNER] Indexando eventos y mercados...", end=" ")
        
        try:
            params = {
                "limit": 50,
                "active": "true",
                "closed": "false",
                "archived": "false",
                "order": "volume24hr", 
                "ascending": "false"
            }
            
            resp = self.session.get(self.gamma_url, params=params, timeout=5)
            data = resp.json()
            
            token_map = {}
            
            for event in data:
                title = event.get('title', '')
                event_id = event.get('id', '')

                # Filtro: Solo cosas de Elon y Tweets
                if "elon" not in title.lower() or "tweet" not in title.lower():
                    continue

                if not event.get('markets'): continue
                
                # Procesamos los mercados de este evento
                for m in event['markets']:
                    if m['closed'] or not m['acceptingOrders']: continue
                    
                    question = m.get('question', '')
                    
                    # Regex para identificar el bucket
                    range_match = re.search(r'(\d+)-(\d+)', question)
                    open_ended_match = re.search(r'(\d+)\+', question)
                    
                    bucket_name = "Otro"
                    if range_match:
                        bucket_name = f"[{range_match.group(1)}-{range_match.group(2)}]"
                    elif open_ended_match:
                        bucket_name = f"[{open_ended_match.group(1)}+]"
                    else:
                        bucket_name = "[General]"

                    try:
                        # Extraemos el Token ID del "YES"
                        token_ids = json.loads(m['clobTokenIds'])
                        yes_token_id = token_ids[0]
                        
                        # --- CLAVE: Guardamos toda la info relacionada ---
                        token_map[yes_token_id] = {
                            "evento": title,         # Ej: "Elon Musk Tweets Jan 14"
                            "bucket": bucket_name,   # Ej: "[10-15]"
                            "event_id": event_id     # Para logs internos
                        }
                        
                    except Exception:
                        continue
            
            if token_map:
                print(f"✅ Indexados {len(token_map)} mercados de interés.")
                return token_map
            else:
                print("❌ No se encontraron mercados.")
                return {}

        except Exception as e:
            print(f"❌ Error Scanner: {e}")
            return {}

async def monitorizar_socket():
    # 1. Obtenemos el mapa detallado
    scanner = ClobMarketScanner()
    info_map = scanner.get_market_map() # Diccionario con info completa
    
    ids_a_vigilar = list(info_map.keys())
    
    if not ids_a_vigilar:
        return

    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    print(f"\n🔌 Conectando Websocket para {len(ids_a_vigilar)} mercados...\n")
    
    async with websockets.connect(uri) as ws:
        print(f"{'EVENTO':<35} | {'BUCKET':<8} | {'TIPO':<4} | {'PRECIO':<6} | {'SHARES':<8} | {'VALOR ($)':<8}")
        print("-" * 90)

        # Suscripción
        payload = {"assets_ids": ids_a_vigilar, "type": "market"}
        await ws.send(json.dumps(payload))

        while True:
            try:
                mensaje = await ws.recv()
                data = json.loads(mensaje)
                evento_tipo = data.get("event_type")

                if evento_tipo == "price_change":
                    for cambio in data.get("price_changes", []):
                        asset_id = cambio.get('asset_id') or data.get('asset_id')
                        
                        # Recuperamos la info del evento
                        info = info_map.get(asset_id)
                        if not info: continue # Si no es uno de los nuestros, ignorar

                        # Datos numéricos
                        precio = float(cambio['price'])
                        cantidad = float(cambio['size'])
                        valor_usd = precio * cantidad # <--- CÁLCULO DE DÓLARES
                        
                        lado = "🟢 C" if cambio['side'] == "BUY" else "🔴 V"
                        
                        # Acortamos el nombre del evento para que quepa en pantalla
                        nombre_evento = (info['evento'][:33] + '..') if len(info['evento']) > 33 else info['evento']

                        print(f"{nombre_evento:<35} | {info['bucket']:<8} | {lado} | {precio:.3f}  | {cantidad:>8.1f} | ${valor_usd:>7.2f}")

                elif evento_tipo == "last_trade_price":
                    asset_id = data.get('asset_id')
                    info = info_map.get(asset_id)
                    if info:
                        p = float(data['price'])
                        s = float(data['size'])
                        val = p * s
                        print(f"💰💰 TRADE REAL: {info['evento']} {info['bucket']} -> ${val:.2f} ({s} shares)")

            except websockets.exceptions.ConnectionClosed:
                print("🔴 Desconectado.")
                break
            except Exception:
                pass

if __name__ == "__main__":
    try:
        asyncio.run(monitorizar_socket())
    except KeyboardInterrupt:
        print("\n👋 Fin.")