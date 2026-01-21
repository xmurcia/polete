import requests
import json
import re
from datetime import datetime

class ClobMarketScanner:
    def __init__(self):
        self.gamma_url = "https://gamma-api.polymarket.com/events"
        self.clob_url = "https://clob.polymarket.com/price"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def _get_clob_price(self, token_id, side="sell"):
        """Consulta el precio (Ask) al CLOB."""
        try:
            params = {"token_id": token_id, "side": side}
            resp = self.session.get(self.clob_url, params=params, timeout=2)
            data = resp.json()
            # Si no hay ofertas de venta, el precio puede venir vacío o null
            if 'price' in data and data['price']:
                return float(data['price'])
            return 0.0
        except Exception:
            return 0.0

    def get_active_tweet_markets(self):
        print("🔎 Escaneando eventos (Estrategia Volumen)...", end=" ")
        
        try:
            # CORRECCIÓN: Usamos parámetros oficiales solamente
            params = {
                "limit": 100,           # Traemos suficientes para encontrar a Elon
                "active": "true",
                "closed": "false",
                "archived": "false",
                "order": "volume24hr",  # TRUCO: Los mercados de Elon tienen mucho volumen, saldrán primero
                "ascending": "false"
            }
            
            resp = self.session.get(self.gamma_url, params=params, timeout=5)
            data = resp.json()
            
            valid_events = []
            
            for event in data:
                # 1. FILTRADO MANUAL (Python)
                title = event.get('title', '')
                # Buscamos "Elon" Y "Tweet" (case insensitive)
                if "elon" not in title.lower() or "tweet" not in title.lower():
                    continue

                if not event.get('markets'): continue
                
                markets_data = []
                
                # 2. Procesar Mercados del Evento
                for m in event['markets']:
                    if m['closed'] or not m['acceptingOrders']: continue
                    
                    # Parsing de Buckets (Regex)
                    question = m.get('question', '')
                    range_match = re.search(r'(\d+)-(\d+)', question)
                    open_ended_match = re.search(r'(\d+)\+', question)
                    
                    bucket_name = "Unknown"
                    min_val, max_val = 0, 99999
                    
                    if range_match:
                        min_val = int(range_match.group(1))
                        max_val = int(range_match.group(2))
                        bucket_name = f"{min_val}-{max_val}"
                    elif open_ended_match:
                        min_val = int(open_ended_match.group(1))
                        bucket_name = f"{min_val}+"
                    else:
                        continue # Si no es un bucket numérico, lo saltamos

                    try:
                        # Obtener Token ID para YES
                        token_ids = json.loads(m['clobTokenIds'])
                        yes_token_id = token_ids[0]
                        
                        # Consultar precio al CLOB
                        price = self._get_clob_price(yes_token_id, side="sell")
                        
                        markets_data.append({
                            'bucket': bucket_name,
                            'price': price,
                            'min': min_val,
                            'max': max_val,
                            'id': m['id']
                        })
                    except Exception:
                        continue
                
                # Ordenar buckets
                markets_data.sort(key=lambda x: x['min'])
                
                if len(markets_data) > 1:
                    valid_events.append({
                        'title': title,
                        'id': event['id'],
                        'buckets': markets_data
                    })

            if len(valid_events) > 0:
                print(f"✅ Encontrados {len(valid_events)} eventos válidos.")
                print(json.dumps(valid_events, indent=4))
            
             
            return valid_events

        except Exception as e:
            print(f"❌ Error Scanner: {e}")
            return []
        
if __name__ == "__main__":
    scanner = ClobMarketScanner().get_active_tweet_markets()