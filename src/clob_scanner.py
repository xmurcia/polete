import time
import json
import re
import requests
from config import *

class ClobMarketScanner:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT_CHROME,
            "Accept": "application/json",
            "Referer": "https://polymarket.com/"
        })
        self.bulk_prices_url = API_CONFIG['clob_url']

    def get_market_prices(self):
        print("   🔎 Escaneando Order Book (Modo Bulk V9)...", end=" ")
        t_start = time.time()
        try:
            params = {
                "limit": API_MARKET_LIMIT, "active": MARKET_STATUS_ACTIVE, "closed": MARKET_STATUS_CLOSED,
                "archived": MARKET_STATUS_ARCHIVED, "order": MARKET_SORT_BY, "ascending": MARKET_SORT_ASCENDING
            }
            resp = self.session.get(API_CONFIG['gamma_url'], params=params, timeout=API_TIMEOUT_SECONDS)
            data = resp.json()

            market_structure = []
            tokens_to_fetch = []

            for event in data:
                title = event.get('title', '').lower()
                if MARKET_FILTER_KEYWORD_1 not in title or MARKET_FILTER_KEYWORD_2 not in title: continue
                if not event.get('markets'): continue

                buckets_list = []
                for m in event['markets']:
                    if m.get('closed') is True: continue
                    q = m.get('question', '')
                    r_match = re.search(r'(\d+)-(\d+)', q)
                    o_match = re.search(r'(\d+)\+', q)

                    min_v, max_v, b_name = 0, BUCKET_MAX_OPEN_ENDED, "Unknown"
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
                        ts = m.get('orderPriceMinTickSize')
                        tick_size = f"{float(ts):g}" if ts is not None else None
                        buckets_list.append({'bucket': b_name, 'min': min_v, 'max': max_v, 'token': yes_token, 'tick_size': tick_size})
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
                        'ask': precios.get('sell', 0.0), 'bid': precios.get('buy', 0.0),
                    'tick_size': b.get('tick_size')
                    })
                final_data.append({'title': mkt['title'], 'buckets': clean_buckets})

            elapsed = time.time() - t_start
            print(f"✅ ({elapsed:.2f}s)")
            return final_data
        except Exception as e:
            print(f"❌ Error Scanner: {e}")
            return []
