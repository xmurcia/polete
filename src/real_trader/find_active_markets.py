#!/usr/bin/env python3
"""Find markets with active orderbooks."""

import requests
import json
from auth import PolyAuth

# Get the event
response = requests.get(
    'https://gamma-api.polymarket.com/events',
    params={
        'limit': 100,
        'active': 'true',
        'closed': 'false',
        'archived': 'false',
        'order': 'volume24hr',
        'ascending': 'false'
    },
    timeout=10
)

events = response.json()
target_event = None

for event in events:
    if event.get('slug') == 'elon-musk-of-tweets-february-13-february-20':
        target_event = event
        break

if target_event:
    markets = target_event.get('markets', [])
    print(f'Event: {target_event.get("title")}')
    print(f'Total markets: {len(markets)}')
    print('\nChecking which markets have active orderbooks...\n')

    auth = PolyAuth()
    client = auth.get_client()

    active_markets = []

    for market in markets:
        question = market.get('question', '')
        clob_token_ids = market.get('clobTokenIds', [])

        if isinstance(clob_token_ids, str):
            clob_token_ids = json.loads(clob_token_ids)

        if clob_token_ids:
            token_id_decimal = clob_token_ids[0]
            token_id = f'0x{hex(int(token_id_decimal))[2:].zfill(64)}'

            try:
                price_data = client.get_price(token_id, 'BUY')
                ask = float(price_data.get('price', 0))

                if ask > 0:
                    active_markets.append({
                        'question': question,
                        'token_id': token_id,
                        'ask': ask
                    })
            except:
                pass

    print(f'Found {len(active_markets)} markets with active orderbooks:\n')
    for m in active_markets:
        print(f'{m["question"][:70]}')
        print(f'  Ask: {m["ask"]*100:.2f}¢ (${m["ask"]:.4f})')
        print(f'  Token: {m["token_id"][:20]}...')
        print()
else:
    print('Event not found')
