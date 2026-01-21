import time
import numpy as np
from collections import deque

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
                            'price': bid,
                            'change': (bid/avg_bid) - 1.0
                        })
                        
        return alerts