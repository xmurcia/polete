from collections import deque
from config import *

class MarketPanicSensor:
    def __init__(self, sensitivity=PANIC_SENSITIVITY):
        self.history = {}; self.sensitivity = sensitivity; self.window_size = PANIC_WINDOW_SIZE
    def analyze(self, market_data):
        alerts = []
        for m in market_data:
            for b in m['buckets']:
                key = f"{m['title']}|{b['bucket']}"
                if key not in self.history: self.history[key] = {'asks': deque(maxlen=PANIC_WINDOW_SIZE), 'bids': deque(maxlen=PANIC_WINDOW_SIZE)}
                h = self.history[key]
                h['asks'].append(b['ask']); h['bids'].append(b['bid'])
                if len(h['asks']) >= PANIC_MIN_HISTORY_LENGTH:
                    avg_a = sum(h['asks'])/len(h['asks']); avg_b = sum(h['bids'])/len(h['bids'])
                    if avg_a > PANIC_MIN_ASKS_FOR_PUMP and b['ask'] > (avg_a * self.sensitivity):
                        alerts.append({'type': 'PUMP', 'market_title': m['title'], 'bucket': b['bucket'], 'price': b['ask'], 'min': b.get('min',0)})
                    if avg_b > PANIC_MIN_BIDS_FOR_DUMP and b['bid'] < (avg_b / self.sensitivity):
                        alerts.append({'type': 'DUMP', 'market_title': m['title'], 'bucket': b['bucket'], 'price': b['bid'], 'min': b.get('min',0)})
        return alerts
