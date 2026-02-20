import requests
import pandas as pd
import dateutil.parser
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *

class PolymarketSensor:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT_SIMPLE})

    def _fetch_tracking_detail(self, t, now):
        try:
            response = self.s.get(f"{API_CONFIG['base_url']}/trackings/{t['id']}?includeStats=true", timeout=API_TIMEOUT_SECONDS).json()
            d = response.get('data', {})
            end_date_str = d.get('endDate') or t.get('endDate')
            hours = 0.0

            if end_date_str:
                try:
                    original_dt = dateutil.parser.isoparse(end_date_str)
                    fixed_end_date = original_dt.replace(hour=END_DATE_FIXED_HOUR, minute=END_DATE_FIXED_MINUTE,
                                                          second=END_DATE_FIXED_SECOND, microsecond=0, tzinfo=timezone.utc)
                    hours = (fixed_end_date - now).total_seconds() / SECONDS_PER_HOUR
                except: pass

            count = d.get('stats', {}).get('total', 0)
            days_elapsed = d.get('stats', {}).get('daysElapsed', 0)
            daily_avg = 0.0
            if days_elapsed > 0: daily_avg = count / days_elapsed

            if hours > -2.0:
                return {
                    'id': t['id'], 'title': t['title'], 'count': count, 'hours': hours,
                    'daily_avg': daily_avg, 'active': True
                }
        except: pass
        return None

    def get_active_counts(self):
        try:
            r = self.s.get(f"{API_CONFIG['base_url']}/users/{API_CONFIG['user']}", timeout=API_TIMEOUT_SECONDS).json()
            trackings = r.get('data', {}).get('trackings', [])
            res = []
            now = datetime.now(timezone.utc)

            candidates = []
            for t in trackings:
                start_str = t.get('startDate'); end_str = t.get('endDate')
                if start_str and end_str:
                    try:
                        end = dateutil.parser.isoparse(end_str)
                        if now <= (end + pd.Timedelta(hours=MARKET_EXPIRATION_TOLERANCE_HOURS)):
                            candidates.append(t)
                            continue
                    except: pass
                if t.get('isActive'): candidates.append(t)

            with ThreadPoolExecutor(max_workers=THREAD_POOL_MAX_WORKERS) as executor:
                futures = {executor.submit(self._fetch_tracking_detail, t, now): t for t in candidates}
                for f in as_completed(futures):
                    result = f.result()
                    if result: res.append(result)
            return res
        except: return []
