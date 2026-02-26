import time
from datetime import datetime, timedelta
from config import *

def ejecutar_moonshot_satelite(trader, m_poly, clob_data, p_count, p_avg_hist, p_hours_left, moonshot_cooldowns):
    """
    Estrategia secundaria que opera SOLO al final del ciclo.
    Busca 'Cisnes Negros' al alza (buckets lejanos y baratos).
    """
    try:
        # 1. Filtro de Seguridad Dinámico
        if p_count > 130 or p_hours_left > MOONSHOT_MIN_EVENT_DURATION:
            total_duration = EVENT_DURATION_HOURS_LONG
        elif p_hours_left < EVENT_DURATION_HOURS_VERY_SHORT:
            total_duration = EVENT_DURATION_HOURS_SHORT
        else:
            total_duration = EVENT_DURATION_HOURS_MID

        if total_duration < MOONSHOT_MIN_EVENT_DURATION:
            return  # Evento demasiado corto para moonshots

        min_hours_required = total_duration * MOONSHOT_MIN_TIME_REMAINING_PCT
        if p_hours_left < min_hours_required or p_count < MOONSHOT_MIN_COUNT_THRESHOLD:
            return

        # 2. Revisar Cartera ACTUAL
        current_positions = trader.portfolio.get('positions', [])
        moonshots_count = 0
        moonshot_buckets_ids = []

        base_proj = p_count + (p_avg_hist / HOURS_PER_DAY * p_hours_left)

        for p in current_positions.values():
            clean_market = ''.join(filter(str.isalnum, p['market'].lower()))
            clean_poly = ''.join(filter(str.isalnum, m_poly['title'].lower()))
            if clean_poly in clean_market or clean_market in clean_poly:
                if p.get('strategy_tag') == 'MOONSHOT':
                    moonshots_count += 1
                    moonshot_buckets_ids.append(p['bucket'])

        if moonshots_count >= MAX_MOONSHOTS_CONCURRENT: return

        rage_target = base_proj + MOONSHOT_RAGE_TARGET_BUFFER
        max_reasonable_distance = p_avg_hist * MOONSHOT_MAX_DISTANCE_MULTIPLIER

        candidates = []
        for b in clob_data.get('buckets', []):
            if b.get('buckets'): return
            ask = b.get('ask', 0)

            if not (MOONSHOT_MIN_PRICE <= ask <= MOONSHOT_MAX_PRICE): continue
            if b['min'] < base_proj: continue
            if b['min'] - base_proj > max_reasonable_distance: continue
            if b['bucket'] in moonshot_buckets_ids: continue

            if b['bucket'] in moonshot_cooldowns:
                if datetime.now() < moonshot_cooldowns[b['bucket']]: continue
                else: del moonshot_cooldowns[b['bucket']]

            if b['max'] >= BUCKET_MAX_OPEN_ENDED: mid = b['min'] + MOONSHOT_DEFAULT_MID_OFFSET
            else: mid = (b['min'] + b['max']) / 2

            dist = abs(mid - rage_target)
            if dist < MOONSHOT_CLUSTER_DISTANCE:
                candidates.append({'bucket': b, 'dist': dist, 'ask': ask})

        if candidates:
            candidates.sort(key=lambda x: x['dist'])
            best = candidates[0]
            print(f"    🛰️ MOONSHOT OPORTUNIDAD: {best['bucket']['bucket']} @ ${best['ask']:.2f}")
            trader.execute(m_poly['title'], best['bucket']['bucket'], "BUY", best['ask'], "Moonshot V33",
                          strategy_tag="MOONSHOT", hours_left=p_hours_left, tweet_count=p_count,
                          market_consensus=None, entry_z_score=None,
                          tick_size=best['bucket'].get('tick_size', '0.01'))
            time.sleep(1.0)

    except Exception as e: pass
