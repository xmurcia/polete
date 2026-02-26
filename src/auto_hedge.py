from config import *
from src.utils import detect_event_type

def gestionar_cobertura_final(trader, m_poly, clob_buckets):
    """
    Analiza si estamos expuestos a riesgo de 'Undershooting' (quedarnos cortos)
    O de 'Overshooting' (pasarnos) en las últimas horas del evento.
    Crea un 'Straddle' defensivo automáticamente.

    ADAPTATIVO: Comportamiento diferente para eventos cortos vs largos
    - Cortos (<72h): Usa ritmo actual × 1.3, activa últimas 6h
    - Largos (≥72h): Usa tasas fijas 3.5/1.0, activa últimas 12h
    """
    try:
        p_hours_left = m_poly.get('hours', HEDGE_MAX_TIME_HOURS)
        current_count = m_poly.get('count', 0)

        # 1. Detect event type (short vs long)
        event_type, bucket_size = detect_event_type(m_poly, clob_buckets)

        # 2. Determine activation threshold based on event type
        if event_type == 'short':
            max_time = HEDGE_MAX_TIME_HOURS_SHORT  # 6h for short events
        else:
            max_time = HEDGE_MAX_TIME_HOURS  # 12h for long events

        if p_hours_left > max_time or p_hours_left < HEDGE_MIN_TIME_HOURS:
            return

        # 3. Calculate projection rates based on event type
        if event_type == 'short':
            # SHORT EVENTS: Use adaptive rates based on current rhythm
            # Calculate current rate from count and estimated hours elapsed
            daily_avg = m_poly.get('daily_avg', 45)
            hourly_rate_avg = daily_avg / 24.0

            # Estimate hours elapsed
            hours_elapsed = current_count / hourly_rate_avg if hourly_rate_avg > 0 else 24

            # Current actual rate
            current_rate = current_count / hours_elapsed if hours_elapsed > 0 else 2.0

            # Apply multipliers with caps
            ceiling_rate = min(current_rate * HEDGE_CEILING_MULTIPLIER_SHORT, HEDGE_CEILING_CAP_SHORT)
            floor_rate = max(current_rate * HEDGE_FLOOR_MULTIPLIER_SHORT, HEDGE_FLOOR_CAP_SHORT)

        else:
            # LONG EVENTS: Use fixed rates (more unpredictable)
            ceiling_rate = HEDGE_CEILING_RATE
            floor_rate = HEDGE_FLOOR_RATE

        # 4. Analyze own positions
        my_buckets = []
        for pid, pos in trader.portfolio['positions'].items():
            clean_market = ''.join(filter(str.isalnum, pos['market'].lower()))
            clean_poly = ''.join(filter(str.isalnum, m_poly['title'].lower()))

            if clean_poly in clean_market or clean_market in clean_poly:
                try:
                    if "+" in pos['bucket']:
                        min_v = int(pos['bucket'].replace("+",""))
                        max_v = BUCKET_MAX_OPEN_ENDED
                    else:
                        parts = pos['bucket'].split("-")
                        min_v = int(parts[0])
                        max_v = int(parts[1])
                    my_buckets.append({'bucket': pos['bucket'], 'min': min_v, 'max': max_v})
                except:
                    pass

        if not my_buckets:
            return

        # Sort to find fortress limits
        my_buckets.sort(key=lambda x: x['min'])
        lowest_owned = my_buckets[0]
        highest_owned = my_buckets[-1]

        # 5. FIX 4: Check if already has natural coverage (consecutive buckets)
        # If we have neighbors, skip hedge
        has_neighbor_below = False
        has_neighbor_above = False

        for b in clob_buckets:
            # Check if there's a bucket just below our lowest
            if b['max'] == lowest_owned['min'] - 1:
                # Check if we own it
                for owned in my_buckets:
                    if owned['max'] == b['max']:
                        has_neighbor_below = True
                        break

            # Check if there's a bucket just above our highest
            if b['min'] == highest_owned['max'] + 1:
                # Check if we own it
                for owned in my_buckets:
                    if owned['min'] == b['min']:
                        has_neighbor_above = True
                        break

        # ---------------------------------------------------------
        # A) PROTECCIÓN ASIMÉTRICA CONTRA CAÍDA (FLOOR HEDGE)
        # ---------------------------------------------------------
        projected_floor = current_count + (floor_rate * p_hours_left)

        if projected_floor < lowest_owned['min'] and not has_neighbor_below:
            # Buscamos el bucket lejano que contiene la proyección de caída
            _ejecutar_hedge(
                trader, m_poly, clob_buckets,
                target_match_func=lambda b: b['min'] <= projected_floor <= b['max'] and b['max'] < lowest_owned['min'],
                reason_tag=f"Floor Risk ({event_type}, proj {projected_floor:.0f})",
                p_hours_left=p_hours_left,
                current_count=current_count
            )

        # ---------------------------------------------------------
        # B) PROTECCIÓN ASIMÉTRICA CONTRA RAGE MODE (CEILING HEDGE)
        # ---------------------------------------------------------
        projected_ceiling = current_count + (ceiling_rate * p_hours_left)

        if highest_owned['max'] < BUCKET_MAX_OPEN_ENDED and projected_ceiling > highest_owned['max'] and not has_neighbor_above:
            # Buscamos el bucket lejano que contiene la proyección de locura
            _ejecutar_hedge(
                trader, m_poly, clob_buckets,
                target_match_func=lambda b: b['min'] <= projected_ceiling <= b['max'] and b['min'] > highest_owned['max'],
                reason_tag=f"Ceiling Risk ({event_type}, proj {projected_ceiling:.0f})",
                p_hours_left=p_hours_left,
                current_count=current_count
            )

    except Exception as e:
        print(f"Error Auto-Hedge: {e}")

def _ejecutar_hedge(trader, m_poly, clob_buckets, target_match_func, reason_tag, p_hours_left, current_count):
    """Función auxiliar para ejecutar la orden de hedge si existe el bucket"""
    candidate = None
    for b in clob_buckets:
        if target_match_func(b):
            candidate = b
            break

    if candidate:
        # Verificar que NO lo tenemos ya (doble check)
        is_owned = False
        for pos in trader.portfolio['positions'].values():
            if pos['bucket'] == candidate['bucket'] and m_poly['title'] in pos['market']:
                is_owned = True
                break

        if not is_owned:
            ask_price = candidate.get('ask', 999)
            # Filtro de precio: No compramos seguros caros automáticamente
            if HEDGE_MIN_PRICE <= ask_price <= HEDGE_MAX_PRICE:
                print(f"    🛡️ AUTO-HEDGE ACTIVADO: Comprando {candidate['bucket']} @ ${ask_price:.2f} ({reason_tag})")
                trader.execute(
                    m_poly['title'],
                    candidate['bucket'],
                    "BUY_HEDGE",
                    ask_price,
                    f"Auto-Hedge: {reason_tag}",
                    strategy_tag="HEDGE",
                    hours_left=p_hours_left,
                    tweet_count=current_count,
                    market_consensus=None,
                    entry_z_score=None,
                    tick_size=candidate.get('tick_size', '0.01')
                )
