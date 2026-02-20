from config import *

def gestionar_cobertura_final(trader, m_poly, clob_buckets):
    """
    Analiza si estamos expuestos a riesgo de 'Undershooting' (quedarnos cortos)
    O de 'Overshooting' (pasarnos) en las últimas 24 horas.
    Crea un 'Straddle' defensivo automáticamente.
    """
    try:
        # 1. Solo aplica en 'End-Game' (últimas 24h)
        p_hours_left = m_poly.get('hours', HEDGE_MAX_TIME_HOURS)
        current_count = m_poly.get('count', 0)

        if p_hours_left > HEDGE_MAX_TIME_HOURS or p_hours_left < HEDGE_MIN_TIME_HOURS: return

        # 2. Analizar Posiciones PROPIAS
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
                except: pass

        if not my_buckets: return

        # Ordenar para encontrar los límites de nuestra "Fortaleza"
        my_buckets.sort(key=lambda x: x['min'])
        lowest_owned = my_buckets[0]
        highest_owned = my_buckets[-1]

        # ---------------------------------------------------------
        # A) PROTECCIÓN CONTRA CAÍDA (FLOOR HEDGE)
        # ---------------------------------------------------------
        pessimistic_rate = HEDGE_FLOOR_RATE
        projected_floor = current_count + (pessimistic_rate * p_hours_left)

        if projected_floor < lowest_owned['min']:
            # Estamos desnudos por abajo. Buscar vecino inferior.
            target_max = lowest_owned['min'] - 1
            _ejecutar_hedge(trader, m_poly, clob_buckets, target_match_func=lambda b: b['max'] == target_max,
                           reason_tag=f"Gap Risk (Floor {projected_floor:.0f})")

        # ---------------------------------------------------------
        # B) PROTECCIÓN CONTRA RAGE MODE (CEILING HEDGE)
        # ---------------------------------------------------------
        optimistic_rate = HEDGE_CEILING_RATE
        projected_ceiling = current_count + (optimistic_rate * p_hours_left)

        if highest_owned['max'] < BUCKET_MAX_OPEN_ENDED and projected_ceiling > highest_owned['max']:
            # Estamos desnudos por arriba. Buscar vecino superior.
            target_min = highest_owned['max'] + 1
            _ejecutar_hedge(trader, m_poly, clob_buckets, target_match_func=lambda b: b['min'] == target_min,
                           reason_tag=f"Rage Risk (Ceiling {projected_ceiling:.0f})")

    except Exception as e:
        print(f"Error Auto-Hedge: {e}")

def _ejecutar_hedge(trader, m_poly, clob_buckets, target_match_func, reason_tag):
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
                is_owned = True; break

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
                    strategy_tag="HEDGE"
                )
