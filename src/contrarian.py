from scipy.stats import norm
from config import (
    MIN_PRICE_ENTRY,
    BUCKET_MAX_OPEN_ENDED, BUCKET_OPEN_ENDED_MID_OFFSET,
    SIGMA_DECAY_BASE_HOURS, SIGMA_DECAY_FACTOR_MIN, SIGMA_DECAY_FACTOR_MAX,
    CONTRARIAN_MAX_TWEETS_ENTRY, CONTRARIAN_MIN_HOURS, CONTRARIAN_FLAT_CURVE_MAX_BID,
    CONTRARIAN_MIN_EDGE, CONTRARIAN_MIN_FAIR, CONTRARIAN_MAX_PRICE_ENTRY,
    CONTRARIAN_MIN_HEADROOM, CONTRARIAN_MAX_CONCURRENT,
)


def ejecutar_contrarian(trader, m_poly, m_clob, final_mean, eff_std, consensus,
                        p_count, p_hours_left, stop_loss_cooldowns,
                        executed_trades_this_cycle):
    """Entra en eventos nuevos con curva plana, aguanta hasta resolución."""
    try:
        # Solo en eventos nuevos: pocos tweets y muchas horas restantes
        if p_count > CONTRARIAN_MAX_TWEETS_ENTRY:
            return
        if p_hours_left < CONTRARIAN_MIN_HOURS:
            return

        # Verificar curva plana: ningún bucket domina aún
        all_buckets = m_clob.get('buckets', [])
        max_bid = max((b.get('bid', 0) for b in all_buckets), default=0)
        if max_bid >= CONTRARIAN_FLAT_CURVE_MAX_BID:
            return

        # Cachear portfolio una sola vez
        positions = trader.get_portfolio().get('positions', {})

        # Cuántas posiciones CONTRARIAN ya tenemos en este mercado
        contrarian_count = sum(
            1 for pos in positions.values()
            if pos.get('strategy_tag') == 'CONTRARIAN'
            and pos.get('market', '') == m_poly['title']
        )
        if contrarian_count >= CONTRARIAN_MAX_CONCURRENT:
            return

        # Hoist: decay es constante para todo el mercado (p_hours_left no cambia por bucket)
        decay_factor = (p_hours_left / SIGMA_DECAY_BASE_HOURS) ** 0.5
        decay_factor = max(SIGMA_DECAY_FACTOR_MIN, min(SIGMA_DECAY_FACTOR_MAX, decay_factor))
        decayed_std = eff_std * decay_factor
        if decayed_std <= 0:
            return

        for b in all_buckets:
            if b['max'] < p_count:
                continue

            ask = b.get('ask', 0)
            if ask < MIN_PRICE_ENTRY or ask > CONTRARIAN_MAX_PRICE_ENTRY:
                continue

            if b['bucket'] in stop_loss_cooldowns:
                continue

            pos_id = f"{m_poly['title']}|{b['bucket']}"
            if pos_id in positions:
                continue

            trade_key = (m_poly['title'], b['bucket'], "BUY")
            if trade_key in executed_trades_this_cycle:
                continue

            if b['max'] - p_count < CONTRARIAN_MIN_HEADROOM:
                continue

            if b['max'] >= BUCKET_MAX_OPEN_ENDED:
                mid = b['min'] + BUCKET_OPEN_ENDED_MID_OFFSET
            else:
                mid = (b['min'] + b['max']) / 2

            p_min = norm.cdf(b['min'], final_mean, decayed_std)
            if b['max'] >= BUCKET_MAX_OPEN_ENDED:
                fair = 1.0 - p_min
            else:
                fair = norm.cdf(b['max'] + 1, final_mean, decayed_std) - p_min

            edge = fair - ask
            if fair < CONTRARIAN_MIN_FAIR or edge < CONTRARIAN_MIN_EDGE:
                continue

            z_score = abs(mid - final_mean) / decayed_std
            reason = f"CTR|new_event+flat+{edge:.2f}"

            print(f"    🔄 CONTRARIAN: {b['bucket']} @ ${ask:.3f}  edge={edge:.2f}  Z={z_score:.1f}")

            res = trader.execute(
                m_poly['title'], b['bucket'], "BUY", ask, reason,
                strategy_tag='CONTRARIAN', hours_left=p_hours_left,
                tweet_count=p_count, market_consensus=consensus,
                entry_z_score=z_score, tick_size=b.get('tick_size')
            )
            if res:
                executed_trades_this_cycle.add(trade_key)
                contrarian_count += 1
                if contrarian_count >= CONTRARIAN_MAX_CONCURRENT:
                    break

    except Exception as e:
        print(f"[Contrarian] ⚠️ Error: {e}")
