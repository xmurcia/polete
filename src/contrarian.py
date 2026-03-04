from datetime import datetime, timedelta
from scipy.stats import norm
from config import (
    MIN_PRICE_ENTRY,
    BUCKET_MAX_OPEN_ENDED, BUCKET_OPEN_ENDED_MID_OFFSET,
    WARMUP_MIN_TWEETS_LONG, WARMUP_MIN_TWEETS_SHORT,
    MOONSHOT_MIN_EVENT_DURATION,
    SIGMA_DECAY_BASE_HOURS, SIGMA_DECAY_FACTOR_MIN, SIGMA_DECAY_FACTOR_MAX,
    CONTRARIAN_MAX_Z, CONTRARIAN_MIN_EDGE, CONTRARIAN_MIN_FAIR,
    CONTRARIAN_MAX_PRICE_ENTRY, CONTRARIAN_MIN_HEADROOM,
    CONTRARIAN_MIN_HOURS, CONTRARIAN_MAX_CONCURRENT,
    CONTRARIAN_CONSENSUS_TOLERANCE, CONTRARIAN_MAX_DISTANCE_FROM_MEAN,
    COOLDOWN_CONTRARIAN_HOURS,
)


def ejecutar_contrarian(trader, m_poly, m_clob, final_mean, eff_std, consensus,
                        p_count, p_hours_left, p_avg_hist, alerts,
                        stop_loss_cooldowns, contrarian_cooldowns,
                        executed_trades_this_cycle):
    try:
        if p_hours_left < CONTRARIAN_MIN_HOURS:
            return

        min_req = WARMUP_MIN_TWEETS_LONG if p_hours_left > MOONSHOT_MIN_EVENT_DURATION else WARMUP_MIN_TWEETS_SHORT
        if p_count < min_req:
            return

        dump_keys = {
            f"{a['market_title']}|{a['bucket']}"
            for a in alerts if a['type'] == 'DUMP'
        }

        contrarian_count = sum(
            1 for pos in trader.get_portfolio().get('positions', {}).values()
            if pos.get('strategy_tag') == 'CONTRARIAN'
            and pos.get('market', '') == m_poly['title']
        )
        if contrarian_count >= CONTRARIAN_MAX_CONCURRENT:
            return

        for b in m_clob.get('buckets', []):
            if b['max'] < p_count:
                continue

            bid = b.get('bid', 0)
            ask = b.get('ask', 0)
            bucket_key = f"{m_poly['title']}|{b['bucket']}"

            if ask < MIN_PRICE_ENTRY or ask > CONTRARIAN_MAX_PRICE_ENTRY:
                continue

            now = datetime.now()
            if b['bucket'] in stop_loss_cooldowns and stop_loss_cooldowns[b['bucket']] > now:
                continue
            if b['bucket'] in contrarian_cooldowns and contrarian_cooldowns[b['bucket']] > now:
                continue

            pos_id = f"{m_poly['title']}|{b['bucket']}"
            if pos_id in trader.get_portfolio().get('positions', {}):
                continue

            bucket_headroom = b['max'] - p_count
            if bucket_headroom < CONTRARIAN_MIN_HEADROOM:
                continue

            trade_key = (m_poly['title'], b['bucket'], "BUY")
            if trade_key in executed_trades_this_cycle:
                continue

            if b['max'] >= BUCKET_MAX_OPEN_ENDED:
                mid = b['min'] + BUCKET_OPEN_ENDED_MID_OFFSET
            else:
                mid = (b['min'] + b['max']) / 2

            decay_factor = (p_hours_left / SIGMA_DECAY_BASE_HOURS) ** 0.5
            decay_factor = max(SIGMA_DECAY_FACTOR_MIN, min(SIGMA_DECAY_FACTOR_MAX, decay_factor))
            decayed_std = eff_std * decay_factor
            if decayed_std <= 0:
                continue

            z_score = abs(mid - final_mean) / decayed_std
            p_min = norm.cdf(b['min'], final_mean, decayed_std)
            if b['max'] >= BUCKET_MAX_OPEN_ENDED:
                fair = 1.0 - p_min
            else:
                fair = norm.cdf(b['max'] + 1, final_mean, decayed_std) - p_min
            edge = fair - ask

            # Signal A: DUMP alert on this bucket
            is_dump = bucket_key in dump_keys

            # Signal B: Hawkes divergence (model sees value, market doesn't)
            is_hawkes_divergence = (
                z_score <= CONTRARIAN_MAX_Z
                and edge >= CONTRARIAN_MIN_EDGE
                and fair >= CONTRARIAN_MIN_FAIR
            )

            # Signal C: Consensus divergence (market consensus vs bucket price)
            is_consensus_divergence = False
            if consensus is not None:
                model_bucket_distance = abs(mid - final_mean)
                consensus_close_to_model = abs(consensus - final_mean) <= CONTRARIAN_CONSENSUS_TOLERANCE
                bucket_near_mean = model_bucket_distance <= (CONTRARIAN_MAX_DISTANCE_FROM_MEAN * decayed_std)
                is_consensus_divergence = consensus_close_to_model and bucket_near_mean and edge > 0

            signals_fired = sum([is_dump, is_hawkes_divergence, is_consensus_divergence])
            if signals_fired < 2:
                continue

            parts = []
            if is_dump:
                parts.append("DUMP")
            if is_hawkes_divergence:
                parts.append(f"Edge{edge:.2f}")
            if is_consensus_divergence:
                parts.append("ConsDivg")
            reason = f"CONTRARIAN|{'+'.join(parts)}"

            print(f"    🔄 CONTRARIAN: {b['bucket']} @ ${ask:.3f} [{reason}] Z={z_score:.1f}")

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
