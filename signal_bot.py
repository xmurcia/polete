"""
signal_bot.py  —  Sistema autónomo de señales D3 + Paper Trading
=================================================================
Fixes v4:
  - burst monitor: xtracker delta (Nitter eliminado — no funciona desde Railway)
  - Paranoid Treasure: exit ≥80% solo en últimas 48h (deja correr antes)
"""

import os, json, math, time, re, sys, threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

import requests
from dotenv import load_dotenv
load_dotenv()

# ── Rutas ──────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
LOGS_DIR = ROOT / "logs" / "signals"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = LOGS_DIR / "signal_bot_state.json"
PAPER_FILE = LOGS_DIR / "paper_portfolio.json"
LOG_FILE   = LOGS_DIR / "signal_bot.log"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── DB ──────────────────────────────────────────────────────────────────
try:
    import database as _db
    import psycopg2.extras as _pge
    _DB_OK = _db._pool is not None
    print("✅ [DB] PostgreSQL" if _DB_OK else "ℹ️  [DB] JSON local")
except Exception as _e:
    _db, _DB_OK, _pge = None, False, None
    print(f"ℹ️  [DB] No disponible ({_e})")

# ── Constantes ──────────────────────────────────────────────────────────
HIST_MEAN       = 343
HIST_SIGMA      = 72
CAPITAL_INIT    = 100.0
RISK_PER_RANGE  = 0.08
POLL_MINUTES    = 15
BURST_POLL_SEC        = 120     # burst monitor cada 2 min
BURST_INTER_TWEET_MIN = 5.0    # minutos entre tweets para considerar burst
BURST_SORPRESA_MULT   = 2.5    # multiplicador sobre AVG histórico de esa hora
BURST_HORAS_MIN       = 2      # horas consecutivas rápidas para considerar sostenido
BURST_SCORE_MIN       = 3      # score mínimo para disparar alerta (sobre 6)
BURST_TIMING_MIN_PCT  = 0.35   # no alertar entrada si evento <35% completado
BURST_LOW_PRICE_THRESHOLD = 0.10  # por debajo de este precio, señal más fiable
XTRACKER_API    = "https://xtracker.polymarket.com/api"
HDR             = {"User-Agent": "Mozilla/5.0"}
MONTH_NAMES     = ["january","february","march","april","may","june",
                   "july","august","september","october","november","december"]

AVG_POR_HORA = {
    0: 1.2, 1: 1.2, 2: 5.0, 3: 3.6, 4: 0.3, 5: 0.0,
    6: 0.0, 7: 1.3, 8: 3.5, 9: 4.3, 10: 0.3, 11: 2.5,
    12: 2.0, 13: 0.8, 14: 2.2, 15: 1.6, 16: 0.4, 17: 0.0,
    18: 2.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.8, 23: 1.0
}


# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════

def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════

class Telegram:
    def __init__(self):
        self.token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.ok      = bool(self.token and self.chat_id)
        log(f"[TG] {'✅ activado' if self.ok else '⚠️  desactivado'}")

    def send(self, msg: str, silent=False) -> bool:
        if not self.ok:
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": msg,
                      "parse_mode": "HTML", "disable_notification": silent},
                timeout=10)
            return r.status_code == 200
        except Exception as e:
            log(f"[TG] Error: {e}")
            return False

    def signal_entry(self, label, day_n, ranges, prices, proj, ev, confidence):
        cost = sum(prices.get(r, 0.05) for r in ranges)
        roi  = (1 / cost - 1) * 100 if cost > 0 else 0
        lines = [f"📊 <b>SEÑAL D{day_n} — {label}</b>", "",
                 f"🎯 <b>SPREAD YES — {len(ranges)} rangos</b>"]
        for rng in ranges:
            lines.append(f"  • YES <b>{rng}</b>  →  {prices.get(rng,0)*100:.1f}¢")
        lines += ["",
                  f"💰 Coste total:    <b>{cost*100:.1f}¢</b> por $1",
                  f"📈 ROI si gana:    <b>+{roi:.0f}%</b>",
                  f"🔮 Proyección:     <b>{proj} tweets</b>",
                  f"⚡ EV esperado:    <b>{ev*100:+.0f}%</b>",
                  f"🎲 Confianza:      <b>{confidence}</b>",
                  "",
                  f"⏰ {datetime.now().strftime('%d/%m %H:%M')} UTC"]
        self.send("\n".join(lines))

    def paper_buy(self, label, rng, price, amount, capital_left):
        self.send(f"📄 <b>PAPER BUY</b>\n"
                  f"  Evento: {label}\n"
                  f"  Rango:  <b>{rng}</b>  @ {price*100:.1f}¢\n"
                  f"  Monto:  <b>${amount:.2f}</b>\n"
                  f"  Capital: ${capital_left:.2f}", silent=True)

    def paper_close(self, label, rng, entry, exit_p, pnl, reason):
        emoji = "✅" if pnl >= 0 else "❌"
        roi   = (exit_p - entry) / entry * 100 if entry > 0 else 0
        self.send(f"{emoji} <b>PAPER CLOSE</b> — {label}\n"
                  f"  Rango:   <b>{rng}</b>\n"
                  f"  {entry*100:.1f}¢ → {exit_p*100:.1f}¢  ({roi:+.0f}%)\n"
                  f"  P&amp;L: <b>${pnl:+.2f}</b>  |  {reason}")

    def portfolio_summary(self, capital, positions, stats):
        roi = (capital - CAPITAL_INIT) / CAPITAL_INIT * 100
        w, l = stats["wins"], stats["losses"]
        wr = 100 * w / (w + l) if (w + l) else 0
        lines = ["📊 <b>Portfolio — SIGNAL PAPER</b>", "",
                 f"  💵 Capital:     <b>${capital:.2f}</b>  (ROI {roi:+.1f}%)",
                 f"  📈 Trades:      {w+l}  ({w}✅ {l}❌)  {wr:.0f}% WR",
                 f"  💰 P&amp;L:     <b>${stats.get('pnl', 0):+.2f}</b>", ""]
        if positions:
            lines.append("  <b>Abiertas:</b>")
            for p in positions[:8]:
                ep = p.get("entry", 0)
                cp = p.get("current_price", ep)
                lines.append(f"    {p['rng']:10}  {ep*100:.0f}¢→{cp*100:.0f}¢  "
                              f"${(cp-ep)*p.get('shares',0):+.2f}")
        lines.append(f"\n  ⏰ {datetime.now().strftime('%d/%m %H:%M')}")
        self.send("\n".join(lines))

    def weekly_summary(self, capital, stats):
        roi = (capital - CAPITAL_INIT) / CAPITAL_INIT * 100
        w, l = stats["wins"], stats["losses"]
        wr = 100 * w / (w + l) if (w + l) else 0
        self.send(f"📅 <b>RESUMEN SEMANAL</b>\n\n"
                  f"  Señales:   {stats['signals']}\n"
                  f"  Trades:    {stats['trades']}\n"
                  f"  Win rate:  {wr:.0f}%  ({w}W / {l}L)\n"
                  f"  P&amp;L:   <b>${stats.get('pnl',0):+.2f}</b>\n"
                  f"  Capital:   <b>${capital:.2f}</b>  (inicio ${CAPITAL_INIT:.0f})\n"
                  f"  ROI total: <b>{roi:+.1f}%</b>\n\n"
                  f"  ⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    def burst_alert(self, label, tweets_hora, inter_tweet_min, avg_hora,
                    horas_consecutivas, pct_completado, proj_normal, rango_normal,
                    proj_burst, rango_burst, spread_status, precio_winner,
                    burst_score, timing_guard_active):
        sorpresa_txt = (f"<b>{tweets_hora/avg_hora:.1f}x</b> sobre AVG histórico ({avg_hora:.1f}/h)"
                        if avg_hora > 0 else "N/A (AVG=0)")
        precio_note = " ← no priceado" if precio_winner < BURST_LOW_PRICE_THRESHOLD else ""
        spread_note = "⚠️ SALIDA RECOMENDADA" if spread_status == "FUERA" else "✓"
        lines = [
            f"🔥 <b>BURST DETECTADO — {label}</b>",
            f"  Tweets esta hora:   <b>{tweets_hora}</b>",
            f"  Inter-tweet:        <b>{inter_tweet_min:.1f} min</b> entre tweets",
            f"  Sorpresa:           {sorpresa_txt}",
            f"  Horas consecutivas: <b>{horas_consecutivas}</b>",
            f"  Progreso evento:    <b>{pct_completado:.0%}</b> completado",
            f"  Proyección normal:  <b>{proj_normal:.0f}</b> tweets → {rango_normal}",
            f"  Proyección burst:   <b>{proj_burst:.0f}</b> tweets → {rango_burst}",
            f"  Spread comprado:    <b>{spread_status}</b> {spread_note}",
            f"  Precio winner:      <b>{precio_winner:.3f}</b>{precio_note}",
            f"  Burst score:        <b>{burst_score}/6</b>",
        ]
        if timing_guard_active:
            lines.append(f"  ⏸ Evento <{BURST_TIMING_MIN_PCT:.0%} completado — solo alerta EXIT activa")
        else:
            lines.append(f"  ⚡ El mercado tardará ~15 min en ajustarse")
        self.send("\n".join(lines))

    def burst_exit_alert(self, label, spread_status, rango_burst, pct_completado):
        self.send(f"⚠️ <b>BURST EXIT — {label}</b>\n"
                  f"  BURST desplaza proyección <b>FUERA</b> del spread → considerar salida\n"
                  f"  Rango burst: {rango_burst}\n"
                  f"  Progreso: {pct_completado:.0%}")

    def scrape_warning(self, label, day_n):
        self.send(f"⚠️ <b>Sin datos tweets D{day_n}</b>\n  {label}", silent=True)

    def bot_start(self, capital):
        self.send(f"🤖 <b>Signal Bot arrancado</b>\n"
                  f"  Capital paper: <b>${capital:.2f}</b>\n"
                  f"  Poll señales:  cada {POLL_MINUTES} min\n"
                  f"  Burst monitor: cada {BURST_POLL_SEC}s (xtracker delta)\n"
                  f"  DB: {'✅ PostgreSQL' if _DB_OK else '📁 JSON local'}\n"
                  f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    def bot_stop(self, capital, stats):
        roi = (capital - CAPITAL_INIT) / CAPITAL_INIT * 100
        w, l = stats["wins"], stats["losses"]
        wr = 100 * w / (w + l) if (w + l) else 0
        self.send(f"🛑 <b>Signal Bot detenido</b>\n"
                  f"  Capital: <b>${capital:.2f}</b>  (ROI {roi:+.1f}%)\n"
                  f"  Win rate: {wr:.0f}%  ({w}W/{l}L)")


# ══════════════════════════════════════════════════════════════════════
# BURST MONITOR  (hilo paralelo — burst score multidimensional)
# 4 dimensiones: frecuencia, sorpresa horaria, sostenibilidad, desplazamiento.
# Heartbeat en log cada 10 ciclos (~20 min) para confirmar que está vivo.
# ══════════════════════════════════════════════════════════════════════

_burst_prev_totals = {}   # tid → (total, datetime)
_burst_hourly_snap = {}   # tid → {hora_utc: tweets_en_esa_hora}
_burst_horas_rapidas = {}  # tid → contador de horas consecutivas rápidas
_burst_last_hora_rapida = {}  # tid → última hora UTC que fue rápida
_last_burst_alert  = {}   # tid → datetime del último alert


def _rango_de(proj, buckets):
    for b in buckets:
        rng = b.get("bucket", "")
        if "-" not in rng:
            continue
        lo, hi = int(rng.split("-")[0]), int(rng.split("-")[1])
        if lo <= proj <= hi:
            return rng
    if buckets:
        maxb = max(buckets, key=lambda b: int(b.get("bucket","0-0").split("-")[-1]))
        return maxb.get("bucket", "???")
    return "???"


def burst_monitor_loop(tg: Telegram, get_events_fn, paper_book=None):
    log("[BURST] Monitor iniciado — burst score multidimensional (6 dims)")
    cycle = 0

    while True:
        try:
            now = datetime.now(tz=timezone.utc)
            hora_actual = now.hour

            r   = requests.get(f"{XTRACKER_API}/users/elonmusk",
                               headers=HDR, timeout=15)
            r.raise_for_status()
            data      = r.json()
            trackings = (data.get("data", {}).get("trackings", [])
                         if isinstance(data.get("data"), dict)
                         else data.get("trackings", []))

            events = get_events_fn()

            for t in trackings:
                tid   = t.get("id", "")
                title = t.get("title", "")
                if not tid or "tweet" not in title.lower():
                    continue

                try:
                    r2    = requests.get(
                        f"{XTRACKER_API}/trackings/{tid}?includeStats=true",
                        headers=HDR, timeout=10)
                    tdata = r2.json().get("data", {})
                    total = int(tdata.get("stats", {}).get("total", 0))
                except Exception:
                    continue

                # Calcular tweets en la hora actual via delta
                snap = _burst_hourly_snap.setdefault(tid, {})
                if tid in _burst_prev_totals:
                    prev_total, prev_ts = _burst_prev_totals[tid]
                    delta = max(0, total - prev_total)
                    if prev_ts.hour == hora_actual:
                        snap[hora_actual] = snap.get(hora_actual, 0) + delta
                    else:
                        snap[hora_actual] = delta
                else:
                    snap[hora_actual] = 0

                _burst_prev_totals[tid] = (total, now)

                tweets_hora = snap.get(hora_actual, 0)
                if tweets_hora < 1:
                    continue

                # ── Burst Score (6 dimensiones) ───────────────────
                burst_score = 0
                inter_tweet_min = 60.0 / tweets_hora if tweets_hora > 0 else 999

                # Dim 1 — Frecuencia (inter-tweet)
                dim1 = False
                if tweets_hora >= 3 and inter_tweet_min < BURST_INTER_TWEET_MIN:
                    burst_score += 1
                    dim1 = True

                # Dim 2 — Sorpresa horaria
                avg_hora = AVG_POR_HORA.get(hora_actual, 1.0)
                if avg_hora > 0 and tweets_hora / avg_hora > BURST_SORPRESA_MULT:
                    burst_score += 1

                # Dim 3 — Sostenibilidad (horas consecutivas rápidas)
                last_rapida = _burst_last_hora_rapida.get(tid)
                if dim1:
                    if last_rapida is not None and hora_actual == (last_rapida + 1) % 24:
                        _burst_horas_rapidas[tid] = _burst_horas_rapidas.get(tid, 1) + 1
                    else:
                        _burst_horas_rapidas[tid] = 1
                    _burst_last_hora_rapida[tid] = hora_actual
                else:
                    _burst_horas_rapidas[tid] = 0

                horas_consecutivas = _burst_horas_rapidas.get(tid, 0)
                if horas_consecutivas >= BURST_HORAS_MIN:
                    burst_score += 1

                # ── Proyecciones ──────────────────────────────────
                ev_match = next((e for e in events
                                 if title.lower() in e.get("label","").lower()
                                 or e.get("label","").lower() in title.lower()), None)
                cum_total = total
                h_rem = ev_match["h_rem"] if ev_match else 0
                buckets = ev_match.get("_buckets", []) if ev_match else []
                prices = ev_match.get("prices", {}) if ev_match else {}

                # pace normal = mediana de días anteriores (del state muskmeter)
                pace_normal = HIST_MEAN / 7  # fallback
                muskmeter_daily = {}
                if ev_match:
                    try:
                        sf = ROOT / "logs" / "signals" / "signal_bot_state.json"
                        if sf.exists():
                            st = json.loads(sf.read_text())
                            muskmeter_daily = st.get("muskmeter", {}).get(ev_match["slug"], {})
                            if muskmeter_daily:
                                vals = sorted(muskmeter_daily.values())
                                pace_normal = vals[len(vals) // 2]
                    except Exception:
                        pass

                horas_restantes = max(0, h_rem)
                proj_normal = cum_total + pace_normal * horas_restantes / 24
                proj_burst  = cum_total + tweets_hora * horas_restantes

                rango_normal = _rango_de(proj_normal, buckets)
                rango_burst  = _rango_de(proj_burst, buckets)

                # ── Refinamiento 1: Timing guard ──────────────────
                pct_completado = cum_total / proj_normal if proj_normal > 0 else 0
                timing_guard_active = pct_completado < BURST_TIMING_MIN_PCT

                if timing_guard_active:
                    log(f"[BURST] timing guard activo — {pct_completado:.0%} completado, "
                        f"mínimo requerido {BURST_TIMING_MIN_PCT:.0%}")

                # ── Dim 4 — Spread check (dentro/fuera del spread comprado, +2/+1/+0)
                spread_status = "N/A"
                has_position = False
                if paper_book and ev_match:
                    slug = ev_match.get("slug", "")
                    owned_ranges = [
                        pos["rng"] for pos_id, pos in paper_book.data.get("positions", {}).items()
                        if pos.get("slug") == slug
                    ]
                    if owned_ranges:
                        has_position = True
                        if rango_burst in owned_ranges:
                            burst_score += 2
                            spread_status = "DENTRO"
                        else:
                            spread_status = "FUERA"
                            # +0, no suma — pero dispara alerta EXIT
                    else:
                        # Sin posición: fallback a lógica anterior (acerca al winner)
                        if rango_burst != rango_normal:
                            burst_score += 1
                        spread_status = "SIN_POS"
                else:
                    # Sin paper_book o sin ev_match: fallback
                    if rango_burst != rango_normal:
                        burst_score += 1
                    spread_status = "SIN_POS"

                # ── Dim 5 — Precio winner bajo (<threshold) ──────
                precio_winner = 0.0
                if prices:
                    precio_winner = prices.get(rango_burst, 0.0)
                    if precio_winner <= 0:
                        # buscar rango más cercano a proj_burst
                        best_dist = float("inf")
                        for rng, p in prices.items():
                            if "-" not in rng:
                                continue
                            mid = (int(rng.split("-")[0]) + int(rng.split("-")[1])) / 2
                            d = abs(mid - proj_burst)
                            if d < best_dist:
                                best_dist = d
                                precio_winner = p

                if precio_winner > 0 and precio_winner < BURST_LOW_PRICE_THRESHOLD:
                    burst_score += 1

                # ── Alerta ────────────────────────────────────────
                cycle += 1
                if cycle % 10 == 0:
                    log(f"[BURST] heartbeat — {title[:40]}  score={burst_score}/6  "
                        f"tw/h={tweets_hora}  pct={pct_completado:.0%}  "
                        f"spread={spread_status}  trackings={len(trackings)}")

                # Alerta EXIT independiente: spread FUERA (incluso con timing guard)
                if spread_status == "FUERA" and has_position:
                    last_alert = _last_burst_alert.get(f"{tid}_exit")
                    cooldown_ok = (last_alert is None or
                                   (now - last_alert).total_seconds() > 1200)
                    if cooldown_ok:
                        _last_burst_alert[f"{tid}_exit"] = now
                        log(f"[BURST] ⚠️ EXIT — proj burst FUERA del spread  {title[:50]}")
                        tg.burst_exit_alert(
                            label=title, spread_status=spread_status,
                            rango_burst=rango_burst, pct_completado=pct_completado)

                # Alerta ENTRY: requiere score >= MIN y timing guard NO activo
                if burst_score >= BURST_SCORE_MIN:
                    last_alert = _last_burst_alert.get(tid)
                    cooldown_ok = (last_alert is None or
                                   (now - last_alert).total_seconds() > 1200)
                    if cooldown_ok:
                        if timing_guard_active:
                            log(f"[BURST] ⏸ score={burst_score}/6 pero timing guard activo "
                                f"({pct_completado:.0%})  {title[:50]}")
                        else:
                            _last_burst_alert[tid] = now
                            log(f"[BURST] 🔥 score={burst_score}/6  tw/h={tweets_hora}  "
                                f"inter={inter_tweet_min:.1f}m  consec={horas_consecutivas}  "
                                f"spread={spread_status}  {title[:50]}")
                        tg.burst_alert(
                            label=title,
                            tweets_hora=tweets_hora,
                            inter_tweet_min=inter_tweet_min,
                            avg_hora=avg_hora,
                            horas_consecutivas=horas_consecutivas,
                            pct_completado=pct_completado,
                            proj_normal=proj_normal,
                            rango_normal=rango_normal,
                            proj_burst=proj_burst,
                            rango_burst=rango_burst,
                            spread_status=spread_status,
                            precio_winner=precio_winner,
                            burst_score=burst_score,
                            timing_guard_active=timing_guard_active)

        except Exception as e:
            log(f"[BURST] Error: {e}")

        time.sleep(BURST_POLL_SEC)


# ══════════════════════════════════════════════════════════════════════
# SCRAPING DE TWEETS
# ══════════════════════════════════════════════════════════════════════

def scrape_tweets(event: dict, state: dict) -> dict:
    """
    Obtiene tweets diarios. MERGE con cache existente (no sobreescribe).
    Returns: dict con todos los días conocidos del evento.
    """
    slug     = event["slug"]
    existing = state.get("muskmeter", {}).get(slug, {})

    # Método 1: muskmeter
    try:
        result = _muskmeter(event)
        if result:
            log(f"[SCRAPE] muskmeter OK: {len(result)} días")
            merged = {**existing, **result}
            return merged
    except Exception as e:
        log(f"[SCRAPE] muskmeter falló: {e}")

    # Método 2: xtracker snapshot (guarda total acumulado por hora)
    try:
        total = _xtracker_total(event)
        if total is not None and total > 0:
            now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:00")
            snaps = state.setdefault("xtracker_snapshots", {}).setdefault(slug, {})
            snaps[now_str] = total
            fresh_days = _daily_from_snapshots(snaps, event["start"])
            merged = {**fresh_days, **existing}
            if merged:
                log(f"[SCRAPE] xtracker OK: total={total}  cache+snap={len(merged)} días")
                return merged
    except Exception as e:
        log(f"[SCRAPE] xtracker falló: {e}")

    if existing:
        log(f"[SCRAPE] usando cache ({len(existing)} días)")
        return existing

    return {}


def _muskmeter(event: dict) -> dict:
    start_ts = event.get("start_ts")
    end_ts   = event.get("end_ts")
    event_id = event.get("event_id", "")

    if not start_ts or not end_ts:
        s = datetime.strptime(event["start"], "%Y-%m-%d").replace(
            hour=17, tzinfo=timezone.utc)
        e = datetime.strptime(event["end"], "%Y-%m-%d").replace(
            hour=17, tzinfo=timezone.utc)
        start_ts, end_ts = int(s.timestamp()), int(e.timestamp())

    url = (f"https://www.muskmeter.live/"
           f"?start={start_ts}&end={end_ts}"
           f"&eventId={event_id}&eventType=weekly&source=3")
    log(f"[SCRAPE] muskmeter URL: {url}")

    r = requests.get(url, headers={**HDR, "Accept": "text/html",
                                   "Referer": "https://www.muskmeter.live/"},
                     timeout=20)
    r.raise_for_status()

    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                  r.text, re.DOTALL)
    if m:
        nd    = json.loads(m.group(1))
        props = nd.get("props", {}).get("pageProps", {}) or {}
        daily = (props.get("dailyData") or props.get("heatmapData") or
                 props.get("dailyCounts") or props.get("tweetsByDay") or {})
        if isinstance(daily, dict) and daily:
            result = {k: int(v) for k, v in daily.items()
                      if re.match(r'\d{4}-\d{2}-\d{2}', str(k))}
            if result:
                return result

    matches = re.findall(r'"(\d{4}-\d{2}-\d{2})"\s*:\s*(\d+)', r.text)
    if matches:
        start_dt = datetime.strptime(event["start"], "%Y-%m-%d")
        return {d: int(c) for d, c in matches
                if abs((datetime.strptime(d, "%Y-%m-%d") - start_dt).days) <= 10}
    return {}


def _xtracker_total(event: dict) -> int | None:
    """Match estricto por mes+día de inicio."""
    try:
        start_dt   = datetime.strptime(event["start"], "%Y-%m-%d")
        search_str = f"{MONTH_NAMES[start_dt.month-1]} {start_dt.day}"
    except:
        return None

    r = requests.get(f"{XTRACKER_API}/users/elonmusk", headers=HDR, timeout=15)
    r.raise_for_status()
    data = r.json()
    trackings = (data.get("data", {}).get("trackings", [])
                 if isinstance(data.get("data"), dict)
                 else data.get("trackings", []))

    for t in trackings:
        title = t.get("title", "").lower()
        tid   = t.get("id", "")
        if not tid or "tweet" not in title or search_str not in title:
            continue
        r2 = requests.get(f"{XTRACKER_API}/trackings/{tid}?includeStats=true",
                          headers=HDR, timeout=15)
        r2.raise_for_status()
        total = r2.json().get("data", {}).get("stats", {}).get("total", 0)
        log(f"[XTRACKER] '{search_str}' → {title[:45]}  total={total}")
        return int(total)

    log(f"[XTRACKER] sin match para '{search_str}'")
    return None


def _daily_from_snapshots(snapshots: dict, start_date: str) -> dict:
    if not snapshots:
        return {}
    by_date = {}
    for ts_str, total in snapshots.items():
        date = ts_str[:10]
        by_date[date] = max(by_date.get(date, 0), total)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    dates = sorted(d for d in by_date
                   if datetime.strptime(d, "%Y-%m-%d") >= start_dt)
    if not dates:
        return {}
    daily, prev = {}, 0
    for date in dates:
        daily[date] = max(0, by_date[date] - prev)
        prev = by_date[date]
    return daily


# ══════════════════════════════════════════════════════════════════════
# EVENTOS ACTIVOS
# ══════════════════════════════════════════════════════════════════════

def fetch_active_events() -> list:
    try:
        from src.polymarket_sensor import PolymarketSensor
        from src.clob_scanner import ClobMarketScanner
        from src.utils import titles_match_paranoid
    except ImportError as e:
        log(f"[EVENTS] ImportError: {e}")
        return []

    sensor    = PolymarketSensor()
    scanner   = ClobMarketScanner()
    now       = datetime.now(tz=timezone.utc)
    m_polys   = sensor.get_active_counts()
    if not m_polys:
        return []
    clob_data = scanner.get_market_prices()
    result    = []

    for mp in m_polys:
        title   = mp.get("title", "")
        count   = mp.get("count", 0)
        h_rem   = mp.get("hours", 0)
        total_h = mp.get("total_hours", 168)

        elapsed_h = max(0, total_h - h_rem)
        day_n     = max(1, int(elapsed_h / 24) + 1)
        start_dt  = now - timedelta(hours=elapsed_h)
        end_dt    = now + timedelta(hours=h_rem)
        series    = "tue" if start_dt.weekday() == 1 else "fri"
        start_ts  = int(start_dt.replace(hour=17, minute=0, second=0,
                                          microsecond=0).timestamp())
        end_ts    = int(end_dt.replace(hour=17, minute=0, second=0,
                                        microsecond=0).timestamp())
        slug = re.sub(r'-+', '-',
               title.lower().replace(" ", "-").replace("#", "")
                    .replace(",", "").replace("?", "")).strip('-')

        prices, buckets = {}, []
        m_clob = next((c for c in clob_data
                       if titles_match_paranoid(title, c["title"])), None)
        if m_clob:
            buckets = m_clob.get("buckets", [])
            for b in buckets:
                rng, ask = b.get("bucket", ""), b.get("ask", 0)
                if rng and ask > 0:
                    prices[rng] = ask

        result.append({
            "slug": slug, "label": title,
            "start": start_dt.strftime("%Y-%m-%d"),
            "end":   end_dt.strftime("%Y-%m-%d"),
            "start_ts": start_ts, "end_ts": end_ts,
            "event_id": "", "day_n": day_n, "series": series,
            "h_rem": h_rem, "count": count,
            "prices": prices, "_buckets": buckets,
        })
        log(f"[EVENTS] {title[:45]}  D{day_n}  {series}  "
            f"h_rem={h_rem:.0f}h  count={count}  precios={len(prices)}")

    return result


# ══════════════════════════════════════════════════════════════════════
# MOTOR DE SEÑALES
# ══════════════════════════════════════════════════════════════════════

def proyectar(daily: dict, start: str) -> tuple:
    days = sorted(daily.items())
    if not days:
        return HIST_MEAN, HIST_SIGMA, 0, 0
    n = len(days)
    vals = [v for _, v in days]
    cum  = sum(vals)
    sv   = sorted(vals)
    pace_med  = sv[n // 2]
    pace_mean = cum / n
    pace_base = pace_med if (max(vals) > 2*pace_med and pace_med > 8) else pace_mean
    last_date = datetime.strptime(days[-1][0], "%Y-%m-%d")
    start_dt  = datetime.strptime(start, "%Y-%m-%d")
    days_left = max(0, 8 - (last_date - start_dt).days - 1)
    proj_pace  = cum + pace_base * days_left
    proj_blend = 0.50*proj_pace + 0.30*HIST_MEAN + 0.20*HIST_MEAN
    sigma      = max(40, HIST_SIGMA - n*4)
    return round(proj_blend), round(sigma), n, round(pace_mean, 1)


def rp(lo, hi, mu, sigma):
    def cdf(x):
        return 0.5*(1+math.erf((x-mu)/(sigma*math.sqrt(2))))
    return max(0, cdf(hi+0.5)-cdf(lo-0.5))


def generar_senal(event: dict, daily: dict, prev_total: int) -> dict | None:
    day_n  = event["day_n"]
    prices = event["prices"]
    if day_n < 3 or day_n > 5 or len(daily) < 3 or not prices:
        return None

    proj, sigma, n_days, pace = proyectar(daily, event["start"])
    vals  = list(daily.values())
    pace3 = sum(vals[:3])/3 if len(vals) >= 3 else pace
    confidence = "ALTA" if (pace3 > 50 or pace3 < 25 or max(vals[:3]) > 60) else "MEDIA"

    cum_total = sum(daily.values())
    rangos = sorted([r for r in prices if "-" in r
                     and int(r.split("-")[1]) > cum_total],
                    key=lambda x: int(x.split("-")[0]))
    best = []
    for i in range(len(rangos)):
        for j in range(i+1, min(i+6, len(rangos))):
            sub   = rangos[i:j+1]
            cost  = sum(prices.get(r, 0) for r in sub)
            p_win = sum(rp(int(r.split("-")[0]), int(r.split("-")[0])+19,
                           proj, sigma) for r in sub)
            if 0.05 < cost < 0.80 and len(sub) >= 2 and p_win > 0.05:
                ev = p_win/cost - 1
                if ev > 0.10:
                    best.append({"ranges": sub, "cost": round(cost,3),
                                 "p_win": round(p_win,3), "ev": round(ev,3),
                                 "roi": round(1/cost-1,2)})
    if not best:
        return None
    best.sort(key=lambda x: -x["ev"])
    top = best[0]

    ci_lo, ci_hi = proj-1.4*sigma, proj+1.4*sigma
    no_signals = [{"range": rng, "price_no": round(1-p,3),
                   "roi_no": round(1/(1-p)-1,3)}
                  for rng, p in prices.items()
                  if "-" in rng
                  and (int(rng.split("-")[1]) < ci_lo-15
                       or int(rng.split("-")[0]) > ci_hi+15)
                  and p > 0.08][:3]

    return {"event": event["label"], "slug": event["slug"],
            "day_n": day_n, "series": event["series"],
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "proj": proj, "sigma": sigma, "pace": pace,
            "confidence": confidence, "spread": top,
            "no_signals": no_signals, "prices": prices}


# ══════════════════════════════════════════════════════════════════════
# PAPER BOOK
# ══════════════════════════════════════════════════════════════════════

class PaperBook:
    def __init__(self):
        self.data = self._load()

    def _load(self):
        if PAPER_FILE.exists():
            try:
                with open(PAPER_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {"capital": CAPITAL_INIT, "positions": {}, "history": [],
                "stats": {"signals":0,"trades":0,"wins":0,"losses":0,"pnl":0.0}}

    def _save(self):
        with open(PAPER_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

    def _db_log(self, **kw):
        if not _DB_OK:
            return
        try:
            _db.log_trade(mode="SIGNAL_PAPER", strategy="SPREAD_YES", **kw)
        except Exception as e:
            log(f"[DB] {e}")

    def _db_upsert(self, pos_id, market, bucket, shares, entry, amount):
        if not _DB_OK:
            return
        try:
            _db.upsert_position(pos_id, {
                "market": market, "bucket": bucket, "shares": shares,
                "entry_price": entry, "invested": amount,
                "strategy_tag": "SPREAD_YES", "mode": "SIGNAL_PAPER"})
            _db.set_state("signal_paper_cash", self.data["capital"])
        except Exception as e:
            log(f"[DB] {e}")

    def _db_close(self, pos_id, pnl):
        if not _DB_OK:
            return
        try:
            _db.close_position(pos_id, pnl)
            _db.set_state("signal_paper_cash", self.data["capital"])
        except Exception as e:
            log(f"[DB] {e}")

    def buy(self, signal: dict, tg: Telegram) -> list:
        spread, prices, event = signal["spread"], signal["prices"], signal["event"]
        bought = []
        for rng in spread["ranges"]:
            pos_id = f"{signal['slug']}|{rng}"
            if pos_id in self.data["positions"]:
                continue
            price = prices.get(rng, 0)
            if price <= 0 or price > 0.80:
                continue
            amount = max(1.0, self.data["capital"] * RISK_PER_RANGE)
            if self.data["capital"] < amount:
                continue
            shares = amount / price
            self.data["capital"] -= amount
            self.data["positions"][pos_id] = {
                "entry": price, "shares": shares, "cost": amount,
                "event": event, "slug": signal["slug"], "rng": rng,
                "ts": datetime.now().isoformat(), "day_n": signal["day_n"]}
            self.data["stats"]["trades"] += 1
            bought.append(rng)
            log(f"[PAPER] BUY {rng} @ {price:.3f}  ${amount:.2f}  cap=${self.data['capital']:.2f}")
            tg.paper_buy(event, rng, price, amount, self.data["capital"])
            self._db_log(action="BUY", market=event, bucket=rng, price=price,
                         shares=shares, reason=f"SIGNAL_D{signal['day_n']}",
                         pnl=0, cash_after=self.data["capital"], pos_id=pos_id,
                         entry_signal_reason=f"ev={spread['ev']:+.0%} proj={signal['proj']}")
            self._db_upsert(pos_id, event, rng, shares, price, amount)
        self._save()
        return bought

    def check_exits(self, active_events: list, tg: Telegram):
        p_slug = {ev["slug"]: ev["prices"] for ev in active_events}
        h_slug = {ev["slug"]: ev["h_rem"]  for ev in active_events}
        to_close = []
        for pos_id, pos in self.data["positions"].items():
            cp    = p_slug.get(pos["slug"], {}).get(pos["rng"], pos["entry"])
            pct   = (cp - pos["entry"]) / pos["entry"]
            h_rem = h_slug.get(pos["slug"], 100)
            reason, exit_p = None, cp

            # Victory Lap: precio casi resuelto en las últimas 48h
            if h_rem <= 48 and cp >= 0.97:
                reason = f"Victory Lap ({cp:.2f})"
            # Paranoid Treasure: beneficio ≥ 80% en las últimas 48h
            # Fuera de esa ventana dejamos correr las ganancias
            elif h_rem <= 48 and pct >= 0.80:
                reason = f"Paranoid Treasure (+{pct*100:.0f}%)"
            # Stop Loss
            elif pos["entry"] < 0.15 and pct < -0.80:
                reason = f"Stop Loss ({pct*100:.0f}%)"
            elif pos["entry"] >= 0.15 and pct < -0.40:
                reason = f"Stop Loss ({pct*100:.0f}%)"
            # Evento expirado
            elif h_rem < -2:
                reason, exit_p = "Evento expirado", max(cp, 0.01)
            # Prune final: precio irrecuperable en últimas 24h
            elif h_rem <= 24 and cp < 0.05:
                reason = f"Prune final ({cp*100:.1f}¢)"

            if reason:
                to_close.append((pos_id, pos, exit_p, reason))

        for pos_id, pos, exit_p, reason in to_close:
            pnl = (exit_p - pos["entry"]) * pos["shares"]
            self.data["capital"] += exit_p * pos["shares"]
            outcome = "win" if pnl > 0 else "loss"
            self.data["stats"]["wins" if outcome=="win" else "losses"] += 1
            self.data["stats"]["pnl"] = round(self.data["stats"].get("pnl",0)+pnl, 4)
            self.data["history"].append({"event":pos["event"],"rng":pos["rng"],
                "entry":pos["entry"],"exit":exit_p,"pnl":round(pnl,4),
                "reason":reason,"ts":datetime.now().isoformat()})
            del self.data["positions"][pos_id]
            log(f"[PAPER] CLOSE {pos['rng']}  pnl={pnl:+.2f}  {reason}")
            tg.paper_close(pos["event"],pos["rng"],pos["entry"],exit_p,pnl,reason)
            self._db_log(action="SELL",market=pos["event"],bucket=pos["rng"],
                         price=exit_p,shares=pos["shares"],reason=reason,
                         pnl=pnl,cash_after=self.data["capital"],pos_id=pos_id,
                         exit_signal_reason=reason,trade_outcome_label=outcome)
            self._db_close(pos_id, pnl)
        if to_close:
            self._save()

    def enrich_positions(self, active_events):
        p_slug = {ev["slug"]: ev["prices"] for ev in active_events}
        return [{**pos, "current_price": p_slug.get(pos["slug"],{}).get(pos["rng"],pos["entry"])}
                for pos in self.data["positions"].values()]


# ══════════════════════════════════════════════════════════════════════
# ESTADO PERSISTENTE
# ══════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"muskmeter":{}, "xtracker_snapshots":{},
            "signals_sent":{}, "prev_totals":{}, "last_weekly":None}


def save_state(state: dict):
    cutoff = (datetime.now()-timedelta(days=14)).strftime("%Y-%m-%d")
    for slug in list(state.get("xtracker_snapshots",{}).keys()):
        state["xtracker_snapshots"][slug] = {
            k:v for k,v in state["xtracker_snapshots"][slug].items()
            if k[:10] >= cutoff}
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("  SIGNAL BOT — Sistema autónomo de señales D3")
    log(f"  Capital: ${CAPITAL_INIT}  |  Risk/rango: {RISK_PER_RANGE*100:.0f}%")
    log(f"  Poll:    cada {POLL_MINUTES} min  |  Burst: cada {BURST_POLL_SEC}s")
    log(f"  DB:      {'PostgreSQL (SIGNAL_PAPER)' if _DB_OK else 'JSON local'}")
    log("=" * 60)

    tg    = Telegram()
    paper = PaperBook()
    state = load_state()

    tg.bot_start(paper.data["capital"])

    # Arrancar burst monitor en hilo paralelo
    _cached_events = []
    _events_lock = threading.Lock()

    def _get_cached_events():
        with _events_lock:
            return list(_cached_events)

    def _set_cached_events(evts):
        with _events_lock:
            _cached_events.clear()
            _cached_events.extend(evts)

    burst_thread = threading.Thread(
        target=burst_monitor_loop,
        args=(tg, _get_cached_events, paper),
        daemon=True)
    burst_thread.start()
    log("[BURST] Hilo iniciado")

    last_weekly = datetime(2000, 1, 1)
    if state.get("last_weekly"):
        try:
            last_weekly = datetime.fromisoformat(state["last_weekly"])
        except:
            pass

    while True:
        try:
            log(f"\n{'─'*50} {datetime.now().strftime('%H:%M')}")

            events = fetch_active_events()
            _set_cached_events(events)
            if not events:
                log("[MAIN] Sin eventos — reintento en 5 min")
                time.sleep(300)
                continue

            paper.check_exits(events, tg)

            for ev in events:
                slug  = ev["slug"]
                day_n = ev["day_n"]

                fresh = scrape_tweets(ev, state)
                if fresh:
                    state["muskmeter"][slug] = fresh
                    save_state(state)
                    log(f"[SCRAPE] {slug}: {len(fresh)} días, {sum(fresh.values())} tweets")
                else:
                    log(f"[SCRAPE] {slug}: sin datos")
                    if day_n >= 3:
                        tg.scrape_warning(ev["label"], day_n)
                    continue

                prev_total = state["prev_totals"].get(ev["series"], HIST_MEAN)
                last_day   = state["signals_sent"].get(slug, 0)

                if day_n >= 3 and day_n != last_day:
                    signal = generar_senal(ev, fresh, prev_total)
                    if signal:
                        paper.data["stats"]["signals"] += 1
                        paper._save()
                        state["signals_sent"][slug] = day_n
                        save_state(state)
                        top = signal["spread"]
                        log(f"[SIGNAL] ✅ D{day_n}  {top['ranges']}  "
                            f"ev={top['ev']:+.0%}  conf={signal['confidence']}")
                        tg.signal_entry(label=ev["label"], day_n=day_n,
                                        ranges=top["ranges"], prices=ev["prices"],
                                        proj=signal["proj"], ev=top["ev"],
                                        confidence=signal["confidence"])
                        bought = paper.buy(signal, tg)
                        if bought:
                            log(f"[PAPER] Comprados: {bought}")
                    else:
                        log(f"[MAIN] {slug} D{day_n}: sin señal "
                            f"(datos={len(fresh)} días, EV insuficiente)")

            now = datetime.now()
            if (now.weekday() == 0 and now.hour == 8
                    and (now-last_weekly).total_seconds() > 86400):
                tg.portfolio_summary(paper.data["capital"],
                                     paper.enrich_positions(events),
                                     paper.data["stats"])
                tg.weekly_summary(paper.data["capital"], paper.data["stats"])
                state["last_weekly"] = now.isoformat()
                save_state(state)
                last_weekly = now

            log(f"[MAIN] OK — próximo en {POLL_MINUTES} min")
            time.sleep(POLL_MINUTES * 60)

        except KeyboardInterrupt:
            log("[MAIN] Bot detenido")
            paper._save()
            save_state(state)
            tg.bot_stop(paper.data["capital"], paper.data["stats"])
            break
        except Exception as e:
            import traceback
            log(f"[ERROR] {e}\n{traceback.format_exc()}")
            tg.send(f"⚠️ <b>Error Signal Bot</b>\n{str(e)[:200]}")
            time.sleep(120)


if __name__ == "__main__":
    main()
