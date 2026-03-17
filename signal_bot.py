"""
signal_bot.py  —  Sistema autónomo de señales D3 + Paper Trading
=================================================================
Fixes v3:
  - scrape no sobreescribe cache: hace merge con datos existentes
  - poll 15 min (era 60)
  - burst detector integrado como hilo paralelo (Nitter RSS)
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
HIST_MEAN      = 343
HIST_SIGMA     = 72
CAPITAL_INIT   = 100.0
RISK_PER_RANGE = 0.08
POLL_MINUTES   = 15          # ← bajado de 60 a 15
BURST_POLL_SEC = 120         # burst monitor cada 2 min
BURST_THRESHOLD = 4.0        # tweets/hora para alertar
XTRACKER_API   = "https://xtracker.polymarket.com/api"
HDR            = {"User-Agent": "Mozilla/5.0"}
MONTH_NAMES    = ["january","february","march","april","may","june",
                  "july","august","september","october","november","december"]
NITTER_MIRRORS = ["https://nitter.privacydev.net", "https://nitter.poast.org",
                  "https://nitter.net"]


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

    def burst_alert(self, label, pace, n_tweets, examples):
        self.send(f"🔥 <b>BURST DETECTADO — {label}</b>\n"
                  f"  Pace:    <b>{pace:.0f} tweets/hora</b>\n"
                  f"  Últimos: <b>{n_tweets} tweets</b> en 10 min\n"
                  f"  Ejemplo: {examples[0][:80] if examples else '—'}\n"
                  f"  ⚡ El mercado tardará ~15 min en ajustarse")

    def scrape_warning(self, label, day_n):
        self.send(f"⚠️ <b>Sin datos tweets D{day_n}</b>\n  {label}", silent=True)

    def bot_start(self, capital):
        self.send(f"🤖 <b>Signal Bot arrancado</b>\n"
                  f"  Capital paper: <b>${capital:.2f}</b>\n"
                  f"  Poll señales:  cada {POLL_MINUTES} min\n"
                  f"  Burst monitor: cada {BURST_POLL_SEC}s\n"
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
# BURST MONITOR  (hilo paralelo, Nitter RSS)
# ══════════════════════════════════════════════════════════════════════

_burst_seen_ids   = set()
_burst_buffer     = []   # (datetime, text)
_last_burst_alert = None
_mirror_idx       = 0


def _fetch_nitter_tweets() -> list:
    """Fetch latest Elon tweets via Nitter RSS. Returns list of (id, text, dt)."""
    global _mirror_idx
    for _ in range(len(NITTER_MIRRORS)):
        mirror = NITTER_MIRRORS[_mirror_idx % len(NITTER_MIRRORS)]
        try:
            r = requests.get(f"{mirror}/elonmusk/rss",
                             headers=HDR, timeout=8)
            items = re.findall(r'<item>(.*?)</item>', r.text, re.DOTALL)
            result = []
            for item in items:
                id_m  = re.search(r'<guid[^>]*>(.*?)</guid>', item)
                txt_m = re.search(r'<title><!\[CDATA\[(.*?)\]\]></title>',
                                  item, re.DOTALL)
                dt_m  = re.search(r'<pubDate>(.*?)</pubDate>', item)
                tid   = id_m.group(1).strip() if id_m else ""
                if not tid or tid in _burst_seen_ids:
                    continue
                text = re.sub(r'<[^>]+>', '', txt_m.group(1) if txt_m else "")
                try:
                    dt = datetime.strptime(dt_m.group(1).strip(),
                                           "%a, %d %b %Y %H:%M:%S %Z"
                                           ).replace(tzinfo=timezone.utc)
                except:
                    dt = datetime.now(tz=timezone.utc)
                result.append((tid, text[:200], dt))
            return result
        except Exception as e:
            log(f"[BURST] Nitter {mirror} falló: {e}")
            _mirror_idx += 1
    return []


def burst_monitor_loop(tg: Telegram, get_events_fn):
    """Hilo paralelo: detecta bursts de tweets de Elon."""
    global _last_burst_alert, _mirror_idx
    log("[BURST] Monitor iniciado")

    while True:
        try:
            new_tweets = _fetch_nitter_tweets()
            now = datetime.now(tz=timezone.utc)

            for tid, text, dt in new_tweets:
                _burst_seen_ids.add(tid)
                _burst_buffer.append((dt, text))

            # Mantener solo últimos 30 min
            cutoff = now - timedelta(minutes=30)
            while _burst_buffer and _burst_buffer[0][0] < cutoff:
                _burst_buffer.pop(0)

            # Calcular pace en ventana de 10 min
            window_10 = now - timedelta(minutes=10)
            recent    = [(dt, t) for dt, t in _burst_buffer if dt >= window_10]
            pace      = len(recent) / (10 / 60)  # tweets/hora

            if pace >= BURST_THRESHOLD:
                cooldown_ok = (_last_burst_alert is None or
                               (now - _last_burst_alert).total_seconds() > 1200)
                if cooldown_ok and recent:
                    _last_burst_alert = now
                    examples = [t for _, t in recent[:2]]
                    log(f"[BURST] 🔥 pace={pace:.0f}/h  {len(recent)} tweets en 10min")
                    tg.burst_alert(
                        label="Elon Musk",
                        pace=pace,
                        n_tweets=len(recent),
                        examples=examples
                    )
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
            merged = {**existing, **result}   # merge, muskmeter gana en conflictos
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
            # Reconstruir solo los días sin datos en el cache existente
            fresh_days = _daily_from_snapshots(snaps, event["start"])
            merged = {**fresh_days, **existing}  # existing tiene prioridad
            if merged:
                log(f"[SCRAPE] xtracker OK: total={total}  cache+snap={len(merged)} días")
                return merged
    except Exception as e:
        log(f"[SCRAPE] xtracker falló: {e}")

    # Fallback: devolver lo que hay en cache
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

    rangos = sorted([r for r in prices if "-" in r],
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
            if h_rem <= 48 and cp >= 0.97:
                reason = f"Victory Lap ({cp:.2f})"
            elif pos["entry"] < 0.15 and pct < -0.80:
                reason = f"Stop Loss ({pct*100:.0f}%)"
            elif pos["entry"] >= 0.15 and pct < -0.40:
                reason = f"Stop Loss ({pct*100:.0f}%)"
            elif h_rem < -2:
                reason, exit_p = "Evento expirado", max(cp, 0.01)
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
    burst_thread = threading.Thread(
        target=burst_monitor_loop,
        args=(tg, lambda: []),
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
            if not events:
                log("[MAIN] Sin eventos — reintento en 5 min")
                time.sleep(300)
                continue

            paper.check_exits(events, tg)

            for ev in events:
                slug  = ev["slug"]
                day_n = ev["day_n"]

                # MERGE: scrape_tweets ya combina nuevo con cache
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
