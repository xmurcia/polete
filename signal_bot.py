"""
signal_bot.py  —  Sistema autónomo de señales D3 + Paper Trading
=================================================================
Completamente autónomo:
  - Scrapa xtracker/muskmeter automáticamente cada hora
  - Detecta eventos activos (Mar→Mar y Vie→Vie en paralelo)
  - Genera señales en D3-D5 con modelo pace+DdS+reglas
  - Ejecuta paper trades automáticamente
  - Persiste en PostgreSQL con mode='SIGNAL_PAPER'
    (separado del bot real, que usa mode='REAL')
  - Envía señales, compras, cierres y resumen semanal a Telegram

Uso:
  python signal_bot.py

Variables .env (las mismas del bot):
  TELEGRAM_BOT_TOKEN=xxx
  TELEGRAM_CHAT_ID=xxx        (canal @nombre o chat_id numérico)
  DATABASE_URL=postgresql://  (ya está en tu .env)
"""

import os, json, math, time, re, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

# ── Rutas ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
SIGNALS_DIR = ROOT / "signals"
SIGNALS_DIR.mkdir(exist_ok=True)

STATE_FILE = SIGNALS_DIR / "signal_bot_state.json"
PAPER_FILE = SIGNALS_DIR / "paper_portfolio.json"
LOG_FILE   = SIGNALS_DIR / "signal_bot.log"

# ── DB (reutiliza database.py del bot existente) ────────────────────────
try:
    sys.path.insert(0, str(ROOT))
    import database as _db
    import psycopg2.extras as _pge
    _DB_OK = _db._pool is not None
    if _DB_OK:
        print("✅ [DB] PostgreSQL conectado — trades en mode=SIGNAL_PAPER")
    else:
        print("ℹ️  [DB] Sin conexión — guardando solo en JSON local")
except Exception as _e:
    _db    = None
    _DB_OK = False
    _pge   = None
    print(f"ℹ️  [DB] No disponible ({_e})")

# ── Constantes ──────────────────────────────────────────────────────────
HIST_MEAN      = 343
HIST_SIGMA     = 72
DOW_MULT       = {"Mon":1.00,"Tue":0.80,"Wed":1.08,"Thu":1.38,
                  "Fri":0.75,"Sat":1.08,"Sun":1.14}
CAPITAL_INIT   = 100.0
RISK_PER_RANGE = 0.08      # 8% del capital por rango del spread
POLL_MINUTES   = 60        # frecuencia de scraping y señales
GAMMA_API      = "https://gamma-api.polymarket.com"
HDR            = {"User-Agent": "Mozilla/5.0"}


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
        log(f"[TG] {'✅ activado' if self.ok else '⚠️  desactivado (falta token/chat_id)'}")

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
        lines = [
            f"📊 <b>SEÑAL D{day_n} — {label}</b>", "",
            f"🎯 <b>SPREAD YES — {len(ranges)} rangos</b>",
        ]
        for rng in ranges:
            lines.append(f"  • YES <b>{rng}</b>  →  {prices.get(rng,0)*100:.1f}¢")
        lines += [
            "",
            f"💰 Coste total:    <b>{cost*100:.1f}¢</b> por $1",
            f"📈 ROI si gana:    <b>+{roi:.0f}%</b>",
            f"🔮 Proyección:     <b>{proj} tweets</b>",
            f"⚡ EV esperado:    <b>{ev*100:+.0f}%</b>",
            f"🎲 Confianza:      <b>{confidence}</b>",
            "",
            f"⏰ {datetime.now().strftime('%d/%m %H:%M')} UTC",
        ]
        self.send("\n".join(lines))

    def paper_buy(self, label, rng, price, amount, capital_left):
        self.send(
            f"📄 <b>PAPER BUY</b>\n"
            f"  Evento: {label}\n"
            f"  Rango:  <b>{rng}</b>  @ {price*100:.1f}¢\n"
            f"  Monto:  <b>${amount:.2f}</b>\n"
            f"  Capital: ${capital_left:.2f}",
            silent=True)

    def paper_close(self, label, rng, entry, exit_p, pnl, reason):
        emoji = "✅" if pnl >= 0 else "❌"
        roi   = (exit_p - entry) / entry * 100 if entry > 0 else 0
        self.send(
            f"{emoji} <b>PAPER CLOSE</b> — {label}\n"
            f"  Rango:   <b>{rng}</b>\n"
            f"  {entry*100:.1f}¢ → {exit_p*100:.1f}¢  ({roi:+.0f}%)\n"
            f"  P&amp;L: <b>${pnl:+.2f}</b>  |  {reason}")

    def portfolio_summary(self, capital, positions, stats):
        roi = (capital - CAPITAL_INIT) / CAPITAL_INIT * 100
        w, l = stats["wins"], stats["losses"]
        total = w + l
        wr = 100 * w / total if total else 0
        lines = [
            "📊 <b>Portfolio — SIGNAL PAPER</b>", "",
            f"  💵 Capital:     <b>${capital:.2f}</b>  (ROI {roi:+.1f}%)",
            f"  📈 Trades:      {total}  ({w}✅ {l}❌)  {wr:.0f}% WR",
            f"  💰 P&amp;L:     <b>${stats.get('pnl', 0):+.2f}</b>", "",
        ]
        if positions:
            lines.append("  <b>Abiertas:</b>")
            for p in positions[:8]:
                ep  = p.get("entry", 0)
                cp  = p.get("current_price", ep)
                pnl = (cp - ep) * p.get("shares", 0)
                lines.append(
                    f"    {p['rng']:10}  {ep*100:.0f}¢→{cp*100:.0f}¢  ${pnl:+.2f}")
        lines.append(f"\n  ⏰ {datetime.now().strftime('%d/%m %H:%M')}")
        self.send("\n".join(lines))

    def weekly_summary(self, capital, stats):
        roi = (capital - CAPITAL_INIT) / CAPITAL_INIT * 100
        w, l = stats["wins"], stats["losses"]
        total = w + l
        wr = 100 * w / total if total else 0
        self.send(
            f"📅 <b>RESUMEN SEMANAL</b>\n\n"
            f"  Señales:   {stats['signals']}\n"
            f"  Trades:    {stats['trades']}\n"
            f"  Win rate:  {wr:.0f}%  ({w}W / {l}L)\n"
            f"  P&amp;L:   <b>${stats.get('pnl',0):+.2f}</b>\n"
            f"  Capital:   <b>${capital:.2f}</b>  (inicio ${CAPITAL_INIT:.0f})\n"
            f"  ROI total: <b>{roi:+.1f}%</b>\n\n"
            f"  ⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    def bot_start(self, capital):
        self.send(
            f"🤖 <b>Signal Bot arrancado</b>\n"
            f"  Capital paper: <b>${capital:.2f}</b>\n"
            f"  Poll: cada {POLL_MINUTES} min\n"
            f"  DB: {'✅ PostgreSQL (SIGNAL_PAPER)' if _DB_OK else '📁 JSON local'}\n"
            f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    def bot_stop(self, capital, stats):
        roi = (capital - CAPITAL_INIT) / CAPITAL_INIT * 100
        w, l = stats["wins"], stats["losses"]
        total = w + l
        wr = 100 * w / total if total else 0
        self.send(
            f"🛑 <b>Signal Bot detenido</b>\n"
            f"  Capital: <b>${capital:.2f}</b>  (ROI {roi:+.1f}%)\n"
            f"  Win rate: {wr:.0f}%  ({w}W/{l}L)")


# ══════════════════════════════════════════════════════════════════════
# SCRAPING  (xtracker primero, muskmeter como fallback)
# ══════════════════════════════════════════════════════════════════════

def scrape_tweets(start_date: str) -> dict:
    """
    Obtiene tweets diarios del evento automáticamente.
    Returns: {"2026-03-13": 23, "2026-03-14": 37, ...}
    """
    try:
        result = _xtracker(start_date)
        if result:
            log(f"[SCRAPE] xtracker OK: {len(result)} días desde {start_date}")
            return result
    except Exception as e:
        log(f"[SCRAPE] xtracker falló: {e}")

    try:
        result = _muskmeter_html(start_date)
        if result:
            log(f"[SCRAPE] muskmeter HTML OK: {len(result)} días")
            return result
    except Exception as e:
        log(f"[SCRAPE] muskmeter falló: {e}")

    return {}


def _xtracker(start_date: str) -> dict:
    r = requests.get(
        "https://xtracker.polymarket.com/api/users/elonmusk",
        headers=HDR, timeout=15)
    r.raise_for_status()
    data = r.json()

    trackings = (data.get("data", {}).get("trackings", [])
                 if isinstance(data.get("data"), dict)
                 else data.get("trackings", []))

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    target = None
    for t in trackings:
        try:
            ts = datetime.fromisoformat(
                t.get("startDate", "").replace("Z", "+00:00"))
            if abs((ts.date() - start_dt.date()).days) <= 1:
                target = t
                break
        except:
            pass

    if not target:
        return {}

    tid = target.get("id", "")
    if not tid:
        return {}

    r2 = requests.get(
        f"https://xtracker.polymarket.com/api/trackings/{tid}?includeStats=true",
        headers=HDR, timeout=15)
    r2.raise_for_status()
    detail = r2.json().get("data", {})

    result = {}
    for ds in detail.get("stats", {}).get("dailyStats", []):
        d = ds.get("date", "")[:10]
        c = ds.get("count", ds.get("total", 0))
        if d:
            result[d] = int(c)
    return result


def _muskmeter_html(start_date: str) -> dict:
    url = f"https://muskmeter.live/{start_date}"
    r   = requests.get(url, headers={**HDR, "Accept": "text/html"}, timeout=20)
    r.raise_for_status()

    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                  r.text, re.DOTALL)
    if not m:
        return {}

    nd    = json.loads(m.group(1))
    props = (nd.get("props", {}).get("pageProps", {}) or
             nd.get("props", {}).get("initialProps", {}))
    daily = (props.get("dailyData") or props.get("heatmapData") or
             props.get("tweetData", {}).get("daily") or {})

    if isinstance(daily, dict):
        return {k: int(v) for k, v in daily.items()
                if re.match(r'\d{4}-\d{2}-\d{2}', k)}

    totals = re.findall(r'"(\d{4}-\d{2}-\d{2})":\s*(\d+)', r.text)
    return {d: int(c) for d, c in totals} if totals else {}


# ══════════════════════════════════════════════════════════════════════
# EVENTOS ACTIVOS  (Gamma API — ambas series Tue/Fri)
# ══════════════════════════════════════════════════════════════════════

def fetch_active_events() -> list:
    try:
        r = requests.get(f"{GAMMA_API}/events",
                         params={"active": "true", "limit": 50},
                         headers=HDR, timeout=10)
        raw = r.json() if isinstance(r.json(), list) else r.json().get("events", [])
    except Exception as e:
        log(f"[EVENTS] Error Gamma: {e}")
        return []

    now    = datetime.now(tz=timezone.utc)
    result = []

    for ev in raw:
        slug = ev.get("slug", "").lower()
        if "elon" not in slug or "tweet" not in slug:
            continue
        try:
            start_dt = datetime.fromisoformat(
                ev.get("startDate", "").replace("Z", "+00:00"))
            end_dt   = datetime.fromisoformat(
                ev.get("endDate", "").replace("Z", "+00:00"))
        except:
            continue

        day_n  = max(1, int((now - start_dt).total_seconds() / 86400) + 1)
        series = "tue" if start_dt.weekday() == 1 else "fri"
        h_rem  = max(0, (end_dt - now).total_seconds() / 3600)

        prices = {}
        for m in ev.get("markets", []):
            rng = m.get("groupItemTitle", "")
            p   = m.get("lastTradePrice") or m.get("bestAsk")
            if rng and p is not None:
                try:
                    prices[rng] = float(p)
                except:
                    pass

        result.append({
            "slug":   slug,
            "label":  ev.get("title", "").replace("Elon Musk # tweets ", ""),
            "start":  start_dt.strftime("%Y-%m-%d"),
            "end":    end_dt.strftime("%Y-%m-%d"),
            "day_n":  day_n,
            "series": series,
            "h_rem":  h_rem,
            "prices": prices,
        })
        log(f"[EVENTS] {slug}  D{day_n}  {series}  "
            f"h_rem={h_rem:.0f}h  precios={len(prices)}")

    return result


# ══════════════════════════════════════════════════════════════════════
# MOTOR DE SEÑALES  (pace + DdS + reglas IF/THEN)
# ══════════════════════════════════════════════════════════════════════

def proyectar(daily: dict, start: str) -> tuple:
    """Proyecta total final. Returns (proj, sigma, n_days, pace_mean)."""
    days = sorted(daily.items())
    if not days:
        return HIST_MEAN, HIST_SIGMA, 0, 0

    n         = len(days)
    vals      = [v for _, v in days]
    cum       = sum(vals)
    sv        = sorted(vals)
    pace_med  = sv[n // 2]
    pace_mean = cum / n
    has_burst = max(vals) > 2 * pace_med and pace_med > 8
    pace_base = pace_med if has_burst else pace_mean

    last_date = datetime.strptime(days[-1][0], "%Y-%m-%d")
    start_dt  = datetime.strptime(start, "%Y-%m-%d")
    days_done = (last_date - start_dt).days + 1
    days_left = max(0, 8 - days_done)

    proj_pace  = cum + pace_base * days_left
    proj_blend = 0.50 * proj_pace + 0.30 * HIST_MEAN + 0.20 * HIST_MEAN
    sigma      = max(40, HIST_SIGMA - n * 4)

    return round(proj_blend), round(sigma), n, round(pace_mean, 1)


def rp(lo, hi, mu, sigma):
    """P(total en [lo, hi]) bajo normal."""
    def cdf(x):
        return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
    return max(0, cdf(hi + 0.5) - cdf(lo - 0.5))


def generar_senal(event: dict, daily: dict, prev_total: int) -> dict | None:
    """Genera señal D3-D5. None si EV < 10% o datos insuficientes."""
    day_n  = event["day_n"]
    prices = event["prices"]

    if day_n < 3 or day_n > 5 or len(daily) < 3 or not prices:
        return None

    proj, sigma, n_days, pace = proyectar(daily, event["start"])

    vals  = list(daily.values())
    pace3 = sum(vals[:3]) / 3 if len(vals) >= 3 else pace
    if pace3 > 50 or pace3 < 25 or max(vals[:3]) > 60:
        confidence = "ALTA"
    else:
        confidence = "MEDIA"

    rangos = sorted([r for r in prices if "-" in r],
                    key=lambda x: int(x.split("-")[0]))
    best = []
    for i in range(len(rangos)):
        for j in range(i + 1, min(i + 6, len(rangos))):
            sub   = rangos[i:j + 1]
            cost  = sum(prices.get(r, 0) for r in sub)
            p_win = sum(rp(int(r.split("-")[0]),
                           int(r.split("-")[0]) + 19,
                           proj, sigma) for r in sub)
            if 0.05 < cost < 0.80 and len(sub) >= 2 and p_win > 0.05:
                ev = p_win / cost - 1
                if ev > 0.10:
                    best.append({
                        "ranges": sub,
                        "cost":   round(cost, 3),
                        "p_win":  round(p_win, 3),
                        "ev":     round(ev, 3),
                        "roi":    round(1 / cost - 1, 2),
                    })

    if not best:
        return None

    best.sort(key=lambda x: -x["ev"])
    top = best[0]

    ci_lo      = proj - 1.4 * sigma
    ci_hi      = proj + 1.4 * sigma
    no_signals = []
    for rng, p_yes in prices.items():
        if "-" not in rng:
            continue
        lo, hi = map(int, rng.split("-"))
        if (hi < ci_lo - 15 or lo > ci_hi + 15) and p_yes > 0.08:
            no_signals.append({
                "range":    rng,
                "price_no": round(1 - p_yes, 3),
                "roi_no":   round(1 / (1 - p_yes) - 1, 3),
            })

    return {
        "event":      event["label"],
        "slug":       event["slug"],
        "day_n":      day_n,
        "series":     event["series"],
        "ts":         datetime.now(tz=timezone.utc).isoformat(),
        "proj":       proj,
        "sigma":      sigma,
        "pace":       pace,
        "confidence": confidence,
        "spread":     top,
        "no_signals": no_signals[:3],
        "prices":     prices,
    }


# ══════════════════════════════════════════════════════════════════════
# PAPER BOOK  (portfolio con persistencia dual JSON + PostgreSQL)
# ══════════════════════════════════════════════════════════════════════

class PaperBook:
    """
    Portfolio paper trading.
    JSON local siempre (fallback offline).
    PostgreSQL cuando disponible, con mode='SIGNAL_PAPER'
    — completamente separado de los trades REAL del bot.
    """

    def __init__(self):
        self.data = self._load()

    def _load(self) -> dict:
        if PAPER_FILE.exists():
            try:
                with open(PAPER_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {
            "capital":   CAPITAL_INIT,
            "positions": {},
            "history":   [],
            "stats":     {"signals": 0, "trades": 0,
                          "wins": 0, "losses": 0, "pnl": 0.0},
        }

    def _save(self):
        with open(PAPER_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

    # ── DB helpers ───────────────────────────────────────────────────

    def _db_log_trade(self, action, market, bucket, price, shares,
                      reason, pnl, cash_after, pos_id,
                      entry_reason=None, exit_reason=None, outcome=None):
        if not _DB_OK:
            return
        try:
            _db.log_trade(
                action=action, market=market, bucket=bucket,
                price=price, shares=shares, reason=reason,
                pnl=pnl, cash_after=cash_after,
                mode="SIGNAL_PAPER", strategy="SPREAD_YES",
                pos_id=pos_id,
                entry_signal_reason=entry_reason,
                exit_signal_reason=exit_reason,
                trade_outcome_label=outcome,
            )
        except Exception as e:
            log(f"[DB] log_trade error: {e}")

    def _db_upsert_pos(self, pos_id, market, bucket, shares, entry_price, amount):
        if not _DB_OK:
            return
        try:
            _db.upsert_position(pos_id, {
                "market": market, "bucket": bucket,
                "shares": shares, "entry_price": entry_price,
                "invested": amount, "strategy_tag": "SPREAD_YES",
                "mode": "SIGNAL_PAPER",
            })
            _db.set_state("signal_paper_cash", self.data["capital"])
        except Exception as e:
            log(f"[DB] upsert_pos error: {e}")

    def _db_close_pos(self, pos_id, pnl):
        if not _DB_OK:
            return
        try:
            _db.close_position(pos_id, pnl)
            _db.set_state("signal_paper_cash", self.data["capital"])
        except Exception as e:
            log(f"[DB] close_pos error: {e}")

    # ── Trading ──────────────────────────────────────────────────────

    def buy(self, signal: dict, tg: Telegram) -> list:
        """Compra todos los rangos del spread. Returns rangos comprados."""
        spread = signal["spread"]
        prices = signal["prices"]
        event  = signal["event"]
        bought = []

        for rng in spread["ranges"]:
            pos_id = f"{signal['slug']}|{rng}"
            if pos_id in self.data["positions"]:
                continue

            price  = prices.get(rng, 0)
            if price <= 0 or price > 0.80:
                continue

            amount = max(1.0, self.data["capital"] * RISK_PER_RANGE)
            if self.data["capital"] < amount:
                log(f"[PAPER] Sin capital para {rng}")
                continue

            shares = amount / price
            self.data["capital"] -= amount
            self.data["positions"][pos_id] = {
                "entry":  price,
                "shares": shares,
                "cost":   amount,
                "event":  event,
                "slug":   signal["slug"],
                "rng":    rng,
                "ts":     datetime.now().isoformat(),
                "day_n":  signal["day_n"],
            }
            self.data["stats"]["trades"] += 1
            bought.append(rng)

            log(f"[PAPER] BUY  {rng} @ {price:.3f}  ${amount:.2f}  "
                f"cap=${self.data['capital']:.2f}")
            tg.paper_buy(event, rng, price, amount, self.data["capital"])

            self._db_log_trade(
                action="BUY", market=event, bucket=rng,
                price=price, shares=shares,
                reason=f"SIGNAL_D{signal['day_n']}",
                pnl=0, cash_after=self.data["capital"],
                pos_id=pos_id,
                entry_reason=(f"spread ev={spread['ev']:+.0%} "
                               f"conf={signal['confidence']} "
                               f"proj={signal['proj']}"),
            )
            self._db_upsert_pos(pos_id, event, rng, shares, price, amount)

        self._save()
        return bought

    def check_exits(self, active_events: list, tg: Telegram):
        """Cierra posiciones según criterios de salida."""
        prices_by_slug = {ev["slug"]: ev["prices"] for ev in active_events}
        h_rem_by_slug  = {ev["slug"]: ev["h_rem"]  for ev in active_events}
        to_close       = []

        for pos_id, pos in self.data["positions"].items():
            slug          = pos["slug"]
            rng           = pos["rng"]
            current_price = prices_by_slug.get(slug, {}).get(rng, pos["entry"])
            profit_pct    = (current_price - pos["entry"]) / pos["entry"]
            h_rem         = h_rem_by_slug.get(slug, 100)

            reason = None
            exit_p = current_price

            if h_rem <= 48 and current_price >= 0.97:
                reason = f"Victory Lap ({current_price:.2f})"
            elif pos["entry"] < 0.15 and profit_pct < -0.80:
                reason = f"Stop Loss ({profit_pct*100:.0f}%)"
            elif pos["entry"] >= 0.15 and profit_pct < -0.40:
                reason = f"Stop Loss ({profit_pct*100:.0f}%)"
            elif h_rem < -2:
                reason = "Evento expirado"
                exit_p = max(current_price, 0.01)
            elif h_rem <= 24 and current_price < 0.05:
                reason = f"Prune final ({current_price*100:.1f}¢)"

            if reason:
                to_close.append((pos_id, pos, exit_p, reason))

        for pos_id, pos, exit_p, reason in to_close:
            pnl     = (exit_p - pos["entry"]) * pos["shares"]
            revenue = exit_p * pos["shares"]
            self.data["capital"] += revenue
            outcome = "win" if pnl > 0 else "loss"

            if outcome == "win":
                self.data["stats"]["wins"] += 1
            else:
                self.data["stats"]["losses"] += 1
            self.data["stats"]["pnl"] = round(
                self.data["stats"].get("pnl", 0) + pnl, 4)

            self.data["history"].append({
                "event": pos["event"], "rng": pos["rng"],
                "entry": pos["entry"], "exit": exit_p,
                "pnl": round(pnl, 4), "reason": reason,
                "ts": datetime.now().isoformat(),
            })
            del self.data["positions"][pos_id]

            log(f"[PAPER] CLOSE {pos['rng']}  pnl={pnl:+.2f}  {reason}")
            tg.paper_close(pos["event"], pos["rng"],
                           pos["entry"], exit_p, pnl, reason)

            self._db_log_trade(
                action="SELL", market=pos["event"], bucket=pos["rng"],
                price=exit_p, shares=pos["shares"], reason=reason,
                pnl=pnl, cash_after=self.data["capital"],
                pos_id=pos_id, exit_reason=reason, outcome=outcome,
            )
            self._db_close_pos(pos_id, pnl)

        if to_close:
            self._save()

    def enrich_positions(self, active_events: list) -> list:
        """Añade current_price a cada posición abierta."""
        prices_by_slug = {ev["slug"]: ev["prices"] for ev in active_events}
        enriched = []
        for pos_id, pos in self.data["positions"].items():
            cp = prices_by_slug.get(pos["slug"], {}).get(pos["rng"], pos["entry"])
            enriched.append({**pos, "current_price": cp})
        return enriched


# ══════════════════════════════════════════════════════════════════════
# DB QUERY  — snapshot del portfolio SIGNAL_PAPER
# ══════════════════════════════════════════════════════════════════════

def db_snapshot() -> dict | None:
    """Lee el portfolio SIGNAL_PAPER directamente de PostgreSQL."""
    if not _DB_OK or _pge is None:
        return None
    try:
        with _db.get_conn() as conn:
            cur = conn.cursor(cursor_factory=_pge.RealDictCursor)

            cur.execute(
                "SELECT value FROM bot_state "
                "WHERE key = 'signal_paper_cash'")
            row     = cur.fetchone()
            capital = float(row["value"]) if row else CAPITAL_INIT

            cur.execute(
                "SELECT pos_id, bucket, market, shares, "
                "       entry_price, current_price, invested "
                "FROM positions WHERE mode = 'SIGNAL_PAPER' "
                "ORDER BY opened_at DESC")
            positions = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT action, bucket, price, pnl, ts, trade_outcome_label "
                "FROM trades WHERE mode = 'SIGNAL_PAPER' "
                "ORDER BY ts DESC LIMIT 50")
            trades = [dict(r) for r in cur.fetchall()]

        wins  = sum(1 for t in trades
                    if t["action"] == "SELL" and float(t.get("pnl") or 0) > 0)
        loss  = sum(1 for t in trades
                    if t["action"] == "SELL" and float(t.get("pnl") or 0) <= 0)
        total_pnl = sum(float(t.get("pnl") or 0)
                        for t in trades if t["action"] == "SELL")
        return {
            "capital": capital, "positions": positions,
            "wins": wins, "losses": loss, "total_pnl": total_pnl,
        }
    except Exception as e:
        log(f"[DB] snapshot error: {e}")
        return None


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
    return {
        "muskmeter":    {},
        "signals_sent": {},
        "prev_totals":  {},
        "last_weekly":  None,
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("  SIGNAL BOT — Sistema autónomo de señales D3")
    log(f"  Capital: ${CAPITAL_INIT}  |  Risk/rango: {RISK_PER_RANGE*100:.0f}%")
    log(f"  Poll:    cada {POLL_MINUTES} min")
    log(f"  DB:      {'PostgreSQL (SIGNAL_PAPER)' if _DB_OK else 'JSON local'}")
    log("=" * 60)

    tg    = Telegram()
    paper = PaperBook()
    state = load_state()

    tg.bot_start(paper.data["capital"])

    last_weekly = datetime(2000, 1, 1)
    if state.get("last_weekly"):
        try:
            last_weekly = datetime.fromisoformat(state["last_weekly"])
        except:
            pass

    while True:
        try:
            log(f"\n{'─'*50} {datetime.now().strftime('%H:%M')}")

            # 1. Detectar eventos activos (Tue + Fri en paralelo)
            events = fetch_active_events()
            if not events:
                log("[MAIN] Sin eventos activos — reintento en 5 min")
                time.sleep(300)
                continue

            # 2. Revisar exits del portfolio paper
            paper.check_exits(events, tg)

            # 3. Procesar cada evento
            for ev in events:
                slug  = ev["slug"]
                day_n = ev["day_n"]

                # Scrapear tweets automáticamente
                fresh = scrape_tweets(ev["start"])
                if fresh:
                    state["muskmeter"][slug] = fresh
                    save_state(state)
                    cum = sum(fresh.values())
                    log(f"[SCRAPE] {slug}: {len(fresh)} días, {cum} tweets total")
                else:
                    fresh = state["muskmeter"].get(slug, {})
                    if not fresh:
                        log(f"[SCRAPE] {slug}: sin datos, saltando")
                        continue
                    log(f"[SCRAPE] {slug}: cache ({len(fresh)} días)")

                prev_total = state["prev_totals"].get(ev["series"], HIST_MEAN)

                # 4. Generar señal si D3-D5 y no enviada aún en este día
                last_day_sent = state["signals_sent"].get(slug, 0)
                if day_n >= 3 and day_n != last_day_sent:
                    signal = generar_senal(ev, fresh, prev_total)

                    if signal:
                        paper.data["stats"]["signals"] += 1
                        paper._save()
                        state["signals_sent"][slug] = day_n
                        save_state(state)

                        top = signal["spread"]
                        log(f"[SIGNAL] ✅ D{day_n}  ranges={top['ranges']}  "
                            f"ev={top['ev']:+.0%}  conf={signal['confidence']}")

                        tg.signal_entry(
                            label=ev["label"], day_n=day_n,
                            ranges=top["ranges"], prices=ev["prices"],
                            proj=signal["proj"], ev=top["ev"],
                            confidence=signal["confidence"],
                        )

                        bought = paper.buy(signal, tg)
                        if bought:
                            log(f"[PAPER] Comprados: {bought}")
                    else:
                        log(f"[MAIN] {slug} D{day_n}: sin señal (EV insuficiente)")

            # 5. Resumen semanal — lunes 08:00 UTC
            now = datetime.now()
            if (now.weekday() == 0 and now.hour == 8
                    and (now - last_weekly).total_seconds() > 86400):
                enriched = paper.enrich_positions(events)
                tg.portfolio_summary(
                    paper.data["capital"], enriched, paper.data["stats"])
                tg.weekly_summary(paper.data["capital"], paper.data["stats"])
                snap = db_snapshot()
                if snap:
                    log(f"[DB] Snapshot: ${snap['capital']:.2f}  "
                        f"{snap['wins']}W/{snap['losses']}L  "
                        f"pnl=${snap['total_pnl']:+.2f}")
                state["last_weekly"] = now.isoformat()
                save_state(state)
                last_weekly = now

            log(f"[MAIN] Ciclo OK — próximo en {POLL_MINUTES} min")
            time.sleep(POLL_MINUTES * 60)

        except KeyboardInterrupt:
            log("[MAIN] Bot detenido")
            paper._save()
            enriched = paper.enrich_positions(events if 'events' in dir() else [])
            tg.bot_stop(paper.data["capital"], paper.data["stats"])
            break

        except Exception as e:
            import traceback
            log(f"[ERROR] {e}\n{traceback.format_exc()}")
            tg.send(f"⚠️ <b>Error Signal Bot</b>\n{str(e)[:200]}")
            time.sleep(120)


if __name__ == "__main__":
    main()
