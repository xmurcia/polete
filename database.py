"""
database.py - Capa de persistencia PostgreSQL para el bot de Polymarket.
Reemplaza: trade_history.csv, portfolio.json, live_history.json,
           logs/snapshots/*.json, logs/market_tape/*.json

Conexión: usa DATABASE_URL (Railway la inyecta automáticamente).
"""

import os
import json
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

# ──────────────────────────────────────────────────────────────
# CONEXIÓN
# ──────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL")
DUAL_WRITE_ENABLED = os.environ.get("DUAL_WRITE_MODE", "false").lower() == "true"

# Pool de conexiones (si está configurado)
_pool = None
if DATABASE_URL:
    try:
        _pool = ThreadedConnectionPool(minconn=1, maxconn=5, dsn=DATABASE_URL)
        print("✅ [DB] Conexión establecida")
    except Exception as e:
        print(f"⚠️  [DB] No se pudo conectar: {e}")
        _pool = None
else:
    if DUAL_WRITE_ENABLED:
        print("⚠️  [DB] DUAL_WRITE_MODE=true pero DATABASE_URL no está definida")
    print("ℹ️  [DB] Dual-write desactivado (solo archivos)")


@contextmanager
def get_conn():
    """Context manager que devuelve una conexión del pool y la libera al salir."""
    if not _pool:
        raise RuntimeError("Database connection pool not initialized")
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def is_db_available() -> bool:
    """Verifica si la base de datos está disponible."""
    return _pool is not None and DUAL_WRITE_ENABLED


# ──────────────────────────────────────────────────────────────
# INICIALIZACIÓN DE TABLAS
# ──────────────────────────────────────────────────────────────

SCHEMA = """
-- Historial de trades (reemplaza trade_history.csv)
CREATE TABLE IF NOT EXISTS trades (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action      TEXT NOT NULL,          -- BUY, SELL, ROTATE, BUY_HEDGE, etc.
    market      TEXT NOT NULL,
    bucket      TEXT NOT NULL,
    price       NUMERIC(10,4) NOT NULL,
    shares      NUMERIC(12,4) NOT NULL,
    reason      TEXT,
    pnl         NUMERIC(10,4) DEFAULT 0,
    cash_after  NUMERIC(10,4),
    mode        TEXT DEFAULT 'PAPER',   -- PAPER | REAL
    strategy    TEXT DEFAULT 'STANDARD', -- STANDARD | MOONSHOT | HEDGE | LOTTO
    hours_left  NUMERIC(8,3),           -- Tiempo restante del mercado en horas
    tweet_count INT,                    -- Conteo actual de tweets
    market_consensus NUMERIC(10,4)      -- Consenso del mercado (probabilidad)
);

CREATE INDEX IF NOT EXISTS idx_trades_ts     ON trades (ts DESC);
CREATE INDEX IF NOT EXISTS idx_trades_action ON trades (action);
CREATE INDEX IF NOT EXISTS idx_trades_bucket ON trades (bucket);
CREATE INDEX IF NOT EXISTS idx_trades_hours_left ON trades (hours_left);
CREATE INDEX IF NOT EXISTS idx_trades_tweet_count ON trades (tweet_count);

-- Posiciones abiertas (reemplaza portfolio.json → positions)
CREATE TABLE IF NOT EXISTS positions (
    pos_id          TEXT PRIMARY KEY,   -- "{market}|{bucket}"
    market          TEXT NOT NULL,
    bucket          TEXT NOT NULL,
    shares          NUMERIC(12,4) NOT NULL,
    entry_price     NUMERIC(10,4) NOT NULL,
    current_price   NUMERIC(10,4),
    invested        NUMERIC(10,4) NOT NULL,
    strategy_tag    TEXT DEFAULT 'STANDARD',
    token_id        TEXT,               -- solo en modo REAL
    max_price_seen  NUMERIC(10,4),
    entry_z_score   NUMERIC(10,4),      -- Z-score en el momento de entrada
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mode            TEXT DEFAULT 'PAPER'
);

-- Estado global del bot (reemplaza portfolio.json → cash)
CREATE TABLE IF NOT EXISTS bot_state (
    key     TEXT PRIMARY KEY,
    value   JSONB NOT NULL,
    updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Eventos de tweets en tiempo real (reemplaza live_history.json)
CREATE TABLE IF NOT EXISTS tweet_events (
    id      SERIAL PRIMARY KEY,
    ts_ms   BIGINT NOT NULL,            -- timestamp en milisegundos
    source  TEXT DEFAULT 'live'
);

CREATE INDEX IF NOT EXISTS idx_tweet_events_ts ON tweet_events (ts_ms DESC);

-- Snapshot de contexto en cada trade (reemplaza logs/snapshots/*.json)
-- Guarda el z_score y pnl en el momento exacto del trade para análisis posterior
CREATE TABLE IF NOT EXISTS trade_snapshots (
    id           SERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action       TEXT NOT NULL,          -- BUY, SMART_ROTATE, etc.
    market       TEXT NOT NULL,
    bucket       TEXT NOT NULL,
    price        NUMERIC(10,4),
    reason       TEXT,
    z_score      NUMERIC(10,4),          -- context.z del snapshot original
    pnl_at_trade NUMERIC(10,4),          -- context.pnl (solo en ventas)
    fair_value   NUMERIC(10,4),          -- fair value predicho por el modelo
    hours_left   NUMERIC(8,3),           -- horas restantes del mercado
    tweet_count  INT,                    -- conteo de tweets en el momento
    mode         TEXT DEFAULT 'PAPER'
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ts     ON trade_snapshots (ts DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_bucket ON trade_snapshots (bucket);
CREATE INDEX IF NOT EXISTS idx_snapshots_action ON trade_snapshots (action);

-- Cabecera del tape: un registro por mercado activo por ciclo de tape
-- (reemplaza logs/market_tape/*.json — guardamos cada 30 min)
CREATE TABLE IF NOT EXISTS market_tape (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ts_unix     NUMERIC,
    market      TEXT NOT NULL,
    tweet_count INT,
    hours_left  NUMERIC(8,3),
    daily_avg   NUMERIC(8,3),
    market_id   TEXT,               -- ID del mercado de Polymarket
    is_active   BOOLEAN DEFAULT TRUE -- Si el mercado está activo
);

CREATE INDEX IF NOT EXISTS idx_tape_ts     ON market_tape (ts DESC);
CREATE INDEX IF NOT EXISTS idx_tape_market ON market_tape (market, ts DESC);

-- Precios por bucket para cada entrada del tape
-- Separado para poder hacer: "dame el bid del bucket 300-319 a lo largo del tiempo"
CREATE TABLE IF NOT EXISTS market_tape_prices (
    id         SERIAL PRIMARY KEY,
    tape_id    INT NOT NULL REFERENCES market_tape(id) ON DELETE CASCADE,
    bucket     TEXT NOT NULL,
    bucket_min INT,
    bucket_max INT,
    bid        NUMERIC(8,4),
    ask        NUMERIC(8,4)
);

CREATE INDEX IF NOT EXISTS idx_tape_prices_tape   ON market_tape_prices (tape_id);
CREATE INDEX IF NOT EXISTS idx_tape_prices_bucket ON market_tape_prices (bucket);
"""


def init_db():
    """Crea las tablas si no existen. Llamar una vez al arrancar el bot."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
    print("✅ [DB] Tablas inicializadas correctamente")


# ──────────────────────────────────────────────────────────────
# TRADES
# ──────────────────────────────────────────────────────────────

def log_trade(
    action: str,
    market: str,
    bucket: str,
    price: float,
    shares: float,
    reason: str = "",
    pnl: float = 0.0,
    cash_after: float = 0.0,
    mode: str = "PAPER",
    strategy: str = "STANDARD",
    hours_left: Optional[float] = None,
    tweet_count: Optional[int] = None,
    market_consensus: Optional[float] = None,
):
    """Guarda un trade. Equivalente a _log_trade() del CSV."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trades (action, market, bucket, price, shares, reason, pnl, cash_after, mode, strategy,
                                   hours_left, tweet_count, market_consensus)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (action, market, bucket, price, shares, reason, pnl, cash_after, mode, strategy,
                 hours_left, tweet_count, market_consensus),
            )


def get_trades(limit: int = 200, mode: Optional[str] = None) -> list[dict]:
    """Devuelve los últimos N trades como lista de dicts."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if mode:
                cur.execute(
                    "SELECT * FROM trades WHERE mode = %s ORDER BY ts DESC LIMIT %s",
                    (mode, limit),
                )
            else:
                cur.execute("SELECT * FROM trades ORDER BY ts DESC LIMIT %s", (limit,))
            return [dict(r) for r in cur.fetchall()]


def get_trade_stats(mode: Optional[str] = None) -> dict:
    """Stats agregados: win_rate, pnl_total, pnl_por_hora."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            where = "WHERE mode = %s AND action IN ('SELL','ROTATE','SMART_ROTATE')" if mode else \
                    "WHERE action IN ('SELL','ROTATE','SMART_ROTATE')"
            params = (mode,) if mode else ()

            cur.execute(f"""
                SELECT
                    COUNT(*)                                        AS total_sells,
                    COUNT(*) FILTER (WHERE pnl > 0)                AS wins,
                    COUNT(*) FILTER (WHERE pnl <= 0)               AS losses,
                    COALESCE(SUM(pnl), 0)                          AS pnl_total,
                    COALESCE(AVG(pnl) FILTER (WHERE pnl > 0), 0)  AS avg_win,
                    COALESCE(AVG(pnl) FILTER (WHERE pnl <= 0), 0) AS avg_loss
                FROM trades {where}
            """, params)
            row = dict(cur.fetchone())

            # PnL por hora del día (para detectar horas malas)
            cur.execute(f"""
                SELECT
                    EXTRACT(HOUR FROM ts)::int AS hora,
                    COALESCE(SUM(pnl), 0)      AS pnl_hora
                FROM trades {where}
                GROUP BY hora
                ORDER BY hora
            """, params)
            pnl_por_hora = {r["hora"]: float(r["pnl_hora"]) for r in cur.fetchall()}

            wins = row["wins"] or 0
            losses = row["losses"] or 0
            total = wins + losses
            row["win_rate"] = round((wins / total * 100), 1) if total > 0 else 0.0
            row["pnl_por_hora"] = pnl_por_hora
            return row


# ──────────────────────────────────────────────────────────────
# POSICIONES (paper mode)
# ──────────────────────────────────────────────────────────────

def upsert_position(pos_id: str, data: dict):
    """Inserta o actualiza una posición abierta."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO positions (pos_id, market, bucket, shares, entry_price, current_price,
                    invested, strategy_tag, token_id, max_price_seen, mode, entry_z_score)
                VALUES (%(pos_id)s, %(market)s, %(bucket)s, %(shares)s, %(entry_price)s,
                    %(current_price)s, %(invested)s, %(strategy_tag)s, %(token_id)s,
                    %(max_price_seen)s, %(mode)s, %(entry_z_score)s)
                ON CONFLICT (pos_id) DO UPDATE SET
                    shares          = EXCLUDED.shares,
                    entry_price     = EXCLUDED.entry_price,
                    current_price   = EXCLUDED.current_price,
                    invested        = EXCLUDED.invested,
                    strategy_tag    = EXCLUDED.strategy_tag,
                    token_id        = EXCLUDED.token_id,
                    max_price_seen  = EXCLUDED.max_price_seen,
                    entry_z_score   = EXCLUDED.entry_z_score
                """,
                {
                    "pos_id":        pos_id,
                    "market":        data.get("market", ""),
                    "bucket":        data.get("bucket", ""),
                    "shares":        data.get("shares", 0),
                    "entry_price":   data.get("entry_price", 0),
                    "current_price": data.get("current_price", data.get("entry_price", 0)),
                    "invested":      data.get("invested", 0),
                    "strategy_tag":  data.get("strategy_tag", "STANDARD"),
                    "token_id":      data.get("token_id"),
                    "max_price_seen":data.get("max_price_seen"),
                    "mode":          data.get("mode", "PAPER"),
                    "entry_z_score": data.get("entry_z_score"),
                },
            )


def delete_position(pos_id: str):
    """Elimina una posición (al cerrarla)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM positions WHERE pos_id = %s", (pos_id,))


def get_positions(mode: Optional[str] = None) -> dict:
    """Devuelve todas las posiciones abiertas como dict {pos_id: data}."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if mode:
                cur.execute("SELECT * FROM positions WHERE mode = %s", (mode,))
            else:
                cur.execute("SELECT * FROM positions")
            rows = cur.fetchall()
            return {r["pos_id"]: dict(r) for r in rows}


# ──────────────────────────────────────────────────────────────
# BOT STATE (cash y otros valores globales)
# ──────────────────────────────────────────────────────────────

def set_state(key: str, value):
    """Guarda un valor en bot_state."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bot_state (key, value, updated) VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated = NOW()
                """,
                (key, json.dumps(value)),
            )


def get_state(key: str, default=None):
    """Lee un valor de bot_state."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM bot_state WHERE key = %s", (key,))
            row = cur.fetchone()
            if not row:
                return default
            # JSONB ya está deserializado por psycopg2, no necesita json.loads()
            return row[0]


# ──────────────────────────────────────────────────────────────
# TWEET EVENTS (reemplaza live_history.json)
# ──────────────────────────────────────────────────────────────

def insert_tweet_event(ts_ms: int):
    """Registra un nuevo evento de tweet."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tweet_events (ts_ms) VALUES (%s)",
                (ts_ms,),
            )


def get_recent_tweet_events(hours: int = 24) -> list[int]:
    """Devuelve timestamps (ms) de tweets de las últimas N horas."""
    cutoff_ms = int((time.time() - hours * 3600) * 1000)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts_ms FROM tweet_events WHERE ts_ms > %s ORDER BY ts_ms ASC",
                (cutoff_ms,),
            )
            return [row[0] for row in cur.fetchall()]


def prune_tweet_events(hours: int = 48):
    """Limpia tweets más antiguos de N horas (mantenimiento)."""
    cutoff_ms = int((time.time() - hours * 3600) * 1000)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tweet_events WHERE ts_ms < %s", (cutoff_ms,))


# ──────────────────────────────────────────────────────────────
# SHADOW WRITE — dual-write seguro durante la transición
# ──────────────────────────────────────────────────────────────

def shadow_write(fn, *args, **kwargs):
    """
    Envuelve cualquier función de escritura en DB para que un fallo
    nunca interrumpa el flujo del bot.
    Los ficheros siguen siendo la fuente de verdad durante la transición.
    """
    if not is_db_available():
        return  # Dual-write desactivado, skip silenciosamente

    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"⚠️  [DB shadow] {fn.__name__}: {e}")


# ──────────────────────────────────────────────────────────────
# TRADE SNAPSHOTS (reemplaza logs/snapshots/*.json)
# ──────────────────────────────────────────────────────────────

def log_snapshot(
    action: str,
    market: str,
    bucket: str,
    price: float,
    reason: str,
    context: dict,
    mode: str = "PAPER",
    hours_left: Optional[float] = None,
    tweet_count: Optional[int] = None,
):
    """
    Guarda el contexto completo de un trade (z_score, pnl en el momento).
    Llamar donde antes se llamaba save_trade_snapshot().
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Convert numpy types to Python native types
            fair_value = context.get("fair")
            if fair_value is not None:
                fair_value = float(fair_value)

            z_score = context.get("z")
            if z_score is not None:
                z_score = float(z_score)

            pnl_value = context.get("pnl")
            if pnl_value is not None:
                pnl_value = float(pnl_value)

            cur.execute(
                """
                INSERT INTO trade_snapshots (action, market, bucket, price, reason, z_score, pnl_at_trade, mode,
                                            fair_value, hours_left, tweet_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    action,
                    market,
                    bucket,
                    price,
                    reason,
                    z_score,
                    pnl_value,
                    mode,
                    fair_value,
                    hours_left,
                    tweet_count,
                ),
            )


def get_snapshots(limit: int = 100, action: Optional[str] = None) -> list[dict]:
    """Devuelve los últimos N snapshots, opcionalmente filtrados por acción."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if action:
                cur.execute(
                    "SELECT * FROM trade_snapshots WHERE action = %s ORDER BY ts DESC LIMIT %s",
                    (action, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM trade_snapshots ORDER BY ts DESC LIMIT %s", (limit,)
                )
            return [dict(r) for r in cur.fetchall()]


# ──────────────────────────────────────────────────────────────
# MARKET TAPE (reemplaza logs/market_tape/*.json)
# ──────────────────────────────────────────────────────────────

def save_tape(ts_unix: float, meta: list, order_book: list):
    """
    Persiste un snapshot del order book completo.
    Llamar donde antes se llamaba save_market_tape().

    Args:
        ts_unix:    timestamp unix del snapshot (float)
        meta:       lista de dicts con {id, title, count, hours, daily_avg}
        order_book: lista de dicts con {title, buckets: [{bucket, min, max, bid, ask}]}
    """
    # Construir mapa title → meta para enriquecer cada fila del tape
    meta_map = {m["title"].strip().lower(): m for m in meta}

    with get_conn() as conn:
        with conn.cursor() as cur:
            for market_ob in order_book:
                market_title = market_ob.get("title", "").strip()
                market_key = market_title.lower()

                # Buscar la meta correspondiente (matching flexible)
                market_meta = meta_map.get(market_key)
                if not market_meta:
                    # Fallback: buscar por overlap de palabras
                    for k, v in meta_map.items():
                        if market_key in k or k in market_key:
                            market_meta = v
                            break

                tweet_count = market_meta["count"] if market_meta else None
                hours_left  = market_meta["hours"] if market_meta else None
                daily_avg   = market_meta["daily_avg"] if market_meta else None
                market_id   = market_meta["id"] if market_meta else None
                is_active   = market_meta.get("active", True) if market_meta else True

                # Insertar cabecera del tape
                cur.execute(
                    """
                    INSERT INTO market_tape (ts_unix, market, tweet_count, hours_left, daily_avg, market_id, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (ts_unix, market_title, tweet_count, hours_left, daily_avg, market_id, is_active),
                )
                tape_id = cur.fetchone()[0]

                # Insertar precios de cada bucket
                buckets = market_ob.get("buckets", [])
                if buckets:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO market_tape_prices (tape_id, bucket, bucket_min, bucket_max, bid, ask)
                        VALUES %s
                        """,
                        [
                            (
                                tape_id,
                                b.get("bucket"),
                                b.get("min"),
                                b.get("max") if b.get("max", 0) < 99999 else None,
                                b.get("bid"),
                                b.get("ask"),
                            )
                            for b in buckets
                        ],
                    )


def get_tape_for_bucket(bucket: str, market_pattern: str, limit: int = 200) -> list[dict]:
    """
    Serie temporal de bid/ask para un bucket concreto.
    Útil para analizar cómo evoluciona el precio a medida que bajan las horas.

    Ejemplo:
        get_tape_for_bucket("300-319", "February 6")
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT t.ts, t.market, t.tweet_count, t.hours_left, t.daily_avg,
                       p.bid, p.ask
                FROM market_tape t
                JOIN market_tape_prices p ON p.tape_id = t.id
                WHERE p.bucket = %s
                  AND t.market ILIKE %s
                ORDER BY t.ts DESC
                LIMIT %s
                """,
                (bucket, f"%{market_pattern}%", limit),
            )
            return [dict(r) for r in cur.fetchall()]


# El tape es el historial más valioso — nunca se borra.
# Si en el futuro el volumen fuera un problema, archivar a cold storage
# en lugar de eliminar.


# ──────────────────────────────────────────────────────────────
# EXPORTACIÓN — reconstruye JSONs compatibles con backtesting
# ──────────────────────────────────────────────────────────────

def export_tape_as_json(
    market_pattern: str = None,
    date_from: str = None,
    date_to: str = None,
) -> list[dict]:
    """
    Reconstruye el formato original de tape_*.json para backtesting.

    Ejemplos:
        export_tape_as_json(market_pattern="February 6")
        export_tape_as_json(date_from="2026-02-01", date_to="2026-02-07")

    Returns:
        Lista de snapshots con el mismo formato que save_market_tape() genera:
        [{"timestamp": ..., "meta": [...], "order_book": [...]}, ...]
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # 1. Cabeceras filtradas
            where_clauses = []
            params = []
            if market_pattern:
                where_clauses.append("t.market ILIKE %s")
                params.append(f"%{market_pattern}%")
            if date_from:
                where_clauses.append("t.ts >= %s")
                params.append(date_from)
            if date_to:
                where_clauses.append("t.ts <= %s")
                params.append(date_to)

            where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            cur.execute(
                f"""
                SELECT DISTINCT ON (t.ts_unix, t.market)
                    t.id, t.ts, t.ts_unix, t.market,
                    t.tweet_count, t.hours_left, t.daily_avg, t.market_id, t.is_active
                FROM market_tape t
                {where_sql}
                ORDER BY t.ts_unix, t.market, t.ts
                """,
                params,
            )
            tape_headers = cur.fetchall()

            if not tape_headers:
                return []

            # 2. Reconstruir snapshot completo por ts_unix
            snapshots_by_ts: dict = {}

            for row in tape_headers:
                ts_unix = float(row["ts_unix"])
                if ts_unix not in snapshots_by_ts:
                    snapshots_by_ts[ts_unix] = {
                        "timestamp": ts_unix,
                        "meta": [],
                        "order_book": [],
                    }

                # Meta
                snapshots_by_ts[ts_unix]["meta"].append({
                    "id":        row["market_id"],
                    "title":     row["market"],
                    "count":     row["tweet_count"],
                    "hours":     float(row["hours_left"]) if row["hours_left"] else None,
                    "daily_avg": float(row["daily_avg"]) if row["daily_avg"] else None,
                    "active":    row.get("is_active", True),
                })

                # Buckets
                cur.execute(
                    """
                    SELECT bucket, bucket_min, bucket_max, bid, ask
                    FROM market_tape_prices
                    WHERE tape_id = %s
                    ORDER BY bucket_min ASC NULLS LAST
                    """,
                    (row["id"],),
                )
                buckets = cur.fetchall()

                snapshots_by_ts[ts_unix]["order_book"].append({
                    "title": row["market"],
                    "buckets": [
                        {
                            "bucket": b["bucket"],
                            "min":    b["bucket_min"],
                            "max":    b["bucket_max"] if b["bucket_max"] else 99999,
                            "bid":    float(b["bid"]) if b["bid"] else 0.0,
                            "ask":    float(b["ask"]) if b["ask"] else 0.0,
                        }
                        for b in buckets
                    ],
                })

            return list(snapshots_by_ts.values())


def dump_tape_to_files(output_dir: str, **kwargs) -> int:
    """
    Exporta el tape de DB a ficheros tape_*.json con el mismo formato
    que genera el bot hoy — compatible con backtesting sin tocar nada.

    Args:
        output_dir: Directorio de salida (se crea si no existe)
        **kwargs:   Filtros opcionales pasados a export_tape_as_json()
                    (market_pattern, date_from, date_to)

    Ejemplos:
        dump_tape_to_files("backtest_logs/feb6_feb13",
                           market_pattern="February 6",
                           date_from="2026-02-06",
                           date_to="2026-02-13")

        dump_tape_to_files("backtest_logs/all")  # Todo el historico

    Returns:
        Numero de ficheros exportados
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    snapshots = export_tape_as_json(**kwargs)

    for snap in snapshots:
        ts_str = datetime.fromtimestamp(snap["timestamp"]).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"tape_{ts_str}.json")
        with open(path, "w") as f:
            json.dump(snap, f)

    print(f"Exportados {len(snapshots)} snapshots -> {output_dir}/")
    return len(snapshots)