# Database Schema - Polymarket Bot

## Diagrama de Relaciones

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRADING OPERATIONS                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│      TRADES          │         │   TRADE_SNAPSHOTS    │
├──────────────────────┤         ├──────────────────────┤
│ 🔑 id (PK)           │         │ 🔑 id (PK)           │
│ 📅 ts                │         │ 📅 ts                │
│ 📝 action            │         │ 📝 action            │
│ 🏪 market            │         │ 🏪 market            │
│ 📦 bucket            │         │ 📦 bucket            │
│ 💰 price             │         │ 💰 price             │
│ 📊 shares            │         │ 📝 reason            │
│ 📝 reason            │         │ 📈 z_score           │
│ 💵 pnl               │         │ 💰 pnl_at_trade      │
│ 💰 cash_after        │         │ 🎮 mode              │
│ 🎮 mode              │         └──────────────────────┘
│ 🎯 strategy          │
└──────────────────────┘
     Historial             Contexto detallado
     completo              en cada trade

┌──────────────────────┐
│     POSITIONS        │
├──────────────────────┤
│ 🔑 pos_id (PK)       │  "{market}|{bucket}"
│ 🏪 market            │
│ 📦 bucket            │
│ 📊 shares            │
│ 💰 entry_price       │
│ 💰 current_price     │
│ 💵 invested          │
│ 🎯 strategy_tag      │
│ 🎫 token_id          │  (solo REAL mode)
│ 📈 max_price_seen    │  (para trailing stop)
│ 📅 opened_at         │
│ 🎮 mode              │
└──────────────────────┘
   Posiciones abiertas
   (se borran al cerrar)

┌─────────────────────────────────────────────────────────────────────────────┐
│                              BOT STATE                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│     BOT_STATE        │
├──────────────────────┤
│ 🔑 key (PK)          │  "cash", "config", etc.
│ 📦 value (JSONB)     │  Flexible JSON storage
│ 📅 updated           │
└──────────────────────┘
   Estado global
   del bot

┌─────────────────────────────────────────────────────────────────────────────┐
│                          MARKET DATA & TAPE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│    MARKET_TAPE       │    1    │ MARKET_TAPE_PRICES   │
├──────────────────────┤───────N─├──────────────────────┤
│ 🔑 id (PK)           │         │ 🔑 id (PK)           │
│ 📅 ts                │         │ 🔗 tape_id (FK)      │────┐
│ 📅 ts_unix           │         │ 📦 bucket            │    │
│ 🏪 market            │         │ 🔢 bucket_min        │    │
│ 🐦 tweet_count       │         │ 🔢 bucket_max        │    │
│ ⏱️  hours_left        │         │ 💰 bid               │    │
│ 📊 daily_avg         │         │ 💰 ask               │    │
└──────────────────────┘         └──────────────────────┘    │
   Cabecera del                    Precios por bucket        │
   snapshot                        en ese momento             │
                                                              │
                                                              │
        Relación: 1 tape → N precios (buckets)  ─────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            TWEET EVENTS                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│   TWEET_EVENTS       │
├──────────────────────┤
│ 🔑 id (PK)           │
│ ⏱️  ts_ms (BIGINT)    │  Timestamp en milisegundos
│ 📝 source            │  'live' | 'historical'
└──────────────────────┘
   Eventos de tweets
   para Hawkes Process
```

## Índices Importantes

### TRADES
- `idx_trades_ts` → Ordenar por tiempo (DESC)
- `idx_trades_action` → Filtrar por tipo de trade
- `idx_trades_bucket` → Analizar buckets específicos

### MARKET_TAPE
- `idx_tape_ts` → Evolución temporal
- `idx_tape_market` → Filtrar por mercado + tiempo

### MARKET_TAPE_PRICES
- `idx_tape_prices_bucket` → Evolución de precio de un bucket
- `idx_tape_prices_tape` → JOIN rápido con market_tape

### TWEET_EVENTS
- `idx_tweet_events_ts` → Ventana temporal para Hawkes

## Campos Clave por Uso

### Para Análisis de Performance
```sql
-- Win rate y P&L
SELECT * FROM trades WHERE action IN ('SELL', 'ROTATE');

-- Mejores buckets
SELECT bucket, COUNT(*), AVG(pnl)
FROM trades
WHERE action = 'SELL'
GROUP BY bucket
ORDER BY AVG(pnl) DESC;
```

### Para Backtesting
```sql
-- Reconstruir tape completo
SELECT t.*, p.*
FROM market_tape t
JOIN market_tape_prices p ON p.tape_id = t.id
WHERE t.market LIKE '%February 13%'
ORDER BY t.ts_unix, p.bucket_min;
```

### Para Debugging en Vivo
```sql
-- Último estado del bot
SELECT * FROM bot_state;

-- Posiciones abiertas
SELECT * FROM positions WHERE mode = 'REAL';

-- Últimos 10 trades
SELECT * FROM trades ORDER BY ts DESC LIMIT 10;
```

### Para Análisis de Estrategias
```sql
-- Performance por estrategia
SELECT
    strategy,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::float / COUNT(*) * 100 as win_rate,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl
FROM trades
WHERE action IN ('SELL', 'ROTATE')
GROUP BY strategy;
```

## Campos que Necesitas Revisar

### ¿Falta algo para tu bot?

Revisa si necesitas agregar:

1. **En TRADES**:
   - ✅ `action` (BUY, SELL, ROTATE, etc.)
   - ✅ `market` (título del mercado)
   - ✅ `bucket` (ej: "300-319")
   - ✅ `price` (precio de ejecución)
   - ✅ `shares` (cantidad)
   - ✅ `reason` (motivo del trade)
   - ✅ `pnl` (profit/loss)
   - ✅ `cash_after` (cash después del trade)
   - ✅ `mode` (PAPER | REAL)
   - ✅ `strategy` (STANDARD | MOONSHOT | HEDGE)
   - ❓ **¿Falta algo?** → hours_left, tweet_count al momento del trade?

2. **En POSITIONS**:
   - ✅ `pos_id` (identificador único)
   - ✅ `market`, `bucket`
   - ✅ `shares`, `entry_price`, `current_price`
   - ✅ `invested` (capital invertido)
   - ✅ `strategy_tag` (STANDARD | MOONSHOT)
   - ✅ `token_id` (para modo REAL)
   - ✅ `max_price_seen` (para trailing stop)
   - ✅ `opened_at` (timestamp de apertura)
   - ✅ `mode` (PAPER | REAL)
   - ❓ **¿Falta algo?** → stop_loss_price, take_profit_price?

3. **En TRADE_SNAPSHOTS**:
   - ✅ `action`, `market`, `bucket`, `price`
   - ✅ `reason` (explicación del trade)
   - ✅ `z_score` (distancia estadística)
   - ✅ `pnl_at_trade` (P&L en el momento)
   - ❓ **¿Falta algo?** → tweet_count, hours_left, prediction?

4. **En MARKET_TAPE**:
   - ✅ `ts_unix` (timestamp)
   - ✅ `market` (título)
   - ✅ `tweet_count` (contador actual)
   - ✅ `hours_left` (tiempo restante)
   - ✅ `daily_avg` (promedio diario)
   - ❓ **¿Falta algo?** → hawkes_prediction, market_consensus?

5. **En BOT_STATE**:
   - ✅ `cash` (efectivo disponible)
   - ✅ Flexible con JSONB para cualquier config
   - ❓ **¿Necesitas guardar más?** → configuración de estrategias, parámetros del bot?

## Próximos Pasos

Una vez revises el schema en DBeaver:

1. ✅ Verificar que todas las columnas necesarias están
2. ✅ Revisar tipos de datos (numeric precision, etc.)
3. ✅ Confirmar que los índices cubren tus queries más frecuentes
4. ➡️ Empezar integración de dual-write en el código del bot
