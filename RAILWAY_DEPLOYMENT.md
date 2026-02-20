# 🚂 Despliegue en Railway - Modo REAL con Dual-Write

## ✅ Pre-requisitos Completados

- [x] Base de datos PostgreSQL creada en Railway
- [x] Schema migrado con 7 tablas + índices
- [x] Rama `feat/implement-real-trader` con dual-write implementado
- [x] Procfile actualizado: `python main.py --real-trading`

---

## 🔧 Paso 1: Configurar Variables de Entorno

En Railway Dashboard:
1. Ve a tu proyecto
2. Click en el servicio del bot (polete)
3. Pestaña **"Variables"**
4. Agregar estas variables:

### Variables Obligatorias

```bash
# Base de Datos (usar Private Network para mejor performance)
DATABASE_URL=postgresql://postgres:PASSWORD@postgres.railway.internal:5432/railway
DUAL_WRITE_MODE=true

# API de Polymarket (MODO REAL)
POLYMARKET_API_KEY=tu_api_key_aqui
POLYMARKET_API_SECRET=tu_api_secret_aqui
POLYMARKET_API_PASSPHRASE=tu_passphrase_aqui

# Telegram (Notificaciones)
TELEGRAM_BOT_TOKEN=tu_bot_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id_aqui

# Python Runtime
PYTHONUNBUFFERED=1
```

### Cómo obtener DATABASE_URL (Private Network)

En Railway:
1. Click en servicio PostgreSQL
2. Tab **"Connect"**
3. Click **"Private Network"**
4. Copiar la URL que se muestra (formato: `postgresql://postgres:...@postgres.railway.internal:5432/railway`)

**IMPORTANTE:** Usa **Private Network** (`.railway.internal`) porque el bot correrá dentro de Railway, no externamente.

---

## 🚀 Paso 2: Configurar el Despliegue

### Opción A: Desde Railway Dashboard

1. Ve a tu servicio del bot
2. Tab **"Settings"**
3. Sección **"Deploy"**
4. Verifica:
   - **Branch**: `feat/implement-real-trader`
   - **Root Directory**: `/` (raíz del proyecto)
   - **Start Command**: (vacío, usará Procfile automáticamente)

### Opción B: Desde CLI

```bash
# Conectar a tu servicio del bot
railway service polete

# Configurar branch de despliegue
railway environment production

# Push para desplegar
git push railway feat/implement-real-trader:main
```

---

## 📦 Paso 3: Verificar Archivos de Configuración

### Procfile (ya actualizado)
```
worker: python main.py --real-trading
```

### requirements.txt
Verificar que incluya:
```
python-dotenv>=1.0.0
psycopg2-binary>=2.9.9
requests>=2.31.0
numpy>=1.26.0
pandas>=2.1.0
scipy>=1.11.0
py-clob-client>=0.15.0
```

### .gitignore
Verificar que NO incluya:
- `main.py`
- `config.py`
- `src/` directory
- `database.py`

---

## 🔍 Paso 4: Verificar el Despliegue

### 4.1 Logs de Deploy

En Railway Dashboard:
1. Tab **"Deployments"**
2. Click en el deployment más reciente
3. Ver logs en tiempo real

**Buscar en los logs:**
```
✅ [DB] Conexión establecida
🤖 ELON-BOT V12.16
🔴 REAL TRADING MODE - Using real money!
[UnifiedTrader] 🔴 REAL TRADING MODE
```

### 4.2 Verificar Conexión a DB

Desde `railway connect postgres`:
```sql
-- Ver si el bot está escribiendo market_tape
SELECT COUNT(*), MAX(ts) FROM market_tape;

-- Ver últimas escrituras
SELECT ts, market, tweet_count, hours_left
FROM market_tape
ORDER BY ts DESC
LIMIT 5;
```

### 4.3 Monitorear Variables

```bash
railway logs --tail 100
```

---

## ⚠️ Checklist Pre-Lanzamiento (MODO REAL)

Antes de activar trading real, verifica:

- [ ] **Credenciales Polymarket válidas** (API Key, Secret, Passphrase)
- [ ] **Balance suficiente** en Polymarket (mínimo $100-500 para empezar)
- [ ] **Telegram configurado** (recibirás notificaciones de trades)
- [ ] **DATABASE_URL correcta** (Private Network: `postgres.railway.internal`)
- [ ] **DUAL_WRITE_MODE=true** (escritura en DB + archivos)
- [ ] **Procfile correcto** (`python main.py --real-trading`)
- [ ] **Logs muestran "REAL TRADING MODE"**
- [ ] **DB recibiendo datos** (market_tape actualizándose)

---

## 🎯 Paso 5: Activar Trading Real

### 5.1 Desplegar

```bash
# Commit el Procfile actualizado
git add Procfile
git commit -m "feat: configure Railway for real trading mode with dual-write"
git push origin feat/implement-real-trader

# Railway detectará el push y desplegará automáticamente
```

### 5.2 Monitorear el Bot

```bash
# Ver logs en tiempo real
railway logs --tail

# Ver logs del servicio específico
railway logs --service polete --tail
```

### 5.3 Verificar Primer Trade

Cuando el bot haga su primer trade real:

**En Logs:**
```
✅ BUY: $40.00
💬 Telegram: Trade ejecutado correctamente
```

**En Railway PostgreSQL:**
```sql
SELECT * FROM trades WHERE mode='REAL' ORDER BY ts DESC LIMIT 1;
```

**En Polymarket:**
- Ve a tu cuenta
- Tab "Portfolio"
- Deberías ver la posición abierta

---

## 📊 Monitoreo Post-Despliegue

### Queries Útiles

```sql
-- Resumen de trading real
SELECT
    DATE(ts) as day,
    COUNT(*) as trades,
    SUM(CASE WHEN action LIKE '%BUY%' THEN 1 ELSE 0 END) as buys,
    SUM(CASE WHEN action LIKE '%SELL%' THEN 1 ELSE 0 END) as sells,
    SUM(pnl) as total_pnl,
    AVG(pnl) FILTER (WHERE pnl != 0) as avg_pnl
FROM trades
WHERE mode = 'REAL'
GROUP BY day
ORDER BY day DESC;

-- Posiciones abiertas
SELECT
    pos_id,
    market,
    bucket,
    shares,
    entry_price,
    current_price,
    strategy_tag,
    entry_z_score,
    opened_at
FROM positions
WHERE mode = 'REAL';

-- Market tape health check
SELECT
    COUNT(*) as snapshots,
    MIN(ts) as first_snapshot,
    MAX(ts) as last_snapshot,
    COUNT(DISTINCT market) as unique_markets
FROM market_tape
WHERE ts > NOW() - INTERVAL '1 hour';
```

---

## 🚨 Troubleshooting

### Error: "DB connection failed"
- **Causa:** DATABASE_URL incorrecta
- **Solución:** Verificar que uses Private Network (`postgres.railway.internal`)

### Error: "Polymarket authentication failed"
- **Causa:** API credentials incorrectas o expiradas
- **Solución:** Regenerar credentials en Polymarket dashboard

### Error: "Insufficient balance"
- **Causa:** No hay fondos suficientes en Polymarket
- **Solución:** Depositar USDC en tu cuenta

### Bot no escribe en DB
- **Causa:** DUAL_WRITE_MODE no está en `true`
- **Solución:** Verificar variables de entorno en Railway

### Logs muestran "PAPER TRADING MODE"
- **Causa:** Procfile no tiene `--real-trading` flag
- **Solución:** Actualizar Procfile y redesplegar

---

## 🔄 Rollback en Caso de Emergencia

Si algo sale mal:

```bash
# Opción 1: Detener el servicio
railway down

# Opción 2: Cambiar a branch anterior
railway environment production
# Ir a Settings > Deploy > Change branch

# Opción 3: Pausar temporalmente
# Railway Dashboard > Service > Settings > Sleep Service
```

---

## 📝 Notas Importantes

1. **Private Network vs Public Network:**
   - Usa **Private Network** para el bot (está dentro de Railway)
   - Usa **Public Network** solo para acceso externo (DBeaver, etc.)

2. **Dual-Write Garantizado:**
   - Archivos JSON/CSV se siguen generando en el volumen montado
   - DB recibe los mismos datos simultáneamente
   - Si DB falla, el bot continúa (shadow mode)

3. **Volumen Persistente:**
   - Railway tiene un volumen montado en `/app/logs`
   - Los archivos `logs/*.json` y `logs/*.csv` persisten entre deploys

4. **Costos:**
   - Railway cobra por:
     - Compute time (bot corriendo)
     - PostgreSQL storage
     - Data transfer (minimal)
   - Estima: ~$5-10/mes para bot + DB pequeña

---

## ✅ Resultado Final

Después de completar estos pasos:

1. ✅ Bot corriendo en Railway en modo REAL
2. ✅ Conectado a PostgreSQL via Private Network
3. ✅ Dual-write activo (archivos + DB)
4. ✅ Notificaciones Telegram funcionando
5. ✅ Trading real ejecutándose con dinero real
6. ✅ Todos los datos capturados en DB para análisis

---

## 🎯 Siguiente Fase: Análisis con IA

Una vez que tengas datos acumulados:

```sql
-- Exportar datos para análisis
\copy (SELECT * FROM trades WHERE mode='REAL') TO 'trades_real.csv' CSV HEADER;
\copy (SELECT * FROM market_tape WHERE ts > NOW() - INTERVAL '7 days') TO 'market_tape_7days.csv' CSV HEADER;
```

Usar estos datos para:
- Entrenar modelos predictivos
- Optimizar parámetros de trading
- Analizar correlaciones entre hours_left, tweet_count y PnL
- Detectar patrones en estrategias ganadoras
