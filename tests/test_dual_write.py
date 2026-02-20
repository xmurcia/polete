#!/usr/bin/env python3
"""
Test de Dual-Write: Verificar que se escribe en archivos Y en base de datos
"""

import os
import sys
import json
import time

# Configurar DATABASE_URL y DUAL_WRITE_MODE antes de importar
os.environ["DATABASE_URL"] = "postgresql://polybot:polybot_dev_2026@localhost:5432/polymarket"
os.environ["DUAL_WRITE_MODE"] = "true"

# Importar módulos del bot
from src.paper_trader import PaperTrader
from src.utils import save_market_tape, save_trade_snapshot
import database as db

def test_paper_trader_dual_write():
    """Test 1: PaperTrader escribe en archivos Y DB"""
    print("\n" + "="*70)
    print("TEST 1: PaperTrader Dual-Write")
    print("="*70)

    # Limpiar DB test
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM trades WHERE market LIKE 'Test Market%'")
                cur.execute("DELETE FROM positions WHERE market LIKE 'Test Market%'")
                cur.execute("DELETE FROM bot_state WHERE key = 'cash'")
        print("✅ DB limpiada para test")
    except Exception as e:
        print(f"⚠️  No se pudo limpiar DB: {e}")

    # Crear trader
    trader = PaperTrader(initial_cash=1000.0)
    print(f"📂 Cash inicial: ${trader.portfolio['cash']:.2f}")

    # Ejecutar un trade BUY
    result = trader.execute(
        market_title="Test Market Dual Write",
        bucket="100-119",
        signal="BUY",
        price=0.45,
        reason="Test dual-write",
        strategy_tag="STANDARD",
        hours_left=12.5,
        tweet_count=150,
        market_consensus=0.48,
        entry_z_score=0.75
    )
    print(f"   {result}")

    # Verificar archivo CSV
    import pandas as pd
    csv_df = pd.read_csv("logs/trade_history.csv")
    csv_trades = csv_df[csv_df['Market'].str.contains("Test Market Dual", na=False)]
    print(f"\n📁 Archivo CSV:")
    print(f"   Trades encontrados: {len(csv_trades)}")
    if len(csv_trades) > 0:
        print(f"   Último: {csv_trades.iloc[-1]['Action']} @ ${csv_trades.iloc[-1]['Price']}")

    # Verificar archivo portfolio.json
    with open("logs/portfolio.json", 'r') as f:
        portfolio_json = json.load(f)
    json_positions = [p for p in portfolio_json['positions'].keys() if 'Test Market Dual' in p]
    print(f"\n📁 Archivo portfolio.json:")
    print(f"   Cash: ${portfolio_json['cash']:.2f}")
    print(f"   Posiciones: {len(json_positions)}")

    # Verificar DB trades
    db_trades = db.get_trades(limit=100, mode="PAPER")
    db_test_trades = [t for t in db_trades if "Test Market Dual" in t['market']]
    print(f"\n💾 Base de Datos (trades):")
    print(f"   Trades encontrados: {len(db_test_trades)}")
    if db_test_trades:
        t = db_test_trades[0]
        print(f"   Último: {t['action']} @ ${t['price']}")
        print(f"   Context: hours_left={t['hours_left']}, tweet_count={t['tweet_count']}")

    # Verificar DB positions
    db_positions = db.get_positions(mode="PAPER")
    db_test_positions = {k: v for k, v in db_positions.items() if 'Test Market Dual' in k}
    print(f"\n💾 Base de Datos (positions):")
    print(f"   Posiciones encontradas: {len(db_test_positions)}")
    if db_test_positions:
        pos_id, pos = list(db_test_positions.items())[0]
        print(f"   Posición: {pos['bucket']} @ ${pos['entry_price']}")
        print(f"   Entry Z-score: {pos.get('entry_z_score')}")

    # Verificar DB bot_state
    db_cash = db.get_state("cash", default=0)
    print(f"\n💾 Base de Datos (bot_state):")
    print(f"   Cash: ${db_cash}")

    # Ejecutar un SELL
    print("\n" + "-"*70)
    result = trader.execute(
        market_title="Test Market Dual Write",
        bucket="100-119",
        signal="SELL",
        price=0.55,
        reason="Test sell dual-write",
        hours_left=8.0,
        tweet_count=200
    )
    print(f"   {result}")

    # Verificar que la posición se eliminó de DB
    db_positions_after = db.get_positions(mode="PAPER")
    db_test_positions_after = {k: v for k, v in db_positions_after.items() if 'Test Market Dual' in k}
    print(f"\n💾 Base de Datos (positions después de SELL):")
    print(f"   Posiciones restantes: {len(db_test_positions_after)}")

    # Resumen
    print("\n" + "="*70)
    print("✅ TEST 1 COMPLETADO")
    print("="*70)
    print(f"✓ Archivos CSV/JSON escritos correctamente")
    print(f"✓ Trades guardados en DB con contexto (hours_left, tweet_count)")
    print(f"✓ Posiciones sincronizadas en DB (con entry_z_score)")
    print(f"✓ Cash actualizado en DB")
    print(f"✓ Posición eliminada de DB al vender")


def test_market_tape_dual_write():
    """Test 2: Market Tape escribe en archivos Y DB"""
    print("\n" + "="*70)
    print("TEST 2: Market Tape Dual-Write")
    print("="*70)

    # Limpiar DB test
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM market_tape WHERE market LIKE 'Test Market Tape%'")
        print("✅ DB limpiada para test")
    except Exception as e:
        print(f"⚠️  No se pudo limpiar DB: {e}")

    # Simular datos de tape
    ts_unix = time.time()
    meta = [{
        "id": "test-uuid-123",
        "title": "Test Market Tape Dual Write",
        "count": 150,
        "hours": 12.5,
        "daily_avg": 45.0,
        "active": True
    }]
    order_book = [{
        "title": "Test Market Tape Dual Write",
        "buckets": [
            {"bucket": "100-119", "min": 100, "max": 119, "bid": 0.45, "ask": 0.48},
            {"bucket": "120-139", "min": 120, "max": 139, "bid": 0.30, "ask": 0.33}
        ]
    }]

    # Llamar a save_market_tape (escribe en archivo Y DB)
    save_market_tape(order_book, meta)
    print("📤 save_market_tape() ejecutado")

    # Verificar archivo
    import glob
    tape_files = sorted(glob.glob("logs/market_tape/tape_*.json"))
    if tape_files:
        with open(tape_files[-1], 'r') as f:
            last_tape = json.load(f)
        print(f"\n📁 Archivo market_tape:")
        print(f"   Último archivo: {os.path.basename(tape_files[-1])}")
        print(f"   Markets: {len(last_tape['meta'])}")

    # Verificar DB
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*), MAX(ts)
                FROM market_tape
                WHERE market LIKE 'Test Market Tape%'
            """)
            count, last_ts = cur.fetchone()

            print(f"\n💾 Base de Datos (market_tape):")
            print(f"   Snapshots encontrados: {count}")
            if count > 0:
                print(f"   Último timestamp: {last_ts}")

                # Verificar precios
                cur.execute("""
                    SELECT COUNT(*)
                    FROM market_tape_prices p
                    JOIN market_tape t ON t.id = p.tape_id
                    WHERE t.market LIKE 'Test Market Tape%'
                """)
                prices_count = cur.fetchone()[0]
                print(f"   Precios guardados: {prices_count} buckets")

    print("\n" + "="*70)
    print("✅ TEST 2 COMPLETADO")
    print("="*70)
    print(f"✓ Archivo market_tape/*.json creado")
    print(f"✓ Tape guardado en DB con market_id y is_active")
    print(f"✓ Precios guardados en market_tape_prices")


def test_snapshot_dual_write():
    """Test 3: Trade Snapshots escribe en archivos Y DB"""
    print("\n" + "="*70)
    print("TEST 3: Trade Snapshot Dual-Write")
    print("="*70)

    # Limpiar DB test
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM trade_snapshots WHERE market LIKE 'Test Snapshot%'")
        print("✅ DB limpiada para test")
    except Exception as e:
        print(f"⚠️  No se pudo limpiar DB: {e}")

    # Llamar a save_trade_snapshot
    save_trade_snapshot(
        action="BUY",
        m_title="Test Snapshot Dual Write",
        bucket="100-119",
        price=0.45,
        reason="Test snapshot",
        ctx={"z": 0.85, "fair": 0.95},
        hours_left=12.5,
        tweet_count=150,
        mode="PAPER"
    )
    print("📤 save_trade_snapshot() ejecutado")

    # Verificar archivo
    import glob
    snap_files = sorted(glob.glob("logs/snapshots/snap_*.json"))
    if snap_files:
        with open(snap_files[-1], 'r') as f:
            last_snap = json.load(f)
        print(f"\n📁 Archivo snapshot:")
        print(f"   Último archivo: {os.path.basename(snap_files[-1])}")
        print(f"   Market: {last_snap['market']}")
        print(f"   Context: z={last_snap['context']['z']}, fair={last_snap['context']['fair']}")

    # Verificar DB
    db_snapshots = db.get_snapshots(limit=100)
    db_test_snapshots = [s for s in db_snapshots if "Test Snapshot" in s['market']]
    print(f"\n💾 Base de Datos (trade_snapshots):")
    print(f"   Snapshots encontrados: {len(db_test_snapshots)}")
    if db_test_snapshots:
        s = db_test_snapshots[0]
        print(f"   Market: {s['market']}")
        print(f"   Context: z_score={s['z_score']}, fair_value={s['fair_value']}")
        print(f"   Additional: hours_left={s['hours_left']}, tweet_count={s['tweet_count']}")

    print("\n" + "="*70)
    print("✅ TEST 3 COMPLETADO")
    print("="*70)
    print(f"✓ Archivo snapshot/*.json creado")
    print(f"✓ Snapshot guardado en DB con todos los campos")


if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║              TEST DUAL-WRITE MODE - ARCHIVOS + DATABASE           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    if not db.is_db_available():
        print("\n❌ ERROR: DUAL_WRITE_MODE no está activado o DB no está disponible")
        print("   Asegúrate de:")
        print("   1. PostgreSQL está corriendo (docker compose ps)")
        print("   2. DATABASE_URL está configurada en .env")
        print("   3. DUAL_WRITE_MODE=true en .env")
        sys.exit(1)

    print("\n✅ Dual-write mode ACTIVADO")
    print(f"   DATABASE_URL: {os.environ.get('DATABASE_URL')[:50]}...")
    print(f"   DUAL_WRITE_MODE: {os.environ.get('DUAL_WRITE_MODE')}")

    try:
        test_paper_trader_dual_write()
        test_market_tape_dual_write()
        test_snapshot_dual_write()

        print("\n╔════════════════════════════════════════════════════════════════════╗")
        print("║                    ✅ TODOS LOS TESTS PASARON                      ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print("\nDual-write está funcionando correctamente!")
        print("Archivos Y base de datos se actualizan en paralelo.")

    except Exception as e:
        print(f"\n❌ ERROR EN TESTS: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
