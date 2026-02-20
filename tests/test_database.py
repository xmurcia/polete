#!/usr/bin/env python3
"""
Script de prueba para verificar la conexión y operaciones básicas con PostgreSQL.
Ejecutar después de arrancar Docker: python test_database.py
"""

import os
import sys
import time
from datetime import datetime

# Configurar DATABASE_URL antes de importar database.py
os.environ["DATABASE_URL"] = "postgresql://polybot:polybot_dev_2026@localhost:5432/polymarket"

try:
    import database as db
except ImportError as e:
    print(f"❌ Error importando database.py: {e}")
    print("   Instala las dependencias: pip install psycopg2-binary")
    sys.exit(1)


def test_connection():
    """Prueba 1: Verificar conexión y creación de tablas"""
    print("\n" + "="*70)
    print("TEST 1: Conexión y creación de schema")
    print("="*70)

    try:
        db.init_db()
        print("✅ Conexión exitosa y tablas creadas")
        return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False


def test_trades():
    """Prueba 2: Operaciones CRUD en tabla trades"""
    print("\n" + "="*70)
    print("TEST 2: Tabla trades")
    print("="*70)

    try:
        # Insertar trade de prueba
        db.log_trade(
            action="BUY",
            market="Test Market",
            bucket="100-119",
            price=0.45,
            shares=10.0,
            reason="Test entry",
            pnl=0.0,
            cash_after=995.0,
            mode="PAPER",
            strategy="STANDARD"
        )
        print("✅ Trade insertado")

        # Leer trades
        trades = db.get_trades(limit=5)
        print(f"✅ Trades recuperados: {len(trades)}")
        if trades:
            print(f"   Último trade: {trades[0]['action']} {trades[0]['bucket']} @ ${trades[0]['price']}")

        # Stats
        stats = db.get_trade_stats()
        print(f"✅ Stats: {stats['total_sells']} ventas, Win rate: {stats['win_rate']}%")

        return True
    except Exception as e:
        print(f"❌ Error en trades: {e}")
        return False


def test_positions():
    """Prueba 3: Operaciones CRUD en tabla positions"""
    print("\n" + "="*70)
    print("TEST 3: Tabla positions")
    print("="*70)

    try:
        pos_id = "Test Market|100-119"

        # Insertar posición
        db.upsert_position(pos_id, {
            "market": "Test Market",
            "bucket": "100-119",
            "shares": 10.0,
            "entry_price": 0.45,
            "current_price": 0.50,
            "invested": 4.5,
            "strategy_tag": "STANDARD",
            "mode": "PAPER"
        })
        print("✅ Posición insertada")

        # Leer posiciones
        positions = db.get_positions(mode="PAPER")
        print(f"✅ Posiciones recuperadas: {len(positions)}")
        if positions:
            print(f"   Posición: {list(positions.keys())[0]}")

        # Actualizar posición
        db.upsert_position(pos_id, {
            "market": "Test Market",
            "bucket": "100-119",
            "shares": 10.0,
            "entry_price": 0.45,
            "current_price": 0.55,  # Cambio
            "invested": 4.5,
            "strategy_tag": "STANDARD",
            "mode": "PAPER"
        })
        print("✅ Posición actualizada")

        # Eliminar posición
        db.delete_position(pos_id)
        positions_after = db.get_positions(mode="PAPER")
        print(f"✅ Posición eliminada (quedan {len(positions_after)})")

        return True
    except Exception as e:
        print(f"❌ Error en positions: {e}")
        return False


def test_bot_state():
    """Prueba 4: Estado global del bot"""
    print("\n" + "="*70)
    print("TEST 4: Bot state (cash, config)")
    print("="*70)

    try:
        # Guardar cash
        db.set_state("cash", 1000.0)
        print("✅ Cash guardado")

        # Leer cash
        cash = db.get_state("cash")
        print(f"✅ Cash recuperado: ${cash}")

        # Guardar config
        db.set_state("config", {"max_positions": 5, "risk_pct": 0.04})
        config = db.get_state("config")
        print(f"✅ Config recuperado: {config}")

        # Leer valor inexistente
        missing = db.get_state("nonexistent", default="default_value")
        print(f"✅ Default value funciona: {missing}")

        return True
    except Exception as e:
        print(f"❌ Error en bot_state: {e}")
        return False


def test_tweet_events():
    """Prueba 5: Eventos de tweets"""
    print("\n" + "="*70)
    print("TEST 5: Tweet events")
    print("="*70)

    try:
        # Insertar eventos
        now_ms = int(time.time() * 1000)
        for i in range(3):
            db.insert_tweet_event(now_ms - i * 60000)  # Cada minuto hacia atrás
        print("✅ 3 eventos insertados")

        # Leer eventos recientes
        events = db.get_recent_tweet_events(hours=1)
        print(f"✅ Eventos recuperados (última hora): {len(events)}")

        # Prueba de limpieza (no borra nada porque son recientes)
        db.prune_tweet_events(hours=48)
        events_after = db.get_recent_tweet_events(hours=1)
        print(f"✅ Limpieza ejecutada (eventos restantes: {len(events_after)})")

        return True
    except Exception as e:
        print(f"❌ Error en tweet_events: {e}")
        return False


def test_snapshots():
    """Prueba 6: Trade snapshots"""
    print("\n" + "="*70)
    print("TEST 6: Trade snapshots")
    print("="*70)

    try:
        # Insertar snapshot
        db.log_snapshot(
            action="BUY",
            market="Test Market",
            bucket="100-119",
            price=0.45,
            reason="Test entry with context",
            context={"z": 0.75, "pnl": 0.0, "hours_left": 12.5},
            mode="PAPER"
        )
        print("✅ Snapshot insertado")

        # Leer snapshots
        snapshots = db.get_snapshots(limit=5)
        print(f"✅ Snapshots recuperados: {len(snapshots)}")
        if snapshots:
            print(f"   Último: {snapshots[0]['action']} Z={snapshots[0]['z_score']}")

        return True
    except Exception as e:
        print(f"❌ Error en snapshots: {e}")
        return False


def test_market_tape():
    """Prueba 7: Market tape (histórico de precios)"""
    print("\n" + "="*70)
    print("TEST 7: Market tape")
    print("="*70)

    try:
        # Simular un snapshot del order book
        ts_unix = time.time()
        meta = [
            {
                "title": "Test Market February 20",
                "count": 308,
                "hours": 4.7,
                "daily_avg": 65.5,
                "active": True
            }
        ]
        order_book = [
            {
                "title": "Test Market February 20",
                "buckets": [
                    {"bucket": "280-299", "min": 280, "max": 299, "bid": 0.05, "ask": 0.06},
                    {"bucket": "300-319", "min": 300, "max": 319, "bid": 0.50, "ask": 0.53},
                    {"bucket": "320-339", "min": 320, "max": 339, "bid": 0.39, "ask": 0.40},
                ]
            }
        ]

        db.save_tape(ts_unix, meta, order_book)
        print("✅ Tape guardado")

        # Leer evolución de un bucket
        history = db.get_tape_for_bucket("300-319", "Test Market", limit=5)
        print(f"✅ Historia del bucket recuperada: {len(history)} snapshots")
        if history:
            print(f"   Último precio: bid=${history[0]['bid']} ask=${history[0]['ask']}")

        return True
    except Exception as e:
        print(f"❌ Error en market_tape: {e}")
        return False


def test_shadow_write():
    """Prueba 8: Shadow write (dual-write seguro)"""
    print("\n" + "="*70)
    print("TEST 8: Shadow write (error handling)")
    print("="*70)

    try:
        # Simular escritura que falla
        def failing_write():
            raise ValueError("Simulated DB error")

        db.shadow_write(failing_write)
        print("✅ Shadow write manejó el error correctamente (no crasheó)")

        # Escritura exitosa
        db.shadow_write(
            db.log_trade,
            action="SELL",
            market="Test",
            bucket="100-119",
            price=0.55,
            shares=10.0,
            pnl=1.0
        )
        print("✅ Shadow write exitoso funcionó")

        return True
    except Exception as e:
        print(f"❌ Error en shadow_write: {e}")
        return False


def run_all_tests():
    """Ejecuta todas las pruebas"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "TEST DATABASE - POLYMARKET BOT" + " "*22 + "║")
    print("╚" + "="*68 + "╝")

    tests = [
        test_connection,
        test_trades,
        test_positions,
        test_bot_state,
        test_tweet_events,
        test_snapshots,
        test_market_tape,
        test_shadow_write,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test falló con excepción: {e}")
            results.append(False)

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Tests pasados: {passed}/{total}")

    if passed == total:
        print("\n🎉 ¡Todos los tests pasaron! La base de datos está lista.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) fallaron. Revisa los errores arriba.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
