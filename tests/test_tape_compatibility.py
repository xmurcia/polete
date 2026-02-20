#!/usr/bin/env python3
"""
Test de compatibilidad: Verificar que los archivos tape exportados desde DB
tienen la misma estructura que los originales (para backtesting).
"""

import os
import json
import glob

os.environ["DATABASE_URL"] = "postgresql://polybot:polybot_dev_2026@localhost:5432/polymarket"

import database as db

def test_export_compatibility():
    """
    1. Lee un tape original de logs/market_tape/
    2. Guarda ese tape en la DB
    3. Exporta desde la DB a un archivo temporal
    4. Compara estructuras
    """

    print("\n" + "="*70)
    print("TEST: Compatibilidad de Estructura de Tape")
    print("="*70 + "\n")

    # 1. Leer un tape original
    tape_files = sorted(glob.glob("logs/market_tape/tape_*.json"))
    if not tape_files:
        print("❌ No hay archivos tape para comparar")
        return False

    original_file = tape_files[0]
    print(f"📂 Leyendo archivo original: {os.path.basename(original_file)}")

    with open(original_file, 'r') as f:
        original_tape = json.load(f)

    print(f"   Timestamp: {original_tape['timestamp']}")
    print(f"   Markets: {len(original_tape['meta'])}")
    print(f"   Order books: {len(original_tape['order_book'])}")

    # 2. Guardar en DB
    print("\n💾 Guardando en base de datos...")
    db.save_tape(
        ts_unix=original_tape['timestamp'],
        meta=original_tape['meta'],
        order_book=original_tape['order_book']
    )
    print("   ✅ Guardado en DB")

    # 3. Exportar desde DB
    print("\n📤 Exportando desde DB...")
    exported_tapes = db.export_tape_as_json()

    if not exported_tapes:
        print("❌ No se pudo exportar desde DB")
        return False

    # Buscar el tape que acabamos de insertar
    exported_tape = None
    for tape in exported_tapes:
        if abs(tape['timestamp'] - original_tape['timestamp']) < 1:
            exported_tape = tape
            break

    if not exported_tape:
        print("❌ No se encontró el tape exportado")
        return False

    print(f"   ✅ Tape exportado encontrado")

    # 4. Comparar estructuras
    print("\n🔍 Comparando estructuras...")

    errors = []

    # Verificar campos top-level
    if 'timestamp' not in exported_tape:
        errors.append("❌ Falta campo 'timestamp'")
    if 'meta' not in exported_tape:
        errors.append("❌ Falta campo 'meta'")
    if 'order_book' not in exported_tape:
        errors.append("❌ Falta campo 'order_book'")

    if errors:
        for err in errors:
            print(f"   {err}")
        return False

    print("   ✅ Campos top-level presentes")

    # Verificar estructura de meta
    if exported_tape['meta']:
        meta_item = exported_tape['meta'][0]
        required_meta_fields = ['title', 'count', 'hours', 'daily_avg', 'active']

        for field in required_meta_fields:
            if field not in meta_item:
                errors.append(f"❌ Meta falta campo '{field}'")

        if errors:
            for err in errors:
                print(f"   {err}")
            return False

        print("   ✅ Estructura de 'meta' correcta")

    # Verificar estructura de order_book
    if exported_tape['order_book']:
        ob_item = exported_tape['order_book'][0]

        if 'title' not in ob_item:
            errors.append("❌ Order book falta campo 'title'")
        if 'buckets' not in ob_item:
            errors.append("❌ Order book falta campo 'buckets'")

        if ob_item.get('buckets'):
            bucket = ob_item['buckets'][0]
            required_bucket_fields = ['bucket', 'min', 'max', 'bid', 'ask']

            for field in required_bucket_fields:
                if field not in bucket:
                    errors.append(f"❌ Bucket falta campo '{field}'")

        if errors:
            for err in errors:
                print(f"   {err}")
            return False

        print("   ✅ Estructura de 'order_book' correcta")

    # 5. Comparar valores (ejemplo)
    print("\n📊 Comparando valores de ejemplo...")

    print(f"   Original timestamp: {original_tape['timestamp']}")
    print(f"   Exported timestamp: {exported_tape['timestamp']}")

    if len(original_tape['meta']) > 0 and len(exported_tape['meta']) > 0:
        orig_title = original_tape['meta'][0]['title']
        exp_title = exported_tape['meta'][0]['title']
        print(f"   Original market: {orig_title}")
        print(f"   Exported market: {exp_title}")

        if orig_title == exp_title:
            print("   ✅ Títulos coinciden")

    # 6. Verificar que el backtesting puede leerlo
    print("\n🧪 Verificando compatibilidad con backtesting...")

    # Guardar exported_tape en un archivo temporal
    temp_file = "/tmp/test_tape_export.json"
    with open(temp_file, 'w') as f:
        json.dump(exported_tape, f)

    # Intentar leerlo como lo haría el backtesting
    try:
        with open(temp_file, 'r') as f:
            backtest_tape = json.load(f)

        # Verificar acceso a campos como lo hace el backtesting
        ts = backtest_tape['timestamp']
        meta = backtest_tape['meta']
        ob = backtest_tape['order_book']

        # Extraer market_title como lo hace tape_backtest.py
        if meta:
            market_title = meta[0]['title']
            print(f"   ✅ Backtesting puede leer: {market_title}")

        # Extraer buckets como lo hace tape_backtest.py
        if ob and ob[0].get('buckets'):
            bucket = ob[0]['buckets'][0]
            print(f"   ✅ Backtesting puede leer buckets: {bucket['bucket']} @ bid={bucket['bid']}")

    except Exception as e:
        print(f"   ❌ Error al leer como backtesting: {e}")
        return False

    finally:
        os.remove(temp_file)

    # 7. Resumen
    print("\n" + "="*70)
    print("✅ TEST PASSED: Estructura compatible con backtesting")
    print("="*70)
    print("\nLos archivos exportados desde DB mantienen la estructura original.")
    print("El backtesting puede seguir usando los archivos JSON sin cambios.\n")

    return True


if __name__ == "__main__":
    success = test_export_compatibility()
    exit(0 if success else 1)
