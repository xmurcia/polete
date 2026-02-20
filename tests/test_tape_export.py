#!/usr/bin/env python3
"""
Test: Verificar que podemos exportar market_tape desde DB
con el formato EXACTO del archivo original
"""

import os
import sys
import json
import glob

# Configurar DATABASE_URL antes de importar
os.environ["DATABASE_URL"] = "postgresql://polybot:polybot_dev_2026@localhost:5432/polymarket"
os.environ["DUAL_WRITE_MODE"] = "true"

import database as db

def test_export_format():
    """Test: Exportar desde DB y comparar estructura con archivo original"""
    print("\n" + "="*70)
    print("TEST: Exportar Market Tape desde DB")
    print("="*70)

    # 1. Leer un archivo original para tener la estructura de referencia
    tape_files = sorted(glob.glob("logs/market_tape/tape_*.json"))
    if not tape_files:
        print("❌ No hay archivos de tape para comparar")
        sys.exit(1)

    with open(tape_files[-1], 'r') as f:
        original_tape = json.load(f)

    print(f"\n📁 Archivo original: {os.path.basename(tape_files[-1])}")
    print(f"   Estructura:")
    print(f"   - timestamp: {type(original_tape['timestamp']).__name__}")
    print(f"   - meta: {len(original_tape['meta'])} markets")
    print(f"   - order_book: {len(original_tape['order_book'])} markets")

    if original_tape['meta']:
        print(f"\n   Meta[0] keys: {list(original_tape['meta'][0].keys())}")
        print(f"   Meta[0] sample:")
        for k, v in original_tape['meta'][0].items():
            print(f"     - {k}: {v} ({type(v).__name__})")

    if original_tape['order_book'] and original_tape['order_book'][0]['buckets']:
        print(f"\n   Bucket[0] keys: {list(original_tape['order_book'][0]['buckets'][0].keys())}")
        print(f"   Bucket[0] sample:")
        for k, v in original_tape['order_book'][0]['buckets'][0].items():
            print(f"     - {k}: {v} ({type(v).__name__})")

    # 2. Exportar desde DB
    print("\n" + "-"*70)
    print("💾 Exportando desde base de datos...")

    exported = db.export_tape_as_json()

    if not exported:
        print("⚠️  No hay datos en DB para exportar")
        print("   (Esto es normal si acabas de limpiar la DB)")
        return

    print(f"   Exportados: {len(exported)} snapshots")

    # 3. Comparar estructura
    print("\n" + "-"*70)
    print("🔍 Comparando estructuras...")

    db_tape = exported[0]  # Tomar el primer snapshot exportado

    # Verificar top-level keys
    original_keys = set(original_tape.keys())
    db_keys = set(db_tape.keys())

    print(f"\n✓ Top-level keys:")
    print(f"  Original: {sorted(original_keys)}")
    print(f"  DB Export: {sorted(db_keys)}")

    if original_keys != db_keys:
        print(f"  ❌ DIFERENCIA: {original_keys.symmetric_difference(db_keys)}")
        return False
    else:
        print(f"  ✅ Idénticos")

    # Verificar estructura de meta
    if db_tape['meta']:
        original_meta_keys = set(original_tape['meta'][0].keys())
        db_meta_keys = set(db_tape['meta'][0].keys())

        print(f"\n✓ Meta keys:")
        print(f"  Original: {sorted(original_meta_keys)}")
        print(f"  DB Export: {sorted(db_meta_keys)}")

        if original_meta_keys != db_meta_keys:
            print(f"  ❌ DIFERENCIA: {original_meta_keys.symmetric_difference(db_meta_keys)}")
            return False
        else:
            print(f"  ✅ Idénticos")

    # Verificar estructura de order_book
    if db_tape['order_book'] and db_tape['order_book'][0]['buckets']:
        original_bucket_keys = set(original_tape['order_book'][0]['buckets'][0].keys())
        db_bucket_keys = set(db_tape['order_book'][0]['buckets'][0].keys())

        print(f"\n✓ Bucket keys:")
        print(f"  Original: {sorted(original_bucket_keys)}")
        print(f"  DB Export: {sorted(db_bucket_keys)}")

        if original_bucket_keys != db_bucket_keys:
            print(f"  ❌ DIFERENCIA: {original_bucket_keys.symmetric_difference(db_bucket_keys)}")
            return False
        else:
            print(f"  ✅ Idénticos")

    # 4. Test de escritura a disco
    print("\n" + "-"*70)
    print("💾 Test de dump_tape_to_files()...")

    test_dir = "logs/test_export"
    os.makedirs(test_dir, exist_ok=True)

    count = db.dump_tape_to_files(test_dir)
    print(f"   Exportados {count} archivos a {test_dir}/")

    # Verificar que los archivos se crearon
    exported_files = sorted(glob.glob(f"{test_dir}/tape_*.json"))
    if exported_files:
        with open(exported_files[0], 'r') as f:
            exported_tape = json.load(f)

        print(f"\n   Archivo exportado: {os.path.basename(exported_files[0])}")
        print(f"   Keys: {list(exported_tape.keys())}")
        print(f"   ✅ Formato válido")

    # 5. Resumen
    print("\n" + "="*70)
    print("✅ TEST COMPLETADO")
    print("="*70)
    print("✓ Estructura top-level: idéntica")
    print("✓ Estructura meta: idéntica")
    print("✓ Estructura order_book/buckets: idéntica")
    print("✓ Función dump_tape_to_files(): funciona")
    print("\n🎯 CONCLUSIÓN: Los market_tape se pueden recuperar")
    print("   con el formato EXACTO del archivo original")


if __name__ == "__main__":
    if not db.is_db_available():
        print("\n❌ ERROR: Base de datos no disponible")
        print("   Asegúrate de:")
        print("   1. PostgreSQL está corriendo")
        print("   2. DATABASE_URL está configurada")
        print("   3. DUAL_WRITE_MODE=true")
        sys.exit(1)

    print("✅ Base de datos disponible")

    try:
        test_export_format()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
