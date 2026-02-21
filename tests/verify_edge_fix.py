#!/usr/bin/env python3
"""
Verification script to confirm the edge multiplier fix
Shows before/after comparison for Feb 17-24 event
"""

print("=" * 80)
print("🔧 VERIFICACIÓN: Cambio de EDGE_STD_MULTIPLIER")
print("=" * 80)

# Event data from current output
event_name = "Elon Musk # tweets February 17 - February 24, 2026?"
tweet_count = 221
hours_left = 75.5
std_dev = 20.2

# Best buckets from output
buckets = [
    ("380-399", 0.165, 0.341, 0.5),  # ask, fair, z_score
    ("400-419", 0.116, 0.336, 0.5),
]

# Constants
MIN_EDGE_BASE = 0.05
MAX_Z_SCORE_ENTRY = 0.85
MIN_PRICE_ENTRY = 0.02

print(f"\nEvento: {event_name}")
print(f"Tweets: {tweet_count}")
print(f"Tiempo restante: {hours_left:.1f}h ({hours_left/24:.1f} días)")
print(f"Volatilidad (σ): {std_dev:.1f}")
print()

# BEFORE
EDGE_STD_MULTIPLIER_OLD = 0.01
dynamic_edge_old = MIN_EDGE_BASE + (std_dev * EDGE_STD_MULTIPLIER_OLD)

print("━" * 80)
print("❌ ANTES (EDGE_STD_MULTIPLIER = 0.01)")
print("━" * 80)
print(f"Edge requerido: {dynamic_edge_old:.3f} ({dynamic_edge_old*100:.1f}%)")
print()

for bucket, ask, fair, z_score in buckets:
    edge = fair - ask
    z_ok = z_score <= MAX_Z_SCORE_ENTRY
    price_ok = ask >= MIN_PRICE_ENTRY
    edge_ok = edge > dynamic_edge_old

    would_buy = z_ok and price_ok and edge_ok
    status = "✅ COMPRARÍA" if would_buy else "❌ NO COMPRA"

    print(f"  {bucket:10} | Ask=${ask:.3f} | Fair={fair:.3f} | Edge={edge:+.3f} ({edge*100:+.1f}%) | {status}")
    if not edge_ok:
        print(f"  {'':10}   └─ Bloqueado: Edge {edge:.3f} ≤ {dynamic_edge_old:.3f}")

# AFTER
EDGE_STD_MULTIPLIER_NEW = 0.005
dynamic_edge_new = MIN_EDGE_BASE + (std_dev * EDGE_STD_MULTIPLIER_NEW)

print()
print("━" * 80)
print("✅ DESPUÉS (EDGE_STD_MULTIPLIER = 0.005)")
print("━" * 80)
print(f"Edge requerido: {dynamic_edge_new:.3f} ({dynamic_edge_new*100:.1f}%)")
print()

for bucket, ask, fair, z_score in buckets:
    edge = fair - ask
    z_ok = z_score <= MAX_Z_SCORE_ENTRY
    price_ok = ask >= MIN_PRICE_ENTRY
    edge_ok = edge > dynamic_edge_new

    would_buy = z_ok and price_ok and edge_ok
    status = "✅ COMPRARÍA" if would_buy else "❌ NO COMPRA"

    print(f"  {bucket:10} | Ask=${ask:.3f} | Fair={fair:.3f} | Edge={edge:+.3f} ({edge*100:+.1f}%) | {status}")
    if would_buy:
        roi = (edge / ask) * 100
        print(f"  {'':10}   └─ ✨ ROI esperado: {roi:.1f}% | Z-score: {z_score:.1f} (muy cerca de predicción)")

print()
print("=" * 80)
print("📊 RESUMEN DEL CAMBIO")
print("=" * 80)
print(f"Edge requerido: {dynamic_edge_old:.3f} → {dynamic_edge_new:.3f} (reducción de {(dynamic_edge_old-dynamic_edge_new)*100:.1f}%)")
print()
print("Resultado:")
print("  ✅ Ahora puede operar más temprano en eventos largos")
print("  ✅ Mantiene protección (edge base 5% + ajuste por volatilidad)")
print("  ✅ Solo compra buckets con Z-score bajo (cerca de predicción)")
print()
print("🎯 El bot debería empezar a operar en Feb 17-24 en la próxima iteración")
