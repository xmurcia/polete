#!/usr/bin/env python3
"""
Test: Simulación de trailing stop para moonshots
"""

print("🧪 SIMULACIÓN: TRAILING STOP ADAPTATIVO PARA MOONSHOTS")
print("="*70)

# Simulamos un moonshot que sube
prices = [
    0.060,  # Entry
    0.080,
    0.120,
    0.200,
    0.350,  # Inmunidad hasta aquí
    0.450,  # Aún no activa trailing
    0.520,  # ✅ ACTIVA TRAILING STOP (peak = 0.52)
    0.580,  # Nuevo peak (0.58)
    0.650,  # Nuevo peak (0.65)
    0.620,  # -0.03 desde peak (OK, sigue dentro)
    0.680,  # Nuevo peak (0.68)
    0.530,  # -0.15 desde 0.68 = ❌ VENDE
]

entry = prices[0]
peak = entry
trailing_active = False

print(f"\n📍 Entry: ${entry:.3f}")
print(f"🎯 Trailing stop se activa a: $0.50")
print(f"📉 Trailing stop threshold: -$0.15 desde peak\n")

print(f"{'Step':<6} | {'Price':<8} | {'Peak':<8} | {'Drawdown':<10} | {'Action'}")
print("-"*70)

for i, price in enumerate(prices, 1):
    # Update peak
    if price > peak:
        peak = price
    
    # Check if trailing is active
    if not trailing_active and peak >= 0.50:
        trailing_active = True
    
    drawdown = peak - price
    
    # Determine action
    action = ""
    if not trailing_active:
        if price < 0.35:
            action = "HOLD (inmunidad)"
        else:
            action = "HOLD (esperando $0.50)"
    else:
        if drawdown >= 0.15:
            action = f"🔴 SELL (trailing -${drawdown:.2f})"
        else:
            action = f"HOLD (trailing ok, -{drawdown:.2f})"
    
    print(f"{i:<6} | ${price:.3f}  | ${peak:.3f}  | ${drawdown:.3f}    | {action}")

print("\n" + "="*70)
print("\n💡 RESUMEN:")
print(f"  Entry: ${entry:.3f}")
print(f"  Peak máximo: ${peak:.3f}")
print(f"  Ganancia potencial: +{(peak-entry)/entry*100:.1f}%")
print(f"  Trailing stop activado en step 7 (cuando llegó a $0.52)")
print(f"  Venta ejecutada en step 12 (cayó -$0.15 desde peak $0.68)")
print(f"  Exit: $0.53 → Ganancia capturada: +{(0.53-entry)/entry*100:.1f}%")
