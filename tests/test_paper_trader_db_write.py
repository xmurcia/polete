#!/usr/bin/env python3
"""
Test rápido: Verificar que PaperTrader escribe en DB cuando hace un trade
"""

import os
import sys

# Configurar environment ANTES de importar
os.environ["DATABASE_URL"] = "postgresql://polybot:polybot_dev_2026@localhost:5432/polymarket"
os.environ["DUAL_WRITE_MODE"] = "true"

# Necesitamos cargar .env primero
from dotenv import load_dotenv
load_dotenv()

from src.paper_trader import PaperTrader
import database as db

def test_single_trade():
    print("\n" + "="*70)
    print("TEST: PaperTrader escribe en DB al ejecutar trade")
    print("="*70)

    if not db.is_db_available():
        print("❌ DB no disponible")
        sys.exit(1)

    print("✅ DB disponible")

    # Contar trades actuales
    trades_before = db.get_trades(limit=1000, mode="PAPER")
    print(f"\n📊 Trades en DB ANTES: {len(trades_before)}")

    # Crear trader y ejecutar un trade
    trader = PaperTrader(initial_cash=1000.0)
    print(f"💵 Cash inicial: ${trader.portfolio['cash']:.2f}")

    # Ejecutar trade BUY
    result = trader.execute(
        market_title="Test DB Write - Paper Trader",
        bucket="100-119",
        signal="BUY",
        price=0.50,
        reason="Test dual-write from PaperTrader",
        strategy_tag="STANDARD",
        hours_left=15.5,
        tweet_count=200,
        market_consensus=0.52,
        entry_z_score=0.65
    )
    print(f"\n✅ Trade ejecutado: {result}")

    # Verificar que se escribió en DB
    trades_after = db.get_trades(limit=1000, mode="PAPER")
    print(f"📊 Trades en DB DESPUÉS: {len(trades_after)}")

    if len(trades_after) > len(trades_before):
        print("\n✅ TRADE SE ESCRIBIÓ EN DB!")
        latest_trade = trades_after[0]
        print(f"   Market: {latest_trade['market']}")
        print(f"   Bucket: {latest_trade['bucket']}")
        print(f"   Price: ${latest_trade['price']}")
        print(f"   Hours left: {latest_trade['hours_left']}")
        print(f"   Tweet count: {latest_trade['tweet_count']}")
        print(f"   Market consensus: {latest_trade['market_consensus']}")
        print(f"   Strategy: {latest_trade['strategy']}")
    else:
        print("\n❌ TRADE NO SE ESCRIBIÓ EN DB")
        print("   Verificar que shadow_write() está funcionando")

    # Verificar posiciones en DB
    db_positions = db.get_positions(mode="PAPER")
    test_positions = {k: v for k, v in db_positions.items() if 'Test DB Write' in k}
    print(f"\n📊 Posiciones en DB: {len(test_positions)}")

    if test_positions:
        print("✅ POSICIÓN SE ESCRIBIÓ EN DB!")
        pos_id, pos = list(test_positions.items())[0]
        print(f"   Position ID: {pos_id}")
        print(f"   Bucket: {pos['bucket']}")
        print(f"   Entry price: ${pos['entry_price']}")
        print(f"   Entry Z-score: {pos.get('entry_z_score')}")
    else:
        print("❌ POSICIÓN NO SE ESCRIBIÓ EN DB")

    # Verificar bot_state (cash)
    db_cash = db.get_state("cash", default=None)
    print(f"\n📊 Cash en DB: {db_cash}")

    if db_cash is not None:
        print(f"✅ CASH SE ESCRIBIÓ EN DB!")
        print(f"   Cash en portfolio.json: ${trader.portfolio['cash']:.2f}")
        print(f"   Cash en DB: ${db_cash:.2f}")
    else:
        print("❌ CASH NO SE ESCRIBIÓ EN DB")

    print("\n" + "="*70)
    print("FIN DEL TEST")
    print("="*70)


if __name__ == "__main__":
    test_single_trade()
