#!/usr/bin/env python3
"""
Test script to validate SELL order decimal precision fix.
This simulates a SELL order without actually executing it.
"""

from decimal import Decimal, ROUND_DOWN

def test_sell_order_rounding(price: float, shares: float):
    """
    Simulate the SELL order rounding logic from order_manager.py
    """
    print(f"\n{'='*60}")
    print(f"Testing SELL order with:")
    print(f"  price = {price}")
    print(f"  shares = {shares}")
    print(f"{'='*60}")

    # Round size (shares sold) to 5 decimals first
    size = float(Decimal(str(shares)).quantize(Decimal('0.00001'), rounding=ROUND_DOWN))
    print(f"\nStep 1: Round size to 5 decimals")
    print(f"  size = {size:.5f}")

    # Then ensure maker_amount (USDC received) has max 2 decimals
    maker_amount = float(Decimal(str(price * size)).quantize(Decimal('0.01'), rounding=ROUND_DOWN))
    print(f"\nStep 2: Calculate maker_amount (USDC received)")
    print(f"  price × size = {price} × {size} = {price * size}")
    print(f"  maker_amount (rounded to 2 dec) = ${maker_amount:.2f}")

    # Recalculate size from rounded maker_amount and round to 5 decimals
    size_calc = maker_amount / price if price > 0 else 0
    size_final = float(Decimal(str(size_calc)).quantize(Decimal('0.00001'), rounding=ROUND_DOWN))
    print(f"\nStep 3: Recalculate size from rounded maker_amount")
    print(f"  size_calc = {maker_amount} / {price} = {size_calc}")
    print(f"  size_final (rounded to 5 dec) = {size_final:.5f}")

    # Verify precision
    maker_amount_check = price * size_final
    print(f"\nVerification:")
    print(f"  price × size_final = {price} × {size_final} = {maker_amount_check:.6f}")
    print(f"  Maker amount has {len(str(maker_amount_check).split('.')[-1])} decimals")

    # Check if it meets Polymarket requirements
    maker_decimals = len(str(maker_amount).split('.')[-1]) if '.' in str(maker_amount) else 0
    size_str = f"{size_final:.10f}".rstrip('0')
    size_decimals = len(size_str.split('.')[-1]) if '.' in size_str else 0

    print(f"\nPolymarket Requirements Check:")
    print(f"  ✓ maker_amount (USDC received): {maker_decimals} decimals (max 2) {'✓' if maker_decimals <= 2 else '❌'}")
    print(f"  ✓ size (shares sold): {size_decimals} decimals (max 5) {'✓' if size_decimals <= 5 else '❌'}")

    return {
        'price': price,
        'size': size_final,
        'maker_amount': maker_amount,
        'valid': maker_decimals <= 2 and size_decimals <= 5
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SELL ORDER DECIMAL PRECISION TEST")
    print("="*60)

    # Test case 1: Current position (380-399)
    # Entry: 31.25 shares @ $0.16
    # Current bid: ~$0.158
    test_sell_order_rounding(price=0.158, shares=31.25)

    # Test case 2: Problematic case (many decimals)
    test_sell_order_rounding(price=0.121, shares=41.32231)

    # Test case 3: Edge case (very small price)
    test_sell_order_rounding(price=0.023, shares=217.391304348)

    # Test case 4: Edge case (high precision shares)
    test_sell_order_rounding(price=0.456, shares=123.456789012)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")
