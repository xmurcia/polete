"""
Test to verify numpy type conversion in database.py works correctly
"""

import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Only test if database is available
import database as db

if not db.is_db_available():
    print("⚠️  Database not available (DUAL_WRITE_MODE not enabled)")
    print("   Set DUAL_WRITE_MODE=true in .env to test")
    exit(0)

print("\n🧪 Testing Database Numpy Type Conversion")
print("=" * 80)

# Test cases with numpy types
test_context = {
    "z": np.float64(0.7744515929863482),
    "fair": np.float64(0.9559398451990877),
    "pnl": None
}

print("\n📊 Test Data:")
print(f"  z_score: {test_context['z']} (type: {type(test_context['z'])})")
print(f"  fair: {test_context['fair']} (type: {type(test_context['fair'])})")
print(f"  pnl: {test_context['pnl']} (type: {type(test_context['pnl'])})")

print("\n🔍 Attempting to insert snapshot with numpy types...")

try:
    db.shadow_write(
        db.log_snapshot,
        action="BUY",
        market="Test Market",
        bucket="65-89",
        price=0.27,
        reason="Test numpy conversion",
        context=test_context,
        mode="PAPER",
        hours_left=None,
        tweet_count=None
    )
    print("  ✅ SUCCESS: Snapshot inserted without errors!")
    print("  ✅ Numpy types were correctly converted to Python native types")

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    if "np" in str(e).lower():
        print("  ❌ Numpy type conversion is NOT working!")
    else:
        print("  ⚠️  Different error (not numpy-related)")

print("\n" + "=" * 80)

# Test with regular Python types too
print("\n🧪 Testing with native Python types (control test)...")

test_context_native = {
    "z": 0.7744515929863482,
    "fair": 0.9559398451990877,
    "pnl": None
}

print("\n📊 Test Data:")
print(f"  z_score: {test_context_native['z']} (type: {type(test_context_native['z'])})")
print(f"  fair: {test_context_native['fair']} (type: {type(test_context_native['fair'])})")

try:
    db.shadow_write(
        db.log_snapshot,
        action="BUY",
        market="Test Market 2",
        bucket="300-319",
        price=0.20,
        reason="Test native types",
        context=test_context_native,
        mode="PAPER",
        hours_left=None,
        tweet_count=None
    )
    print("\n  ✅ SUCCESS: Snapshot with native types inserted correctly")

except Exception as e:
    print(f"\n  ❌ FAILED: {e}")

print("\n" + "=" * 80)
print("🏁 Test Complete\n")
