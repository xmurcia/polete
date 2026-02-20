"""
Tests for PolyAuth authentication.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import PolyAuth


def test_auth_initialization():
    """Test that PolyAuth initializes correctly"""
    auth = PolyAuth()

    assert auth is not None
    assert auth.private_key is not None
    assert auth.wallet_address is not None
    assert auth.chain_id == 137
    assert auth.host == "https://clob.polymarket.com"


def test_get_wallet_address():
    """Test wallet address getter"""
    auth = PolyAuth()
    wallet = auth.get_wallet_address()

    assert wallet is not None
    assert wallet.startswith("0x")
    assert len(wallet) == 42


def test_client_initialization():
    """Test that client can be initialized"""
    auth = PolyAuth()
    client = auth.get_client()

    assert client is not None
    assert auth.client is not None


def test_client_is_cached():
    """Test that client is cached after first call"""
    auth = PolyAuth()

    client1 = auth.get_client()
    client2 = auth.get_client()

    # Should return same instance
    assert client1 is client2


def test_test_connection():
    """Test connection to Polymarket API"""
    auth = PolyAuth()

    # This will make real API call
    success = auth.test_connection()

    assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
