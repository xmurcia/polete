"""
Tests for UnifiedTrader - Paper/Real wrapper
"""

import pytest
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unified_trader import UnifiedTrader


class TestUnifiedTraderPaperMode:
    """Test paper trading mode"""

    def test_paper_mode_initialization(self):
        """Test paper mode init"""
        trader = UnifiedTrader(use_real=False, initial_cash=1000.0)
        trader.initialize()

        assert trader.use_real is False
        assert trader._paper_trader is not None
        assert trader._paper_trader.portfolio["cash"] == 1000.0

    def test_paper_execute_buy(self):
        """Test paper BUY execution"""
        trader = UnifiedTrader(use_real=False, initial_cash=1000.0)
        trader.initialize()

        result = trader.execute(
            market_title="Test Market",
            bucket="300-319",
            signal="BUY",
            price=0.20,
            reason="Test",
            strategy_tag="STANDARD"
        )

        assert result is not None
        assert "BUY" in result or "✅" in result
        portfolio = trader.get_portfolio()
        assert portfolio["cash"] < 1000.0  # Money spent
        assert len(portfolio["positions"]) == 1

    def test_paper_portfolio_property(self):
        """Test portfolio property access"""
        trader = UnifiedTrader(use_real=False, initial_cash=1000.0)
        trader.initialize()

        # Should work with property access
        assert trader.portfolio["cash"] == 1000.0

        # Should be same as get_portfolio()
        assert trader.portfolio == trader.get_portfolio()

    def test_paper_multiple_positions(self):
        """Test multiple positions in paper mode"""
        trader = UnifiedTrader(use_real=False, initial_cash=1000.0)
        trader.initialize()

        # Buy position 1
        trader.execute(
            market_title="Test Market 1",
            bucket="300-319",
            signal="BUY",
            price=0.20,
            reason="Test 1",
            strategy_tag="STANDARD"
        )

        # Buy position 2
        trader.execute(
            market_title="Test Market 2",
            bucket="400-419",
            signal="BUY",
            price=0.15,
            reason="Test 2",
            strategy_tag="STANDARD"
        )

        portfolio = trader.get_portfolio()
        assert len(portfolio["positions"]) == 2
        assert portfolio["cash"] < 1000.0


class TestUnifiedTraderRealMode:
    """Test real trading mode (integration)"""

    def test_real_mode_initialization(self):
        """Test real mode init"""
        trader = UnifiedTrader(use_real=True)
        trader.initialize()

        assert trader.use_real is True
        assert trader.balance_mgr is not None
        assert trader.order_mgr is not None
        assert trader.position_tracker is not None

        # Balance should be from blockchain
        portfolio = trader.get_portfolio()
        assert "cash" in portfolio
        assert "positions" in portfolio
        assert isinstance(portfolio["cash"], (int, float))

    def test_real_mode_portfolio_format(self):
        """Test real portfolio returns correct format"""
        trader = UnifiedTrader(use_real=True)
        trader.initialize()

        portfolio = trader.get_portfolio()

        # Should match PaperTrader format
        assert "cash" in portfolio
        assert "positions" in portfolio
        assert "history" in portfolio

        # Positions should be dict
        assert isinstance(portfolio["positions"], dict)

    def test_real_mode_portfolio_property(self):
        """Test portfolio property in real mode"""
        trader = UnifiedTrader(use_real=True)
        trader.initialize()

        # Property access should work
        cash = trader.portfolio["cash"]
        assert isinstance(cash, (int, float))
        assert cash >= 0

    def test_real_mode_print_summary(self):
        """Test print_summary doesn't crash"""
        trader = UnifiedTrader(use_real=True)
        trader.initialize()

        # Should not crash
        trader.print_summary([])


class TestUnifiedTraderSwitching:
    """Test switching between modes"""

    def test_paper_and_real_separate_instances(self):
        """Test that paper and real are separate"""
        paper = UnifiedTrader(use_real=False, initial_cash=1000.0)
        paper.initialize()

        real = UnifiedTrader(use_real=True)
        real.initialize()

        # Should be different balances
        paper_cash = paper.portfolio["cash"]
        real_cash = real.portfolio["cash"]

        assert paper_cash == 1000.0
        # Real cash depends on blockchain state
        assert isinstance(real_cash, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
