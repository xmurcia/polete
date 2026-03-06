import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """
    Backtesting performance metrics.
    """
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_profit: float
    avg_loss: float
    profit_factor: float
    equity_curve: List[Dict]

    def print_summary(self):
        """Print formatted performance summary"""
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS: {self.strategy_name}")
        print("="*60)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${self.final_value:,.2f}")
        print(f"Total Return: {self.total_return*100:+.2f}%")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {self.max_drawdown*100:.2f}%")
        print("-"*60)
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades} ({self.win_rate*100:.1f}%)")
        print(f"Losing Trades: {self.losing_trades}")
        print(f"Avg Profit per Winner: ${self.avg_profit:.2f}")
        print(f"Avg Loss per Loser: ${self.avg_loss:.2f}")
        print(f"Profit Factor: {self.profit_factor:.2f}")
        print("="*60 + "\n")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_profit": self.avg_profit,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor
        }

def calculate_metrics(
    strategy_name: str,
    equity_curve: List[Dict],
    trades: List[Dict],
    initial_capital: float
) -> PerformanceMetrics:
    """
    Calculate performance metrics from equity curve and trade history.

    Args:
        strategy_name: Name of strategy
        equity_curve: List of {timestamp, equity, cash} dicts
        trades: List of completed trades with profit/loss
        initial_capital: Starting capital
    """
    if not equity_curve:
        raise ValueError("Empty equity curve")

    # Extract equity series
    equity_series = pd.Series([e['equity'] for e in equity_curve])
    final_value = equity_series.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # Calculate returns
    returns = equity_series.pct_change().dropna()

    # Sharpe Ratio (annualized, assuming daily data)
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min()

    # Trade statistics
    total_trades = len(trades)
    if total_trades > 0:
        winning = [t for t in trades if t.get('profit', 0) > 0]
        losing = [t for t in trades if t.get('profit', 0) <= 0]

        win_rate = len(winning) / total_trades
        avg_profit = np.mean([t['profit'] for t in winning]) if winning else 0
        avg_loss = np.mean([t['profit'] for t in losing]) if losing else 0

        total_gains = sum(t['profit'] for t in winning)
        total_losses = abs(sum(t['profit'] for t in losing))
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
    else:
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
        profit_factor = 0
        winning = []
        losing = []

    return PerformanceMetrics(
        strategy_name=strategy_name,
        start_date=equity_curve[0].get('timestamp', 'N/A'),
        end_date=equity_curve[-1].get('timestamp', 'N/A'),
        initial_capital=initial_capital,
        final_value=final_value,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=total_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        avg_profit=avg_profit,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        equity_curve=equity_curve
    )
