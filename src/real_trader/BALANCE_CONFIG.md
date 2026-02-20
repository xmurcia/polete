# Balance Manager Configuration

El `BalanceManager` ahora replica la lógica de sizing del `PaperTrader` del bot principal.

## Variables de Entorno

Añade estas variables a tu `.env`:

```bash
# Risk Management - Position Sizing
RISK_PCT_NORMAL=0.04      # 4% del capital por trade normal
RISK_PCT_LOTTO=0.01       # 1% del capital para lottery tickets
RISK_PCT_MOONSHOT=0.01    # 1% del capital para moonshots
MAX_MOONSHOT_BET=10.0     # Máximo $10 por moonshot
MIN_BET=5.0               # Apuesta mínima de $5

# Risk Limits
MAX_DAILY_LOSS=30         # Pérdida máxima diaria en $
MAX_EXPOSURE=0.99         # Exposición máxima (99% del capital)
```

## Uso del calculate_bet_size()

```python
from real_trader import BalanceManager, PolyAuth

auth = PolyAuth()
balance_mgr = BalanceManager(auth)
await balance_mgr.initialize()

# Trade normal (4% del capital)
bet_amount, shares = await balance_mgr.calculate_bet_size(
    price=0.20,
    strategy_tag="STANDARD"
)

# Trade con edge detectado (Kelly multiplier)
bet_amount, shares = await balance_mgr.calculate_bet_size(
    price=0.20,
    strategy_tag="STANDARD",
    edge_value=0.25  # 25% edge → multiplier 1.5x
)

# Moonshot (1% capital, max $10)
bet_amount, shares = await balance_mgr.calculate_bet_size(
    price=0.008,
    strategy_tag="MOONSHOT"
)

# Lottery ticket (1% capital)
bet_amount, shares = await balance_mgr.calculate_bet_size(
    price=0.05,
    strategy_tag="LOTTO"
)

# Hedge (2.5% capital)
bet_amount, shares = await balance_mgr.calculate_bet_size(
    price=0.15,
    is_hedge=True
)
```

## Lógica de Sizing

### 1. Base Percentage
- **STANDARD**: 4% del capital disponible
- **LOTTO**: 1% del capital
- **MOONSHOT**: 1% del capital
- **HEDGE**: 2.5% del capital

### 2. Kelly Multiplier (Sniper Mode)
Solo se aplica a trades STANDARD con edge detectado:
- Edge ≥ 40% → multiplier 2.0x
- Edge ≥ 20% → multiplier 1.5x
- Edge < 20% → multiplier 1.0x (sin boost)

### 3. Safety Belt
- Cap máximo: 10% del capital por trade
- Moonshot cap: $10 máximo
- Bet mínimo: $5

### 4. Validación
- No se puede apostar más del balance disponible
- Si el balance < min_bet, el trade se rechaza

## Ejemplos con $1000 de capital

| Strategy | Price | Edge | Base | Multiplier | Final % | Bet Amount | Shares |
|----------|-------|------|------|------------|---------|------------|--------|
| STANDARD | $0.20 | None | 4%   | 1.0x       | 4%      | $40        | 200    |
| STANDARD | $0.20 | 25%  | 4%   | 1.5x       | 6%      | $60        | 300    |
| STANDARD | $0.20 | 45%  | 4%   | 2.0x       | 8%      | $80        | 400    |
| MOONSHOT | $0.01 | None | 1%   | 1.0x       | 1%      | $10*       | 1000   |
| LOTTO    | $0.05 | None | 1%   | 1.0x       | 1%      | $10        | 200    |
| HEDGE    | $0.15 | N/A  | 2.5% | 1.0x       | 2.5%    | $25        | 166    |

*Moonshot capped at $10 max
