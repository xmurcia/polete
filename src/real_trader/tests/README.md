# Real Trader Tests

Complete test suite for the real_trader module.

## Test Structure

```
tests/
├── test_auth.py              # Authentication tests
├── test_models.py            # Data model tests
├── test_order_manager.py     # Order management tests
├── test_position_tracker.py  # Position tracking tests
├── test_balance_manager.py   # Balance management tests
└── test_integration.py       # Full integration tests
```

## Running Tests

### All Tests
```bash
cd real_trader
./venv/bin/pytest tests/ -v
```

### Specific Test File
```bash
./venv/bin/pytest tests/test_auth.py -v
```

### Single Test
```bash
./venv/bin/pytest tests/test_auth.py::test_auth_initialization -v
```

### With Output
```bash
./venv/bin/pytest tests/ -v -s
```

### Coverage Report
```bash
./venv/bin/pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### Unit Tests
Test individual components in isolation:
- `test_auth.py` - Authentication logic
- `test_models.py` - Data structures
- `test_order_manager.py` - Order management logic
- `test_position_tracker.py` - Position tracking logic
- `test_balance_manager.py` - Balance management logic

### Integration Tests
Test full workflows with real API calls:
- `test_integration.py` - Complete trading lifecycle

## Important Notes

### Real API Calls
Some tests make **real API calls** to Polymarket:
- `test_auth.py::test_test_connection`
- `test_integration.py` - All tests
- Tests marked with `@pytest.mark.asyncio`

These tests require:
- Valid `.env` configuration
- Active internet connection
- Sufficient POL for gas (for write operations)

### Safe Tests
These tests **do not** make real API calls:
- `test_models.py` - All tests
- Most unit tests (non-async)

## Test Fixtures

Common fixtures available:
- `auth` - Initialized PolyAuth instance
- `order_manager` - Initialized OrderManager
- `tracker` - Initialized PositionTracker
- `balance_manager` - Initialized BalanceManager
- `full_setup` - All managers initialized together

## Adding New Tests

1. Create test file: `test_<module>.py`
2. Import necessary modules
3. Add fixtures if needed
4. Write test functions (prefix with `test_`)
5. Use `@pytest.mark.asyncio` for async tests

Example:
```python
import pytest
from auth import PolyAuth

@pytest.fixture
def auth():
    return PolyAuth()

def test_something(auth):
    result = auth.get_wallet_address()
    assert result is not None

@pytest.mark.asyncio
async def test_async_operation(auth):
    client = auth.get_client()
    # ... async test
```

## CI/CD Integration

To run tests in CI:
```bash
pip install pytest pytest-asyncio pytest-cov
pytest tests/ -v --cov=.
```

## Troubleshooting

### Import Errors
Ensure parent directory is in path:
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

### Async Test Failures
Install pytest-asyncio:
```bash
pip install pytest-asyncio
```

### API Connection Failures
- Check `.env` configuration
- Verify internet connection
- Check Polymarket API status
