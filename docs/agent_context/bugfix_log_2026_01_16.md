# Code Review & Bug Fixes Log (2026-01-16)

This document records all issues identified and fixed during the comprehensive code review session.

---

## Session Summary

| Category | Files Modified | Issues Fixed |
|----------|----------------|--------------|
| Training Pipeline | `research/improved_evolution.py` | 5 |
| Exchange Client | `exchange/ccxt_exchange.py` | 1 |
| Bot Entry Point | `run_live.py` | 2 |
| Strategy Manager | `strategies/strategy_manager.py` | 1 |

---

## 1. Training Pipeline Fixes (`improved_evolution.py`)

### 🔴 P0-1: Feature Leakage Risk - Incomplete Exclusion List (Line 362)

**Problem**: The `non_feature_cols` blacklist was missing critical columns that could leak information:
- `symbol` column (categorical, not numeric)
- Raw OHLCV (`open`, `high`, `low`, `close`, `volume`) - prefer derived features
- Legacy columns like `open_time`, `close_time`

**Fix**: Expanded exclusion list:
```python
non_feature_cols = [
    "ts", "dt", "date", "timestamp", "time", "symbol",
    "y", "future_high", "future_low", "future_close", "future_ret",
    "open", "high", "low", "close", "volume",
    "vol_ccy", "vol_ccy_quote", "confirm", "open_time", "close_time",
    "index", "level_0"
]
```

---

### 🔴 P0-2: Insufficient Cross-Validation Folds (Line 599)

**Problem**: `TimeSeriesSplit(n_splits=2)` - Only 2 folds provided:
- Extremely high variance in stability score calculation
- `std_sharpe` from 2 samples is statistically meaningless
- High risk of Optuna selecting overfit parameters

**Fix**: Increased to 5 folds for statistical reliability:
```python
tscv = TimeSeriesSplit(n_splits=5)
```

---

### 🟡 P1-1: Incorrect Annualization Factor (Line 512)

**Problem**: Used stock market's 252-day calendar instead of crypto's 365-day:
```python
trading_periods_per_year = {"1H": 252 * 24}.get(interval, 252)  # Wrong!
```
This **underestimated Sharpe Ratio by ~17%**.

**Fix**: Corrected to crypto calendar:
```python
trading_periods_per_year = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1H": 365 * 24,
    "4H": 365 * 6,
    "1D": 365
}.get(interval, 365 * 24)
```

---

### 🟡 P1-2: Dead Optuna Pruner (Line 682)

**Problem**: `MedianPruner` was configured but never received intermediate values:
- No `trial.report()` calls in the objective function
- Pruner could never prune any trial (dead code)

**Fix**: Added intermediate reporting and pruning logic:
```python
for fold_idx, (train_index, test_index) in enumerate(tscv.split(X_train)):
    # ... training logic ...
    
    if all_fold_sharpes:
        intermediate_score = np.mean(all_fold_sharpes)
        trial.report(intermediate_score, fold_idx)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
```

---

### 🟢 P2-1: Dead Code Cleanup

- Removed unused `y_val` variable (Line 538)
- Removed unused `matplotlib.pyplot` import (Line 876)
- Removed duplicate comments (Lines 596-598, 715-716)

---

## 2. Exchange Client Fixes (`ccxt_exchange.py`)

### 🟡 P1: No Retry for Maintenance Errors (Line 283)

**Problem**: When OKX returned `50001` (Service Temporarily Unavailable), the code:
1. Logged a warning
2. Returned `None` immediately
3. Bot waited until next hourly cycle to retry

For a 1-hour cycle bot, this meant losing opportunities during 5-minute maintenance windows.

**Fix**: Added immediate retry with conservative exponential backoff:
```python
except (RateLimitExceeded, RequestTimeout, NetworkError, ExchangeNotAvailable) as e:
    retry_intervals = [30, 60, 120]  # ~3.5 min total
    for attempt, wait_time in enumerate(retry_intervals, 1):
        logger.warning(f"Retry {attempt}/{max_retries} in {wait_time}s...")
        await asyncio.sleep(wait_time)
        try:
            result = await self.exchange.create_order(...)
            return result
        except (...) as retry_e:
            if attempt == max_retries:
                self._record_error()
                return None
```

---

## 3. Bot Entry Point Fixes (`run_live.py`)

### 🔴 P0-1: Indentation Error (Lines 140-142)

**Problem**: Debug log was incorrectly nested inside the Transformer try block:
```python
    # Note: 'funding_arb' is intentionally ignored here...

        trader_logger.info(f"DEBUG: Added Transformer...")  # Wrong indent!
```

**Fix**: Moved log outside the conditional block with correct indentation.

---

### 🟡 P1-1: Mock Mode Infinite Loop (Lines 296-301)

**Problem**: When market data fetch failed in mock mode:
```python
if not data_for_pm:
    if not args.mock:
        await asyncio.sleep(60)
    continue  # Instant retry in mock -> infinite fast loop!
```

**Fix**: Added small delay even in mock mode:
```python
await asyncio.sleep(1 if args.mock else 60)
```

---

## 4. Strategy Manager Fixes (`strategy_manager.py`)

### 🔴 P0-1: AttributeError - Wrong Attribute Name (Lines 236, 337+)

**Problem**: Code referenced `self.active_strategy` (singular) but the actual attribute was `self.active_strategies` (plural list):
```python
best_match = self.active_strategy if ...  # AttributeError!
```

**Fix**: Changed all references to use `self.active_strategies[0]` where a single strategy was needed.

---

## Verification

All modified files passed Python syntax check:
```bash
python -m py_compile research/improved_evolution.py
python -m py_compile exchange/ccxt_exchange.py
python -m py_compile run_live.py
python -m py_compile strategies/strategy_manager.py
```

---

## Recommendations for Future

1. **Add Unit Tests**: Critical paths like `create_order` retry logic should have unit tests.
2. **Strategy Weights from Config**: Move hardcoded weights in `run_live.py` to `settings.yaml`.
3. **Periodic State Sync**: Consider calling `sync_with_exchange()` at the start of each cycle.
4. **Feature Whitelist**: Use a whitelist approach instead of blacklist for feature selection.
