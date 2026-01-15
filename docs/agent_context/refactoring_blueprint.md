# Refactoring Blueprint: Introducing Tenacity for Robust Retries

## Goal
Replace ad-hoc `try-except` blocks and manual retry loops in `exchange/ccxt_exchange.py` with declarative, robust `tenacity` retry logic.

## Strategy

### 1. Global Retry Configuration
We will define a reusable retry configuration for network operations.
- **Stop**: After 3 attempts (balance between trying and failing fast).
- **Wait**: Exponential backoff (1s, 2s, 4s...) + Jitter (randomness to prevent thundering herd).
- **Retry Condition**: On `NetworkError`, `RequestTimeout`, `ExchangeNotAvailable`, `RateLimitExceeded`.
- **Failure Callback**: If all retries fail, log a warning and return a default "safe" value (Empty DataFrame, 0.0, None) instead of raising an exception. This maintains the "Graceful Degradation" for secondary exchanges.

### 2. Methods to Target
*   `fetch_candles` -> Returns `pd.DataFrame()` on failure.
*   `get_current_price` -> Returns `0.0` on failure.
*   `get_balance` -> Returns `0.0` on failure.
*   `load_markets` -> Just Logs on failure (already retry logic present, replace with tenacity).

### 3. Implementation Details

#### Imports
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, return_last_value, RetryError
from tenacity import before_sleep_log
```

#### Retry Decorator Factory
Create a helper to generate decorators easily:
```python
def robust_exchange_retry(retry_count=3, default_return=None):
    def return_default_on_failure(retry_state):
        ex_id = "unknown"
        # Try to extract 'self.exchange.id' from the first arg (self)
        try:
            if retry_state.args and hasattr(retry_state.args[0], 'exchange'):
                 ex_id = retry_state.args[0].exchange.id
        except:
             pass
        
        logger.warning(f"🛑 All {retry_count} retries failed for {retry_state.fn.__name__} on {ex_id}. Returning default: {default_return}. Final Error: {retry_state.outcome.exception()}")
        return default_return

    return retry(
        stop=stop_after_attempt(retry_count),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((NetworkError, RequestTimeout, ExchangeNotAvailable, RateLimitExceeded)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry_error_callback=return_default_on_failure 
    )
```
*Note: `retry_error_callback` is the key to preventing crashes while still retrying aggressively.*

## Execution Steps
1.  Install `tenacity`.
2.  Modify `exchange/ccxt_exchange.py` to import `tenacity` and define the wrapper.
3.  Decorate critical methods.
4.  Verify functionality.
