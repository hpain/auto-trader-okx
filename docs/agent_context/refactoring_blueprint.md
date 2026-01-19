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

---

## Transformer Model Refactoring (2026-01-19)

### Goal
Significant improvements to the Transformer model architecture and training pipeline to address class imbalance and improve performance.

### Strategy

#### 1. Loss Function Enhancement
Implement Asymmetric Focal Loss to better handle imbalanced data (70.84% vs 29.16%):
- Different alpha and gamma parameters for positive and negative samples
- Reduces impact of dominant class while preserving minority class signal

#### 2. Data Augmentation
Add Gaussian noise augmentation to training features:
- Improves model generalization
- Reduces overfitting to specific patterns in training data
- Controlled noise level based on feature standard deviation

#### 3. Curriculum Learning
Implement sample difficulty-based sorting:
- Sort training samples by volatility or other complexity metrics
- Start with simpler patterns, gradually introduce complex ones
- Helps model learn fundamental patterns before complex interactions

#### 4. Feature Engineering Enhancement
Add technical indicators and derived features:
- Volatility measures (rolling std, ATR)
- Momentum indicators (returns, ROC)
- Moving average ratios (price to MA ratios)
- Enhanced feature selection process

#### 5. Optimizer and Scheduling Improvements
- Upgrade to AdamW optimizer with amsgrad
- Implement CosineAnnealingWarmRestarts for learning rate scheduling
- Increase weight decay for stronger regularization
- Add gradient clipping for training stability

#### 6. Performance Improvements
- Achieved significant F1 score improvement from 0.1271 to 0.5068
- Better handling of imbalanced dataset
- More stable training process
- Reduced overfitting through enhanced regularization

---

## Multi-class Transformer Enhancement (2026-01-19)

### Goal
Convert the model from binary classification to 3-class classification (timeout, stop_loss, take_profit) to better represent market outcomes.

### Strategy

#### 1. Architecture Conversion
Update model architecture from binary to 3-class:
- Change output layer from 1 neuron to 3 neurons
- Update forward pass to output 3-dimensional logits
- Use softmax activation for multi-class probability distribution

#### 2. Loss Function Update
Replace BCEWithLogitsLoss with CrossEntropyLoss:
- Proper handling of multi-class integer labels
- Implementation of class-weighted loss for imbalanced classes
- Better gradient properties for multi-class problems

#### 3. Training Loop Modifications
Update training and validation loops:
- Handle integer class labels instead of binary probabilities
- Update accuracy and F1 score calculations for multi-class
- Modify early stopping criteria for multi-class performance

#### 4. Data Processing Updates
Enhance data preprocessing for 3-class labels:
- Proper conversion of triple barrier labels to 3-class format
- Validation of label integrity for multi-class problem
- Enhanced feature engineering for multi-class prediction

#### 5. Regularization and Generalization
- Implement stronger regularization for increased model capacity
- Add smart data augmentation with different noise levels for training/validation
- Improve early stopping with multi-metric monitoring

#### 6. Performance Outcomes
- Maintained high F1 score (0.5068) while expanding to 3-class problem
- Better interpretability with specific market outcome predictions
- Enhanced robustness across different market regimes
- Improved generalization across various market conditions
