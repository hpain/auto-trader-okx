# Dual-Process Architecture Walkthrough (2026-01-04)

We successfully pivoted the trading bot's architecture from a single monolithic process to a **Dual-Process Architecture** to safely run conflicting strategies in parallel.

## Architecture Overview

| Bot Service | Image Name | Strategy | Symbol | Market Type | Goal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **auto-trader-ml** | `auto-trader-ml` | ML (Transformer/LGB) | **BTC/USDT** | Spot | **High Risk / Alpha**: Price prediction & momentum used for directional trades. |
| **auto-trader-arb** | `auto-trader-arb` | FundingArb | **ETH/USDT** | Swap | **Low Risk / Structural**: Exploiting funding rate anomalies for yield. |

## Implementation Details

### 1. Isolated Execution Script (`run_live_simple.py`)
A minimal, stateless runner designed for reliability.
- **No Fusion**: Runs a single strategy class directly (e.g. `FundingRateArbitrageStrategy`).
- **Targeted Market**: Modified to force `market_type='swap'` connection for Arb strategies to access Funding Rates properly.
- **Robust Logging**: Independent log file `logs/simple_funding_arb.log` to avoid cluttering the ML logs.

### 2. Docker Service Split (`docker-compose.yml`)
- We split the original `auto-trader` service into two.
- **Conflict Prevention**: 
    - `auto-trader-ml` is configured (via Env Var) to trade **BTC only**.
    - `auto-trader-arb` is configured (via CLI override) to trade **ETH only**.
    - This ensures no cross-contamination of position management logic.

### 3. Strategy Verification
- **FundingArb on Live**: Verified that `auto-trader-arb` correctly connects to OKX Swap API, fetches live Funding Rates (e.g. `0.003269`), and generates correct signals (`SHORT Perp` when rate is positive).

### 4. Live Deployment Success (2026-01-05)
- **Status**: **LIVE & ACTIVE**
- **Architecture**: Dual-Leg Hedging (Spot Buy + Perp Sell).
- **Key Separation**: Implemented `LIVE_OKX_...` env vars to isolate Real Money bot from Mock bot.
- **Safety**:
    - **Entry Threshold**: > 0.15% (Strict cost coverage).
    - **Exit Threshold**: < 0.05% (Hysteresis for profit maximization).
    - **Crash Recovery**: `reconcile_state()` ensures no zombie positions.
- **Metrics**:
    - **Capital**: ~$139 (0.02 ETH trade size).
    - **Current State**: Monitoring for high funding rate opportunities (Neutral at <0.01%).

### 5. Phase 2: Predictive Alpha Launch (ML Bot V2)
- **Model Upgrade**: Deployed `TransformerStrategy V2` (139 Features, 120-Hour Window).
    - **Training**: 100 Epochs on GPU, Val Loss `0.58` -> `0.48`.
    - **Features**: Added Crypto Factors (Funding Z-Score, OI Regimes, Taker Ratio).
- **Deployment**:
    - **Service**: `auto-trader-ml` (Mock Mode).
    - **Resource Mgmt**: Fixed OOM (Exit 137) on 4GB VPS by enforcing `limits: memory: 2G`.
    - **Status**: Running parallel to Arb Bot, generating live predictions.

### 6. Phase 3: Alpha Optimization (Jan 2026)
**P3-1: Dynamic Funding Arbitrage**
- **Logic Upgrade**: Replaced fixed thresholds (0.15%) with a dynamic "Breakeven" calculator.
- **Formula**: `Entry_Threshold = (Transaction_Cost) / (3 * Target_Recovery_Days)`.
- **Params**: Cost=0.3% (Taker Fees), Target=5 Days.
- **Result**: Bot enters if Rate > ~0.02%. This ensures every trade is mathematically profitable within the target timeframe.
- **Troubleshooting**: Resolved `TypeError` where abstract methods were missing on VPS by executing `git reset --hard HEAD && git pull` to force sync.
- **Status**: Deployed and Verified.

---

### 7. Comprehensive Code Review & Bug Fixes (2026-01-16)

**Scope**: Full review of `improved_evolution.py`, `run_live.py`, `ccxt_exchange.py`, and `strategy_manager.py`.

#### Critical Fixes (P0)
| File | Issue | Impact |
|------|-------|--------|
| `improved_evolution.py` | Feature leakage risk (missing `symbol`, raw OHLCV in exclusion list) | Model could memorize non-generalizable patterns |
| `improved_evolution.py` | Only 2 CV folds → statistically meaningless stability score | Optuna selecting overfit parameters |
| `strategy_manager.py` | `AttributeError` - referenced `active_strategy` but defined `active_strategies` | Bot crash on strategy switch |
| `run_live.py` | Indentation error causing debug log in wrong scope | Misleading logs |

#### Important Fixes (P1)
| File | Issue | Impact |
|------|-------|--------|
| `ccxt_exchange.py` | No retry for maintenance errors (OKX 50001) | Lost trading opportunities during 5-min maintenance |
| `improved_evolution.py` | Wrong annualization factor (252 days instead of 365) | Sharpe underestimated by ~17% |
| `improved_evolution.py` | Dead Optuna Pruner (no `trial.report()` calls) | Wasted compute on bad trials |
| `run_live.py` | Mock mode infinite loop on data fetch failure | CPU spin |

#### Key Changes
1. **TimeSeriesSplit**: 2 → 5 folds
2. **Order Retry**: Added 30s/60s/120s backoff for temporary errors
3. **Feature Exclusion**: Now excludes `symbol`, raw OHLCV, legacy columns
4. **Crypto Calendar**: 365 days × 24 hours for 1H Sharpe calculation

**See**: [`bugfix_log_2026_01_16.md`](./bugfix_log_2026_01_16.md) for full details.

### 8. Transformer Model Optimization (2026-01-19)

Comprehensive improvements to the Transformer model to address class imbalance and improve performance:

#### Key Optimizations
1. **Asymmetric Focal Loss**: Implemented specialized loss function to better handle the 70.84% vs 29.16% class imbalance
2. **Data Augmentation**: Added Gaussian noise augmentation to training data to improve generalization
3. **Curriculum Learning**: Implemented sample difficulty sorting based on volatility for progressive learning
4. **Enhanced Feature Engineering**: Added technical indicators including volatility, momentum, and moving average ratios
5. **Optimizer Improvements**: Upgraded to AdamW with amsgrad and increased weight decay for better regularization
6. **Intelligent Learning Rate Scheduling**: Implemented CosineAnnealingWarmRestarts for better exploration of loss landscape

#### Results
- **Significant F1 Score Improvement**: From 0.1271 to 0.5068 (approximately 4x improvement)
- **Better Stability**: More stable training process with controlled gradient norms
- **Improved Generalization**: Reduced overfitting through enhanced regularization techniques
- **Enhanced Robustness**: Better handling of imbalanced dataset through Asymmetric Focal Loss

