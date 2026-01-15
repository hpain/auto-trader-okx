# Implementation Plan: Strategic Pivot to Structural Alpha

## Goal
Shift the bot's core logic from "Predicting Price Direction" (Low Alpha) to "Capturing Structural Yield" (High Alpha), as recommended in the expert review.

## User Review Required
> [!IMPORTANT]
> This is a major change in strategy philosophy. We are moving away from "AI predicting the market" to "Math capturing market inefficiencies".

## Proposed Changes

### P0: Paradigm Shift & Cognitive Refactor (Immediate)
> *Ref: improve_plan.txt "Cognitive & Paradigm Refactor"*
- **Rule**: All new strategies MUST start in `research/*.ipynb`. No direct coding in `src/`.
- **Role Definition**:
    - **Model != Decision Maker**.
    - **Model == Risk Filter / State Gating**.

### P1: Strategy Layer Refactor (The Life Line)
> *Ref: improve_plan.txt "P1-1: Immediate Funding Rate Arbitrage"*
- **Action**: Implement `FundingRateArbitrageStrategy`.
    - **Goal**: 5-15% APY, Delta Neutral.
    - **Logic**: 
        - Monitor Funding Rates (via `run_live_simple.py`).
        - **Entry**: Rate > 0.15% (Strict breakeven filter).
        - **Exit**: Rate < 0.05% (Hysteresis profit maximization).
    - **Execution**: Dual-Leg Hedging (Short Perp + Buy Spot) with State Reconciliation.

### P1-2: Execution Entry Point Refactor (The Isolator)
> *Ref: verify_alpha_analysis.txt "Run Live Minimal Version"*
- **Action**: Create `run_live_simple.py` (or refactor `run_live.py`).
- **Goal**: Run **ONE** strategy, **ONE** symbol, **ZERO** assumption.
- **Changes**:
    - Remove hardcoded strategy weights/fusion.
    - Remove automatic "Feature Engineering" pipeline (unless strategy demands it).
    - Allow CLI arg: `--strategy funding_arb`.

### P2: Research Layer (Alpha Boundaries)
> *Ref: improve_plan.txt "P2: Standard Research Template"*
- Define "Allowed Alpha": Funding Rates, Basis, Cross-Exchange Spreads.
- Ban "Forbidden Alpha": Simple Technical Analysis Directional Prediction.

### P3: Engineering & Structure (Mid Priority)
> *Ref: improve_plan.txt "Module Decoupling & Cleanup"*
- **Archive**: Move old experiments (`analysis/*`) to `archive/`.
- **Refactor**: Split `strategies/` into `arbitrage/` and `trend/`.

## Verification Plan (For P1)

### Automated Tests
- `tests/test_funding_arb.py`:
    - Mock positive/negative funding rates.
    - Verify correct Long/Short hedging orders are generated.
    - Verify position sizing is equal (Delta Neutral).

### Manual Verification
- **Unit Logic**: Run a script to fetch current OKX funding rates and print potential yield.
- **System Logic**: Deploy updates to Docker and verify logs show "Funding Arb" initialized.
### P2: Predictive Alpha (Transformer v2)
> *Ref: P2-Series in task.md*
- **Architecture**: `TransformerStrategy` with 120-hour window.
- **Input**: 139 Features (Price + Funding + Regime + Taker Flow).
- **Execution**: Mock Mode (Simulation) running in parallel with Arb Bot.
- **Infrastructure**: GPU Training -> Artifact Transfer -> Docker Deployment with 2G Limits.

## Verification Plan (For P2)
- **Dimensions**: `verify_real_model.py` checks 139/120 alignment.
- **OOM Safety**: Enforce `deploy.resources.limits.memory: 2G` in Docker.
- **Logic**: Inspect Logs for `Transformer Prediction Prob`.

### P3: Dynamic Alpha Optimization
- **Logic**: Rate > Cost / (Cycles * RecoveryDays).
- **Implementation**: `FundingRateArbitrageStrategy` auto-calculates threshold based on 0.3% cost and 5-day target.
- **Goal**: "Never miss a profitable trade" (Entry ~0.02%).
