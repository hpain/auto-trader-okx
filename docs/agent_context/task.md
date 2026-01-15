# Task: Strategic Pivot to Structural Alpha

## Objective
Shift from pure price prediction to structural arbitrage. Implement Funding Rate Arbitrage as the first "Real Alpha" baseline.

## Todo List
- [x] **P1-1: Implement Funding Rate Arbitrage**
    - [x] Create `strategies/funding_arb.py` (Core Logic).
    - [x] Create `tests/test_funding_arb.py` (Unit Tests).
    - [x] Register in `strategies/strategy_manager.py`.
    - [x] Enable in `config/settings.yaml`.
    - [x] Create verification script `scripts/check_funding_rates.py`.
- [x] **Deployment**
    - [x] Git Push changes to `main_lt`.
    - [x] User: Pull & Restart on Cloud Server (Dual Process Architecture).
- [x] **P1-2: Refactor Run Live (Minimal)**
    - [x] Create `run_live_simple.py` (Blueprint created).
    - [x] Implement Dual-Leg Service in `docker-compose.yml`.
    - [x] **VERIFY**: Arb Bot running on ETH/USDT in parallel with ML Bot.
- [ ] **Phase 2: Predictive Alpha (ML Bot)**
    - [x] **P2-1: Advanced Feature Engineering**
        - [x] Mine Funding Rate & OI Logic (from Phase 1).
        - [x] Implement `features/crypto_factors.py` (L/S Ratio, Taker Vol).
        - [x] Generate comprehensive dataset (Integrated in pipeline).
    - [x] **P2-2: Model Upgrade**
        - [x] Refactor `TransformerStrategy` for Multi-Head Attention.
        - [x] Optimize LightGBM defaults (Skipped for Transformer focus).
        - [x] Validate on Backtest (Verified via generic script).
    - [x] **P2-3: Live Simulation & Launch**
        - [x] Fix `auto-trader-ml` Docker config (Rebuilt).
        - [x] Deploy ML Bot in `Mock Mode` (Verified Running).
    - [x] **P2-4: Verification**
    - [x] **P2-4: Verification**
        - [x] Verify Feature/Model Dimensions (139/120) via `verify_real_model.py`.
        - [x] Confirm Remote Health (Logs checked).
    - [x] **P2-5: Stabilization (Current)**
        - [x] Fix Exit 137 (OOM) on `auto-trader-ml` startup (Fixed via Memory Limit & Cleanup).
        - [ ] Optimize Feature Generation Memory Usage (Defer to Phase 3).

- [ ] **Phase 3: Alpha Optimization**
    - [x] **P3-1: Aggressive Arb Tuning**
        - [x] Lower Funding Arb Entry via Dynamic Logic (`Rate > Cost/Days`).
        - [x] Update `run_live_simple.py` logic.
    - [x] **P3-2: System Robustness (Tenacity)**
        - [x] Implement `tenacity` retry logic in `CcxtExchange`.
        - [x] Downgrade 502/Network errors to WARNING (No Crash).
        - [x] Sanitize HTML logs for cleaner debugging.
        - [x] Sanitize HTML logs for cleaner debugging.
        - [x] Optimize Docker build cache order.
        - [x] Implement Auto-Healing for InsufficientFunds (Precision mismatch fix).
    - [ ] **P3-3: Dynamic Sensitivity Tuning**
        - [x] Analyze V2 baseline bias (0.69) vs Live (0.51).
        - [ ] Decision: Maintain 0.55 threshold (User: "No changes for now").
    - [/] **P3-4: Retrain LGB with Full Data**
        - [x] Confirm data availability (BTCUSDT_FULL_2020_2025).
        - [x] Fix `UnboundLocalError` in `improved_evolution.py`.
        - [x] Train BTC Model (Best Score: 0.0933).
        - [x] Merge BTC+ETH Data for Universal Model (Applied +20y Shift).
        - [x] Merge BTC+ETH Data for Universal Model (Applied +20y Shift).
        - [x] Train Universal Model (Completed - Score: 0.20, Val Trades: 0).
            - [x] Fix Sparse Feature Dropping (OI/Sentiment).
            - [x] Optimize Hyperparams (Expanded Search Space).
            - [x] Tune Labels (TP 0.8%, SL 0.5%).
        - [x] **Optional**: Train Universal Transformer (Completed - Loss: 0.5604).
        - [x] **Fallback**: Deploy BTC 5-Year Model (Score: 0.0933) - *Current Live Config*.

- [ ] **Phase 3: Alpha Optimization (Clean Data Era)**
    - [ ] **P3-1: Dataset Purification**
        - [ ] Create `CLEAN_FULL_2022_2025.csv` (Start from 2021-12-01).
        - [ ] Verify zero NaN in OI/Funding columns.
    - [ ] **P3-2: Retrain Models on Clean Data**
        - [ ] Train LightGBM (Expect >0.20 score & valid trades).
        - [ ] Train Transformer V4 (Compare vs V3).
    
    - [x] **P3-3: Aggressive Arb Tuning**
        - [x] Lower Funding Arb Entry via Dynamic Logic (`Rate > Cost/Days`).
        - [x] Update `run_live_simple.py` logic.
    - [x] **P3-4: System Robustness**
        - [x] Implement `tenacity` retry logic.
        - [x] Auto-Healing for InsufficientFunds.
