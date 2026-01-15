# Project Status & Strategy Update (2026-01-15)

## 1. Current Status (Phase 2 Wrap-up)

### A. Model Training Results
1.  **Universal Transformer (V3)**
    *   **Data**: Full 5Y Dataset (2020-2025) with "Fill 0" for missing features.
    *   **Config**: 100 Epochs, Window 120, Batch 256.
    *   **Result**: Converged well (Loss 0.56).
    *   **Assessment**: **Ready for Deployment**. Structurally robust to data noise.

2.  **Universal LightGBM**
    *   **Result**: Score 0.20 (Positive but low).
    *   **Issue**: Validation set showed **0 trades**.
    *   **Assessment**: Failed Experiment. Model became too conservative ("shy") likely due to confusing signals from the "Fill 0" period (2020-2021).

### B. Data Infrastructure Audit
We performed a deep probe of Binance historical archives (`data.binance.vision`) to explain the data gaps:
*   **2020 - 2021.06**:
    *   **Open Interest / Long-Short Ratio**: **MISSING (404)**.
    *   **Funding Rate**: **MISSING** in daily dumps (`premiumIndex`), but **AVAILABLE** in `premiumIndexKlines` (requires complex scraping/parsing).
*   **2021.12 - Present**:
    *   All data (OI, Funding, Price) is fully available and high quality.

---

## 2. Strategic Pivot: Phase 3 "Clean Data"

Based on the audit, we are shifting strategy to prioritize **Data Quality over Data Quantity**.

### The "Clean Data" Hypothesis
*   **Problem**: The "Fill 0" strategy for 2020-2021 forces models to learn two conflicting regimes:
    1.  "Technical Only" (2020-2021, where OI=0).
    2.  "Technical + Alpha" (2022-2025, where OI=Real).
*   **Solution**: Discard the "Technical Only" era. Train solely on the "Alpha Era" (2021-12 to Present).
*   **Benefit**: Models will no longer receive conflicting signals. Decision boundaries will be sharper.

### Action Plan
1.  **Immediate (Phase 2.5)**:
    *   Deploy **Transformer V3 (Full Data)** to production.
    *   *Rationale*: "Better than nothing". It has seen 5 years of market cycles (Bull/Bear/Chop) which provides macro-robustness, even if micro-precision is noisy.

2.  **Next Step (Phase 3.1)**:
    *   Create `CLEAN_FULL_2022_2025.csv` (Start date: **2021-12-01**).
    *   Retrain **LightGBM** on this clean dataset (Expectation: Fixes the "0 trades" issue).
    *   Retrain **Transformer V4** on this clean dataset (Expectation: Faster convergence, higher precision).

---

## 3. Key Decisions Log
*   **Data Imputation**: We confirmed that manual filling of 2020 OI data is impossible (source does not exist).
*   **Pipeline**: We will maintain two parallel pipelines for now:
    *   `Legacy`: 5-Year Full Data (for robustness checks).
    *   `Modern`: 3.5-Year Clean Data (for Alpha generation).
