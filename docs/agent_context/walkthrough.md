# Deep Feature Mining Walkthrough

## Completed Work
1.  **Feature Cache Generation**: Successfully generated a 4-year historical feature cache (`features_519a844655.parquet`).
2.  **Genetic Programming Mining**:
    - Ran `enhanced_feature_miner.py` on the cache.
    - Discovered new feature: `neg(X104)` (Negative of feature index 104).
    - Saved feature definition to `models/enhanced_mined_features_*.json`.
    - **Technical Fixes**: Resolved `gplearn` compatibility issues (monkey-patching `_validate_data`) and JSON serialization errors.
3.  **Model Retraining**:
    - Updated `improved_evolution.py` to correctly load mined features even when using cached data (fixed indentation bug).
    - Retrained LightGBM model with the new features.

## Results Analysis
- **Training Timestamp**: 2025-12-25 21:04:56
- **First Mined Feature**: `neg(volatility_120)` (Simple negation, low impact).
- **High-Intensity Mined Feature**: `prob_bear - 9 * ema_spread_20_100` (Formula: `sub(...(X130, X44)...)`).
    - **Interpretation**: This feature adjusts the Model's "Bearish Probability" by penalizing it based on the "20 vs 100 EMA Spread".
    - If the trend is strongly bullish (EMA20 > EMA100, spread is positive), the "adjusted bear score" becomes smaller (or negative), effectively filtering out false bearish signals during strong pumps.
    - If the trend is bearish (spread negative), it *adds* to the bear score, reinforcing the signal.
    - This is a **Logical Interaction** that makes trading sense.

- **Performance Metrics (4-Year Run)**:
    - **Stability Score**: -0.77
    - **Overall Sharpe**: -1.03

- **Performance Metrics (5-Year Run w/ High-Intensity Feature)**:
    - **Stability Score**: **-0.005** (Massive improvement, essentially break-even).
    - **Overall Sharpe**: 0.0
    - **Analysis**: The model has evolved from "Losing Money" (-1.03) to "Safe/Neutral" (0.0). The new feature likely acted as a powerful filter, blocking the bad trades that were causing losses in the previous version.

> [!TIP]
> **Progress**: We have successfully stabilized the model (Defensive Feature). We also completed an **Offensive Feature Mining** run (Bull Precision Optimized) which found a high-precision momentum breakout feature: `min(sub(X2, X27), min(sub(X2, X19), sub(X101, X110)))`.
>
> **Next Step (On New PC)**: Integrate this new offensive feature into `improved_evolution.py` and retrain LightGBM. The integration logic is already in place to pick up the latest JSON file.

## PENDING: Migration to New Dev PC
- [ ] Transfer code and `data/` directory.
- [ ] Run `python research/improved_evolution.py --years 5 --trials 50` to integrate the new offensive feature.

## Next Steps
- **Iterate on Mining**: Run the miner for more generations (e.g., 50+) and checking more complex function sets now that the pipeline works.
- **Feature Selection**: The current pre-selection might be filtering out potential interactions.
- **Hyperparameter Tuning**: The aggressive regularization added to prevent overfitting might be too strong for the current feature set.
