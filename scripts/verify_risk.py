import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader.risk_engine import RiskEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_risk_engine():
    logger.info("--- Testing Risk Engine ---")
    
    # 1. Initialize Engine (Target Vol = 20%)
    engine = RiskEngine(target_volatility=0.20)
    
    # 2. Simulate LOW Volatility (Stable Market)
    # Annualized Vol of 0.01 per hour roughly translates to ~29% annualized if *sqrt(24*365)
    # Let's make it very stable: 0.1% hourly moves -> ~9% annualized
    # returns = np.random.normal(0, 0.001, 100)
    logger.info("\n[Scenario 1: Stable Market (Low Vol)]")
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    low_vol_returns = pd.Series(np.random.normal(0, 0.001, 100), index=dates) # ~9% annualized
    
    scalar_low = engine.calculate_volatility_scalar(low_vol_returns)
    logger.info(f"Target Vol: 20%")
    logger.info(f"Realized Vol: {low_vol_returns.std() * np.sqrt(365*24):.2%}")
    logger.info(f"Calculated Scalar: {scalar_low:.4f}")
    
    if scalar_low > 1.0:
        logger.info("✅ PASS: Scalar > 1.0 (Increased size in stable market)")
    else:
        logger.warning("❌ FAIL: Scalar should be > 1.0")

    # 3. Simulate HIGH Volatility (Crash)
    # 1% hourly moves -> ~93% annualized
    logger.info("\n[Scenario 2: High Volatility (Crash)]")
    high_vol_returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
    
    scalar_high = engine.calculate_volatility_scalar(high_vol_returns)
    logger.info(f"Target Vol: 20%")
    logger.info(f"Realized Vol: {high_vol_returns.std() * np.sqrt(365*24):.2%}")
    logger.info(f"Calculated Scalar: {scalar_high:.4f}")
    
    if scalar_high < 1.0:
        logger.info("✅ PASS: Scalar < 1.0 (Reduced size in volatile market)")
    else:
        logger.warning("❌ FAIL: Scalar should be < 1.0")
        
    logger.info("\n--- Risk Engine Verification Complete ---")

if __name__ == "__main__":
    test_risk_engine()
