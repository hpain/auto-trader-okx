import pandas as pd
import numpy as np
import logging
import riskfolio as rp
from typing import Dict, List, Optional, Union

class RiskEngine:
    """
    Advanced Risk Management Engine using Riskfolio-Lib.
    Focuses on Hierarchical Risk Parity (HRP) and Volatility Targeting.
    """
    
    def __init__(self, target_volatility: float = 0.20):
        """
        :param target_volatility: Target annualized volatility (e.g., 0.20 for 20%)
        """
        self.logger = logging.getLogger(__name__)
        self.target_vol = target_volatility
        self.logger.info(f"RiskEngine initialized with Target Vol: {self.target_vol:.2%}")

    def calculate_hrp_weights(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio weights using Hierarchical Risk Parity (HRP).
        
        :param returns_df: DataFrame of asset returns (index=datetime, cols=assets)
        :return: Dictionary of asset weights {asset: weight}
        """
        if returns_df.empty or len(returns_df.columns) < 2:
            self.logger.warning("Insufficient data for HRP (need >1 asset). Returning equal weights.")
            cols = returns_df.columns if not returns_df.empty else []
            if len(cols) == 0: return {}
            return {col: 1.0 / len(cols) for col in cols}

        try:
            # Building the Portfolio Object
            port = rp.Portfolio(returns=returns_df)
            
            # Estimate inputs
            # method_mu='hist': Historical mean returns
            # method_cov='hist': Historical covariance
            port.assets_stats(method_mu='hist', method_cov='hist')
            
            # Estimate optimal portfolio:
            # model='HRP': Hierarchical Risk Parity
            # codependence='pearson': Pearson correlation
            # rm='MV': Mean Variance (Minimize Variance)
            # rf=0: Risk free rate
            # linkage='single': Single Linkage clustering
            # leaf_order=True: Optimal leaf ordering
            w = port.optimization(model='HRP', codependence='pearson', rm='MV', rf=0, linkage='single', leaf_order=True)
            
            weights = w.T.to_dict(orient='records')[0]
            # Ensure weights sum to 1 (Riskfolio usually does this, but good to check)
            
            self.logger.info(f"HRP Optimization Successful. Weights: {weights}")
            return weights

        except Exception as e:
            self.logger.error(f"HRP Optimization Failed: {e}", exc_info=True)
            # Fallback to equal weights
            cols = returns_df.columns
            return {col: 1.0 / len(cols) for col in cols}

    def calculate_volatility_scalar(self, returns_df: pd.DataFrame, window: int = 30) -> float:
        """
        Calculate a scalar to adjust position size based on recent volatility.
        If recent vol > target vol, scalar < 1 (reduce size).
        If recent vol < target vol, scalar > 1 (increase size, capped at 1.5).
        
        :param returns_df: DataFrame of returns (can be single asset or portfolio)
        """
        if len(returns_df) < window:
            return 1.0
            
        try:
            # Calculate recent annualized volatility
            recent_vol = returns_df.tail(window).std() * np.sqrt(365 * 24) # Assuming hourly data
            
            # If multi-asset, take the mean vol of the assets (simple approx) or portfolio vol if provided
            if isinstance(recent_vol, pd.Series):
                 current_vol = recent_vol.mean()
            else:
                 current_vol = recent_vol
                 
            if current_vol == 0:
                return 1.0

            # Volatility Scalar = Target / Current
            scalar = self.target_vol / current_vol
            
            # Cap the scalar to avoid excessive leverage or tiny positions
            # Let's say max leverage boost is 1.5x, min is 0.1x
            scalar = max(0.1, min(1.5, scalar))
            
            self.logger.info(f"Volatility Sizing: Target={self.target_vol:.2%}, Current={current_vol:.2%}, Scalar={scalar:.2f}")
            return scalar
            
        except Exception as e:
            self.logger.error(f"Vol Scalar Calc Failed: {e}")
            return 1.0
