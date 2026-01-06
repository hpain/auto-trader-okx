import logging
import pandas as pd
from typing import Dict, Optional, Any
from .base_strategy import BaseStrategy

class FundingRateArbitrageStrategy(BaseStrategy):
    """
    P1-1: Structural Alpha - Funding Rate Arbitrage.
    
    Logic:
    - Monitor Perpetual Funding Rates.
    - If Rate > positive_threshold:
        - Market is Overheated (Longs paying Shorts).
        - Action: SHORT Perp + LONG Spot (Earn Funding Fee).
    - If Rate < negative_threshold:
        - Market is Oversold (Shorts paying Longs).
        - Action: LONG Perp + SHORT Spot (Earn Funding Fee).
    - If Rate returns to neutral_threshold:
        - Close Positions.
        
    Risk:
    - Delta Neutral (Hedging Spot vs Perp).
    - Exposure: Only to spread divergence (Basis Risk) and Liquidation risk if leverage used.
    """
    
    def __init__(self, 
                 strategy_name="FundingArb_v1", 
                 positive_threshold=None,  # If None, use dynamic calculation
                 negative_threshold=None, 
                 neutral_threshold=0.0001,
                 transaction_cost=0.003,   # 0.3% Round Trip Estimate (Fees+Slippage)
                 target_days=5.0,          # Target Breakeven Time (Days)
                 leverage=1.0):
        super().__init__(strategy_name, {}) 
        self.logger = logging.getLogger(__name__)
        
        # Fixed Thresholds (Optional)
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.neutral_threshold = neutral_threshold
        
        # Dynamic Parameters
        self.transaction_cost = transaction_cost
        self.target_days = target_days
        self.leverage = leverage
        
        self.current_state = "NEUTRAL" 

    def _calculate_dynamic_threshold(self) -> float:
        """
        Calculate required funding rate to break even within target_days.
        Formula: Required_Daily_Yield = Cost / Days
                 Required_8h_Rate = Required_Daily_Yield / 3
        """
        required_daily_yield = self.transaction_cost / self.target_days
        required_rate = required_daily_yield / 3.0
        return required_rate

    def generate_signal(self, df: pd.DataFrame, symbol: str = "", funding_rate: Optional[float] = None) -> float:
        if funding_rate is None:
            if df is not None and 'fundingRate' in df.columns:
                funding_rate = float(df['fundingRate'].iloc[-1])
            else:
                self.logger.warning(f"[{symbol}] No funding rate data available for Arb.")
                return 0.0

        # Determine Effective Thresholds
        if self.positive_threshold is not None:
            eff_pos_thresh = self.positive_threshold
            eff_neg_thresh = self.negative_threshold if self.negative_threshold else -self.positive_threshold
        else:
            # Dynamic Calculation
            dynamic_min = self._calculate_dynamic_threshold()
            eff_pos_thresh = dynamic_min
            eff_neg_thresh = -dynamic_min
            
        score = 0.0
        
        # Check Entry
        if funding_rate > eff_pos_thresh:
            score = -1.0
            if self.current_state != "POSITIVE_ARB":
                self.logger.info(f"[{symbol}] Rate {funding_rate:.6f} > DynThresh {eff_pos_thresh:.6f} (Cost:{self.transaction_cost*100:.1f}%, Days:{self.target_days}). Signal SHORT Perp.")
                self.current_state = "POSITIVE_ARB"
                
        elif funding_rate < eff_neg_thresh:
            score = 1.0
            if self.current_state != "NEGATIVE_ARB":
                self.logger.info(f"[{symbol}] Rate {funding_rate:.6f} < DynThresh {eff_neg_thresh:.6f}. Signal LONG Perp.")
                self.current_state = "NEGATIVE_ARB"
                
        # Check Exit (Neutral)
        elif abs(funding_rate) < self.neutral_threshold:
            score = 0.0
            if self.current_state != "NEUTRAL":
                self.logger.info(f"[{symbol}] Rate {funding_rate:.6f} normalized. Signal NEUTRAL.")
                self.current_state = "NEUTRAL"
        
        return score

