import logging
import pandas as pd
from typing import Dict, Optional
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
                 positive_threshold=0.0001,  # 0.01% per 8h (approx 11% APY)
                 negative_threshold=-0.0001, 
                 neutral_threshold=0.00005,
                 leverage=1.0):
        super().__init__(strategy_name)
        self.logger = logging.getLogger(__name__)
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.neutral_threshold = neutral_threshold
        self.leverage = leverage
        
        # State tracking
        self.current_state = "NEUTRAL" # NEUTRAL, POSITIVE_ARB, NEGATIVE_ARB

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of abstract method.
        Wraps generate_signal to return a DataFrame of signals.
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        # Funding rate might be in data or fetched separately
        # For bulk processing, we iterate or use vector ops if column exists
        if 'fundingRate' in data.columns:
             # Vectorized logic for backtesting efficiently
            signals['signal'] = 0.0
            # Short Perp (Signal -1) if Rate > Positive Threshold
            signals.loc[data['fundingRate'] > self.positive_threshold, 'signal'] = -1.0
            # Long Perp (Signal 1) if Rate < Negative Threshold
            signals.loc[data['fundingRate'] < self.negative_threshold, 'signal'] = 1.0
        else:
            # Fallback for single row/live loop if needed, though this method is mostly for backtest
            pass
            
        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Implementation of abstract method.
        """
        return {
            "name": self.strategy_name,
            "type": "Arbitrage",
            "thresholds": {
                "positive": self.positive_threshold,
                "negative": self.negative_threshold,
                "neutral": self.neutral_threshold
            },
            "current_state": self.current_state
        }

    def generate_signal(self, df: pd.DataFrame, symbol: str = "", funding_rate: Optional[float] = None) -> float:
        """
        Generate signal based on funding rate.
        Returns:
            1.0 (Open Long Perp Arbitrage / Close Short Perp Arbitrage)
            -1.0 (Open Short Perp Arbitrage / Close Long Perp Arbitrage)
            0.0 (Hold)
        
        Note: The return signal follows standard convention:
            1.0 = Bullish on the instrument (Buy Perp) -> Implies Sell Spot if Arb
            -1.0 = Bearish on the instrument (Sell Perp) -> Implies Buy Spot if Arb
            
            However, BaseStrategy usually handles simple Directional bets. 
            For Arb, we need to handle the Dual-Leg execution in ExecutionHandler.
            Here we just signal the 'Bias' of the Perp leg.
        """
        if funding_rate is None:
            # Try to get from last row if available as column
            if df is not None and 'fundingRate' in df.columns:
                funding_rate = float(df['fundingRate'].iloc[-1])
            else:
                self.logger.warning(f"[{symbol}] No funding rate data available for Arb.")
                return 0.0

        score = 0.0
        
        # Logic Loop
        # 1. Check Entry
        if funding_rate > self.positive_threshold:
            # Positive Funding: Longs pay Shorts. We want to be SHORT Perp.
            # Signal -1.0
            score = -1.0
            if self.current_state != "POSITIVE_ARB":
                self.logger.info(f"[{symbol}] Funding Rate {funding_rate:.6f} > {self.positive_threshold}. Signal SHORT Perp (Arb).")
                self.current_state = "POSITIVE_ARB"
                
        elif funding_rate < self.negative_threshold:
            # Negative Funding: Shorts pay Longs. We want to be LONG Perp.
            # Signal 1.0
            score = 1.0
            if self.current_state != "NEGATIVE_ARB":
                self.logger.info(f"[{symbol}] Funding Rate {funding_rate:.6f} < {self.negative_threshold}. Signal LONG Perp (Arb).")
                self.current_state = "NEGATIVE_ARB"
                
        # 2. Check Exit (Regression to Mean)
        elif abs(funding_rate) < self.neutral_threshold:
            # Rate is normal. Close items.
            # If we were SHORT Perp (Positive Arb), we now Buy to close -> 1.0 (to neutral)
            # If we were LONG Perp (Negative Arb), we now Sell to close -> -1.0 (to neutral)
            # Actually, standard logic: Signal 0 means 'Close/Neutral'? 
            # Or PortfolioManager handles 'Hold if 0'?
            # Let's align with PortfolioManager: 
            # If Signal is 0, PM usually Holds. 
            # To force close, we might need a specific 'Close' logic or rely on PM's "Time based exit" or manual "Exit" signal.
            # For now, we return 0.0. The Arb Logic wrapper in PM needs to handle "Exit if condition lost".
            score = 0.0
            if self.current_state != "NEUTRAL":
                self.logger.info(f"[{symbol}] Funding Rate {funding_rate:.6f} normalized. Signal NEUTRAL.")
                self.current_state = "NEUTRAL"
        
        return score
