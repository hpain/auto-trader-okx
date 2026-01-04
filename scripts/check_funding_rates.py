import sys
import os
import pandas as pd
import logging
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exchange.factory import ExchangeFactory
from strategies.funding_arb import FundingRateArbitrageStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CheckFunding")

def main():
    # 1. Setup Exchange (OKX)
    exchange = ExchangeFactory.create_exchange('okx')
    
    # 2. Setup Strategy
    strategy = FundingRateArbitrageStrategy(
        positive_threshold=0.0001, # 0.01%
        negative_threshold=-0.0001
    )
    
    symbol = "BTC/USDT"
    logger.info(f"Fetching Funding Rate for {symbol}...")
    
    # 3. Fetch Funding Rate (Mocking fetch via exchange if method exists, else using hardcoded check)
    # Check if exchange has fetch_funding_rate
    # If not, we might need to rely on ccxt direct
    try:
        # Assuming exchange has a method or we use ccxt
        # aggregated_exchange usually exposes fetch_funding_rates
        # Let's try to get t from exchange client directly if it's CCXT based
        if hasattr(exchange, 'client') and hasattr(exchange.client, 'fetch_funding_rate'):
            funding_info = exchange.client.fetch_funding_rate(symbol)
            current_rate = funding_info['fundingRate']
            predicted_rate = funding_info.get('predictedFundingRate', 0)
            logger.info(f"Current Funding Rate: {current_rate:.6f} ({current_rate*100:.4f}%)")
            logger.info(f"Next Predicted Rate: {predicted_rate:.6f}")
            
            # 4. Generate Signal
            signal = strategy.generate_signal(pd.DataFrame(), symbol, funding_rate=current_rate)
            
            action = "NEUTRAL"
            if signal == 1.0: action = "LONG PERP (Negative Arb)"
            elif signal == -1.0: action = "SHORT PERP (Positive Arb)"
            
            logger.info(f"Strategy Signal: {signal} -> {action}")
            
            # Annualized Yield Estimate
            apy = abs(current_rate) * 3 * 365 * 100 # 3 times a day
            logger.info(f"Estimated APY (if rate holds): {apy:.2f}%")
            
        else:
            logger.warning("Exchange does not support fetch_funding_rate directly.")
            
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")

if __name__ == "__main__":
    main()
