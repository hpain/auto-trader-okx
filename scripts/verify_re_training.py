
import pandas as pd
import numpy as np
import os
import sys
import json
import asyncio

# Add project root to path
sys.path.append(os.getcwd())

from exchange.factory import ExchangeFactory
from features.feature_engineering import generate_features, apply_triple_barrier

async def verify():
    print("Initializing exchange...")
    exchange = await ExchangeFactory.create_exchange('aggregated', sandbox=False)
    
    symbol = 'BTC-USDT'
    interval = '1h'
    years = 4.0 # Emulating the long training run
    
    print(f"Fetching historical data for {symbol} ({years} years)...")
    df = await exchange.fetch_historical_data(symbol, interval, years)
    
    # Check funding rates
    since_ms = int((pd.Timestamp.now() - pd.Timedelta(days=years*365.25)).timestamp() * 1000)
    funding_df = await exchange.fetch_funding_rates(symbol, interval, since=since_ms)
    
    # Feature Engineering
    print("Generating features...")
    derivatives_dfs = {'funding_rates': funding_df} if not funding_df.empty else {}
    
    data = generate_features(df, derivatives_dfs=derivatives_dfs)
    
    print(f"Columns available in final data: {len(data.columns)}")
    
    enhanced_cols = [c for c in data.columns if 'enhanced' in c]
    print(f"Enhanced (mined) features successfully added: {len(enhanced_cols)}")
    if enhanced_cols:
         print(f"Sample mined features: {enhanced_cols[:3]}")

    # Apply labels
    tp, sl, timeout = 0.015, 0.01, 12
    data_labeled = apply_triple_barrier(data, tp=tp, sl=sl, timeout=timeout)
    
    label_counts = data_labeled['y'].value_counts()
    print("\nLabel Distribution:")
    print(label_counts)

    await exchange.close()

if __name__ == "__main__":
    asyncio.run(verify())
