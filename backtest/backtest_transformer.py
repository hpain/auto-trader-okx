import sys
import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engineering import generate_features
from strategies.transformer_strategy import TransformerStrategy
from trader.portfolio_manager import get_portfolio_manager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest():
    logger.info("--- Starting Transformer + Risk Engine Backtest (CPU) ---")
    
    # 1. Load Data
    data_path = 'data/history/binance_BTCUSDT_1h_1y.csv'
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Rename columns to standard format
    df.rename(columns={
        'ts': 'timestamp', 
        'o': 'open', 
        'h': 'high', 
        'l': 'low', 
        'c': 'close', 
        'v': 'volume'
    }, inplace=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    logger.info(f"Loaded {len(df)} candles.")
    
    # 2. Feature Engineering
    logger.info("Generating Features...")
    df = generate_features(df)
    df.dropna(inplace=True)
    
    # Create Targets for training
    df['target_up'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 3. Split Train/Test
    # Simple Time Split: 80% Train, 20% Test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    logger.info(f"Train Size: {len(train_df)}, Test Size: {len(test_df)}")
    
    # 4. Initialize Components
    # Strategy
    strategy = TransformerStrategy(window_size=60)
    strategy.build_model(input_dim=len(strategy.features)) # Auto-detects input dim from default features
    
    # Portfolio Manager (Ensure Risk Engine is ON)
    # We cheat a bit by modifying global config directly or mocking
    config = {
        'multi_asset': {
            'symbols': ['BTC-USDT'],
            'risk_per_trade': 0.02,
            'max_portfolio_risk': 0.10
        },
        'risk_management': {
            'use_risk_engine': True,
            'target_volatility': 0.20,
            'volatility_window': 30
        }
    }
    pm = get_portfolio_manager(config)
    pm.capital = 10000.0 # Start with 10k
    pm.risk_engine_enabled = True # Force enable if init logic missed it
    # Re-init risk engine to be sure
    from trader.risk_engine import RiskEngine
    pm.risk_engine = RiskEngine(target_volatility=0.20)
    
    # 5. Train Model
    logger.info("Training Model on CPU (2 Epochs)...")
    strategy.train_model(train_df, epochs=2, batch_size=32)
    
    # 6. Simulation Loop
    logger.info("Simulating Trading Loop...")
    
    capital_curve = [pm.capital]
    positions = [0.0]
    dates = []
    
    # We walk forward one step at a time
    # This is slow but simulates reality. optimize for verifying logic.
    # To speed up CPU test, we step every 1 hour but only predict/trade 
    
    current_cash = pm.capital
    current_position = 0.0
    
    # For quick verification, just iterate
    for i in range(len(test_df)):
        try:
            current_bar = test_df.iloc[i]
            # Need a window of history ending at i
            # Context df must be length window_size
            if i < strategy.window_size: 
                dates.append(current_bar.name)
                capital_curve.append(current_cash + current_position * current_bar['close'])
                continue
                
            # Get window for inference
            # We need the last 60 bars from the FULL dataframe (including test history so far)
            # The test_df is a slice, so we need to validly index backward
            # Easiest way: use rolling window on test_df if i is large enough
            window_df = test_df.iloc[i-strategy.window_size : i] # Correct window: [t-60 : t]
            
            # 1. Signal
            signal = strategy.generate_signal(window_df)
            
            # 2. Risk Engine Scalar
            # Need recent history for volatility
            vol_scalar = 1.0
            if pm.risk_engine:
                # Use last 30 bars of closes
                history_closes = window_df['close'].tail(31) # Need 31 points to get 30 returns
                history_returns = history_closes.pct_change().dropna()
                vol_scalar = pm.risk_engine.calculate_volatility_scalar(history_returns)
            
            # 3. Size
            # Mock portfolio manager logic manually for clear logging
            price = current_bar['close']
            
            # Check Exit
            if signal == -1 and current_position > 0:
                # Sell
                revenue = current_position * price
                current_cash += revenue
                current_position = 0.0
                logger.debug(f"[{current_bar.name}] SELL @ {price:.2f}")
                
            # Check Entry
            elif signal == 1 and current_position == 0:
                # Buy
                risk_amt = current_cash * 0.02 * vol_scalar # 2% base risk * scalar
                quantity = risk_amt / price
                cost = quantity * price
                
                if cost < current_cash:
                    current_cash -= cost
                    current_position = quantity
                    logger.debug(f"[{current_bar.name}] BUY  @ {price:.2f} (Vol Scalar: {vol_scalar:.2f})")
            
            # Update Equity
            equity = current_cash + (current_position * price)
            capital_curve.append(equity)
            dates.append(current_bar.name)
        except Exception as e:
            logger.error(f"Error at step {i}: {e}")
            break
            
    # 7. Results
    final_equity = capital_curve[-1]
    ret = (final_equity - 10000) / 10000 * 100
    
    logger.info("--- Backtest Complete ---")
    logger.info(f"Final Equity: ${final_equity:.2f}")
    logger.info(f"Return: {ret:.2f}%")
    
    # Plot
    # plt.plot(dates, capital_curve[1:]) # Skip initial
    # plt.title(f"Transformer + Risk Engine (Return: {ret:.2f}%)")
    # plt.savefig("backtest_transformer_result.png")
    # logger.info("Saved plot to backtest_transformer_result.png")

if __name__ == "__main__":
    run_backtest()
