import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_proper_triple_barrier(df: pd.DataFrame, tp: float = 0.015, sl: float = 0.01, timeout: int = 12) -> pd.DataFrame:
    """
    Apply Proper Triple Barrier Method without look-ahead bias.
    
    This method properly implements the triple barrier without using future information.
    It checks for take-profit, stop-loss, and timeout conditions using only information
    available at the time of the signal.
    
    Args:
        df: DataFrame with OHLCV data
        tp: Take profit threshold (e.g., 0.015 for 1.5%)
        sl: Stop loss threshold (e.g., 0.01 for 1%)
        timeout: Maximum holding period in bars
    
    Returns:
        DataFrame with 'y' column containing labels:
        - 1: Take profit hit first
        - 0: Stop loss hit first  
        - -1: Timeout reached without hitting either barrier
    """
    df = df.copy()
    df['y'] = -1  # Default: timeout (no clear direction)
    
    # Calculate barriers for each starting point
    for i in range(len(df) - timeout):
        start_price = df.iloc[i]['close']
        tp_price = start_price * (1 + tp)  # Take profit target
        sl_price = start_price * (1 - sl)  # Stop loss target
        
        # Look at the next 'timeout' bars to see which condition is met first
        future_bars = df.iloc[i+1:i+1+timeout][['high', 'low']].values
        
        # Check each future bar to see if barriers are hit
        barrier_hit = False
        for j, (high, low) in enumerate(future_bars):
            if high >= tp_price:  # Take profit hit
                df.iloc[i, df.columns.get_loc('y')] = 1
                barrier_hit = True
                break
            elif low <= sl_price:  # Stop loss hit
                df.iloc[i, df.columns.get_loc('y')] = 0  # Changed to 0 for stop loss (down movement)
                barrier_hit = True
                break
        
        # If no barrier hit within timeout, keep as -1 (timeout/no clear signal)
    
    # Only keep rows where we could determine the outcome
    # Remove the last 'timeout' rows since we can't determine their outcome
    df = df[:-timeout] if timeout > 0 else df
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    logger.info(f"Applied proper triple barrier. Labels distribution: {df['y'].value_counts().to_dict()}")
    return df


def apply_proper_triple_barrier_vectorized(df: pd.DataFrame, tp: float | np.ndarray = 0.015, sl: float | np.ndarray = 0.01, timeout: int = 12) -> pd.DataFrame:
    """
    Vectorized implementation of proper triple barrier method without look-ahead bias.
    Supports both static (float) and dynamic (array) barriers.
    
    Args:
        df: DataFrame with OHLCV data
        tp: Take profit threshold (float or array of same length as df)
        sl: Stop loss threshold (float or array of same length as df)
        timeout: Maximum holding period in bars
    
    Returns:
        DataFrame with 'y' column containing labels
    """
    df = df.copy()
    
    # Initialize labels
    df['y'] = -1  # Default: timeout
    
    # Create arrays for vectorized operations
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    n = len(df)
    
    # Handle dynamic vs static barriers
    if isinstance(tp, (pd.Series, np.ndarray)):
        if len(tp) != n:
            raise ValueError(f"Dynamic TP length {len(tp)} != DataFrame length {n}")
        tp_values = np.array(tp)
    else:
        tp_values = np.full(n, tp)
        
    if isinstance(sl, (pd.Series, np.ndarray)):
        if len(sl) != n:
            raise ValueError(f"Dynamic SL length {len(sl)} != DataFrame length {n}")
        sl_values = np.array(sl)
    else:
        sl_values = np.full(n, sl)
    
    # Calculate barriers for all positions at once
    # Shape: (n, 1) to broadcast against future prices
    tp_levels = (close_prices * (1 + tp_values)).reshape(-1, 1)
    sl_levels = (close_prices * (1 - sl_values)).reshape(-1, 1)
    
    # For each starting position, check the next 'timeout' bars
    for i in range(n - timeout):
        # Get future high and low prices for the next 'timeout' periods
        future_highs = high_prices[i+1:i+1+timeout]
        future_lows = low_prices[i+1:i+1+timeout]
        
        # Check if take profit was hit in any of the next 'timeout' bars
        # current barrier is tp_levels[i, 0]
        tp_hit = np.any(future_highs >= tp_levels[i, 0])
        
        # Check if stop loss was hit in any of the next 'timeout' bars
        # Only check if take profit wasn't hit first
        if tp_hit:
            df.iloc[i, df.columns.get_loc('y')] = 1  # Take profit
        elif np.any(future_lows <= sl_levels[i, 0]):
            df.iloc[i, df.columns.get_loc('y')] = 0  # Stop loss
        # Otherwise, remains -1 (timeout)
    
    # Remove the last 'timeout' rows since we can't determine their outcome
    df = df[:-timeout] if timeout > 0 else df
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    logger.info(f"Applied proper triple barrier (vectorized). Labels distribution: {df['y'].value_counts().to_dict()}")
    return df


def apply_simple_forward_label(df: pd.DataFrame, horizon: int = 24, threshold: float = 0.005) -> pd.DataFrame:
    """
    Simple forward-looking label without using future price directly in features.
    This creates a supervised learning problem without look-ahead bias in the features.
    
    Args:
        df: DataFrame with OHLCV data
        horizon: Number of periods ahead to predict
        threshold: Threshold for positive return
    
    Returns:
        DataFrame with 'y' column containing binary labels
    """
    df = df.copy()
    
    # Calculate future returns (this is the TARGET, not a feature)
    future_close = df['close'].shift(-horizon)
    df['future_return'] = (future_close - df['close']) / df['close']
    
    # Create binary target: 1 if future return > threshold, else 0
    df['y'] = (df['future_return'] > threshold).astype(int)
    
    # Add future prices for backtesting (but these should NOT be used as features)
    df['future_high'] = df['high'].shift(-horizon)
    df['future_low'] = df['low'].shift(-horizon)
    df['future_close'] = future_close
    
    # Only keep rows where we have target values
    df = df.dropna(subset=['y', 'future_return', 'future_high', 'future_low', 'future_close'])
    
    logger.info(f"Applied simple forward label. Labels distribution: {df['y'].value_counts().to_dict()}")
    return df