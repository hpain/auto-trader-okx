import pandas as pd
import numpy as np

def calculate_funding_factors(df: pd.DataFrame, windows=[8, 16, 24, 72, 168], col_name='funding_rate') -> pd.DataFrame:
    """
    Calculates statistical features for Funding Rates to detect overheating types.
    
    Generates:
    - Z-Score: How extreme is the current rate relative to recent history?
    - MA Deviation: Is rate trending up/down?
    """
    if col_name not in df.columns:
        return pd.DataFrame(index=df.index)
        
    out = pd.DataFrame(index=df.index)
    series = df[col_name]
    
    # 1. Z-Scores (Standardized Deviation)
    for w in windows:
        roll_mean = series.rolling(window=w).mean()
        roll_std = series.rolling(window=w).std().replace(0, 0.000001) # Avoid div by zero
        out[f'funding_z_{w}'] = (series - roll_mean) / roll_std
        
        # 2. Level vs MA (Trend)
        out[f'funding_ma_dev_{w}'] = series - roll_mean

    # 3. Absolute Level check (Is it extraordinarily high/low?)
    # 0.01% is baseline (0.0001). 
    out['funding_high_regime'] = (series > 0.0003).astype(int) # > 0.03% (High)
    out['funding_neg_regime'] = (series < 0).astype(int)        # Negative funding
    
    return out

def calculate_oi_factors(df: pd.DataFrame, windows=[4, 12, 24], col_name='open_interest') -> pd.DataFrame:
    """
    Calculates Open Interest Price Factors to detect Squeezes.
    
    Generates:
    - OI % Change
    - OI / Volume Ratio (if volume avail)
    - Price/OI Divergence Signs
    """
    if col_name not in df.columns:
        return pd.DataFrame(index=df.index)
        
    out = pd.DataFrame(index=df.index)
    oi = df[col_name]
    
    # 1. Momentum (% Change)
    for w in windows:
        out[f'oi_pct_chg_{w}'] = oi.pct_change(periods=w)
    
    # 2. Acceleration (Change of Change) - Detects explosion
    out['oi_accel'] = out[f'oi_pct_chg_{windows[0]}'].diff()
    
    # 3. OI vs Price Correlation (Windowed)
    if 'close' in df.columns:
        for w in windows:
            # Correlation between Price Change and OI Change
            # High +Corr: Trend confirmed by new money
            # High -Corr: Liquidation cascade?
            out[f'oi_price_corr_{w}'] = df['close'].rolling(w).corr(oi)

    return out

def calculate_ls_ratio_factors(df: pd.DataFrame, windows=[24, 72], col_name='long_short_ratio') -> pd.DataFrame:
    """
    Calculates features for Long/Short Ratio (Retail vs Smart Money proxy).
    """
    if col_name not in df.columns:
        return pd.DataFrame(index=df.index)
        
    out = pd.DataFrame(index=df.index)
    ls = df[col_name]
    
    # 1. Deviation from Norm
    for w in windows:
        # Is the crowd excessively Long?
        out[f'ls_z_{w}'] = (ls - ls.rolling(w).mean()) / ls.rolling(w).std().replace(0, 0.001)
        
    # 2. Extreme Crowding check
    # > 2.0 usually means too many retail longs (Bearish signal)
    # < 0.5 usually means too many retail shorts (Bullish squeeze signal)
    out['ls_crowded_long'] = (ls > 2.0).astype(int)
    out['ls_crowded_short'] = (ls < 0.8).astype(int)
    
    return out

def generate_crypto_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to generate all crypto-specific factors.
    Merges them into a single DataFrame aligned with input index.
    """
    features_list = []
    
    # 1. Funding
    # Map common names
    f_col = 'funding_rate' if 'funding_rate' in df.columns else 'fundingRate'
    if f_col in df.columns:
        features_list.append(calculate_funding_factors(df, col_name=f_col))
        
    # 2. Open Interest
    oi_col = 'open_interest' if 'open_interest' in df.columns else 'openInterest'
    if oi_col in df.columns:
        features_list.append(calculate_oi_factors(df, col_name=oi_col))
        
    # 3. Long/Short Ratio
    ls_col = 'long_short_ratio' if 'long_short_ratio' in df.columns else 'longShortRatio'
    if ls_col in df.columns:
        features_list.append(calculate_ls_ratio_factors(df, col_name=ls_col))
        
    # 4. Global Ratio (if avail)
    gls_col = 'global_long_short_ratio'
    if gls_col in df.columns:
        features_list.append(calculate_ls_ratio_factors(df, col_name=gls_col).add_prefix('global_'))
        
    # Combine
    if features_list:
        combined = pd.concat(features_list, axis=1)
        # Ensure alignment
        combined = combined.reindex(df.index)
        return combined
    else:
        return pd.DataFrame(index=df.index)
