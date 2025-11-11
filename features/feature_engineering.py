# features/feature_engineering.py
import pandas as pd
import os
import json
import numpy as np
import ta

    # Import news processing functions
from data.news import load_news_from_csv
from features.sentiment_analysis import process_news_sentiment
# Import news processing functions
from data.news import load_news_from_csv, aggregate_daily_sentiment
# Import the new Bayesian detector
from analysis.bayesian_regime_detector import BayesianRegimeDetector
# Import feature generation parameters from config
from config.model_config import FEATURE_PARAMS
# Import sentiment analysis utilities
from features.sentiment_utils import merge_price_and_sentiment

def check_data_length(df: pd.DataFrame, min_length: int = 200) -> bool:
    """
    检查输入数据是否满足最小长度要求
    """
    return len(df) >= min_length

def handle_insufficient_data(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    处理数据长度不足的情况，返回一个填充了NaN的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for feature in feature_names:
        result[feature] = np.nan
    return result
def _add_bayesian_regime_features(df: pd.DataFrame, ma_period: int = 20, atr_period: int = 14) -> pd.DataFrame:
    """
    Adds Bayesian regime probabilities as features to the DataFrame.
    This is a helper function for generate_features.
    """
    work_df = df.copy()

    # 1. Calculate indicators needed for evidence
    temp_indicators_df = pd.DataFrame(index=work_df.index)

    # MA Slope - Check if SMA is already in work_df (from programmatic features)
    ma_col = f'sma_{ma_period}'
    if ma_col not in work_df.columns:
        temp_indicators_df[ma_col] = ta.trend.SMAIndicator(work_df["close"], window=ma_period).sma_indicator()
    else:
        temp_indicators_df[ma_col] = work_df[ma_col] # Use existing if present
    
    ma_slope_col = f'ma_slope_{ma_period}'
    temp_indicators_df[ma_slope_col] = temp_indicators_df[ma_col].diff()

    # Normalized ATR - Check if ATR is already in work_df (from programmatic features)
    atr_col = f'atr_{atr_period}'
    if atr_col not in work_df.columns:
        temp_indicators_df[atr_col] = ta.volatility.AverageTrueRange(
            high=work_df["high"], low=work_df["low"], close=work_df["close"], window=atr_period
        ).average_true_range()
    else:
        temp_indicators_df[atr_col] = work_df[atr_col] # Use existing if present
    
    atr_norm_col = f'atr_norm_{atr_period}'
    temp_indicators_df[atr_norm_col] = temp_indicators_df[atr_col] / work_df['close']
    
    # --- 核心修复：在合并前处理重复索引，防止数据错乱 ---
    work_df = work_df[~work_df.index.duplicated(keep='first')]
    temp_indicators_df = temp_indicators_df[~temp_indicators_df.index.duplicated(keep='first')]

    # Temporarily combine base data with calculated indicators for sequential processing
    calc_df = pd.concat([work_df, temp_indicators_df], axis=1)
    
    # 2. Initialize detector and apply it row-by-row
    detector = BayesianRegimeDetector()
    posteriors_history = []

    # Use .iterrows() for sequential state updates
    # Ensure the required columns exist and are not NaN for the iteration
    temp_cols_for_iteration = [ma_slope_col, atr_norm_col]
    # Drop rows where these critical columns are NaN, as the detector cannot update without them
    calc_df_for_iter = calc_df.dropna(subset=temp_cols_for_iteration)

    # --- 关键修复：处理数据量不足的情况 ---
    if calc_df_for_iter.empty:
        regime_df = pd.DataFrame(index=df.index)
        regime_df['prob_bull'] = 0.0
        regime_df['prob_bear'] = 0.0
        regime_df['prob_ranging'] = 0.0
        return regime_df

    for index, row in calc_df_for_iter.iterrows():
        ma_slope = row[ma_slope_col]
        atr_normalized = row[atr_norm_col]
        
        # The detector's state (posteriors) is updated sequentially
        updated_posteriors = detector.update(ma_slope, atr_normalized)
        posteriors_history.append(updated_posteriors)
    
    # 3. Create DataFrame for regime probabilities
    regime_df = pd.DataFrame(index=df.index) # Ensure index matches original df
    if posteriors_history:
        # Create a DataFrame from posteriors_history with the index of the iterated data
        temp_regime_df = pd.DataFrame(posteriors_history, index=calc_df_for_iter.index)
        temp_regime_df.columns = ['prob_bull', 'prob_bear', 'prob_ranging']
        # Reindex to match the original df, filling NaNs for rows that were dropped
        regime_df = regime_df.merge(temp_regime_df, left_index=True, right_index=True, how='left')
        
    # Fill any NaNs (e.g., from initial rows where indicators were NaN) with 0.0
    for col in ['prob_bull', 'prob_bear', 'prob_ranging']:
        if col in regime_df.columns:
            regime_df[col] = regime_df[col].fillna(0.0)
        else: # If no posteriors were generated at all
            regime_df[col] = 0.0

    return regime_df


def _generate_programmatic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a wide array of technical analysis features programmatically.
    Includes proper handling of insufficient data.
    """
    # Start with a copy of the original DataFrame for calculations
    calc_df = df.copy()

    # 处理边界情况
    # --- 核心修复：如果输入数据为空，则提前返回一个空的DataFrame ---
    if calc_df.empty:
        return pd.DataFrame(index=calc_df.index)
    
    # 检查数据长度是否足够计算指标
    min_required_length = max(FEATURE_PARAMS.get("other_indicator_windows", [14])) * 2
    if len(calc_df) < min_required_length:
        # 如果数据长度不足，返回一个包含基础特征的DataFrame
        result = pd.DataFrame(index=calc_df.index)
        result["return"] = calc_df["close"].pct_change()
        return result


    # Define parameter ranges
    lag_windows = FEATURE_PARAMS.get("lag_windows", [2, 3, 5, 10, 15, 20])
    ma_windows = FEATURE_PARAMS.get("ma_windows", [10, 20, 30, 50, 100, 200])
    rsi_windows = FEATURE_PARAMS.get("rsi_windows", [7, 14, 21, 30])
    volatility_windows = FEATURE_PARAMS.get("volatility_windows", [10, 20, 30, 60, 120])
    other_indicator_windows = FEATURE_PARAMS.get("other_indicator_windows", [14, 20, 30])

    # List to hold all newly generated feature DataFrames
    new_features_dfs = []

    # --- Base & Lag Features ---
    temp_df_base_lag = pd.DataFrame(index=calc_df.index)
    temp_df_base_lag["return"] = calc_df["close"].pct_change()
    for lag in lag_windows:
        temp_df_base_lag[f'return_lag_{lag}'] = temp_df_base_lag['return'].shift(lag)
        temp_df_base_lag[f'vol_lag_{lag}'] = calc_df['vol'].shift(lag)
    new_features_dfs.append(temp_df_base_lag)

    # --- Moving Averages & Spreads ---
    temp_df_ma = pd.DataFrame(index=calc_df.index)
    for window in ma_windows:
        temp_df_ma[f'sma_{window}'] = ta.trend.SMAIndicator(calc_df["close"], window=window).sma_indicator()
        temp_df_ma[f'ema_{window}'] = ta.trend.EMAIndicator(calc_df["close"], window=window).ema_indicator()
    
    # Generate spreads between different speed MAs
    for fast in [10, 20, 50]:
        for slow in [50, 100, 200]:
            if slow > fast:
                if f'sma_{fast}' in temp_df_ma.columns and f'sma_{slow}' in temp_df_ma.columns:
                    temp_df_ma[f'sma_spread_{fast}_{slow}'] = (temp_df_ma[f'sma_{fast}'] - temp_df_ma[f'sma_{slow}']) / calc_df['close']
                if f'ema_{fast}' in temp_df_ma.columns and f'ema_{slow}' in temp_df_ma.columns:
                    temp_df_ma[f'ema_spread_{fast}_{slow}'] = (temp_df_ma[f'ema_{fast}'] - temp_df_ma[f'ema_{slow}']) / calc_df['close']
    new_features_dfs.append(temp_df_ma)

    # --- Volatility Indicators ---
    temp_df_volatility = pd.DataFrame(index=calc_df.index)
    for window in volatility_windows:
        temp_df_volatility[f'volatility_{window}'] = temp_df_base_lag['return'].rolling(window).std()
    
    for window in other_indicator_windows:
        # --- 核心修复：增加数据长度检查，防止 ta 库在数据量过少时崩溃 ---
        if len(calc_df) >= window:
            temp_df_volatility[f'atr_{window}'] = ta.volatility.AverageTrueRange(
                high=calc_df["high"], low=calc_df["low"], close=calc_df["close"], window=window
            ).average_true_range()
            bollinger = ta.volatility.BollingerBands(close=calc_df["close"], window=window, window_dev=2)
            temp_df_volatility[f'bb_mavg_{window}'] = bollinger.bollinger_mavg()
            temp_df_volatility[f'bb_hband_{window}'] = bollinger.bollinger_hband()
            temp_df_volatility[f'bb_lband_{window}'] = bollinger.bollinger_lband()
            temp_df_volatility[f'bb_width_{window}'] = (temp_df_volatility[f'bb_hband_{window}'] - temp_df_volatility[f'bb_lband_{window}']) / temp_df_volatility[f'bb_mavg_{window}']
    new_features_dfs.append(temp_df_volatility)

    # --- Momentum Indicators ---
    temp_df_momentum = pd.DataFrame(index=calc_df.index)
    for window in rsi_windows:
        temp_df_momentum[f'rsi_{window}'] = ta.momentum.RSIIndicator(calc_df["close"], window=window).rsi()
    
    for window in other_indicator_windows:
        # --- 核心修复：增加数据长度检查 ---
        if len(calc_df) >= window:
            temp_df_momentum[f'roc_{window}'] = ta.momentum.ROCIndicator(calc_df["close"], window=window).roc()
            temp_df_momentum[f'williams_r_{window}'] = ta.momentum.WilliamsRIndicator(
                high=calc_df["high"], low=calc_df["low"], close=calc_df["close"], lbp=window
            ).williams_r()
            stoch = ta.momentum.StochasticOscillator(
                high=calc_df["high"], low=calc_df["low"], close=calc_df["close"], window=window, smooth_window=3
            )
            temp_df_momentum[f'stoch_k_{window}'] = stoch.stoch()
            temp_df_momentum[f'stoch_d_{window}'] = stoch.stoch_signal()
    new_features_dfs.append(temp_df_momentum)

    # --- Trend Indicators ---
    temp_df_trend = pd.DataFrame(index=calc_df.index)
    for window in other_indicator_windows:
        # --- 核心修复：增加数据长度检查，防止 ta 库在数据量过少时崩溃 ---
        if len(calc_df) >= window:
            temp_df_trend[f'cci_{window}'] = ta.trend.CCIIndicator(
                high=calc_df["high"], low=calc_df["low"], close=calc_df["close"], window=window
            ).cci()
            adx_indicator = ta.trend.ADXIndicator(
                high=calc_df["high"], low=calc_df["low"], close=calc_df["close"], window=window
            )
            temp_df_trend[f'adx_{window}'] = adx_indicator.adx()
            temp_df_trend[f'adx_pos_{window}'] = adx_indicator.adx_pos()
            temp_df_trend[f'adx_neg_{window}'] = adx_indicator.adx_neg()
            vortex_indicator = ta.trend.VortexIndicator(
                high=calc_df["high"], low=calc_df["low"], close=calc_df["close"], window=window
            )
            temp_df_trend[f'vi_pos_{window}'] = vortex_indicator.vortex_indicator_pos()
            temp_df_trend[f'vi_neg_{window}'] = vortex_indicator.vortex_indicator_neg()
    new_features_dfs.append(temp_df_trend)

    # --- Volume Indicators ---
    temp_df_volume = pd.DataFrame(index=calc_df.index)
    temp_df_volume["obv"] = ta.volume.OnBalanceVolumeIndicator(close=calc_df["close"], volume=calc_df["vol"]).on_balance_volume()
    for window in other_indicator_windows:
        # --- 核心修复：增加数据长度检查 ---
        if len(calc_df) >= window:
            temp_df_volume[f'mfi_{window}'] = ta.volume.MFIIndicator(
                high=calc_df["high"], low=calc_df["low"], close=calc_df["close"], volume=calc_df["vol"], window=window
            ).money_flow_index()
            temp_df_volume[f'cmf_{window}'] = ta.volume.ChaikinMoneyFlowIndicator(
                high=calc_df["high"], low=calc_df["low"], close=calc_df["close"], volume=calc_df["vol"], window=window
            ).chaikin_money_flow()
    new_features_dfs.append(temp_df_volume)

    # Concatenate all generated features into a single DataFrame
    # Filter out empty DataFrames from new_features_dfs list
    new_features_dfs = [f for f in new_features_dfs if not f.empty]
    
    # Combine original df with all new programmatic features
    combined_programmatic_df = pd.concat([calc_df] + new_features_dfs, axis=1)

    # --- Interaction Features (calculated on the combined DataFrame) ---
    temp_df_interaction = pd.DataFrame(index=combined_programmatic_df.index)
    for adx_w in other_indicator_windows:
        for rsi_w in rsi_windows:
            if f'adx_{adx_w}' in combined_programmatic_df.columns and f'rsi_{rsi_w}' in combined_programmatic_df.columns:
                temp_df_interaction[f'adx_{adx_w}_x_rsi_{rsi_w}'] = combined_programmatic_df[f'adx_{adx_w}'] * combined_programmatic_df[f'rsi_{rsi_w}']
    
    if not temp_df_interaction.empty:
        combined_programmatic_df = pd.concat([combined_programmatic_df, temp_df_interaction], axis=1)

    # --- 核心修复: 只返回新生成的特征列 ---
    # 移除原始的 OHLCV 列，因为它们已经存在于主 DataFrame 中
    original_cols = df.columns.tolist()
    new_feature_cols = [col for col in combined_programmatic_df.columns if col not in original_cols]
    
    return combined_programmatic_df[new_feature_cols]

def _add_onchain_features(df: pd.DataFrame, onchain_dfs: dict) -> pd.DataFrame:
    """
    Merges on-chain data into the main DataFrame.
    Assumes on-chain data is daily and needs to be forward-filled.
    """
    if not onchain_dfs:
        return pd.DataFrame(index=df.index)

    all_features = pd.DataFrame(index=df.index)

    for name, onchain_df in onchain_dfs.items():
        if onchain_df.empty:
            continue
        
        temp_df = onchain_df.copy()
        
        time_col = None
        if 'time' in temp_df.columns:
            time_col = 'time'
        elif 'day' in temp_df.columns:
            time_col = 'day'
        
        if time_col:
            temp_df.index = pd.to_datetime(temp_df[time_col])
        else:
            temp_df.index = pd.to_datetime(temp_df.index)

        if df.index.tz is not None:
            if temp_df.index.tz is None:
                temp_df.index = temp_df.index.tz_localize('UTC').tz_convert(df.index.tz)
            else:
                temp_df.index = temp_df.index.tz_convert(df.index.tz)
        
        temp_df = temp_df.add_prefix(f"onchain_{name}_")
        all_features = all_features.merge(temp_df, how='left', left_index=True, right_index=True)

    # Forward-fill the daily data and infer dtypes to avoid FutureWarning
    all_features = all_features.ffill().infer_objects(copy=False)

    # Drop columns that are still entirely NaN after ffill
    all_features.dropna(axis=1, how='all', inplace=True)

    # Address the FutureWarning by inferring object types
    if not all_features.empty:
        all_features = all_features.infer_objects(copy=False)
    
    new_cols = [col for col in all_features.columns if col not in df.columns]
    return all_features[new_cols]

def _add_derivatives_features(df: pd.DataFrame, derivatives_dfs: dict) -> pd.DataFrame:
    """Placeholder for adding derivatives features."""
    if not derivatives_dfs:
        return pd.DataFrame(index=df.index)
    # TODO: Implement actual feature engineering for derivatives data
    return pd.DataFrame(index=df.index)

def _add_multi_symbol_features(df: pd.DataFrame, feature_dfs: dict) -> pd.DataFrame:
    """Placeholder for adding multi-symbol features."""
    if not feature_dfs:
        return pd.DataFrame(index=df.index)
    # TODO: Implement actual feature engineering for multi-symbol data
    return pd.DataFrame(index=df.index)


def generate_features(df: pd.DataFrame, news_csv_path: str = None, mined_features_path: str = None, feature_dfs: dict = None, derivatives_dfs: dict = None, onchain_dfs: dict = None) -> pd.DataFrame:
    """
    Generates a streamlined, high-importance set of features, including multi-period
    indicators, lag features, and additional time-based features.
    Generates a wide and extensible set of features for the model.
    This function now orchestrates feature generation by calling helper functions.
    """
    out = df.copy()
    
    # Check for required columns
    for col in ["open", "high", "low", "close", "vol"]:
        if col not in out.columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain column: {col}")
    
    # 处理空数据的情况
    if df.empty:
        return handle_insufficient_data(df, [
            'return', 'atr_14', 'rsi_14', 'volatility_10', 'sma_10', 'ema_50',
            'adx_14', 'sent_mean', 'sent_median', 'count'
        ])

    # 检查数据长度是否足够
    min_required_length = 30  # 根据技术指标的计算需要设定最小长度
    if not check_data_length(df, min_required_length):
        # 如果数据长度不足，返回填充了NaN的DataFrame
        # 但是仍然尝试加载任何提供的 mined_features（如果有），以符合测试对短数据的期望
        result = handle_insufficient_data(out, [
            'return', 'atr_14', 'rsi_14', 'volatility_10', 'sma_10', 'ema_50',
            'adx_14', 'sent_mean', 'sent_median', 'count'
        ])

        # This section for mined_features_path seems to be from another version, let's ignore it for now.
        if mined_features_path and os.path.exists(mined_features_path):
            try:
                with open(mined_features_path, 'r') as f:
                    mined_data = json.load(f)
                formula_str = mined_data.get("formula")
                base_features = mined_data.get("base_features", [])
                new_feature_name = mined_data.get("name")

                # Only proceed if base features exist in the original df
                if new_feature_name and all(feature in df.columns for feature in base_features):
                    safe_dict = {
                        'add': np.add,
                        'sub': np.subtract,
                        'mul': np.multiply,
                        'div': lambda a, b: np.divide(a, np.where(b == 0, 1e-9, b)),
                        'sqrt': lambda a: np.sqrt(np.abs(a)),
                        'log': lambda a: np.log(np.abs(a) + 1e-9),
                        'neg': np.negative,
                        'inv': lambda a: 1 / np.where(a == 0, 1e-9, a),
                        'sin': np.sin,
                        'cos': np.cos,
                        'tan': np.tan,
                        'max': np.maximum,
                        'min': np.minimum,
                        'abs': np.abs
                    }
                    for i, feature_name in enumerate(base_features):
                        safe_dict[f'X{i}'] = df[feature_name].values.astype(float)
                    try:
                        result_vals = eval(formula_str, {"__builtins__": None}, safe_dict)
                        # Ensure series alignment and preserve integer dtype when possible
                        series_result = pd.Series(result_vals, index=df.index)
                        if base_features and all(np.issubdtype(df[f].dtype, np.integer) for f in base_features):
                            series_result = series_result.astype(df[base_features[0]].dtype)
                        result[new_feature_name] = series_result
                    except Exception as e:
                        print(f"Error evaluating mined feature formula in short-data path: {e}")
            except Exception as e:
                print(f"Error loading mined features in short-data path: {e}")

        return result

    # --- Programmatic Technical Indicators ---
    programmatic_features_df = _generate_programmatic_features(df)
    
    # --- Time-based Features ---
    time_features_df = pd.DataFrame(index=df.index)
    time_features_df['hour_of_day'] = df.index.hour
    time_features_df['day_of_week'] = df.index.dayofweek

    # --- News Sentiment Features --- (Refactored for clarity)
    sentiment_features_df = pd.DataFrame(index=df.index)
    try:
        news_df = load_news_from_csv(news_csv_path) if news_csv_path and os.path.exists(news_csv_path) else pd.DataFrame()
        if not news_df.empty:
            daily_sent_df = aggregate_daily_sentiment(news_df)
            merged_df = merge_price_and_sentiment(df, daily_sent_df)
            # Ensure the columns exist before trying to slice
            cols_to_extract = [c for c in ['sent_mean', 'sent_median', 'count'] if c in merged_df.columns]
            if cols_to_extract:
                sentiment_features_df = merged_df[cols_to_extract].copy()
    except Exception as e:
        logging.warning(f"Could not process sentiment features: {e}")

    # Robustly create and fill sentiment columns
    for col in ['sent_mean', 'sent_median', 'count']:
        if col not in sentiment_features_df:
            sentiment_features_df[col] = 0.0
        else:
            # If column exists but has NaNs, fill them using the recommended assignment method
            sentiment_features_df[col] = sentiment_features_df[col].fillna(0.0)
    
    # Special case for 'count' which should be integer
    if 'count' in sentiment_features_df.columns:
        sentiment_features_df['count'] = sentiment_features_df['count'].astype(int)

    # --- Bayesian Regime Features ---
    # This must be called after the necessary indicators (SMA, ATR) are calculated.
    # Combine original data with new programmatic features to ensure indicators like SMA are available
    # for the Bayesian calculation.
    # --- 核心修复: 在拼接前处理重复列 ---
    cols_to_use = programmatic_features_df.columns.difference(df.columns)
    temp_combined_df_for_bayesian = pd.concat([df, programmatic_features_df[cols_to_use]], axis=1)
    bayesian_features_df = _add_bayesian_regime_features(temp_combined_df_for_bayesian)

    # --- On-Chain, Derivatives, and Multi-Symbol Features ---
    onchain_features_df = _add_onchain_features(df, onchain_dfs)
    derivatives_features_df = _add_derivatives_features(df, derivatives_dfs)
    multi_symbol_features_df = _add_multi_symbol_features(df, feature_dfs)

    # --- Mined Features (from Genetic Programming) ---
    mined_features_df = pd.DataFrame(index=df.index)
    if mined_features_path and os.path.exists(mined_features_path):
        try:
            with open(mined_features_path, 'r') as f:
                mined_data = json.load(f)
            
            formula_str = mined_data.get("formula")
            base_features = mined_data.get("base_features", [])
            new_feature_name = mined_data.get("name")

            if not all([formula_str, base_features, new_feature_name]):
                print(f"Warning: Mined features file is incomplete: {mined_features_path}")
            else:
                print(f"Processing mined feature: {new_feature_name}")
                # 确保基础特征存在
                if all(feature in df.columns for feature in base_features):
                    # Create a safe evaluation environment for the formula
                    safe_dict = {
                        'add': np.add,
                        'sub': np.subtract,
                        'mul': np.multiply,
                        'div': lambda a, b: np.divide(a, np.where(b == 0, 1e-9, b)),
                        'sqrt': lambda a: np.sqrt(np.abs(a)),
                        'log': lambda a: np.log(np.abs(a) + 1e-9),
                        'neg': np.negative,
                        'inv': lambda a: 1 / np.where(a == 0, 1e-9, a),
                        'sin': np.sin,
                        'cos': np.cos,
                        'tan': np.tan,
                        'max': np.maximum,
                        'min': np.minimum,
                        'abs': np.abs
                    }
                    
                    # Map X0, X1, etc. to the actual DataFrame columns
                    for i, feature_name in enumerate(base_features):
                        safe_dict[f'X{i}'] = df[feature_name].values.astype(float)

                    # 使用numpy函数进行安全的特征计算
                    try:
                        result = eval(formula_str, {"__builtins__": None}, safe_dict)
                        # Align to index and preserve integer dtype when appropriate
                        series_result = pd.Series(result, index=df.index)
                        if base_features and all(np.issubdtype(df[f].dtype, np.integer) for f in base_features):
                            series_result = series_result.astype(df[base_features[0]].dtype)
                        mined_features_df[new_feature_name] = series_result  # 添加挖掘特征到DataFrame
                        print(f"Successfully added mined feature: {new_feature_name}")
                    except Exception as e:
                        print(f"Error evaluating formula {formula_str}: {e}")
                else:
                    missing_features = [f for f in base_features if f not in df.columns]
                    print(f"Warning: Missing required base features: {missing_features}")
        except Exception as e:
            print(f"Error processing mined features file: {e}")
            
    # --- Combine All Features ---
    all_feature_dfs = [
        df, # Start with the original dataframe
        programmatic_features_df,
        time_features_df,
        sentiment_features_df,
        bayesian_features_df,
        onchain_features_df,
        derivatives_features_df,
        multi_symbol_features_df,
        mined_features_df
    ]

    # Filter out any None or empty DataFrames
    valid_feature_dfs = [f for f in all_feature_dfs if f is not None and not f.empty]
    
    # Concatenate all valid feature dataframes
    final_df = pd.concat(valid_feature_dfs, axis=1)

    # Remove duplicated columns, keeping the first occurrence
    final_df = final_df.loc[:, ~final_df.columns.duplicated(keep='first')]

    return final_df



def make_supervised(
    df: pd.DataFrame, horizon: int = 1, threshold: float = 0.005
) -> pd.DataFrame:
    """
    构建监督学习数据集（分类模式）。

    此函数会创建一个新的 DataFrame，其中包含：
    1. 原始特征。
    2. 目标变量 'y'：如果未来 `horizon` 周期的回报率超过 `threshold`，则为1，否则为0。
    3. 未来价格 'future_*' 和 'future_ret'：可用于后续的回测分析（如止盈止损），但不应作为模型的训练特征，以防数据泄露。

    Args:
        df: 包含价格数据的 DataFrame。
        horizon: 预测未来的时间周期数。
        threshold: 定义正向标签（1）的最低回报率阈值。

    Returns:
        一个新的 DataFrame，包含特征、目标变量 'y' 和未来价格信息。
    """
    # 创建副本以避免修改原始 DataFrame
    out_df = df.copy()

    # 计算未来N个周期的价格，用于生成目标和回测分析
    # 警告：这些 'future_*' 列不应作为模型训练的特征！
    out_df["future_high"] = out_df["high"].shift(-horizon)
    out_df["future_low"] = out_df["low"].shift(-horizon)
    out_df["future_close"] = out_df["close"].shift(-horizon)

    # 计算未来回报率
    future_ret = (out_df["future_close"] / out_df["close"]) - 1
    out_df["future_ret"] = future_ret
    
    # 目标变量：分类标签
    out_df["y"] = (future_ret > threshold).astype(int)
    
    # 移除因 shift 操作在末尾产生的 NaN 值，这些行无法用于训练或回测
    out_df = out_df.dropna(subset=["future_high", "future_low", "future_close", "future_ret", "y"])
    
    return out_df


def merge_price_and_sentiment(price_df: pd.DataFrame, daily_sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    将日级情绪聚合对齐到K线：
    - 取K线时间的date作为merge key（处理时区问题）
    - 用前向填充方式把日情绪扩展到当日所有bar
    """
    p = price_df.copy()

    # Build a date key from price index (preserve timezone-awareness if present)
    if p.index.tz is not None:
        # Convert price times to UTC and floor to day (tz-aware)
        p_dates = p.index.tz_convert('UTC').floor('D')
    else:
        p_dates = p.index.floor('D')
    p = p.copy()
    p['date'] = p_dates

    # Prepare sentiment index to be timezone-aware in UTC for matching
    sent = daily_sent_df.copy()
    if sent.index.tz is None:
        # If naive, assume UTC and localize
        try:
            sent.index = sent.index.tz_localize('UTC')
        except Exception:
            # Fallback: coerce index to datetime then localize
            sent.index = pd.to_datetime(sent.index).tz_localize('UTC')
    else:
        sent.index = sent.index.tz_convert('UTC')

    # Now merge on the date key (left_on is tz-aware if p had tz)
    m = p.merge(sent, how='left', left_on='date', right_index=True)

    # Forward-fill sentiment within the time series so each bar inherits the latest daily value
    for col in ['sent_mean', 'sent_median', 'count']:
        if col in m.columns:
            m[col] = m[col].ffill()
        else:
            # Ensure columns exist even if sentiment df was empty
            m[col] = 0.0 if col != 'count' else 0

    # Fill any remaining NaNs (e.g., at the start of the series) with 0
    m.fillna({'sent_mean': 0.0, 'sent_median': 0.0, 'count': 0}, inplace=True)

    # Drop helper date column
    m.drop(columns=['date'], inplace=True)

    return m
