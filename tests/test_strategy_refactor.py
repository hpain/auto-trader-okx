import pandas as pd
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入我们需要验证的策略
from strategies.moving_average import MovingAverageStrategy
from strategies.lgb_strategy import LGBStrategy

# 导入数据加载和特征工程工具
from utils.load_history import load_history_csv
from features.feature_engineering import generate_features
from utils.data_normalization import normalize_binance_df

def verify_strategies():
    """
    验证已重构的策略是否符合BaseStrategy标准并能正常工作。
    """
    print("--- Verification Started: Testing Refactored Strategies ---")

    # 1. 加载测试数据 (使用本地的一小部分数据即可)
    try:
        print("\n1. Loading test data...")
        data_path = 'data/history/binance_BTCUSDT_1h_1y.csv' 
        df_raw = load_history_csv(data_path)
        if df_raw.empty:
            raise FileNotFoundError("Loaded data is empty.")
        
        # 关键修复：将'ts'列设置为时间序列索引
        if 'ts' in df_raw.columns:
            df_raw = df_raw.set_index('ts')
        df_raw.sort_index(inplace=True)

        df_raw = normalize_binance_df(df_raw)
        print(f"Data loaded and indexed successfully. Shape: {df_raw.shape}")

    except Exception as e:
        print(f"ERROR: Could not load data. {e}")
        print("Please ensure you have some historical data or run data download first.")
        return

    # 2. 生成LGB策略所需的特征
    print("\n2. Generating features for all strategies...")
    featured_data = generate_features(df_raw, news_csv_path=None)
    print(f"Features generated. Shape: {featured_data.shape}")

    # 3. 验证 MovingAverageStrategy
    print("\n--- 3. Verifying MovingAverageStrategy ---")
    try:
        ma_config = {'short_window': 10, 'long_window': 30}
        ma_strategy = MovingAverageStrategy(strategy_name="Test_MA", config=ma_config)
        
        print("Calling generate_signals() on MovingAverageStrategy...")
        ma_signals_df = ma_strategy.generate_signals(featured_data)

        print("Verification Checks:")
        assert isinstance(ma_signals_df, pd.DataFrame), "Return type should be a DataFrame."
        assert 'signal' in ma_signals_df.columns, "'signal' column is missing."
        assert ma_signals_df.index.equals(featured_data.index), "Index does not match input data."
        print("  - PASSED: Correct return type and structure.")

        print("\nSignal distribution:")
        print(ma_signals_df['signal'].value_counts())
        
        print("\nHead of signals:")
        print(ma_signals_df.head())

        print("\nTail of signals:")
        print(ma_signals_df.tail())
        print("--- MovingAverageStrategy Verification PASSED ---")

    except Exception as e:
        print(f"--- MovingAverageStrategy Verification FAILED: {e}", exc_info=True)

    # 4. 验证 LGBStrategy
    print("\n--- 4. Verifying LGBStrategy ---")
    try:
        lgb_config = {
            'model_dir': 'models',
            'model_name': 'best_model.pkl',
            'metadata_name': 'metadata.json'
        }
        lgb_strategy = LGBStrategy(strategy_name="Test_LGB", config=lgb_config)

        # 检查模型是否成功加载
        if lgb_strategy.model is None:
            raise ConnectionError("LGB model was not loaded. Cannot proceed with verification.")

        print("Calling generate_signals() on LGBStrategy...")
        lgb_signals_df = lgb_strategy.generate_signals(featured_data)

        print("Verification Checks:")
        assert isinstance(lgb_signals_df, pd.DataFrame), "Return type should be a DataFrame."
        assert 'signal' in lgb_signals_df.columns, "'signal' column is missing."
        assert lgb_signals_df.index.equals(featured_data.index), "Index does not match input data."
        print("  - PASSED: Correct return type and structure.")

        print("\nSignal distribution:")
        print(lgb_signals_df['signal'].value_counts())

        print("\nHead of signals:")
        print(lgb_signals_df.head())

        print("\nTail of signals:")
        print(lgb_signals_df.tail())
        print("--- LGBStrategy Verification PASSED ---")

    except Exception as e:
        print(f"--- LGBStrategy Verification FAILED: {e}", exc_info=True)

    print("\n--- Verification Finished ---")

if __name__ == "__main__":
    verify_strategies()