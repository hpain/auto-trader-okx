import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from models.backtest import simple_backtest
import pprint
from exchange.factory import ExchangeFactory
from utils.logger import logger
import os

# --- Configuration ---
YEARS_OF_DATA = 4
SYMBOL = 'BTC/USDT' # Use '/' for CCXT compatibility
TIMEFRAME = '1h'
CACHE_PATH = f'data/history/aggregated_{SYMBOL.replace("/", "")}_{TIMEFRAME}_{YEARS_OF_DATA}y.csv'

def run_model_backtest():
    # --- 1. 加载数据和特征工程 ---
    print("加载数据和计算特征...")
    
    # Check for cached data first
    if os.path.exists(CACHE_PATH):
        logger.info(f"从缓存加载历史数据: {CACHE_PATH}")
        data = pd.read_csv(CACHE_PATH)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp').set_index('timestamp')
    else:
        logger.info("缓存未找到，通过API获取历史数据...")
        try:
            import asyncio
            
            async def fetch_data_async():
                # Use the factory to get the aggregated exchange instance
                exchange = await ExchangeFactory.create_exchange('aggregated')
                try:
                    return await exchange.fetch_historical_data(SYMBOL, TIMEFRAME, YEARS_OF_DATA)
                finally:
                    if hasattr(exchange, 'close'):
                        await exchange.close()

            data = asyncio.run(fetch_data_async())
            
            # Save data to cache for future runs
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            data.to_csv(CACHE_PATH)
            logger.info(f"数据已缓存到: {CACHE_PATH}")

        except Exception as e:
            logger.error(f"通过API获取数据时出错: {e}")
            return

    # The rest of the script remains the same...

    # --- 基础指标 ---
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['SMA_30'] = data['close'].rolling(window=30).mean()
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # --- 新增指标 ---
    print("计算新增特征: MACD, Bollinger Bands, ATR...")
    # MACD
    exp12 = data['close'].ewm(span=12, adjust=False).mean()
    exp26 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma_20 = data['close'].rolling(window=20).mean()
    std_20 = data['close'].rolling(window=20).std()
    data['BB_upper'] = sma_20 + (std_20 * 2)
    data['BB_lower'] = sma_20 - (std_20 * 2)

    # ATR
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=14).mean()

    # --- 2. 定义目标和特征 ---
    data['target'] = np.sign(data['close'].shift(-5) - data['close'])
    data = data.dropna()
    data = data[data['target'] != 0]

    features = [
        'open', 'high', 'low', 'close', 'volume', 
        'SMA_10', 'SMA_30', 'RSI_14',
        'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR'
    ]
    X = data[features]
    y = data['target']

    # --- 3. 划分训练集和测试集 (时间序列划分) ---
    split_ratio = 0.7
    split_index = int(len(data) * split_ratio)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    print(f"数据划分完成。训练集: {len(X_train)}条, 测试集: {len(X_test)}条")

    # --- 4. 训练模型 ---
    print("训练随机森林分类器...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"模型在测试集上的准确率: {accuracy:.4f}")

    # --- 5. 获取预测概率 ---
    print("获取预测概率...")
    probabilities = model.predict_proba(X_test)[:, 1]
    
    test_data = X_test.copy()
    test_data['proba'] = probabilities
    test_data['close'] = data['close'].loc[X_test.index]

    # --- 6. 执行回测 ---
    print("\n执行回测...")
    backtest_results = simple_backtest(test_data, proba_col="proba", buy_th=0.55, sell_th=0.45, fee=0.001)
    
    print("\n--- 回测结果 ---")
    pprint.pprint(backtest_results)

if __name__ == '__main__':
    run_model_backtest()
