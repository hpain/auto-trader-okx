import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import json
from sklearn.metrics import accuracy_score

# 1. 加载数据
try:
    file_path = 'data/history/binance_BTCUSDT_1h_4y.csv'
    data = pd.read_csv(file_path)
    data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp').set_index('timestamp')
    print("数据加载成功。")
    print(data.head())
except FileNotFoundError:
    print(f"错误: 数据文件未找到 at {file_path}")
    data = pd.DataFrame(np.random.rand(200, 8), columns=['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'SMA_30', 'RSI_14'])
    data['target'] = np.random.choice([-1, 1], 200)

# 2. 准备特征和目标
print("\n计算额外的技术指标作为基础特征...")
data['SMA_10'] = data['close'].rolling(window=10).mean()
data['SMA_30'] = data['close'].rolling(window=30).mean()
delta = data['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI_14'] = 100 - (100 / (1 + rs))

# 改变预测目标为分类问题：预测5小时后价格是涨(1)还是跌(-1)
data['target'] = np.sign(data['close'].shift(-5) - data['close'])

data = data.dropna()
data = data[data['target'] != 0]

features = ['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'SMA_30', 'RSI_14']
X = data[features].values
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. 初始化并运行遗传编程符号分类器
function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']

gp = SymbolicClassifier(generations=20, population_size=1000,
                        function_set=function_set,
                        parsimony_coefficient=0.0005,
                        max_samples=0.9, verbose=1,
                        random_state=0, n_jobs=-1)

print("\n开始进行遗传编程因子挖掘 (分类模式)...")
gp.fit(X_train, y_train)

print("\n遗传编程完成。")

# 4. 提取并展示最佳因子
print("\n发现的最佳分类程序:")
print(gp._program)
best_program_str = str(gp._program)

# 5. 保存发现的因子到文件
output_path = 'models/mined_features.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

mined_feature_data = {
    "name": f"gp_feature_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M')}",
    "formula": best_program_str,
    "base_features": features, # The list of features (X0, X1, ...) corresponds to this list
    "performance": {
        "accuracy": gp.score(X_test, y_test)
    }
}
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(mined_feature_data, f, indent=4)
print(f"\n已将发现的最佳因子保存到: {output_path}")

# 5. 评估模型性能
print("\n--- 模型性能评估 ---")

# 评估遗传编程分类器 (gp) 本身的性能
score_gp = gp.score(X_test, y_test)
print(f"遗传编程分类器 (GP) 的准确率: {score_gp:.4f}")

# 评估作为基准的随机森林分类器
print("\n评估作为基准的随机森林分类器 (使用原始特征)...")
rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
rf_orig.fit(X_train, y_train)
score_orig = rf_orig.score(X_test, y_test)
print(f"基准随机森林 (Baseline RF) 的准确率: {score_orig:.4f}")

print("\n--- 最终结论 ---")
if score_gp > score_orig:
    print("遗传编程分类器优于基准随机森林模型。")
elif score_gp > 0.52: # 设定一个比随机猜测（50%）稍高的阈值
    print("遗传编程分类器表现优于随机猜测，但未超越更复杂的基准模型。")
else:
    print("所有模型的表现均不理想，接近或低于随机猜测水平。这表明在当前设置下，市场未来5小时的涨跌方向是不可预测的。")
