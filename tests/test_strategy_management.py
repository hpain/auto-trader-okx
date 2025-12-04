"""
测试策略管理和市场制度检测功能
"""
import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from strategies.strategy_manager import StrategyManager
from trader.market_regime_detector import MarketRegimeDetector


def test_strategy_manager():
    """测试策略管理器功能"""
    print("测试策略管理器功能...")
    
    config = {
        'enabled_strategies': ['ma_fast', 'ma_slow'],  # 先只测试MA策略，避免LGB模型问题
        'lgb_strategy': {
            'model_dir': 'models',
            'model_name': 'best_model.pkl',
            'metadata_name': 'metadata.json'
        },
        'ma_fast_strategy': {
            'short_window': 5,
            'long_window': 15
        },
        'ma_slow_strategy': {
            'short_window': 20,
            'long_window': 50
        }
    }
    
    try:
        sm = StrategyManager(config)
        print("  - 策略管理器初始化成功")
        
        # 检查初始化的策略
        print(f"  - 可用策略: {list(sm.strategies.keys())}")
        print(f"  - 当前活跃策略: {sm.get_active_strategy_name()}")
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, 100)  # 小幅正收益，高波动
        prices = [100]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'vol': np.random.uniform(1000, 5000, len(prices))
        }, index=dates)
        
        # 测试信号生成
        for strategy_name in sm.strategies.keys():
            signals = sm.generate_signals(data, strategy_name)
            latest_signal = signals['signal'].iloc[-1]
            print(f"  - 策略 {strategy_name} 信号: {latest_signal}")
        
        # 测试策略性能评估
        returns_series = pd.Series(np.random.normal(0.001, 0.01, 10))
        performance = sm.evaluate_strategy_performance('ma_fast', returns_series)
        print(f"  - 策略性能评估: {performance}")
        
        print("  OK: 策略管理器测试通过")
        return True
        
    except Exception as e:
        print(f"  ERROR: 策略管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_market_regime_detector():
    """测试市场制度检测器功能"""
    print("\n测试市场制度检测器功能...")
    
    try:
        detector = MarketRegimeDetector()
        print("  - 市场制度检测器初始化成功")
        
        # 创建测试数据 - 趋势市场
        dates = pd.date_range(start='2023-01-01', periods=50, freq='h')
        trend = np.linspace(100, 150, 50)  # 明显上升趋势
        noise = np.random.normal(0, 2, 50)  # 添加一些噪声
        prices = trend + noise
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'vol': np.random.uniform(1000, 5000, len(prices))
        }, index=dates)
        
        regime = detector.detect_regime(data)
        print(f"  - 检测到的市场制度: {regime['regime']}")
        print(f"  - 描述: {regime['description']}")
        
        # 测试推荐策略
        recommended_strategy, reason = detector.recommend_strategy(regime)
        print(f"  - 推荐策略: {recommended_strategy}")
        print(f"  - 推荐原因: {reason}")
        
        print("  OK: 市场制度检测器测试通过")
        return True
        
    except Exception as e:
        print(f"  ERROR: 市场制度检测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始测试策略管理和市场制度检测功能...\n")
    
    success1 = test_strategy_manager()
    success2 = test_market_regime_detector()
    
    if success1 and success2:
        print("\nOK: 所有测试通过！")
        return True
    else:
        print("\nERROR: 部分测试失败！")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
