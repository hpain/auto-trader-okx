import pandas as pd
import numpy as np
import pytest

# 确保能找到 analysis 模块
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.market_regime_detection_new import detect_regime_change

@pytest.fixture
def stable_market_data():
    """
    生成代表稳定、低波动市场的假数据。
    价格在一个窄幅区间内随机游走。
    """
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="h"))
    price = 100 + (np.random.randn(100) * 0.1).cumsum()
    df = pd.DataFrame({"close": price}, index=dates)
    return df

@pytest.fixture
def volatile_market_data():
    """
    生成代表波动率剧增的假数据。
    前50个周期稳定，后50个周期波动性显著放大。
    """
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="h"))
    stable_part = 100 + np.random.randn(50).cumsum() * 0.1
    volatile_part = stable_part[-1] + np.random.randn(50).cumsum() * 2.0 # 波动放大20倍
    price = np.concatenate([stable_part, volatile_part])
    df = pd.DataFrame({"close": price}, index=dates)
    return df

def test_detect_regime_change_in_stable_market(stable_market_data):
    """
    测试：在稳定的市场中，不应检测到状态突变。
    """
    print("Testing with stable market data...")
    # 传入原始价格数据，让函数内部自己处理
    result = detect_regime_change(stable_market_data)
    
    assert isinstance(result, dict)
    assert 'f_value' in result
    assert result['regime_changed'] == False, f"在稳定市场中检测到状态改变，F统计量为{result['f_value']:.2f}，效应量为{result.get('effect_size', 0):.2f}"

def test_detect_regime_change_in_volatile_market(volatile_market_data):
    """
    测试：在市场波动率剧增时，应该能检测到状态突变。
    """
    print("Testing with volatile market data...")
    # 传入原始价格数据，让函数内部自己处理
    result = detect_regime_change(volatile_market_data)

    assert isinstance(result, dict)
    assert 'f_value' in result
    assert result['regime_changed'] == True, f"在波动市场中未能检测到状态改变，F统计量为{result['f_value']:.2f}，效应量为{result.get('effect_size', 0):.2f}"