import pandas as pd
import numpy as np

def run_backtest(
    predictions: pd.Series,
    probabilities: np.ndarray,
    test_data: pd.DataFrame,
    confidence_threshold: float = 0.95,
    stop_loss_pct: float = 0.02,
) -> tuple[float, float, float, int]:
    """
    执行基于高置信度和止损的回测。

    Args:
        predictions (pd.Series): 模型预测的标签 (0 或 1).
        probabilities (np.ndarray): 模型预测的概率 (对应标签1的概率).
        test_data (pd.DataFrame): 包含价格和未来价格的测试数据.
        confidence_threshold (float): 执行交易的最低置信度.
        stop_loss_pct (float): 止损百分比.

    Returns:
        tuple: (总收益率, 最大回撤, 达标交易成功率, 交易次数)
    """
    returns = []
    trade_count = 0
    successful_trades = 0

    # 确保索引对齐
    aligned_predictions = predictions.loc[test_data.index]
    aligned_probs = pd.Series(probabilities, index=test_data.index)

    for i in range(len(test_data)):
        row = test_data.iloc[i]
        prob = aligned_probs.iloc[i]
        pred = aligned_predictions.iloc[i]

        # 决策：只有当预测为1且概率高于阈值时才开仓
        if pred == 1 and prob >= confidence_threshold:
            trade_count += 1
            entry_price = row["close"]
            stop_loss_price = entry_price * (1 - stop_loss_pct)

            # 判断交易结果：
            # 1. 如果未来最低价触及止损线，则以止损价出场
            if row["future_low"] <= stop_loss_price:
                trade_return = -stop_loss_pct
            # 2. 否则，以未来收盘价出场
            else:
                trade_return = (row["future_close"] / entry_price) - 1
                if trade_return >= 0.005: # 检查是否达到0.5%的目标
                    successful_trades += 1
            
            returns.append(trade_return)
        else:
            # 不交易，当日收益为0
            returns.append(0.0)

    if not returns:
        return 0.0, 0.0, 0.0, 0

    returns_series = pd.Series(returns)
    total_return = (1 + returns_series).prod() - 1
    
    # 计算最大回撤
    cum_returns = (1 + returns_series).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    # 计算达标交易成功率
    success_rate = successful_trades / trade_count if trade_count > 0 else 0.0

    return total_return, max_drawdown, success_rate, trade_count