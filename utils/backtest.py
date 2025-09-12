import pandas as pd
import numpy as np

def run_backtest(
    predictions: pd.Series,
    probabilities: np.ndarray,
    test_data: pd.DataFrame,
    confidence_threshold: float = 0.95,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.05, # New parameter
) -> tuple[float, float, float, int, pd.Series]:
    """
    执行基于高置信度、止损和止盈的回测。

    Args:
        predictions (pd.Series): 模型预测的标签 (0 或 1).
        probabilities (np.ndarray): 模型预测的概率 (对应标签1的概率).
        test_data (pd.DataFrame): 包含价格和未来价格的测试数据.
        confidence_threshold (float): 执行交易的最低置信度.
        stop_loss_pct (float): 止损百分比.
        take_profit_pct (float): 止盈百分比.

    Returns:
        tuple: (总收益率, 最大回撤, 达标交易成功率, 交易次数, 回报序列)
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

        # 决策：只有当预测为1且概率高于置信度门槛时才开仓
        if pred == 1 and prob >= confidence_threshold:
            trade_count += 1
            entry_price = row["close"]
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)

            # 交易结果判断 (优先级: 止损 > 止盈 > 期末收盘)
            # 1. 检查是否触发止损
            if row["future_low"] <= stop_loss_price:
                trade_return = -stop_loss_pct
            # 2. 检查是否触发止盈
            elif row["future_high"] >= take_profit_price:
                trade_return = take_profit_pct
                successful_trades += 1 # 止盈即为成功
            # 3. 如果都未触发，则以未来收盘价出场
            else:
                trade_return = (row["future_close"] / entry_price) - 1
                if trade_return >= 0.005: # 检查是否达到0.5%的目标
                    successful_trades += 1
            
            returns.append(trade_return)
        else:
            # 不交易，当日收益为0
            returns.append(0.0)

    if not returns:
        return 0.0, 0.0, 0.0, 0, pd.Series(dtype=float)

    returns_series = pd.Series(returns)
    total_return = (1 + returns_series).prod() - 1
    
    # 计算最大回撤
    cum_returns = (1 + returns_series).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    # 计算达标交易成功率
    success_rate = successful_trades / trade_count if trade_count > 0 else 0.0

    return total_return, max_drawdown, success_rate, trade_count, returns_series

def run_backtest_regression(
    predicted_returns: pd.Series,
    test_data: pd.DataFrame,
    entry_threshold: float,
    stop_loss_pct: float,
    take_profit_pct: float, # New parameter
) -> tuple[float, float, float, int, pd.Series]:
    """
    执行基于回归预测值和止损的回测。

    Args:
        predicted_returns (pd.Series): 模型预测的未来连续回报率.
        test_data (pd.DataFrame): 包含价格和未来价格的测试数据.
        entry_threshold (float): 决定是否交易的最低预测回报率阈值.
        stop_loss_pct (float): 止损百分比.

    Returns:
        tuple: (总收益率, 最大回撤, 成功率, 交易次数, 回报序列)
    """
    returns = []
    trade_count = 0
    successful_trades = 0

    # 确保索引对齐
    aligned_predictions = predicted_returns.loc[test_data.index]

    for i in range(len(test_data)):
        row = test_data.iloc[i]
        predicted_ret = aligned_predictions.iloc[i]

        # 决策：只有当预测回报率 > 入场阈值时才开仓
        if predicted_ret > entry_threshold:
            trade_count += 1
            entry_price = row["close"]
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct) # New line

            # 交易结果判断 (优先级: 止损 > 止盈 > 期末收盘)
            # 1. 检查是否触发止损
            if row["future_low"] <= stop_loss_price:
                trade_return = -stop_loss_pct
            # 2. 检查是否触发止盈
            elif row["future_high"] >= take_profit_price:
                trade_return = take_profit_pct
                successful_trades += 1 # 止盈即为成功
            # 3. 如果都未触发，则以未来收盘价出场
            else:
                trade_return = (row["future_close"] / entry_price) - 1
            
            if trade_return > 0:
                successful_trades += 1
            
            returns.append(trade_return)
        else:
            # 不交易，当日收益为0
            returns.append(0.0)

    if not returns:
        return 0.0, 0.0, 0.0, 0, pd.Series(dtype=float)

    returns_series = pd.Series(returns, index=test_data.index)
    total_return = (1 + returns_series).prod() - 1
    
    # 计算最大回撤
    cum_returns = (1 + returns_series).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    # 计算成功率 (盈利的交易 / 总交易)
    success_rate = successful_trades / trade_count if trade_count > 0 else 0.0

    return total_return, max_drawdown, success_rate, trade_count, returns_series