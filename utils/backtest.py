import pandas as pd
import numpy as np

def run_backtest(
    predictions: pd.Series,
    probabilities: np.ndarray,
    test_data: pd.DataFrame,
    confidence_threshold: float = 0.95,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.05,
    fee: float = 0.001,  # Default fee at 0.1%
    timeout: int = 12     # Default to matching the triple barrier timeout
) -> tuple[float, float, float, int, pd.Series]:
    """
    执行基于高置信度、三极大值逻辑(止损、止盈、时间窗口)的回测。
    """
    returns = []
    trade_count = 0
    successful_trades = 0

    close_prices = test_data["close"].values
    high_prices = test_data["high"].values
    low_prices = test_data["low"].values
    n = len(test_data)

    # 确保索引对齐
    aligned_predictions = predictions.loc[test_data.index]
    aligned_probs = pd.Series(probabilities, index=test_data.index)

    for i in range(n):
        prob = aligned_probs.iloc[i]
        pred = aligned_predictions.iloc[i]

        # 决策：只有当预测为1且概率高于置信度门槛时才开仓
        if pred == 1 and prob >= confidence_threshold:
            trade_count += 1
            entry_price = close_prices[i]
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)

            trade_return_gross = 0.0
            
            # 寻找在 timeout 窗口内哪个边界先被触发
            # 限制 lookahead 不会超过 test_data 边界
            horizon = min(timeout, n - i - 1)
            
            if horizon == 0:
                # 最后一根，无法回测，直接给 0
                returns.append(0.0)
                continue

            hit_tp = False
            hit_sl = False
            final_bar_idx = i + horizon
            
            for j in range(i + 1, i + 1 + horizon):
                h = high_prices[j]
                l = low_prices[j]
                
                # 优先级: 止损 > 止盈 (保守估计)
                if l <= stop_loss_price:
                    hit_sl = True
                    trade_return_gross = -stop_loss_pct
                    break
                if h >= take_profit_price:
                    hit_tp = True
                    trade_return_gross = take_profit_pct
                    successful_trades += 1
                    break
            
            # 如果超时未触发 TP/SL，则在 timeout 结束时的收盘价出场
            if not hit_tp and not hit_sl:
                exit_price = close_prices[final_bar_idx]
                trade_return_gross = (exit_price / entry_price) - 1
                if trade_return_gross >= 0.005: 
                    successful_trades += 1
            
            # 从毛利润中减去双边手续费
            trade_return = trade_return_gross - (2 * fee)
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