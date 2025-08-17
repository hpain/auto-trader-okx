# models/backtest.py
import numpy as np
import pandas as pd

def simple_backtest(df: pd.DataFrame, proba_col="proba", buy_th=0.55, sell_th=0.45, fee=0.0005):
    """
    非持仓型二分类策略：
    - proba > buy_th -> 多头（下根bar按close变化计收益）
    - proba < sell_th -> 空头（若交易所不允许做空，可改为观望）
    - 其他 -> 观望
    这里演示：只做多（空头变观望），包含单边手续费fee。
    """
    s = df.copy()
    s["position"] = 0
    s.loc[s[proba_col] > buy_th, "position"] = 1
    # 可选：允许空头
    # s.loc[s[proba_col] < sell_th, "position"] = -1

    # 下期收益
    s["ret_1"] = s["close"].pct_change().shift(-1)  # 用下一根close变化
    # 策略收益（含手续费，开仓/平仓扣一次；简化处理）
    s["trade"] = s["position"].diff().abs().fillna(0)
    s["fee_cost"] = s["trade"] * fee
    s["strat_ret"] = s["position"] * s["ret_1"] - s["fee_cost"]

    s["equity"] = (1 + s["strat_ret"].fillna(0)).cumprod()

    # 指标
    total_ret = s["equity"].iloc[-1] - 1
    ann_factor = max(1, int(365*24/1))  # 粗略：1H数据≈8760/年；可根据bar实际间隔替换
    ret_series = s["strat_ret"].dropna()
    ann_ret = ret_series.mean() * ann_factor
    ann_vol = ret_series.std(ddof=1) * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # 最大回撤
    roll_max = s["equity"].cummax()
    drawdown = s["equity"] / roll_max - 1
    mdd = drawdown.min()

    return {
        "total_return": float(total_ret),
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "trades": int(s["trade"].sum())
    }
