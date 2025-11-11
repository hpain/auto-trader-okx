import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from strategies.base_strategy import BaseStrategy

class Backtester:
    """
    一个通用的、基于事件驱动的向量化回测引擎。
    它严格按照时间顺序处理数据，避免了未来函数。
    """

    def __init__(self, strategy: BaseStrategy, data: pd.DataFrame, initial_capital=100000, commission=0.001, stop_loss=None, take_profit=None, regime_filter_period=None):
        self.strategy = strategy
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.regime_filter_period = regime_filter_period
        self.trades = []
        self.equity_curve = None

    def run(self, stats_output_path=None):
        """执行回测。"""
        print(f"开始回测策略: {self.strategy.get_name()}...")
        print(f"初始资金: {self.initial_capital:,.2f}")
        if self.stop_loss:
            print(f"止损设置: {self.stop_loss:.2%}")
        if self.take_profit:
            print(f"止盈设置: {self.take_profit:.2%}")
        if self.regime_filter_period:
            print(f"市场状态过滤器: EMA({self.regime_filter_period})")
            self.data['regime_ma'] = self.data['close'].ewm(span=self.regime_filter_period, adjust=False).mean()

        print(f"时间范围: {self.data.index[0]} -> {self.data.index[-1]}")

        capital = self.initial_capital
        position = 0
        entry_price = 0
        equity = [self.initial_capital]
        
        start_index = max(50, self.regime_filter_period if self.regime_filter_period else 0)

        for i in range(start_index, len(self.data)):
            price = self.data['close'].iloc[i]

            # 1. Check for exit conditions (stop-loss or take-profit)
            if position > 0:
                if self.stop_loss and (price <= entry_price * (1 - self.stop_loss)):
                    capital = position * price * (1 - self.commission)
                    self.trades.append(('stop-loss', self.data.index[i], price, position))
                    position = 0
                    entry_price = 0
                    equity.append(capital)
                    continue

                if self.take_profit and (price >= entry_price * (1 + self.take_profit)):
                    capital = position * price * (1 - self.commission)
                    self.trades.append(('take-profit', self.data.index[i], price, position))
                    position = 0
                    entry_price = 0
                    equity.append(capital)
                    continue

            # 2. Get trading signal from strategy
            current_slice = self.data.iloc[0:i+1]
            signal = self.strategy.generate_signals(current_slice)

            # 3. Apply Regime Filter and Execute trading signal
            is_bull_regime = True
            if self.regime_filter_period:
                is_bull_regime = price > self.data['regime_ma'].iloc[i]

            if signal == 'buy' and position == 0:
                if is_bull_regime:
                    position = (capital * (1 - self.commission)) / price
                    entry_price = price
                    capital = 0
                    self.trades.append(('buy', self.data.index[i], price, position))
            
            elif signal == 'sell' and position > 0:
                capital = position * price * (1 - self.commission)
                self.trades.append(('sell', self.data.index[i], price, position))
                position = 0
                entry_price = 0

            current_equity = capital + position * price
            equity.append(current_equity)

        self.equity_curve = pd.Series(equity, index=self.data.index[start_index-1:])
        print("回测执行完毕。")
        return self.generate_stats(output_path=stats_output_path)

    def generate_stats(self, output_path=None):
        """计算并返回性能指标。"""
        if self.equity_curve is None or self.equity_curve.empty:
            print("没有市值曲线，无法生成报告。")
            return {}

        total_return = (self.equity_curve.iloc[-1] / self.initial_capital) - 1
        returns = self.equity_curve.pct_change().dropna()
        
        # 假设数据是1小时的，年化因子为 252 * 24
        trading_hours_per_year = 252 * 24 
        annual_return = returns.mean() * trading_hours_per_year
        annual_volatility = returns.std() * np.sqrt(trading_hours_per_year)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        stats = {
            "Total Return": f"{total_return:.2%}",
            "Annual Return": f"{annual_return:.2%}",
            "Annual Volatility": f"{annual_volatility:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Total Trades": len(self.trades)
        }
        
        print("--- 回测性能报告 ---")
        for key, value in stats.items():
            print(f"{key:<20} {value}")
        print("---------------------\n")

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
            print(f"性能报告已保存至: {output_path}")
        
        return stats

    def plot_equity(self, output_path="equity_curve.png"):
        """绘制市值曲线并保存到文件。"""
        if self.equity_curve is None:
            print("没有市值曲线可供绘制。")
            return
            
        plt.figure(figsize=(12, 6))
        self.equity_curve.plot(label=f'{self.strategy.get_name()} Equity Curve', legend=True)
        plt.title(f"Strategy Performance: {self.strategy.get_name()}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        
        plt.savefig(output_path)
        plt.close()
        print(f"资金曲线图已保存至: {output_path}")