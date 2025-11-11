
import json
import pandas as pd
import numpy as np
from pathlib import Path

class PerformanceMonitor:
    """
    负责读取交易周期日志，分析并计算关键性能指标(KPIs)。
    这是实现自动反馈循环的第二阶段的核心组件。
    """

    def __init__(self, log_file_path: str = 'logs/trading_cycles.log'):
        """
        初始化性能监视器。
        :param log_file_path: 结构化日志文件的路径。
        """
        self.log_file = Path(log_file_path)
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found at: {self.log_file}")
        self.raw_logs = []
        self.log_df = pd.DataFrame()

    def _load_logs(self):
        """
        从日志文件中加载所有JSON对象。
        """
        self.raw_logs = []
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    self.raw_logs.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed log line: {line.strip()}")
        
        if not self.raw_logs:
            print("Warning: No logs found to analyze.")
            return False
            
        self.log_df = pd.DataFrame(self.raw_logs)
        self.log_df['cycle_id'] = pd.to_datetime(self.log_df['cycle_id'])
        self.log_df = self.log_df.set_index('cycle_id').sort_index()
        return True

    def calculate_kpis(self, window: int = 30, periods_per_year: int = 365*24) -> dict:
        """
        计算指定窗口期内的关键性能指标。

        :param window: 计算滚动指标的回看窗口大小。
        :param periods_per_year: 用于年化夏普比率的周期数 (例如，1H数据为365*24)。
        :return: 一个包含所有KPIs的字典。
        """
        if not self._load_logs() or len(self.log_df) < 2:
            return {'status': 'NO_DATA', 'message': 'Not enough log data to calculate KPIs.'}

        # 1. 提取总资产净值并计算回报率
        try:
            portfolio_df = self.log_df['portfolio'].apply(pd.Series)
            if 'total_value' not in portfolio_df.columns:
                raise KeyError("'total_value' not found in the portfolio log data.")
            
            self.log_df['total_value'] = portfolio_df['total_value']
            self.log_df['returns'] = self.log_df['total_value'].pct_change().fillna(0)
        except Exception as e:
            return {'status': 'ERROR', 'message': f"Failed to process portfolio data: {e}"}

        # 2. 计算累计回报
        self.log_df['cumulative_returns'] = (1 + self.log_df['returns']).cumprod()
        total_cumulative_return = self.log_df['cumulative_returns'].iloc[-1] - 1

        # 3. 计算最大回撤
        running_max = self.log_df['cumulative_returns'].cummax()
        drawdown = (self.log_df['cumulative_returns'] - running_max) / running_max
        max_drawdown = drawdown.min()

        # 4. 计算胜率 (周期回报为正的比例)
        win_rate = (self.log_df['returns'] > 0).mean()

        # 5. 计算滚动年化夏普比率
        rolling_mean = self.log_df['returns'].rolling(window=window).mean()
        rolling_std = self.log_df['returns'].rolling(window=window).std()
        # 避免除以零
        rolling_std[rolling_std == 0] = np.nan
        
        annualized_sharpe_ratio = (rolling_mean * periods_per_year) / (rolling_std * np.sqrt(periods_per_year))
        latest_sharpe_ratio = annualized_sharpe_ratio.iloc[-1] if not annualized_sharpe_ratio.empty else None

        kpis = {
            'status': 'SUCCESS',
            'log_entries': len(self.log_df),
            'first_log_time': self.log_df.index[0].isoformat(),
            'last_log_time': self.log_df.index[-1].isoformat(),
            'total_cumulative_return_pct': total_cumulative_return * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate * 100,
            f'sharpe_ratio_annualized_{window}_periods': latest_sharpe_ratio,
        }

        return kpis

if __name__ == '__main__':
    # 一个简单的使用示例
    try:
        monitor = PerformanceMonitor()
        kpi_results = monitor.calculate_kpis()
        print("\n--- Performance Monitor KPIs ---")
        for key, value in kpi_results.items():
            print(f"{key}: {value}")
        print("---------------------------------")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
