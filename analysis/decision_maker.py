
import pandas as pd
from analysis.performance_monitor import PerformanceMonitor
from analysis.market_regime_detection import detect_regime_change
from data.binance import get_klines_bian
from exchange.okx_exchange import OKXExchange
from utils.data_normalization import normalize_binance_df

class DecisionMaker:
    """
    决策者模块。

    整合性能指标和市场状态，以决定是否需要触发模型再训练。
    """

    def __init__(self, config: dict):
        """
        初始化决策者。
        :param config: 包含决策阈值等参数的配置字典。
        """
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        # 使用一个无密钥的客户端获取公共市场数据
        self.public_client = OKXExchange(sandbox=True)

    def should_retrain(self) -> dict:
        """
        根据预设规则判断是否应该触发再训练。

        :return: 一个包含决策结果和原因的字典。
        """
        retrain_triggers = []

        # 1. 评估性能指标 (KPIs)
        kpi_thresholds = self.config.get('kpi_thresholds', {})
        try:
            kpis = self.performance_monitor.calculate_kpis()
            if kpis.get('status') == 'SUCCESS':
                sharpe_key = [k for k in kpis.keys() if 'sharpe_ratio' in k][0]
                latest_sharpe = kpis.get(sharpe_key)
                max_drawdown = kpis.get('max_drawdown_pct')

                # 规则1: 夏普比率过低
                sharpe_threshold = kpi_thresholds.get('min_sharpe_ratio', 0.5)
                if latest_sharpe is not None and latest_sharpe < sharpe_threshold:
                    retrain_triggers.append(f"Sharpe ratio ({latest_sharpe:.2f}) is below threshold ({sharpe_threshold}).")

                # 规则2: 最大回撤过大
                drawdown_threshold = kpi_thresholds.get('max_drawdown_pct', -15.0)
                if max_drawdown is not None and max_drawdown < drawdown_threshold:
                    retrain_triggers.append(f"Max drawdown ({max_drawdown:.2f}%) exceeds threshold ({drawdown_threshold}%)." )
            else:
                print(f"Warning: Could not get valid KPIs. Reason: {kpis.get('message')}")

        except Exception as e:
            print(f"Error calculating KPIs: {e}")

        # 2. 评估市场状态变化
        try:
            # 获取用于市场状态分析的数据
            market_data = get_klines_bian(self.public_client, symbol="BTC-USDT", interval="1H", years=1.0)
            if market_data is not None and not market_data.empty:
                market_data = normalize_binance_df(market_data)
                regime_result = detect_regime_change(market_data)

                # 规则3: 市场模式发生变化
                if regime_result.get('regime_changed', False):
                    retrain_triggers.append(f"Market regime change detected: {regime_result.get('reason')}")
            else:
                print("Warning: Could not fetch market data for regime analysis.")

        except Exception as e:
            print(f"Error analyzing market regime: {e}")

        # 3. 最终决策
        if retrain_triggers:
            return {
                "should_retrain": True,
                "reasons": retrain_triggers
            }
        else:
            return {
                "should_retrain": False,
                "reasons": ["All performance and market metrics are within acceptable limits."]
            }

if __name__ == '__main__':
    # 示例配置
    decision_config = {
        'kpi_thresholds': {
            'min_sharpe_ratio': 0.5,      # 当30周期年化夏普比率低于0.5时
            'max_drawdown_pct': -15.0,  # 当最大回撤超过-15%时
        }
    }

    # 创建实例并做出决策
    decision_maker = DecisionMaker(config=decision_config)
    decision = decision_maker.should_retrain()

    print("\n--- Retraining Decision ---")
    print(f"Decision to Retrain: {decision['should_retrain']}")
    print("Reasons:")
    for reason in decision['reasons']:
        print(f"- {reason}")
    print("-------------------------")
