import time
import pandas as pd

# 导入我们所有的新组件
from strategies.moving_average import MovingAverageStrategy
from strategies.lgb_strategy import LGBStrategy
from trader.portfolio_manager import PortfolioManager
from trader.execution_handler import ExecutionHandler
from trader.strategy_manager import StrategyManager

# 导入新的工厂和配置加载器
from exchange.factory import ExchangeFactory
from utils.config_loader import load_config
from features.feature_engineering import generate_features
from utils.logger import setup_trader_logger, CycleLogger

def main_loop():
    """
    全新的自动化交易主循环，集成了市场状态检测和策略管理。
    """
    print("--- System Initializing ---")

    # 1. 初始化日志记录器
    trader_logger = setup_trader_logger()
    print("Logger initialized.")

    # 2. 加载配置
    config = load_config('config/settings.yaml')
    print("Configuration loaded.")

    # 3. 初始化所有组件
    # 3.1 初始化聚合交易所客户端
    # 使用 ExchangeFactory 创建聚合交易所实例
    api_credentials = config.get('okx', {}) # 以okx的凭证为例，将来可以做得更通用
    exchange_client = ExchangeFactory.create_exchange(
        'aggregated',
        api_key=api_credentials.get('api_key'),
        api_secret=api_credentials.get('secret_key'),
        passphrase=api_credentials.get('passphrase'),
        sandbox=api_credentials.get('sandbox', True)
    )
    print(f"Aggregated exchange client initialized.")

    # 3.2 初始化策略 (与之前相同)
    strategy_1_config = {'short_window': 10, 'long_window': 30}
    strategy_2_config = {'short_window': 20, 'long_window': 60}
    strategy_3_config = {
        'model_dir': 'models',
        'model_name': 'best_model.pkl',
        'metadata_name': 'metadata.json'
    }
    ma_strategy_1 = MovingAverageStrategy(strategy_name="MA_10_30", config=strategy_1_config)
    ma_strategy_2 = MovingAverageStrategy(strategy_name="MA_20_60", config=strategy_2_config)
    lgb_strategy = LGBStrategy(strategy_name="LGB_Main", config=strategy_3_config)
    strategy_army = [ma_strategy_1, ma_strategy_2, lgb_strategy]
    print(f"Initialized {len(strategy_army)} strategies.")

    # 3.3 初始化StrategyManager
    strategy_manager = StrategyManager(strategies=strategy_army)
    print(f"Initialized StrategyManager with {len(strategy_army)} strategies.")

    # 3.4 初始化PortfolioManager
    initial_capital = 10000.0
    risk_config = {
        'risk_per_trade': 0.01,
        'max_portfolio_risk': 0.05,
        'stop_loss_window': 20
    }
    portfolio_manager = PortfolioManager(strategies=strategy_army, capital=initial_capital, risk_config=risk_config, exchange_client=exchange_client)

    # 3.5 初始化ExecutionHandler
    execution_handler = ExecutionHandler(exchange_client=exchange_client)

    print("--- Initialization Complete. Starting Live Trading Loop ---")

    # 4. 主循环
    while True:
        cycle_timestamp = pd.Timestamp.now(tz='UTC').isoformat()
        cycle_logger = CycleLogger(logger=trader_logger, cycle_id=cycle_timestamp)
        
        try:
            print(f"\n{'='*20} New Cycle at {pd.Timestamp.now()} {'='*20}")
            
            # 4.1 获取多个交易对的最新市场数据
            print("Fetching latest market data...")
            symbols = ["BTC/USDT", "ETH/USDT"]  # 使用 / 分隔符
            interval = "1H"
            
            # 获取所有交易对的数据
            data_for_pm = {}
            for symbol in symbols:
                # 使用新的聚合交易所客户端获取数据
                # 获取最近200条K线用于特征计算
                raw_data = exchange_client.fetch_candles(symbol, interval, limit=200)
                
                if raw_data is None or raw_data.empty:
                    print(f"Warning: Failed to fetch market data for {symbol}, skipping...")
                    continue

                # 特征工程 (fetch_candles返回的数据已标准化，无需normalize)
                featured_data = generate_features(raw_data, news_csv_path=None)
                data_for_pm[symbol] = featured_data

            if not data_for_pm:
                raise ValueError("Failed to fetch market data for any symbol.")

            # 4.2 (后续逻辑与之前相同...)
            print("Analyzing market regime for each asset and selecting strategies...")
            for symbol in data_for_pm.keys():
                featured_data = data_for_pm[symbol]
                
                regime_info = strategy_manager.analyze_market_regime(featured_data, cycle_logger)
                recommended_strategy_name = strategy_manager.get_recommended_strategy(regime_info, cycle_logger)
                
                strategy_manager.switch_strategy(recommended_strategy_name, cycle_logger)
                
                selected_strategies = strategy_manager.get_active_strategies()
                portfolio_manager.asset_strategies[symbol] = selected_strategies

            trade_orders, _ = portfolio_manager.rebalance(data_for_pm, cycle_logger)

            if trade_orders:
                execution_report = execution_handler.execute_trades(trade_orders, cycle_logger)
                portfolio_manager.update_positions(execution_report)
            else:
                print("No new trade orders to execute.")
                cycle_logger.set_status("NO_ACTION")

            total_position_value = 0.0
            for symbol, qty in portfolio_manager.positions.items():
                if symbol in data_for_pm and not data_for_pm[symbol].empty:
                    current_price = data_for_pm[symbol].iloc[-1]['close']
                    total_position_value += qty * current_price
            
            total_value = portfolio_manager.capital + total_position_value

            cycle_logger.add_portfolio_info(
                final_positions=portfolio_manager.positions,
                capital=portfolio_manager.capital,
                position_value=total_position_value,
                total_value=total_value
            )

            wait_seconds = 3600 # 1 hour
            print(f"Cycle finished. Waiting for {wait_seconds / 60:.1f} minutes...")
            time.sleep(wait_seconds)

        except KeyboardInterrupt:
            print("\nUser interrupted the process. Shutting down.")
            cycle_logger.set_error("User interrupted.")
            cycle_logger.commit()
            break
        except Exception as e:
            print(f"FATAL ERROR in main loop: {e}")
            cycle_logger.set_error(str(e))
            time.sleep(60)
        finally:
            cycle_logger.commit()

def run():
    """
    运行交易系统
    """
    main_loop()

if __name__ == "__main__":
    main_loop()