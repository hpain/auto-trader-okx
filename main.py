import asyncio
import pandas as pd

# 导入我们所有的新组件
from strategies.moving_average import MovingAverageStrategy
from strategies.lgb_strategy import LGBStrategy
from trader.portfolio_manager import PortfolioManager
from trader.execution_handler import ExecutionHandler
from trader.strategy_manager import StrategyManager

# 导入新的组件和配置加载器
from exchange.aggregated_exchange import AggregatedExchange
from utils.config_loader import load_config
from features.feature_engineering import generate_features
from utils.logger import setup_trader_logger, CycleLogger

async def main():
    """
    全新的自动化交易主循环，以异步原生方式运行，并确保资源被优雅释放。
    """
    print("--- System Initializing ---")
    trader_logger = setup_trader_logger()
    print("Logger initialized.")
    config = load_config('config/settings.yaml')
    print("Configuration loaded.")

    exchange_client = None
    try:
        # 1. 初始化所有组件
        # 1.1 异步初始化聚合交易所客户端
        exchange_config = config['exchange']
        api_credentials = config.get('okx', {})
        dune_credentials = config.get('dune', {})
        
        exchange_client = await AggregatedExchange.create_async(
            primary_exchange_id=exchange_config['primary_exchange'],
            secondary_exchange_ids=exchange_config['secondary_exchanges'],
            api_key=api_credentials.get('api_key'),
            api_secret=api_credentials.get('secret_key'),
            passphrase=api_credentials.get('passphrase'),
            sandbox=str(api_credentials.get('flag', '0')) == '0',
            dune_api_key=dune_credentials.get('api_key')
        )
        print("Aggregated exchange client initialized.")

        # 1.2 初始化策略
        strategy_1_config = {'short_window': 10, 'long_window': 30}
        strategy_2_config = {'short_window': 20, 'long_window': 60}
        strategy_3_config = {'model_dir': 'models', 'model_name': 'best_model.pkl', 'metadata_name': 'metadata.json'}
        ma_strategy_1 = MovingAverageStrategy(strategy_name="MA_10_30", config=strategy_1_config)
        ma_strategy_2 = MovingAverageStrategy(strategy_name="MA_20_60", config=strategy_2_config)
        lgb_strategy = LGBStrategy(strategy_name="LGB_Main", config=strategy_3_config)
        strategy_army = [ma_strategy_1, ma_strategy_2, lgb_strategy]
        print(f"Initialized {len(strategy_army)} strategies.")

        # 1.3 初始化管理器
        strategy_manager = StrategyManager(strategies=strategy_army)
        portfolio_manager = PortfolioManager(strategies=strategy_army, capital=10000.0, risk_config={'risk_per_trade': 0.01, 'max_portfolio_risk': 0.05, 'stop_loss_window': 20}, exchange_client=exchange_client)
        execution_handler = ExecutionHandler(exchange_client=exchange_client)

        print("--- Initialization Complete. Starting Live Trading Loop ---")

        # 2. 主循环
        while True:
            cycle_timestamp = pd.Timestamp.now(tz='UTC').isoformat()
            cycle_logger = CycleLogger(logger=trader_logger, cycle_id=cycle_timestamp)
            
            try:
                print(f"\n{'='*20} New Cycle at {pd.Timestamp.now()} {'='*20}")
                
                # 2.1 异步获取市场数据
                print("Fetching latest market data...")
                symbols = ["BTC-USDT", "ETH-USDT"]
                interval = "1H"
                
                data_for_pm = {}
                for symbol in symbols:
                    raw_data = await exchange_client.fetch_candles(symbol, interval, limit=200)
                    if raw_data is None or raw_data.empty:
                        print(f"Warning: Failed to fetch market data for {symbol}, skipping...")
                        continue
                    featured_data = generate_features(raw_data, news_csv_path=None)
                    data_for_pm[symbol] = featured_data

                if not data_for_pm:
                    raise ValueError("Failed to fetch market data for any symbol.")

                # 2.2 决策
                print("Analyzing market regime and rebalancing portfolio...")
                for symbol in data_for_pm.keys():
                    featured_data = data_for_pm[symbol]
                    regime_info = strategy_manager.analyze_market_regime(featured_data, cycle_logger)
                    recommended_strategy_name = strategy_manager.get_recommended_strategy(regime_info, cycle_logger)
                    strategy_manager.switch_strategy(recommended_strategy_name, cycle_logger)
                    selected_strategies = strategy_manager.get_active_strategies()
                    portfolio_manager.asset_strategies[symbol] = selected_strategies

                trade_orders, _ = portfolio_manager.rebalance(data_for_pm, cycle_logger)

                # 2.3 执行
                if trade_orders:
                    execution_report = await execution_handler.execute_trades(trade_orders, cycle_logger)
                    portfolio_manager.update_positions(execution_report)
                else:
                    print("No new trade orders to execute.")
                    cycle_logger.set_status("NO_ACTION")

                # 2.4 记录状态
                total_position_value = 0.0
                for symbol, qty in portfolio_manager.positions.items():
                    if symbol in data_for_pm and not data_for_pm[symbol].empty:
                        current_price = data_for_pm[symbol].iloc[-1]['close']
                        total_position_value += qty * current_price
                total_value = portfolio_manager.capital + total_position_value
                cycle_logger.add_portfolio_info(final_positions=portfolio_manager.positions, capital=portfolio_manager.capital, position_value=total_position_value, total_value=total_value)

                # 2.5 等待
                wait_seconds = 3600
                print(f"Cycle finished. Waiting for {wait_seconds / 60:.1f} minutes...")
                await asyncio.sleep(wait_seconds)

            except KeyboardInterrupt:
                print("\nUser interrupted the process. Shutting down.")
                cycle_logger.set_error("User interrupted.")
                break
            except Exception as e:
                print(f"FATAL ERROR in main loop: {e}")
                cycle_logger.set_error(str(e))
                await asyncio.sleep(60)
            finally:
                cycle_logger.commit()
    finally:
        if exchange_client:
            print("\nClosing exchange connections...")
            await exchange_client.close()


if __name__ == "__main__":
    asyncio.run(main())