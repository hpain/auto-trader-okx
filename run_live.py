import time
import asyncio
import pandas as pd
import os
import sys
import argparse

# 导入我们所有的新组件
from strategies.moving_average import MovingAverageStrategy
from strategies.lgb_strategy import LGBStrategy
from strategies.transformer_strategy import TransformerStrategy
from trader.portfolio_manager import PortfolioManager
from trader.execution_handler import ExecutionHandler
from strategies.strategy_manager import StrategyManager

# 导入新的工厂和配置加载器
from exchange.factory import ExchangeFactory
from utils.config_loader import load_config
from features.feature_engineering import generate_features
from utils.logger import setup_trader_logger, CycleLogger

async def main_loop(args):
    """
    全新的自动化交易主循环，集成了市场状态检测和策略管理。
    """
    # --- SAFETY CHECK: PREVENT ACCIDENTAL LIVE/PAPER TRADING ON DEV MACHINE ---
    if sys.platform == 'win32' and not args.mock and not os.environ.get('ALLOW_LOCAL_TRADING'):
        print("\n" + "!" * 80)
        print("CRITICAL SAFETY STOP: Windows Development Environment Detected")
        print("!" * 80)
        print("To protect your capital, Live and Paper trading are DISABLED on this machine.")
        print("You are attempting to run the bot without '--mock'.")
        print("\nALLOWED ACTIONS on this machine:")
        print("1. Run with mock data:   python run_live.py --mock")
        print("2. Run unit tests:       pytest")
        print("\nIf you REALLY want to trade from this laptop, set env var: ALLOW_LOCAL_TRADING=1")
        print("!" * 80 + "\n")
        return
    # --------------------------------------------------------------------------

    print("--- System Initializing ---")
    if args.mock:
        print("!!! RUNNING IN MOCK MODE !!!")

    # 1. 初始化日志记录器
    trader_logger = setup_trader_logger()
    print("Logger initialized.")

    # 2. 加载配置
    try:
        config = load_config('config/settings.yaml')
        print("Configuration loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load config: {e}")
        return

    # 3. 初始化所有组件
    # 3.1 初始化聚合交易所客户端
    # 使用 ExchangeFactory 创建聚合交易所实例
    api_credentials = config.get('okx', {})
    
    # 解析 sandbox 模式：如果 flag 为 "0"，则是 sandbox 模式
    # 默认为 True (安全起见) 如果没有找到配置
    flag = str(api_credentials.get('flag', '0'))
    is_sandbox = (flag == '0')
    print(f"Exchange Mode: {'SANDBOX' if is_sandbox else 'LIVE'}")
    
    try:
        exchange_client = await ExchangeFactory.create_exchange(
            'aggregated',
            api_key=api_credentials.get('api_key'),
            api_secret=api_credentials.get('secret_key'),
            passphrase=api_credentials.get('passphrase'),
            sandbox=is_sandbox,
            mock=args.mock # Pass mock argument
        )
        print(f"Aggregated exchange client initialized.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize exchange: {e}")
        return

    # 3.2 初始化策略
    # 从配置中读取策略参数
    strat_mgmt_config = config.get('strategy_management', {})
    
    # MA Fast
    ma_fast_cfg = strat_mgmt_config.get('ma_fast_strategy', {'short_window': 10, 'long_window': 30})
    ma_strategy_1 = MovingAverageStrategy(strategy_name="MA_Fast", config=ma_fast_cfg)
    
    # MA Slow
    ma_slow_cfg = strat_mgmt_config.get('ma_slow_strategy', {'short_window': 20, 'long_window': 60})
    ma_strategy_2 = MovingAverageStrategy(strategy_name="MA_Slow", config=ma_slow_cfg)
    
    # LGB Strategy
    lgb_cfg = strat_mgmt_config.get('lgb_strategy', {
        'model_dir': 'models',
        'model_name': 'best_model.pkl',
        'metadata_name': 'metadata.json'
    })
    lgb_strategy = LGBStrategy(strategy_name="LGB_Main", config=lgb_cfg)
    
    # Transformer Strategy (New)
    transformer_cfg = strat_mgmt_config.get('transformer_strategy')
    transformer_strategy = None
    if transformer_cfg and transformer_cfg.get('enabled', False):
        print("Initializing Transformer Strategy...")
        transformer_strategy = TransformerStrategy(
            strategy_name="Transformer_Main",
            window_size=transformer_cfg.get('window_size', 60),
            features=None, # Will be auto-loaded/set during load_model or inference
            buy_threshold=transformer_cfg.get('buy_threshold', 0.55),
            sell_threshold=transformer_cfg.get('sell_threshold', 0.45)
        )
        # Load model weights if provided
        model_path = transformer_cfg.get('model_path')
        scaler_path = transformer_cfg.get('scaler_path') # We might need to pass this to strategy or handler
        input_dim = transformer_cfg.get('input_dim', 108) # Default approx
        
        try:
             # Just build/load. Note: input_dim might need dynamic check from scaler in real implementation
             transformer_strategy.build_model(input_dim)
             transformer_strategy.load_model(model_path, input_dim)
             if scaler_path:
                 transformer_strategy.load_scaler(scaler_path)
             print(f"Transformer loaded from {model_path}")
        except Exception as e:
             print(f"Failed to load Transformer: {e}")
    
    print("11111111111111111111111111111111")
    strategy_army = [ma_strategy_1, ma_strategy_2, lgb_strategy]
    print(f"DEBUG: Strategy Army Base: {[s.strategy_name for s in strategy_army]}")
    
    if transformer_strategy:
        # Insert at 0 to make it the DEFAULT active strategy
        strategy_army.insert(0, transformer_strategy)
        print(f"DEBUG: Added Transformer. Army now: {[s.strategy_name for s in strategy_army]}")
        
    print(f"Initialized {len(strategy_army)} strategies: {[s.strategy_name for s in strategy_army]}")

    # 3.3 初始化StrategyManager
    strategy_manager = StrategyManager(strategies=strategy_army)
    print(f"Initialized StrategyManager.")

    # 3.4 初始化PortfolioManager
    # 尝试从配置读取初始资金，如果没有则默认 10000
    risk_mgmt_config = config.get('risk_management', {})
    initial_capital = risk_mgmt_config.get('min_account_balance', 10000.0)
    
    risk_config = {
        'risk_per_trade': config.get('position_sizing', {}).get('risk_per_trade', 0.01),
        'max_portfolio_risk': 0.05, # 可以添加到配置中
        'stop_loss_window': 20
    }

    # ========== Determine Symbols FIRST ==========
    symbols = config.get('trading', {}).get('symbols')
    if not symbols:
            symbols = config.get('backtest', {}).get('symbols_to_test')
    if not symbols:
        symbols = ["BTC/USDT", "ETH/USDT"]
        print("Warning: No symbols found in config, using default: BTC/USDT, ETH/USDT")
    # Standardize symbols
    symbols = [s.replace('-', '/') for s in symbols]
    print(f"Trading Symbols: {symbols}")
    # ===========================================

    # ========== Strategy Weights (Silver Tier) ==========
    strategy_weights = {
        'Transformer_Main': 1.5,
        'LGB_Main': 1.2,
        'MA_Slow': 1.0,
        'MA_Fast': 0.5
    }
    # ====================================================

    # ====================================================

    print(f"DEBUG: FINAL STRATEGY CHECK. ID: {id(strategy_army)}")
    print(f"DEBUG: STRATEGY COUNT: {len(strategy_army)}")
    print(f"DEBUG: STRATEGY NAMES: {[s.strategy_name for s in strategy_army]}")

    portfolio_manager = PortfolioManager(
        strategies=strategy_army, 
        capital=initial_capital, 
        risk_config=risk_config, 
        exchange_client=exchange_client,
        symbols=symbols, # Pass symbols here
        strategy_weights=strategy_weights # Pass weights here
    )

    # 3.5 初始化ExecutionHandler
    execution_handler = ExecutionHandler(exchange_client=exchange_client)

    # 3. Sync State (Position & Balance)
    # This prevents the bot from buying what it already has on restart
    await portfolio_manager.sync_with_exchange()

    print("--- Initialization Complete. Starting Live Trading Loop ---")

    cycle_count = 0

    # 4. 主循环
    while True:
        # Check cycle limit
        if args.cycles is not None and cycle_count >= args.cycles:
            print(f"Reached cycle limit of {args.cycles}. Exiting.")
            break

        # ========== Emergency Stop Check ==========
        if os.path.exists('emergency_stop.flag'):
            msg = "🚨 CRITICAL: emergency_stop.flag detected! Shutting down immediately."
            print(msg)
            trader_logger.critical(msg)
            break
        # ==========================================

        cycle_timestamp = pd.Timestamp.now(tz='UTC').isoformat()
        cycle_logger = CycleLogger(logger=trader_logger, cycle_id=cycle_timestamp)
        
        try:
            print(f"\n{'='*20} New Cycle {cycle_count+1} at {pd.Timestamp.now()} {'='*20}")
            
            # 4.1 获取多个交易对的最新市场数据
            print("Fetching latest market data...")
            
            # (Symbols are already determined above)
            interval = config.get('trading', {}).get('interval', '1H')
            
            # 获取所有交易对的数据
            data_for_pm = {}
            for symbol in symbols:
                # 使用新的聚合交易所客户端获取数据
                # 获取最近200条K线用于特征计算
                raw_data = await exchange_client.fetch_candles(symbol, interval, limit=1000)
                
                if raw_data is None or raw_data.empty:
                    print(f"Warning: Failed to fetch market data for {symbol}, skipping...")
                    continue

                # --- 4.1.2 获取衍生品数据 (Live Data Upgrade) ---
                # 为了配合 Transformer 模型，我们需要实时的资金费率和持仓量
                trader_logger.info(f"Fetching derivatives data for {symbol}...")
                derivatives_data = {}
                
                try:
                    # 1. Funding Rate
                    funding_df = await exchange_client.fetch_funding_rates(symbol, interval, limit=1000)
                    if not funding_df.empty:
                        derivatives_data['funding'] = funding_df
                    
                    # 2. Open Interest
                    oi_df = await exchange_client.fetch_open_interest(symbol, interval, limit=1000)
                    if not oi_df.empty:
                        derivatives_data['oi'] = oi_df
                    
                    # 3. Long/Short Ratio
                    ls_df = await exchange_client.fetch_long_short_ratio(symbol, interval, limit=1000)
                    if not ls_df.empty:
                        derivatives_data['ls_ratio'] = ls_df
                        
                    # 4. Global Long/Short Ratio (NEW)
                    global_ls_df = await exchange_client.fetch_global_long_short_ratio(symbol, interval, limit=1000)
                    if not global_ls_df.empty:
                        derivatives_data['global_ls'] = global_ls_df

                    # 5. Taker Buy/Sell Volume Ratio (NEW)
                    taker_df = await exchange_client.fetch_taker_buy_sell_vol_ratio(symbol, interval, limit=1000)
                    if not taker_df.empty:
                         derivatives_data['taker_ratio'] = taker_df
                        
                except Exception as e:
                    print(f"Warning: Failed to fetch derivatives data for {symbol}: {e}. Continuing with Price only.")
                    # 不因衍生品数据缺失而中断交易，但模型可能会受到影响

                # 特征工程 (注入衍生品数据)
                featured_data = generate_features(raw_data, news_csv_path=None, derivatives_dfs=derivatives_data)
                data_for_pm[symbol] = featured_data

            if not data_for_pm:
                print("Failed to fetch market data for any symbol. Retrying in 60 seconds...")
                # Skip sleep if in mock mode to speed up debugging, unless explicitly waiting
                if not args.mock:
                    await asyncio.sleep(60)
                continue

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
                execution_report = await execution_handler.execute_trades(trade_orders, cycle_logger)
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

            cycle_count += 1
            
            # Calculate sleep time
            if args.mock:
                 print("Mock mode: Skipping sleep.")
            else:
                now = pd.Timestamp.now(tz='UTC')
                seconds_remaining = 3600 - (now.minute * 60 + now.second)
                wait_seconds = seconds_remaining + 5
                print(f"Cycle finished. Waiting for {wait_seconds / 60:.1f} minutes to align with next hour...")
                await asyncio.sleep(wait_seconds)

        except KeyboardInterrupt:
            print("\nUser interrupted the process. Shutting down.")
            cycle_logger.set_error("User interrupted.")
            cycle_logger.commit()
            break
        except Exception as e:
            print(f"FATAL ERROR in main loop: {e}")
            cycle_logger.set_error(str(e))
            if not args.mock:
                await asyncio.sleep(60)
            else:
                break # Break on error in mock mode
        finally:
            cycle_logger.commit()
    
    # 关闭交易所连接
    if 'exchange_client' in locals():
        await exchange_client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Auto Trader.')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode with simulated exchange data.')
    parser.add_argument('--cycles', type=int, default=None, help='Number of cycles to run before exiting (default: infinite).')
    
    args = parser.parse_args()
    
    # FIX: Confirmed necessary for Windows stability (prevents WinError 121 / SSL timeouts)
    if sys.platform == 'win32':
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception as e:
            print(f"Warning: Failed to set WindowsSelectorEventLoopPolicy: {e}")

    try:
        asyncio.run(main_loop(args))
    except KeyboardInterrupt:
        pass