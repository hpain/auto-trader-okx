import sys
print("DEBUG: Script started", file=sys.stderr)
import time
import logging
import os
import json
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
import multiprocessing
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 导入我们需要的模块 ---
from config import config
# from exchange.okx_exchange import OKXExchange # 移除旧的同步实现
from exchange.factory import ExchangeFactory # 使用工厂类
from strategies.lgb_strategy import LGBStrategy
from strategies.strategy_manager import StrategyManager
from analysis.market_regime_detection import detect_regime_change
from analysis.performance_monitor import PerformanceMonitor
from tools.model_pipeline import run_training_pipeline
from features.feature_engineering import generate_features
from trader.risk_monitor import RiskMonitor
from trader.risk_manager import RiskManager
from trader.market_regime_detector import MarketRegimeDetector
from trader.enhanced_monitor import get_enhanced_monitor
from trader.slippage_monitor import get_slippage_monitor
from trader.portfolio_manager import get_portfolio_manager
from models.model_interpretability import get_model_interpretability
from utils.system_monitor import get_system_monitor


class AutonomousTrader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.trader_config = config.get('trader', {})
        self.okx_config = config.get('okx', {})
        self.strategy_config = config.get('strategy', {})
        self.model_update_flag_path = os.path.join(config.get('paths', {}).get('model_dir', 'models'), 'model_update_complete.flag')
        self.training_lock_path = os.path.join(config.get('paths', {}).get('model_dir', 'models'), 'training.lock')
        self.training_state_path = os.path.join(config.get('paths', {}).get('model_dir', 'models'), 'training_state.json')

        print("DEBUG: Starting AutonomousTrader initialization...")
        # 交易所初始化延迟到 initialize_async 方法中
        self.exchange = None

        # Initialize StrategyManager for multiple strategy support
        try:
            print("DEBUG: Initializing StrategyManager...")
            # 从配置中获取策略管理配置
            sm_config = config.get('strategy_management', {})
            self.strategy_manager = StrategyManager(config=sm_config)
            self.strategy = self.strategy_manager.strategies.get('lgb')
            self.logger.info("StrategyManager Initialized with multiple strategies.")
        except Exception as e:
            print(f"DEBUG: StrategyManager initialization failed: {e}")
            self.logger.error(f"Failed to initialize StrategyManager: {e}. Falling back to single LGB strategy.", exc_info=True)
            self.strategy = LGBStrategy(
                strategy_name="LGBStrategy_Autonomous",
                config=self.strategy_config
            )
        try:
            print("DEBUG: Initializing PerformanceMonitor...")
            log_path = self.trader_config.get('performance_log_path', 'logs/trading_cycles.log')
            self.performance_monitor = PerformanceMonitor(log_file_path=log_path)
            self.logger.info("Performance Monitor Initialized.")
        except FileNotFoundError:
            print("DEBUG: PerformanceMonitor initialization failed (FileNotFound).")
            self.logger.warning(f"Performance log file not found. Monitor will not work until logs are generated.")
            self.performance_monitor = None

            self.logger.warning(f"Performance log file not found. Monitor will not work until logs are generated.")
            self.performance_monitor = None
        
        self.last_training_time = self._load_last_training_time()
        
        # Initialize Enhanced Monitor for comprehensive monitoring and alerts
        try:
            self.enhanced_monitor = get_enhanced_monitor(
                db_path=config.get('paths', {}).get('database_path', 'trader_state.db')
            )
            self.logger.info("Enhanced Monitor Initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize EnhancedMonitor: {e}. Continuing without enhanced monitoring.", exc_info=True)
            self.enhanced_monitor = None
            # Fallback to the original risk monitor
            try:
                self.risk_monitor = RiskMonitor(
                    db_path=config.get('paths', {}).get('database_path', 'trader_state.db'),
                    config=config.get('risk_monitoring', {})  # Get risk monitoring config from settings
                )
                self.logger.info("Risk Monitor (fallback) Initialized.")
            except Exception as e2:
                self.logger.error(f"Failed to initialize RiskMonitor: {e2}. Continuing without risk monitoring.", exc_info=True)
                self.risk_monitor = None
        
        # Initialize Slippage Monitor for liquidity and slippage checks
        try:
            self.slippage_monitor = get_slippage_monitor(config)
            self.logger.info("Slippage Monitor Initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SlippageMonitor: {e}. Continuing without slippage monitoring.", exc_info=True)
            self.slippage_monitor = None
        
        # Initialize Portfolio Manager for multi-asset management
        try:
            self.portfolio_manager = get_portfolio_manager(config)
            self.logger.info("Portfolio Manager Initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize PortfolioManager: {e}. Continuing with single asset mode.", exc_info=True)
            self.portfolio_manager = None
        
        # Initialize Model Interpretability for explaining predictions
        try:
            self.model_interpretability = get_model_interpretability(config)
            self.logger.info("Model Interpretability Initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelInterpretability: {e}. Continuing without model interpretability.", exc_info=True)
            self.model_interpretability = None
        
        # Initialize System Monitor for resource monitoring
        try:
            self.system_monitor = get_system_monitor(config)
            # Add alert callback to send alerts through our enhanced monitor
            if hasattr(self, 'enhanced_monitor') and self.enhanced_monitor:
                self.system_monitor.add_alert_callback(self._handle_system_alert)
            self.system_monitor.start_monitoring()
            self.logger.info("System Monitor Initialized and started.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SystemMonitor: {e}. Continuing without system monitoring.", exc_info=True)
            self.system_monitor = None
        
        # Initialize StateManager for tracking positions and orders
        try:
            from trader.state_manager import StateManager
            self.state_manager = StateManager(
                db_path=config.get('paths', {}).get('database_path', 'trader_state.db')
            )
            self.logger.info("StateManager Initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize StateManager: {e}. This is critical for operation.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Failed to initialize StateManager: {e}. This is critical for operation.", exc_info=True)
            raise
        
        # Initialize Risk Manager (Pre-trade checks)
        try:
            self.risk_manager = RiskManager(
                balance=0.0, # Will be updated before checks
                config=config,
                db_path=config.get('paths', {}).get('database_path', 'trader_state.db'),
                state_manager=self.state_manager
            )
            self.logger.info("RiskManager Initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize RiskManager: {e}", exc_info=True)
            self.risk_manager = None
        # Initialize Market Regime Detector
        try:
            self.market_regime_detector = MarketRegimeDetector()
            self.logger.info("Market Regime Detector Initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Market Regime Detector: {e}", exc_info=True)
            self.market_regime_detector = None
        
        # ========== 新增: 初始化新闻监控 (遵循开闭原则) ==========
        try:
            from data.news_monitor import get_news_monitor
            news_config = config.get('news_monitoring', {})
            if news_config.get('enabled', False):
                self.news_monitor = get_news_monitor(config, self.logger)
                self.logger.info("News Monitor Initialized.")
            else:
                self.news_monitor = None
                self.logger.info("News Monitor disabled in config.")
        except Exception as e:
            self.logger.error(f"Failed to initialize News Monitor: {e}. Continuing without news monitoring.", exc_info=True)
            self.news_monitor = None
        
        # ========== 新增: 初始化事件日历 (遵循开闭原则) ==========
        try:
            from data.event_calendar import get_event_calendar
            event_config = config.get('event_calendar', {})
            if event_config.get('enabled', False):
                self.event_calendar = get_event_calendar(event_config, self.logger)
                self.logger.info("Event Calendar Initialized.")
            else:
                self.event_calendar = None
                self.logger.info("Event Calendar disabled in config.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Event Calendar: {e}. Continuing without event calendar.", exc_info=True)
            self.event_calendar = None
        # ========== END 新增 ==========
        
        self.logger.info("Autonomous Trader Initialized Successfully.")

    async def sense(self):
        """
        感知阶段：获取市场数据，计算特征，检测市场状态。
        """
        self.logger.info("--- Stage: SENSE ---")
        
        # ========== 新增: 检查宏观事件日历 ==========
        if hasattr(self, 'event_calendar') and self.event_calendar:
            try:
                upcoming = self.event_calendar.check_upcoming_events(
                    days_ahead=self.config.get('event_calendar', {}).get('check_days_ahead', 3)
                )
                
                if upcoming['has_event']:
                    event = upcoming['event']
                    days = upcoming['days_until']
                    
                    self.logger.warning(f"📅 Upcoming event in {days} days: {event['description']}")
                    self.logger.warning(f"   Impact: {event.get('impact', 'UNKNOWN')}, Action: {event.get('action', 'normal')}")
                    
                    # 根据事件影响调整仓位
                    position_multiplier = self.event_calendar.get_position_multiplier(event)
                    
                    if position_multiplier == 0.0:
                        # 暂停交易
                        self.logger.critical(f"⛔ Pausing trading due to {event['description']}")
                        with open('event_pause.flag', 'w') as f:
                            f.write(f"Event: {event['description']}\n")
                            f.write(f"Date: {event.get('date', 'recurring')}\n")
                            f.write(f"Impact: {event.get('impact')}\n")
                        
                        # 发送告警
                        if hasattr(self, 'enhanced_monitor') and self.enhanced_monitor:
                            if hasattr(self.enhanced_monitor, 'send_alert'):
                                self.enhanced_monitor.send_alert(
                                    level='WARNING',
                                    message=f'Trading paused due to upcoming event: {event["description"]}',
                                    details=event
                                )
                        
                        return None, None, None  # 跳过此周期
                    
                    elif position_multiplier < 1.0:
                        # 降低仓位
                        self.logger.warning(f"🔻 Reducing position to {position_multiplier*100}% due to upcoming event")
                        # 设置仓位调整系数 (在 decide_and_act 中使用)
                        self.event_position_multiplier = position_multiplier
                    else:
                        self.event_position_multiplier = 1.0
                else:
                    self.event_position_multiplier = 1.0
                    
            except Exception as e:
                self.logger.error(f"Event calendar check failed: {e}", exc_info=True)
                self.event_position_multiplier = 1.0
        else:
            self.event_position_multiplier = 1.0
        # ========== END 新增 ==========
        
        # 确定要交易的资产列表
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            symbols = self.config.get('multi_asset', {}).get('symbols', [self.trader_config.get('symbol', 'BTC-USDT')])
        else:
            symbols = [self.trader_config.get('symbol', 'BTC-USDT')]

        all_featured_data = {}
        performance_kpis = {} # Placeholder

        for symbol in symbols:
            try:
                # 获取K线数据
                limit = self.trader_config.get('lookback_period', 100)
                timeframe = self.trader_config.get('timeframe', '1h')
                
                candles = await self.exchange.fetch_candles(symbol, timeframe=timeframe, limit=limit)
                
                # DEBUG: Inspect candles type
                # print(f"DEBUG: Fetched candles for {symbol}, type: {type(candles)}")
                
                is_df = isinstance(candles, pd.DataFrame)
                if not is_df and hasattr(candles, 'columns') and hasattr(candles, 'empty'):
                    is_df = True
                
                if is_df:
                    if candles.empty:
                        self.logger.warning(f"No candles fetched for {symbol}")
                        continue
                    df = candles
                    if 'volume' in df.columns and 'vol' not in df.columns:
                        df = df.rename(columns={'volume': 'vol'})
                    if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                        df = df.reset_index()
                elif not candles:
                    self.logger.warning(f"No candles fetched for {symbol}")
                    continue
                else:
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Ensure timestamp is index for feature engineering
                df = df.set_index('timestamp')
                
                # 特征工程
                featured_df = generate_features(df)
                all_featured_data[symbol] = featured_df
                
            except Exception as e:
                self.logger.error(f"Error in sense stage for {symbol}: {e}", exc_info=True)
        
        # 市场状态检测
        regime_info = {}
        if self.market_regime_detector and symbols:
             # Use the first symbol for regime detection for now
             primary_symbol = symbols[0]
             if primary_symbol in all_featured_data:
                 df = all_featured_data[primary_symbol]
                 regime_info = self.market_regime_detector.detect_regime(df)

        return all_featured_data, regime_info, performance_kpis

    async def initialize_async(self):
        """异步初始化交易所连接"""
        print("DEBUG: Entering initialize_async()...")
        # 优先使用环境变量 (Docker兼容)，否则使用配置文件
        api_key = os.environ.get('OKX_API_KEY') or self.okx_config.get('api_key')
        secret_key = os.environ.get('OKX_SECRET_KEY') or self.okx_config.get('secret_key')
        passphrase = os.environ.get('OKX_PASSPHRASE') or self.okx_config.get('passphrase')
        # 环境变量 OKX_FLAG=1 为实盘，0 为模拟盘
        env_flag = os.environ.get('OKX_FLAG')
        if env_flag is not None:
            sandbox = str(env_flag) != "1"
        else:
            sandbox = self.okx_config.get('sandbox', True)

        # 检查是否启用 Mock 模式
        mock_mode = os.environ.get('MOCK_MODE', '0') == '1'
        if not (api_key and secret_key and passphrase) and not mock_mode:
            self.logger.warning("Missing API credentials. Switching to MOCK MODE.")
            mock_mode = True

        if mock_mode:
            self.logger.warning("!!! RUNNING IN MOCK MODE - NO REAL TRADES WILL BE EXECUTED !!!")

        print(f"DEBUG: Creating exchange (mock={mock_mode})...")
        # 使用工厂类创建异步交易所实例
        self.exchange = await ExchangeFactory.create_exchange(
            'okx',
            market_type='swap', # 假设主要交易永续合约，根据需要调整
            api_key=api_key,
            api_secret=secret_key,
            passphrase=passphrase,
            sandbox=sandbox,
            mock=mock_mode
        )
        print("DEBUG: Exchange created.")
        # 加载市场信息
        if hasattr(self.exchange, 'load'):
            print("DEBUG: Loading exchange markets...")
            await self.exchange.load()
            print("DEBUG: Exchange markets loaded.")
            
        self.logger.info("Async Exchange Initialized.")

    async def decide_and_act(self, all_featured_data, regime_info):
        """
        决策与执行阶段：根据感知到的数据生成信号并执行交易。
        """
        self.logger.info("--- Stage: DECIDE & ACT ---")
        
        # ========== BUG FIX 1: 强制执行每日止损 ==========
        if hasattr(self, 'enhanced_monitor') and self.enhanced_monitor:
            try:
                # 获取今日盈亏
                daily_pnl = self.enhanced_monitor.get_daily_pnl()
                daily_loss_threshold = self.config.get('risk_monitoring', {}).get('daily_loss_threshold', -500.0)
                
                if daily_pnl < daily_loss_threshold:
                    self.logger.critical(f"🚨 DAILY LOSS LIMIT REACHED! Daily PnL: ${daily_pnl:.2f}, Threshold: ${daily_loss_threshold:.2f}")
                    self.logger.critical("⛔ STOPPING ALL TRADING FOR TODAY")
                    
                    # 发送紧急告警
                    if hasattr(self.enhanced_monitor, 'send_alert'):
                        self.enhanced_monitor.send_alert(
                            level='CRITICAL',
                            message=f'Daily loss limit reached: ${daily_pnl:.2f}. Trading stopped.',
                            details={'daily_pnl': daily_pnl, 'threshold': daily_loss_threshold}
                        )
                    
                    # 创建紧急停止标志文件
                    emergency_flag = 'emergency_stop.flag'
                    with open(emergency_flag, 'w') as f:
                        f.write(f'Daily loss limit reached at {datetime.utcnow().isoformat()}\n')
                        f.write(f'Daily PnL: ${daily_pnl:.2f}\n')
                    
                    return  # 立即返回,不执行任何交易
                    
            except Exception as e:
                self.logger.error(f"Failed to check daily loss limit: {e}", exc_info=True)
        # ========== END BUG FIX 1 ==========
        
        # ========== 新增: 实时新闻监控检查 ==========
        if hasattr(self, 'news_monitor') and self.news_monitor:
            try:
                # 获取最新新闻
                news_config = self.config.get('news_monitoring', {})
                currencies = news_config.get('currencies', ['BTC', 'ETH'])
                
                news = self.news_monitor.fetch_latest_news(currencies=currencies, limit=20)
                impact = self.news_monitor.analyze_news_impact(
                    news,
                    recent_minutes=news_config.get('recent_news_window', 5)
                )
                
                if impact['action'] == 'emergency_stop':
                    self.logger.critical(f"🚨 CRITICAL NEWS DETECTED!")
                    self.logger.critical(f"📰 {impact['news'][0]['title']}")
                    self.logger.critical(f"🔑 Keyword: {impact['news'][0]['keyword']}")
                    self.logger.critical(f"⛔ STOPPING TRADING")
                    
                    # 创建紧急停止标志
                    with open('emergency_stop.flag', 'w') as f:
                        f.write(f"Critical news at {datetime.utcnow().isoformat()}\n")
                        f.write(f"Reason: {impact['reason']}\n")
                        f.write(f"News: {impact['news'][0]['title']}\n")
                        f.write(f"URL: {impact['news'][0]['url']}\n")
                    
                    # 发送告警
                    if hasattr(self, 'enhanced_monitor') and self.enhanced_monitor:
                        if hasattr(self.enhanced_monitor, 'send_alert'):
                            self.enhanced_monitor.send_alert(
                                level='CRITICAL',
                                message=f'Critical news: {impact["news"][0]["title"]}',
                                details=impact
                            )
                    
                    return  # 立即停止
                
                elif impact['action'] == 'reduce_position':
                    self.logger.warning(f"⚠️ WARNING NEWS DETECTED")
                    self.logger.warning(f"📰 {len(impact['news'])} warning news in last {news_config.get('recent_news_window', 5)} minutes")
                    for news_item in impact['news'][:3]:  # 只显示前3条
                        self.logger.warning(f"   - {news_item['title']}")
                    
                    # 降低仓位
                    news_position_multiplier = news_config.get('warning_position_multiplier', 0.5)
                    self.logger.warning(f"🔻 Reducing position to {news_position_multiplier*100}% due to news")
                    self.news_position_multiplier = news_position_multiplier
                else:
                    self.news_position_multiplier = 1.0
                    
            except Exception as e:
                self.logger.error(f"News monitoring failed: {e}", exc_info=True)
                self.news_position_multiplier = 1.0
        else:
            self.news_position_multiplier = 1.0
        # ========== END 新增 ==========

        if all_featured_data is None:
            self.logger.warning("No data available for decision making.")
            return

        # 获取主要交易对
        primary_symbol = self.trader_config.get('symbol', 'BTC-USDT')
        
        # 确定要交易的资产列表
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            symbols = self.config.get('multi_asset', {}).get('symbols', [primary_symbol])
        else:
            symbols = [primary_symbol]
            
        all_signals = {}
        main_signal = 0
        
        # 为主要资产生成信号（兼容旧逻辑）
        if primary_symbol in all_featured_data:
            primary_featured_data = all_featured_data[primary_symbol]
            
            # 使用策略生成信号
            try:
                # 确保数据包含所需的特征
                if primary_featured_data.empty:
                    self.logger.warning(f"Featured data for {primary_symbol} is empty. Skipping strategy execution.")
                    return

                # 准备预测数据 (取最后一行)
                latest_data = primary_featured_data.iloc[[-1]].copy()
                
                # 生成信号
                main_signal = self.strategy.generate_signal(latest_data)
                self.logger.info(f"Strategy generated signal for {primary_symbol}: {main_signal}")
                
                # 模型可解释性分析
                try:
                    # 获取模型路径
                    model_dir = self.config.get('paths', {}).get('model_dir', 'models')
                    model_path = os.path.join(model_dir, 'best_model.pkl')
                    
                    if os.path.exists(model_path):
                        interpretation = self.model_interpretability.integrate_with_trading_system(
                            model_path=model_path,
                            X_current=primary_featured_data.tail(1),  # 使用最新的数据点
                            model_type="lightgbm"
                        )
                        
                        if interpretation:
                            self.logger.info(f"Model interpretation completed:")
                            self.logger.info(f"  Most positive contributor: {interpretation['prediction_explanation']['most_positive_contributor']}")
                            self.logger.info(f"  Most negative contributor: {interpretation['prediction_explanation']['most_negative_contributor']}")
                            
                            # 可以将解释信息保存到日志中供后续分析
                            interpretation_log_path = 'logs/model_interpretation.log'
                            os.makedirs(os.path.dirname(interpretation_log_path), exist_ok=True)
                            with open(interpretation_log_path, 'a') as f:
                                f.write(json.dumps({
                                    'timestamp': interpretation['timestamp'],
                                    'top_feature_contributions': interpretation['top_feature_contributions'][:5]  # 只记录前5个贡献
                                }) + '\n')
                    else:
                        self.logger.warning("Model file not found for interpretability analysis.")
                except Exception as e:
                    self.logger.error(f"Error in model interpretability: {e}", exc_info=True)

            except Exception as e:
                self.logger.error(f"Error in strategy execution: {e}", exc_info=True)
                return
        
        # 如果有投资组合管理器，执行投资组合级别的决策
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            # 获取所有相关资产的信号（这里简化处理，实际应用中可能需要为每个资产单独生成信号）
            for symbol in symbols:
                if symbol in all_featured_data:
                    # 对于每个资产，暂时使用主信号，实际应用中可能需要为每个资产单独生成信号
                    all_signals[symbol] = main_signal
            
            # 检查是否需要再平衡
            if self.portfolio_manager.should_rebalance():
                self.logger.info("Portfolio rebalancing required.")
                
                # 获取当前价格
                current_prices = {}
                for symbol in symbols:
                    try:
                        current_prices[symbol] = await self.exchange.get_current_price(symbol)
                    except Exception as e:
                        self.logger.error(f"Failed to get current price for {symbol}: {e}")
                
                # 计算再平衡订单
                rebalance_orders = self.portfolio_manager.calculate_rebalance_orders(
                    current_prices, 
                    await self._get_total_portfolio_value()
                )
                
                # 执行再平衡订单
                for order in rebalance_orders:
                    await self._execute_single_order(order['symbol'], order['side'], order['quantity'], 
                                             current_prices[order['symbol']])
                
                # 更新再平衡时间
                self.portfolio_manager.last_rebalance_time = datetime.utcnow()
            else:
                # 根据信号和投资组合状态决定交易哪些资产
                active_symbols = self.portfolio_manager.get_optimal_symbols_to_trade(
                    all_signals, 
                    {s: all_featured_data[s]['close'].iloc[-1] for s in symbols if s in all_featured_data}
                )
        else:
            # 单资产模式，只交易主要资产
            active_symbols = [primary_symbol] if main_signal != 0 else []
        
        # 执行交易
        for symbol in active_symbols:
            if symbol not in all_featured_data:
                continue
                
            current_price = all_featured_data[symbol]['close'].iloc[-1]
            signal = all_signals.get(symbol, main_signal)  # 如果没有特定信号，使用主信号
            
            # 获取当前持仓
            base_currency, quote_currency = symbol.split('-')
            try:
                base_balance = await self.exchange.get_balance(currency=base_currency)
                quote_balance = await self.exchange.get_balance(currency=quote_currency)
                self.logger.info(f"Current balance for {symbol}: {base_balance:.4f} {base_currency}, {quote_balance:.2f} {quote_currency}")
            except Exception as e:
                self.logger.error(f"Failed to get balance for {symbol}: {e}", exc_info=True)
                continue
            
            # 检查滑点和流动性条件
            if hasattr(self, 'slippage_monitor') and self.slippage_monitor:
                should_restrict, restriction_reason = self.slippage_monitor.should_restrict_trading(symbol, self.exchange)
                if should_restrict:
                    self.logger.warning(f"Trading restricted for {symbol}: {restriction_reason}")
                    continue  # 如果应该限制交易，则跳过此资产

            trade_amount_quote = self.trader_config.get('trade_amount_quote', 100)
            
            # ========== BUG FIX 3: 使用价值而非数量判断仓位 ==========
            position_value = base_balance * current_price
            min_position_value = 10  # 最小持仓价值 $10
            # ========== END BUG FIX 3 ==========
            
            # --- 风险管理：计算止损和止盈 ---
            atr = None
            if 'ATR_14' in all_featured_data[symbol].columns:
                atr = all_featured_data[symbol]['ATR_14'].iloc[-1]
            
            # 更新 RiskManager 余额
            if self.risk_manager:
                self.risk_manager.balance = quote_balance

            if signal == 1 and position_value < min_position_value:
                self.logger.info(f"BUY signal for {symbol} and no significant position held (value: ${position_value:.2f}). Executing BUY order.")
                
                try:
                    quantity = trade_amount_quote / current_price
                    
                    # 风险检查
                    if self.risk_manager:
                        is_approved, reason = self.risk_manager.assess_trade(
                            proposed_quantity=quantity,
                            current_price=current_price,
                            symbol=symbol
                        )
                        if not is_approved:
                            self.logger.warning(f"Risk check failed for {symbol}: {reason}")
                            continue
                    
                    # 计算止损止盈
                    stop_loss_price, take_profit_price = self.risk_manager.calculate_sl_tp(
                        entry_price=current_price,
                        side='buy',
                        volatility=atr
                    )
                    
                    self.logger.info(f"Preparing Buy Order: Price={current_price}, SL={stop_loss_price}, TP={take_profit_price}")

                    # 尝试 OCO 订单
                    order_result = None
                    if hasattr(self.exchange, 'place_oco_order'):
                        try:
                            order_result = await self.exchange.place_oco_order(
                                symbol=symbol,
                                side='buy',
                                amount=quantity,
                                take_profit_price=take_profit_price,
                                stop_loss_price=stop_loss_price
                            )
                        except Exception as oco_e:
                            self.logger.warning(f"OCO order failed: {oco_e}. Falling back to Limit Order.")
                    
                    # 如果 OCO 失败或不支持，使用限价单 (BUG FIX 2 logic)
                    if not order_result:
                        # 计算限价 (允许 0.2% 滑点)
                        limit_price = current_price * 1.002
                        expected_buy_price = limit_price
                        
                        self.logger.info(f"Placing LIMIT BUY order: {quantity:.6f} {base_currency} @ ${limit_price:.2f}")
                        
                        order_result = await self.exchange.create_order(
                            symbol=symbol,
                            order_type='limit',
                            side='buy',
                            amount=quantity,
                            price=limit_price
                        )
                    
                    # ========== BUG FIX 4: 添加订单成功检查 ==========
                    if not order_result or (isinstance(order_result, dict) and order_result.get('code') and order_result.get('code') != '0'):
                        self.logger.error(f"❌ BUY order FAILED for {symbol}: {order_result}")
                        continue
                    
                    # 适配不同的返回格式
                    order_id = None
                    executed_buy_price = current_price
                    
                    if isinstance(order_result, dict):
                        if 'data' in order_result and order_result['data']:
                            order_id = order_result['data'][0]['ordId']
                        elif 'id' in order_result:
                            order_id = order_result['id']
                            
                    if order_id:
                        self.logger.info(f"✅ BUY order placed successfully. Order ID: {order_id}")
                        
                        # 记录订单到状态管理器
                        # 检查滑点
                        if hasattr(self, 'slippage_monitor') and self.slippage_monitor:
                            slippage_result = self.slippage_monitor.monitor_trade_execution(
                                symbol=symbol,
                                side='buy',
                                quantity=quantity,
                                expected_price=current_price,
                                executed_price=executed_buy_price,
                                exchange_interface=self.exchange
                            )
                            if slippage_result['is_excessive_slippage']:
                                self.logger.warning(f"Excessive slippage in BUY order for {symbol}: {slippage_result['warning']}")
                        
                        # 使用StateManager的集成TradeTracker记录交易进入
                        self.state_manager.trade_tracker.record_trade_entry(
                            trade_id=order_id,
                            symbol=symbol,
                            side='buy',
                            entry_price=executed_buy_price,
                            quantity=quantity
                        )
                        
                        # 更新持仓
                        self.state_manager.update_position(symbol, quantity, executed_buy_price)
                        
                        # 更新投资组合管理器中的持仓
                        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                            self.portfolio_manager.update_position(symbol, quantity, executed_buy_price)
                            
                            except Exception as e:
                                self.logger.error(f"Failed to execute BUY order for {symbol}: {e}", exc_info=True)
                            elif signal == -1 and position_value > min_position_value:  # 修复: 只有明确的卖出信号(-1)才卖出
                                self.logger.info(f"SELL signal (-1) for {symbol} and position held (value: ${position_value:.2f}). Executing SELL order.")                
                try:
                    # ========== 使用限价单 ==========
                    # 计算限价 (允许 0.2% 滑点,卖出时价格略低)
                    limit_price = current_price * 0.998
                    expected_sell_price = limit_price
                    
                    self.logger.info(f"Placing LIMIT SELL order: {base_balance:.6f} {base_currency} @ ${limit_price:.2f}")
                    
                    order_result = await self.exchange.create_order(
                        symbol=symbol,
                        order_type='limit',
                        side='sell',
                        amount=base_balance,
                        price=limit_price
                    )
                    
                    # 检查订单是否成功
                    if not order_result or (isinstance(order_result, dict) and order_result.get('code') and order_result.get('code') != '0'):
                        self.logger.error(f"❌ SELL order FAILED for {symbol}: {order_result}")
                        continue
                    
                    # 适配不同的返回格式
                    order_id = None
                    executed_sell_price = limit_price
                    
                    if isinstance(order_result, dict):
                        if 'data' in order_result and order_result['data']:
                            order_id = order_result['data'][0]['ordId']
                        elif 'id' in order_result:
                            order_id = order_result['id']

                    if order_id:
                        self.logger.info(f"✅ SELL order placed successfully. Order ID: {order_id}")
                        
                        # 检查滑点
                        if hasattr(self, 'slippage_monitor') and self.slippage_monitor:
                            slippage_result = self.slippage_monitor.monitor_trade_execution(
                                symbol=symbol,
                                side='sell',
                                quantity=base_balance,
                                expected_price=expected_sell_price,
                                executed_price=executed_sell_price,
                                exchange_interface=self.exchange
                            )
                            if slippage_result['is_excessive_slippage']:
                                self.logger.warning(f"Excessive slippage in SELL order for {symbol}: {slippage_result['warning']}")
                        
                        # 使用StateManager的集成TradeTracker记录交易退出
                        self.state_manager.trade_tracker.record_trade_exit(
                            trade_id=order_id,
                            exit_price=executed_sell_price
                        )
                        
                        # 更新持仓（清空持仓）
                        self.state_manager.update_position(symbol, 0, 0)
                        
                        # 更新投资组合管理器中的持仓
                        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                            self.portfolio_manager.update_position(symbol, 0, 0)
                            
                except Exception as e:
                    self.logger.error(f"Failed to execute SELL order for {symbol}: {e}", exc_info=True)
            else:
                self.logger.info(f"No action needed for {symbol}. Signal: {signal} (0=Hold), Position Value: ${position_value:.2f}")

    async def _execute_single_order(self, symbol: str, side: str, quantity: float, price: float):
        """
        执行单个订单（用于再平衡）
        """
        # 检查滑点和流动性条件
        if hasattr(self, 'slippage_monitor') and self.slippage_monitor:
            should_restrict, restriction_reason = self.slippage_monitor.should_restrict_trading(symbol, self.exchange)
            if should_restrict:
                self.logger.warning(f"Trading restricted for {symbol}: {restriction_reason}")
                return  # 如果应该限制交易，则不做任何操作

        expected_price = price
        try:
            if side == 'buy':
                order_result = await self.exchange.create_order(
                    symbol=symbol,
                    order_type='market',
                    side='buy',
                    amount=quantity
                )
            elif side == 'sell':
                order_result = await self.exchange.create_order(
                    symbol=symbol,
                    order_type='market',
                    side='sell',
                    amount=quantity
                )
            else:
                self.logger.error(f"Invalid order side: {side}")
                return
                
            self.logger.info(f"{side.upper()} order for {symbol} executed. Result: {order_result}")
            
            # 记录订单到状态管理器
            if order_result and 'id' in order_result:
                order_id = order_result['id']
                executed_price = price  # 对于市价单，使用传入的价格
                
                # 检查滑点
                if hasattr(self, 'slippage_monitor') and self.slippage_monitor:
                    slippage_result = self.slippage_monitor.monitor_trade_execution(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        expected_price=expected_price,
                        executed_price=executed_price,
                        exchange_interface=self.exchange
                    )
                    if slippage_result['is_excessive_slippage']:
                        self.logger.warning(f"Excessive slippage in {side.upper()} order for {symbol}: {slippage_result['warning']}")
                    elif not slippage_result['liquidity_ok']:
                        self.logger.warning(f"Liquidity concern in {side.upper()} order for {symbol}: {slippage_result['warning']}")
                
                order_data = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': executed_price,
                    'status': 'filled'
                }
                
                if side == 'buy':
                    # 使用StateManager的集成TradeTracker记录交易进入
                    self.state_manager.trade_tracker.record_trade_entry(
                        trade_id=order_id,
                        symbol=symbol,
                        side='buy',
                        entry_price=executed_price,
                        quantity=quantity
                    )
                else:  # side == 'sell'
                    # 使用StateManager的集成TradeTracker记录交易退出
                    self.state_manager.trade_tracker.record_trade_exit(
                        trade_id=order_id,
                        exit_price=executed_price
                    )
                
                # 更新持仓
                if side == 'buy':
                    self.state_manager.update_position(symbol, quantity, executed_price)
                else:  # side == 'sell'
                    # 对于卖出，需要获取当前持仓数量并减去卖出数量，这里简化处理为清空整个持仓
                    self.state_manager.update_position(symbol, 0, 0)
                
                # 更新投资组合管理器中的持仓
                if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                    if side == 'buy':
                        self.portfolio_manager.update_position(symbol, quantity, executed_price)
                    else:  # side == 'sell'
                        self.portfolio_manager.update_position(symbol, 0, 0)
                        
        except Exception as e:
            self.logger.error(f"Failed to execute {side} order for {symbol}: {e}", exc_info=True)

    async def _get_total_portfolio_value(self) -> float:
        """
        获取投资组合总价值
        """
        total_value = 0.0
        
        # 确定要交易的资产列表
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            symbols = self.config.get('multi_asset', {}).get('symbols', [self.trader_config.get('symbol', 'BTC-USDT')])
        else:
            symbols = [self.trader_config.get('symbol', 'BTC-USDT')]
        
        for symbol in symbols:
            base_currency, quote_currency = symbol.split('-')
            try:
                # 获取资产余额
                base_balance = await self.exchange.get_balance(currency=base_currency)
                quote_balance = await self.exchange.get_balance(currency=quote_currency)
                
                # 获取当前价格
                current_price = await self.exchange.get_current_price(symbol)
                
                # 计算该资产的价值
                asset_value = quote_balance + base_balance * current_price
                total_value += asset_value
                
            except Exception as e:
                self.logger.error(f"Failed to calculate value for {symbol}: {e}")
                # 如果无法获取某资产价值，跳过（或可以使用其他估值方法）
                continue
        
        return total_value

    def learn(self, performance_kpis, regime_info):
        """
        学习阶段：评估表现和市场状态，决定是否启动再训练模型子进程。
        """
        self.logger.info("--- Stage: LEARN ---")
        
        # 检查持久化的训练锁
        if os.path.exists(self.training_lock_path):
            # 可以在这里添加更复杂的逻辑，比如检查锁文件的时间戳，如果过旧则认为进程已死
            self.logger.info(f"Training lock file found at {self.training_lock_path}. Assuming training is in progress. Skipping.")
            return

        retrain_needed = False
        reasons = []

        # 0. 检查定期训练触发器
        retraining_interval_days = self.trader_config.get('retraining_interval_days', 7)
        if self.last_training_time:
            time_since_last_training = datetime.utcnow() - self.last_training_time
            if time_since_last_training.days >= retraining_interval_days:
                retrain_needed = True
                reasons.append(f"Scheduled retraining triggered (last training was {time_since_last_training.days} days ago).")
        else:
            # 如果从未训练过，则立即触发一次
            retrain_needed = True
            reasons.append("Initial training triggered (no previous training record).")

        # 1. 评估性能指标
        if performance_kpis and performance_kpis.get('status') == 'SUCCESS':
            kpi_thresholds = self.trader_config.get('kpi_thresholds', {})
            sharpe_key = [k for k in performance_kpis.keys() if 'sharpe_ratio' in k][0]
            latest_sharpe = performance_kpis.get(sharpe_key)
            sharpe_threshold = kpi_thresholds.get('min_sharpe_ratio', 0.5)

            if latest_sharpe is not None and latest_sharpe < sharpe_threshold:
                retrain_needed = True
                reasons.append(f"Sharpe ratio ({latest_sharpe:.2f}) dropped below threshold ({sharpe_threshold}).")
        
        # 2. 评估市场状态
        if regime_info and regime_info.get('regime_changed', False):
            retrain_needed = True
            reasons.append(f"Market regime change detected: {regime_info.get('reason')}")
        
        # 3. 检查投资组合风险（如果启用多资产）
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            is_over_risk, risk_warnings = self.portfolio_manager.check_risk_limits()
            if is_over_risk:
                retrain_needed = True
                reasons.extend([f"Portfolio risk limit exceeded: {warning}" for warning in risk_warnings])

        # 3. 执行决策
        if retrain_needed:
            self.logger.warning(f"Retraining triggered! Reasons: {reasons}")
            try:
                # 使用子进程异步运行训练流水线
                training_process = multiprocessing.Process(target=run_training_pipeline, kwargs={'n_trials': 50})
                training_process.daemon = True # 设置为守护进程，主进程退出时子进程也会退出
                training_process.start()
                self.last_training_time = datetime.utcnow()
                self._save_last_training_time()
                self.logger.info(f"Started training pipeline in a separate process (PID: {training_process.pid}).")
            except Exception as e:
                self.logger.error(f"Failed to start training pipeline process: {e}", exc_info=True)
        else:
            self.logger.info("No retraining needed. Performance and market regime are stable.")

    async def log_portfolio_status(self):
        """记录当前投资组合的总价值到性能日志。"""
        self.logger.info("Logging portfolio status...")
        try:
            # 确定要交易的资产列表
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                symbols = self.config.get('multi_asset', {}).get('symbols', [self.trader_config.get('symbol', 'BTC-USDT')])
            else:
                symbols = [self.trader_config.get('symbol', 'BTC-USDT')]
            
            portfolio_details = {}
            total_value = 0.0
            
            for symbol in symbols:
                base_currency, quote_currency = symbol.split('-')

                base_balance = await self.exchange.get_balance(currency=base_currency)
                quote_balance = await self.exchange.get_balance(currency=quote_currency)
                current_price = await self.exchange.get_current_price(symbol)

                if current_price > 0:
                    asset_value = quote_balance + base_balance * current_price
                    total_value += asset_value
                    
                    portfolio_details[symbol] = {
                        "base_currency": base_currency,
                        "base_balance": base_balance,
                        "quote_currency": quote_currency,
                        "quote_balance": quote_balance,
                        "current_price": current_price,
                        "asset_value": asset_value
                    }
                else:
                    self.logger.warning(f"Could not get current price for {symbol}, skipping in portfolio calculation.")

            self.logger.info(f"Total portfolio value: {total_value:.2f} {quote_currency}")

            log_entry = {
                "cycle_id": datetime.utcnow().isoformat(),
                "portfolio": {
                    "total_value": total_value,
                    "assets": portfolio_details
                }
            }
            
            # 添加投资组合管理器的额外信息（如果存在）
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                log_entry["portfolio"]["portfolio_manager_info"] = {
                    "allocation_strategy": portfolio_summary['allocation_strategy'],
                    "last_rebalance_time": portfolio_summary['last_rebalance_time'],
                    "should_rebalance": portfolio_summary['should_rebalance'],
                    "asset_allocations": {symbol: details['weight'] for symbol, details in portfolio_summary['assets'].items()}
                }
            
            log_path = self.trader_config.get('performance_log_path', 'logs/trading_cycles.log')
            # 确保日志目录存在
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            self.logger.error(f"Failed to log portfolio status: {e}", exc_info=True)

    def _check_and_reload_model(self):
        """检查模型更新标志，如果存在，则热重载模型。"""
        # 检查模型是否已更新
        if os.path.exists(self.model_update_flag_path):
            self.logger.info("Model update flag found. A new model has been deployed.")
            self.logger.info("Reloading strategy to activate the new model...")
            try:
                self.strategy = LGBStrategy(
                    strategy_name="LGBStrategy_Autonomous",
                    config=self.strategy_config
                )
                os.remove(self.model_update_flag_path) # 删除标志文件
                self.logger.info("Strategy reloaded and update flag removed. System is now using the new model.")
            except Exception as e:
                self.logger.error(f"Failed to reload strategy after model update: {e}", exc_info=True)
        
        # 检查训练锁是否“陈旧”（意味着训练进程可能已崩溃）
        if os.path.exists(self.training_lock_path):
            try:
                with open(self.training_lock_path, 'r') as f:
                    lock_data = json.load(f)
                pid = lock_data.get('pid')
                # 检查具有该PID的进程是否存在。这在Linux/macOS上更可靠。
                # 在Windows上，我们可以检查锁文件的时间戳。
                lock_time = datetime.fromisoformat(lock_data.get('start_time'))
                if (datetime.now() - lock_time).total_seconds() > 3600 * 3: # 例如，如果训练超过3小时
                    self.logger.warning(f"Stale training lock file found (PID: {pid}, started at {lock_time}). Assuming training process crashed. Removing lock.")
                    os.remove(self.training_lock_path)
            except Exception as e:
                self.logger.error(f"Error checking stale training lock: {e}. Removing lock as a precaution.", exc_info=True)
                os.remove(self.training_lock_path)

    def _load_last_training_time(self) -> Optional[datetime]:
        """从文件加载上次训练时间。"""
        if os.path.exists(self.training_state_path):
            try:
                with open(self.training_state_path, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data.get('last_training_time'))
            except Exception as e:
                self.logger.error(f"Failed to load last training time: {e}")
        return None

    def _save_last_training_time(self):
        """保存上次训练时间到文件。"""
        try:
            with open(self.training_state_path, 'w') as f:
                json.dump({'last_training_time': self.last_training_time.isoformat()}, f)
        except Exception as e:
            self.logger.error(f"Failed to save last training time: {e}")

    def _handle_system_alert(self, alert_data: dict):
        """处理系统监控告警"""
        if hasattr(self, 'enhanced_monitor') and self.enhanced_monitor:
            self.enhanced_monitor.record_alert(
                alert_type='SYSTEM_RESOURCE',
                message=f"{alert_data['metric']} usage high: {alert_data['value']}%",
                severity='WARNING'
            )

    async def run(self):
        """
        启动自主交易循环。
        """
        self.logger.info("Starting Autonomous Trader Loop...")
        
        # 异步初始化
        await self.initialize_async()
        
        while True:
            try:
                # ========== BUG FIX 5: 紧急停止开关 ==========
                emergency_flag = 'emergency_stop.flag'
                if os.path.exists(emergency_flag):
                    self.logger.critical("🚨 EMERGENCY STOP FLAG DETECTED!")
                    self.logger.critical("⛔ STOPPING ALL TRADING IMMEDIATELY")
                    
                    # 读取停止原因
                    try:
                        with open(emergency_flag, 'r') as f:
                            reason = f.read()
                        self.logger.critical(f"Stop reason:\n{reason}")
                    except:
                        pass
                    
                    # 发送紧急告警
                    if hasattr(self, 'enhanced_monitor') and self.enhanced_monitor:
                        if hasattr(self.enhanced_monitor, 'send_alert'):
                            self.enhanced_monitor.send_alert(
                                level='CRITICAL',
                                message='Emergency stop activated. Trading halted.',
                                details={'flag_file': emergency_flag}
                            )
                    
                    self.logger.critical("To resume trading, delete the emergency_stop.flag file")
                    break  # 退出主循环
                # ========== END BUG FIX 5 ==========
                
                # Circuit Breaker Check
                if hasattr(self.exchange, 'check_circuit_breaker'):
                    if not self.exchange.check_circuit_breaker():
                        self.logger.warning("Circuit breaker active. Skipping cycle.")
                        await asyncio.sleep(10)
                        continue

                # 0. 检查并热重载模型
                self._check_and_reload_model()
                
                # 1. Sense
                print("DEBUG: Entering SENSE stage...")
                all_featured_data, regime_info, performance_kpis = await self.sense()
                
                if all_featured_data is not None:
                    # 2. Decide & Act
                    print("DEBUG: Entering DECIDE & ACT stage...")
                    await self.decide_and_act(all_featured_data, regime_info)
                    
                    # 3. Learn (异步触发训练，不阻塞主循环)
                    print("DEBUG: Entering LEARN stage...")
                    self.learn(performance_kpis, regime_info)
                    
                    # 4. Log Status
                    print("DEBUG: Entering LOG STATUS stage...")
                    await self.log_portfolio_status()
                
                # Sleep for the interval
                # 简单起见，这里固定休眠，实际应用可能需要更精确的定时
                self.logger.info("Cycle completed. Sleeping for 60 seconds...")
                print("DEBUG: Cycle completed. Sleeping...")
                await asyncio.sleep(60)

            except KeyboardInterrupt:
                self.logger.info("Stopping Autonomous Trader...")
                break
            except Exception as e:
                self.logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
                await asyncio.sleep(60) # 出错后等待一段时间再重试
        
        # 关闭交易所连接
        if self.exchange:
            await self.exchange.close()

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/autonomous_trader.log"),
            logging.StreamHandler()
        ]
    )
    
    trader = AutonomousTrader()
    asyncio.run(trader.run())
