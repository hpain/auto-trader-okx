import time
import logging
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import multiprocessing

# --- 导入我们需要的模块 ---
from config import config
from exchange.okx_exchange import OKXExchange
from strategies.lgb_strategy import LGBStrategy
from strategies.strategy_manager import StrategyManager
from analysis.market_regime_detection import detect_regime_change
from analysis.performance_monitor import PerformanceMonitor
from tools.model_pipeline import run_training_pipeline
from features.feature_engineering import generate_features
from trader.risk_monitor import RiskMonitor
from trader.market_regime_detector import MarketRegimeDetector
from trader.enhanced_monitor import get_enhanced_monitor
from trader.slippage_monitor import get_slippage_monitor
from trader.portfolio_manager import get_portfolio_manager
from models.model_interpretability import get_model_interpretability
from utils.system_monitor import get_system_monitor


class AutonomousTrader:
    def __init__(self):
        # ... (之前的 __init__ 代码保持不变) ...
        self.logger = logging.getLogger(__name__)
        self.trader_config = config.get('trader', {})
        self.okx_config = config.get('okx', {})
        self.strategy_config = config.get('strategy', {})
        self.model_update_flag_path = os.path.join(config.get('paths', {}).get('model_dir', 'models'), 'model_update_complete.flag')
        self.training_lock_path = os.path.join(config.get('paths', {}).get('model_dir', 'models'), 'training.lock')
        self.training_state_path = os.path.join(config.get('paths', {}).get('model_dir', 'models'), 'training_state.json')

        self.exchange = OKXExchange(
            api_key=self.okx_config.get('api_key'),
            api_secret=self.okx_config.get('secret_key'),
            passphrase=self.okx_config.get('passphrase'),
            sandbox=self.okx_config.get('sandbox', True)
        )
        self.logger.info("OKX Exchange Initialized.")
        # Initialize StrategyManager for multiple strategy support
        try:
            # 从配置中获取策略管理配置
            sm_config = config.get('strategy_management', {})
            self.strategy_manager = StrategyManager(config=sm_config)
            self.strategy = self.strategy_manager.strategies.get('lgb')
            self.logger.info("StrategyManager Initialized with multiple strategies.")
        except Exception as e:
            self.logger.error(f"Failed to initialize StrategyManager: {e}. Falling back to single LGB strategy.", exc_info=True)
            self.strategy = LGBStrategy(
                strategy_name="LGBStrategy_Autonomous",
                config=self.strategy_config
            )
        try:
            log_path = self.trader_config.get('performance_log_path', 'logs/trading_cycles.log')
            self.performance_monitor = PerformanceMonitor(log_file_path=log_path)
            self.logger.info("Performance Monitor Initialized.")
        except FileNotFoundError:
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
            raise
        
        # Initialize Market Regime Detector
        try:
            self.market_regime_detector = MarketRegimeDetector()
            self.logger.info("Market Regime Detector Initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Market Regime Detector: {e}", exc_info=True)
            self.market_regime_detector = None
        
        self.logger.info("Autonomous Trader Initialized Successfully.")

    def sense(self):
        # ... (更新后的 sense 代码，支持多资产) ...
        self.logger.info("--- Stage: SENSE ---")
        
        # 确定要交易的资产列表
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            symbols = self.config.get('multi_asset', {}).get('symbols', [self.trader_config.get('symbol', 'BTC-USDT')])
        else:
            symbols = [self.trader_config.get('symbol', 'BTC-USDT')]
        
        # 获取市场数据
        all_market_data = {}
        interval = self.trader_config.get('interval', '1H')
        
        for symbol in symbols:
            self.logger.info(f"Fetching latest market data for {symbol}...")
            try:
                market_data = self.exchange.fetch_candles(symbol=symbol, timeframe=interval, limit=100)
                if market_data.empty:
                    self.logger.warning(f"Market data is empty for {symbol}. Skipping.")
                    continue
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], unit='ms')
                market_data.set_index('timestamp', inplace=True)
                all_market_data[symbol] = market_data
                self.logger.info(f"Successfully fetched {len(market_data)} candles for {symbol}.")
            except Exception as e:
                self.logger.error(f"Failed to fetch market data for {symbol}: {e}", exc_info=True)
                continue
        
        if not all_market_data:
            self.logger.warning("No valid market data fetched. Skipping cycle.")
            return None, None, None
        
        # 为每个资产生成特征
        all_featured_data = {}
        for symbol, market_data in all_market_data.items():
            self.logger.info(f"Generating features for {symbol}...")
            try:
                featured_data = generate_features(market_data.copy())
                all_featured_data[symbol] = featured_data
                self.logger.info(f"Feature generation for {symbol} complete. DataFrame shape: {featured_data.shape}")
            except Exception as e:
                self.logger.error(f"Failed to generate features for {symbol}: {e}", exc_info=True)
                continue
        
        # 使用主要资产的数据进行市场制度检测和性能计算
        primary_symbol = symbols[0]  # 主要资产为列表第一个
        primary_featured_data = all_featured_data.get(primary_symbol)
        if primary_featured_data is None:
            self.logger.error(f"Failed to get featured data for primary symbol {primary_symbol}")
            return None, None, None
            
        self.logger.info("Detecting market regime...")
        regime_info = detect_regime_change(primary_featured_data)
        self.logger.info(f"Analysis regime detection result: {regime_info.get('reason')}")
        
        # 使用我们自己的市场制度检测器
        if self.market_regime_detector:
            try:
                market_regime = self.market_regime_detector.detect_regime(primary_featured_data)
                self.logger.info(f"Market regime: {market_regime.get('regime', 'unknown')} - {market_regime.get('description', '')}")
                
                # 如果策略管理器存在，根据市场制度推荐策略
                if hasattr(self, 'strategy_manager') and self.strategy_manager:
                    recommended_strategy, reason = self.market_regime_detector.recommend_strategy(market_regime)
                    self.logger.info(f"Strategy recommendation: {recommended_strategy} - {reason}")
                    
                    # 如果推荐的策略与当前不同，可以考虑切换（这里暂时只记录，不自动切换）
                    current_strategy = self.strategy_manager.get_active_strategy_name()
                    if current_strategy and recommended_strategy != current_strategy:
                        self.logger.info(f"Different strategy recommended ({recommended_strategy}) vs current ({current_strategy})")
                        
            except Exception as e:
                self.logger.error(f"Error in market regime detection: {e}", exc_info=True)
                market_regime = {'regime': 'error', 'description': 'Error in regime detection'}
        else:
            market_regime = {'regime': 'unknown', 'description': 'Regime detector not available'}
        
        self.logger.info("Calculating performance KPIs...")
        if self.performance_monitor:
            performance_kpis = self.performance_monitor.calculate_kpis()
            if performance_kpis.get('status') == 'SUCCESS':
                self.logger.info(f"Latest Sharpe Ratio: {performance_kpis.get('sharpe_ratio_annualized_30_periods', 'N/A')}")
        else:
            performance_kpis = None
            self.logger.info("Performance monitor not available.")
        
        return all_featured_data, regime_info, performance_kpis

    def decide_and_act(self, all_featured_data, regime_info):
        # ... (更新后的多资产 decide_and_act 代码) ...
        self.logger.info("--- Stage: DECIDE & ACT ---")
        
        # 确定要交易的资产列表
        if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
            symbols = self.config.get('multi_asset', {}).get('symbols', [self.trader_config.get('symbol', 'BTC-USDT')])
        else:
            symbols = [self.trader_config.get('symbol', 'BTC-USDT')]
        
        # 为每个资产生成交易信号
        all_signals = {}
        primary_symbol = symbols[0]  # 使用第一个资产作为主要资产决定交易信号
        primary_featured_data = all_featured_data.get(primary_symbol)
        
        # 生成主要资产的信号
        if hasattr(self, 'strategy_manager') and self.strategy_manager:
            # 检查是否需要切换策略
            if self.strategy_manager.should_switch_strategy(lookback_days=7):
                best_strategy = self.strategy_manager.get_best_performing_strategy(lookback_days=7)
                if best_strategy:
                    self.strategy_manager.switch_strategy(best_strategy)
                    self.logger.info(f"Switched to strategy: {best_strategy}")
            
            # 使用当前策略生成主要资产的信号
            active_strategy_name = self.strategy_manager.get_active_strategy_name()
            signals_df = self.strategy_manager.generate_signals(primary_featured_data, active_strategy_name)
            self.logger.info(f"Generated signals for {primary_symbol} using strategy: {active_strategy_name}")
        else:
            # 后备策略
            signals_df = self.strategy.generate_signals(primary_featured_data)
            self.logger.info(f"Generated signals for {primary_symbol} using fallback LGB strategy")
        
        # 将主要资产的信号用作主要决策信号
        main_signal = signals_df['signal'].iloc[-1]
        self.logger.info(f"Main signal from {primary_symbol}: {main_signal}")
        
        # 如果启用模型可解释性，生成预测解释
        if hasattr(self, 'model_interpretability') and self.model_interpretability:
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
                                'top_contributions': interpretation['top_feature_contributions'][:5]  # 只记录前5个贡献
                            }) + '\n')
                else:
                    self.logger.warning("Model file not found for interpretability analysis.")
            except Exception as e:
                self.logger.error(f"Error in model interpretability: {e}", exc_info=True)
        
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
                        current_prices[symbol] = self.exchange.get_current_price(symbol)
                    except Exception as e:
                        self.logger.error(f"Failed to get current price for {symbol}: {e}")
                
                # 计算再平衡订单
                rebalance_orders = self.portfolio_manager.calculate_rebalance_orders(
                    current_prices, 
                    self._get_total_portfolio_value()
                )
                
                # 执行再平衡订单
                for order in rebalance_orders:
                    self._execute_single_order(order['symbol'], order['side'], order['quantity'], 
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
                base_balance = self.exchange.get_balance(currency=base_currency)
                quote_balance = self.exchange.get_balance(currency=quote_currency)
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
            
            if signal == 1 and base_balance < 0.001:
                self.logger.info(f"BUY signal for {symbol} and no significant position held. Executing BUY order.")
                
                # 预期价格（对于市价单，我们使用当前价格作为参考）
                expected_buy_price = current_price
                
                try:
                    quantity = trade_amount_quote / current_price
                    order_result = self.exchange.create_order(
                        symbol=symbol,
                        order_type='market',
                        side='buy',
                        amount=quantity
                    )
                    self.logger.info(f"BUY order for {symbol} executed. Result: {order_result}")
                    
                    # 记录订单到状态管理器
                    if order_result and order_result.get('code') == '0' and order_result.get('data'):
                        order_id = order_result['data'][0]['ordId']
                        executed_buy_price = current_price  # 对于市价单，实际成交价通常是执行时的价格
                        
                        # 检查滑点
                        if hasattr(self, 'slippage_monitor') and self.slippage_monitor:
                            slippage_result = self.slippage_monitor.monitor_trade_execution(
                                symbol=symbol,
                                side='buy',
                                quantity=quantity,
                                expected_price=expected_buy_price,
                                executed_price=executed_buy_price,
                                exchange_interface=self.exchange
                            )
                            if slippage_result['is_excessive_slippage']:
                                self.logger.warning(f"Excessive slippage in BUY order for {symbol}: {slippage_result['warning']}")
                            elif not slippage_result['liquidity_ok']:
                                self.logger.warning(f"Liquidity concern in BUY order for {symbol}: {slippage_result['warning']}")
                        
                        order_data = {
                            'order_id': order_id,
                            'symbol': symbol,
                            'side': 'buy',
                            'quantity': quantity,
                            'price': executed_buy_price,
                            'status': 'filled'  # 市场订单通常立即成交
                        }
                        
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
            elif signal == 0 and base_balance > 0.001:
                self.logger.info(f"SELL signal for {symbol} and position held. Executing SELL order.")
                
                # 预期价格（对于市价单，我们使用当前价格作为参考）
                expected_sell_price = current_price
                
                try:
                    order_result = self.exchange.create_order(
                        symbol=symbol,
                        order_type='market',
                        side='sell',
                        amount=base_balance
                    )
                    self.logger.info(f"SELL order for {symbol} executed. Result: {order_result}")
                    
                    # 记录订单到状态管理器，并完成交易记录
                    if order_result and order_result.get('code') == '0' and order_result.get('data'):
                        order_id = order_result['data'][0]['ordId']
                        executed_sell_price = current_price  # 对于市价单，实际成交价通常是执行时的价格
                        
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
                            elif not slippage_result['liquidity_ok']:
                                self.logger.warning(f"Liquidity concern in SELL order for {symbol}: {slippage_result['warning']}")
                        
                        order_data = {
                            'order_id': order_id,
                            'symbol': symbol,
                            'side': 'sell',
                            'quantity': base_balance,
                            'price': executed_sell_price,
                            'status': 'filled'
                        }
                        
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
                self.logger.info(f"No action needed for {symbol} based on current signal ({signal}) and position ({base_balance}).")

    def _execute_single_order(self, symbol: str, side: str, quantity: float, price: float):
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
                order_result = self.exchange.create_order(
                    symbol=symbol,
                    order_type='market',
                    side='buy',
                    amount=quantity
                )
            elif side == 'sell':
                order_result = self.exchange.create_order(
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
            if order_result and order_result.get('code') == '0' and order_result.get('data'):
                order_id = order_result['data'][0]['ordId']
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

    def _get_total_portfolio_value(self) -> float:
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
                base_balance = self.exchange.get_balance(currency=base_currency)
                quote_balance = self.exchange.get_balance(currency=quote_currency)
                
                # 获取当前价格
                current_price = self.exchange.get_current_price(symbol)
                
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

    def log_portfolio_status(self):
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

                base_balance = self.exchange.get_balance(currency=base_currency)
                quote_balance = self.exchange.get_balance(currency=quote_currency)
                current_price = self.exchange.get_current_price(symbol)

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
        """从状态文件中加载上次训练的时间。"""
        if not os.path.exists(self.training_state_path):
            return None
        try:
            with open(self.training_state_path, 'r') as f:
                state = json.load(f)
            return datetime.fromisoformat(state['last_training_time'])
        except (Exception, json.JSONDecodeError):
            self.logger.warning(f"Could not read or parse training state file at {self.training_state_path}.")
            return None

    def _save_last_training_time(self):
        """将当前的训练时间保存到状态文件。"""
        try:
            with open(self.training_state_path, 'w') as f:
                json.dump({'last_training_time': self.last_training_time.isoformat()}, f)
        except Exception as e:
            self.logger.error(f"Failed to save last training time: {e}", exc_info=True)

    def _monitor_risk(self):
        """
        监控风险指标并发出告警
        """
        if self.enhanced_monitor is None:
            # Fallback to original risk monitor
            if self.risk_monitor is None:
                self.logger.debug("RiskMonitor not available, skipping risk monitoring.")
                return

            try:
                # 获取当前账户余额
                symbol = self.trader_config.get('symbol', 'BTC-USDT')
                base_currency, quote_currency = symbol.split('-')
                quote_balance = self.exchange.get_balance(currency=quote_currency)
                base_balance = self.exchange.get_balance(currency=base_currency)
                current_price = self.exchange.get_current_price(symbol)
                
                if current_price > 0:
                    total_value = quote_balance + base_balance * current_price
                    risk_alerts = self.risk_monitor.monitor_and_alert(current_balance=total_value)
                    
                    if risk_alerts:
                        self.logger.warning(f"Risk monitoring detected {len(risk_alerts)} risk conditions:")
                        for alert in risk_alerts:
                            self.logger.warning(f"  - [{alert['severity']}] {alert['message']}")
                    else:
                        self.logger.debug("Risk monitoring: All indicators within normal range.")
                else:
                    self.logger.warning("Could not get current price for risk monitoring.")
                    
            except Exception as e:
                self.logger.error(f"Error during risk monitoring: {e}", exc_info=True)
            return

        try:
            # 获取当前账户余额
            symbol = self.trader_config.get('symbol', 'BTC-USDT')
            base_currency, quote_currency = symbol.split('-')
            quote_balance = self.exchange.get_balance(currency=quote_currency)
            base_balance = self.exchange.get_balance(currency=base_currency)
            current_price = self.exchange.get_current_price(symbol)
            
            if current_price > 0:
                total_value = quote_balance + base_balance * current_price
                
                # 使用增强监控器进行所有类型的监控
                all_alerts = self.enhanced_monitor.monitor_all(current_balance=total_value)
                
                # 发送所有类型的告警
                sent_alerts_results = self.enhanced_monitor.send_alerts(all_alerts)
                
                # 统计告警数量
                total_alerts = sum(len(alerts) for alerts in all_alerts.values())
                if total_alerts > 0:
                    self.logger.warning(f"Enhanced monitoring detected {total_alerts} conditions:")
                    for category, alerts in all_alerts.items():
                        for alert in alerts:
                            self.logger.warning(f"  - [{alert['severity']}] {alert['type']}: {alert['message']}")
                else:
                    self.logger.debug("Enhanced monitoring: All indicators within normal range.")
            else:
                self.logger.warning("Could not get current price for risk monitoring.")
                
        except Exception as e:
            self.logger.error(f"Error during enhanced risk monitoring: {e}", exc_info=True)

    def _evaluate_strategy_performance(self, featured_data: pd.DataFrame, current_signal: int):
        """
        评估当前策略的性能
        """
        if not hasattr(self, 'strategy_manager') or not self.strategy_manager:
            return
            
        try:
            # 获取当前使用的策略名称
            active_strategy_name = self.strategy_manager.get_active_strategy_name()
            if not active_strategy_name:
                return
                
            # 这里我们可以基于一些指标来评估策略性能，比如回测最近的信号
            # 暂时简单实现，实际应用中可能需要更复杂的逻辑
            latest_price = featured_data['close'].iloc[-1]
            previous_price = featured_data['close'].iloc[-2] if len(featured_data) > 1 else latest_price
            price_change = (latest_price - previous_price) / previous_price if previous_price != 0 else 0
            
            # 模拟收益计算（这里只是示意，实际应用中需要更精确的计算）
            # 根据信号和价格变化来评估策略表现
            if current_signal != 0:  # 有交易信号
                # 简单基于信号与价格变化的一致性来评估
                if (current_signal > 0 and price_change > 0) or (current_signal < 0 and price_change < 0):
                    simulated_return = abs(price_change)  # 正确方向的收益
                else:
                    simulated_return = -abs(price_change)  # 错误方向的损失
            else:
                simulated_return = 0  # 无信号，持币不动
            
            # 记录策略性能
            returns_series = pd.Series([simulated_return])
            performance = self.strategy_manager.evaluate_strategy_performance(
                active_strategy_name, 
                returns_series
            )
            
            self.logger.debug(f"Strategy {active_strategy_name} performance updated: {performance}")
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy performance: {e}", exc_info=True)

    def _handle_system_alert(self, status: Dict):
        """
        处理系统警报
        """
        if hasattr(self, 'enhanced_monitor') and self.enhanced_monitor:
            # 发送系统警报
            alert_message = f"System Alert: {', '.join(status['alerts'])}"
            self.enhanced_monitor.notification_manager.send_risk_alert("SYSTEM_RESOURCE_ALERT", alert_message)
        else:
            # 如果没有增强监控器，记录日志
            self.logger.warning(f"System Alert: {', '.join(status['alerts'])}")

    def run_loop(self):
        # 主循环
        while True:
            try:
                # 0. 检查模型是否已更新并重载
                self._check_and_reload_model()

                # 1. 感知
                featured_data, regime_info, performance_kpis = self.sense()

                # 如果感知阶段失败，则跳过此循环
                if featured_data is None:
                    time.sleep(60)
                    continue

                # 2. 决策与行动
                self.decide_and_act(featured_data, regime_info)

                # 3. 记录状态 (为下一次学习做准备)
                self.log_portfolio_status()

                # 4. 监控风险
                self._monitor_risk()

                # 5. 学习
                self.learn(performance_kpis, regime_info)

                self.logger.info("Cycle complete. Waiting for next iteration...")
                time.sleep(60) # 每分钟循环一次

            except KeyboardInterrupt:
                self.logger.info("Shutdown signal received.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred: {e}", exc_info=True)
                time.sleep(300) # 出错后等待5分钟

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 确保在Windows上使用 'spawn' 启动方式以避免问题
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn', force=True)
    trader = AutonomousTrader()
    
    try:
        trader.run_loop()
    except KeyboardInterrupt:
        logging.info("Received shutdown signal.")
    finally:
        # 确保所有连接都被正确关闭
        if hasattr(trader, 'state_manager') and trader.state_manager:
            trader.state_manager.close()
        if hasattr(trader, 'risk_monitor') and hasattr(trader.risk_monitor, 'trade_tracker') and trader.risk_monitor.trade_tracker:
            trader.risk_monitor.trade_tracker.conn.close()
        if hasattr(trader, 'enhanced_monitor') and trader.enhanced_monitor and hasattr(trader.enhanced_monitor, 'risk_monitor') and trader.enhanced_monitor.risk_monitor.trade_tracker:
            trader.enhanced_monitor.risk_monitor.trade_tracker.conn.close()
        
        # 停止系统监控
        if hasattr(trader, 'system_monitor') and trader.system_monitor:
            trader.system_monitor.stop_monitoring()
        
        logging.info("All connections closed. Shutdown complete.")

if __name__ == '__main__':
    main()
