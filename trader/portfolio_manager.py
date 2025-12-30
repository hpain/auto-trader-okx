"""
多资产组合管理器，用于管理多个交易对的投资组合
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


class PortfolioManager:
    """
    投资组合管理器，负责多资产的配置、风险管理和再平衡
    """
    
    def __init__(self, strategies: List = None, capital: float = 10000.0, risk_config: Dict = None, 
                 exchange_client: Any = None, config: Dict = None, symbols: List[str] = None, 
                 strategy_weights: Dict[str, float] = None):
        """
        初始化投资组合管理器
        """
        # DEBUG INIT
        print(f"DEBUG: PortfolioManager Init. Strategies: {len(strategies) if strategies else 0}, Config passed: {config is not None}")
        
        # 支持新旧两种初始化方式
        # If strategies are explicitly passed, use them! Don't overwrite with empty list from config.
        if config is not None and not strategies:
            # 旧版（或 Config-driven）初始化方式
            self.config = config.get('multi_asset', {})
            self.symbols = self.config.get('symbols', ['BTC-USDT'])
            self.allocation_strategy = self.config.get('allocation_strategy', 'equal')
            self.max_assets = self.config.get('max_assets', 5)
            self.rebalance_frequency_days = self.config.get('rebalance_frequency_days', 7)
            self.per_asset_risk_limit = self.config.get('per_asset_risk_limit', 0.10)
            self.overall_risk_limit = self.config.get('overall_risk_limit', 0.15)
            
            # 为旧版初始化设置默认值
            self.strategies = []
            self.capital = 10000.0
            self.risk_config = {}
            self.exchange_client = None
        else:

            # 新版初始化方式，支持与 run_live.py 兼容
            self.strategies = strategies or []
            self.capital = capital
            self.risk_config = risk_config or {}
            self.exchange_client = exchange_client
            self.symbols = symbols or ['BTC-USDT']  # 使用传入的symbols或默认值
            self.allocation_strategy = 'equal'
            self.max_assets = 5
            self.rebalance_frequency_days = 7
            self.per_asset_risk_limit = 0.10
            self.overall_risk_limit = 0.15
            
        self.strategy_weights = strategy_weights or {}
        # Default weight 1.0 if not specified
        for s in self.strategies:
            if hasattr(s, 'strategy_name') and s.strategy_name not in self.strategy_weights:
                self.strategy_weights[s.strategy_name] = 1.0
        
        self.logger = logging.getLogger(__name__)
        self.current_allocations = {}
        self.last_rebalance_time = datetime.utcnow()
        
        # 初始化资产数据
        self.asset_data = {symbol: {
            'position': 0.0,
            'value': 0.0,
            'pnl': 0.0,
            'weight': 0.0
        } for symbol in self.symbols}
        
    # 新增初始化 RiskEngine
        self.risk_engine_enabled = self.risk_config.get('use_risk_engine', False)
        if self.risk_engine_enabled:
            from trader.risk_engine import RiskEngine
            target_vol = self.risk_config.get('target_volatility', 0.20)
            self.risk_engine = RiskEngine(target_volatility=target_vol)
            self.logger.info("RiskEngine integrated into PortfolioManager.")
        else:
            self.risk_engine = None

        # 新增：用于跟踪各资产的策略分配
        self.asset_strategies = {symbol: self.strategies for symbol in self.symbols}
        
        # 新增：用于跟踪各资产的持仓
        self.positions = {}

        # ========== 移植 1.3: 每日盈亏跟踪 ==========
        self.daily_pnl = 0.0
        self.last_pnl_reset = datetime.utcnow().date()
        self.daily_loss_limit = -500.0  # 硬编码阈值，建议后续放入 config

    def _calculate_position_size(self, symbol: str, price: float, signal: int, volatility_scalar: float = 1.0) -> float:
        """
        计算头寸大小
        Args:
            symbol: 交易对符号
            price: 当前价格
            signal: 信号 (1=buy, 0=hold, -1=sell)
            volatility_scalar: 波动率调整系数 (默认 1.0)
        Returns:
            计算出的头寸大小
        """
        if price <= 0:
            return 0.0
        
        # 基于风险配置计算头寸大小
        risk_per_trade = self.risk_config.get('risk_per_trade', 0.01)
        
        # 使用总资本的一定比例计算订单大小
        # Apply volatility scalar: High Vol -> scalar < 1 -> smaller size
        risk_amount = self.capital * risk_per_trade * volatility_scalar
        quantity = risk_amount / price
        
        # Log if scalar is effectively active
        if abs(volatility_scalar - 1.0) > 0.05 and signal > 0:
            self.logger.info(f"RiskEngine applied: {symbol} size adjusted by {volatility_scalar:.2f}x (Vol Target)")
        
        # 如果是卖出信号，全仓卖出
        if signal < 0:
            current_position = self.positions.get(symbol, 0.0)
            quantity = current_position
        
        return quantity
        
        return quantity
    
    def calculate_target_allocations(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        根据配置的策略计算目标资产配置
        
        Args:
            market_data: 包含各资产市场数据的字典
            
        Returns:
            各资产的目标配置权重字典
        """
        if self.allocation_strategy == 'equal':
            # 等权重分配
            active_symbols = [s for s in self.symbols if s in market_data and len(market_data[s]) > 0]
            if not active_symbols:
                return {}
                
            equal_weight = 1.0 / len(active_symbols)
            target_allocations = {symbol: equal_weight for symbol in active_symbols}
            
        elif self.allocation_strategy == 'risk_parity':
            # 风险平价分配
            target_allocations = self._calculate_risk_parity_allocations(market_data)
            
        elif self.allocation_strategy == 'volatility_weighted':
            # 波动率加权分配
            target_allocations = self._calculate_volatility_weighted_allocations(market_data)
            
        else:
            # 默认使用等权重
            active_symbols = [s for s in self.symbols if s in market_data and len(market_data[s]) > 0]
            equal_weight = 1.0 / len(active_symbols)
            target_allocations = {symbol: equal_weight for symbol in active_symbols}
        
        return target_allocations
    
    def _calculate_risk_parity_allocations(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        计算风险平价配置
        """
        active_symbols = [s for s in self.symbols if s in market_data and len(market_data[s]) > 0 and 'close' in market_data[s].columns]
        
        if not active_symbols:
            return {}
        
        # 计算各资产的波动率（作为风险的代理）
        volatilities = {}
        for symbol in active_symbols:
            prices = market_data[symbol]['close']
            returns = prices.pct_change().dropna()
            if len(returns) > 1:
                volatilities[symbol] = returns.std() * np.sqrt(252)  # 年化波动率
            else:
                volatilities[symbol] = 0.0
        
        # 如果所有波动率都为0，使用等权重
        if all(v == 0 for v in volatilities.values()):
            equal_weight = 1.0 / len(active_symbols)
            return {symbol: equal_weight for symbol in active_symbols}
        
        # 简化的风险平价计算（实际应用中可能需要更复杂的方法）
        # 风险平价通常与波动率成反比
        inverse_volatilities = {s: 1.0 / max(v, 0.001) for s, v in volatilities.items()}  # 避免除以0
        total_inverse_vol = sum(inverse_volatilities.values())
        
        if total_inverse_vol > 0:
            return {s: iv / total_inverse_vol for s, iv in inverse_volatilities.items()}
        else:
            # 如果计算出问题，返回等权重
            equal_weight = 1.0 / len(active_symbols)
            return {symbol: equal_weight for symbol in active_symbols}
    
    def _calculate_volatility_weighted_allocations(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        计算波动率加权配置
        """
        active_symbols = [s for s in self.symbols if s in market_data and len(market_data[s]) > 0 and 'close' in market_data[s].columns]
        
        if not active_symbols:
            return {}
        
        # 计算各资产的波动率
        volatilities = {}
        for symbol in active_symbols:
            prices = market_data[symbol]['close']
            returns = prices.pct_change().dropna()
            if len(returns) > 1:
                volatilities[symbol] = returns.std() * np.sqrt(252)  # 年化波动率
            else:
                volatilities[symbol] = 0.0
        
        # 使用波动率的反比作为权重
        max_vol = max(volatilities.values()) if volatilities else 1.0
        if max_vol == 0:
            equal_weight = 1.0 / len(active_symbols)
            return {symbol: equal_weight for symbol in active_symbols}
        
        # 波动率越低，分配权重越高
        vol_weights = {s: (max_vol - v) / max_vol for s, v in volatilities.items()}
        total_vol_weight = sum(vol_weights.values())
        
        if total_vol_weight > 0:
            return {s: w / total_vol_weight for s, w in vol_weights.items()}
        else:
            equal_weight = 1.0 / len(active_symbols)
            return {symbol: equal_weight for symbol in active_symbols}
    
    def should_rebalance(self) -> bool:
        """
        检查是否需要再平衡
        """
        time_since_rebalance = datetime.utcnow() - self.last_rebalance_time
        return time_since_rebalance.days >= self.rebalance_frequency_days
    
    def calculate_rebalance_orders(self, current_prices: Dict[str, float], 
                                 total_portfolio_value: float) -> List[Dict]:
        """
        计算再平衡订单
        
        Args:
            current_prices: 当前价格字典
            total_portfolio_value: 投资组合总价值
            
        Returns:
            订单列表
        """
        target_allocations = self.calculate_target_allocations({})  # 简化版本，实际应用中需要市场数据
        
        orders = []
        for symbol, target_weight in target_allocations.items():
            if symbol not in current_prices or current_prices[symbol] <= 0:
                continue
                
            target_value = total_portfolio_value * target_weight
            target_quantity = target_value / current_prices[symbol]
            
            # 获取当前持仓
            current_quantity = self.asset_data.get(symbol, {}).get('position', 0.0)
            
            # 计算需要调整的数量
            quantity_diff = target_quantity - current_quantity
            
            if abs(quantity_diff) > 0.0001:  # 避免过于微小的调整
                side = 'buy' if quantity_diff > 0 else 'sell'
                order = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': abs(quantity_diff),
                    'target_weight': target_weight,
                    'current_position': current_quantity,
                    'target_position': target_quantity,
                    'current_price': current_prices[symbol]
                }
                orders.append(order)
        
        return orders
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """
        更新特定资产的持仓
        """
        if symbol not in self.asset_data:
            self.asset_data[symbol] = {
                'position': 0.0,
                'value': 0.0,
                'pnl': 0.0,
                'weight': 0.0
            }
        
        self.asset_data[symbol]['position'] = quantity
        self.asset_data[symbol]['value'] = quantity * price if price > 0 else 0.0
        self.logger.info(f"Updated position for {symbol}: {quantity} units at {price}, value: {self.asset_data[symbol]['value']:.2f}")
    
    def get_portfolio_summary(self) -> Dict:
        """
        获取投资组合摘要
        """
        total_value = sum(asset['value'] for asset in self.asset_data.values())
        
        summary = {
            'total_value': total_value,
            'last_rebalance_time': self.last_rebalance_time.isoformat(),
            'should_rebalance': self.should_rebalance(),
            'allocation_strategy': self.allocation_strategy,
            'assets': {}
        }
        
        for symbol, data in self.asset_data.items():
            if total_value > 0:
                weight = data['value'] / total_value
            else:
                weight = 0.0
                
            summary['assets'][symbol] = {
                'position': data['position'],
                'value': data['value'],
                'weight': weight,
                'pnl': data['pnl']
            }
        
        return summary
    
    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """
        检查是否超过风险限制
        
        Returns:
            (是否超过风险限制, 风险警告列表)
        """
        total_value = sum(asset['value'] for asset in self.asset_data.values())
        if total_value <= 0:
            return False, []
        
        risk_warnings = []
        
        # 检查单个资产风险限制
        for symbol, data in self.asset_data.items():
            asset_weight = data['value'] / total_value if total_value > 0 else 0
            if asset_weight > self.per_asset_risk_limit:
                risk_warnings.append(f"Asset {symbol} weight {asset_weight:.2%} exceeds limit {self.per_asset_risk_limit:.2%}")
        
        # 检查总体风险限制（这里可以扩展更多风险指标）
        if len(risk_warnings) > 0:
            return True, risk_warnings
        
        return False, []
    
    def rebalance(self, data: Dict[str, pd.DataFrame], cycle_logger=None) -> Tuple[List[Dict], Dict]:
        """
        根据市场数据重新平衡投资组合
        
        Args:
            data: 包含各资产市场数据的字典 {symbol: DataFrame}
            cycle_logger: 周期日志记录器
            
        Returns:
            (交易订单列表, 附加信息字典)
        """
        # ========== 移植 1.3: 每日止损检查 ==========
        current_date = datetime.utcnow().date()
        if current_date > self.last_pnl_reset:
            self.daily_pnl = 0.0
            self.last_pnl_reset = current_date
            self.logger.info("Daily PnL reset for new day.")
            
        if self.daily_pnl < self.daily_loss_limit:
            msg = f"🚨 DAILY LOSS LIMIT REACHED (${self.daily_pnl:.2f} < ${self.daily_loss_limit:.2f}). No new orders will be generated."
            self.logger.critical(msg)
            if cycle_logger:
                cycle_logger.add_error(msg)
            return [], {'status': 'STOPPED_DAILY_LOSS'}
        # ==========================================

        if cycle_logger:
            cycle_logger.add_info("Starting portfolio rebalance process")
        
        trade_orders = []
        
        # 遍历所有资产，为每个资产生成信号和订单
        for symbol, df in data.items():
            # 为每个资产选择合适的策略（这里可以集成MarketRegimeDetector的建议）
            selected_strategies = self.asset_strategies.get(symbol, self.strategies)
            # DEBUG LOG
            if cycle_logger:
                 cycle_logger.add_info(f"DEBUG: Selected {len(selected_strategies)} strategies for {symbol}: {[s.strategy_name for s in selected_strategies]}")
            
            
            # 从所有策略获取信号并进行融合
            strategy_signals = {}
            valid_signals_list = []
            
            for strategy in selected_strategies:
                try:
                    # 获取策略信号
                    signal = strategy.generate_signal(df, symbol=symbol)
                    strategy_signals[strategy.strategy_name] = signal
                    valid_signals_list.append(signal)
                except Exception as e:
                    if cycle_logger:
                        cycle_logger.add_warning(f"Error getting signal from strategy {strategy.strategy_name}: {e}")
                    continue
            
            # 融合多个策略的信号 (Weighted Fuse)
            # Filter strategies to only those that succeeded for the weighted calculation
            succeeded_strategies = [s for s in selected_strategies if s.strategy_name in strategy_signals]
            final_signal, fusion_details = self._fuse_signals_weighted(valid_signals_list, succeeded_strategies)
            
            # --- Enhanced Consolidated Logging ---
            if cycle_logger:
                score_str = f"{final_signal:+.1f}"
                action_str = "BUY" if final_signal > 0.5 else ("SELL" if final_signal < -0.5 else "HOLD")
                
                # Build detail string: "Trans(+1.5) MA_S(-1.0)..."
                details_parts = []
                # Iterate over ALL selected strategies to show who failed
                for s in selected_strategies:
                    s_name = getattr(s, 'strategy_name', 'Unknown')
                    display_name = s_name.replace('Transformer_Main', 'Trans').replace('MovingAverage', 'MA').replace('LGB_Main', 'LGB').replace('Strategy', '')
                    
                    if s_name in strategy_signals:
                        sig = strategy_signals[s_name]
                        weight = self.strategy_weights.get(s_name, 1.0)
                        contrib = sig * weight
                        details_parts.append(f"{display_name}({contrib:+.1f})")
                    else:
                        details_parts.append(f"{display_name}(ERR)")
                
                details_str = " ".join(details_parts)
                cycle_logger.add_info(f"[{symbol}] Score: {score_str} ({action_str}) | {details_str}")
            # -------------------------------------
            
            # 决定是否交易
            should_trade = self._should_trade(symbol, final_signal)
            
            if should_trade:
                # 计算订单大小
                current_price = df['close'].iloc[-1] if not df.empty else 0
                
                # --- Volatility Scalar Calculation ---
                vol_scalar = 1.0
                if self.risk_engine and final_signal > 0:
                    try:
                        # Use RiskEngine to calculate scalar based on recent volatility
                        vol_scalar = self.risk_engine.calculate_volatility_scalar(
                            df['close'], 
                            window=self.risk_config.get('volatility_window', 30)
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate vol scalar for {symbol}: {e}")
                # -------------------------------------

                quantity = self._calculate_position_size(symbol, current_price, final_signal, volatility_scalar=vol_scalar)
                
                # ========== 移植 1.3: 最小交易价值检查 (防止尘埃单) ==========
                min_trade_value = 10.0 # USD
                estimated_value = quantity * current_price
                
                if quantity > 0 and estimated_value >= min_trade_value:
                    side = 'buy' if final_signal > 0 else 'sell'
                    
                    order = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': abs(quantity),
                        'price': current_price, # 确保传递价格给 ExecutionHandler
                        'strategy': getattr(selected_strategies[0], 'strategy_name', 'unknown') if selected_strategies else 'unknown',
                        'signal_strength': final_signal
                    }
                    
                    # 风险检查
                    if self._check_risk_limits(order):
                        trade_orders.append(order)
                    else:
                        if cycle_logger:
                            cycle_logger.add_warning(f"Order for {symbol} rejected by risk management: {order}")
                elif quantity > 0:
                     if cycle_logger:
                        cycle_logger.add_info(f"Ignored dust trade for {symbol}: Value ${estimated_value:.2f} < ${min_trade_value}")
        
        # 执行投资组合级别的风险检查
        if trade_orders:
            trade_orders = self._filter_orders_by_portfolio_risk(trade_orders)
        
        info = {
            'num_orders': len(trade_orders),
            'symbols_traded': [order['symbol'] for order in trade_orders]
        }
        
        if cycle_logger:
            cycle_logger.add_info(f"Generated {len(trade_orders)} trade orders: {info['symbols_traded']}")
        
        return trade_orders, info

    def _fuse_signals_weighted(self, signals: List[int], strategies: List) -> Tuple[float, Dict]:
        """
        Weighted fusion of strategy signals.
        Returns: (Weighted Score, Details Dict)
        """
        if not signals or not strategies:
            return 0.0, {}
            
        total_score = 0.0
        details = {}
        
        for strategy, signal in zip(strategies, signals):
            name = getattr(strategy, 'strategy_name', 'Unknown')
            weight = self.strategy_weights.get(name, 1.0)
            score = signal * weight
            total_score += score
            details[name] = score
            
        # Decision Logic: 
        # Score > 0.5 -> Buy (1)
        # Score < -0.5 -> Sell (-1)
        # Else -> Hold (0)
        # However, we return the raw score for the 'signal_strength' field, 
        # but for _should_trade we need integer logic direction.
        
        return total_score, details

    def _fuse_signals(self, signals: List[int]) -> int:
         # Deprecated legacy method
        return sum(signals)

    def _should_trade(self, symbol: str, signal: int) -> bool:
        """
        决定是否对指定资产进行交易
        
        Args:
            symbol: 交易对符号
            signal: 策略信号
        
        Returns:
            是否应该交易
        """
        # 获取当前持仓
        current_position = self.positions.get(symbol, 0.0)
        
        # 如果有买入信号且无持仓，则买入
        if signal > 0 and current_position == 0:
            return True
        # 如果有卖出信号且有持仓，则卖出
        elif signal < 0 and current_position > 0:
            return True
        # 其他情况不交易
        else:
            return False



    def _check_risk_limits(self, order: Dict) -> bool:
        """
        检查订单是否符合风险限制
        
        Args:
            order: 订单字典
        
        Returns:
            是否通过风险检查
        """
        try:
            order_value = order['quantity'] * order['price']
            
            # 检查单笔订单风险
            if order_value > self.capital * self.risk_config.get('max_portfolio_risk', 0.05):
                return False
            
            return True
        except Exception:
            return False

    def _filter_orders_by_portfolio_risk(self, orders: List[Dict]) -> List[Dict]:
        """
        根据投资组合风险过滤订单
        
        Args:
            orders: 订单列表
        
        Returns:
            过滤后的订单列表
        """
        # 检查总风险敞口
        total_order_value = sum(order['quantity'] * order['price'] for order in orders)
        max_risk_value = self.capital * self.risk_config.get('max_portfolio_risk', 0.05)
        
        if total_order_value > max_risk_value:
            # 按信号强度排序并选择最强的订单
            orders.sort(key=lambda x: abs(x['signal_strength']), reverse=True)
            # 只保留部分订单以控制风险
            selected_orders = []
            current_value = 0.0
            for order in orders:
                order_value = order['quantity'] * order['price']
                if current_value + order_value <= max_risk_value:
                    selected_orders.append(order)
                    current_value += order_value
                else:
                    break
            return selected_orders
        
        return orders

    def update_positions(self, execution_report: List[Dict]):
        """
        根据执行报告更新持仓
        
        Args:
            execution_report: 执行报告列表
        """
        for report in execution_report:
            symbol = report.get('symbol', '')
            executed_qty = report.get('executed_qty', 0.0)
            side = report.get('side', '').lower()
            
            current_position = self.positions.get(symbol, 0.0)
            
            if side == 'buy':
                new_position = current_position + executed_qty
            elif side == 'sell':
                new_position = current_position - executed_qty
            else:
                continue  # 忽略其他操作
            
            # 确保持仓不会变成负数（除做空外）
            self.positions[symbol] = max(0.0, new_position)
            
            # 同时更新资产数据
            if symbol in self.asset_data:
                current_price = report.get('price', 0.0)
                self.asset_data[symbol]['position'] = new_position
                self.asset_data[symbol]['value'] = new_position * current_price if current_price > 0 else 0.0

    def update_position(self, symbol: str, quantity: float, price: float):
        """
        更新特定资产的持仓
        """
        if symbol not in self.asset_data:
            self.asset_data[symbol] = {
                'position': 0.0,
                'value': 0.0,
                'pnl': 0.0,
                'weight': 0.0
            }
        
        self.asset_data[symbol]['position'] = quantity
        self.asset_data[symbol]['value'] = quantity * price if price > 0 else 0.0
        self.positions[symbol] = quantity  # 同步更新positions
        self.logger.info(f"Updated position for {symbol}: {quantity} units at {price}, value: {self.asset_data[symbol]['value']:.2f}")

    def get_optimal_symbols_to_trade(self, signals: Dict[str, int], 
                                   current_prices: Dict[str, float]) -> List[str]:
        """
        根据信号和当前持仓情况，确定应该交易的最优资产列表
        
        Args:
            signals: 各资产的交易信号字典
            current_prices: 当前价格字典
            
        Returns:
            应该交易的资产列表
        """
        # 如果需要再平衡，优先考虑再平衡
        if self.should_rebalance():
            return list(set(signals.keys()))  # 返回所有有信号的资产
        
        # 根据当前配置和信号决定交易哪些资产
        active_symbols = []
        for symbol, signal in signals.items():
            if symbol not in current_prices or current_prices[symbol] <= 0:
                continue
                
            # 获取当前持仓
            current_position = self.asset_data.get(symbol, {}).get('position', 0.0)
            
            # 根据信号和当前持仓决定是否交易
            if signal == 1 and current_position <= 0.001:  # 买入信号且无持仓
                active_symbols.append(symbol)
            elif signal == 0 and current_position > 0.001:  # 卖出信号且有持仓
                active_symbols.append(symbol)
        
        # 如果资产数量超过限制，只选择最有可能盈利的资产
        if len(active_symbols) > self.max_assets:
            # 这里可以添加更复杂的排序逻辑，比如根据信号强度或技术指标
            active_symbols = active_symbols[:self.max_assets]
        
        return active_symbols

    async def sync_with_exchange(self):
        """
        Synchronize internal portfolio state with the actual exchange balances.
        This is CRITICAL to prevent state drift on restarts.
        """
        if not self.exchange_client:
            self.logger.warning("Cannot sync portfolio: No exchange client connected.")
            return

        self.logger.info("Synchronizing portfolio state with exchange...")
        
        # Sync Capital (USDT)
        try:
            # Assuming USDT is the quote currency for all pairs for now
            usdt_balance = await self.exchange_client.get_balance('USDT')
            self.capital = float(usdt_balance)
            self.logger.info(f"Synced Capital (USDT): {self.capital:.2f}")
        except Exception as e:
            self.logger.error(f"Failed to sync capital: {e}")

        # Sync Positions
        for symbol in self.symbols:
            try:
                # Parse base currency (e.g., BTC/USDT -> BTC)
                if '/' in symbol:
                    base_currency = symbol.split('/')[0]
                elif '-' in symbol:
                    base_currency = symbol.split('-')[0]
                else:
                    self.logger.warning(f"Skipping sync for unparseable symbol: {symbol}")
                    continue
                
                # Fetch balance for base currency
                balance = await self.exchange_client.get_balance(base_currency)
                balance = float(balance)
                
                # Update internal state
                self.positions[symbol] = balance
                
                # Update asset_data
                if symbol not in self.asset_data:
                    self.asset_data[symbol] = {}
                self.asset_data[symbol]['position'] = balance
                
                # Try to update value if we can get a price (best effort)
                try:
                    price = await self.exchange_client.get_current_price(symbol)
                    if price:
                        value = balance * price
                        self.asset_data[symbol]['value'] = value
                        self.logger.info(f"Synced {symbol}: {balance:.6f} (Value: ${value:.2f})")
                except Exception:
                    self.logger.info(f"Synced {symbol}: {balance:.6f} (Price unavailable)")
                    
            except Exception as e:
                self.logger.error(f"Failed to sync position for {symbol}: {e}")

        self.logger.info("Portfolio synchronization complete.")


# 全局实例
_portfolio_manager = None


def get_portfolio_manager(config: Dict) -> PortfolioManager:
    """
    获取或创建投资组合管理器实例
    """
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager(config=config)
    return _portfolio_manager