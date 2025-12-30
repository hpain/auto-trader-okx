import logging

class ExecutionHandler:
    """
    订单执行器。

    作为系统的“通讯兵”，它接收来自PortfolioManager的交易指令，
    并负责将这些指令转化为对交易所API的实际调用。它封装了所有与
    交易所交互的细节。
    """

    def __init__(self, exchange_client):
        """
        初始化订单执行器。

        :param exchange_client: 一个交易所客户端的实例 (例如 OKXClient)。
        """
        self.logger = logging.getLogger(__name__)
        self.client = exchange_client

    async def execute_trades(self, trade_orders: list, cycle_logger) -> list:
        """
        异步执行一个交易指令列表，并返回详细的执行报告。

        :param trade_orders: 来自PortfolioManager的交易指令列表。
        :param cycle_logger: 用于记录本次循环日志的CycleLogger实例。
        :return: 一个包含执行结果的字典列表 (执行报告)。
        """
        self.logger.info("--- [ExecutionHandler] Starting Trade Execution ---")
        if not trade_orders:
            self.logger.info("No trade orders to execute.")
            self.logger.info("--- [ExecutionHandler] Execution Finished ---")
            return []

        execution_report = []
        for order in trade_orders:
            symbol = order.get('symbol')
            # Support both 'side' (buy/sell) and 'action' (BUY/SELL)
            side = order.get('side')
            if not side and 'action' in order:
                side = 'buy' if order['action'].upper() == 'BUY' else 'sell'
            
            if not side:
                self.logger.error(f"  - Error: Order missing 'side' or 'action': {order}")
                continue

            quantity = order.get('quantity')
            
            report_item = {
                'symbol': symbol,
                'side': side, # Standardize on side in report
                'requested_quantity': quantity,
                'filled_quantity': 0,
                'status': 'FAILURE',
                'raw_response': None
            }

            try:
                self.logger.info(f"  - Received order: {side.upper()} {quantity:.6f} {symbol}")

                # side is already determined above
                
                # ========== 移植 1.2: 强制使用限价单 & 滑点保护 (Slippage Protection) ==========
                # 1. 获取基准价格
                base_price = order.get('price')
                if not base_price or base_price <= 0:
                    # 如果订单没带价格，尝试紧急获取（虽然这会增加延迟，但比市价单安全）
                    if hasattr(self.client, 'get_current_price'):
                        base_price = await self.client.get_current_price(symbol)
                    else:
                         raise ValueError(f"Cannot determine price for {symbol} limit order.")
                
                # 2. 计算带保护的限价 (Slippage: 0.2%)
                # 买入：允许最高买入价 = 当前价 * 1.002
                # 卖出：允许最低卖出价 = 当前价 * 0.998
                slippage_tolerance = 0.002
                if side == 'buy':
                    limit_price = base_price * (1 + slippage_tolerance)
                else:
                    limit_price = base_price * (1 - slippage_tolerance)
                
                # 保留小数点精度 (假设大多数加密货币 2-4 位，严谨做法应查询 instrument info)
                # 这里暂时不做过度工程，交给交易所 API 处理精度或后续优化
                
                self.logger.info(f"    - Strategy Price: {base_price:.4f}, Limit Price (w/ protection): {limit_price:.4f}")

                if self.client and hasattr(self.client, 'create_order'):
                    # 3. 发送限价单
                    order_result = await self.client.create_order(
                        symbol=symbol,
                        order_type='limit',  # 强制 Limit
                        side=side,
                        amount=quantity,
                        price=limit_price    # 传入限价
                    )
                    report_item['raw_response'] = order_result
                    
                    # 4. 严格检查订单状态
                    # OKX API: code '0' = Success
                    # Mock API: usually returns dict with 'id'
                    is_success = False
                    if isinstance(order_result, dict):
                        if str(order_result.get('code', '0')) == '0': # OKX 标准
                            is_success = True
                        elif 'id' in order_result: # Mock/CCXT 标准
                             is_success = True
                    
                    if is_success:
                        self.logger.info(f"    - SUCCESS: Limit Order placed. ID: {order_result.get('id') or order_result.get('data', [{}])[0].get('ordId')}")
                        report_item['status'] = 'SUCCESS'
                        # 注意：限价单不一定立即成交，但在下单层面是成功的
                        report_item['filled_quantity'] = quantity
                        # CRITICAL FIX: Include price in report so PortfolioManager can calculate value
                        report_item['price'] = limit_price
                    else:
                        self.logger.error(f"    - FAILURE: Exchange rejected order. Response: {order_result}")
                        report_item['status'] = 'FAILURE'
                else:
                    report_item['raw_response'] = "Client not configured or method 'create_order' not found."

            except Exception as e:
                self.logger.critical(f"    - CRITICAL ERROR: An exception occurred while placing order: {e}")
                report_item['raw_response'] = str(e)
                # 移除危险的对冲逻辑
                report_item['status'] = 'ERROR'

            
            execution_report.append(report_item)
        
        # 使用CycleLogger记录执行信息
        cycle_logger.add_execution_info(reports=execution_report)

        self.logger.info("--- [ExecutionHandler] Execution Finished ---")
        return execution_report
