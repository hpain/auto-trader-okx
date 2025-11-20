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
        self.client = exchange_client

    async def execute_trades(self, trade_orders: list, cycle_logger) -> list:
        """
        异步执行一个交易指令列表，并返回详细的执行报告。

        :param trade_orders: 来自PortfolioManager的交易指令列表。
        :param cycle_logger: 用于记录本次循环日志的CycleLogger实例。
        :return: 一个包含执行结果的字典列表 (执行报告)。
        """
        print("\n--- [ExecutionHandler] Starting Trade Execution ---")
        if not trade_orders:
            print("No trade orders to execute.")
            print("--- [ExecutionHandler] Execution Finished ---")
            return []

        execution_report = []
        for order in trade_orders:
            symbol = order['symbol']
            action = order['action']
            quantity = order['quantity']
            
            report_item = {
                'symbol': symbol,
                'action': action,
                'requested_quantity': quantity,
                'filled_quantity': 0,
                'status': 'FAILURE',
                'raw_response': None
            }

            try:
                print(f"  - Received order: {action} {quantity:.6f} {symbol}")

                side = 'buy' if action == 'BUY' else 'sell'
                order_type = 'market'

                if self.client and hasattr(self.client, 'create_order'):
                    # 使用 await 调用异步方法，并修正方法名为 create_order
                    order_result = await self.client.create_order(
                        symbol=symbol,
                        order_type=order_type,
                        side=side,
                        amount=quantity
                    )
                    report_item['raw_response'] = order_result
                    
                    # 假设成功的API调用返回的字典中包含'info'和'status'
                    if order_result and order_result.get('info', {}).get('sCode') == '0':
                        print(f"    - SUCCESS: Order API call successful.")
                        report_item['status'] = 'SUCCESS'
                        # 假设市价单完全成交
                        report_item['filled_quantity'] = quantity
                    else:
                        print(f"    - FAILURE: Order placement failed. Response: {order_result}")
                        report_item['status'] = 'FAILURE'
                else:
                    report_item['raw_response'] = "Client not configured or method 'create_order' not found."

            except Exception as e:
                print(f"    - CRITICAL ERROR: An exception occurred while placing order: {e}")
                report_item['raw_response'] = str(e)
            
            execution_report.append(report_item)
        
        # 使用CycleLogger记录执行信息
        cycle_logger.add_execution_info(reports=execution_report)

        print("--- [ExecutionHandler] Execution Finished ---")
        return execution_report
