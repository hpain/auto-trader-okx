print("DEBUG: Starting Imports...", flush=True)
import unittest
import logging
import pandas as pd
import asyncio
import sys
import os

# Add project root to sys.path to ensure 'trader' module is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock, AsyncMock

print("DEBUG: Importing PortfolioManager...", flush=True)
from trader.portfolio_manager import PortfolioManager
print("DEBUG: Importing ExecutionHandler...", flush=True)
from trader.execution_handler import ExecutionHandler
print("DEBUG: Imports Done.", flush=True)

class TestOCOExecution(unittest.TestCase):
    def setUp(self):
        print("DEBUG: Setting up Test...", flush=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TestOCO")
        
        # 1. Mock Strategy
        self.mock_strategy = MagicMock()
        self.mock_strategy.strategy_name = "TestStrategy"
        self.mock_strategy.generate_signal.return_value = 1.0 # Strong BUY
        
        # 2. Mock Exchange Client (Async)
        self.mock_client = AsyncMock()
        self.mock_client.place_oco_order.return_value = {'code': '0', 'data': [{'ordId': 'OCO_123'}]}
        self.mock_client.create_order.return_value = {'code': '0', 'data': [{'ordId': 'LIMIT_123'}]}
        self.mock_client.get_current_price.return_value = 50000.0
        
        # 3. Setup PortfolioManager
        self.pm = PortfolioManager(strategies=[self.mock_strategy], config={})
        self.pm.exchange_client = self.mock_client
        self.pm.high_watermarks = {} # Reset
        self.pm.positions = {'BTC/USDT': 0.0} # Empty position
        
        # 4. Setup ExecutionHandler
        self.eh = ExecutionHandler(exchange_client=self.mock_client)
        print("DEBUG: Setup Complete.", flush=True)

    def test_oco_order_generation_and_execution(self):
        symbol = "BTC/USDT"
        current_price = 50000.0
        
        # Mock Market Data
        df = pd.DataFrame({
            'close': [48000.0, 49000.0, 50000.0],
            'open': [48000.0, 49000.0, 50000.0],
            'high': [48000.0, 49000.0, 50000.0],
            'low': [48000.0, 49000.0, 50000.0],
            'vol': [100, 100, 100]
        })
        market_data = {symbol: df}
        
        # --- Step 1: Generate Orders via PortfolioManager ---
        self.logger.info("Step 1: Rebalancing to generate orders...")
        orders, info = self.pm.rebalance(market_data)
        
        self.assertEqual(len(orders), 1, "Should generate exactly 1 buy order")
        order = orders[0]
        
        # Verify SL/TP calculation
        self.assertEqual(order['side'], 'buy')
        self.assertIn('stop_loss_price', order)
        self.assertIn('take_profit_price', order)
        
        sl_price = order['stop_loss_price']
        tp_price = order['take_profit_price']
        
        # Default Logic: SL -2%, TP +4% (approx)
        expected_sl = 50000.0 * 0.98
        expected_tp = 50000.0 * 1.04
        
        self.logger.info(f"Generated Order: {order}")
        self.assertAlmostEqual(sl_price, expected_sl, delta=50.0, msg="SL price calculation off")
        self.assertAlmostEqual(tp_price, expected_tp, delta=50.0, msg="TP price calculation off")
        
        # --- Step 2: Execute Orders via ExecutionHandler ---
        self.logger.info("Step 2: Executing orders via ExecutionHandler...")
        
        # Need to run async method in sync test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        reports = loop.run_until_complete(self.eh.execute_trades(orders, cycle_logger=MagicMock()))
        loop.close()
        
        # Verify Execution Report
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertEqual(report['status'], 'SUCCESS')
        self.assertEqual(report['type'], 'OCO', "Should report type as OCO")
        
        # Verify Client Call
        self.mock_client.place_oco_order.assert_called_once()
        call_args = self.mock_client.place_oco_order.call_args[1]
        self.assertEqual(call_args['symbol'], symbol)
        self.assertEqual(call_args['side'], 'buy')
        self.assertEqual(call_args['stop_loss_price'], sl_price)
        self.assertEqual(call_args['take_profit_price'], tp_price)
        
        self.logger.info("Test Passed: OCO Order correctly generated and dispatched.")

if __name__ == '__main__':
    unittest.main()
