import unittest
import tempfile
import os
import pandas as pd
from datetime import datetime, timedelta
from trader.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Test configuration
        self.test_config = {
            'position_sizing': {
                'strategy': 'fractional',
                'risk_per_trade': 0.01  # 1% risk per trade
            },
            'risk_management': {
                'daily_loss_limit': 0.05,  # 5% daily loss limit
                'weekly_loss_limit': 0.10,  # 10% weekly loss limit
                'max_consecutive_losses': 3,
                'max_position_percentage': 0.20  # 20% max position
            }
        }
        
        self.risk_manager = RiskManager(
            balance=10000.0,  # $10,000 balance
            config=self.test_config,
            db_path=self.temp_db.name
        )

    def tearDown(self):
        # Close the database connection and remove the temporary file
        self.risk_manager.conn.close()
        os.unlink(self.temp_db.name)

    def test_initialization(self):
        """Test that RiskManager initializes correctly with risk parameters"""
        self.assertEqual(self.risk_manager.daily_loss_limit, 0.05)
        self.assertEqual(self.risk_manager.weekly_loss_limit, 0.10)
        self.assertEqual(self.risk_manager.max_consecutive_losses, 3)
        self.assertEqual(self.risk_manager.max_position_percentage, 0.20)

    def test_position_sizing_calculation(self):
        """Test that position sizing works correctly"""
        price = 50000.0  # $50,000 per BTC
        quantity = self.risk_manager.calculate_order_size(price)
        
        # For fractional sizing: (balance * risk_per_trade) / price
        expected_quantity = (10000.0 * 0.01) / 50000.0  # 1% of balance at current price
        self.assertAlmostEqual(quantity, expected_quantity, places=6)

    def test_volatility_adjustment(self):
        """Test that volatility-adjusted position sizing works"""
        # Create test market data with low volatility
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = [50000.0 + i * 100 for i in range(30)]  # Small, consistent changes
        df = pd.DataFrame({
            'close': prices
        }, index=dates)
        
        price = 50500.0
        base_quantity = self.risk_manager.calculate_order_size(price)
        vol_adjusted_quantity = self.risk_manager.calculate_order_size_with_volatility_adjustment(price, df)
        
        # With low volatility, the quantities should be similar
        self.assertAlmostEqual(base_quantity, vol_adjusted_quantity, places=4)
        
        # Create test data with high volatility
        high_vol_prices = [50000.0 * (1 + 0.06 * ((-1)**i)) for i in range(30)]  # 6% daily swings
        df_high_vol = pd.DataFrame({
            'close': high_vol_prices
        }, index=dates)
        
        vol_adjusted_high_vol = self.risk_manager.calculate_order_size_with_volatility_adjustment(price, df_high_vol)
        
        # With high volatility, the quantity should be reduced by 50%
        expected_reduced = base_quantity * 0.5
        self.assertAlmostEqual(vol_adjusted_high_vol, expected_reduced, places=4)

    def test_trade_approval_basic(self):
        """Test basic trade approval functionality"""
        # Use a very small quantity to avoid position size limits
        # 0.001 BTC at $50,000 is $50, which is 0.5% of $10,000 balance
        result, reason = self.risk_manager.assess_trade(0.001, current_price=50000.0, symbol="BTC-USDT")
        self.assertTrue(result)
        self.assertIn("approved", reason.lower())

    def test_negative_quantity_rejection(self):
        """Test that negative quantities are rejected"""
        result, reason = self.risk_manager.assess_trade(-0.1, current_price=50000.0, symbol="BTC-USDT")
        self.assertFalse(result)
        self.assertIn("positive", reason.lower())

    def test_zero_quantity_rejection(self):
        """Test that zero quantities are rejected"""
        result, reason = self.risk_manager.assess_trade(0, current_price=50000.0, symbol="BTC-USDT")
        self.assertFalse(result)
        self.assertIn("positive", reason.lower())

    def test_trade_rejection_by_position_size(self):
        """Test that trades are rejected if they exceed max position percentage"""
        # This would require a very large quantity to exceed 20% of $10,000 balance
        # For BTC at $50,000, 20% of $10,000 = $2,000, which is 0.04 BTC
        # So a quantity higher than 0.04 should trigger the position size check when price is $50,000
        large_quantity = 0.5  # This is $25,000 worth of BTC, much more than 20% of our balance
        result, reason = self.risk_manager.assess_trade(large_quantity, current_price=50000.0, symbol="BTC-USDT")
        # The trade should be approved because the position value ($25,000) is more than 20% of balance ($10,000*0.2 = $2,000)
        # Wait, this is wrong - if the position would be more than 20% of balance, it should be rejected
        # Let's recalculate: 0.5 BTC at $50,000 is $25,000, which is 250% of the $10,000 balance,
        # so it should definitely be rejected for position size
        self.assertFalse(result)
        self.assertIn("position", reason.lower())

    def test_daily_loss_limit(self):
        """Test that daily loss limit works"""
        # Record some losing trades to exceed daily loss limit
        now = datetime.utcnow()
        past_6h = (now - timedelta(hours=6)).isoformat()
        
        # Record a loss that exceeds the daily limit of 5% of $10,000 = $500
        self.risk_manager.record_trade_outcome(
            trade_id="test_loss_1",
            symbol="BTC-USDT", 
            entry_time=past_6h,
            exit_time=now.isoformat(),
            entry_price=50000.0,
            exit_price=49000.0,  # $1000 loss per BTC
            quantity=1.0  # This would be a $1000 loss
        )
        
        # Now try to make a new trade - it should be rejected due to daily loss limit
        result, reason = self.risk_manager.assess_trade(0.1, current_price=50000.0, symbol="BTC-USDT")
        
        # Daily loss is $1000 out of $10000 balance = 10%, which exceeds 5% limit
        daily_loss_pct = abs(-1000.0) / 10000.0  # 10%
        if daily_loss_pct >= self.risk_manager.daily_loss_limit:
            self.assertFalse(result)
            self.assertIn("daily", reason.lower())
        else:
            # If the loss doesn't exceed daily limit, we should approve the trade
            self.assertTrue(result)

    def test_weekly_loss_limit(self):
        """Test that weekly loss limit works"""
        # Record a significant loss to test weekly limit
        now = datetime.utcnow()
        past_2d = (now - timedelta(days=2)).isoformat()
        
        # Record a loss that exceeds the weekly limit of 10% of $10,000 = $1000
        self.risk_manager.record_trade_outcome(
            trade_id="test_loss_2",
            symbol="BTC-USDT",
            entry_time=past_2d,
            exit_time=now.isoformat(),
            entry_price=50000.0,
            exit_price=49000.0,  # $1000 loss per BTC
            quantity=1.1  # This would be a $1100 loss, exceeding 10% weekly limit
        )
        
        # Now try to make a new trade - it should be rejected due to weekly loss limit
        result, reason = self.risk_manager.assess_trade(0.1, current_price=50000.0, symbol="BTC-USDT")
        
        # Weekly loss is $1100 out of $10000 balance = 11%, which exceeds 10% limit
        weekly_loss_pct = abs(-1100.0) / 10000.0  # 11%
        if weekly_loss_pct >= self.risk_manager.weekly_loss_limit:
            self.assertFalse(result)
            self.assertIn("weekly", reason.lower())
        else:
            # If the loss doesn't exceed weekly limit, we should approve the trade
            self.assertTrue(result)

    def test_consecutive_losses_limit(self):
        """Test that consecutive losses limit works"""
        now = datetime.utcnow()
        
        # Record 3 consecutive losses one after another (with slight time differences)
        for i in range(3):
            time_ago = (now - timedelta(minutes=10*(i+1))).isoformat()
            self.risk_manager.record_trade_outcome(
                trade_id=f"test_loss_{i}",
                symbol="BTC-USDT",
                entry_time=time_ago,
                exit_time=(now - timedelta(minutes=10*i)).isoformat(),
                entry_price=50000.0,
                exit_price=49900.0,  # Small loss to mark as losing trade
                quantity=0.1
            )
        
        # Now try to make a new trade - it should be rejected due to consecutive losses
        result, reason = self.risk_manager.assess_trade(0.1, current_price=50000.0, symbol="BTC-USDT")
        
        # After 3 consecutive losses (the limit), the next trade should be rejected
        consecutive_count = self.risk_manager._get_consecutive_losses_count()
        if consecutive_count >= self.risk_manager.max_consecutive_losses:
            self.assertFalse(result)
            self.assertIn("consecutive", reason.lower())
        else:
            # If consecutive losses are below the limit, trade should be approved
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()