import unittest
import logging

from trader.risk_manager import RiskManager

logging.disable(logging.CRITICAL)

class TestRiskManager(unittest.TestCase):

    def setUp(self):
        self.balance = 1000
        self.price = 50000

    def test_fixed_position_sizing(self):
        config = {'position_sizing': {'strategy': 'fixed', 'fixed_quantity': 0.005}}
        risk_manager = RiskManager(balance=self.balance, config=config)
        quantity = risk_manager.calculate_order_size(price=self.price)
        self.assertEqual(quantity, 0.005)

    def test_fractional_position_sizing(self):
        config = {'position_sizing': {'strategy': 'fractional', 'risk_per_trade': 0.02}}
        risk_manager = RiskManager(balance=self.balance, config=config)
        expected_quantity = (1000 * 0.02) / 50000
        quantity = risk_manager.calculate_order_size(price=self.price)
        self.assertAlmostEqual(quantity, expected_quantity)

    def test_assess_trade_approval(self):
        config = {'position_sizing': {'strategy': 'fixed', 'fixed_quantity': 0.01}}
        risk_manager = RiskManager(balance=self.balance, config=config)
        is_approved, reason = risk_manager.assess_trade(proposed_quantity=0.01)
        self.assertTrue(is_approved)

    def test_assess_trade_rejection_for_zero_quantity(self):
        config = {'position_sizing': {'strategy': 'fixed', 'fixed_quantity': 0.01}}
        risk_manager = RiskManager(balance=self.balance, config=config)
        is_approved, reason = risk_manager.assess_trade(proposed_quantity=0)
        self.assertFalse(is_approved)