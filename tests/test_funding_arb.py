import unittest
import pandas as pd
from strategies.funding_arb import FundingRateArbitrageStrategy

class TestFundingArb(unittest.TestCase):
    def setUp(self):
        self.strategy = FundingRateArbitrageStrategy(
            positive_threshold=0.0001,
            negative_threshold=-0.0001,
            neutral_threshold=0.00005
        )
        self.df_mock = pd.DataFrame({'close': [100.0]})

    def test_positive_funding_entry(self):
        """Test entry when funding is high positive (Should Short Perp)"""
        rate = 0.0002 # > 0.0001
        signal = self.strategy.generate_signal(self.df_mock, "BTC/USDT", funding_rate=rate)
        
        # Expect -1.0 (Short Perp)
        self.assertEqual(signal, -1.0)
        self.assertEqual(self.strategy.current_state, "POSITIVE_ARB")

    def test_negative_funding_entry(self):
        """Test entry when funding is high negative (Should Long Perp)"""
        rate = -0.0002 # < -0.0001
        signal = self.strategy.generate_signal(self.df_mock, "BTC/USDT", funding_rate=rate)
        
        # Expect 1.0 (Long Perp)
        self.assertEqual(signal, 1.0)
        self.assertEqual(self.strategy.current_state, "NEGATIVE_ARB")

    def test_neutral_zone_hold(self):
        """Test hold when funding is normal"""
        rate = 0.00008 # Between 0.00005 and 0.0001
        # Signal should maintan previous state or 0 if start. 
        # Logic says: if > pos_th -> -1. if < neg_th -> 1. elif < neutral_th -> 0.
        # This 0.00008 is < pos (0.0001) but > neutral (0.00005). 
        # The current logic:
        # if > pos: ...
        # elif < neg: ...
        # elif abs < neutral: ...
        # else: ... (Implicitly, it falls through to return score=0.0 default but doesn't change state?)
        
        # Let's verify 'Gray Zone' behavior
        signal = self.strategy.generate_signal(self.df_mock, "BTC/USDT", funding_rate=rate)
        self.assertEqual(signal, 0.0)

    def test_exit_condition(self):
        """Test exit when rate returns to almost zero"""
        # First enter
        self.strategy.generate_signal(self.df_mock, "BTC/USDT", funding_rate=0.0002)
        self.assertEqual(self.strategy.current_state, "POSITIVE_ARB")
        
        # Then normalize
        rate = 0.00001 # < 0.00005
        signal = self.strategy.generate_signal(self.df_mock, "BTC/USDT", funding_rate=rate)
        
        self.assertEqual(signal, 0.0)
        self.assertEqual(self.strategy.current_state, "NEUTRAL")

if __name__ == '__main__':
    unittest.main()
