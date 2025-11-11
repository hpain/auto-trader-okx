import unittest
import pandas as pd
from analysis.market_regime_detection import detect_regime_change

class TestMarketRegimeDetection(unittest.TestCase):

    def test_detect_regime_change_no_change(self):
        # Create a DataFrame with stable price data
        data = {'close': [100, 100.1, 99.9, 100.2, 100.0]}
        df = pd.DataFrame(data)
        
        # Call detect_regime_change and assert it returns False
        self.assertFalse(detect_regime_change(df["close"]), "Should not detect regime change in stable data")

    def test_detect_regime_change_sudden_increase(self):
        # Create a DataFrame with a sudden price increase
        data = {'close': [100, 101, 102, 103, 110]}
        df = pd.DataFrame(data)
        
        # Call detect_regime_change and assert it returns True
        self.assertTrue(detect_regime_change(df["close"]), "Should detect regime change after a sudden increase")

    def test_detect_regime_change_sudden_decrease(self):
        # Create a DataFrame with a sudden price decrease
        data = {'close': [100, 99, 98, 97, 90]}
        df = pd.DataFrame(data)
        
        # Call detect_regime_change and assert it returns True
        self.assertTrue(detect_regime_change(df["close"]), "Should detect regime change after a sudden decrease")

    def test_detect_regime_change_oscillating(self):
        # Create a DataFrame with oscillating price data
        data = {'close': [100, 105, 95, 110, 90]}
        df = pd.DataFrame(data)
        
        # Call detect_regime_change and assert it returns True
        self.assertTrue(detect_regime_change(df["close"]), "Should detect regime change in oscillating data")

    def test_detect_regime_change_with_realistic_data(self):
        # Create a DataFrame with more realistic, but still clearly changing, data
        data = {'close': [
            29700, 29750, 29800, 29850, 29900,  # Gradual increase
            29880, 29860, 29840, 29820, 29800,  # Slight decrease
            29700, 29600, 29500, 29400, 29300   # Significant decrease
        ]}
        df = pd.DataFrame(data)

        # Call detect_regime_change and assert it returns True
        self.assertTrue(detect_regime_change(df["close"]), "Should detect regime change with mixed but trending data")

    def test_detect_regime_change_with_minimal_realistic_data(self):
        # Create a DataFrame with very few data points and a clear change
        data = {'close': [30000, 30010, 29900]}  # Slight increase then a big drop
        df = pd.DataFrame(data)

        # Call detect_regime_change and assert it returns True
        self.assertTrue(detect_regime_change(df["close"]), "Should reliably detect regime change even with few points")

    def test_detect_regime_change_with_very_stable_data(self):
        # Create a DataFrame with almost constant data
        data = {'close': [150.01, 150.02, 150.01, 150.02, 150.01]}
        df = pd.DataFrame(data)

        # It should still be stable even with the slight variations
        self.assertFalse(detect_regime_change(df["close"]), "Should remain stable with nearly constant data")

if __name__ == '__main__':
    unittest.main()