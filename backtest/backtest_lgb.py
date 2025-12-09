import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.lgb_strategy import LGBStrategy
from config import config

from features.feature_engineering import generate_features

class Backtester:
    def __init__(self, data_path, strategy, initial_balance=10000, fee_rate=0.001, slippage=0.0005):
        self.data_path = data_path
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.balance = initial_balance
        self.position = 0
        self.trades = []
        self.equity_curve = []

    def load_data(self):
        # Load data (assuming CSV for now, adapt as needed)
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
            # Rename columns to match standard format
            rename_map = {
                'ts': 'timestamp', 'o': 'open', 'h': 'high', 
                'l': 'low', 'c': 'close', 'v': 'vol'
            }
            self.data = self.data.rename(columns=rename_map)
            
            # Ensure timestamp is datetime
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.set_index('timestamp', inplace=True)
        elif self.data_path.endswith('.parquet'):
            self.data = pd.read_parquet(self.data_path)
        else:
            raise ValueError("Unsupported data format")
        
        # Ensure features exist (mocking feature generation if needed)
        # In a real scenario, you'd run the feature engineering pipeline here
        print("Generating features...")
        self.data = generate_features(self.data)
        print(f"Loaded {len(self.data)} rows of data with {len(self.data.columns)} features.")

    def run(self):
        print("Generating signals...")
        signals_df = self.strategy.generate_signals(self.data)
        self.data['signal'] = signals_df['signal']
        
        print("Running backtest loop...")
        for i in range(1, len(self.data)):
            current_bar = self.data.iloc[i]
            prev_bar = self.data.iloc[i-1]
            timestamp = self.data.index[i]
            price = current_bar['close']
            signal = current_bar['signal']
            
            # Calculate equity
            current_equity = self.balance + (self.position * price)
            self.equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

            # Execute trades based on signal
            # 1: Buy, -1: Sell, 0: Hold
            
            if signal == 1 and self.position == 0:
                # Buy Entry
                quantity = (self.balance * 0.99) / price # Use 99% of balance
                cost = quantity * price
                fee = cost * self.fee_rate
                slippage_cost = cost * self.slippage
                
                if self.balance >= cost + fee:
                    self.balance -= (cost + fee)
                    self.position = quantity
                    self.trades.append({
                        'id': len(self.trades),
                        'type': 'buy',
                        'time': timestamp,
                        'price': price,
                        'quantity': quantity,
                        'fee': fee,
                        'slippage': slippage_cost
                    })
            
            elif signal == -1 and self.position > 0:
                # Sell Exit
                revenue = self.position * price
                fee = revenue * self.fee_rate
                slippage_cost = revenue * self.slippage
                
                self.balance += (revenue - fee)
                self.trades.append({
                    'id': len(self.trades),
                    'type': 'sell',
                    'time': timestamp,
                    'price': price,
                    'quantity': self.position,
                    'fee': fee,
                    'slippage': slippage_cost,
                    'pnl': self.balance - self.initial_balance # Approximate cumulative PnL
                })
                self.position = 0

        # Finalize
        final_equity = self.balance + (self.position * self.data.iloc[-1]['close'])
        print(f"Backtest completed. Final Equity: {final_equity:.2f}")
        return final_equity

    def analyze_results(self):
        if not self.trades:
            print("No trades executed.")
            return

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve).set_index('timestamp')
        
        total_trades = len(trades_df[trades_df['type'] == 'sell'])
        winning_trades = len(trades_df[(trades_df['type'] == 'sell') & (trades_df['pnl'] > 0)]) # Simplified PnL check
        # Note: accurate PnL per trade needs better tracking in the loop
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252 * 24) * (returns.mean() / returns.std()) if len(returns) > 0 else 0
        
        max_drawdown = (equity_df['equity'] / equity_df['equity'].cummax() - 1).min()

        print("\n=== Performance Report ===")
        print(f"Total Trades: {total_trades}")
        print(f"Final Equity: {self.equity_curve[-1]['equity']:.2f}")
        print(f"Return: {(self.equity_curve[-1]['equity'] - self.initial_balance) / self.initial_balance * 100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'], label='Equity')
        plt.title('Backtest Equity Curve')
        plt.legend()
        plt.savefig('backtest_results.png')
        print("Equity curve saved to backtest_results.png")

if __name__ == "__main__":
    # Example usage
    # You need to point this to a real data file with features
    data_file = "data/history/binance_BTCUSDT_1h_1y.csv" 
    
    # Mock config for strategy
    strategy_config = {
        'model_dir': 'models',
        'model_name': 'best_model.pkl',
        'metadata_name': 'metadata.json'
    }
    
    try:
        strategy = LGBStrategy(strategy_name="LGB_Backtest", config=strategy_config)
        
        if os.path.exists(data_file):
            backtester = Backtester(data_file, strategy)
            backtester.load_data()
            backtester.run()
            backtester.analyze_results()
        else:
            print(f"Data file {data_file} not found. Please generate data with features first.")
            # Create a dummy file for testing if needed
            # ...
    except Exception as e:
        print(f"Backtest failed: {e}")
