# Auto Trader for OKX

An automated trading bot that uses a machine learning model (LightGBM) to predict market movements and execute trades on OKX. The system includes a robust backtesting and model evolution framework using data from Binance.

## Core Features

- **Machine Learning Driven:** Uses a LightGBM model to generate trading signals based on a rich set of technical indicators.
- **Automated Trading:** Connects to OKX to execute `buy`/`sell` orders automatically.
- **Robust Backtesting:** Leverages Binance for deep historical data to train and validate models.
- **Hyperparameter Optimization:** Uses Optuna for evolving and finding the best model parameters.
- **Fallback Strategy:** Includes a simple Moving Average (MA) strategy as a fallback if the ML model fails.
- **Configurable:** All major settings, including API keys and file paths, are managed in a central `settings.yaml` file.

## Project Structure

```
auto-trader-okx/
├── config/               # Configuration files (settings.yaml)
├── data/                 # Data fetching (Binance, OKX) and caching
├── features/             # Feature engineering and normalization
├── models/               # Model training, evolution, and storage
├── research/             # Scripts for model research and evolution (evolve.py)
├── trader/               # Live trading execution logic
├── utils/                # Utility scripts
├── main.py               # Main entry point for the live trader
└── requirements.txt      # Project dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd auto-trader-okx
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the bot:**
    - It is recommended to copy `config/settings.yaml` to `config/settings.yaml.example` and add it to your `.gitignore`.
    - Open `config/settings.yaml` and fill in your OKX API credentials:
      ```yaml
      okx:
        api_key: "YOUR_API_KEY"
        secret_key: "YOUR_SECRET_KEY"
        passphrase: "YOUR_PASSPHRASE"
        # flag: "1" for live trading, "0" for demo/paper trading
        flag: "0"
      ```

## Usage

### 1. Model Training

The core of the trading strategy is the machine learning model. You need to train it first using historical data. The `research/evolve.py` script is used for this purpose. It fetches data from Binance, engineers features, and uses Optuna to find the best model.

**Example command:**

```bash
python research/evolve.py --years 2 --trials 500 --models lgb --profit-threshold 0.003 --stop-loss-pct 0.022
```

**Key Arguments:**
- `--years`: Number of years of historical data to fetch from Binance.
- `--trials`: Number of optimization trials for Optuna to run.
- `--models`: Comma-separated list of models to train (e.g., `lgb`, `rf`).
- `--ignore-local`: Force re-downloading of historical data and re-generation of feature caches, ignoring local files.
- `--profit-threshold`: The minimum profit target for a trade to be considered successful in the backtest.
- `--confidence-threshold`: The minimum model confidence probability required to execute a trade.
- `--stop-loss-pct`: The stop-loss percentage for the backtesting strategy.
- `--max-drawdown`: The maximum drawdown limit for the backtesting period.
- `--success-rate-threshold`: The minimum success rate of trades for a trial to be considered successful.
- `--patience`: (Currently unused) Optuna early stopping patience.

The best model will be saved as `models/best_model.pkl`.

### 2. Live Trading

Once you have a trained model, you can start the live trading bot.

**To run the bot:**

```bash
python main.py
```

The bot will:
1.  Load the best model from `models/best_model.pkl`.
2.  Fetch the latest market data from OKX.
3.  Generate a prediction (`buy`, `sell`, or `hold`).
4.  Place an order on OKX if a `buy` or `sell` signal is generated.
5.  If the model cannot be loaded or fails, it will fall back to a simple Moving Average strategy.

## Configuration (`config/settings.yaml`)

- **`okx`**: Your API credentials and trading mode (`flag`).
- **`trade`**: Trading parameters like the symbol (`BTC-USDT`) and order quantity.
- **`paths`**: Centralized paths for storing models, cached data, etc. You shouldn't need to change these unless you restructure the project.

---
*Disclaimer: Trading cryptocurrencies involves significant risk. This bot is provided for educational purposes only. Use it at your own risk.*