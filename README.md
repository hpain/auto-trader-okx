# Auto Trader for OKX

An automated trading bot that uses a machine learning model (LightGBM) to predict market movements and execute trades on OKX. The system includes a robust backtesting and model evolution framework using data from Binance.

## Core Features

- **Machine Learning Driven:** Uses a LightGBM model to generate trading signals based on a rich set of technical indicators.
- **Automated Trading:** Connects to OKX to execute `buy`/`sell` orders automatically.
- **Continuous Operation:** Designed to run 24/7 with a persistent execution loop and robust error handling.
- **Robust Backtesting:** Leverages Binance for deep historical data to train and validate models.
- **Hyperparameter Optimization:** Uses Optuna for evolving and finding the best model parameters.
- **Live Monitoring:** Generates structured logs and a machine-readable status file for easy monitoring.
- **Configurable:** All major settings, including API keys and file paths, are managed in a central `settings.yaml` file.

## Project Structure

```
auto-trader-okx/
├── config/               # Configuration files (settings.yaml)
├── data/                 # Data fetching (Binance, OKX) and caching
├── features/             # Feature engineering and normalization
├── models/               # Model training, evolution, and storage
├── research/             # Scripts for model research and evolution (evolve.py)
├── trader/               # Core trading execution logic
├── utils/                # Utility scripts (including the logger)
├── logs/                 # Application log files (e.g., trader.log)
├── run_live.py           # Main entry point for continuous live/demo trading
├── status.json           # Live status report of the bot
└── requirements.txt      # Project dependencies
```

## Setup (Local)

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
    - Copy `config/settings.yaml.example` to `config/settings.yaml`.
    - Open `config/settings.yaml` and fill in your OKX API credentials. For demo trading, use your paper trading API keys.
      ```yaml
      okx:
        api_key: "YOUR_API_KEY"
        secret_key: "YOUR_SECRET_KEY"
        passphrase: "YOUR_PASSPHRASE"
        # flag: "1" for live trading, "0" for demo/paper trading
        flag: "0"
      ```

## One-Click Deployment with Docker

For a more reliable and portable setup, you can use Docker to run the application in an isolated container. This is the recommended method for deployment.

**Prerequisites:**
- [Docker](https://www.docker.com/get-started) installed on your system (Linux, Windows, or Mac).

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd auto-trader-okx
    ```

2.  **Configure Docker:**
    - Open the `docker-compose.yml` file.
    - Inside the `environment` section, replace the placeholder values with your actual OKX API credentials.
      ```yaml
      environment:
        - OKX_API_KEY=YOUR_API_KEY
        - OKX_SECRET_KEY=YOUR_SECRET_KEY
        - OKX_PASSPHRASE=YOUR_PASSPHRASE
        - OKX_FLAG=0 # Use "1" for live trading
      ```

3.  **Build and Run:**
    - Run the following command in your terminal from the project root directory:
      ```bash
      docker-compose up --build -d
      ```
    - This command will build the Docker image, create a container, and run the trading bot in the background (`-d` for detached mode).

**Monitoring the Dockerized Bot:**

- **View Logs:** To see the real-time logs from the running container:
  ```bash
  docker-compose logs -f
  ```

- **Stop the Bot:** To stop the container:
  ```bash
  docker-compose down
  ```

- **Check Status:** The `status.json` file will be created in your local project directory, so you can still check it directly.

## Usage Workflow

### 1. Model Training & Optimization

The project's workflow separates model training from trading. First, you must train a model on historical data. This is an "offline" process that finds the best model and saves it.

The main script for this process is `research/evolve.py`. It acts as the **orchestrator** for the entire training and hyperparameter optimization workflow. It fetches historical data, generates features, and uses advanced techniques (like Optuna) to find the most profitable model configuration.

It's important to understand the roles of the key scripts:
- **`research/evolve.py`**: This is the high-level script you run to find and save the best model.
- **`models/build_model.py`**: This is a low-level "model factory" used internally by `evolve.py`. You do not run it directly.

**To start training, run `research/evolve.py`:**

The `evolve.py` script accepts several command-line arguments to control the training process:

| Argument                   | Description                                                                                                | Default Value |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------- |
| `--years`                  | The number of years of historical data to download for training.                                           | `3`           |
| `--trials`                 | The number of optimization trials for Optuna to run. More trials increase the chance of finding a better model. | `100`         |
| `--ignore-local`           | If set, forces re-downloading of historical data and re-generation of features, ignoring any local caches.   | `False`       |
| `--stop-loss-pct`          | The percentage at which to set a stop-loss order to limit potential losses. (e.g., 0.02 for 2%)              | `0.02`        |
| `--take-profit-pct`        | The percentage at which to set a take-profit order. (e.g., 0.05 for 5%)                                      | `0.05`        |
| `--max-drawdown`           | The maximum acceptable drawdown during backtesting for a model to be considered valid. (e.g., 0.1 for 10%)    | `0.1`         |
| `--success-rate-threshold` | The minimum acceptable trade success rate during backtesting.                                              | `0.75`        |

**Legacy / Deprecated Arguments:**

The following arguments are still present in the script but are related to the older classification-based model logic. They are not actively used by the current regression-based evolution process in `train_evolve`.

| Argument                 | Description                                                     | Default Value |
| ------------------------ | --------------------------------------------------------------- | ------------- |
| `--models`               | Specifies model types (e.g., "rf,lgb"). Now hardcoded to LGBM.  | `rf,lgb`      |
| `--profit-threshold`     | The profit target used to create the binary `y` label.          | `0.005`       |
| `--confidence-threshold` | The confidence level required for a classification model to trade. | `0.95`        |

**Example Usage:**

To run a short optimization for 50 trials using 2 years of data:
```bash
python research/evolve.py --years 2 --trials 50
```

To run an extensive search with 1000 trials, forcing a full data refresh:
```bash
python research/evolve.py --trials 1000 --ignore-local
```

The best model will be saved as `models/best_model.pkl`.

### 2. Deployment & Monitoring

Once you have a trained model, you can start the bot for continuous (demo or live) trading using either the local setup or the recommended Docker method.

**To run the bot (local setup):**

```bash
python run_live.py
```

This script runs an infinite loop that will:
1.  Call the core trading logic from `trader/executor.py`.
2.  Handle any errors gracefully without crashing the main loop.
3.  Sleep for the duration specified by the `interval` in your config file (e.g., 1 hour for "1H") before starting the next cycle.

**Monitoring the bot (local setup):**

There are two ways to monitor the bot's activity:

- **Detailed Logs:** For real-time, detailed information about every step the bot takes (fetching data, generating signals, placing orders, errors, etc.), you can inspect the log file.
  ```bash
  # On Linux/macOS
  tail -f logs/trader.log
  
  # On Windows (PowerShell)
  Get-Content logs/trader.log -Wait
  ```

- **Status Summary:** For a high-level, at-a-glance view of the bot's last known state, you can check the `status.json` file. This file is updated at the end of every trading cycle and contains information like the last signal, action taken, and current account equity.

## Configuration

Configuration can be managed in two ways:

1.  **`config/settings.yaml` (for local runs):** This file is used for local development.
2.  **Environment Variables (for Docker):** When running with Docker, settings are controlled via the `environment` section in `docker-compose.yml`. These variables will override any values present in `settings.yaml`.

- **`okx` / `OKX_*`**: Your API credentials and trading mode (`flag`).
- **`trade` / `TRADE_*`**: Trading parameters like the symbol (`BTC-USDT`) and order quantity.
- **`paths`**: Centralized paths for storing models, cached data, etc. (Generally not changed).

---
*Disclaimer: Trading cryptocurrencies involves significant risk. This bot is provided for educational purposes only. Use it at your own risk.*
