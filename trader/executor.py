# trader/executor.py
import json
import os
import logging
from datetime import datetime
import pandas as pd
from joblib import load

from utils.config_loader import load_config
from strategies.moving_average import generate_signal as ma_signal, plot_moving_averages
from exchange.okx_exchange import OKXExchange
from trader.risk_manager import RiskManager
from utils.logger import setup_logger
from trader.state_manager import StateManager
from trader.trade_tracker import TradeTracker
from trader.risk_monitor import RiskMonitor


def _write_status(status_data: dict):
    """Write the status dictionary to a JSON file."""
    try:
        # Ensure all values are serializable, especially datetime objects
        for key, value in status_data.items():
            if isinstance(value, (pd.Timestamp, datetime)):
                status_data[key] = value.isoformat()

        with open("status.json", "w", encoding="utf-8") as f:
            json.dump(status_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Failed to write status.json file: {e}", exc_info=True)


def _load_model(config: dict):
    """Load model and metadata using the provided config."""
    try:
        model_dir = config["paths"]["model_dir"]
        model_path = os.path.join(model_dir, "best_model.pkl")
        meta_path = os.path.join(model_dir, "metadata.json")
        pipe = load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return pipe, meta
    except FileNotFoundError:
        return None, None
    except Exception:
        logging.error("Failed to load model due to an unexpected error:", exc_info=True)
        return None, None

def run():
    setup_logger()
    state_manager = None  # Initialize to ensure it's available in the finally block
    try:
        config = load_config('config/settings.yaml')
        state_manager = StateManager(db_path=config.get('paths', {}).get('database_path', 'trader_state.db'))
    except FileNotFoundError as e:
        logging.error(f"CRITICAL: {e}. The application cannot start without a configuration file.")
        return
    except Exception as e:
        logging.error(f"CRITICAL: Failed to initialize StateManager: {e}. Application cannot start.", exc_info=True)
        return

    status_report = {
        "last_update": datetime.utcnow().isoformat(),
        "signal": "N/A",
        "outcome": "Cycle starting",
        "account_equity": "N/A"
    }

    # Use the ExchangeFactory to create the client
    from exchange.factory import ExchangeFactory
    api_credentials = config.get('okx', {}) # Credentials can be generalized later
    client = ExchangeFactory.create_exchange(
        'aggregated',
        api_key=api_credentials.get('api_key'),
        api_secret=api_credentials.get('secret_key'),
        passphrase=api_credentials.get('passphrase'),
        sandbox=api_credentials.get('sandbox', True)
    )

    # Initialize TradeTracker for enhanced monitoring
    trade_tracker = TradeTracker(db_path=config.get('paths', {}).get('database_path', 'trader_state.db'))
    
    # Initialize RiskMonitor for real-time risk monitoring and alerts
    risk_monitor = RiskMonitor(
        db_path=config.get('paths', {}).get('database_path', 'trader_state.db'),
        config=config.get('risk_monitoring', {})  # Get risk monitoring config from settings
    )

    try:
        trading_config = config['trading']
        symbol = trading_config['symbol']
        interval = trading_config['interval']
        live_strategy = trading_config['strategy']

        # --- State Recovery Check ---
        open_orders = state_manager.get_open_orders(symbol)
        if open_orders:
            logging.warning(f"Found {len(open_orders)} open orders from a previous run for {symbol}. Aborting.")
            logging.warning("Manual intervention required to check order status on the exchange.")
            status_report["outcome"] = "Safety Stop: Found previous open orders."
            return

        current_position = state_manager.get_position(symbol)
        if current_position:
            logging.info(f"Found existing position from previous run: {current_position}")
        # --- End State Recovery Check ---

        equity = client.get_balance('USDT')
        if equity is None:
            logging.error("Could not retrieve equity. Aborting run.")
            status_report["outcome"] = "Error: Could not retrieve equity."
            return
        
        risk_config = {
            "position_sizing": config.get("position_sizing", {}),
            "risk_management": config.get("risk_management", {})
        }
        risk_manager = RiskManager(balance=equity, config=risk_config, db_path=config.get('paths', {}).get('database_path', 'trader_state.db'))
        status_report["account_equity"] = equity

        df = client.fetch_candles(symbol=symbol, timeframe=interval, limit=400)
        if df is None or df.empty:
            logging.error("Failed to get kline data, aborting run.")
            status_report["outcome"] = "Error: Failed to get kline data"
            return

        current_price = df['close'].iloc[-1]
        # Use volatility-adjusted position sizing
        quantity = risk_manager.calculate_order_size_with_volatility_adjustment(price=current_price, market_data=df)
        logging.info(f"Calculated quantity by RiskManager: {quantity}")

        if quantity <= 0:
            logging.warning("Calculated quantity is 0, no trade will be placed.")

        ma_params = config['strategy_params']['ma_crossover']['live_fallback']
        short_window = ma_params['short_window']
        long_window = ma_params['long_window']
        df['ma_short'] = df['close'].rolling(window=short_window, min_periods=short_window).mean()
        df['ma_long'] = df['close'].rolling(window=long_window, min_periods=long_window).mean()
        plot_moving_averages(df, symbol)

        sig = "hold"
        
        logging.info(f"Executing live strategy: '{live_strategy}'")
        if live_strategy == 'ml':
            pipe, meta = _load_model(config)
            if not pipe or not meta:
                logging.warning("ML strategy selected, but model not found. Using MA fallback.")
                sig = ma_signal(df)
            else:
                feature_cols = meta["feature_cols"]
                missing = [c for c in feature_cols if c not in df.columns]
                if missing:
                    logging.warning(f"Model feature columns missing: {missing}. Using MA fallback.")
                    sig = ma_signal(df)
                else:
                    x = df[feature_cols].tail(1)
                    try:
                        proba = float(pipe.predict_proba(x)[:, 1][0])
                        conf_threshold = meta["best_params"].get("confidence_threshold", 0.5)
                        
                        logging.info(f"Model probability: {proba:.4f}, Confidence Threshold: {conf_threshold:.4f}")
                        if proba >= conf_threshold:
                            sig = "buy"
                        else:
                            sig = "hold"

                    except Exception as e:
                        logging.error(f"Model inference failed: {e}. Using MA fallback.", exc_info=True)
                        sig = ma_signal(df)

        elif live_strategy == 'ma_crossover':
            logging.info("Using MA crossover strategy as configured.")
            sig = ma_signal(df)
        else:
            logging.error(f"Unknown strategy '{live_strategy}' in config. Using MA fallback.")
            sig = ma_signal(df)

        logging.info(f"Final Signal: {sig.upper()}")
        status_report["signal"] = sig.upper()

        # If we already have a position, a 'buy' signal means 'hold'
        if current_position and sig == "buy":
            logging.info(f"Already in a position for {symbol}, treating 'buy' signal as 'hold'.")
            sig = "hold"
        
        # If we are not in a position, a 'sell' signal means 'hold'
        if not current_position and sig == "sell":
            logging.info(f"Not in a position for {symbol}, treating 'sell' signal as 'hold'.")
            sig = "hold"

        is_approved, reason = risk_manager.assess_trade(quantity, current_price=current_price, symbol=symbol)
        if not is_approved:
            logging.warning(f"Trade rejected by RiskManager: {reason}")
            status_report["outcome"] = f"Trade rejected: {reason}"
            return

        # --- Trade Execution and State Update ---
        if sig == "buy" and quantity > 0:
            risk_params = config['risk_management']
            stop_loss_price = current_price * (1 - risk_params['stop_loss_pct'])
            take_profit_price = current_price * (1 + risk_params['take_profit_pct'])
            
            order_result = client.place_oco_order(symbol, "buy", quantity, take_profit_price, stop_loss_price)

            if order_result and order_result.get('code') == '0' and order_result.get('data'):
                order_id = order_result['data'][0]['ordId']
                logging.info(f"Successfully placed BUY order with ID: {order_id}")
                order_data = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': 'buy',
                    'quantity': quantity,
                    'price': current_price,
                    'status': 'open' # OCO orders are open until triggered
                }
                state_manager.upsert_order(order_data)
                state_manager.update_position(symbol, quantity, current_price)
                
                # Record the trade entry for enhanced monitoring and risk management
                from datetime import datetime
                risk_manager.record_trade_outcome(
                    trade_id=order_id,
                    symbol=symbol,
                    entry_time=datetime.utcnow().isoformat(),
                    exit_time=None,  # Will be updated when OCO order executes and fills
                    entry_price=current_price,
                    exit_price=None,  # Will be updated when OCO order executes and fills
                    quantity=quantity
                )
                
                # Use TradeTracker for additional monitoring
                trade_tracker.record_trade_entry(
                    trade_id=order_id,
                    symbol=symbol,
                    side='buy',
                    entry_price=current_price,
                    quantity=quantity
                )
                
                status_report["outcome"] = f"Placed BUY order {order_id} for {quantity} {symbol}."
            else:
                error_code = order_result.get('data', [{}])[0].get('sCode', 'N/A')
                error_msg = order_result.get('data', [{}])[0].get('sMsg', 'Unknown error')
                logging.error(f"Failed to place BUY order. Code: {error_code}, Msg: {error_msg}")
                status_report["outcome"] = f"Failed to place BUY order: {error_msg}"

        elif sig == "sell" and quantity > 0 and current_position:
            logging.info("Sell signal received. Placing a market sell order to close position.")
            order_result = client.create_order(symbol=symbol, order_type='market', side='sell', amount=quantity)

            if order_result and order_result.get('code') == '0' and order_result.get('data'):
                order_id = order_result['data'][0]['ordId']
                logging.info(f"Successfully placed SELL order with ID: {order_id}")
                order_data = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': quantity,
                    'price': current_price,
                    'status': 'filled' # Market orders are assumed to be filled quickly
                }
                state_manager.upsert_order(order_data)
                state_manager.update_position(symbol, 0, 0) # Close position
                
                # Record the completed trade outcome for risk management
                from datetime import datetime
                # For sell orders (position closures), use the entry price from the existing position
                entry_price = current_position['entry_price'] if current_position and 'entry_price' in current_position else current_price
                risk_manager.record_trade_outcome(
                    trade_id=order_id,
                    symbol=symbol,
                    entry_time=current_position.get('last_update', datetime.utcnow().isoformat()) if current_position else datetime.utcnow().isoformat(),  # Use position's last update time as entry time
                    exit_time=datetime.utcnow().isoformat(),
                    entry_price=entry_price,
                    exit_price=current_price,
                    quantity=quantity
                )
                
                # Use TradeTracker to record trade exit with P&L calculation
                trade_tracker.record_trade_exit(
                    trade_id=order_id,
                    exit_price=current_price
                )
                
                status_report["outcome"] = f"Placed SELL order {order_id} to close position."
            else:
                error_code = order_result.get('data', [{}])[0].get('sCode', 'N/A')
                error_msg = order_result.get('data', [{}])[0].get('sMsg', 'Unknown error')
                logging.error(f"Failed to place SELL order. Code: {error_code}, Msg: {error_msg}")
                status_report["outcome"] = f"Failed to place SELL order: {error_msg}"
        else:
            logging.info("Signal is 'hold' or quantity is zero, no order placed.")
            status_report["outcome"] = "Signal is 'hold' or quantity is zero, no order placed."

    except Exception as e:
        logging.exception(f"An unhandled error occurred during the run cycle: {e}")
        status_report["outcome"] = f"Error: {e}"
    finally:
        logging.info("Updating final status report.")
        if status_report.get("account_equity") == "N/A":
             try:
                equity = client.get_balance('USDT')
                status_report["account_equity"] = equity if equity is not None else "Failed to retrieve"
             except Exception as e:
                logging.error(f"Could not retrieve account equity for status report: {e}")
                status_report["account_equity"] = "Error retrieving equity"

        # Check for any unresolved risk events and update status report
        unresolved_risk_events = trade_tracker.get_unresolved_risk_events()
        if unresolved_risk_events:
            logging.warning(f"Found {len(unresolved_risk_events)} unresolved risk events")
            # Add risk event information to status report
            status_report["risk_events"] = len(unresolved_risk_events)
            status_report["latest_risk_event"] = unresolved_risk_events[0]['event_description'] if unresolved_risk_events else "None"
        else:
            status_report["risk_events"] = 0
            status_report["latest_risk_event"] = "None"

        # Run risk monitoring and alerts
        try:
            # Get current equity to pass to risk monitor
            current_equity = status_report.get("account_equity", "N/A")
            if current_equity != "N/A" and current_equity != "Failed to retrieve" and current_equity != "Error retrieving equity":
                risk_alerts = risk_monitor.monitor_and_alert(current_balance=current_equity)
                if risk_alerts:
                    logging.info(f"Detected {len(risk_alerts)} risk conditions")
                    status_report["active_risk_alerts"] = len(risk_alerts)
                    status_report["latest_risk_alert"] = risk_alerts[0]['message'] if risk_alerts else "None"
                else:
                    status_report["active_risk_alerts"] = 0
                    status_report["latest_risk_alert"] = "None"
                
                # Add performance summary to status report
                perf_summary = risk_monitor.get_performance_summary()
                status_report["performance_summary"] = perf_summary
            else:
                status_report["active_risk_alerts"] = "N/A (Equity not available)"
                status_report["latest_risk_alert"] = "N/A (Equity not available)"
        except Exception as e:
            logging.error(f"Error during risk monitoring: {e}", exc_info=True)
            status_report["active_risk_alerts"] = "Error in monitoring"
            status_report["latest_risk_alert"] = f"Error: {str(e)}"

        status_report["last_update"] = datetime.utcnow().isoformat() + "Z"
        _write_status(status_report)
        logging.info("Status file 'status.json' has been updated.")

        if state_manager:
            state_manager.close()
            logging.info("StateManager connection closed.")
        
        # Close trade tracker connection
        if hasattr(trade_tracker, 'conn'):
            trade_tracker.conn.close()
            logging.info("TradeTracker connection closed.")
        
        # Close risk monitor connection
        if hasattr(risk_monitor, 'trade_tracker') and hasattr(risk_monitor.trade_tracker, 'conn'):
            risk_monitor.trade_tracker.conn.close()
            logging.info("RiskMonitor connection closed.")
