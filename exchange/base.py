from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd

class Exchange(ABC):
    """
    Abstract base class for exchange interfaces.
    """

    @abstractmethod
    def __init__(self, api_key: str = None, api_secret: str = None, passphrase: str = None, sandbox: bool = True):
        pass

    @abstractmethod
    def fetch_candles(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles.
        """
        pass

    @abstractmethod
    def fetch_historical_data(self, symbol: str, timeframe: str, years: float) -> pd.DataFrame:
        """
        Fetch a large amount of historical data for backtesting or training.
        """
        pass

    @abstractmethod
    def fetch_funding_rates(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch funding rates. Can fetch historical data using 'years' or recent data using 'since' and 'limit'.
        """
        pass

    @abstractmethod
    def fetch_open_interest(self, symbol: str, timeframe: str, years: Optional[float] = None, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch open interest data. Can fetch historical data using 'years' or recent data using 'since' and 'limit'.
        """
        pass

    @abstractmethod
    def get_balance(self, currency: str) -> float:
        """
        Get the balance for a specific currency.
        """
        pass

    @abstractmethod
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a new order.
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get details of a specific order.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an order.
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price of a symbol.
        """
        pass

    @abstractmethod
    def place_oco_order(self, symbol: str, side: str, amount: float, take_profit_price: float, stop_loss_price: float) -> Dict[str, Any]:
        """
        Place a One-Cancels-the-Other order.
        """
        pass
