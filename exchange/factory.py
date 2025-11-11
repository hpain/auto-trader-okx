from typing import Any, Dict, List
from exchange.base import Exchange
from utils.config_loader import load_config # Assuming a config loader utility
import logging
import asyncio

logger = logging.getLogger(__name__)
import logging

logger = logging.getLogger(__name__)

class ExchangeFactory:
    """
    A factory class to dynamically create exchange instances.
    """
    @staticmethod
    async def create_exchange(exchange_id: str, market_type: str = 'spot', **kwargs) -> Exchange:
        """
        Creates an exchange instance based on the provided exchange_id asynchronously.

        :param exchange_id: The identifier of the exchange to create.
                            Can be 'aggregated', 'binance', 'okx', etc.
        :param market_type: The market type ('spot' or 'swap') for the exchange instance.
        :param kwargs: API credentials and other parameters.
        :return: An instance of an Exchange subclass.
        """
        exchange_id = exchange_id.lower()
        logger.info(f"Creating exchange for id: '{exchange_id}' (market_type: {market_type})")

        if exchange_id == 'aggregated':
            from exchange.aggregated_exchange import AggregatedExchange
            # Load main config to get the list of exchanges
            config = load_config('config/settings.yaml')
            primary = config['exchange']['primary_exchange']
            secondaries = config['exchange']['secondary_exchanges']
            
            # API keys from kwargs or config can be used here
            # AggregatedExchange will now have an async factory method
            return await AggregatedExchange.create_async(
                primary_exchange_id=primary,
                secondary_exchange_ids=secondaries,
                **kwargs
            )
        else:
            # For any other exchange, use the generic CcxtExchange
            from exchange.ccxt_exchange import CcxtExchange
            kwargs['exchange_id'] = exchange_id
            kwargs['market_type'] = market_type
            # Run synchronous CcxtExchange constructor in a separate thread
            return await asyncio.to_thread(CcxtExchange, **kwargs)

