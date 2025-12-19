from typing import Any, Dict, List
import os
from exchange.base import Exchange
from utils.config_loader import load_config
import logging
import asyncio
from exchange.ccxt_exchange import CcxtExchange
# from exchange.aggregated_exchange import AggregatedExchange # Moved inside method to avoid circular import
from exchange.mock_exchange import MockExchange

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

        mock = kwargs.pop('mock', False)
        if mock:
            # Pass exchange_id and market_type to MockExchange
            return MockExchange(exchange_id=exchange_id, market_type=market_type, **kwargs)

        # Auto-load credentials from env if not provided
        if 'api_key' not in kwargs or not kwargs['api_key']:
            env_prefix = exchange_id.upper()
            kwargs['api_key'] = os.environ.get(f'{env_prefix}_API_KEY')
            kwargs['api_secret'] = os.environ.get(f'{env_prefix}_SECRET_KEY')
            kwargs['passphrase'] = os.environ.get(f'{env_prefix}_PASSPHRASE')
            
            if kwargs['api_key']:
                logger.info(f"Loaded credentials for {exchange_id} from environment variables.")

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
            kwargs['exchange_id'] = exchange_id
            kwargs['market_type'] = market_type
            # 直接在当前事件循环中实例化，避免跨线程导致的事件循环绑定问题
            # CcxtExchange 的初始化主要是配置设置，不会阻塞
            exchange = CcxtExchange(**kwargs)
            # 如果需要自动加载市场，可以在这里调用 await exchange.load()
            # 但为了保持工厂方法的纯粹性，我们返回实例，由调用者决定何时加载
            # (注意：AggregatedExchange 会自动调用 load)
            return exchange
