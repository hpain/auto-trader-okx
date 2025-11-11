import os
import pandas as pd
from abc import ABC, abstractmethod
from dune_client.types import QueryParameter
from dune_client.client import DuneClient
from dune_client.query import QueryBase
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OnChainClient(ABC):
    """Abstract base class for on-chain data clients."""
    @abstractmethod
    def get_data(self, query_id: str, params: dict = None) -> pd.DataFrame:
        """Fetch data from the on-chain data source."""
        pass

class DuneAnalyticsClient(OnChainClient):
    """Client for fetching data from Dune Analytics."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = DuneClient(self.api_key)
        logging.info("DuneAnalyticsClient initialized.")

    def get_data(self, query_id: str, params: dict = None) -> pd.DataFrame:
        """
        Fetch data from a Dune Analytics query.
        
        :param query_id: The query ID on Dune Analytics.
        :param params: A dictionary of parameters for the query.
        :return: A pandas DataFrame with the query results.
        """
        query_parameters = []
        if params:
            for key, value in params.items():
                query_parameters.append(QueryParameter.text_type(name=key, value=str(value)))

        # 最终修复：不再使用 run_query_dataframe，而是统一使用 execute -> poll -> fetch 流程
        # 这能解决 404 Not Found 的问题
        logging.info(f"Executing Dune query by ID: {query_id} with params: {params}")
        query = QueryBase(
            query_id=int(query_id),
            params=query_parameters
        )
        # 最终修复：调用专属的 _execute_and_wait 方法，并传入 QueryBase 对象
        return self._execute_and_wait(query_object=query)

    def get_data_from_sql(self, sql_query: str, performance: str = "medium", timeout: float = 300.0) -> pd.DataFrame:
        """
        Executes a raw SQL query on Dune and returns the results.
        This method encapsulates the execute -> poll -> fetch result workflow.

        :param sql_query: The raw SQL query string to execute.
        :param performance: The performance tier for the query (e.g., "medium").
        :param timeout: Timeout in seconds for waiting for query completion.
        :return: A pandas DataFrame with the query results.
        """
        logging.info(f"Executing raw Dune SQL query...")
        # 最终修复：调用专属的 _execute_and_wait 方法，并传入原始 SQL 字符串
        return self._execute_and_wait(sql_query=sql_query, performance=performance)

    def _execute_and_wait(self, sql_query: str = None, query_object: QueryBase = None, performance: str = "medium", timeout: float = 300.0) -> pd.DataFrame:
        """
        Private helper method to handle the execution loop for both raw SQL and Query objects.
        """
        try:
            # Step 1: Execute the query using the CORRECT method based on input type
            if sql_query:
                # 官方方式：使用 execute_sql() 执行原始 SQL 字符串
                execution = self.client.execute_sql(
                    query_sql=sql_query,
                    performance=performance
                )
            elif query_object:
                # 官方方式：使用 execute() 执行 QueryBase 对象
                execution = self.client.execute(
                    query=query_object,
                    performance=performance
                )
            else:
                logging.error("Internal error: _execute_and_wait called with no query.")
                return pd.DataFrame()

            execution_id = execution.execution_id
            logging.info(f"Dune execution started with ID: {execution_id}")

            # Step 2: Wait for the query to complete
            start_time = time.time()
            while True:
                status = self.client.get_status(execution_id)
                # 修正：处理 status.state 可能是字符串或枚举对象的情况
                state_value = status.state.value if hasattr(status.state, 'value') else status.state
                logging.info(f"Dune execution status: {state_value}")

                # 最终修复：根据 TypeError，is_execution_finished 是一个属性，而不是一个方法
                if status.is_execution_finished:
                    break

                if time.time() - start_time > timeout:
                    logging.error(f"Dune query {execution_id} timed out after {timeout} seconds.")
                    self.client.cancel_execution(execution_id)
                    return pd.DataFrame() # type: ignore
                
                time.sleep(5) # Poll every 5 seconds

            # Step 3: Fetch the results if completed successfully
            state_value = status.state.value if hasattr(status.state, 'value') else status.state
            if state_value == 'QUERY_STATE_COMPLETED':
                results_response = self.client.get_execution_results(execution_id)
                records = results_response.result.rows
                results_df = pd.DataFrame.from_records(records)
                logging.info(f"Successfully fetched {len(results_df)} rows from Dune execution {execution_id}.")
                return results_df
            else:
                # 修正：同样处理 state 可能是字符串的情况
                state_value = status.state.value if hasattr(status.state, 'value') else status.state
                logging.error(f"Dune execution {execution_id} failed with state: {state_value}")
                error_details = getattr(status, 'error', None)
                if error_details:
                    logging.error(f"Error details: {error_details}")
                return pd.DataFrame()

        except Exception as e:
            logging.error(f"Failed to execute or fetch data from Dune: {e}", exc_info=True)
            return pd.DataFrame()

class GlassnodeClient(OnChainClient):
    """
    Client for fetching data from Glassnode API.
    Note: This client is currently a placeholder and requires a valid API key 
    and implementation of the 'requests' library to function.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.glassnode.com"
        logging.info("GlassnodeClient initialized (placeholder).")

    def get_data(self, query_id: str, params: dict = None) -> pd.DataFrame:
        """
        Placeholder for fetching data from Glassnode.
        This method needs to be implemented.
        """
        logging.warning("GlassnodeClient.get_data is not implemented.")
        # This is a placeholder implementation.
        # You would typically use a library like 'requests' to make API calls.
        # Example:
        # endpoint = f"/v1/metrics/some_metric"
        # full_url = f"{self.base_url}{endpoint}"
        # response = requests.get(full_url, params={'a': 'BTC', 'api_key': self.api_key})
        # data = response.json()
        # return pd.DataFrame(data)
        return pd.DataFrame()

def get_onchain_client(config: dict) -> OnChainClient:
    """
    Factory function to get the appropriate on-chain data client based on config.
    """
    onchain_config = config.get('onchain_data', {})
    provider = onchain_config.get('provider', 'dune').lower()
    
    api_key = None
    if provider == 'dune':
        api_key = os.getenv('DUNE_API_KEY') or onchain_config.get('api_key')
        if not api_key:
            raise ValueError("Dune API key not found. Set DUNE_API_KEY environment variable or api_key in settings.yaml.")
        return DuneAnalyticsClient(api_key=api_key)
    
    elif provider == 'glassnode':
        api_key = os.getenv('GLASSNODE_API_KEY') or onchain_config.get('api_key')
        if not api_key:
            raise ValueError("Glassnode API key not found. Set GLASSNODE_API_KEY environment variable or api_key in settings.yaml.")
        return GlassnodeClient(api_key=api_key)
        
    else:
        raise ValueError(f"Unsupported on-chain data provider: {provider}")