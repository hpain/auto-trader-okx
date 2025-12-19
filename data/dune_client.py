import os
import pandas as pd
from dotenv import load_dotenv

# Optional import for DuneClient
try:
    from dune_client.client import DuneClient
    from dune_client.query import QueryBase, QueryParameter
    DUNE_AVAILABLE = True
except ImportError:
    DuneClient = None
    QueryBase = None
    QueryParameter = None
    DUNE_AVAILABLE = False
    print("Warning: 'dune-client' library not found. Dune Analytics integration will be disabled.")

load_dotenv()

class DuneAnalyticsClient:
    """
    A client for interacting with the Dune Analytics API.
    """

    def __init__(self, api_key: str = None):
        """
        Initializes the Dune Analytics client.

        Args:
            api_key (str, optional): The Dune Analytics API key. 
                                     If not provided, it will be read from the DUNE_API_KEY environment variable.
        """
        if not DUNE_AVAILABLE:
            self.client = None
            print("DuneClient is not available (library missing).")
            return

        self.api_key = api_key or os.getenv("DUNE_API_KEY")
        if not self.api_key:
            # Instead of raising error, just warn and disable
             print("Warning: Dune API key not provided. Dune integration disabled.")
             self.client = None
             return

        self.client = DuneClient(self.api_key)

    def get_query_results(self, query_id: int, params: dict = None) -> pd.DataFrame:
        """
        Executes a Dune Analytics query and returns the results as a Pandas DataFrame.
        """
        if not self.client:
            print("DuneClient not initialized. Returning empty DataFrame.")
            return pd.DataFrame()
        
        query_params = []
        if params:
            for name, value in params.items():
                query_params.append(QueryParameter.text_type(name=name, value=str(value)))

        query = QueryBase(
            name="Sample Query",  # You can change this name
            query_id=query_id,
            params=query_params,
        )

        try:
            results_df = self.client.run_query_dataframe(query)
            return results_df
        except Exception as e:
            print(f"An error occurred while fetching query results: {e}")
            return pd.DataFrame()


if __name__ == '__main__':
    # Example usage:
    # 1. Make sure you have a .env file in your project root with your Dune API key:
    #    DUNE_API_KEY="your_dune_api_key"
    # 2. Replace the query_id with the ID of your Dune Analytics query.
    
    # You need to get your query ID from the URL of your query on Dune Analytics.
    # For example, if the URL is https://dune.com/queries/12345, your query ID is 12345.
    example_query_id = 1215383  # Replace with your query ID

    # If your query has parameters, you can pass them as a dictionary.
    # For example:
    # params = {"param1": "value1", "param2": "value2"}
    # dune_client = DuneAnalyticsClient()
    # results = dune_client.get_query_results(example_query_id, params=params)
    
    dune_client = DuneAnalyticsClient()
    results = dune_client.get_query_results(example_query_id)

    if not results.empty:
        print("Query Results:")
        print(results)