
import unittest
import asyncio
import os
import pandas as pd
from dotenv import load_dotenv

from exchange.aggregated_exchange import AggregatedExchange
from exchange.factory import ExchangeFactory

# Load environment variables from .env file
load_dotenv()

class TestDuneIntegration(unittest.TestCase):

    def test_fetch_onchain_data_integration(self):
        """
        Integration test for fetching on-chain data from Dune Analytics.
        This test requires a DUNE_API_KEY to be set in the environment.
        """
        dune_api_key = os.getenv("DUNE_API_KEY")
        if not dune_api_key:
            self.skipTest("DUNE_API_KEY environment variable not set. Skipping integration test.")

        # We are only testing Dune, so we can use a dummy exchange for the primary.
        # The factory will likely fail for a dummy ID, but we can mock it just for this.
        # A better approach would be to make the exchanges optional if we only need on-chain data.
        # For now, let's just provide a valid exchange that can be initialized without keys in sandbox mode.
        primary_exchange = 'binance' 
        secondary_exchanges = []

        async def run_test():
            # Create the aggregated exchange instance
            agg_exchange = await AggregatedExchange.create_async(
                primary_exchange_id=primary_exchange,
                secondary_exchange_ids=secondary_exchanges,
                sandbox=True, # Use sandbox mode for exchanges
                dune_api_key=dune_api_key
            )

            self.assertIsNotNone(agg_exchange.dune_client, "Dune client should be initialized")

            # Use the example query ID from dune_client.py
            example_query_id = 1215383 
            
            # Fetch the on-chain data
            onchain_df = agg_exchange.fetch_onchain_data(query_id=example_query_id)

            # Perform assertions
            self.assertIsInstance(onchain_df, pd.DataFrame, "Should return a pandas DataFrame")
            self.assertFalse(onchain_df.empty, f"DataFrame should not be empty for query {example_query_id}")
            print(f"\nSuccessfully fetched {len(onchain_df)} rows from Dune query {example_query_id}.")
            print(onchain_df.head())

        # Run the async test
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
