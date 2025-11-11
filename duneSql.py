import requests
import time

API_KEY = "kYbxk5wVHTwxnj5l3IRvNgfQJ721koe6"  # 替换为你的 Key
ENDPOINT = "https://api.dune.com/api/v1/sql/execute"

def execute_sql(sql: str, performance: str = "medium"):
    headers = {
        "Content-Type": "application/json",
        "X-DUNE-API-KEY": API_KEY
    }
    body = {
        "sql": sql,
        "performance": "medium"
    }
    resp = requests.post(ENDPOINT, json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    execution_id = data.get("execution_id")
    state = data.get("state")
    print("Execution ID:", execution_id)
    print("Initial state:", state)
    return execution_id

def get_execution_results(execution_id: str, offset: int = 0, limit: int = 1000):
    url = f"https://api.dune.com/api/v1/execution/{execution_id}/results"
    params = {
        "offset": offset,
        "limit": limit
    }
    headers = {
        "X-DUNE-API-KEY": API_KEY
    }
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()

def wait_for_completion(execution_id: str, interval: float = 5.0, timeout: float = 300.0):
    url = f"https://api.dune.com/api/v1/execution/{execution_id}/status"
    headers = {
        "X-DUNE-Api-Key": API_KEY
    }
    start = time.time()
    while True:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state")
        print("Current state:", state)
        if state in ("QUERY_STATE_COMPLETED", "QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            return data
        if time.time() - start > timeout:
            raise TimeoutError("Waiting for execution timed out")
        time.sleep(interval)

if __name__ == "__main__":
    # 你的 SQL 查询语句
    sql_query = """
    SELECT
      block_time,
      hash,
      "from",
      "to",
      value
    FROM ethereum.transactions
    WHERE block_time >= NOW() - interval \'2\' day
    LIMIT 10
    """
    
    sql_query1 = """
    SELECT
      block_time,
      hash,
      "from",
      "to",
      value / 1e18 AS eth_value
    FROM ethereum.transactions
    WHERE block_time >= NOW() - INTERVAL '1' DAY
      AND value / 1e18 >= 10  -- 例如 ≥10 ETH
      ORDER BY value DESC
    LIMIT 50;

    
    """

    execution_id = execute_sql(sql_query1, performance="medium")
    # 等待执行完成
    status = wait_for_completion(execution_id)
    if status.get("state") == "QUERY_STATE_COMPLETED":
        results = get_execution_results(execution_id, offset=0, limit=10)
        print("Results metadata:", results.get("result", {}).get("metadata"))
        for row in results.get("result", {}).get("rows", []):
            print(row)
    else:
        print("Execution failed or cancelled:", status)
