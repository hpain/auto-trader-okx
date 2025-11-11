import unittest
import pandas as pd
from unittest.mock import MagicMock, patch

# 在导入我们自己的模块之前，先 mock 掉外部依赖。
# 最终修复：不再导入 GetStatusResponse 或 ExecutionStatus，因为版本不确定。
# 我们将直接 mock get_execution_status 方法的返回值。
from dune_client.models import ExecutionResult, DuneError

# 现在可以安全地导入我们的客户端了
from data.onchain import DuneAnalyticsClient

class TestDuneAnalyticsClient(unittest.TestCase):

    def setUp(self):
        """在每个测试开始前，初始化一个带有 mock 客户端的 DuneAnalyticsClient 实例"""
        self.api_key = "test_api_key"
        self.client = DuneAnalyticsClient(api_key=self.api_key)
        # 创建一个 mock 对象来替换真实的 dune_client.client.DuneClient
        self.mock_dune_raw_client = MagicMock()
        self.client.client = self.mock_dune_raw_client

    def test_get_data_by_query_id_success(self):
        """测试通过 query_id 成功获取数据"""
        # 最终修复：重构此测试以匹配 get_data 的新实现（调用 get_data_from_sql）
        print("\n--- Running test_get_data_by_query_id_success ---")
        
        # 模拟 "执行 -> 轮询 -> 获取结果" 流程
        mock_execution_id = "execution_abc"
        mock_execution_response = MagicMock()
        mock_execution_response.execution_id = mock_execution_id

        mock_status_response = MagicMock()
        mock_status_response.state = 'QUERY_STATE_COMPLETED'
        mock_status_response.is_execution_finished.return_value = True

        mock_results_response = ExecutionResult(rows=[{'value': 100}, {'value': 200}], metadata=None)
        mock_get_results_response = MagicMock()
        mock_get_results_response.result = mock_results_response

        self.mock_dune_raw_client.execute.return_value = mock_execution_response
        self.mock_dune_raw_client.get_execution_status.return_value = mock_status_response
        self.mock_dune_raw_client.get_execution_results.return_value = mock_get_results_response

        # 调用被测试的方法
        result_df = self.client.get_data(query_id="12345")

        # 断言：检查 execute 是否被正确调用
        self.mock_dune_raw_client.execute.assert_called_once()
        # 断言：检查返回的 DataFrame 是否正确
        self.assertFalse(result_df.empty)
        self.assertEqual(len(result_df), 2)
        print("OK: Successfully tested fetching data by query_id.")

    def test_get_data_by_query_id_failure(self):
        """测试通过 query_id 获取数据时发生 API 异常"""
        # 最终修复：重构此测试以匹配 get_data 的新实现
        print("\n--- Running test_get_data_by_query_id_failure ---")
        # 配置 mock 对象，使其在被调用时抛出异常
        self.mock_dune_raw_client.execute.side_effect = Exception("Dune API Error")

        # 调用被测试的方法
        result_df = self.client.get_data(query_id="12345")

        # 断言：检查返回的是否为一个空的 DataFrame
        self.assertTrue(result_df.empty)
        print("OK: Correctly handled API failure by returning an empty DataFrame.")

    def test_get_data_from_sql_success(self):
        """测试通过原始 SQL 成功获取数据（模拟执行->完成->获取结果的流程）"""
        print("\n--- Running test_get_data_from_sql_success ---")
        # 准备 mock 的 API 响应
        mock_execution_id = "execution_xyz"
        mock_execution_response = MagicMock()
        mock_execution_response.execution_id = mock_execution_id
        
        # 最终修复：直接创建一个 mock 对象来模拟 status 对象
        # 我们只需要确保它有 .state.value 和 .is_execution_finished() 即可
        mock_status_response = MagicMock()
        # 模拟 status.state.value == 'QUERY_STATE_COMPLETED'
        mock_status_response.state.value = 'QUERY_STATE_COMPLETED'
        mock_status_response.state = 'QUERY_STATE_COMPLETED' # 直接模拟 state 为字符串
        # 模拟 status.is_execution_finished() 返回 True
        mock_status_response.is_execution_finished.return_value = True
        
        # 模拟返回的查询结果
        # 修复 TypeError: ExecutionResult.__init__() missing 1 required positional argument: 'metadata'
        # 为 ExecutionResult 的构造函数添加 metadata=None
        mock_results_response = ExecutionResult(rows=[{'value': 500}, {'value': 600}], metadata=None)
        mock_get_results_response = MagicMock()
        mock_get_results_response.result = mock_results_response

        # 配置 mock 对象的行为
        self.mock_dune_raw_client.execute.return_value = mock_execution_response
        self.mock_dune_raw_client.get_execution_status.return_value = mock_status_response
        self.mock_dune_raw_client.get_execution_results.return_value = mock_get_results_response

        # 调用被测试的方法
        sql_query = "SELECT 1"
        result_df = self.client.get_data_from_sql(sql_query)

        # 断言
        self.mock_dune_raw_client.execute.assert_called_once_with(query=sql_query, performance="medium")
        self.mock_dune_raw_client.get_execution_status.assert_called_once_with(mock_execution_id)
        self.mock_dune_raw_client.get_execution_results.assert_called_once_with(mock_execution_id)
        
        self.assertFalse(result_df.empty)
        self.assertEqual(len(result_df), 2)
        self.assertEqual(result_df['value'].iloc[0], 500)
        print("OK: Successfully tested fetching data via raw SQL.")

    def test_get_data_from_sql_failure_state(self):
        """测试通过原始 SQL 查询时，查询执行失败"""
        print("\n--- Running test_get_data_from_sql_failure_state ---")
        # 最终修复：直接创建一个 mock 对象来模拟失败的状态
        mock_status_response_failed = MagicMock()
        mock_status_response_failed.state.value = 'QUERY_STATE_FAILED'
        mock_status_response_failed.is_execution_finished.return_value = True
        mock_status_response_failed.state = 'QUERY_STATE_FAILED' # 直接模拟 state 为字符串
        # 模拟错误详情
        # 最终修复：不再创建 DuneError 实例，因为它在日志记录时会出错。
        # 只需创建一个可以被字符串化的 mock 对象即可。
        mock_error_obj = MagicMock()
        mock_error_obj.__str__.return_value = "Mocked DuneError: Execution failed: Syntax error"
        mock_status_response_failed.error = mock_error_obj

        self.mock_dune_raw_client.execute.return_value.execution_id = "execution_fail"
        self.mock_dune_raw_client.get_execution_status.return_value = mock_status_response_failed

        result_df = self.client.get_data_from_sql("SELECT * FROM non_existent_table")

        self.assertTrue(result_df.empty)
        print("OK: Correctly handled failed execution state by returning an empty DataFrame.")

if __name__ == '__main__':
    unittest.main()