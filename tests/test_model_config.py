# tests/test_model_config.py
# --- BEGIN VADER SENTIMENT MONKEY-PATCH ---
# This patch fixes a compatibility issue between older versions of
# vaderSentiment and newer Python versions (3.10+). The issue causes a
# TypeError during the lexicon file reading. We replace the faulty
# __init__ method with a corrected one.
try:
    import os
    import sys
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    def patched_vader_init(self, lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt"):
        # Correctly locate the lexicon files relative to the vaderSentiment package
        vader_module = sys.modules[SentimentIntensityAnalyzer.__module__]
        vader_path = os.path.dirname(os.path.abspath(vader_module.__file__))
        
        # Load Lexicon
        lexicon_full_filepath = os.path.join(vader_path, lexicon_file)
        with open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon = {}
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.lexicon[parts[0]] = float(parts[1])

        # Load Emoji Lexicon
        emoji_full_filepath = os.path.join(vader_path, emoji_lexicon)
        with open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_lexicon = {}
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.emoji_lexicon[parts[0]] = parts[1]

    SentimentIntensityAnalyzer.__init__ = patched_vader_init
except Exception as e:
    print(f"Could not apply VADER monkey-patch: {e}")
# --- END VADER SENTIMENT MONKEY-PATCH ---
import unittest
import pandas as pd
import numpy as np

# 导入要测试的配置文件
import config.model_config as model_config

# 导入用于一致性检查的特征生成函数
from features.feature_engineering import generate_features

class TestModelConfig(unittest.TestCase):

    def test_config_import_and_types(self):
        """
        测试：配置文件可以被导入，且各项类型正确。
        """
        self.assertIsInstance(model_config.Y_COL, str, "Y_COL 应该是一个字符串")
        self.assertIsInstance(model_config.FEATURES, list, "FEATURES 应该是一个列表")
        self.assertIsInstance(model_config.MODEL_PARAMS, dict, "MODEL_PARAMS 应该是一个字典")
        self.assertIsInstance(model_config.DIRECTION, str, "DIRECTION 应该是一个字符串")
        self.assertIsInstance(model_config.METRIC_TO_OPTIMIZE, str, "METRIC_TO_OPTIMIZE 应该是一个字符串")
        self.assertIsInstance(model_config.N_TRIALS, int, "N_TRIALS 应该是一个整数")

    def test_features_list_not_empty(self):
        """
        测试：特征列表不为空，且内容都是字符串。
        """
        self.assertGreater(len(model_config.FEATURES), 0, "特征列表不应为空")
        for feature in model_config.FEATURES:
            self.assertIsInstance(feature, str, f"特征 '{feature}' 应为字符串")

    def test_model_params_schema(self):
        """
        测试：MODEL_PARAMS 字典结构和超参数范围的有效性。
        """
        self.assertTrue(all(k in model_config.MODEL_PARAMS for k in ["logreg", "rf", "lgb"]))
        
        for model, params in model_config.MODEL_PARAMS.items():
            for param_name, value in params.items():
                if param_name.endswith("_range"):
                    self.assertIsInstance(value, list, f"{model}.{param_name} 应该是一个列表")
                    self.assertEqual(len(value), 2, f"{model}.{param_name} 应该包含两个元素 [min, max]")
                    self.assertLessEqual(value[0], value[1], f"{model}.{param_name} 的最小值应小于或等于最大值")

    def test_feature_consistency_with_generator(self):
        """
        测试：config中的特征列表与 feature_engineering.py 生成的特征完全一致。
        这是最重要的测试，确保配置和代码同步。
        如果此测试失败，通常意味着您需要更新 config/model_config.py 中的 FEATURES 列表。
        """
        # 1. 创建一个最小的、符合要求的假 DataFrame
        # 需要足够多的行数（例如100）以确保所有技术指标都能被计算
        data = {
            'open': np.random.random(100),
            'high': np.random.random(100),
            'low': np.random.random(100),
            'close': np.random.random(100),
            'vol': np.random.random(100)
        }
        index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='h'))
        dummy_df = pd.DataFrame(data, index=index)
        
        # 2. 记录初始列名
        initial_columns = set(dummy_df.columns)
        
        # 3. 运行特征生成函数
        # 传入一个不存在的新闻文件路径，以测试情绪特征列的默认创建
        df_with_features = generate_features(dummy_df, news_csv_path='non_existent_file.csv')
        
        # 4. 提取实际生成的特征列
        # 注意: generate_features 函数最后会 drop 掉一些中间MA特征，这很好，
        # 因为我们的配置文件里也不包含它们。
        generated_features = set(df_with_features.columns) - initial_columns
        
        # 5. 从配置文件加载特征
        config_features = set(model_config.FEATURES)
        
        # 6. 比较两个集合
        missing_in_config = generated_features - config_features
        unexpected_in_config = config_features - generated_features
        
        error_message = ""
        if missing_in_config:
            error_message += f"\n以下特征由代码生成，但缺失于配置文件中: {sorted(list(missing_in_config))}"
        if unexpected_in_config:
            error_message += f"\n以下特征在配置文件中，但代码未生成: {sorted(list(unexpected_in_config))}"
            
        self.assertEqual(config_features, generated_features, "配置文件中的特征与代码生成的特征不匹配。" + error_message)

if __name__ == '__main__':
    unittest.main()
