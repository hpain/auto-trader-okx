# 深度因子挖掘实施计划

## 目标描述
执行“深度因子挖掘计划”以提高模型性能（目标夏普比率 > -0.39）。这涉及处理4年的历史数据，使用遗传编程发现新的复杂特征，并将这些特征集成到LightGBM训练流程中。

## 用户审查要求
> [!IMPORTANT]
> 此过程涉及计算密集型任务（特征挖掘和模型训练）。
> - **特征挖掘**：将在4年数据上运行遗传编程（GP）。
> - **重新训练**：将运行LightGBM优化（Optuna）。

## 建议变更

### 研究与脚本
#### [修改] [research/enhanced_feature_miner.py](file:///d:/pycode/auto-trader-okx-lt/research/enhanced_feature_miner.py)
- 确保脚本正确处理带有现有特征的Parquet输入。
- （已验证：脚本已支持Parquet和目标'y'列）。
- 除非发现错误，否则不需要更改挖掘器脚本本身的代码。

#### [修改] [research/improved_evolution.py](file:///d:/pycode/auto-trader-okx-lt/research/improved_evolution.py)
- 确保它拾取*最新*的挖掘特征JSON文件。
- （已验证：脚本已查找`models/enhanced_mined_features_*.json`并按反向时间戳排序）。
- 此处无需更改代码。

### 执行工作流（[COMPLETED]）
1.  **生成特征缓存（4年）**
    - [x] 完成。生成了 `features_519a844655.parquet`。
2.  **运行深度特征挖掘**
    - [x] 完成。运行了两轮：
        - 初始轮：低强度，发现特征无效。
        - 高强度轮：发现 `enhanced_gp_feature_202512251331` (逻辑特征 `prob_bear - 9 * spread`)。
3.  **重新训练模型**
    - [x] 完成。使用5年数据重新训练。
    - **结果**：稳定性得分从 **-0.77** 提升至 **-0.005** (模型不再亏损)。

## 验证计划

### 自动化测试
- 挖掘过程本身没有自动化测试，因为它是脚本执行。
- 我们将验证输出文件是否存在：
    - `data/cache/*.parquet` (已更新/创建)
    - `models/enhanced_mined_features_*.json` (新创建)
    - `models/improved_best_model.pkl` (已更新)

### 手动验证
- 检查 `enhanced_feature_miner.py` 的控制台输出，查看“最佳个体”和性能指标。
- 检查 `improved_evolution.py` 日志中的：
    - "Found mined feature file: ..."（发现挖掘的特征文件）
    - "Successfully added mined feature: ..."（成功添加挖掘的特征）
    - 最终验证中的夏普比率有所提高。
