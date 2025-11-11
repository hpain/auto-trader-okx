# Project Summary

## Overall Goal
开发一个基于机器学习（LightGBM）的自动化交易机器人，能够在OKX交易所进行实盘或模拟交易，并具有完善的数据获取、特征工程、模型训练、回测和部署功能。

## Key Knowledge
- 项目使用Python实现，采用LightGBM作为机器学习模型
- 系统架构包括数据获取（Binance/OKX）、特征工程、模型训练与优化（使用Optuna）、回测和实盘交易模块
- 项目支持分类和回归两种模型类型，但当前主要实现的是分类模型
- 有完整的Docker部署方案和本地运行方式
- 使用YAML配置文件管理设置，支持通过环境变量覆盖配置
- 实盘交易通过run_live.py启动，使用trader/executor.py执行交易逻辑
- 模型训练通过research/evolve.py进行，支持历史数据下载和特征缓存

## Recent Actions
- 分析了项目当前状态，确认了主要模块的功能和实现方式
- 发现项目虽然为回归模型准备了回测函数(run_backtest_regression)，但核心训练流程仍基于分类模型
- 开始在models/evolution.py中实现回归模型训练函数train_evolve_regression，以支持完整的回归模型训练流程

## Current Plan
1. [IN PROGRESS] 在models/evolution.py中添加完整的回归模型训练函数train_evolve_regression
2. [TODO] 修改research/evolve.py以支持选择使用分类或回归模型进行训练
3. [TODO] 更新trader/executor.py以支持加载和使用回归模型进行预测
4. [TODO] 测试完整的回归模型训练和部署流程

---

## Summary Metadata
**Update time**: 2025-09-15T16:25:09.697Z 
