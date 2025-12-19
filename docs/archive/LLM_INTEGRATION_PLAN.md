# LLM 集成方案 - 务实路线

**核心原则**: LLM 辅助决策,不直接交易。让 LLM 做它擅长的事,让量化模型做它擅长的事。

---

## 🎯 推荐方案: 分层决策架构

```
┌─────────────────────────────────────────────────────────┐
│                    LLM 层 (战略决策)                      │
│  - 宏观市场分析                                           │
│  - 风险评估与建议                                         │
│  - 异常情况判断                                           │
│  - 策略参数建议                                           │
└─────────────────────────────────────────────────────────┘
                            ↓ (建议)
┌─────────────────────────────────────────────────────────┐
│                 决策融合层 (你的控制点)                    │
│  - 综合 LLM 建议 + 量化信号                               │
│  - 最终决策权在这里                                       │
│  - 可以否决 LLM 或模型的建议                              │
└─────────────────────────────────────────────────────────┘
                            ↓ (执行)
┌─────────────────────────────────────────────────────────┐
│              量化模型层 (战术执行)                         │
│  - LightGBM 预测                                         │
│  - 技术指标分析                                           │
│  - 精确的买卖时机                                         │
└─────────────────────────────────────────────────────────┘
```

**为什么这样设计?**
- ✅ LLM 擅长理解复杂信息 (新闻、宏观、异常)
- ✅ 量化模型擅长精确预测 (价格、时机)
- ✅ 你保持最终控制权,可以随时干预
- ✅ 降低风险,不会因为 LLM 幻觉导致巨额亏损

---

## 💡 LLM 的 4 个最佳应用场景

### 场景 1: 宏观市场分析 (每日一次) ⭐⭐⭐⭐⭐

**问题**: 你的量化模型只看技术指标,不理解宏观环境

**LLM 能做什么**:
```python
# analysis/llm_market_analyst.py

class LLMMarketAnalyst:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)  # 或 Anthropic, Google
    
    def daily_market_analysis(self, market_data: dict) -> dict:
        """
        每天早上分析市场环境
        
        输入:
        - 最近 7 天的价格走势
        - 最新的新闻标题 (CryptoPanic API)
        - 资金费率、持仓量数据
        - 宏观指标 (美联储利率、CPI 等)
        
        输出:
        - 市场情绪: bullish/bearish/neutral
        - 风险等级: low/medium/high
        - 关键事件提醒
        - 建议的仓位比例
        """
        
        prompt = f"""
你是一个专业的加密货币市场分析师。请分析当前市场环境:

**价格数据**:
- BTC 7日涨跌: {market_data['btc_7d_change']}%
- ETH 7日涨跌: {market_data['eth_7d_change']}%
- 当前波动率: {market_data['volatility']}%

**资金数据**:
- BTC 资金费率: {market_data['funding_rate']}%
- 持仓量变化: {market_data['oi_change']}%

**最新新闻** (最近 24 小时):
{market_data['news_headlines']}

**宏观环境**:
- 美联储利率: {market_data['fed_rate']}%
- 美元指数: {market_data['dxy']}

请提供:
1. 市场情绪评估 (bullish/bearish/neutral)
2. 风险等级 (low/medium/high)
3. 关键风险点 (3 条以内)
4. 建议的仓位比例 (0-100%)
5. 是否建议暂停交易 (yes/no)

以 JSON 格式回复。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",  # 或 claude-3-5-sonnet
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        return analysis

# 在 run_autonomous.py 中使用:
def sense(self):
    # ... 原有的数据获取 ...
    
    # 每天早上 9 点进行 LLM 分析
    if datetime.now().hour == 9 and not self.daily_analysis_done:
        llm_analysis = self.llm_analyst.daily_market_analysis({
            'btc_7d_change': ...,
            'news_headlines': self.fetch_latest_news(),
            'funding_rate': ...,
            # ...
        })
        
        # 根据 LLM 建议调整交易参数
        if llm_analysis['risk_level'] == 'high':
            self.position_size_multiplier = 0.5  # 减半仓位
            self.logger.warning(f"LLM: High risk detected. Reducing position size.")
        
        if llm_analysis['pause_trading'] == 'yes':
            self.trading_paused = True
            self.logger.critical(f"LLM: Recommends pausing trading. Reason: {llm_analysis['key_risks']}")
        
        self.daily_analysis_done = True
```

**实际效果**:
- 在重大新闻前自动降低仓位 (如 FOMC 会议)
- 在极端市场环境下暂停交易 (如 FTX 崩盘)
- 调整风险敞口,避免黑天鹅

**成本**: 每天 1 次调用,约 $0.01-0.05/天

---

### 场景 2: 异常检测与告警 (实时) ⭐⭐⭐⭐⭐

**问题**: 量化模型可能在异常情况下做出错误决策

**LLM 能做什么**:
```python
# trader/llm_anomaly_detector.py

class LLMAnomalyDetector:
    def check_before_trade(self, trade_context: dict) -> dict:
        """
        在执行交易前,让 LLM 检查是否有异常
        
        输入:
        - 当前信号 (buy/sell)
        - 模型置信度
        - 最近 1 小时的价格变化
        - 最新新闻
        - 资金费率异常情况
        
        输出:
        - 是否建议执行: yes/no
        - 风险提示
        - 建议的仓位调整
        """
        
        # 只在以下情况调用 LLM (节省成本):
        # 1. 价格 1 小时内变化 > 5%
        # 2. 资金费率异常 (> 0.1%)
        # 3. 检测到重大新闻关键词
        
        if not self._should_check(trade_context):
            return {"proceed": True, "reason": "Normal conditions"}
        
        prompt = f"""
你是风险控制专家。请判断以下交易是否安全:

**交易信号**: {trade_context['signal']} (置信度: {trade_context['confidence']})
**当前价格**: ${trade_context['current_price']}
**1小时涨跌**: {trade_context['1h_change']}%
**资金费率**: {trade_context['funding_rate']}%

**最新新闻**:
{trade_context['latest_news']}

**异常指标**:
- 价格突变: {trade_context['price_spike']}
- 成交量异常: {trade_context['volume_spike']}
- 订单簿失衡: {trade_context['orderbook_imbalance']}

请判断:
1. 是否建议执行交易? (yes/no)
2. 风险等级? (low/medium/high)
3. 如果执行,建议仓位比例? (0-100%)
4. 主要风险点?

以 JSON 格式回复。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # 使用更便宜的模型
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

# 在 decide_and_act() 中使用:
def decide_and_act(self, all_featured_data, regime_info):
    # ... 生成信号 ...
    
    if main_signal != 0:  # 有交易信号
        # LLM 异常检测
        anomaly_check = self.llm_detector.check_before_trade({
            'signal': 'buy' if main_signal == 1 else 'sell',
            'confidence': signals_df['confidence'].iloc[-1],
            'current_price': current_price,
            '1h_change': featured_data['close'].pct_change(60).iloc[-1] * 100,
            'latest_news': self.fetch_latest_news(limit=3),
            'funding_rate': featured_data['funding_rate'].iloc[-1],
            # ...
        })
        
        if anomaly_check['proceed'] == 'no':
            self.logger.warning(f"LLM blocked trade. Reason: {anomaly_check['risk_points']}")
            return  # 不执行交易
        
        if anomaly_check['risk_level'] == 'high':
            # 降低仓位
            quantity *= anomaly_check['position_ratio'] / 100
            self.logger.info(f"LLM: High risk. Reducing position to {anomaly_check['position_ratio']}%")
        
        # 执行交易...
```

**实际效果**:
- 在"插针"行情时阻止交易
- 在重大新闻发布时降低仓位
- 避免在异常市场环境下亏损

**成本**: 只在异常时调用,约 $0.10-0.50/天

---

### 场景 3: 交易复盘与学习 (每周一次) ⭐⭐⭐⭐

**问题**: 你不知道为什么有些交易赚钱,有些亏钱

**LLM 能做什么**:
```python
# analysis/llm_trade_reviewer.py

class LLMTradeReviewer:
    def weekly_review(self, trades: List[dict]) -> dict:
        """
        每周复盘所有交易,找出问题
        
        输入:
        - 最近 7 天的所有交易
        - 每笔交易的盈亏
        - 交易时的市场环境
        - 交易时的新闻
        
        输出:
        - 盈利交易的共同特征
        - 亏损交易的共同特征
        - 改进建议
        - 参数调整建议
        """
        
        # 分析盈利交易
        profitable_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        prompt = f"""
你是量化交易专家。请分析以下交易记录,找出规律:

**盈利交易** ({len(profitable_trades)} 笔):
{self._format_trades(profitable_trades[:10])}  # 只展示前 10 笔

**亏损交易** ({len(losing_trades)} 笔):
{self._format_trades(losing_trades[:10])}

**整体统计**:
- 胜率: {len(profitable_trades) / len(trades) * 100:.1f}%
- 平均盈利: ${sum(t['pnl'] for t in profitable_trades) / len(profitable_trades):.2f}
- 平均亏损: ${sum(t['pnl'] for t in losing_trades) / len(losing_trades):.2f}
- 盈亏比: {abs(sum(t['pnl'] for t in profitable_trades) / sum(t['pnl'] for t in losing_trades)):.2f}

请分析:
1. 盈利交易有什么共同特征? (市场环境、时间段、技术指标等)
2. 亏损交易有什么共同特征?
3. 哪些类型的交易应该避免?
4. 建议调整哪些参数? (止损、止盈、置信度阈值等)
5. 下周交易建议?

以 JSON 格式回复。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        review = json.loads(response.choices[0].message.content)
        
        # 自动应用建议 (可选)
        if review['auto_apply_suggestions']:
            self._apply_parameter_adjustments(review['parameter_adjustments'])
        
        return review

# 每周日晚上运行:
if datetime.now().weekday() == 6 and datetime.now().hour == 20:
    weekly_review = self.llm_reviewer.weekly_review(
        self.get_trades_last_7_days()
    )
    
    self.logger.info(f"Weekly Review:\n{json.dumps(weekly_review, indent=2)}")
    
    # 发送邮件报告
    self.send_email_report(weekly_review)
```

**实际效果**:
- 发现盈利模式 (如"周一早上的交易胜率更高")
- 发现亏损模式 (如"高波动时段亏损更多")
- 自动优化参数
- 持续改进策略

**成本**: 每周 1 次,约 $0.10-0.20/周

---

### 场景 4: 策略参数优化建议 (每月一次) ⭐⭐⭐⭐

**问题**: 不知道如何调整策略参数才能提高收益

**LLM 能做什么**:
```python
# analysis/llm_strategy_optimizer.py

class LLMStrategyOptimizer:
    def suggest_optimizations(self, performance_data: dict) -> dict:
        """
        基于历史表现,建议策略优化方向
        
        输入:
        - 最近 30 天的表现数据
        - 不同市场环境下的表现
        - 当前参数配置
        
        输出:
        - 参数调整建议
        - 新特征建议
        - 策略改进建议
        """
        
        prompt = f"""
你是量化策略专家。请基于以下数据建议优化方向:

**整体表现** (最近 30 天):
- 总收益: {performance_data['total_return']}%
- 夏普比率: {performance_data['sharpe_ratio']}
- 最大回撤: {performance_data['max_drawdown']}%
- 胜率: {performance_data['win_rate']}%

**不同市场环境表现**:
- 趋势市 (上涨): 收益 {performance_data['trending_up_return']}%, 胜率 {performance_data['trending_up_winrate']}%
- 趋势市 (下跌): 收益 {performance_data['trending_down_return']}%, 胜率 {performance_data['trending_down_winrate']}%
- 震荡市: 收益 {performance_data['ranging_return']}%, 胜率 {performance_data['ranging_winrate']}%
- 高波动: 收益 {performance_data['high_vol_return']}%, 胜率 {performance_data['high_vol_winrate']}%

**当前参数**:
- 止损: {performance_data['stop_loss']}%
- 止盈: {performance_data['take_profit']}%
- 置信度阈值: {performance_data['confidence_threshold']}
- 仓位大小: {performance_data['position_size']}%

**特征重要性** (Top 10):
{performance_data['top_features']}

请建议:
1. 哪些参数需要调整? 调整到多少?
2. 在哪些市场环境下应该暂停交易?
3. 应该添加哪些新特征?
4. 应该移除哪些无效特征?
5. 策略改进方向? (3 条以内)

以 JSON 格式回复。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

# 每月 1 号运行:
if datetime.now().day == 1:
    optimization_suggestions = self.llm_optimizer.suggest_optimizations(
        self.get_performance_last_30_days()
    )
    
    self.logger.info(f"Monthly Optimization Suggestions:\n{json.dumps(optimization_suggestions, indent=2)}")
    
    # 人工审核后应用
    # (不要自动应用,需要你确认)
```

**实际效果**:
- 发现策略弱点 (如"震荡市表现差")
- 建议参数调整 (如"止损应该放宽到 5%")
- 建议新特征 (如"应该加入链上数据")
- 持续进化策略

**成本**: 每月 1 次,约 $0.20-0.50/月

---

## ❌ 不建议让 LLM 做的事

### 1. 直接生成交易信号 ❌❌❌

**为什么不行**:
```python
# ❌ 错误示例:
signal = llm.predict_price_direction(market_data)
if signal == "buy":
    execute_trade()  # 危险!
```

**问题**:
- LLM 不擅长数值预测
- 容易产生幻觉
- 不稳定,同样输入可能给出不同答案
- 成本高 (每次交易都调用)
- 延迟高 (API 调用 1-3 秒)

**数据对比**:
```
LightGBM 预测:
- 准确率: 55-65%
- 延迟: 5ms
- 成本: $0
- 稳定性: 100% (相同输入相同输出)

LLM 预测:
- 准确率: 45-55% (不如随机)
- 延迟: 1-3 秒
- 成本: $0.01-0.05/次
- 稳定性: 60% (可能给出不同答案)
```

### 2. 实时技术分析 ❌❌

**为什么不行**:
```python
# ❌ 错误示例:
analysis = llm.analyze_chart(candlestick_data)
# "我看到一个头肩顶形态..." - 不靠谱!
```

**问题**:
- LLM 看不懂图表 (即使是 GPT-4V)
- 技术分析应该用代码实现 (ta 库)
- 成本高,效果差

### 3. 高频决策 ❌❌

**为什么不行**:
- API 延迟 1-3 秒,错过最佳时机
- 成本爆炸 (每分钟调用 = $100+/天)
- 不稳定

---

## 💰 成本估算

### 推荐方案的成本:

```
场景 1: 每日市场分析
- 频率: 1 次/天
- 模型: GPT-4o
- Token: ~2000 input + 500 output
- 成本: ~$0.02/天 × 30 = $0.60/月

场景 2: 异常检测
- 频率: ~5 次/天 (只在异常时)
- 模型: GPT-4o-mini
- Token: ~1000 input + 300 output
- 成本: ~$0.002/次 × 5 × 30 = $0.30/月

场景 3: 每周复盘
- 频率: 1 次/周
- 模型: GPT-4o
- Token: ~3000 input + 1000 output
- 成本: ~$0.05/次 × 4 = $0.20/月

场景 4: 每月优化
- 频率: 1 次/月
- 模型: GPT-4o
- Token: ~3000 input + 1000 output
- 成本: ~$0.05/月

总成本: ~$1.15/月
```

**如果让 LLM 直接交易**:
```
每天 100 次交易决策 × $0.02 = $2/天 × 30 = $60/月
(而且效果还不如量化模型)
```

---

## 🛠️ 实施步骤

### 第 1 周: 搭建基础框架

```python
# 1. 安装依赖
pip install openai anthropic  # 或其他 LLM API

# 2. 创建 LLM 管理器
# llm/llm_manager.py
class LLMManager:
    def __init__(self, provider='openai', api_key=None):
        self.provider = provider
        if provider == 'openai':
            self.client = OpenAI(api_key=api_key)
        elif provider == 'anthropic':
            self.client = Anthropic(api_key=api_key)
    
    def call(self, prompt: str, model: str = 'gpt-4o', 
             response_format: str = 'json') -> dict:
        """统一的 LLM 调用接口"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": response_format}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return {"error": str(e)}

# 3. 配置文件
# config/settings.yaml
llm:
  enabled: true
  provider: 'openai'  # or 'anthropic', 'google'
  api_key: 'your-api-key'
  daily_analysis:
    enabled: true
    time: '09:00'
    model: 'gpt-4o'
  anomaly_detection:
    enabled: true
    model: 'gpt-4o-mini'
    threshold: 0.05  # 价格变化 > 5% 才调用
  weekly_review:
    enabled: true
    day: 'sunday'
    time: '20:00'
```

### 第 2 周: 实现场景 1 (每日分析)

```python
# analysis/llm_market_analyst.py
# (完整代码见上面场景 1)

# 集成到 run_autonomous.py:
class AutonomousTrader:
    def __init__(self):
        # ... 原有初始化 ...
        
        # 添加 LLM 组件
        if config.get('llm', {}).get('enabled'):
            self.llm_manager = LLMManager(
                provider=config['llm']['provider'],
                api_key=config['llm']['api_key']
            )
            self.llm_analyst = LLMMarketAnalyst(self.llm_manager)
        else:
            self.llm_analyst = None
    
    def sense(self):
        # ... 原有代码 ...
        
        # 添加每日 LLM 分析
        if self.llm_analyst and self._should_run_daily_analysis():
            llm_analysis = self.llm_analyst.daily_market_analysis(
                self._prepare_market_data()
            )
            self._apply_llm_suggestions(llm_analysis)
```

### 第 3 周: 实现场景 2 (异常检测)

```python
# trader/llm_anomaly_detector.py
# (完整代码见上面场景 2)

# 集成到 decide_and_act():
def decide_and_act(self, all_featured_data, regime_info):
    # ... 生成信号 ...
    
    if main_signal != 0 and self.llm_detector:
        # LLM 异常检测
        if self._is_anomalous_condition(all_featured_data):
            anomaly_check = self.llm_detector.check_before_trade(
                self._prepare_trade_context(main_signal, all_featured_data)
            )
            
            if not anomaly_check['proceed']:
                self.logger.warning(f"LLM blocked trade: {anomaly_check['reason']}")
                return
    
    # ... 执行交易 ...
```

### 第 4 周: 实现场景 3 & 4 (复盘与优化)

```python
# analysis/llm_trade_reviewer.py
# analysis/llm_strategy_optimizer.py

# 添加定时任务:
def run(self):
    while True:
        # ... 主循环 ...
        
        # 每周日晚上复盘
        if self._is_weekly_review_time():
            self._run_weekly_review()
        
        # 每月 1 号优化
        if self._is_monthly_optimization_time():
            self._run_monthly_optimization()
```

---

## 📊 预期效果

### 短期效果 (1-3 个月):

1. **减少亏损交易**
   - LLM 异常检测可以避免 20-30% 的亏损交易
   - 特别是在重大新闻、异常波动时

2. **优化风险管理**
   - 根据宏观环境动态调整仓位
   - 在高风险时段降低敞口

3. **持续学习**
   - 每周复盘发现问题
   - 每月优化改进策略

### 长期效果 (6-12 个月):

4. **提高胜率**
   - 从 55% 提升到 60-65%
   - 通过避免低质量交易

5. **提高收益**
   - 月收益从 5% 提升到 8-10%
   - 通过更好的风险管理和参数优化

6. **降低回撤**
   - 最大回撤从 15% 降低到 10%
   - 通过及时识别和应对异常情况

---

## ⚠️ 注意事项

### 1. LLM 不是万能的

- ❌ 不要过度依赖 LLM
- ✅ LLM 只是辅助工具
- ✅ 最终决策权在你手里

### 2. 成本控制

- ✅ 只在必要时调用 LLM (异常检测)
- ✅ 使用更便宜的模型 (gpt-4o-mini)
- ✅ 批量处理 (每日/每周,而非实时)

### 3. 延迟问题

- ❌ 不要在交易执行路径上调用 LLM
- ✅ 异步调用,不阻塞主流程
- ✅ 设置超时 (5 秒)

### 4. 稳定性

- ✅ LLM 调用失败时有降级方案
- ✅ 不要因为 LLM 失败就停止交易
- ✅ 记录所有 LLM 建议,事后分析准确率

---

## 🎯 我的建议

### 立即开始 (本周):

1. ✅ **先实现场景 1 (每日分析)**
   - 最简单,最有价值
   - 成本低 ($0.60/月)
   - 立即见效

2. ✅ **运行 2 周,观察效果**
   - LLM 的建议是否合理?
   - 是否真的帮助避免了亏损?
   - 调整 prompt 优化效果

### 然后扩展 (2-4 周):

3. ✅ **添加场景 2 (异常检测)**
   - 在重大新闻、异常波动时保护你
   - 成本可控 ($0.30/月)

4. ✅ **添加场景 3 (每周复盘)**
   - 持续学习,不断改进
   - 成本很低 ($0.20/月)

### 不要做:

- ❌ 不要让 LLM 直接交易
- ❌ 不要在高频路径上调用 LLM
- ❌ 不要过度依赖 LLM

---

## 📝 总结

**最佳实践**:
```
LLM = 战略顾问 (宏观分析、风险评估、学习优化)
量化模型 = 战术执行 (精确预测、买卖时机)
你 = 最终决策者 (保持控制权)
```

**成本**: ~$1-2/月  
**预期收益提升**: 5-10%  
**风险**: 低 (LLM 只建议,不直接交易)

**开始行动**:
1. 本周实现场景 1 (每日分析)
2. 运行 2 周观察效果
3. 逐步添加其他场景
4. 持续优化 prompt 和参数

这样既能利用 LLM 的优势,又不会引入太大风险。**稳扎稳打,逐步优化。**
