# 消息面感知与应对方案

**核心原则**: 消息面用于**防守**(避免亏损),技术面用于**进攻**(寻找机会)

---

## 🎯 三层防御体系

```
Layer 1: 实时新闻监控 (秒级) → 紧急暂停交易
Layer 2: 社交媒体监控 (分钟级) → 调整仓位
Layer 3: 宏观事件日历 (天级) → 提前规避
```

---

## 📰 Layer 1: 实时新闻监控 (最重要!)

### 目标
在重大消息发布后 **30 秒内**暂停交易,避免被收割

### 数据源

#### 1. CryptoPanic (推荐,免费)
```python
# data/news_monitor.py
import requests
import time
from datetime import datetime

class CryptoPanicMonitor:
    def __init__(self, api_key=None):
        self.api_key = api_key  # 免费版无需 API key
        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        self.last_check_time = datetime.utcnow()
        
        # 关键词列表 (触发紧急暂停)
        self.critical_keywords = [
            # 监管相关
            'sec', 'regulation', 'ban', 'lawsuit', 'investigation',
            # 交易所相关
            'hack', 'hacked', 'exploit', 'bankruptcy', 'insolvent',
            'withdraw', 'suspension', 'halt',
            # 宏观相关
            'fed', 'interest rate', 'fomc', 'cpi', 'inflation',
            # 重大事件
            'crash', 'dump', 'liquidation', 'delisting'
        ]
        
        # 警告词列表 (降低仓位)
        self.warning_keywords = [
            'concern', 'warning', 'risk', 'volatile', 'uncertainty',
            'delay', 'postpone', 'investigation'
        ]
    
    def fetch_latest_news(self, currencies=['BTC', 'ETH'], limit=20):
        """获取最新新闻"""
        try:
            params = {
                'currencies': ','.join(currencies),
                'filter': 'important',  # 只获取重要新闻
                'public': 'true'
            }
            
            if self.api_key:
                params['auth_token'] = self.api_key
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('results', [])
            
        except Exception as e:
            print(f"Failed to fetch news: {e}")
            return []
    
    def analyze_news_impact(self, news_items):
        """
        分析新闻影响
        
        返回:
        {
            'action': 'emergency_stop' | 'reduce_position' | 'normal',
            'reason': '触发原因',
            'news': [相关新闻列表]
        }
        """
        critical_news = []
        warning_news = []
        
        for item in news_items:
            title = item.get('title', '').lower()
            published_at = item.get('published_at', '')
            
            # 只看最近 5 分钟的新闻
            if not self._is_recent(published_at, minutes=5):
                continue
            
            # 检查关键词
            for keyword in self.critical_keywords:
                if keyword in title:
                    critical_news.append({
                        'title': item.get('title'),
                        'url': item.get('url'),
                        'published_at': published_at,
                        'keyword': keyword,
                        'votes': item.get('votes', {}).get('positive', 0)
                    })
                    break
            
            for keyword in self.warning_keywords:
                if keyword in title:
                    warning_news.append({
                        'title': item.get('title'),
                        'url': item.get('url'),
                        'published_at': published_at,
                        'keyword': keyword
                    })
                    break
        
        # 决策逻辑
        if critical_news:
            return {
                'action': 'emergency_stop',
                'reason': f'Critical news detected: {critical_news[0]["keyword"]}',
                'news': critical_news,
                'severity': 'CRITICAL'
            }
        
        if len(warning_news) >= 3:  # 5分钟内3条警告新闻
            return {
                'action': 'reduce_position',
                'reason': f'{len(warning_news)} warning news in 5 minutes',
                'news': warning_news,
                'severity': 'WARNING'
            }
        
        return {
            'action': 'normal',
            'reason': 'No significant news',
            'news': [],
            'severity': 'INFO'
        }
    
    def _is_recent(self, published_at, minutes=5):
        """检查新闻是否在最近 N 分钟内"""
        try:
            from dateutil import parser
            pub_time = parser.parse(published_at)
            now = datetime.utcnow()
            
            # 处理时区
            if pub_time.tzinfo is None:
                pub_time = pub_time.replace(tzinfo=None)
            else:
                pub_time = pub_time.replace(tzinfo=None)
                now = now.replace(tzinfo=None)
            
            delta = (now - pub_time).total_seconds()
            return delta <= minutes * 60
        except:
            return False

# 使用示例:
monitor = CryptoPanicMonitor()
news = monitor.fetch_latest_news(['BTC', 'ETH'])
impact = monitor.analyze_news_impact(news)

if impact['action'] == 'emergency_stop':
    # 立即暂停交易
    with open('emergency_stop.flag', 'w') as f:
        f.write(f"Critical news: {impact['reason']}\n")
```

#### 2. Twitter/X 监控 (KOL 发言)

**重要 KOL 列表**:
- @elonmusk (影响 DOGE, BTC)
- @cz_binance (Binance CEO)
- @VitalikButerin (ETH)
- @APompliano (宏观)
- @DocumentingBTC (BTC 新闻)

```python
# data/twitter_monitor.py
import tweepy

class TwitterKOLMonitor:
    def __init__(self, bearer_token):
        self.client = tweepy.Client(bearer_token=bearer_token)
        
        # KOL 列表
        self.kols = {
            'elonmusk': {'影响币种': ['BTC', 'DOGE'], '权重': 10},
            'cz_binance': {'影响币种': ['BNB', 'BTC'], '权重': 9},
            'VitalikButerin': {'影响币种': ['ETH'], '权重': 8},
        }
        
        # 负面关键词
        self.negative_keywords = [
            'sell', 'dump', 'crash', 'scam', 'fraud',
            'concern', 'warning', 'risk'
        ]
        
        # 正面关键词
        self.positive_keywords = [
            'buy', 'bullish', 'moon', 'adoption', 'partnership'
        ]
    
    def check_kol_tweets(self, username, hours=1):
        """检查 KOL 最近的推文"""
        try:
            # 获取用户最近的推文
            user = self.client.get_user(username=username)
            tweets = self.client.get_users_tweets(
                user.data.id,
                max_results=10,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return None
            
            recent_tweets = []
            for tweet in tweets.data:
                # 只看最近 N 小时的推文
                if self._is_recent(tweet.created_at, hours=hours):
                    sentiment = self._analyze_sentiment(tweet.text)
                    recent_tweets.append({
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'likes': tweet.public_metrics['like_count'],
                        'retweets': tweet.public_metrics['retweet_count'],
                        'sentiment': sentiment
                    })
            
            return recent_tweets
            
        except Exception as e:
            print(f"Failed to fetch tweets: {e}")
            return None
    
    def _analyze_sentiment(self, text):
        """简单的情感分析"""
        text_lower = text.lower()
        
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        
        if negative_count > positive_count:
            return 'negative'
        elif positive_count > negative_count:
            return 'positive'
        else:
            return 'neutral'
```

### 集成到交易系统

```python
# 在 run_autonomous.py 的 decide_and_act() 开头添加:

def decide_and_act(self, all_featured_data, regime_info):
    self.logger.info("--- Stage: DECIDE & ACT ---")
    
    # ========== 每日止损检查 (已有) ==========
    # ...
    
    # ========== 新增: 实时新闻检查 ==========
    if hasattr(self, 'news_monitor'):
        try:
            # 每次交易前检查新闻 (只需 1-2 秒)
            news = self.news_monitor.fetch_latest_news(['BTC', 'ETH'])
            impact = self.news_monitor.analyze_news_impact(news)
            
            if impact['action'] == 'emergency_stop':
                self.logger.critical(f"🚨 CRITICAL NEWS DETECTED!")
                self.logger.critical(f"📰 {impact['news'][0]['title']}")
                self.logger.critical(f"⛔ STOPPING TRADING")
                
                # 创建紧急停止标志
                with open('emergency_stop.flag', 'w') as f:
                    f.write(f"Critical news at {datetime.utcnow().isoformat()}\n")
                    f.write(f"Reason: {impact['reason']}\n")
                    f.write(f"News: {impact['news'][0]['title']}\n")
                
                return  # 立即停止
            
            elif impact['action'] == 'reduce_position':
                self.logger.warning(f"⚠️ WARNING NEWS DETECTED")
                self.logger.warning(f"📰 {len(impact['news'])} warning news")
                
                # 降低仓位到 50%
                self.position_size_multiplier = 0.5
                self.logger.warning(f"🔻 Reducing position size to 50%")
                
        except Exception as e:
            self.logger.error(f"News monitoring failed: {e}")
    
    # 继续正常交易流程...
```

---

## 📱 Layer 2: 社交媒体情绪监控

### Reddit 情绪指标

```python
# data/reddit_monitor.py
import praw

class RedditSentimentMonitor:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # 监控的 subreddit
        self.subreddits = ['cryptocurrency', 'bitcoin', 'ethereum']
    
    def get_sentiment_score(self, hours=24):
        """
        获取社交媒体情绪分数
        
        返回: -1.0 到 1.0 (负面到正面)
        """
        total_score = 0
        total_posts = 0
        
        for sub_name in self.subreddits:
            subreddit = self.reddit.subreddit(sub_name)
            
            # 获取热门帖子
            for post in subreddit.hot(limit=50):
                # 只看最近 N 小时的帖子
                if not self._is_recent(post.created_utc, hours=hours):
                    continue
                
                # 简单的情绪评分: (upvotes - downvotes) / total
                score = (post.score - 0) / max(post.score + 1, 1)
                total_score += score
                total_posts += 1
        
        if total_posts == 0:
            return 0
        
        return total_score / total_posts
    
    def detect_fomo_or_fud(self):
        """
        检测 FOMO (贪婪) 或 FUD (恐惧)
        
        返回: 'extreme_greed' | 'greed' | 'neutral' | 'fear' | 'extreme_fear'
        """
        sentiment = self.get_sentiment_score(hours=6)
        
        if sentiment > 0.7:
            return 'extreme_greed'  # 可能是顶部,考虑减仓
        elif sentiment > 0.4:
            return 'greed'
        elif sentiment < -0.7:
            return 'extreme_fear'  # 可能是底部,考虑加仓
        elif sentiment < -0.4:
            return 'fear'
        else:
            return 'neutral'
```

### 应用到交易

```python
# 在 decide_and_act() 中:
if hasattr(self, 'reddit_monitor'):
    fomo_fud = self.reddit_monitor.detect_fomo_or_fud()
    
    if fomo_fud == 'extreme_greed':
        # 市场过热,降低仓位
        self.position_size_multiplier = 0.5
        self.logger.warning("🔥 Extreme greed detected. Reducing position.")
    
    elif fomo_fud == 'extreme_fear':
        # 市场恐慌,可能是机会,但要谨慎
        self.confidence_threshold_multiplier = 1.2  # 提高置信度要求
        self.logger.info("😱 Extreme fear detected. Increasing confidence threshold.")
```

---

## 📅 Layer 3: 宏观事件日历

### 重要事件列表

```python
# data/event_calendar.py
from datetime import datetime, timedelta

class MacroEventCalendar:
    def __init__(self):
        # 定期事件
        self.recurring_events = {
            'FOMC': {
                'description': 'Federal Reserve Meeting',
                'impact': 'CRITICAL',
                'action': 'pause_trading',
                'dates': [  # 2025 年 FOMC 会议日期
                    '2025-01-29', '2025-03-19', '2025-05-07',
                    '2025-06-18', '2025-07-30', '2025-09-17',
                    '2025-11-05', '2025-12-17'
                ]
            },
            'CPI': {
                'description': 'Consumer Price Index',
                'impact': 'HIGH',
                'action': 'reduce_position',
                'frequency': 'monthly',  # 每月第二周
            },
            'NFP': {
                'description': 'Non-Farm Payrolls',
                'impact': 'MEDIUM',
                'action': 'reduce_position',
                'frequency': 'monthly',  # 每月第一个周五
            }
        }
        
        # 一次性事件 (手动添加)
        self.one_time_events = [
            {
                'date': '2025-01-20',
                'description': 'US Presidential Inauguration',
                'impact': 'HIGH',
                'action': 'pause_trading'
            },
            # 可以手动添加其他重要事件
        ]
    
    def check_upcoming_events(self, days_ahead=3):
        """
        检查未来 N 天是否有重要事件
        
        返回: {
            'has_event': True/False,
            'event': {...},
            'days_until': N
        }
        """
        today = datetime.utcnow().date()
        
        # 检查定期事件
        for event_type, event_info in self.recurring_events.items():
            if 'dates' in event_info:
                for date_str in event_info['dates']:
                    event_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    days_until = (event_date - today).days
                    
                    if 0 <= days_until <= days_ahead:
                        return {
                            'has_event': True,
                            'event': {
                                'type': event_type,
                                **event_info
                            },
                            'days_until': days_until
                        }
        
        # 检查一次性事件
        for event in self.one_time_events:
            event_date = datetime.strptime(event['date'], '%Y-%m-%d').date()
            days_until = (event_date - today).days
            
            if 0 <= days_until <= days_ahead:
                return {
                    'has_event': True,
                    'event': event,
                    'days_until': days_until
                }
        
        return {'has_event': False}
    
    def get_action_for_event(self, event_impact):
        """根据事件影响级别决定行动"""
        if event_impact == 'CRITICAL':
            return 'pause_trading'  # 暂停交易
        elif event_impact == 'HIGH':
            return 'reduce_position'  # 降低仓位到 30%
        elif event_impact == 'MEDIUM':
            return 'reduce_position'  # 降低仓位到 50%
        else:
            return 'normal'
```

### 集成到系统

```python
# 在 sense() 阶段检查:
def sense(self):
    self.logger.info("--- Stage: SENSE ---")
    
    # ========== 检查宏观事件 ==========
    if hasattr(self, 'event_calendar'):
        upcoming = self.event_calendar.check_upcoming_events(days_ahead=3)
        
        if upcoming['has_event']:
            event = upcoming['event']
            days = upcoming['days_until']
            
            self.logger.warning(f"📅 Upcoming event in {days} days: {event['description']}")
            
            if event['action'] == 'pause_trading':
                self.logger.critical(f"⛔ Pausing trading due to {event['description']}")
                
                # 创建临时暂停标志
                with open('event_pause.flag', 'w') as f:
                    f.write(f"Event: {event['description']}\n")
                    f.write(f"Date: {event.get('date', 'recurring')}\n")
                
                # 暂停交易直到事件结束
                self.trading_paused_until = datetime.utcnow() + timedelta(days=days+1)
            
            elif event['action'] == 'reduce_position':
                if event['impact'] == 'HIGH':
                    self.position_size_multiplier = 0.3  # 降到 30%
                else:
                    self.position_size_multiplier = 0.5  # 降到 50%
                
                self.logger.warning(f"🔻 Reducing position to {self.position_size_multiplier*100}%")
    
    # 继续正常的数据获取...
```

---

## 🎯 实战案例

### 案例 1: OKX 销毁 OKB

**场景**: OKX 突然宣布销毁 7 亿 OKB

**传统量化系统**:
- ❌ 完全不知道这个消息
- ❌ 可能在暴涨时追高
- ❌ 或者错过机会

**集成消息面后**:
```python
# CryptoPanic 会在 30 秒内抓取到新闻:
# "OKX announces 700M OKB token burn"

# 系统判断:
if 'okb' in title and 'burn' in title:
    # 这是重大利好!
    if current_position['OKB'] > 0:
        # 持有 OKB,不要卖出
        self.logger.info("🔥 OKB burn news! Holding position.")
        return  # 不执行卖出信号
    else:
        # 没有持仓,考虑买入 (如果技术面也支持)
        if technical_signal == 'buy':
            self.position_size_multiplier = 1.5  # 加大仓位
```

### 案例 2: SEC 起诉 Binance

**场景**: SEC 突然宣布起诉 Binance

**系统反应**:
```python
# CryptoPanic 检测到关键词: 'SEC', 'Binance', 'lawsuit'
impact = monitor.analyze_news_impact(news)

if impact['action'] == 'emergency_stop':
    # 立即暂停所有交易
    with open('emergency_stop.flag', 'w') as f:
        f.write("SEC lawsuit against Binance\n")
    
    # 如果持有 BNB,考虑紧急平仓
    if current_position['BNB'] > 0:
        self.emergency_close_position('BNB')
```

### 案例 3: FOMC 会议

**场景**: 美联储 FOMC 会议前 3 天

**系统反应**:
```python
# 事件日历检测到 FOMC 会议
upcoming = event_calendar.check_upcoming_events(days_ahead=3)

if upcoming['event']['type'] == 'FOMC':
    # 提前 3 天降低仓位
    self.position_size_multiplier = 0.3
    self.logger.warning("📅 FOMC meeting in 3 days. Reducing position to 30%")
    
    # 会议当天暂停交易
    if upcoming['days_until'] == 0:
        self.trading_paused = True
        self.logger.critical("⛔ FOMC meeting today. Trading paused.")
```

---

## 💰 成本估算

### 数据源成本

| 数据源 | 免费额度 | 付费价格 | 推荐 |
|--------|----------|----------|------|
| **CryptoPanic** | 无限 (有延迟) | $10/月 (实时) | ⭐⭐⭐⭐⭐ |
| **Twitter API** | 免费 (限制) | $100/月 | ⭐⭐⭐ |
| **Reddit API** | 免费 | - | ⭐⭐⭐⭐ |
| **事件日历** | 免费 (手动) | - | ⭐⭐⭐⭐⭐ |

**推荐配置**: 
- CryptoPanic 免费版 (足够用)
- Reddit API (免费)
- 事件日历 (手动维护)

**总成本**: $0/月

---

## 🛠️ 实施步骤

### 第 1 周: 实现 Layer 3 (最简单)

```python
# 1. 创建事件日历
calendar = MacroEventCalendar()

# 2. 手动添加重要事件
calendar.one_time_events.append({
    'date': '2025-12-01',  # 示例
    'description': 'Some important event',
    'impact': 'HIGH',
    'action': 'reduce_position'
})

# 3. 集成到 sense() 阶段
# (代码见上面)
```

### 第 2 周: 实现 Layer 1 (最重要)

```python
# 1. 注册 CryptoPanic (可选)
# https://cryptopanic.com/developers/api/

# 2. 创建新闻监控器
monitor = CryptoPanicMonitor()

# 3. 集成到 decide_and_act() 开头
# (代码见上面)

# 4. 测试
news = monitor.fetch_latest_news(['BTC'])
impact = monitor.analyze_news_impact(news)
print(impact)
```

### 第 3 周: 实现 Layer 2 (可选)

```python
# 1. 注册 Reddit API
# https://www.reddit.com/prefs/apps

# 2. 创建情绪监控器
reddit_monitor = RedditSentimentMonitor(
    client_id='your_id',
    client_secret='your_secret',
    user_agent='your_app'
)

# 3. 集成到系统
# (代码见上面)
```

---

## ⚠️ 注意事项

### 1. 不要过度反应

```python
# ❌ 错误: 看到任何负面新闻就停止交易
if 'negative' in sentiment:
    stop_trading()  # 太敏感!

# ✅ 正确: 只对真正重大的新闻反应
if impact['severity'] == 'CRITICAL':
    stop_trading()
```

### 2. 设置冷静期

```python
# 新闻触发暂停后,设置冷静期
self.news_pause_until = datetime.utcnow() + timedelta(hours=2)

# 2 小时后自动恢复
if datetime.utcnow() > self.news_pause_until:
    self.trading_paused = False
```

### 3. 人工确认

```python
# 对于 CRITICAL 级别的新闻,发送邮件通知
if impact['severity'] == 'CRITICAL':
    send_email(
        subject='🚨 Critical News Alert',
        body=f"News: {impact['news'][0]['title']}\n"
             f"Action: Trading paused\n"
             f"Please review and confirm."
    )
```

---

## 📊 预期效果

### 避免的亏损 (保守估计)

```
假设每年遇到 3 次重大负面新闻:
- 没有消息面监控: 每次亏损 10-20%
- 有消息面监控: 提前暂停,亏损 0-2%

每次避免亏损: 10-18%
年度避免亏损: 30-54%

本金 10000 USDT:
避免亏损 = 3000-5400 USDT/年
```

### 抓住的机会

```
假设每年遇到 2 次重大利好 (如 OKB 销毁):
- 没有消息面监控: 错过或追高
- 有消息面监控: 及时调整仓位

每次额外收益: 5-10%
年度额外收益: 10-20%

本金 10000 USDT:
额外收益 = 1000-2000 USDT/年
```

**总价值**: 4000-7400 USDT/年

---

## 🎯 我的建议

### 立即实施 (本周):

1. ✅ **实现事件日历** (Layer 3)
   - 最简单
   - 效果明显
   - 0 成本

2. ✅ **实现新闻监控** (Layer 1)
   - 最重要
   - 避免重大亏损
   - 0 成本 (免费 API)

### 可选实施 (下周):

3. 🔧 **Reddit 情绪监控** (Layer 2)
   - 锦上添花
   - 需要额外开发

### 不要做:

- ❌ 不要让新闻监控过于敏感
- ❌ 不要完全依赖消息面
- ❌ 不要忽视技术面

**核心原则**: 消息面是**安全网**,不是**交易信号**。用它来避免亏损,而不是寻找机会。
