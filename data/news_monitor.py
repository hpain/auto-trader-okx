"""
实时新闻监控模块
用于检测重大市场消息并触发风险控制
"""
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dateutil import parser


class CryptoPanicMonitor:
    """
    CryptoPanic 新闻监控器
    
    功能:
    1. 获取最新加密货币新闻
    2. 分析新闻影响级别
    3. 触发相应的风险控制措施
    """
    
    def __init__(self, api_key: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        初始化新闻监控器
        
        Args:
            api_key: CryptoPanic API key (可选,免费版无需)
            logger: 日志记录器
        """
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        self.last_check_time = datetime.utcnow()
        
        # 关键词配置 - 触发紧急暂停
        self.critical_keywords = [
            # 监管相关
            'sec', 'regulation', 'ban', 'lawsuit', 'investigation',
            'regulatory', 'illegal', 'fine', 'penalty',
            # 交易所风险
            'hack', 'hacked', 'exploit', 'exploited', 'bankruptcy', 
            'insolvent', 'insolvency', 'withdraw', 'withdrawal',
            'suspension', 'halt', 'halted', 'freeze', 'frozen',
            # 宏观重大事件
            'fed', 'federal reserve', 'interest rate', 'fomc',
            # 市场崩盘
            'crash', 'crashed', 'dump', 'dumped', 'liquidation',
            'liquidated', 'delisting', 'delisted'
        ]
        
        # 警告词列表 - 降低仓位
        self.warning_keywords = [
            'concern', 'concerned', 'warning', 'warns', 'risk', 'risky',
            'volatile', 'volatility', 'uncertainty', 'uncertain',
            'delay', 'delayed', 'postpone', 'postponed',
            'investigation', 'probe', 'scrutiny'
        ]
        
        self.logger.info("CryptoPanicMonitor initialized")
    
    def fetch_latest_news(self, currencies: List[str] = None, limit: int = 20) -> List[Dict]:
        """
        获取最新新闻
        
        Args:
            currencies: 币种列表,如 ['BTC', 'ETH']
            limit: 获取新闻数量
            
        Returns:
            新闻列表
        """
        if currencies is None:
            currencies = ['BTC', 'ETH']
        
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
            results = data.get('results', [])
            
            self.logger.debug(f"Fetched {len(results)} news items")
            return results
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch news: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in fetch_latest_news: {e}", exc_info=True)
            return []
    
    def analyze_news_impact(self, news_items: List[Dict], 
                           recent_minutes: int = 5) -> Dict:
        """
        分析新闻影响
        
        Args:
            news_items: 新闻列表
            recent_minutes: 只分析最近 N 分钟的新闻
            
        Returns:
            {
                'action': 'emergency_stop' | 'reduce_position' | 'normal',
                'severity': 'CRITICAL' | 'WARNING' | 'INFO',
                'reason': '触发原因',
                'news': [相关新闻列表]
            }
        """
        critical_news = []
        warning_news = []
        
        for item in news_items:
            title = item.get('title', '').lower()
            published_at = item.get('published_at', '')
            
            # 只看最近 N 分钟的新闻
            if not self._is_recent(published_at, minutes=recent_minutes):
                continue
            
            # 检查关键词
            matched_critical = False
            for keyword in self.critical_keywords:
                if keyword in title:
                    critical_news.append({
                        'title': item.get('title'),
                        'url': item.get('url'),
                        'published_at': published_at,
                        'keyword': keyword,
                        'votes': item.get('votes', {}).get('positive', 0)
                    })
                    matched_critical = True
                    break
            
            # 如果不是关键新闻,检查警告词
            if not matched_critical:
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
            self.logger.warning(f"Critical news detected: {critical_news[0]['title']}")
            return {
                'action': 'emergency_stop',
                'severity': 'CRITICAL',
                'reason': f"Critical keyword detected: {critical_news[0]['keyword']}",
                'news': critical_news
            }
        
        if len(warning_news) >= 3:  # 5分钟内3条警告新闻
            self.logger.warning(f"{len(warning_news)} warning news detected in {recent_minutes} minutes")
            return {
                'action': 'reduce_position',
                'severity': 'WARNING',
                'reason': f'{len(warning_news)} warning news in {recent_minutes} minutes',
                'news': warning_news
            }
        
        return {
            'action': 'normal',
            'severity': 'INFO',
            'reason': 'No significant news',
            'news': []
        }
    
    def _is_recent(self, published_at: str, minutes: int = 5) -> bool:
        """
        检查新闻是否在最近 N 分钟内
        
        Args:
            published_at: 发布时间字符串
            minutes: 时间窗口(分钟)
            
        Returns:
            是否在时间窗口内
        """
        try:
            pub_time = parser.parse(published_at)
            now = datetime.utcnow()
            
            # 处理时区
            if pub_time.tzinfo is not None:
                pub_time = pub_time.replace(tzinfo=None)
            
            delta = (now - pub_time).total_seconds()
            return delta <= minutes * 60
            
        except Exception as e:
            self.logger.debug(f"Failed to parse timestamp {published_at}: {e}")
            return False
    
    def get_summary(self) -> str:
        """获取监控器状态摘要"""
        return (f"CryptoPanicMonitor: "
                f"{len(self.critical_keywords)} critical keywords, "
                f"{len(self.warning_keywords)} warning keywords")


def get_news_monitor(config: Dict = None, logger: logging.Logger = None) -> CryptoPanicMonitor:
    """
    工厂函数: 创建新闻监控器实例
    
    Args:
        config: 配置字典
        logger: 日志记录器
        
    Returns:
        CryptoPanicMonitor 实例
    """
    api_key = None
    if config:
        api_key = config.get('news_monitoring', {}).get('cryptopanic_api_key')
    
    return CryptoPanicMonitor(api_key=api_key, logger=logger)
