import feedparser
import logging
import threading
import time
from datetime import datetime, timedelta

class NewsAggregator:
    """
    Fetches and aggregates news from RSS feeds for Macro and Crypto sentiment analysis.
    Sources: Yahoo Finance, CNBC, CoinTelegraph, Decrypt.
    """
    def __init__(self):
        self.logger = logging.getLogger("NewsAggregator")
        
        # RSS Feed Sources
        self.feeds = {
            "macro_yahoo": "https://finance.yahoo.com/news/rssindex",
            "macro_cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", # World Politics
            "crypto_cointelegraph": "https://cointelegraph.com/rss",
            "crypto_decrypt": "https://decrypt.co/feed"
        }
        
        self.cache = {}
        self.cache_expiry = 3600 # 1 hour cache
        self.last_fetch = 0
        self.lock = threading.Lock()

    def fetch_news_summary(self, lookback_hours=24):
        """
        Fetch news from all sources and return a formatted summary string.
        """
        with self.lock:
            # Check cache
            if time.time() - self.last_fetch < self.cache_expiry and self.cache:
                self.logger.info("Using cached news data.")
                return self._format_summary(self.cache)

            self.logger.info("Fetching fresh news from RSS feeds...")
            aggregated_data = {
                "macro": [],
                "crypto": []
            }

            for name, url in self.feeds.items():
                try:
                    feed = feedparser.parse(url)
                    category = "macro" if "macro" in name else "crypto"
                    
                    if not feed.entries:
                        self.logger.warning(f"No entries found for {name}")
                        continue
                        
                    count = 0
                    for entry in feed.entries:
                        # Parse published time
                        published = None
                        if hasattr(entry, 'published_parsed'):
                            published = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                        elif hasattr(entry, 'updated_parsed'):
                            published = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
                        
                        # Filter by age
                        if published and (datetime.now() - published) > timedelta(hours=lookback_hours):
                            continue
                            
                        # Clean summary (remove HTML tags if simple)
                        summary = getattr(entry, 'summary', '')
                        # Basic HTML tag strip (heuristic)
                        if '<' in summary:
                            summary = summary.split('<')[0] 
                        
                        aggregated_data[category].append({
                            "source": name,
                            "title": entry.title,
                            "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                            "link": entry.link,
                            "time": published.strftime("%Y-%m-%d %H:%M") if published else "N/A"
                        })
                        
                        count += 1
                        if count >= 5: # Limit per source
                            break
                            
                except Exception as e:
                    self.logger.error(f"Failed to fetch {name}: {e}")

            self.cache = aggregated_data
            self.last_fetch = time.time()
            
            return self._format_summary(aggregated_data)

    def _format_summary(self, data):
        """
        Format the aggregated data into a token-efficient string for the LLM.
        """
        lines = ["=== GLOBAL NEWS CONTEXT (Past 24H) ==="]
        
        lines.append("\n[MACRO & GEOPOLITICS]")
        if not data['macro']:
            lines.append("No recent macro news found.")
        for item in data['macro']:
            lines.append(f"- [{item['time']}] {item['title']}")
            # lines.append(f"  Context: {item['summary']}") # Summary implies more tokens, maybe just title for now if list is long

        lines.append("\n[CRYPTO MARKET]")
        if not data['crypto']:
            lines.append("No recent crypto news found.")
        for item in data['crypto']:
             lines.append(f"- [{item['time']}] {item['title']}")
        
        return "\n".join(lines)

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    agg = NewsAggregator()
    print(agg.fetch_news_summary())
