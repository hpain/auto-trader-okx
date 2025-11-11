import sys, os
sys.path.append(os.getcwd())
import pandas as pd
from data.news import load_news_from_csv
from features.sentiment_analysis import process_news_sentiment

mock_news = pd.DataFrame({'timestamp': pd.to_datetime(['2023-01-01T10:00:00Z','2023-01-02T12:00:00Z']), 'title': ["Good news for crypto, prices are up","Bad news for crypto, prices are down"]})
print('mock_news:')
print(mock_news)
print('\nprocessed daily_sent:')
print(process_news_sentiment(mock_news))
