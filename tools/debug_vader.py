import sys, os
sys.path.append(os.getcwd())
from features.sentiment_analysis import analyze_text_sentiment

texts = ["Good news for crypto, prices are up", "Bad news for crypto, prices are down"]
for t in texts:
    print(t, '->', analyze_text_sentiment(t))
