"""
Sentiment analysis functions.
"""

import pandas as pd
from typing import Optional, Dict, List
from src.fmp_client import FMPClient

# Lazy import and initialization of NLTK to avoid import errors
_analyzer = None

def _get_analyzer():
    """Get or initialize the VADER sentiment analyzer."""
    global _analyzer
    if _analyzer is None:
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            # Download VADER lexicon if not already downloaded
            try:
                nltk.data.find('tokenizers/vader_lexicon')
            except LookupError:
                try:
                    nltk.download('vader_lexicon', quiet=True)
                except Exception:
                    # If download fails, try alternative path
                    try:
                        nltk.data.find('vader_lexicon')
                    except LookupError:
                        nltk.download('vader_lexicon', quiet=True)
            
            _analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            # If NLTK fails, create a dummy analyzer
            class DummyAnalyzer:
                def polarity_scores(self, text):
                    return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
            _analyzer = DummyAnalyzer()
    
    return _analyzer


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text using VADER.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dict with sentiment scores
    """
    if not text or not isinstance(text, str):
        return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    try:
        analyzer = _get_analyzer()
        scores = analyzer.polarity_scores(text)
        return scores
    except Exception:
        # Return neutral sentiment if analysis fails
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}


def get_company_news_sentiment(symbol: str, client: Optional[FMPClient] = None) -> pd.DataFrame:
    """
    Get company news and analyze sentiment.
    
    Reference: https://site.financialmodelingprep.com/developer/docs#news
    Endpoints: v3/stock_news (primary), v4/company-outlook (fallback)
    
    Args:
        symbol: Stock symbol
        client: FMP client instance
    
    Returns:
        DataFrame with news and sentiment scores
    """
    if client is None:
        client = FMPClient()
        should_close = True
    else:
        should_close = False
    
    try:
        # Try stock_news endpoint first (as shown in Company Sentiment.ipynb)
        # Reference: https://site.financialmodelingprep.com/developer/docs#news
        # Note: v3/stock_news uses different base URL
        try:
            news_data_list = client._get("v3/stock_news", params={"tickers": symbol, "limit": 100})
        except Exception as e:
            # If v3 endpoint fails, try alternative
            news_data_list = None
        
        if not news_data_list:
            # Fallback to company-outlook endpoint
            outlook = client.get_company_outlook(symbol)
            if outlook and 'news' in outlook:
                news_data_list = outlook['news']
            else:
                return pd.DataFrame()
        
        if not news_data_list:
            return pd.DataFrame()
        
        # Analyze sentiment for each news item
        news_data = []
        for item in news_data_list[:100]:  # Limit to 100 most recent
            text = item.get('text', '') or item.get('title', '') or item.get('content', '')
            if not text:
                continue
                
            sentiment = analyze_sentiment(text)
            
            news_data.append({
                'symbol': symbol,
                'publishedDate': item.get('publishedDate', '') or item.get('date', ''),
                'title': item.get('title', ''),
                'text': text[:500],  # Truncate for display
                'url': item.get('url', ''),
                'site': item.get('site', '') or item.get('source', ''),
                'sentiment_compound': sentiment['compound'],
                'sentiment_pos': sentiment['pos'],
                'sentiment_neu': sentiment['neu'],
                'sentiment_neg': sentiment['neg'],
            })
        
        if not news_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(news_data)
        
        if 'publishedDate' in df.columns:
            df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
            df = df.sort_values('publishedDate', ascending=False)
        
        return df
    
    finally:
        if should_close:
            client.close()


def get_aggregate_sentiment(symbol: str, client: Optional[FMPClient] = None) -> Dict[str, float]:
    """
    Get aggregate sentiment score for a company.
    
    Args:
        symbol: Stock symbol
        client: FMP client instance
    
    Returns:
        Dict with aggregate sentiment metrics
    """
    news_df = get_company_news_sentiment(symbol, client)
    
    if news_df.empty:
        return {
            'avg_compound': 0.0,
            'avg_pos': 0.0,
            'avg_neu': 0.0,
            'avg_neg': 0.0,
            'news_count': 0
        }
    
    return {
        'avg_compound': news_df['sentiment_compound'].mean(),
        'avg_pos': news_df['sentiment_pos'].mean(),
        'avg_neu': news_df['sentiment_neu'].mean(),
        'avg_neg': news_df['sentiment_neg'].mean(),
        'news_count': len(news_df)
    }

