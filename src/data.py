"""
Data fetching functions ported from Folder 2 Stock Dashboard.
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Optional, Dict
from datetime import datetime, timedelta
from src.fmp_fallback import (
    get_profile_with_fallback,
    get_quote_with_fallback,
    get_key_metrics_with_fallback
)
from src.fmp_client import FMPClient

logger = logging.getLogger(__name__)

# Try to import Agno for web scraping
try:
    from agno.agent import Agent
    try:
        from agno.models.gemini import GeminiChat
        GEMINI_AVAILABLE = True
    except ImportError:
        # Try Google's Gemini SDK and create a wrapper
        try:
            import google.generativeai as genai
            from agno.models.openai import OpenAIChat
            
            class GeminiChatWrapper(OpenAIChat):
                """Wrapper to use Gemini API with Agno framework."""
                def __init__(self, id=None, api_key=None, **kwargs):
                    if not api_key:
                        raise ValueError("Gemini API key is required")
                    genai.configure(api_key=api_key)
                    model_map = {
                        'gemini-1.5-pro': 'gemini-1.5-pro',
                        'gemini-1.5-flash': 'gemini-1.5-flash',
                        'gemini-pro': 'gemini-pro',
                    }
                    self.gemini_model_name = model_map.get(id, id or 'gemini-1.5-flash')  # Default to fastest
                    # Use fake API key to prevent OpenAI initialization
                    super().__init__(id='gpt-3.5-turbo', api_key='dummy-key-to-prevent-openai-init', **kwargs)
                    # Override any OpenAI client references
                    if hasattr(self, 'client'):
                        self.client = None
                    if hasattr(self, '_client'):
                        self._client = None
                    self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                    self._gemini_api_key = api_key
                
                def _invoke(self, messages, **kwargs):
                    """Override to use Gemini API - prevents OpenAI calls."""
                    try:
                        prompt_parts = []
                        for msg in messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            if role == "system":
                                prompt_parts.append(f"System: {content}")
                            elif role == "user":
                                prompt_parts.append(f"User: {content}")
                            elif role == "assistant":
                                prompt_parts.append(f"Assistant: {content}")
                        prompt = "\n".join(prompt_parts)
                        response = self.gemini_model.generate_content(prompt)
                        return {"content": response.text, "role": "assistant"}
                    except Exception as e:
                        logger.error(f"Gemini API error: {e}")
                        raise
                
                def invoke(self, messages, **kwargs):
                    """Override invoke to use Gemini."""
                    return self._invoke(messages, **kwargs)
            
            GeminiChat = GeminiChatWrapper
            GEMINI_AVAILABLE = True
        except ImportError:
            GeminiChat = None
            GEMINI_AVAILABLE = False
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
            GeminiChat = None
            GEMINI_AVAILABLE = False
    from agno.tools.duckduckgo import DuckDuckGoTools
    from agno.tools.exa import ExaTools
    try:
        from scripts.agents.config import GEMINI_API_KEY, GEMINI_MODEL, EXA_API_KEY
        # Backward compatibility aliases
        OPENAI_API_KEY = GEMINI_API_KEY
        OPENAI_MODEL = GEMINI_MODEL
    except ImportError:
        import os
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
        EXA_API_KEY = os.getenv('EXA_API_KEY')
        # Backward compatibility aliases
        OPENAI_API_KEY = GEMINI_API_KEY
        OPENAI_MODEL = GEMINI_MODEL
    
    # Import SafeDuckDuckGoTools
    try:
        from ddgs.exceptions import DDGSException
    except ImportError:
        DDGSException = Exception
    
    class SafeDuckDuckGoTools(DuckDuckGoTools):
        """Wrapper that handles 'No results found' errors gracefully."""
        def duckduckgo_search(self, query: str, max_results: int = 5) -> list:
            try:
                return super().duckduckgo_search(query=query, max_results=max_results)
            except DDGSException as e:
                if "No results found" in str(e):
                    logger.debug(f"DuckDuckGo search returned no results for query: {query}")
                    return []
                else:
                    raise
            except Exception as e:
                logger.warning(f"DuckDuckGo search error for query '{query}': {e}")
                return []
    
    AGNO_AVAILABLE = bool(GEMINI_API_KEY)
except ImportError:
    AGNO_AVAILABLE = False
    logger.debug("Agno not available for web scraping")


def get_company_info(symbol: str, client=None) -> Dict:
    """
    Get company information using FMP API and web scraping.
    
    Args:
        symbol: Stock symbol
        client: Optional FMPClient instance (if None, uses fallback functions)
    
    Returns:
        Dict with company info
    """
    try:
        # Use FMP API with web scraping fallback
        profile = get_profile_with_fallback(symbol, fmp_client=client)
        quote = get_quote_with_fallback(symbol, fmp_client=client)
        
        if not profile:
            logger.warning(f"No profile data available for {symbol} from any source")
            return {}
        
        company_info = {
            'Name': profile.get('companyName', ''),
            'Exchange': profile.get('exchangeShortName') or profile.get('exchange', ''),
            'Currency': profile.get('currency', ''),
            'Country': profile.get('country', ''),
            'Sector': profile.get('sector', ''),
            'Market Cap': profile.get('mktCap') or profile.get('marketCap', 0),
            'Price': profile.get('price', quote.get('price', 0) if quote else 0),
            'Beta': profile.get('beta', 0),
            'Price change': profile.get('changes', quote.get('change', 0) if quote else 0),
            'Website': profile.get('website', ''),
            'Image': profile.get('image', '')
        }
        
        return company_info
    except Exception as e:
        logger.error(f"Error getting company info for {symbol}: {e}")
        return {}


def get_stock_price(symbol: str, years: int = 5,
                   client=None) -> pd.DataFrame:
    """
    Get historical stock prices using FMP API.
    
    Args:
        symbol: Stock symbol
        years: Number of years of history
        client: Optional FMPClient instance (if None, creates new client)
    
    Returns:
        DataFrame with date index and Price column
    """
    try:
        fmp_client = client or FMPClient()
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        
        hist = fmp_client.get_historical_prices(symbol, start_date=start_date)
        
        if hist.empty:
            return pd.DataFrame()
        
        # Extract Close price
        if 'close' in hist.columns:
            df = pd.DataFrame({'Price': hist['close']})
        elif 'Close' in hist.columns:
            df = pd.DataFrame({'Price': hist['Close']})
        else:
            # Try to find price column
            price_col = [col for col in hist.columns if 'close' in col.lower() or 'price' in col.lower()]
            if price_col:
                df = pd.DataFrame({'Price': hist[price_col[0]]})
            else:
                return pd.DataFrame()
        
        df.index.name = 'Date'
        
        return df
    except Exception as e:
        logger.error(f"Error getting stock price for {symbol}: {e}")
        return pd.DataFrame()


def get_income_statement(symbol: str, limit: int = 5,
                        client=None) -> pd.DataFrame:
    """
    Get income statement using FMP API first, then web scraping.
    
    Args:
        symbol: Stock symbol
        limit: Number of years
        client: Optional FMPClient instance (if None, creates new client)
    
    Returns:
        DataFrame with income statement
    """
    try:
        fmp_client = client or FMPClient()
        income = fmp_client.get_income_statement(symbol, period='annual', limit=limit)
        if not income.empty:
            return income
    except Exception as e:
        logger.debug(f"FMP income statement failed for {symbol}: {e}")
    
    # Fallback to web scraping
    return _get_financial_statement_from_web(symbol, statement_type='income', limit=limit)


def get_balance_sheet(symbol: str, limit: int = 5,
                     client=None) -> pd.DataFrame:
    """
    Get balance sheet using FMP API first, then web scraping.
    
    Args:
        symbol: Stock symbol
        limit: Number of years
        client: Optional FMPClient instance (if None, creates new client)
    
    Returns:
        DataFrame with balance sheet
    """
    try:
        fmp_client = client or FMPClient()
        balance = fmp_client.get_balance_sheet(symbol, period='annual', limit=limit)
        if not balance.empty:
            return balance
    except Exception as e:
        logger.debug(f"FMP balance sheet failed for {symbol}: {e}")
    
    # Fallback to web scraping
    return _get_financial_statement_from_web(symbol, statement_type='balance', limit=limit)


def get_cash_flow(symbol: str, limit: int = 5,
                  client=None) -> pd.DataFrame:
    """
    Get cash flow statement using FMP API first, then web scraping.
    
    Args:
        symbol: Stock symbol
        limit: Number of years
        client: Optional FMPClient instance (if None, creates new client)
    
    Returns:
        DataFrame with cash flow
    """
    try:
        fmp_client = client or FMPClient()
        cashflow = fmp_client.get_cash_flow(symbol, period='annual', limit=limit)
        if not cashflow.empty:
            return cashflow
    except Exception as e:
        logger.debug(f"FMP cash flow failed for {symbol}: {e}")
    
    # Fallback to web scraping
    return _get_financial_statement_from_web(symbol, statement_type='cashflow', limit=limit)


def get_key_metrics(symbol: str, limit: int = 5,
                   client=None) -> pd.DataFrame:
    """
    Get key metrics using FMP API first, then web scraping.
    
    Args:
        symbol: Stock symbol
        limit: Number of years
        client: Optional FMPClient instance (if None, uses fallback functions)
    
    Returns:
        DataFrame with key metrics
    """
    return get_key_metrics_with_fallback(symbol, fmp_client=client, limit=limit)


def get_financial_ratios(symbol: str, limit: int = 5,
                         client=None) -> pd.DataFrame:
    """
    Get financial ratios using FMP API.
    
    Args:
        symbol: Stock symbol
        limit: Number of years
        client: Optional FMPClient instance (if None, creates new client)
    
    Returns:
        DataFrame with financial ratios
    """
    try:
        fmp_client = client or FMPClient()
        ratios = fmp_client.get_ratios(symbol, period='annual', limit=limit)
        if not ratios.empty:
            return ratios
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting financial ratios for {symbol}: {e}")
        return pd.DataFrame()


def enrich_ticker_data(symbol: str, client=None) -> Dict:
    """
    Enrich a ticker with comprehensive financial data using FMP API and web scraping.
    This is the main analytics function that should be used by orchestrators.
    
    Args:
        symbol: Stock symbol
        client: Optional FMPClient instance (if None, uses fallback functions)
    
    Returns:
        Dict with enriched data including:
        - profile: Company profile (from FMP/web)
        - quote: Current quote (from FMP)
        - key_metrics: Key financial metrics (from FMP)
        - ratios: Financial ratios (from FMP)
        - income_statement: Income statement (from FMP/web)
        - balance_sheet: Balance sheet (from FMP/web)
        - cashflow: Cash flow statement (from FMP/web)
    """
    enriched = {'symbol': symbol}
    
    try:
        # Get profile and quote (basic info) using FMP API with web scraping fallback
        profile = get_profile_with_fallback(symbol, fmp_client=client)
        if profile:
            enriched['profile'] = profile
            logger.debug(f"Profile retrieved for {symbol} (source: {profile.get('source', 'fmp')})")
        else:
            logger.warning(f"No profile data available for {symbol}")
        
        quote = get_quote_with_fallback(symbol, fmp_client=client)
        if quote:
            enriched['quote'] = quote
            logger.debug(f"Quote retrieved for {symbol}")
        else:
            logger.warning(f"No quote data available for {symbol}")
        
        # Get key metrics (analytics) using FMP API
        metrics = get_key_metrics_with_fallback(symbol, fmp_client=client, limit=5)
        if not metrics.empty:
            enriched['key_metrics'] = metrics
            logger.debug(f"Key metrics retrieved for {symbol}")
        
        # Get ratios (analytics) using FMP API
        try:
            ratios = get_financial_ratios(symbol, limit=5, client=client)
            if not ratios.empty:
                enriched['ratios'] = ratios
        except Exception as e:
            logger.debug(f"Ratios fetch failed for {symbol}: {e}")
        
        # Financial statements using FMP API with web scraping fallback
        income = get_income_statement(symbol, limit=5, client=client)
        if not income.empty:
            enriched['income_statement'] = income
        
        balance = get_balance_sheet(symbol, limit=5, client=client)
        if not balance.empty:
            enriched['balance_sheet'] = balance
        
        cashflow = get_cash_flow(symbol, limit=5, client=client)
        if not cashflow.empty:
            enriched['cashflow'] = cashflow
        
        # Get analyst rating (optional - don't fail if unavailable)
        try:
            from src.analyst import get_rating
            rating_data = get_rating(symbol, client=client)
            if rating_data:
                enriched['rating'] = rating_data
                logger.debug(f"Rating retrieved for {symbol}")
        except Exception as e:
            logger.debug(f"Rating fetch failed for {symbol}: {e}")
        
        # Get sentiment data (optional - don't fail if unavailable)
        try:
            from src.sentiment import get_aggregate_sentiment
            sentiment_data = get_aggregate_sentiment(symbol, client=client)
            if sentiment_data:
                enriched['sentiment'] = sentiment_data
                logger.debug(f"Sentiment retrieved for {symbol}")
        except Exception as e:
            logger.debug(f"Sentiment fetch failed for {symbol}: {e}")
        
        # Get analyst estimates (optional)
        try:
            from src.analyst import get_analyst_estimates
            analyst_estimates = get_analyst_estimates(symbol, limit=5, client=client)
            if not analyst_estimates.empty:
                enriched['analyst_estimates'] = analyst_estimates
                logger.debug(f"Analyst estimates retrieved for {symbol}")
        except Exception as e:
            logger.debug(f"Analyst estimates fetch failed for {symbol}: {e}")
        
        # Get earnings surprises (optional)
        try:
            from src.earnings import get_earnings_surprises
            earnings_surprises = get_earnings_surprises(symbol, limit=20, client=client)
            if not earnings_surprises.empty:
                enriched['earnings_surprises'] = earnings_surprises
                logger.debug(f"Earnings surprises retrieved for {symbol}")
        except Exception as e:
            logger.debug(f"Earnings surprises fetch failed for {symbol}: {e}")
        
        # Get institutional holders (optional)
        try:
            from src.institutional import get_institutional_holders
            institutional_holders = get_institutional_holders(symbol, client=client)
            if not institutional_holders.empty:
                enriched['institutional_holders'] = institutional_holders
                logger.debug(f"Institutional holders retrieved for {symbol}")
        except Exception as e:
            logger.debug(f"Institutional holders fetch failed for {symbol}: {e}")
        
        # Get analyst grades (optional) - for sentiment count by grade
        try:
            if client:
                grades_data = client._get(f"v3/grade/{symbol}", params={"limit": 500})
                if grades_data:
                    analyst_grades = pd.DataFrame(grades_data)
                    if not analyst_grades.empty:
                        enriched['analyst_grades'] = analyst_grades
                        logger.debug(f"Analyst grades retrieved for {symbol}")
        except Exception as e:
            logger.debug(f"Analyst grades fetch failed for {symbol}: {e}")
    
    except Exception as e:
        # Log error but don't fail completely
        logger.error(f"Error enriching {symbol}: {e}")
        enriched['error'] = str(e)
    
    return enriched


def _get_financial_statement_from_web(symbol: str, statement_type: str, limit: int = 5) -> pd.DataFrame:
    """
    Get financial statement from web using AI agents.
    
    Args:
        symbol: Stock symbol
        statement_type: 'income', 'balance', or 'cashflow'
        limit: Number of years
    
    Returns:
        DataFrame with financial statement data
    """
    if not AGNO_AVAILABLE or not GEMINI_API_KEY:
        logger.debug(f"Web scraping not available for {statement_type} statement of {symbol}")
        return pd.DataFrame()
    
    try:
        # Create AI agent for financial statement scraping
        tools = []
        if EXA_API_KEY:
            try:
                tools.append(ExaTools(api_key=EXA_API_KEY, num_results=5, text=True))
            except Exception as e:
                logger.debug(f"ExaTools initialization failed: {e}")
        
        try:
            tools.append(SafeDuckDuckGoTools())
        except Exception as e:
            logger.debug(f"SafeDuckDuckGoTools initialization failed: {e}")
        
        if not tools:
            logger.debug("No web scraping tools available")
            return pd.DataFrame()
        
        statement_names = {
            'income': 'income statement',
            'balance': 'balance sheet',
            'cashflow': 'cash flow statement'
        }
        statement_name = statement_names.get(statement_type, statement_type)
        
        agent = Agent(
            name=f"Financial Statement Scraper",
            model=GeminiChat(id=GEMINI_MODEL or 'gemini-1.5-pro', api_key=GEMINI_API_KEY),
            tools=tools,
            instructions=f"""
            Search for the {statement_name} for stock ticker {symbol}.
            Find the most recent {limit} years of financial data from reliable sources like:
            - SEC filings (10-K, 10-Q)
            - Company investor relations pages
            - Financial data websites (Yahoo Finance, MarketWatch, etc.)
            
            Extract the data in a structured format with:
            - Years as columns or index
            - Financial line items as rows
            - Numerical values (not text descriptions)
            
            Return the data in a clear, structured format that can be parsed into a table.
            """
        )
        
        query = f"Get the {statement_name} for stock ticker {symbol} for the last {limit} years. Include all major line items with numerical values."
        response = agent.run(query)
        
        # Try to parse the response into a DataFrame
        # This is a simplified parser - in production, use more sophisticated parsing
        try:
            # Look for structured data in the response
            response_text = str(response)
            
            # Try to extract tabular data
            # Look for patterns like "Year: 2023, Revenue: 1000000" or similar
            # This is a basic implementation - can be improved
            
            # For now, return empty DataFrame with a note that parsing needs improvement
            logger.info(f"Web scraping retrieved {statement_name} data for {symbol}, but parsing needs enhancement")
            
            # Return empty DataFrame for now - the agent response would need more sophisticated parsing
            return pd.DataFrame()
            
        except Exception as parse_error:
            logger.debug(f"Error parsing financial statement response for {symbol}: {parse_error}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.debug(f"Web scraping {statement_type} statement failed for {symbol}: {e}")
        return pd.DataFrame()

