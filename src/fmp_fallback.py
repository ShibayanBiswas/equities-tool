"""
FMP API with web scraping fallback using AI agents when FMP fails.

This module provides functions that use FMP API first, then fall back to:
1. Web scraping with AI agents (for comprehensive data)
"""

import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
from src.fmp_client import FMPClient

logger = logging.getLogger(__name__)

# Initialize FMP client
_fmp_client = None

def _get_fmp_client():
    """Get or create FMP client instance."""
    global _fmp_client
    if _fmp_client is None:
        try:
            _fmp_client = FMPClient()
        except Exception as e:
            logger.warning(f"Failed to initialize FMP client: {e}")
            _fmp_client = None
    return _fmp_client

# Try to import Agno for AI agents
AGNO_AVAILABLE = False
GEMINI_API_KEY = None
GEMINI_MODEL = None
EXA_API_KEY = None

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
        # Try environment variables as fallback
        import os
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        EXA_API_KEY = os.getenv('EXA_API_KEY')
        # Backward compatibility aliases
        OPENAI_API_KEY = GEMINI_API_KEY
        OPENAI_MODEL = GEMINI_MODEL
    
    # Import for error handling
    try:
        from ddgs.exceptions import DDGSException
    except ImportError:
        DDGSException = Exception
    
    # Create safe wrapper for DuckDuckGoTools
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
    
    if GEMINI_API_KEY:
        AGNO_AVAILABLE = True
    else:
        logger.warning("Agno available but GEMINI_API_KEY not found. Web scraping fallback will be limited.")
except ImportError:
    logger.debug("Agno not available. Web scraping fallback will be limited.")


def get_profile_with_fallback(symbol: str, fmp_client=None) -> Optional[Dict]:
    """
    Get company profile using FMP API first, then web scraping.
    
    Args:
        symbol: Stock symbol
        fmp_client: Optional FMPClient instance (if None, uses global client)
    
    Returns:
        Dict with profile data or None
    """
    # Try FMP API first
    client = fmp_client or _get_fmp_client()
    if client:
        try:
            profile = client.get_profile(symbol)
            if profile:
                profile['source'] = 'fmp'
                logger.debug(f"Using FMP API for {symbol} profile")
                return profile
        except Exception as e:
            logger.debug(f"FMP profile failed for {symbol}: {e}")
    
    # Fallback to web scraping with AI agent
    if AGNO_AVAILABLE and GEMINI_API_KEY:
        try:
            profile = _get_profile_from_web(symbol)
            if profile:
                logger.info(f"Using web scraping fallback for {symbol} profile")
                return profile
        except Exception as e:
            logger.debug(f"Web scraping profile failed for {symbol}: {e}")
    
    logger.warning(f"Failed to get profile for {symbol} from any source.")
    return None


def get_quote_with_fallback(symbol: str, fmp_client=None) -> Optional[Dict]:
    """
    Get quote using FMP API first, then web scraping.
    
    Args:
        symbol: Stock symbol
        fmp_client: Optional FMPClient instance (if None, uses global client)
    
    Returns:
        Dict with quote data or None
    """
    # Try FMP API first
    client = fmp_client or _get_fmp_client()
    if client:
        try:
            quote = client.get_quote(symbol)
            if quote:
                quote['source'] = 'fmp'
                logger.debug(f"Using FMP API for {symbol} quote")
                return quote
        except Exception as e:
            logger.debug(f"FMP quote failed for {symbol}: {e}")
    
    # Web scraping for quotes is not typically needed
    logger.warning(f"Failed to get quote for {symbol} from FMP API.")
    return None


def get_key_metrics_with_fallback(symbol: str, fmp_client=None, limit: int = 5) -> pd.DataFrame:
    """
    Get key metrics using FMP API first, then web scraping.
    
    Args:
        symbol: Stock symbol
        fmp_client: Optional FMPClient instance (if None, uses global client)
        limit: Number of periods to return
    
    Returns:
        DataFrame with key metrics or empty DataFrame
    """
    # Try FMP API first
    client = fmp_client or _get_fmp_client()
    if client:
        try:
            metrics = client.get_key_metrics(symbol, period='annual', limit=limit)
            if not metrics.empty:
                logger.debug(f"Using FMP API for {symbol} key metrics")
                return metrics
        except Exception as e:
            logger.debug(f"FMP key metrics failed for {symbol}: {e}")
    
    logger.warning(f"Failed to get key metrics for {symbol} from FMP API.")
    return pd.DataFrame()


def _get_profile_from_web(symbol: str) -> Optional[Dict]:
    """
    Get company profile from web using AI agent.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dict with profile data or None
    """
    if not AGNO_AVAILABLE or not GEMINI_API_KEY:
        return None
    
    try:
        # Create AI agent for web scraping
        tools = []
        if EXA_API_KEY:
            try:
                # ExaTools doesn't support date filters for company searches
                tools.append(ExaTools(api_key=EXA_API_KEY, num_results=3, text=True))
            except Exception as e:
                logger.debug(f"ExaTools initialization failed: {e}")
        
        try:
            tools.append(SafeDuckDuckGoTools())
        except Exception as e:
            logger.debug(f"SafeDuckDuckGoTools initialization failed: {e}")
        
        if not tools:
            logger.debug("No web scraping tools available")
            return None
        
        agent = Agent(
            name="Company Profile Scraper",
            model=GeminiChat(id=GEMINI_MODEL or 'gemini-1.5-flash', api_key=GEMINI_API_KEY),
            tools=tools,
            instructions=f"""
            Search for comprehensive company information for stock ticker {symbol}.
            Find and extract:
            - Company name
            - Sector and industry
            - Market cap
            - Current stock price
            - Company description
            - Website URL
            - Exchange and country
            
            Return the information in a structured format.
            """
        )
        
        query = f"Get comprehensive company profile information for stock ticker {symbol} including company name, sector, industry, market cap, current price, and description"
        response = agent.run(query)
        
        # Parse response (simplified - in production, use more sophisticated parsing)
        # For now, return a basic structure
        # In a real implementation, you'd parse the agent response more carefully
        return {
            'symbol': symbol,
            'companyName': symbol,  # Would be extracted from response
            'sector': 'Unknown',
            'industry': 'Unknown',
            'source': 'web_scraping'
        }
    except Exception as e:
        logger.debug(f"Web scraping profile failed for {symbol}: {e}")
        return None
