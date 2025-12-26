"""
Agno-based agents for index cache building.
Uses Agno framework (Agent, Team) for agentic orchestration.

ARCHITECTURE NOTE:
-----------------
This module contains ONLY orchestration agents. These agents handle:
- Index discovery and constituent fetching
- Ticker validation
- Sector/industry segregation
- Market intelligence gathering

All analytics (data enrichment, calculations, valuations) are handled by
the src/ folder modules. This ensures clear separation of concerns:
- scripts/agents/ = Orchestration & data gathering
- src/ = Analytics & calculations
"""

import logging
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from textwrap import dedent
import re
import requests

logger = logging.getLogger(__name__)

from agno.agent import Agent
from agno.team import Team
from agno.tools import tool

# Try to import Gemini from Agno
GEMINI_AVAILABLE = False
GeminiChat = None

try:
    from agno.models.gemini import GeminiChat
    GEMINI_AVAILABLE = True
    logger.info("Using Agno GeminiChat model")
except ImportError:
    # Try Google's Gemini SDK and create a proper wrapper
    try:
        import google.generativeai as genai
        from agno.models.openai import OpenAIChat
        
        # Create a Gemini wrapper that inherits from OpenAIChat to work with Agno
        class GeminiChatWrapper(OpenAIChat):
            """Wrapper to use Gemini API with Agno framework by extending OpenAIChat interface."""
            def __init__(self, id=None, api_key=None, **kwargs):
                if not api_key:
                    raise ValueError("Gemini API key is required")
                
                # Configure Gemini FIRST before parent init
                genai.configure(api_key=api_key)
                
                # Map Gemini model names
                model_map = {
                    'gemini-1.5-pro': 'gemini-1.5-pro',
                    'gemini-1.5-flash': 'gemini-1.5-flash',
                    'gemini-pro': 'gemini-pro',
                }
                self.gemini_model_name = model_map.get(id, id or 'gemini-1.5-flash')  # Default to fastest
                
                # Initialize parent with dummy values but prevent OpenAI client creation
                # Pass a fake API key to prevent actual OpenAI initialization
                try:
                    super().__init__(
                        id='gpt-3.5-turbo',  # Dummy for compatibility
                        api_key='dummy-key-to-prevent-openai-init',  # Fake key to prevent OpenAI calls
                        **kwargs
                    )
                    # Override the client to None to prevent OpenAI API calls
                    if hasattr(self, 'client'):
                        self.client = None
                    if hasattr(self, '_client'):
                        self._client = None
                except Exception as e:
                    logger.debug(f"Parent init warning (expected): {e}")
                
                # Initialize Gemini model with fallback chain
                # Try multiple models in order: gemini-1.5-flash -> gemini-pro -> gemini-1.5-pro
                self._gemini_api_key = api_key
                self._fallback_models = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.5-pro']
                self._current_model_index = 0
                self._initialize_gemini_model()
                self.id = self.gemini_model_name  # Set the actual model ID
            
            def _initialize_gemini_model(self):
                """Initialize Gemini model, trying fallback models if needed."""
                while self._current_model_index < len(self._fallback_models):
                    model_name = self._fallback_models[self._current_model_index]
                    try:
                        logger.debug(f"Trying to initialize Gemini model: {model_name}")
                        self.gemini_model = genai.GenerativeModel(model_name)
                        self.gemini_model_name = model_name
                        logger.info(f"Successfully initialized Gemini model: {model_name}")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to initialize {model_name}: {type(e).__name__}: {str(e)}")
                        self._current_model_index += 1
                
                # If all models failed, set to None
                logger.error(f"Failed to initialize any Gemini model from: {self._fallback_models}")
                self.gemini_model = None
                self.gemini_model_name = self._fallback_models[0]  # Keep original name for ID
            
            def _try_fallback_model(self):
                """Try the next fallback model when current one fails."""
                if self._current_model_index < len(self._fallback_models) - 1:
                    self._current_model_index += 1
                    logger.info(f"Switching to fallback model: {self._fallback_models[self._current_model_index]}")
                    self._initialize_gemini_model()
                    return self.gemini_model is not None
                return False
            
            def _invoke(self, messages, **kwargs):
                """Override the invoke method to use Gemini API - prevents OpenAI calls."""
                try:
                    # Convert messages to Gemini format
                    prompt_parts = []
                    for msg in messages:
                        # Handle both dict and Message object formats
                        if isinstance(msg, dict):
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                        else:
                            # Handle Message objects (from Agno or Gemini SDK)
                            # Try common attributes
                            role = getattr(msg, "role", None) or getattr(msg, "role_name", None) or "user"
                            content = getattr(msg, "content", None) or getattr(msg, "text", None) or str(msg)
                        
                        if role == "system":
                            prompt_parts.append(f"System: {content}")
                        elif role == "user":
                            prompt_parts.append(f"User: {content}")
                        elif role == "assistant":
                            prompt_parts.append(f"Assistant: {content}")
                    
                    prompt = "\n".join(prompt_parts)
                    
                    # Generate response using Gemini (NOT OpenAI)
                    if self.gemini_model is None:
                        # Try to initialize if not already done
                        self._initialize_gemini_model()
                        if self.gemini_model is None:
                            raise ValueError("Gemini model not initialized and all fallback models failed")
                    
                    # Try to generate content, with automatic fallback on error
                    max_retries = len(self._fallback_models) - self._current_model_index
                    for attempt in range(max_retries):
                        try:
                            response = self.gemini_model.generate_content(prompt)
                            
                            # Extract text from response (handle different response formats)
                            if hasattr(response, 'text'):
                                content = response.text
                            elif hasattr(response, 'content'):
                                # Handle if response.content is a list or object
                                if isinstance(response.content, list) and len(response.content) > 0:
                                    content = getattr(response.content[0], 'text', str(response.content[0]))
                                else:
                                    content = str(response.content)
                            else:
                                content = str(response)
                            
                            # Return in Agno format
                            return {
                                "content": content,
                                "role": "assistant"
                            }
                        except Exception as e:
                            error_str = str(e)
                            # Check if it's a model not found error
                            if "not found" in error_str.lower() or "404" in error_str or "not supported" in error_str.lower():
                                logger.warning(f"Gemini model {self.gemini_model_name} failed: {error_str}")
                                if attempt < max_retries - 1:
                                    # Try next fallback model
                                    if self._try_fallback_model():
                                        logger.info(f"Retrying with fallback model: {self.gemini_model_name}")
                                        continue
                                    else:
                                        logger.error("No more fallback models available")
                                        raise
                                else:
                                    logger.error(f"All Gemini models failed. Last error: {error_str}")
                                    raise
                            else:
                                # Other errors, don't retry with different model
                                logger.error(f"Gemini API error: {type(e).__name__}: {error_str}")
                                raise
                    
                    # Should not reach here, but just in case
                    raise ValueError("Failed to generate content with any available Gemini model")
                except Exception as e:
                    logger.error(f"Gemini API error: {type(e).__name__}: {str(e)}")
                    raise
            
            def invoke(self, messages, **kwargs):
                """Override invoke to use Gemini."""
                return self._invoke(messages, **kwargs)
        
        GeminiChat = GeminiChatWrapper
        GEMINI_AVAILABLE = True
        logger.info("Using Google Gemini SDK with Agno-compatible wrapper")
    except ImportError:
        logger.warning("Gemini not available. Install google-generativeai: pip install google-generativeai")
        GeminiChat = None
        GEMINI_AVAILABLE = False
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini: {e}")
        GeminiChat = None
        GEMINI_AVAILABLE = False
from agno.tools.exa import ExaTools
from agno.tools.duckduckgo import DuckDuckGoTools

# Import for error handling
try:
    from ddgs.exceptions import DDGSException
except ImportError:
    DDGSException = Exception


class SafeDuckDuckGoTools(DuckDuckGoTools):
    """
    Wrapper for DuckDuckGoTools that handles "No results found" errors gracefully.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def duckduckgo_search(self, query: str, max_results: int = 5) -> list:
        """
        Safe wrapper that catches DDGSException and returns empty list instead of raising.
        """
        try:
            return super().duckduckgo_search(query=query, max_results=max_results)
        except DDGSException as e:
            if "No results found" in str(e):
                logger.debug(f"DuckDuckGo search returned no results for query: {query}")
                return []
            else:
                # Re-raise other exceptions
                raise
        except Exception as e:
            logger.warning(f"DuckDuckGo search error for query '{query}': {e}")
            return []

from scripts.agents.config import (
    GEMINI_API_KEY, FMP_API_KEY, EXA_API_KEY,
    GEMINI_MODEL, GEMINI_MODEL_FAST, EXA_INCLUDE_DOMAINS,
    EXA_START_DATE, EXA_END_DATE
)
logger = logging.getLogger(__name__)

# Import FMP client
from src.fmp_client import FMPClient

# Create FMP client instance
_fmp_client = None

def _get_fmp_client():
    """Get or create FMP client instance."""
    global _fmp_client
    if _fmp_client is None:
        try:
            logger.info("_get_fmp_client: Initializing FMP client")
            _fmp_client = FMPClient()
            logger.info("_get_fmp_client: FMP client initialized successfully")
        except Exception as e:
            logger.error(f"_get_fmp_client: Failed to initialize FMP client: {type(e).__name__}: {str(e)}")
            import traceback
            logger.debug(f"_get_fmp_client: Traceback: {traceback.format_exc()}")
            _fmp_client = None
    else:
        logger.debug("_get_fmp_client: Returning existing FMP client instance")
    return _fmp_client

# Create Exa tool if API key available
exa_tool = None
if EXA_API_KEY:
    try:
        # ExaTools doesn't support date filters for all search types
        # Create without date filters to avoid errors
        exa_tool = ExaTools(
            api_key=EXA_API_KEY,
            include_domains=EXA_INCLUDE_DOMAINS,
            # Removed start_published_date and end_published_date as they're not supported for company searches
            type="auto",
            num_results=5,
            text=True,
        )
        logger.info("ExaTools created successfully")
    except Exception as e:
        logger.warning(f"Could not create ExaTools: {e}")


# ============================================================================
# Agno Tools for Index Cache Building
# ============================================================================

def _get_index_constituents_impl(index_symbol: str) -> Dict[str, Any]:
    """
    Get list of ticker symbols for an index using FMP API.
    
    Args:
        index_symbol: Index symbol (e.g., '^GSPC')
    
    Returns:
        Dict with constituents list
    """
    # Try FMP API first
    client = _get_fmp_client()
    if client:
        try:
            constituents = client.get_index_constituents(index_symbol)
            if constituents:
                return {
                    "constituents": constituents,
                    "count": len(constituents),
                    "source": "fmp"
                }
        except Exception as e:
            logger.debug(f"FMP index constituents failed for {index_symbol}: {e}")
    
    # Fallback to web scraping
    return get_index_constituents_from_web(index_symbol)


@tool
def get_index_constituents(index_symbol: str) -> Dict[str, Any]:
    """
    Get list of ticker symbols for an index (Agno tool wrapper).
    Uses FMP API first, falls back to web scraping if FMP fails.
    """
    return _get_index_constituents_impl(index_symbol)


def get_index_constituents_direct(index_symbol: str) -> Dict[str, Any]:
    """
    Direct callable version (not wrapped as Agno tool).
    Get list of ticker symbols for an index.
    Uses FMP API first, falls back to web scraping if FMP fails.
    """
    return _get_index_constituents_impl(index_symbol)


def get_index_constituents_from_web(index_symbol: str) -> Dict[str, Any]:
    """
    Get index constituents using web scraping as fallback.
    
    Args:
        index_symbol: Index symbol (e.g., '^GSPC')
    
    Returns:
        Dict with constituents list
    """
    index_names = {
        '^GSPC': 'S&P 500',
    }
    
    index_name = index_names.get(index_symbol, index_symbol)
    
    # Method 1: Try web scraping for S&P 500 (most reliable free source)
    if index_symbol == '^GSPC':
        symbols = get_sp500_from_web()
        if symbols and len(symbols) > 400:
            return {"constituents": symbols, "count": len(symbols), "source": "web_scraping"}
    
    # Method 2: Use AI agent to search for index constituents
    try:
        # Create a specialized agent for index constituent discovery
        if not GEMINI_AVAILABLE or GeminiChat is None:
            logger.debug("Gemini not available, skipping AI agent method")
            return {"error": "Gemini not available", "constituents": []}
        
        discovery_agent = create_index_discovery_agent()
        if discovery_agent is None:
            return {"error": "Failed to create discovery agent", "constituents": []}
        
        query = f"What are the current ticker symbols (stock symbols) that make up the {index_name} ({index_symbol})? Provide a comprehensive list of all constituent companies."
        
        response = discovery_agent.run(query)
        
        # Parse the response to extract ticker symbols
        symbols = parse_tickers_from_ai_response(str(response), index_symbol)
        
        if symbols:
            return {"constituents": symbols, "count": len(symbols), "source": "AI_agent"}
        else:
            return {"error": "Could not extract ticker symbols from AI response", "constituents": []}
    
    except Exception as e:
        logger.error(f"AI agent method failed: {e}")
        return {"error": f"All methods failed: {str(e)}", "constituents": []}


def get_sp500_from_web() -> List[str]:
    """Get S&P 500 list from web using requests and BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
        
        # Try to get from Wikipedia - most reliable public source
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find the table with S&P 500 companies (id='constituents' or class-based)
            table = soup.find('table', {'id': 'constituents'})
            if not table:
                # Try alternative table identifiers
                table = soup.find('table', class_='wikitable')
            
            if table:
                symbols = []
                for row in table.find_all('tr')[1:]:  # Skip header
                    cells = row.find_all('td')
                    if cells and len(cells) > 0:
                        # First cell usually contains the ticker symbol
                        symbol = cells[0].text.strip().split('\n')[0]  # Get first line if multiline
                        # Clean up the symbol (remove any extra characters)
                        symbol = re.sub(r'[^A-Z.]', '', symbol.upper())
                        if symbol and 1 <= len(symbol) <= 5:
                            symbols.append(symbol)
                
                if len(symbols) > 400:  # S&P 500 should have ~500 companies
                    logger.info(f"Successfully scraped {len(symbols)} S&P 500 symbols from Wikipedia")
                    return symbols
                else:
                    logger.warning(f"Only found {len(symbols)} symbols, expected ~500")
    except ImportError:
        logger.warning("BeautifulSoup4 not installed. Install with: pip install beautifulsoup4")
    except Exception as e:
        logger.warning(f"Web scraping failed: {e}")
    
    return []


def parse_tickers_from_ai_response(response: str, index_symbol: str) -> List[str]:
    """Parse ticker symbols from AI agent response."""
    import re
    
    # Common patterns for ticker symbols
    # Tickers are usually 1-5 uppercase letters, sometimes with dots
    ticker_pattern = r'\b([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b'
    
    # Extract potential tickers
    potential_tickers = re.findall(ticker_pattern, response)
    
    # Extended false positives list (common words, index symbols, etc.)
    false_positives = {
        # Common words
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 
        'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 
        'WAY', 'USE', 'MAN', 'YEAR', 'YOUR', 'THEM', 'THESE', 'THOSE', 'WHAT', 'WHEN', 'WHERE', 'WHICH', 
        'WHILE', 'AFTER', 'BEFORE', 'ABOUT', 'ABOVE', 'ACROSS', 'AGAIN', 'ALONG', 'AMONG', 'AROUND', 
        'BECAUSE', 'BECOME', 'BECAME', 'BEHIND', 'BELOW', 'BESIDE', 'BETWEEN', 'BEYOND', 'DURING', 'EXCEPT', 
        'INSIDE', 'INSTEAD', 'INTO', 'NEAR', 'NEITHER', 'NOR', 'ONCE', 'ONLY', 'OTHER', 'OUTSIDE', 'OVER', 
        'PERHAPS', 'RATHER', 'REALLY', 'SEVERAL', 'SHOULD', 'SINCE', 'SOME', 'SUCH', 'SUDDEN', 'SURE', 
        'THAN', 'THAT', 'THERE', 'THESE', 'THEY', 'THING', 'THINK', 'THOSE', 'THREE', 'THROUGH', 
        'THROUGHOUT', 'THUS', 'TOGETHER', 'TOWARD', 'TOWARDS', 'UNDER', 'UNDERNEATH', 'UNLESS', 'UNTIL', 
        'UPON', 'USED', 'USING', 'USUALLY', 'VARIOUS', 'VERSUS', 'VERY', 'VIA', 'WANT', 'WANTS', 'WAS', 
        'WERE', 'WHAT', 'WHEN', 'WHERE', 'WHETHER', 'WHICH', 'WHILE', 'WHO', 'WHOM', 'WHOSE', 'WILL', 
        'WITH', 'WITHIN', 'WITHOUT', 'WOULD', 'YEAR', 'YEARS', 'YES', 'YET', 'YOU', 'YOUR', 'YOURS', 
        'YOURSELF', 'YOURSELVES',
        # Index symbols and common abbreviations
        'GSPC', 'SPX',
        # Single letters that are likely false positives
        'I', 'A', 'S', 'P', 'N', 'D', 'C', 'M', 'T', 'E', 'R', 'O', 'L', 'F', 'U', 'G', 'H', 'B', 'V', 
        'K', 'J', 'Q', 'W', 'X', 'Y', 'Z'
    }
    
    # Filter and clean
    valid_tickers = []
    for ticker in potential_tickers:
        ticker_upper = ticker.upper()
        # Skip if it's a false positive, too short/long, or not alphabetic
        if (ticker_upper not in false_positives and 
            2 <= len(ticker_upper) <= 5 and  # Require at least 2 characters
            ticker_upper.isalpha()):
            valid_tickers.append(ticker_upper)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in valid_tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)
    
    return unique_tickers


def create_index_discovery_agent() -> Agent:
    """Create an AI agent specialized in discovering index constituents."""
    tools_list = []
    if exa_tool:
        tools_list.append(exa_tool)
    tools_list.append(SafeDuckDuckGoTools())
    
    if not GEMINI_AVAILABLE or GeminiChat is None:
        logger.error("Gemini is not available. AI agent features will be disabled.")
        return None
    
    return Agent(
        name="Index Constituent Discovery Agent",
        model=GeminiChat(
            id=GEMINI_MODEL_FAST,  # Use fastest model
            api_key=GEMINI_API_KEY,  # Using Gemini API
        ),
        tools=tools_list,
        instructions=dedent("""
        You are a financial data research specialist. Your task is to find comprehensive, 
        accurate lists of stock ticker symbols that make up major stock market indexes.
        
        When asked about index constituents:
        1. Search for the most current and official list of constituents
        2. Extract all ticker symbols accurately
        3. Provide the list in a clear, structured format
        4. Focus on accuracy - only include tickers that are actually part of the index
        
        Format your response with ticker symbols clearly listed, one per line or in a comma-separated list.
        """),
    )


def _validate_tickers_batch_impl(tickers: List[str]) -> Dict[str, Any]:
    """
    Validate tickers using FMP API.
    
    Args:
        tickers: List of ticker symbols
    
    Returns:
        Dict with validated tickers and coverage
    """
    logger.info(f"_validate_tickers_batch_impl: Starting validation for {len(tickers)} tickers")
    
    if not tickers:
        logger.warning("_validate_tickers_batch_impl: Empty tickers list provided")
        return {"validated": [], "coverage": 0.0, "source": "none"}
    
    # Use FMP API
    logger.debug("_validate_tickers_batch_impl: Getting FMP client")
    client = _get_fmp_client()
    if not client:
        logger.error("_validate_tickers_batch_impl: FMP client not available")
        return {"error": "FMP client not available", "validated": [], "coverage": 0.0, "source": "none"}
    
    logger.info(f"_validate_tickers_batch_impl: FMP client obtained successfully")
    validated = []
    
    # Process in batches to avoid rate limiting
    batch_size = 50
    total_batches = (len(tickers) - 1) // batch_size + 1
    logger.info(f"_validate_tickers_batch_impl: Processing {total_batches} batches of size {batch_size}")
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"_validate_tickers_batch_impl: Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)")
        logger.debug(f"_validate_tickers_batch_impl: Batch {batch_num} symbols: {batch[:5]}..." if len(batch) > 5 else f"_validate_tickers_batch_impl: Batch {batch_num} symbols: {batch}")
        
        try:
            # Use FMP batch profile endpoint to validate
            logger.debug(f"_validate_tickers_batch_impl: Calling get_profiles_batch for batch {batch_num}")
            profiles = client.get_profiles_batch(batch)
            logger.debug(f"_validate_tickers_batch_impl: Batch {batch_num} - get_profiles_batch returned {len(profiles)} profiles")
            
            batch_validated = 0
            for symbol in batch:
                if symbol in profiles and profiles[symbol]:
                    validated.append(symbol)
                    batch_validated += 1
                else:
                    logger.debug(f"_validate_tickers_batch_impl: Symbol {symbol} not found in profiles or profile is empty")
            
            logger.info(f"_validate_tickers_batch_impl: Batch {batch_num} - {batch_validated}/{len(batch)} validated")
            
        except Exception as e:
            logger.warning(f"_validate_tickers_batch_impl: FMP batch validation failed for batch {batch_num}: {e}")
            logger.debug(f"_validate_tickers_batch_impl: Exception details: {type(e).__name__}: {str(e)}")
            # Fallback to individual validation
            individual_validated = 0
            for symbol in batch:
                try:
                    logger.debug(f"_validate_tickers_batch_impl: Trying individual validation for {symbol}")
                    profile = client.get_profile(symbol)
                    if profile:
                        validated.append(symbol)
                        individual_validated += 1
                        logger.debug(f"_validate_tickers_batch_impl: {symbol} validated individually")
                    else:
                        logger.debug(f"_validate_tickers_batch_impl: {symbol} - profile returned None/empty")
                except Exception as e2:
                    logger.debug(f"_validate_tickers_batch_impl: Individual validation failed for {symbol}: {type(e2).__name__}: {str(e2)}")
                    continue
            logger.info(f"_validate_tickers_batch_impl: Batch {batch_num} fallback - {individual_validated}/{len(batch)} validated individually")
    
    coverage = len(validated) / len(tickers) if tickers else 0
    logger.info(f"_validate_tickers_batch_impl: Validation complete - {len(validated)}/{len(tickers)} validated ({coverage:.2%} coverage)")
    
    result = {
        "validated": validated,
        "coverage": coverage,
        "total": len(tickers),
        "valid_count": len(validated),
        "source": "fmp"
    }
    logger.debug(f"_validate_tickers_batch_impl: Returning result: {result}")
    return result




@tool
def validate_tickers_batch(tickers: List[str]) -> Dict[str, Any]:
    """
    Validate tickers are available in FMP (Agno tool wrapper).
    """
    return _validate_tickers_batch_impl(tickers)


def validate_tickers_batch_direct(tickers: List[str]) -> Dict[str, Any]:
    """
    Direct callable version (not wrapped as Agno tool).
    Validate tickers using FMP API.
    """
    return _validate_tickers_batch_impl(tickers)


def _segregate_by_sector_impl(tickers: List[str]) -> Dict[str, Any]:
    """
    Implementation for segregating tickers by sector and industry.
    Uses FMP API first, then web scraping if needed.
    
    Args:
        tickers: List of validated ticker symbols
    
    Returns:
        Dict with sector_groups and industry_groups
    """
    logger.info(f"_segregate_by_sector_impl: Starting segregation for {len(tickers)} tickers")
    
    if not tickers:
        logger.warning("_segregate_by_sector_impl: Empty tickers list provided")
        return {"sector_groups": {}, "industry_groups": {}}
    
    logger.debug(f"_segregate_by_sector_impl: First 10 tickers: {tickers[:10]}")
    
    sector_groups: Dict[str, List[str]] = {}
    industry_groups: Dict[str, List[str]] = {}
    
    logger.debug("_segregate_by_sector_impl: Getting FMP client")
    client = _get_fmp_client()
    
    if not client:
        logger.warning("_segregate_by_sector_impl: FMP client not available, will use fallback methods only")
    
    fmp_success = 0
    fmp_partial = 0
    fallback_used = 0
    unknown_count = 0
    
    try:
        # Use FMP API first
        logger.info(f"_segregate_by_sector_impl: Processing {len(tickers)} tickers")
        
        for idx, symbol in enumerate(tickers, 1):
            if idx % 50 == 0 or idx == 1:
                logger.info(f"_segregate_by_sector_impl: Processing ticker {idx}/{len(tickers)}: {symbol}")
            
            sector = None
            industry = None
            source = None
            
            # Try FMP API
            if client:
                try:
                    logger.debug(f"_segregate_by_sector_impl: Fetching FMP profile for {symbol}")
                    profile = client.get_profile(symbol)
                    if profile:
                        sector = profile.get('sector', '')
                        industry = profile.get('industry', '')
                        if sector and industry:
                            fmp_success += 1
                            source = "fmp"
                            logger.debug(f"_segregate_by_sector_impl: {symbol} - FMP: sector={sector}, industry={industry}")
                        elif sector or industry:
                            fmp_partial += 1
                            source = "fmp_partial"
                            logger.debug(f"_segregate_by_sector_impl: {symbol} - FMP partial: sector={sector or 'None'}, industry={industry or 'None'}")
                        else:
                            logger.debug(f"_segregate_by_sector_impl: {symbol} - FMP returned empty sector/industry")
                    else:
                        logger.debug(f"_segregate_by_sector_impl: {symbol} - FMP profile returned None")
                except Exception as e:
                    logger.debug(f"_segregate_by_sector_impl: FMP profile failed for {symbol}: {type(e).__name__}: {str(e)}")
            
            # Fallback to web scraping if FMP didn't provide both
            if not sector or not industry:
                if source != "fmp":
                    logger.debug(f"_segregate_by_sector_impl: {symbol} - Trying fallback method for sector/industry")
                    try:
                        fallback_data = _get_sector_industry_with_fallback(symbol)
                        if fallback_data:
                            sector = fallback_data.get('sector', sector) or 'Unknown'
                            industry = fallback_data.get('industry', industry) or 'Unknown'
                            if sector != 'Unknown' or industry != 'Unknown':
                                fallback_used += 1
                                source = "fallback"
                                logger.debug(f"_segregate_by_sector_impl: {symbol} - Fallback: sector={sector}, industry={industry}")
                            else:
                                unknown_count += 1
                                logger.debug(f"_segregate_by_sector_impl: {symbol} - Fallback returned Unknown")
                        else:
                            sector = sector or 'Unknown'
                            industry = industry or 'Unknown'
                            unknown_count += 1
                            logger.debug(f"_segregate_by_sector_impl: {symbol} - No fallback data, using Unknown")
                    except Exception as e:
                        logger.debug(f"_segregate_by_sector_impl: {symbol} - Fallback method failed: {type(e).__name__}: {str(e)}")
                        sector = sector or 'Unknown'
                        industry = industry or 'Unknown'
                        unknown_count += 1
                else:
                    # FMP provided partial data, fill in missing with Unknown
                    sector = sector or 'Unknown'
                    industry = industry or 'Unknown'
                    if sector == 'Unknown' or industry == 'Unknown':
                        unknown_count += 1
            
            # Group by sector
            if sector not in sector_groups:
                sector_groups[sector] = []
                logger.debug(f"_segregate_by_sector_impl: New sector group created: {sector}")
            sector_groups[sector].append(symbol)
            
            # Group by industry
            if industry not in industry_groups:
                industry_groups[industry] = []
                logger.debug(f"_segregate_by_sector_impl: New industry group created: {industry}")
            industry_groups[industry].append(symbol)
        
        logger.info(f"_segregate_by_sector_impl: Segregation complete")
        logger.info(f"_segregate_by_sector_impl: FMP success: {fmp_success}, FMP partial: {fmp_partial}, Fallback used: {fallback_used}, Unknown: {unknown_count}")
        logger.info(f"_segregate_by_sector_impl: Created {len(sector_groups)} sector groups and {len(industry_groups)} industry groups")
        
        # Log sector distribution
        sector_dist = {sector: len(symbols) for sector, symbols in sector_groups.items()}
        logger.info(f"_segregate_by_sector_impl: Sector distribution: {dict(sorted(sector_dist.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        result = {
            "sector_groups": sector_groups,
            "industry_groups": industry_groups,
            "sector_count": len(sector_groups),
            "industry_count": len(industry_groups)
        }
        logger.debug(f"_segregate_by_sector_impl: Returning result with {len(sector_groups)} sectors and {len(industry_groups)} industries")
        return result
    
    except Exception as e:
        logger.error(f"_segregate_by_sector_impl: Error segregating tickers: {type(e).__name__}: {str(e)}")
        import traceback
        logger.debug(f"_segregate_by_sector_impl: Traceback: {traceback.format_exc()}")
        return {
            "sector_groups": sector_groups,
            "industry_groups": industry_groups,
            "sector_count": len(sector_groups),
            "industry_count": len(industry_groups),
            "error": str(e)
        }


@tool
def segregate_by_sector(tickers: List[str]) -> Dict[str, Any]:
    """
    Segregate tickers by sector and industry (Agno tool wrapper).
    Uses FMP API first, then web scraping if needed.
    """
    return _segregate_by_sector_impl(tickers)


def segregate_by_sector_direct(tickers: List[str]) -> Dict[str, Any]:
    """
    Direct callable version (not wrapped as Agno tool).
    Segregate tickers by sector and industry.
    Uses FMP API first, then web scraping if needed.
    """
    return _segregate_by_sector_impl(tickers)


def _get_sector_industry_from_ai_agent(symbol: str) -> Optional[Dict[str, str]]:
    """
    Get sector and industry using AI agent search.
    Uses AI agents to search for sector and industry information.
    
    Args:
        symbol: Ticker symbol
    
    Returns:
        Dict with 'sector' and 'industry' keys, or None if unavailable
    """
    # Use AI agent to search for sector and industry
    try:
        if not GEMINI_AVAILABLE or GeminiChat is None:
            logger.debug(f"_get_sector_industry_from_ai_agent: Gemini not available for {symbol}")
            return None
        
        logger.debug(f"_get_sector_industry_from_ai_agent: Creating discovery agent for {symbol}")
        discovery_agent = create_sector_industry_discovery_agent()
        if discovery_agent is None:
            logger.debug(f"_get_sector_industry_from_ai_agent: Failed to create discovery agent for {symbol}")
            return None
        
        query = f"What is the sector and industry classification for stock ticker {symbol}? Provide the exact sector name and industry name."
        logger.debug(f"_get_sector_industry_from_ai_agent: Querying AI agent for {symbol}")
        
        response = discovery_agent.run(query)
        response_text = str(response)
        
        logger.debug(f"_get_sector_industry_from_ai_agent: AI agent response for {symbol}: {response_text[:200]}...")
        
        # Parse the response to extract sector and industry
        sector, industry = _parse_sector_industry_from_text(response_text, symbol)
        
        if sector or industry:
            logger.info(f"_get_sector_industry_from_ai_agent: {symbol} - AI agent found: sector={sector}, industry={industry}")
            return {
                'sector': sector if sector else 'Unknown',
                'industry': industry if industry else 'Unknown'
            }
        else:
            logger.debug(f"_get_sector_industry_from_ai_agent: {symbol} - Could not extract sector/industry from AI response")
    except Exception as e:
        logger.warning(f"_get_sector_industry_from_ai_agent: AI agent search failed for {symbol}: {type(e).__name__}: {str(e)}")
        import traceback
        logger.debug(f"_get_sector_industry_from_ai_agent: Traceback: {traceback.format_exc()}")
    
    return None


def _parse_sector_industry_from_text(text: str, symbol: str) -> tuple:
    """
    Parse sector and industry from text response.
    
    Args:
        text: Text to parse
        symbol: Ticker symbol (for context)
    
    Returns:
        Tuple of (sector, industry) or (None, None)
    """
    sector = None
    industry = None
    
    # Common sector patterns
    sector_patterns = [
        r'Sector[:\s]+([A-Za-z\s&,]+?)(?:\n|Industry|$)',
        r'sector[:\s]+([A-Za-z\s&,]+?)(?:\n|industry|$)',
        r'GICS Sector[:\s]+([A-Za-z\s&,]+?)(?:\n|$)',
    ]
    
    # Common industry patterns
    industry_patterns = [
        r'Industry[:\s]+([A-Za-z\s&,]+?)(?:\n|$)',
        r'industry[:\s]+([A-Za-z\s&,]+?)(?:\n|$)',
        r'GICS Industry[:\s]+([A-Za-z\s&,]+?)(?:\n|$)',
    ]
    
    # Try to extract sector
    for pattern in sector_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            sector = match.group(1).strip()
            # Clean up common false positives
            if len(sector) > 3 and len(sector) < 100 and not sector.lower().startswith('the '):
                break
    
    # Try to extract industry
    for pattern in industry_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            industry = match.group(1).strip()
            # Clean up common false positives
            if len(industry) > 3 and len(industry) < 100 and not industry.lower().startswith('the '):
                break
    
    # Common sectors list for validation
    common_sectors = [
        'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
        'Consumer Defensive', 'Energy', 'Industrials', 'Communication Services',
        'Utilities', 'Real Estate', 'Basic Materials', 'Consumer Staples'
    ]
    
    # Validate sector if found
    if sector:
        sector_upper = sector.upper()
        # Check if it matches a common sector (fuzzy match)
        for common_sector in common_sectors:
            if common_sector.upper() in sector_upper or sector_upper in common_sector.upper():
                sector = common_sector
                break
    
    return (sector, industry)


def create_sector_industry_discovery_agent() -> Agent:
    """Create an AI agent specialized in discovering sector and industry information."""
    tools_list = []
    if exa_tool:
        tools_list.append(exa_tool)
    tools_list.append(SafeDuckDuckGoTools())
    
    if not GEMINI_AVAILABLE or GeminiChat is None:
        logger.error("Gemini is not available. AI agent features will be disabled.")
        return None
    
    return Agent(
        name="Sector Industry Discovery Agent",
        model=GeminiChat(
            id=GEMINI_MODEL_FAST,  # Use fastest model
            api_key=GEMINI_API_KEY,  # Using Gemini API
        ),
        tools=tools_list,
        instructions=dedent("""
        You are a financial data research specialist. Your task is to find accurate sector and industry 
        classifications for stock ticker symbols.
        
        When asked about a stock's sector and industry:
        1. Search for the most current and official sector and industry classification
        2. Use reliable sources like Yahoo Finance, MarketWatch, company websites, or SEC filings
        3. Provide the exact sector name and industry name in a clear format
        4. Format your response as: "Sector: [sector name]" and "Industry: [industry name]"
        
        Be precise and use standard sector/industry classifications (e.g., GICS classifications).
        """),
    )


def _get_sector_industry_with_fallback(symbol: str) -> Optional[Dict[str, str]]:
    """
    Get sector and industry with fallback chain: FMP API -> AI agents.
    
    Args:
        symbol: Ticker symbol
    
    Returns:
        Dict with 'sector' and 'industry' keys, or None if unavailable
    """
    logger.debug(f"_get_sector_industry_with_fallback: Starting for {symbol}")
    
    # Try FMP API first
    client = _get_fmp_client()
    if client:
        try:
            logger.debug(f"_get_sector_industry_with_fallback: Trying FMP API for {symbol}")
            profile = client.get_profile(symbol)
            if profile:
                sector = profile.get('sector', '')
                industry = profile.get('industry', '')
                if sector and industry:
                    logger.debug(f"_get_sector_industry_with_fallback: {symbol} - FMP provided both: sector={sector}, industry={industry}")
                    return {
                        'sector': sector,
                        'industry': industry
                    }
                else:
                    logger.debug(f"_get_sector_industry_with_fallback: {symbol} - FMP partial: sector={sector or 'None'}, industry={industry or 'None'}")
        except Exception as e:
            logger.debug(f"_get_sector_industry_with_fallback: FMP profile failed for {symbol}: {type(e).__name__}: {str(e)}")
    else:
        logger.debug(f"_get_sector_industry_with_fallback: FMP client not available for {symbol}")
    
    # If FMP didn't provide both, try AI agent
    logger.debug(f"_get_sector_industry_with_fallback: Trying AI agent for {symbol}")
    ai_result = _get_sector_industry_from_ai_agent(symbol)
    if ai_result:
        logger.info(f"_get_sector_industry_with_fallback: {symbol} - AI agent provided: sector={ai_result.get('sector')}, industry={ai_result.get('industry')}")
        return ai_result
    
    logger.debug(f"_get_sector_industry_with_fallback: {symbol} - All methods failed, returning None")
    return None


# ============================================================================
# FMP Validation Tools
# ============================================================================

@tool
def validate_fmp_profile(symbol: str) -> Dict[str, Any]:
    """
    Validate profile data using FMP API.
    
    Args:
        symbol: Ticker symbol to validate
    
    Returns:
        Dict with validation results and data quality metrics
    """
    try:
        client = _get_fmp_client()
        if not client:
            return {
                "valid": False,
                "error": "FMP client not available",
                "symbol": symbol
            }
        
        profile = client.get_profile(symbol)
        quote = client.get_quote(symbol)
        
        if not profile:
            return {
                "valid": False,
                "error": "No profile data returned",
                "symbol": symbol
            }
        
        # Check required fields
        required_fields = ['symbol', 'companyName', 'sector', 'industry', 'marketCap']
        missing_fields = [field for field in required_fields if not profile.get(field)]
        
        has_price = quote is not None and quote.get('price') is not None
        
        # Calculate completeness score
        completeness_score = ((len(required_fields) - len(missing_fields)) / len(required_fields)) * 100
        
        return {
            "valid": len(missing_fields) == 0 and has_price,
            "symbol": symbol,
            "has_profile": True,
            "has_quote": has_price,
            "missing_fields": missing_fields,
            "completeness_score": completeness_score,
            "market_cap": profile.get('marketCap') or profile.get('mktCap', 0),
            "sector": profile.get('sector'),
            "industry": profile.get('industry'),
            "price": quote.get('price') if quote else None
        }
    
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "symbol": symbol
        }


@tool
def validate_fmp_batch(tickers: List[str]) -> Dict[str, Any]:
    """
    Validate multiple profiles in batch using FMP API.
    
    Args:
        tickers: List of ticker symbols to validate
    
    Returns:
        Dict with validation summary and per-ticker results
    """
    if not tickers:
        return {"validated": [], "invalid": [], "summary": {}}
    
    validated = []
    invalid = []
    results = {}
    
    try:
        client = _get_fmp_client()
        if not client:
            return {"validated": [], "invalid": tickers, "summary": {"error": "FMP client not available"}}
        
        # Fetch profiles and quotes using FMP API batch endpoints
        profiles = client.get_profiles_batch(tickers[:100])
        quotes = client.get_quotes_batch(tickers[:100])
        
        for symbol in tickers[:100]:
            profile = profiles.get(symbol)
            quote = quotes.get(symbol)
            
            if profile and quote:
                # Check data quality
                has_required = all([
                    profile.get('symbol'),
                    profile.get('companyName'),
                    profile.get('sector'),
                    quote.get('price')
                ])
                
                if has_required:
                    validated.append(symbol)
                    results[symbol] = {
                        "valid": True,
                        "market_cap": profile.get('marketCap'),
                        "price": quote.get('price'),
                        "sector": profile.get('sector')
                    }
                else:
                    invalid.append(symbol)
                    results[symbol] = {"valid": False, "reason": "Missing required fields"}
            else:
                invalid.append(symbol)
                results[symbol] = {"valid": False, "reason": "No data returned"}
        
        return {
            "validated": validated,
            "invalid": invalid,
            "results": results,
            "summary": {
                "total": len(tickers[:100]),
                "valid_count": len(validated),
                "invalid_count": len(invalid),
                "coverage": len(validated) / len(tickers[:100]) if tickers else 0
            }
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "validated": [],
            "invalid": tickers[:100],
            "summary": {}
        }


# ============================================================================
# FMP Validation Tools
# ============================================================================

@tool
def validate_ticker_data_quality(symbol: str) -> Dict[str, Any]:
    """
    Comprehensive data quality validation using FMP API.
    
    Args:
        symbol: Ticker symbol to validate
    
    Returns:
        Dict with comprehensive validation results and quality score
    """
    fmp_val = validate_fmp_profile(symbol)
    
    # Calculate quality score (0-100)
    score = 0
    max_score = 100
    
    if fmp_val.get("valid"):
        score += 80  # FMP data available
    if fmp_val.get("completeness_score", 0) >= 80:
        score += 20  # High completeness
    
    quality_rating = "EXCELLENT" if score >= 90 else "GOOD" if score >= 70 else "FAIR" if score >= 50 else "POOR"
    
    return {
        "symbol": symbol,
        "quality_score": score,
        "quality_rating": quality_rating,
        "fmp_validation": fmp_val,
        "recommendation": "USE" if score >= 70 else "REVIEW" if score >= 50 else "SKIP"
    }


# ============================================================================
# Agno Agents
# ============================================================================

def create_constituent_fetcher_agent() -> Agent:
    """Create agent for fetching index constituents."""
    return Agent(
        model=GeminiChat(id=GEMINI_MODEL_FAST, api_key=GEMINI_API_KEY),
        name="Constituent Fetcher Agent",
        role=dedent("""
            You are responsible for fetching index constituents.
            Use the get_index_constituents tool to retrieve ticker symbols for an index.
            Return the list of constituents in a structured format.
        """),
        tools=[get_index_constituents],
        markdown=True,
    )


def create_ticker_validator_agent() -> Agent:
    """Create agent for validating tickers with FMP API."""
    tools = [validate_tickers_batch, validate_fmp_batch, validate_fmp_profile, validate_ticker_data_quality]
    
    return Agent(
        model=GeminiChat(id=GEMINI_MODEL_FAST, api_key=GEMINI_API_KEY),
        name="Ticker Validator Agent",
        role=dedent("""
            You are responsible for validating tickers using FMP API.
            Use FMP tools to validate ticker data.
            Check data quality, completeness, and consistency.
            Report any data quality issues.
            Calculate and report coverage ratio with quality metrics.
        """),
        tools=tools,
        markdown=True,
    )


def create_sector_segregator_agent() -> Agent:
    """Create agent for segregating by sector/industry with validation."""
    tools = [segregate_by_sector, validate_fmp_profile]
    
    return Agent(
        model=GeminiChat(id=GEMINI_MODEL_FAST, api_key=GEMINI_API_KEY),
        name="Sector Segregator Agent",
        role=dedent("""
            You are responsible for segregating tickers by sector and industry.
            Use the segregate_by_sector tool to group tickers.
            Validate sector/industry classifications using FMP API.
            Cross-check sector assignments for accuracy.
            Provide insights on sector distribution with validation notes.
        """),
        tools=tools,
        markdown=True,
    )


def create_market_intelligence_agent() -> Agent:
    """Create agent for gathering market intelligence."""
    tools = []
    if exa_tool:
        tools.append(exa_tool)
    tools.append(SafeDuckDuckGoTools())
    
    return Agent(
        model=GeminiChat(id=GEMINI_MODEL_FAST, api_key=GEMINI_API_KEY),
        name="Market Intelligence Agent",
        role=dedent(f"""
            You are responsible for gathering market intelligence using search tools.
            Use Exa search (if available) or DuckDuckGo to find:
            - Index-level market analysis
            - Sector-specific trends
            - Recent news and developments
            
            Provide structured intelligence reports with citations.
            Only use URLs from search results - never fabricate URLs.
        """),
        tools=tools,
        markdown=True,
    )


def create_data_validation_agent() -> Agent:
    """Create agent for comprehensive data validation."""
    tools = [
        validate_fmp_profile,
        validate_fmp_batch,
        validate_tickers_batch,
        validate_ticker_data_quality
    ]
    
    return Agent(
        model=GeminiChat(id=GEMINI_MODEL_FAST, api_key=GEMINI_API_KEY),
        name="Data Validation Agent",
        role=dedent("""
            You are responsible for comprehensive data validation using FMP API.
            Check data quality, completeness, and consistency.
            Identify discrepancies and data quality issues.
            Provide validation reports with quality scores and recommendations.
            
            Validation checks:
            - Profile completeness (required fields)
            - Price data accuracy
            - Sector/industry classification consistency
            - Market cap consistency
            - Data freshness
            
            Flag any tickers with:
            - Missing required fields
            - Sector/industry mismatches
            - Missing or stale data
        """),
        tools=tools,
        markdown=True,
    )


def create_index_cache_team() -> Team:
    """Create Agno Team for index cache building with validation."""
    members = [
        create_constituent_fetcher_agent(),
        create_ticker_validator_agent(),
        create_sector_segregator_agent(),
        create_data_validation_agent(),
    ]
    
    # Add market intelligence agent if Exa is available
    if exa_tool:
        members.append(create_market_intelligence_agent())
    
    return Team(
        model=GeminiChat(id=GEMINI_MODEL_FAST, api_key=GEMINI_API_KEY),
        name="Index Cache Builder Team",
        members=members,
        role=dedent("""
            You are a team of specialized agents working together to build validated index cache.
            
            Workflow:
            1. Constituent Fetcher: Fetch all tickers from index
            2. Ticker Validator: Validate tickers using FMP API
            3. Sector Segregator: Group tickers by sector/industry (with validation)
            4. Data Validator: Comprehensive quality checks and cross-validation
            5. Market Intelligence: Gather market insights (if available)
            
            All data must be validated across multiple sources before being cached.
            Flag any discrepancies or quality issues for review.
            Coordinate to build comprehensive, validated index cache with all required data.
        """),
        markdown=True,
    )

