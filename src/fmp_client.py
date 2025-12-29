"""
FMP API client with sync/async support, retries, backoff, and batch endpoints.
"""

import os
import time
import requests
import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional, Any, Union
from functools import wraps
import pandas as pd
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class FMPClient:
    """Financial Modeling Prep API client with retry logic and batch support."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP client.
        
        Args:
            api_key: FMP API key. If None, reads from FMP_API_KEY env var or Streamlit secrets.
        """
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.getenv('FMP_API_KEY')
        if not self.api_key:
            # Try Streamlit secrets
            try:
                import streamlit as st
                self.api_key = st.secrets.get("FMP_API_KEY")
            except:
                pass
        if not self.api_key:
            raise ValueError("FMP_API_KEY must be provided, set as environment variable, or in Streamlit secrets")
        
        # Use stable API base URL as per FMP documentation
        # Documentation: https://financialmodelingprep.com/stable/
        self.base_url = "https://financialmodelingprep.com/stable"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'EquitiesTool/1.0'})
        # Configure session to handle redirects properly
        self.session.max_redirects = 5  # Limit redirects to prevent loops
        
    def _retry_request(self, max_retries: int = 3, backoff_factor: float = 1.0):
        """Decorator for retrying requests with exponential backoff."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except (requests.exceptions.RequestException, 
                            requests.exceptions.HTTPError) as e:
                        if attempt == max_retries - 1:
                            raise
                        wait_time = backoff_factor * (2 ** attempt)
                        time.sleep(wait_time)
                return None
            return wrapper
        return decorator
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request with retry logic."""
        if params is None:
            params = {}
        # Add API key to params (FMP uses ?apikey= parameter)
        params['apikey'] = self.api_key
        
        # Remove leading slash from endpoint if present
        endpoint = endpoint.lstrip('/')
        
        # Handle v3/v4 endpoints - they should use /api/ prefix
        if endpoint.startswith('v3/') or endpoint.startswith('v4/'):
            base_url = "https://financialmodelingprep.com/api"
        else:
            base_url = self.base_url
        
        # Construct URL properly
        if endpoint.startswith('http'):
            url = endpoint  # Already a full URL
        else:
            # Remove leading slash and construct URL
            endpoint_clean = endpoint.lstrip('/')
            url = f"{base_url}/{endpoint_clean}"
        
        for attempt in range(3):
            try:
                logger.debug(f"_get: Making request to {url} with params (excluding apikey)")
                # Make request with redirect handling
                response = self.session.get(url, params=params, timeout=30, allow_redirects=True)
                logger.debug(f"_get: Response status code: {response.status_code}, final URL: {response.url}")
                
                # Check if we got redirected to an error page or wrong endpoint
                if response.status_code >= 400:
                    # If it's a 404, try alternative endpoint format
                    if response.status_code == 404 and endpoint.startswith('v3/'):
                        # Try without v3 prefix using stable API
                        alt_endpoint = endpoint.replace('v3/', '').replace('v4/', '')
                        if alt_endpoint != endpoint:
                            logger.debug(f"_get: Trying alternative endpoint: {alt_endpoint}")
                            alt_url = f"{self.base_url}/{alt_endpoint}"
                            response = self.session.get(alt_url, params=params, timeout=30, allow_redirects=True)
                            logger.debug(f"_get: Alternative endpoint status: {response.status_code}")
                
                response.raise_for_status()
                data = response.json()
                logger.debug(f"_get: Response type: {type(data)}, length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
                if isinstance(data, list) and len(data) == 0:
                    logger.warning(f"_get: Empty list returned from {endpoint}")
                elif not data:
                    logger.warning(f"_get: Empty/None data returned from {endpoint}")
                return data
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                
                if status_code == 402:
                    # 402 Payment Required - endpoint requires subscription upgrade
                    error_msg = (
                        f"402 Payment Required: This endpoint requires a subscription upgrade. "
                        f"Your current FMP plan does not include access to this endpoint.\n"
                        f"Please upgrade your subscription at https://financialmodelingprep.com/"
                    )
                    raise requests.exceptions.HTTPError(error_msg, response=e.response) from e
                
                if status_code == 403:
                    # 403 Forbidden - usually means invalid API key or no access
                    error_msg = (
                        f"403 Forbidden: API access denied. "
                        f"This usually means:\n"
                        f"1. Your API key is invalid or expired\n"
                        f"2. Your API key doesn't have access to this endpoint\n"
                        f"3. You've exceeded your API quota\n\n"
                        f"Please check your FMP_API_KEY in the .env file or Streamlit secrets."
                    )
                    raise requests.exceptions.HTTPError(error_msg, response=e.response) from e
                
                if status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                
                if status_code == 401:
                    error_msg = (
                        f"401 Unauthorized: Invalid API key. "
                        f"Please check your FMP_API_KEY in the .env file or Streamlit secrets."
                    )
                    raise requests.exceptions.HTTPError(error_msg, response=e.response) from e
                
                raise
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(1 * (2 ** attempt))
        
        return []
    
    # Profile endpoints
    def get_profile(self, symbol: str) -> Optional[Dict]:
        """Get company profile for a single symbol."""
        logger.debug(f"get_profile: Fetching profile for {symbol}")
        try:
            # Use stable API format: /profile?symbol={symbol}
            data = self._get("profile", params={"symbol": symbol})
            # Handle both list and dict responses
            if isinstance(data, list):
                result = data[0] if data else None
            else:
                result = data if data else None
            
            if result:
                logger.debug(f"get_profile: Successfully retrieved profile for {symbol}")
            else:
                logger.debug(f"get_profile: No profile data returned for {symbol}")
            return result
        except Exception as e:
            logger.warning(f"get_profile: Error fetching profile for {symbol}: {type(e).__name__}: {str(e)}")
            raise
    
    def get_profiles_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get profiles for multiple symbols (batch endpoint).
        
        Uses FMP stable API: /profile?symbol={symbol1,symbol2,...}
        According to FMP docs: https://site.financialmodelingprep.com/developer/docs
        
        Falls back to individual fetches if batch fails.
        """
        if not symbols:
            logger.debug("get_profiles_batch called with empty symbols list")
            return {}
        
        logger.info(f"get_profiles_batch: Processing {len(symbols)} symbols")
        result = {}
        
        # Process in smaller batches to avoid URL length issues
        # FMP API supports comma-separated symbols in the symbol parameter
        batch_size = 50  # Conservative batch size to avoid URL length limits
        total_batches = (len(symbols) - 1) // batch_size + 1
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            batch_num = i // batch_size + 1
            logger.debug(f"get_profiles_batch: Processing batch {batch_num}/{total_batches} with {len(batch)} symbols")
            
            # Try stable API first with comma-separated symbols
            # Official FMP API: /profile?symbol={symbol1,symbol2,...}
            try:
                symbols_str = ','.join(batch)
                logger.debug(f"get_profiles_batch: Calling FMP stable API for batch {batch_num} with symbols: {batch[:5]}...")
                data = self._get("profile", params={"symbol": symbols_str})
                
                logger.debug(f"get_profiles_batch: Received {len(data) if isinstance(data, list) else type(data).__name__} from API")
                if isinstance(data, list) and len(data) > 0:
                    logger.debug(f"get_profiles_batch: First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
                
                # Process the response
                if data:
                    processed_count = 0
                    for item in data:
                        if item and isinstance(item, dict) and 'symbol' in item:
                            result[item['symbol']] = item
                            processed_count += 1
                        else:
                            logger.debug(f"get_profiles_batch: Skipping item - type: {type(item)}, has symbol: {isinstance(item, dict) and 'symbol' in item if isinstance(item, dict) else False}")
                    logger.info(f"get_profiles_batch: Batch {batch_num} - Processed {processed_count} profiles, added to result: {len([s for s in batch if s in result])}")
                else:
                    logger.warning(f"get_profiles_batch: Batch {batch_num} returned empty data from stable API, trying individual fetches")
                    # If batch returned empty, try individual fetches
                    individual_success = 0
                    for symbol in batch:
                        try:
                            profile = self.get_profile(symbol)
                            if profile:
                                result[symbol] = profile
                                individual_success += 1
                        except Exception as e2:
                            logger.debug(f"get_profiles_batch: Individual fetch failed for {symbol}: {e2}")
                            continue
                    logger.info(f"get_profiles_batch: Batch {batch_num} fallback - {individual_success}/{len(batch)} succeeded individually")
            except Exception as e:
                # Fallback: fetch individually if batch fails
                logger.warning(f"get_profiles_batch: Batch {batch_num} failed with error: {type(e).__name__}: {str(e)}, trying individual fetches")
                individual_success = 0
                for symbol in batch:
                    try:
                        profile = self.get_profile(symbol)
                        if profile:
                            result[symbol] = profile
                            individual_success += 1
                    except Exception as e2:
                        logger.debug(f"get_profiles_batch: Individual fetch failed for {symbol}: {e2}")
                        continue
                logger.info(f"get_profiles_batch: Batch {batch_num} fallback - {individual_success}/{len(batch)} succeeded individually")
        
        logger.info(f"get_profiles_batch: Completed - {len(result)}/{len(symbols)} profiles retrieved")
        return result
    
    # Quote endpoints
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for a symbol."""
        # Try stable API first: /quote?symbol={symbol}
        try:
            data = self._get("quote", params={"symbol": symbol})
            # Handle both list and dict responses
            if isinstance(data, list):
                return data[0] if data else None
            return data if data else None
        except Exception as e:
            logger.debug(f"get_quote: Stable API failed for {symbol}, trying v3: {e}")
            # Fallback to v3 format
            try:
                data = self._get(f"v3/quote/{symbol}")
                return data[0] if data else None
            except Exception as e2:
                logger.warning(f"get_quote: Both stable and v3 failed for {symbol}: {e2}")
                return None
    
    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols (batch endpoint).
        
        Uses FMP stable API: /quote?symbol={symbol1,symbol2,...}
        According to FMP docs: https://site.financialmodelingprep.com/developer/docs
        """
        if not symbols:
            return {}
        
        result = {}
        
        # Try stable API first with comma-separated symbols
        try:
            symbols_str = ','.join(symbols[:100])
            data = self._get("quote", params={"symbol": symbols_str})
            
            if data:
                if isinstance(data, list):
                    for item in data:
                        if item and isinstance(item, dict) and 'symbol' in item:
                            result[item['symbol']] = item
                elif isinstance(data, dict) and 'symbol' in data:
                    result[data['symbol']] = data
                
                if result:
                    logger.debug(f"get_quotes_batch: Stable API returned {len(result)} quotes")
                    return result
        except Exception as e:
            logger.debug(f"get_quotes_batch: Stable API failed, trying v3: {e}")
        
        # Fallback to v3 format
        try:
            symbols_str = ','.join(symbols[:100])
            data = self._get(f"v3/quote/{symbols_str}")
            
            if data:
                for item in data:
                    if item and 'symbol' in item:
                        result[item['symbol']] = item
                logger.debug(f"get_quotes_batch: v3 API returned {len(result)} quotes")
        except Exception as e:
            logger.warning(f"get_quotes_batch: Both stable and v3 failed: {e}")
        
        return result
    
    # Financial statements
    def get_income_statement(self, symbol: str, period: str = 'annual', limit: int = 5) -> pd.DataFrame:
        """Get income statement."""
        # Use stable API format: /income-statement?symbol={symbol}
        params = {'symbol': symbol, 'period': period, 'limit': limit}
        data = self._get("income-statement", params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index(ascending=False)
        
        return df
    
    def get_balance_sheet(self, symbol: str, period: str = 'annual', limit: int = 5) -> pd.DataFrame:
        """Get balance sheet."""
        # Try different endpoint names for balance sheet
        params = {'symbol': symbol, 'period': period, 'limit': limit}
        data = self._get("balance-sheet-statement", params)
        # If empty, try alternative names
        if not data:
            data = self._get("balance-sheet", params)
        if not data:
            # Fallback to v3 format
            data = self._get(f"v3/balance-sheet-statement/{symbol}", {'period': period, 'limit': limit})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index(ascending=False)
        
        return df
    
    def get_cash_flow(self, symbol: str, period: str = 'annual', limit: int = 5) -> pd.DataFrame:
        """Get cash flow statement."""
        # Use stable API format: /cash-flow-statement?symbol={symbol}
        params = {'symbol': symbol, 'period': period, 'limit': limit}
        data = self._get("cash-flow-statement", params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index(ascending=False)
        
        return df
    
    # Key metrics and ratios
    def get_key_metrics(self, symbol: str, period: str = 'annual', limit: int = 5) -> pd.DataFrame:
        """Get key metrics."""
        # Try stable API format first, fallback to v3 if needed
        params = {'symbol': symbol, 'period': period, 'limit': limit}
        data = self._get("key-metrics", params)
        # If empty, try v3 format as fallback
        if not data:
            data = self._get(f"v3/key-metrics/{symbol}", {'period': period, 'limit': limit})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index(ascending=False)
        
        return df
    
    def get_ratios(self, symbol: str, period: str = 'annual', limit: int = 5) -> pd.DataFrame:
        """Get financial ratios."""
        # Try stable API format first, fallback to v3 if needed
        params = {'symbol': symbol, 'period': period, 'limit': limit}
        data = self._get("ratios", params)
        # If empty, try v3 format as fallback
        if not data:
            data = self._get(f"v3/ratios/{symbol}", {'period': period, 'limit': limit})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index(ascending=False)
        
        return df
    
    # Ratings
    def get_rating(self, symbol: str) -> Optional[Dict]:
        """
        Get analyst rating.
        
        Reference: https://site.financialmodelingprep.com/developer/docs#analyst
        Endpoints: rating (stable), v3/rating/{symbol}, rating-bulk
        """
        # According to FMP docs, rating endpoint might be under different path
        # Try multiple endpoint formats
        endpoints_to_try = [
            ("rating", {"symbol": symbol}),  # Stable API format
            (f"v3/rating/{symbol}", None),   # v3 format
            ("rating-bulk", {"symbol": symbol}),  # Bulk format (might work for single)
        ]
        
        for endpoint, params in endpoints_to_try:
            try:
                if params:
                    data = self._get(endpoint, params=params)
                else:
                    data = self._get(endpoint)
                
                if data:
                    if isinstance(data, list):
                        # If it's a list, find the item matching our symbol or return first
                        for item in data:
                            if isinstance(item, dict) and item.get('symbol') == symbol:
                                return item
                        return data[0] if data else None
                    elif isinstance(data, dict):
                        return data
            except Exception as e:
                # Continue to next endpoint if this one fails
                continue
        
        # If all endpoints fail, return None (rating is optional)
        return None
    
    # DCF
    def get_dcf(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """Get DCF values."""
        # Try stable API format first, fallback to v3 if needed
        params = {'symbol': symbol, 'limit': limit}
        data = self._get("discounted-cash-flow", params)
        # If empty, try v3 format as fallback
        if not data:
            data = self._get(f"v3/discounted-cash-flow/{symbol}", {'limit': limit})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index(ascending=False)
        
        return df
    
    # Analyst estimates
    def get_analyst_estimates(self, symbol: str, period: str = 'annual', limit: int = 5) -> pd.DataFrame:
        """
        Get analyst estimates.
        
        Reference: https://site.financialmodelingprep.com/developer/docs#analyst
        Endpoint: analyst-estimates (stable) or v3/analyst-estimates/{symbol}
        """
        # Try stable API format first, fallback to v3 if needed
        params = {'symbol': symbol, 'period': period, 'limit': limit}
        data = self._get("analyst-estimates", params)
        # If empty, try v3 format as fallback
        if not data:
            data = self._get(f"v3/analyst-estimates/{symbol}", {'period': period, 'limit': limit})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index(ascending=False)
        
        return df
    
    # Earnings surprises
    def get_earnings_surprises(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """
        Get earnings surprises.
        
        Reference: https://site.financialmodelingprep.com/developer/docs#earnings-transcript
        Endpoint: earnings-surprises (stable), v3/earnings-surprises/{symbol} (fallback)
        """
        # Try stable API format first
        params = {'symbol': symbol, 'limit': limit}
        try:
            data = self._get("earnings-surprises", params)
        except Exception as e:
            logger.debug(f"get_earnings_surprises: stable API failed: {e}, trying v3")
            data = None
        
        # If empty or error, try v3 format as fallback
        if not data:
            try:
                data = self._get(f"v3/earnings-surprises/{symbol}", {'limit': limit})
            except Exception as e:
                logger.debug(f"get_earnings_surprises: v3 fallback failed: {e}")
                data = None
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index(ascending=False)
        
        return df
    
    # Historical prices
    def get_historical_prices(self, symbol: str, start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """Get historical price data."""
        endpoint = f"v3/historical-price-full/{symbol}"
        params = {}
        
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
        
        data = self._get(endpoint, params)
        
        if not data or 'historical' not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data['historical'])
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
        
        return df
    
    # Institutional holders
    def get_institutional_holders(self, symbol: str) -> pd.DataFrame:
        """
        Get institutional holders (Form 13F data).
        
        Reference: https://site.financialmodelingprep.com/developer/docs#form-13f
        Endpoint: v3/institutional-holder/{symbol}
        """
        # Try multiple endpoint formats
        endpoints_to_try = [
            (f"v3/institutional-holder/{symbol}", None),  # Primary endpoint per docs
            ("v3/institutional-holder", {"symbol": symbol}),
            ("institutional-holder", {"symbol": symbol}),
        ]
        
        for endpoint, params in endpoints_to_try:
            try:
                data = self._get(endpoint, params)
                if data:
                    df = pd.DataFrame(data)
                    if not df.empty:
                        if 'dateReported' in df.columns:
                            df['dateReported'] = pd.to_datetime(df['dateReported'], errors='coerce')
                        return df
            except Exception as e:
                logger.debug(f"FMPClient.get_institutional_holders: Attempt with {endpoint} failed for {symbol}: {e}")
                continue
        
        return pd.DataFrame()
    
    def get_press_releases(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get press releases for a company."""
        data = self._get(f"v3/press-releases/{symbol}", {'limit': limit})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date', ascending=False)
        
        return df
    
    def get_institutional_holders_historical(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """Get historical institutional holdings."""
        # Note: FMP may have different endpoint for historical
        # This is a placeholder - adjust based on actual API
        data = self._get(f"v3/institutional-holder/{symbol}")
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'dateReported' in df.columns:
            df['dateReported'] = pd.to_datetime(df['dateReported'])
            df = df.sort_values('dateReported', ascending=False).head(limit)
        
        return df
    
    # Company outlook (for sentiment/news)
    def get_company_outlook(self, symbol: str) -> Optional[Dict]:
        """
        Get company outlook including news.
        
        Reference: https://site.financialmodelingprep.com/developer/docs#news
        Endpoint: v4/company-outlook
        """
        # Try v4 format (may be available in some plans)
        data = self._get("v4/company-outlook", params={"symbol": symbol})
        return data if data else None
    
    # Index constituents
    def get_index_constituents(self, index_symbol: str) -> List[str]:
        """Get list of tickers in an index."""
        # Use stable API endpoints as per FMP documentation
        index_map = {
            '^GSPC': ['sp500-constituent'],
        }
        
        endpoints = index_map.get(index_symbol, [])
        for endpoint in endpoints:
            try:
                data = self._get(endpoint)
                if data:
                    symbols = []
                    for item in data:
                        if isinstance(item, dict):
                            symbol = item.get('symbol') or item.get('ticker') or item.get('companySymbol')
                            if symbol:
                                symbols.append(symbol.upper())
                        elif isinstance(item, str):
                            symbols.append(item.upper())
                    return symbols
            except requests.exceptions.HTTPError as e:
                # If 402, this endpoint requires subscription - skip to next
                if e.response.status_code == 402:
                    continue
                # For other errors, continue to next endpoint
                continue
            except:
                continue
        
        return []
    
    def get_economic_indicators(self, name: str, from_date: Optional[str] = None, to_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get economic indicators.
        
        Args:
            name: Indicator name (GDP, realGDP, nominalPotentialGDP, etc.)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with economic indicator data
        """
        params = {'name': name}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        data = self._get("economic-indicators", params)
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index(ascending=False)
        
        return df
    
    def get_market_risk_premium(self) -> pd.DataFrame:
        """
        Get market risk premium data.
        
        Returns:
            DataFrame with market risk premium data
        """
        data = self._get("market-risk-premium")
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    
    def close(self):
        """Close the session."""
        self.session.close()

