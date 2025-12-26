"""
Index management and data loading.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from src.fmp_client import FMPClient
from src.utils import load_index_cache


def get_index_constituents(index_symbol: str, use_cache: bool = True, 
                          cache_dir: str = "data/index_cache") -> List[str]:
    """
    Get list of ticker symbols for an index.
    
    Args:
        index_symbol: Index symbol (e.g., '^GSPC')
        use_cache: Whether to use cached data
        cache_dir: Cache directory
    
    Returns:
        List of ticker symbols
    """
    if use_cache:
        index_slug = index_symbol.replace('^', '').lower()
        cache = load_index_cache(index_slug, cache_dir)
        
        if cache and 'constituents' in cache:
            return cache['constituents']['symbol'].tolist()
    
    # Fallback to live fetch
    client = FMPClient()
    try:
        # Use stable API endpoints as per FMP documentation
        index_map = {
            '^GSPC': 'sp500-constituent',
        }
        
        endpoint = index_map.get(index_symbol)
        if endpoint:
            data = client._get(endpoint)
            if data:
                symbols = []
                for item in data:
                    if isinstance(item, dict):
                        symbol = item.get('symbol') or item.get('ticker')
                        if symbol:
                            symbols.append(symbol)
                    elif isinstance(item, str):
                        symbols.append(item)
                return symbols
    finally:
        client.close()
    
    return []


def get_index_universe_features(index_symbol: str, use_cache: bool = True,
                                cache_dir: str = "data/index_cache") -> Optional[pd.DataFrame]:
    """
    Get universe features DataFrame for an index.
    
    Args:
        index_symbol: Index symbol
        use_cache: Whether to use cached data
        cache_dir: Cache directory
    
    Returns:
        DataFrame with universe features or None
    """
    if use_cache:
        index_slug = index_symbol.replace('^', '').lower()
        cache = load_index_cache(index_slug, cache_dir)
        
        if cache and 'universe_features' in cache:
            return cache['universe_features']
    
    return None


def get_index_sector_groups(index_symbol: str, use_cache: bool = True,
                           cache_dir: str = "data/index_cache") -> Dict[str, List[str]]:
    """
    Get sector groups for an index.
    
    Args:
        index_symbol: Index symbol
        use_cache: Whether to use cached data
        cache_dir: Cache directory
    
    Returns:
        Dict mapping sector -> list of tickers
    """
    if use_cache:
        index_slug = index_symbol.replace('^', '').lower()
        cache = load_index_cache(index_slug, cache_dir)
        
        if cache and 'sector_groups' in cache:
            return cache['sector_groups']
    
    return {}


def get_index_industry_groups(index_symbol: str, use_cache: bool = True,
                              cache_dir: str = "data/index_cache") -> Dict[str, List[str]]:
    """
    Get industry groups for an index.
    
    Args:
        index_symbol: Index symbol
        use_cache: Whether to use cached data
        cache_dir: Cache directory
    
    Returns:
        Dict mapping industry -> list of tickers
    """
    if use_cache:
        index_slug = index_symbol.replace('^', '').lower()
        cache = load_index_cache(index_slug, cache_dir)
        
        if cache and 'industry_groups' in cache:
            return cache['industry_groups']
    
    return {}

