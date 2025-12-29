"""
Analyst estimates and ratings functions.
"""

import pandas as pd
from typing import Optional
from src.fmp_client import FMPClient


def get_analyst_estimates(symbol: str, period: str = 'annual', limit: int = 5,
                          client: Optional[FMPClient] = None) -> pd.DataFrame:
    """
    Get analyst estimates for a stock.
    
    Args:
        symbol: Stock symbol
        period: 'annual' or 'quarter'
        limit: Number of periods to fetch
        client: FMP client instance
    
    Returns:
        DataFrame with analyst estimates
    """
    if client is None:
        client = FMPClient()
        should_close = True
    else:
        should_close = False
    
    try:
        df = client.get_analyst_estimates(symbol, period, limit)
        return df
    finally:
        if should_close:
            client.close()


def get_rating(symbol: str, client: Optional[FMPClient] = None) -> Optional[dict]:
    """
    Get analyst rating for a stock.
    
    Args:
        symbol: Stock symbol
        client: FMP client instance
    
    Returns:
        Dict with rating data or None
    """
    if client is None:
        client = FMPClient()
        should_close = True
    else:
        should_close = False
    
    try:
        rating = client.get_rating(symbol)
        return rating
    finally:
        if should_close:
            client.close()

