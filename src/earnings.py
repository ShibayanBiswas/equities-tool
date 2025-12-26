"""
Earnings data functions.
"""

import pandas as pd
from typing import Optional
from src.fmp_client import FMPClient


def get_earnings_surprises(symbol: str, limit: int = 20,
                           client: Optional[FMPClient] = None) -> pd.DataFrame:
    """
    Get earnings surprises for a stock.
    
    Args:
        symbol: Stock symbol
        limit: Number of quarters to fetch
        client: FMP client instance
    
    Returns:
        DataFrame with earnings surprises
    """
    if client is None:
        client = FMPClient()
        should_close = True
    else:
        should_close = False
    
    try:
        df = client.get_earnings_surprises(symbol, limit)
        return df
    finally:
        if should_close:
            client.close()

