"""
Institutional holdings functions.
"""

import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
from src.fmp_client import FMPClient


def get_institutional_holders(symbol: str,
                              client: Optional[FMPClient] = None) -> pd.DataFrame:
    """
    Get current institutional holders for a stock.
    
    Args:
        symbol: Stock symbol
        client: FMP client instance
    
    Returns:
        DataFrame with institutional holders
    """
    if client is None:
        client = FMPClient()
        should_close = True
    else:
        should_close = False
    
    try:
        df = client.get_institutional_holders(symbol)
        
        if not df.empty:
            # Sort by shares descending
            if 'shares' in df.columns:
                df = df.sort_values('shares', ascending=False)
        
        return df
    finally:
        if should_close:
            client.close()


def get_institutional_holders_historical(symbol: str, years: int = 5,
                                         client: Optional[FMPClient] = None) -> pd.DataFrame:
    """
    Get historical institutional holdings (5 years).
    
    Args:
        symbol: Stock symbol
        years: Number of years of history
        client: FMP client instance
    
    Returns:
        DataFrame with historical holdings
    """
    if client is None:
        client = FMPClient()
        should_close = True
    else:
        should_close = False
    
    try:
        # FMP may have different endpoint for historical
        # For now, use the standard endpoint which may include historical data
        df = client.get_institutional_holders_historical(symbol, limit=years * 4)
        
        if not df.empty and 'dateReported' in df.columns:
            # Filter to last N years
            cutoff_date = datetime.now() - timedelta(days=years * 365)
            df['dateReported'] = pd.to_datetime(df['dateReported'])
            df = df[df['dateReported'] >= cutoff_date]
            df = df.sort_values('dateReported', ascending=False)
        
        return df
    finally:
        if should_close:
            client.close()


def get_institutional_summary(symbol: str,
                              client: Optional[FMPClient] = None) -> dict:
    """
    Get summary statistics for institutional holdings.
    
    Args:
        symbol: Stock symbol
        client: FMP client instance
    
    Returns:
        Dict with summary statistics
    """
    df = get_institutional_holders(symbol, client)
    
    if df.empty:
        return {
            'total_shares': 0,
            'total_investors': 0,
            'net_change': 0,
            'top_holder': None,
            'top_holder_shares': 0
        }
    
    total_shares = df['shares'].sum() if 'shares' in df.columns else 0
    total_investors = len(df)
    net_change = df['change'].sum() if 'change' in df.columns else 0
    
    top_holder = None
    top_holder_shares = 0
    if not df.empty and 'shares' in df.columns:
        top_row = df.iloc[0]
        top_holder = top_row.get('name', 'Unknown')
        top_holder_shares = top_row.get('shares', 0)
    
    return {
        'total_shares': total_shares,
        'total_investors': total_investors,
        'net_change': net_change,
        'top_holder': top_holder,
        'top_holder_shares': top_holder_shares
    }

