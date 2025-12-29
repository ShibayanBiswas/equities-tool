"""
Valuation and DCF functions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from src.fmp_client import FMPClient


def get_dcf(symbol: str, limit: int = 10,
            client: Optional[FMPClient] = None) -> pd.DataFrame:
    """
    Get DCF values for a stock.
    
    Args:
        symbol: Stock symbol
        limit: Number of historical DCF values
        client: FMP client instance
    
    Returns:
        DataFrame with DCF data
    """
    if client is None:
        client = FMPClient()
        should_close = True
    else:
        should_close = False
    
    try:
        df = client.get_dcf(symbol, limit)
        
        if not df.empty and 'dcf' in df.columns and 'Stock Price' in df.columns:
            df['margin_of_safety'] = (df['dcf'] - df['Stock Price']) / df['Stock Price']
        
        return df
    finally:
        if should_close:
            client.close()


def calculate_intrinsic_value_metrics(dcf_df: pd.DataFrame, 
                                     margin_of_safety_threshold: float = 0.30) -> Dict:
    """
    Calculate intrinsic value metrics from DCF data.
    
    Args:
        dcf_df: DataFrame with DCF data
        margin_of_safety_threshold: Threshold for margin of safety
    
    Returns:
        Dict with metrics
    """
    if dcf_df.empty:
        return {
            'latest_dcf': None,
            'latest_price': None,
            'margin_of_safety': None,
            'investment_decision': 'Data unavailable'
        }
    
    latest = dcf_df.iloc[0]
    
    latest_dcf = latest.get('dcf', None)
    latest_price = latest.get('Stock Price', None)
    margin = latest.get('margin_of_safety', None)
    
    if margin is not None and margin >= margin_of_safety_threshold:
        decision = 'Good asset!'
    elif margin is not None:
        decision = 'Insufficient margin, keep searching!'
    else:
        decision = 'Data unavailable, cannot make a decision.'
    
    return {
        'latest_dcf': latest_dcf,
        'latest_price': latest_price,
        'margin_of_safety': margin,
        'investment_decision': decision
    }

