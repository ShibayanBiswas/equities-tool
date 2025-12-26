"""
Stock scoring and ranking functions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal, List


def convert_rating_to_score(rating: str) -> float:
    """
    Convert analyst rating string to numeric score.
    
    Args:
        rating: Rating string (e.g., "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")
    
    Returns:
        Numeric score (higher is better)
    """
    if pd.isna(rating) or not rating:
        return 0.0
    
    rating_lower = str(rating).lower().strip()
    
    # Map ratings to scores (higher is better)
    rating_map = {
        'strong buy': 5.0,
        'strongbuy': 5.0,
        'buy': 4.0,
        'outperform': 4.0,
        'overweight': 4.0,
        'hold': 3.0,
        'neutral': 3.0,
        'equal-weight': 3.0,
        'underperform': 2.0,
        'underweight': 2.0,
        'sell': 1.0,
        'strong sell': 0.0,
        'strongsell': 0.0,
    }
    
    # Try exact match first
    if rating_lower in rating_map:
        return rating_map[rating_lower]
    
    # Try partial match
    for key, value in rating_map.items():
        if key in rating_lower:
            return value
    
    # Default to neutral if unknown
    return 3.0


def get_analyst_rating_score(df: pd.DataFrame) -> pd.Series:
    """
    Get analyst rating score from DataFrame.
    Uses ratingScore if available, otherwise converts rating string.
    
    Args:
        df: DataFrame with 'rating' and/or 'ratingScore' columns
    
    Returns:
        Series with rating scores
    """
    scores = pd.Series(index=df.index, dtype=float)
    
    # First try to use ratingScore if available
    if 'ratingScore' in df.columns:
        scores = df['ratingScore'].fillna(0.0)
        # Normalize ratingScore to 0-5 scale if it's in a different range
        if scores.max() > 5:
            scores = (scores / scores.max()) * 5.0
    elif 'rating' in df.columns:
        # Convert rating strings to scores
        scores = df['rating'].apply(convert_rating_to_score)
    else:
        scores = pd.Series(0.0, index=df.index)
    
    return scores


def normalize_scores(df: pd.DataFrame, columns: List[str], 
                     method: Literal['zscore', 'minmax'] = 'zscore') -> pd.DataFrame:
    """
    Normalize scores in DataFrame.
    
    Args:
        df: DataFrame with scores
        columns: Columns to normalize
        method: Normalization method ('zscore' or 'minmax')
    
    Returns:
        DataFrame with normalized scores
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f'{col}_normalized'] = (df[col] - mean) / std
            else:
                df[f'{col}_normalized'] = 0
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f'{col}_normalized'] = 0
    
    return df


def calculate_composite_score_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate composite score out of 5 based on new criteria.
    
    Args:
        df: DataFrame with stock data including all required columns
        
    Returns:
        DataFrame with composite_score column (0-5 scale)
    """
    df = df.copy()
    
    # Initialize composite score
    composite_scores = []
    
    # Count total parameters for equal weightage
    # Parameters: ratingScore (1) + 6 other rating scores + 4 criteria + 6 financial ratios + 1 analyst sentiment + 1 earnings = 20 parameters
    total_params = 20
    weight_per_param = 5.0 / total_params  # Equal weight per parameter (0.25 points each)
    
    for idx, row in df.iterrows():
        score = 0.0
        max_score = 0.0
        
        # Step 1: Rating scores (each scored 0-5, equal weight)
        # ratingScore
        rating_score = row.get('ratingScore', 0) or 0
        if pd.notna(rating_score) and rating_score > 0:
            try:
                rating_score_val = float(rating_score)
                if 0 <= rating_score_val <= 5:
                    # Each parameter gets equal weight: score (0-5) * (weight_per_param / 5.0)
                    score += rating_score_val * (weight_per_param / 5.0)
            except (ValueError, TypeError):
                pass
        max_score += weight_per_param
        
        # Other rating scores (each scored 0-5, equal weight)
        for col in ['ratingDetailsDCFScore', 'ratingDetailsROEScore', 
                    'ratingDetailsROAScore', 'ratingDetailsDEScore', 'ratingDetailsPEScore', 
                    'ratingDetailsPBScore']:
            val = row.get(col, 0) or 0
            if pd.notna(val) and val > 0:
                try:
                    num_val = float(val)
                    if 0 <= num_val <= 5:
                        score += num_val * (weight_per_param / 5.0)
                except (ValueError, TypeError):
                    pass
            max_score += weight_per_param
        
        # Step 2: Market cap, volume, beta, dividend criteria (each scored 0-5, equal weight)
        # Market Cap: 0 if < 1B, 5 if >= 1B
        market_cap = row.get('marketCap', 0) or 0
        market_cap_score = 5.0 if market_cap >= 1_000_000_000 else 0.0
        score += market_cap_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Volume: 0 if < 10000, 5 if >= 10000
        volume = row.get('volume', 0) or 0
        volume_score = 5.0 if volume >= 10000 else 0.0
        score += volume_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Beta: 0 if <= 1, 5 if > 1
        beta = row.get('beta', 0) or 0
        beta_score = 5.0 if beta > 1 else 0.0
        score += beta_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Dividend Yield: 0 if <= 1%, 5 if > 1%
        dividend_yield = row.get('dividendYield', 0) or 0
        dividend_score = 5.0 if dividend_yield > 0.01 else 0.0
        score += dividend_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Step 3: Financial ratios criteria (each scored 0-5, equal weight)
        # ROE: 0 if < 0.08, 5 if >= 0.08
        roe = row.get('roe', 0) or 0
        roe_score = 5.0 if roe >= 0.08 else 0.0
        score += roe_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Debt to Equity: 0 if > 0.50, 5 if <= 0.50
        debt_eq_ratio = row.get('debtToEquity', 0) or 0
        debt_score = 5.0 if debt_eq_ratio <= 0.50 else 0.0
        score += debt_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Current Ratio: 0 if < 1.50, 5 if >= 1.50
        current_ratio = row.get('currentRatio', 0) or 0
        current_ratio_score = 5.0 if current_ratio >= 1.50 else 0.0
        score += current_ratio_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # FCF Growth: 0 if <= 0, 5 if > 0
        fcf_growth = row.get('fcfGrowth', 0) or 0
        fcf_score = 5.0 if fcf_growth > 0 else 0.0
        score += fcf_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Net Income Growth: 0 if <= 0, 5 if > 0
        net_income_growth = row.get('netIncomeGrowth', 0) or 0
        net_income_score = 5.0 if net_income_growth > 0 else 0.0
        score += net_income_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Operating Margin: 0 if <= 0, 5 if > 0
        operating_margin = row.get('operatingMargin', 0) or 0
        operating_margin_score = 5.0 if operating_margin > 0 else 0.0
        score += operating_margin_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Avg Shares Dil Growth: 0 if > 0, 5 if <= 0
        avg_shares_dil_growth = row.get('avgSharesDilGrowth', 0) or 0
        shares_growth_score = 5.0 if avg_shares_dil_growth <= 0 else 0.0
        score += shares_growth_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Step 4: Analyst Sentiment (scored 0-5, equal weight)
        buy_count = row.get('analystBuyCount', 0) or 0
        hold_count = row.get('analystHoldCount', 0) or 0
        sell_count = row.get('analystSellCount', 0) or 0
        total_analyst_count = buy_count + hold_count + sell_count
        
        if total_analyst_count > 0:
            buy_ratio = buy_count / total_analyst_count
            analyst_score = buy_ratio * 5.0  # Convert 0-1 to 0-5
        else:
            analyst_score = 0.0
        score += analyst_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Step 5: Earnings Surprises (scored 0-5, equal weight)
        earnings_positive_surprises = row.get('earningsPositiveSurprises', 0) or 0
        earnings_total_surprises = row.get('earningsSurprisesScore', 0) or 0
        
        if earnings_total_surprises > 0:
            positive_ratio = earnings_positive_surprises / earnings_total_surprises
            earnings_score = positive_ratio * 5.0  # Convert 0-1 to 0-5
        else:
            earnings_score = 0.0
        score += earnings_score * (weight_per_param / 5.0)
        max_score += weight_per_param
        
        # Final score is already on 0-5 scale (sum of all weighted parameters)
        final_score = score
        composite_scores.append(final_score)
    
    df['composite_score'] = composite_scores
    return df


def calculate_composite_score(df: pd.DataFrame, 
                              normalize_within_sector: bool = True,
                              sector_col: str = 'sector') -> pd.DataFrame:
    """
    Calculate composite score for ranking stocks.
    
    Args:
        df: DataFrame with stock data
        normalize_within_sector: If True, normalize within sector; else normalize across all
        sector_col: Column name for sector
    
    Returns:
        DataFrame with composite_score column
    """
    df = df.copy()
    
    # Get analyst rating scores (convert rating strings to numeric if needed)
    # Priority: ratingScore (numeric) > rating (string conversion)
    if 'ratingScore' not in df.columns:
        # No ratingScore column, try to create from rating string
        if 'rating' in df.columns:
            df['ratingScore'] = get_analyst_rating_score(df)
        else:
            df['ratingScore'] = 0.0
    else:
        # ratingScore column exists, but may have NaN values
        # Fill NaN values by converting rating strings if available
        if 'rating' in df.columns:
            missing_mask = df['ratingScore'].isna()
            if missing_mask.any():
                # Convert rating strings for missing values
                converted_scores = df.loc[missing_mask, 'rating'].apply(convert_rating_to_score)
                df.loc[missing_mask, 'ratingScore'] = converted_scores
        
        # Fill any remaining NaN with 0 (neutral)
        df['ratingScore'] = df['ratingScore'].fillna(0.0)
    
    # Define scoring columns (higher is better)
    # Only sentiment, analyst, and earnings-based metrics
    score_columns = {
        'ratingScore': 1.5,  # FMP Rating Calculation
        'sentimentScore': 1.0,  # Sentiment Scores for Articles
        'avgAnalystSentiment': 1.0,  # Average Analyst Sentiment
        'analystSentimentCount': 1.0,  # Count Recent Analyst Sentiment
        'analystEstimatesScore': 1.0,  # Analyst Estimates
        'earningsSurprisesScore': 1.0,  # Earnings Surprises
        'earningsPositiveSurprises': 1.0,  # Earnings Surprises: Actual > Estimated
    }
    
    # Filter to available columns
    available_cols = {k: v for k, v in score_columns.items() if k in df.columns}
    
    if not available_cols:
        df['composite_score'] = 0
        return df
    
    # Normalize scores
    if normalize_within_sector and sector_col in df.columns:
        # Normalize within each sector
        normalized_dfs = []
        for sector in df[sector_col].unique():
            sector_df = df[df[sector_col] == sector].copy()
            for col in available_cols.keys():
                if col in sector_df.columns:
                    mean = sector_df[col].mean()
                    std = sector_df[col].std()
                    if std > 0:
                        sector_df[f'{col}_norm'] = (sector_df[col] - mean) / std
                    else:
                        sector_df[f'{col}_norm'] = 0
            normalized_dfs.append(sector_df)
        df_normalized = pd.concat(normalized_dfs, ignore_index=True)
    else:
        # Normalize across all stocks
        df_normalized = df.copy()
        for col in available_cols.keys():
            if col in df_normalized.columns:
                mean = df_normalized[col].mean()
                std = df_normalized[col].std()
                if std > 0:
                    df_normalized[f'{col}_norm'] = (df_normalized[col] - mean) / std
                else:
                    df_normalized[f'{col}_norm'] = 0
    
    # Calculate composite score
    composite_scores = []
    for idx, row in df_normalized.iterrows():
        score = 0
        weight_sum = 0
        
        for col, direction in available_cols.items():
            norm_col = f'{col}_norm'
            if norm_col in df_normalized.columns and pd.notna(row[norm_col]):
                # Apply direction (multiply by -1 if lower is better)
                score += row[norm_col] * direction
                weight_sum += abs(direction)
        
        composite_scores.append(score / weight_sum if weight_sum > 0 else 0)
    
    df_normalized['composite_score'] = composite_scores
    
    # Merge back to original df
    if 'composite_score' not in df.columns:
        df = df.merge(
            df_normalized[['symbol', 'composite_score']],
            on='symbol',
            how='left'
        )
        df['composite_score'] = df['composite_score'].fillna(0)
    else:
        df['composite_score'] = df_normalized['composite_score'].values
    
    return df


def rank_stocks(df: pd.DataFrame, sector: Optional[str] = None,
                normalize_within_sector: bool = True,
                top_n: int = 50) -> pd.DataFrame:
    """
    Rank stocks by composite score.
    
    Args:
        df: DataFrame with stock data
        sector: If provided, filter to this sector
        normalize_within_sector: Normalization method
        top_n: Number of top stocks to return
    
    Returns:
        Ranked DataFrame
    """
    # Filter by sector if provided
    if sector and 'sector' in df.columns:
        df = df[df['sector'] == sector].copy()
    
    # Calculate composite score using new algorithm
    if 'ratingDetailsDCFScore' in df.columns or 'analystBuyCount' in df.columns:
        # Use new scoring algorithm if new columns are present
        df = calculate_composite_score_v2(df)
    else:
        # Fallback to old algorithm
        df = calculate_composite_score(df, normalize_within_sector)
    
    # Sort by composite score
    df = df.sort_values('composite_score', ascending=False)
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    return df.head(top_n)

