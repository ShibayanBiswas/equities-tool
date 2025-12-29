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
        'equal weight': 3.0,
        'sector perform': 3.0,
        'sectorperform': 3.0,
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
    
    for idx, row in df.iterrows():
        # Step 1: Average of 7 rating scores (already out of 5)
        rating_scores = []
        
        # All rating scores in the same group
        for col in ['ratingScore', 'ratingDetailsDCFScore', 'ratingDetailsROEScore', 
                    'ratingDetailsROAScore', 'ratingDetailsDEScore', 'ratingDetailsPEScore', 
                    'ratingDetailsPBScore']:
            val = row.get(col, 0) or 0
            if pd.notna(val) and val > 0:
                try:
                    num_val = float(val)
                    if 0 <= num_val <= 5:
                        rating_scores.append(num_val)
                except (ValueError, TypeError):
                    pass
        
        # Calculate average of rating scores (out of 5)
        if rating_scores:
            avg_rating_score = np.mean(rating_scores)
        else:
            avg_rating_score = 0.0
        
        # Step 2: Score other parameters out of 5
        other_scores = []
        
        # Market Cap: 0 if < 1B, 5 if >= 1B
        market_cap = row.get('marketCap', 0) or 0
        market_cap_score = 5.0 if market_cap >= 1_000_000_000 else 0.0
        other_scores.append(market_cap_score)
        
        # Volume: 0 if < 10000, 5 if >= 10000
        volume = row.get('volume', 0) or 0
        volume_score = 5.0 if volume >= 10000 else 0.0
        other_scores.append(volume_score)
        
        # Beta: 0 if <= 1, 5 if > 1
        beta = row.get('beta', 0) or 0
        beta_score = 5.0 if beta > 1 else 0.0
        other_scores.append(beta_score)
        
        # Dividend Yield: 0 if <= 1%, 5 if > 1%
        dividend_yield = row.get('dividendYield', 0) or 0
        dividend_score = 5.0 if dividend_yield > 0.01 else 0.0
        other_scores.append(dividend_score)
        
        # ROE: 0 if < 0.08, 5 if >= 0.08
        roe = row.get('roe', 0) or 0
        roe_score = 5.0 if roe >= 0.08 else 0.0
        other_scores.append(roe_score)
        
        # Debt to Equity: 0 if > 0.50, 5 if <= 0.50
        debt_eq_ratio = row.get('debtToEquity', 0) or 0
        debt_score = 5.0 if debt_eq_ratio <= 0.50 else 0.0
        other_scores.append(debt_score)
        
        # Current Ratio: 0 if < 1.50, 5 if >= 1.50
        current_ratio = row.get('currentRatio', 0) or 0
        current_ratio_score = 5.0 if current_ratio >= 1.50 else 0.0
        other_scores.append(current_ratio_score)
        
        # FCF Growth: 0 if <= 0, 5 if > 0
        fcf_growth = row.get('fcfGrowth', 0) or 0
        fcf_score = 5.0 if fcf_growth > 0 else 0.0
        other_scores.append(fcf_score)
        
        # Net Income Growth: 0 if <= 0, 5 if > 0
        net_income_growth = row.get('netIncomeGrowth', 0) or 0
        net_income_score = 5.0 if net_income_growth > 0 else 0.0
        other_scores.append(net_income_score)
        
        # Operating Margin: 0 if <= 0, 5 if > 0
        operating_margin = row.get('operatingMargin', 0) or 0
        operating_margin_score = 5.0 if operating_margin > 0 else 0.0
        other_scores.append(operating_margin_score)
        
        # Avg Shares Dil Growth: 0 if > 0, 5 if <= 0
        avg_shares_dil_growth = row.get('avgSharesDilGrowth', 0) or 0
        shares_growth_score = 5.0 if avg_shares_dil_growth <= 0 else 0.0
        other_scores.append(shares_growth_score)
        
        # Analyst Sentiment: 0-5 based on counts of newGrade values
        # Count each grade type from newGrade and calculate weighted average using rating_map scores (lines 27-42)
        analyst_strong_buy_count = row.get('analystStrongBuyCount', 0) or 0
        analyst_buy_count = row.get('analystBuyCountSpecific', 0) or 0
        analyst_outperform_count = row.get('analystOutperformCount', 0) or 0
        analyst_overweight_count = row.get('analystOverweightCount', 0) or 0
        analyst_hold_count = row.get('analystHoldCountSpecific', 0) or 0
        analyst_neutral_count = row.get('analystNeutralCount', 0) or 0
        analyst_equal_weight_count = row.get('analystEqualWeightCount', 0) or 0
        analyst_sector_perform_count = row.get('analystSectorPerformCount', 0) or 0
        analyst_underperform_count = row.get('analystUnderperformCount', 0) or 0
        analyst_underweight_count = row.get('analystUnderweightCount', 0) or 0
        analyst_sell_count = row.get('analystSellCountSpecific', 0) or 0
        analyst_strong_sell_count = row.get('analystStrongSellCount', 0) or 0
        
        # Calculate weighted average using scores from rating_map (lines 27-42)
        total_count = (analyst_strong_buy_count + analyst_buy_count + analyst_outperform_count + 
                      analyst_overweight_count + analyst_hold_count + analyst_neutral_count + 
                      analyst_equal_weight_count + analyst_sector_perform_count + 
                      analyst_underperform_count + analyst_underweight_count + 
                      analyst_sell_count + analyst_strong_sell_count)
        
        if total_count > 0:
            # Weighted average: (count * score) / total_count for each grade type
            weighted_sum = (
                analyst_strong_buy_count * 5.0 +      # Strong Buy: 5.0
                analyst_buy_count * 4.0 +              # Buy: 4.0
                analyst_outperform_count * 4.0 +        # Outperform: 4.0
                analyst_overweight_count * 4.0 +        # Overweight: 4.0
                analyst_hold_count * 3.0 +             # Hold: 3.0
                analyst_neutral_count * 3.0 +          # Neutral: 3.0
                analyst_equal_weight_count * 3.0 +      # Equal Weight: 3.0
                analyst_sector_perform_count * 3.0 +    # Sector Perform: 3.0
                analyst_underperform_count * 2.0 +      # Underperform: 2.0
                analyst_underweight_count * 2.0 +       # Underweight: 2.0
                analyst_sell_count * 1.0 +             # Sell: 1.0
                analyst_strong_sell_count * 0.0        # Strong Sell: 0.0
            )
            analyst_score = weighted_sum / total_count  # Already on 0-5 scale
        else:
            analyst_score = 0.0
        
        other_scores.append(analyst_score)
        
        # Earnings Surprises: 0-5 based on positive ratio
        earnings_positive_surprises = row.get('earningsPositiveSurprises', 0) or 0
        earnings_total_surprises = row.get('earningsSurprisesScore', 0) or 0
        
        if earnings_total_surprises > 0:
            positive_ratio = earnings_positive_surprises / earnings_total_surprises
            earnings_score = positive_ratio * 5.0  # Convert 0-1 to 0-5
        else:
            earnings_score = 0.0
        other_scores.append(earnings_score)
        
        # Institutional Net Change: 0-5 based on net change / total shares ratio
        institutional_net_change_pct = row.get('institutionalNetChangePct', 0) or 0
        if pd.notna(institutional_net_change_pct):
            try:
                # Convert percentage to ratio
                net_change_ratio = float(institutional_net_change_pct) / 100.0
                
                if net_change_ratio >= 0.10:
                    institutional_score = 5.0
                elif net_change_ratio >= 0.05:
                    institutional_score = 4.0
                elif net_change_ratio >= 0.02:
                    institutional_score = 3.0
                elif net_change_ratio > 0:
                    institutional_score = 2.0
                else:
                    institutional_score = 0.0
            except (ValueError, TypeError):
                institutional_score = 0.0
        else:
            institutional_score = 0.0
        other_scores.append(institutional_score)
        
        # Calculate average of other parameters (out of 5)
        if other_scores:
            avg_other_score = np.mean(other_scores)
        else:
            avg_other_score = 0.0
        
        # Step 3: Final aggregate score with 50% weightage to each group
        # 50% weightage to rating scores average, 50% to other parameters average
        final_score = (avg_rating_score * 0.5) + (avg_other_score * 0.5)
        
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
            df['ratingScore'] = df['rating'].apply(convert_rating_to_score)
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
        
        # Normalize ratingScore to 0-5 scale if it's in a different range
        if df['ratingScore'].max() > 5:
            df['ratingScore'] = (df['ratingScore'] / df['ratingScore'].max()) * 5.0
    
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

