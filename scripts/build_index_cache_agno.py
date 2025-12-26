"""
Automated Index Cache Builder - Agno Framework Version

ARCHITECTURE:
-----------
This orchestrator coordinates between two distinct layers:

1. scripts/agents/ (AI Agent Layer - Orchestration Only):
   - Discovers indexes
   - Fetches index constituents
   - Validates tickers
   - Segregates by sector/industry
   - Gathers market intelligence
   - NO analytics - only data gathering/orchestration

2. src/ (Analytics Layer):
   - All financial data enrichment (enrich_ticker_data)
   - All calculations (valuation, scoring, etc.)
   - All analytics functions (DCF, sentiment, ratios, etc.)
   - Used by both orchestrator and Streamlit app

Uses Agno Team and Agent framework for agentic orchestration:
- Agno Agent: Individual specialized agents
- Agno Team: Coordinates multiple agents
- Agno Tools: Structured tool interfaces
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import time
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('index_cache_builder.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import enrich_ticker_data
from scripts.agents.agno_agents import (
    create_index_cache_team,
    get_index_constituents,
    get_index_constituents_direct,
    validate_tickers_batch,
    validate_tickers_batch_direct,
    segregate_by_sector,
    segregate_by_sector_direct,
    exa_tool,
    create_market_intelligence_agent
)
from scripts.agents.config import GEMINI_API_KEY, EXA_API_KEY


class IndexCacheOrchestratorAgno:
    """Orchestrates index cache building using Agno framework."""
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "data/index_cache"):
        """
        Initialize orchestrator.
        
        Args:
            api_key: Deprecated parameter (kept for compatibility, ignored)
            output_dir: Output directory for cached data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Agno team
        self.team = create_index_cache_team()
        print("[OK] Agno Team initialized")
    
    def discover_indexes(self) -> List[Dict[str, str]]:
        """Discover available indexes using FMP API with web scraping fallback."""
        discovered = []
        
        # Index definitions
        index_definitions = {
            '^GSPC': 'S&P 500',
        }
        
        print("\n[INFO] Discovering indexes...")
        print("  [NOTE] Using FMP API with web scraping fallback\n")
        
        for index_symbol, name in index_definitions.items():
            print(f"  Checking {name} ({index_symbol})...", end=" ")
            
            # Use the get_index_constituents function which handles FMP -> web scraping fallback
            try:
                constituents_result = get_index_constituents_direct(index_symbol)
                
                if constituents_result.get('error'):
                    error_msg = constituents_result.get('error', 'Unknown error')
                    print(f"    [ERROR] {error_msg}")
                elif constituents_result.get('constituents'):
                    symbols = constituents_result['constituents']
                    source = constituents_result.get('source', 'unknown')
                    discovered.append({
                        'symbol': index_symbol,
                        'name': name,
                        'constituent_count': len(symbols),
                        'endpoint': source,
                        'source': source
                    })
                    print(f"[OK] Found {len(symbols)} constituents (source: {source})")
                else:
                    print(f"[ERROR] No constituents found")
            except Exception as e:
                print(f"[ERROR] Exception: {str(e)[:100]}")
        
        return discovered
    
    def _check_cache_exists(self, index_symbol: str) -> bool:
        """Check if cache already exists for this index."""
        index_slug = index_symbol.replace('^', '').lower()
        index_dir = self.output_dir / index_slug
        
        # Check if main cache files exist
        meta_file = index_dir / 'meta.json'
        sector_file = index_dir / 'sector_groups.json'
        industry_file = index_dir / 'industry_groups.json'
        
        if meta_file.exists() and sector_file.exists() and industry_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    print(f"[INFO] Cache already exists for {index_symbol}")
                    print(f"       Built: {meta.get('build_date', 'Unknown')}")
                    print(f"       Constituents: {meta.get('constituent_count', 0)}")
                    print(f"       Coverage: {meta.get('coverage', 0):.2%}")
                    return True
            except Exception as e:
                logger.debug(f"Error reading cache metadata: {e}")
                return False
        return False
    
    def _convert_dataframe_to_dict(self, df):
        """Convert pandas DataFrame to JSON-serializable dict."""
        if df is None:
            return None
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return None
            # Convert DataFrame to dict with records orientation
            return df.to_dict(orient='records')
        return df
    
    def _serialize_enriched_data(self, enriched_data: List[Dict]) -> List[Dict]:
        """Convert enriched data to JSON-serializable format (handle DataFrames)."""
        logger.debug(f"_serialize_enriched_data: Input length: {len(enriched_data)}")
        serialized = []
        for idx, item in enumerate(enriched_data):
            if idx < 3:  # Log first 3 items for debugging
                logger.debug(f"_serialize_enriched_data: Item {idx} type: {type(item)}, keys: {list(item.keys()) if isinstance(item, dict) else 'N/A'}")
            serialized_item = {}
            for key, value in item.items():
                if isinstance(value, pd.DataFrame):
                    # Convert DataFrame to list of dicts
                    if not value.empty:
                        serialized_item[key] = value.to_dict(orient='records')
                    else:
                        serialized_item[key] = None
                elif pd.api.types.is_datetime64_any_dtype(type(value)):
                    # Handle datetime objects
                    serialized_item[key] = str(value)
                elif isinstance(value, (pd.Series, pd.Index)):
                    # Convert Series/Index to list
                    serialized_item[key] = value.tolist()
                else:
                    # Keep as-is (should be JSON-serializable)
                    serialized_item[key] = value
            serialized.append(serialized_item)
        return serialized
    
    def _save_incremental(self, index_symbol: str, index_name: str, 
                         validated_tickers: List[str],
                         sector_groups: Dict[str, List[str]],
                         industry_groups: Dict[str, List[str]],
                         enriched_data: List[Dict],
                         coverage: float,
                         market_intelligence: Dict,
                         partial: bool = False):
        """Save cache incrementally (can be called multiple times during building)."""
        try:
            index_slug = index_symbol.replace('^', '').lower()
            index_dir = self.output_dir / index_slug
            index_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metadata (update if exists)
            meta = {
                'index_symbol': index_symbol,
                'index_name': index_name,
                'constituent_count': len(validated_tickers),
                'coverage': coverage,
                'build_date': datetime.now().isoformat(),
                'sector_count': len(sector_groups),
                'industry_count': len(industry_groups),
                'market_intelligence': market_intelligence,
                'partial': partial,  # Indicate if this is a partial save
                'enriched_count': len(enriched_data),
            }
            
            with open(index_dir / 'meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Save sector and industry groups (overwrite if exists)
            with open(index_dir / 'sector_groups.json', 'w') as f:
                json.dump(sector_groups, f, indent=2)
            
            with open(index_dir / 'industry_groups.json', 'w') as f:
                json.dump(industry_groups, f, indent=2)
            
            # Save enriched data incrementally (append mode)
            enriched_file = index_dir / 'enriched_data.json'
            
            # For partial saves, always merge with existing data to avoid losing progress
            if enriched_file.exists():
                # Load existing data with better error handling
                existing_data = []
                try:
                    with open(enriched_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # Only try to parse if file is not empty
                            existing_data = json.loads(content)
                    if not isinstance(existing_data, list):
                        logger.warning(f"_save_incremental: Existing data is not a list, resetting")
                        existing_data = []
                    logger.debug(f"_save_incremental: Loaded {len(existing_data)} existing enriched entries")
                except json.JSONDecodeError as e:
                    # Corrupted JSON - backup and start fresh
                    logger.warning(f"_save_incremental: Corrupted JSON file (line {e.lineno}, col {e.colno}): {e.msg}")
                    backup_file = enriched_file.with_suffix('.json.backup')
                    try:
                        import shutil
                        shutil.copy2(enriched_file, backup_file)
                        logger.info(f"_save_incremental: Backed up corrupted file to {backup_file}")
                    except Exception as backup_error:
                        logger.warning(f"_save_incremental: Failed to backup corrupted file: {backup_error}")
                    existing_data = []
                except Exception as e:
                    logger.warning(f"_save_incremental: Error loading existing enriched_data: {type(e).__name__}: {e}")
                    existing_data = []
            else:
                existing_data = []
            
            # Serialize new enriched data (convert DataFrames to dicts)
            logger.debug(f"_save_incremental: Input enriched_data length: {len(enriched_data)}")
            logger.debug(f"_save_incremental: Input enriched_data types: {[type(item) for item in enriched_data[:3]] if len(enriched_data) > 0 else 'empty'}")
            
            serialized_new_data = self._serialize_enriched_data(enriched_data)
            logger.debug(f"_save_incremental: Serialized new_data length: {len(serialized_new_data)}")
            logger.debug(f"_save_incremental: First serialized item keys: {list(serialized_new_data[0].keys()) if len(serialized_new_data) > 0 else 'N/A'}")
            
            # Merge with new data (avoid duplicates)
            existing_symbols = {item.get('symbol') for item in existing_data if 'symbol' in item}
            logger.debug(f"_save_incremental: Existing symbols count: {len(existing_symbols)}")
            logger.debug(f"_save_incremental: Existing symbols sample: {list(existing_symbols)[:5] if existing_symbols else 'none'}")
            
            new_data = [item for item in serialized_new_data if item.get('symbol') not in existing_symbols]
            logger.debug(f"_save_incremental: New data after deduplication: {len(new_data)} items")
            if len(new_data) > 0:
                logger.debug(f"_save_incremental: New data symbols: {[item.get('symbol') for item in new_data[:5]]}")
            
            merged_data = existing_data + new_data
            logger.info(f"_save_incremental: Merging {len(existing_data)} existing + {len(new_data)} new = {len(merged_data)} total")
            
            # Save merged data
            logger.debug(f"_save_incremental: About to write to file: {enriched_file}")
            logger.debug(f"_save_incremental: Merged data length: {len(merged_data)}, type: {type(merged_data)}")
            try:
                with open(enriched_file, 'w', encoding='utf-8') as f:
                    json.dump(merged_data, f, indent=2, default=str)  # default=str handles any remaining non-serializable types
                logger.info(f"_save_incremental: Successfully saved {len(merged_data)} enriched entries to {enriched_file}")
                # Verify file was written
                if enriched_file.exists():
                    file_size = enriched_file.stat().st_size
                    logger.debug(f"_save_incremental: File written successfully, size: {file_size} bytes")
                else:
                    logger.error(f"_save_incremental: File was not created after write!")
            except TypeError as e:
                # If there are still non-serializable types, try to handle them
                logger.error(f"_save_incremental: JSON serialization error: {e}")
                # Try to save with more aggressive serialization
                def json_serializer(obj):
                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='records') if not obj.empty else None
                    elif isinstance(obj, (pd.Series, pd.Index)):
                        return obj.tolist()
                    elif hasattr(obj, 'isoformat'):  # datetime objects
                        return obj.isoformat()
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                with open(enriched_file, 'w', encoding='utf-8') as f:
                    json.dump(merged_data, f, indent=2, default=json_serializer)
                logger.info(f"_save_incremental: Saved with custom serializer")
            
            if partial:
                print(f"  [SAVED] Incremental save: {len(merged_data)} tickers enriched so far ({len(new_data)} new in this batch)")
                logger.debug(f"_save_incremental: Saved {len(merged_data)} total enriched entries (partial=True)")
            return True
        except Exception as e:
            logger.error(f"Error saving incremental cache: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"_save_incremental traceback: {traceback.format_exc()}")
            return False
    
    def build_index_cache(self, index_symbol: str, index_name: str, min_coverage: float = 0.9, 
                         force_rebuild: bool = False):
        """
        Build cache for a single index using Agno team.
        
        Args:
            index_symbol: Index symbol (e.g., '^GSPC')
            index_name: Index display name
            min_coverage: Minimum coverage threshold
            force_rebuild: If True, rebuild even if cache exists
        """
        print(f"\n{'='*60}")
        print(f"Building cache for {index_name} ({index_symbol})")
        print(f"{'='*60}\n")
        
        # Check if cache already exists
        if not force_rebuild and self._check_cache_exists(index_symbol):
            print("[SKIP] Cache already exists. Use force_rebuild=True to rebuild.")
            return True
        
        try:
            # Step 1: Fetch constituents using Agno agent
            print("[Agno Team] Fetching constituents...")
            constituents_result = get_index_constituents_direct(index_symbol)
            
            if constituents_result.get('error') or not constituents_result.get('constituents'):
                print(f"[ERROR] Failed to fetch constituents: {constituents_result.get('error', 'Unknown error')}")
                return False
            
            raw_constituents = constituents_result['constituents']
            print(f"[OK] Fetched {len(raw_constituents)} constituents")
            
            # Step 2: Validate tickers
            print("\n[Agno Team] Validating tickers...")
            logger.info(f"build_index_cache: Starting validation for {len(raw_constituents)} raw constituents")
            logger.debug(f"build_index_cache: First 10 constituents: {raw_constituents[:10]}")
            
            validation_result = validate_tickers_batch_direct(raw_constituents)
            
            logger.info(f"build_index_cache: Validation result keys: {list(validation_result.keys())}")
            logger.debug(f"build_index_cache: Validation result: {validation_result}")
            
            if validation_result.get('error'):
                error_msg = validation_result['error']
                logger.error(f"build_index_cache: Validation error: {error_msg}")
                print(f"[ERROR] Validation error: {error_msg}")
                return False
            
            validated_tickers = validation_result.get('validated', [])
            coverage = validation_result.get('coverage', 0.0)
            total = validation_result.get('total', len(raw_constituents))
            valid_count = validation_result.get('valid_count', len(validated_tickers))
            
            logger.info(f"build_index_cache: Validation summary - Total: {total}, Validated: {valid_count}, Coverage: {coverage:.2%}")
            logger.debug(f"build_index_cache: First 10 validated tickers: {validated_tickers[:10]}")
            
            print(f"[OK] Validated {len(validated_tickers)}/{len(raw_constituents)} tickers ({coverage:.2%} coverage)")
            
            if coverage < min_coverage:
                print(f"[WARN] Coverage {coverage:.2%} below threshold {min_coverage:.2%}")
                return False
            
            # Step 3: Segregate by sector
            print("\n[Agno Team] Segregating by sector/industry...")
            logger.info(f"build_index_cache: Starting sector/industry segregation for {len(validated_tickers)} tickers")
            logger.debug(f"build_index_cache: First 10 tickers to segregate: {validated_tickers[:10]}")
            
            segregation_result = segregate_by_sector_direct(validated_tickers)
            
            logger.info(f"build_index_cache: Segregation result keys: {list(segregation_result.keys())}")
            
            if segregation_result.get('error'):
                error_msg = segregation_result['error']
                logger.error(f"build_index_cache: Segregation error: {error_msg}")
                print(f"[WARN] Segregation error: {error_msg}")
                sector_groups = {}
                industry_groups = {}
            else:
                sector_groups = segregation_result.get('sector_groups', {})
                industry_groups = segregation_result.get('industry_groups', {})
                sector_count = segregation_result.get('sector_count', len(sector_groups))
                industry_count = segregation_result.get('industry_count', len(industry_groups))
                
                logger.info(f"build_index_cache: Segregation complete - {sector_count} sectors, {industry_count} industries")
                logger.debug(f"build_index_cache: Sector groups: {list(sector_groups.keys())[:10]}...")
                logger.debug(f"build_index_cache: Industry groups: {list(industry_groups.keys())[:10]}...")
                
                # Log distribution details
                if sector_groups:
                    top_sectors = sorted(sector_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                    logger.info(f"build_index_cache: Top 5 sectors by count: {[(s, len(t)) for s, t in top_sectors]}")
                
                print(f"[OK] Segregated into {len(sector_groups)} sectors and {len(industry_groups)} industries")
            
            # Step 4: Gather market intelligence using FMP API (simple, reliable)
            # Initialize market_intelligence before using it
            market_intelligence = {}
            print("\n[Market Intelligence] Gathering index statistics...")
            logger.info(f"build_index_cache: Gathering market intelligence for {index_name}")
            
            try:
                # Use FMP client to gather basic index statistics
                from src.fmp_client import FMPClient
                fmp_client = FMPClient()
                
                # Get sample of top constituents for market intelligence
                sample_size = min(10, len(validated_tickers))
                sample_tickers = validated_tickers[:sample_size] if validated_tickers else []
                
                intelligence_data = {
                    'index_name': index_name,
                    'index_symbol': index_symbol,
                    'total_constituents': len(validated_tickers),
                    'sample_size': sample_size,
                    'sector_distribution': {sector: len(tickers) for sector, tickers in sector_groups.items()},
                    'top_sectors': sorted(sector_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5],
                    'sample_companies': []
                }
                
                # Get basic info for sample companies
                for symbol in sample_tickers:
                    try:
                        profile = fmp_client.get_profile(symbol)
                        quote = fmp_client.get_quote(symbol)
                        if profile:
                            company_info = {
                                'symbol': symbol,
                                'name': profile.get('companyName', ''),
                                'sector': profile.get('sector', ''),
                                'market_cap': profile.get('mktCap') or profile.get('marketCap', 0),
                                'price': quote.get('price') if quote else profile.get('price', 0),
                            }
                            intelligence_data['sample_companies'].append(company_info)
                    except Exception as e:
                        logger.debug(f"build_index_cache: Error getting intelligence for {symbol}: {e}")
                        continue
                
                # Calculate aggregate statistics
                if intelligence_data['sample_companies']:
                    total_market_cap = sum(c.get('market_cap', 0) for c in intelligence_data['sample_companies'])
                    avg_price = sum(c.get('price', 0) for c in intelligence_data['sample_companies']) / len(intelligence_data['sample_companies'])
                    intelligence_data['sample_stats'] = {
                        'total_market_cap': total_market_cap,
                        'average_price': avg_price,
                        'companies_analyzed': len(intelligence_data['sample_companies'])
                    }
                
                market_intelligence = {
                    'source': 'fmp',
                    'method': 'index_statistics',
                    'data': intelligence_data,
                    'generated_at': datetime.now().isoformat()
                }
                
                logger.info(f"build_index_cache: Market intelligence gathered - {len(intelligence_data['sample_companies'])} sample companies analyzed")
                print(f"[OK] Market intelligence gathered ({len(intelligence_data['sample_companies'])} sample companies)")
                
            except Exception as e:
                logger.warning(f"build_index_cache: Market intelligence gathering failed: {type(e).__name__}: {str(e)}")
                # Create minimal intelligence with just sector distribution
                market_intelligence = {
                    'source': 'fmp_fallback',
                    'method': 'sector_distribution_only',
                    'data': {
                        'index_name': index_name,
                        'index_symbol': index_symbol,
                        'total_constituents': len(validated_tickers),
                        'sector_distribution': {sector: len(tickers) for sector, tickers in sector_groups.items()},
                    },
                    'error': str(e),
                    'generated_at': datetime.now().isoformat()
                }
                print(f"[WARN] Market intelligence gathering failed (non-critical): {str(e)[:100]}")
            
            # Save incrementally after segregation and market intelligence
            logger.info(f"build_index_cache: Saving incremental data BEFORE enrichment - enriched_data is empty (as expected)")
            logger.debug(f"build_index_cache: About to save - validated_tickers: {len(validated_tickers)}, sector_groups: {len(sector_groups)}, industry_groups: {len(industry_groups)}")
            save_result = self._save_incremental(
                index_symbol, index_name, validated_tickers,
                sector_groups, industry_groups, [],
                coverage, market_intelligence, partial=True
            )
            logger.info(f"build_index_cache: Pre-enrichment save result: {save_result}")
            
            # Step 5: Enrich with analytics data (using src analytics functions)
            # NOTE: Agents only handle orchestration (fetching, validating, segregating)
            # All analytics are done by src/ functions
            print("\n[Analytics] Enriching tickers using src analytics functions...")
            logger.info(f"build_index_cache: ===== STARTING ENRICHMENT STEP =====")
            logger.info(f"build_index_cache: Starting enrichment for {len(validated_tickers)} validated tickers")
            logger.debug(f"build_index_cache: Validated tickers sample: {validated_tickers[:10] if validated_tickers else 'none'}")
            
            # Initialize enriched_data to ensure it always exists
            enriched_data = []
            logger.debug(f"build_index_cache: Initialized enriched_data as empty list")
            
            if not validated_tickers:
                logger.warning("build_index_cache: No validated tickers to enrich!")
                print("[WARN] No validated tickers to enrich")
            else:
                logger.debug(f"build_index_cache: First 10 tickers to enrich: {validated_tickers[:10]}")
                
                batch_size = 50
                total_batches = (len(validated_tickers) - 1) // batch_size + 1
                logger.info(f"build_index_cache: Processing {total_batches} enrichment batches of size {batch_size}")
                logger.info(f"build_index_cache: Total tickers to process: {len(validated_tickers)}")
                print(f"[INFO] Will process {total_batches} batches of {batch_size} tickers each")
                print(f"[INFO] Total tickers to enrich: {len(validated_tickers)}")
                
                # Force enrichment to run - wrap in try/except but ensure it completes
                logger.info(f"build_index_cache: Entering enrichment loop - starting batch processing")
                try:
                    for i in range(0, len(validated_tickers), batch_size):
                        batch = validated_tickers[i:i+batch_size]
                        batch_num = i // batch_size + 1
                        
                        print(f"  Processing batch {batch_num}/{total_batches}...")
                        logger.info(f"build_index_cache: Enrichment batch {batch_num}/{total_batches} - {len(batch)} tickers")
                        
                        batch_success = 0
                        batch_errors = 0
                        batch_no_quote = 0
                        batch_with_data = 0
                        
                        for idx, symbol in enumerate(batch, 1):
                            try:
                                # Progress indicator
                                if idx % 10 == 0 or idx == 1:
                                    print(f"    Enriching {symbol} ({idx}/{len(batch)} in batch {batch_num})...")
                                
                                logger.debug(f"build_index_cache: Enriching {symbol} ({idx}/{len(batch)} in batch {batch_num})")
                                # Use src analytics function (no FMP client needed)
                                enriched = enrich_ticker_data(symbol)
                                
                                # Always append something - even if it's just the symbol
                                if enriched and isinstance(enriched, dict):
                                    # Ensure symbol is in the enriched data
                                    if 'symbol' not in enriched:
                                        enriched['symbol'] = symbol
                                    enriched_data.append(enriched)
                                    
                                    # Check what data we got
                                    has_profile = 'profile' in enriched and enriched.get('profile')
                                    has_quote = 'quote' in enriched and enriched.get('quote')
                                    has_metrics = 'key_metrics' in enriched and not enriched.get('key_metrics', pd.DataFrame()).empty
                                    has_statements = any(key in enriched for key in ['income_statement', 'balance_sheet', 'cashflow'])
                                    
                                    if has_profile or has_metrics or has_statements:
                                        batch_with_data += 1
                                    
                                    if not has_quote:
                                        batch_no_quote += 1
                                        logger.debug(f"build_index_cache: {symbol} enriched but no quote (has profile: {has_profile}, has metrics: {has_metrics})")
                                    
                                    batch_success += 1
                                    if batch_success % 10 == 0:
                                        logger.info(f"build_index_cache: Batch {batch_num} progress: {batch_success}/{len(batch)} enriched")
                                        print(f"    Progress: {batch_success}/{len(batch)} enriched in batch {batch_num}")
                                else:
                                    # Still add entry even if enrichment returned None/empty
                                    batch_errors += 1
                                    enriched_data.append({'symbol': symbol, 'error': 'enrich_ticker_data returned None/empty', 'enriched': False})
                                    logger.warning(f"build_index_cache: enrich_ticker_data returned None/empty for {symbol}")
                                
                                time.sleep(0.1)  # Rate limiting
                            except Exception as e:
                                batch_errors += 1
                                # Still add entry with error - never skip a ticker
                                error_msg = f"{type(e).__name__}: {str(e)}"
                                enriched_data.append({'symbol': symbol, 'error': error_msg, 'enriched': False})
                                logger.warning(f"build_index_cache: Error enriching {symbol}: {error_msg}")
                                # Don't let one failure stop the process
                                continue
                        
                        logger.info(f"build_index_cache: Batch {batch_num} summary - Success: {batch_success}, Errors: {batch_errors}, No quote: {batch_no_quote}, With data: {batch_with_data}")
                        print(f"  Batch {batch_num} completed: {batch_success} success ({batch_with_data} with data, {batch_no_quote} missing quotes), {batch_errors} errors")
                        
                        # Save incrementally after each batch
                        logger.info(f"build_index_cache: Saving incremental data after batch {batch_num} - {len(enriched_data)} total enriched")
                        logger.debug(f"build_index_cache: Enriched data sample - first item keys: {list(enriched_data[0].keys()) if len(enriched_data) > 0 else 'N/A'}")
                        save_result = self._save_incremental(
                            index_symbol, index_name, validated_tickers,
                            sector_groups, industry_groups, enriched_data,
                            coverage, market_intelligence, partial=True
                        )
                        if not save_result:
                            logger.error(f"build_index_cache: WARNING - Save failed after batch {batch_num}!")
                            print(f"  [ERROR] Failed to save after batch {batch_num}")
                        else:
                            logger.info(f"build_index_cache: Successfully saved after batch {batch_num}")
                    
                    # Calculate enrichment statistics
                    total_enriched = len(enriched_data)
                    with_profile = sum(1 for e in enriched_data if e.get('profile'))
                    with_quote = sum(1 for e in enriched_data if e.get('quote'))
                    with_metrics = sum(1 for e in enriched_data if 'key_metrics' in e and not e.get('key_metrics', pd.DataFrame()).empty)
                    with_errors = sum(1 for e in enriched_data if 'error' in e)
                    
                    logger.info(f"build_index_cache: Enrichment complete - {total_enriched} tickers enriched")
                    logger.info(f"build_index_cache: Enrichment stats - Profile: {with_profile}, Quote: {with_quote}, Metrics: {with_metrics}, Errors: {with_errors}")
                    print(f"[OK] Enriched {total_enriched} tickers using src analytics")
                    print(f"     Stats: {with_profile} with profile, {with_quote} with quote, {with_metrics} with metrics, {with_errors} with errors")
                except Exception as e:
                    logger.error(f"build_index_cache: Critical error during enrichment: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"build_index_cache: Enrichment traceback: {traceback.format_exc()}")
                    print(f"[ERROR] Enrichment failed: {str(e)}")
                    # Try to continue with individual enrichment as last resort
                    if len(enriched_data) == 0 and len(validated_tickers) > 0:
                        logger.warning(f"build_index_cache: Attempting emergency individual enrichment for first 10 tickers")
                        print("[WARN] Attempting emergency individual enrichment...")
                        for symbol in validated_tickers[:10]:
                            try:
                                enriched = enrich_ticker_data(symbol)
                                if enriched:
                                    enriched_data.append(enriched)
                                    logger.info(f"build_index_cache: Emergency enrichment succeeded for {symbol}")
                            except Exception as e2:
                                logger.debug(f"build_index_cache: Emergency enrichment failed for {symbol}: {e2}")
                                enriched_data.append({'symbol': symbol, 'error': str(e2)})
                    logger.warning(f"build_index_cache: Continuing with {len(enriched_data)} enriched tickers despite errors")
            
            # Final check - ensure enriched_data exists
            if not enriched_data and len(validated_tickers) > 0:
                logger.error(f"build_index_cache: CRITICAL - enriched_data is empty but we have {len(validated_tickers)} validated tickers!")
                print(f"[ERROR] Enrichment produced no data! Attempting minimal enrichment...")
                # Last resort: try to enrich at least a few tickers
                for symbol in validated_tickers[:5]:
                    try:
                        enriched = enrich_ticker_data(symbol)
                        if enriched:
                            enriched_data.append(enriched)
                            logger.info(f"build_index_cache: Last resort enrichment succeeded for {symbol}")
                    except Exception as e:
                        logger.debug(f"build_index_cache: Last resort enrichment failed for {symbol}: {e}")
                        enriched_data.append({'symbol': symbol, 'error': str(e)})
            
            # Log final enrichment status
            logger.info(f"build_index_cache: Final enrichment status - {len(enriched_data)} tickers enriched")
            if len(enriched_data) == 0 and len(validated_tickers) > 0:
                logger.error(f"build_index_cache: WARNING - No tickers were enriched despite {len(validated_tickers)} validated tickers!")
                print(f"[WARN] No tickers were enriched! This may indicate an issue with enrich_ticker_data function.")
            
            # Step 6: Build and save cache
            print("\n[Cache Builder] Building cache structure...")
            logger.info(f"build_index_cache: Saving final cache with {len(enriched_data)} enriched tickers")
            success = self._save_cache(
                index_symbol, index_name, validated_tickers,
                sector_groups, industry_groups, enriched_data,
                coverage, market_intelligence
            )
            
            if success:
                logger.info(f"build_index_cache: Cache saved successfully for {index_name}")
                print(f"\n[SUCCESS] Cache built successfully for {index_name}!")
                print(f"         - {len(validated_tickers)} validated tickers")
                print(f"         - {len(enriched_data)} enriched tickers")
                print(f"         - {len(sector_groups)} sectors")
                print(f"         - {len(industry_groups)} industries")
                return True
            else:
                logger.error(f"build_index_cache: Failed to save cache for {index_name}")
                print(f"\n[ERROR] Failed to save cache")
                return False
            
        except Exception as e:
            print(f"\n[ERROR] Error building cache: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # NOTE: _enrich_ticker removed - now using src.data.enrich_ticker_data()
    # This ensures proper separation: agents handle orchestration, src handles analytics
    
    def _build_universe_features(self, tickers_data: List[Dict]) -> pd.DataFrame:
        """Build merged universe features table."""
        rows = []
        
        for ticker_data in tickers_data:
            symbol = ticker_data.get('symbol')
            profile = ticker_data.get('profile', {})
            if not isinstance(profile, dict):
                profile = {}
            quote = ticker_data.get('quote', {})
            if not isinstance(quote, dict):
                quote = {}
            rating = ticker_data.get('rating', {})
            if not isinstance(rating, dict):
                rating = {}
            
            # Handle metrics - can be DataFrame or list of dicts (from JSON cache)
            metrics = ticker_data.get('key_metrics')
            latest_metrics = {}
            if isinstance(metrics, pd.DataFrame) and not metrics.empty:
                latest_metrics = metrics.iloc[0].to_dict() if hasattr(metrics.iloc[0], 'to_dict') else dict(metrics.iloc[0])
            elif isinstance(metrics, list) and len(metrics) > 0:
                # Data loaded from JSON cache will be a list of dicts
                latest_metrics = metrics[0] if isinstance(metrics[0], dict) else {}
            
            # Handle ratios - can be DataFrame or list of dicts (from JSON cache)
            ratios = ticker_data.get('ratios')
            latest_ratios = {}
            if isinstance(ratios, pd.DataFrame) and not ratios.empty:
                latest_ratios = ratios.iloc[0].to_dict() if hasattr(ratios.iloc[0], 'to_dict') else dict(ratios.iloc[0])
            elif isinstance(ratios, list) and len(ratios) > 0:
                # Data loaded from JSON cache will be a list of dicts
                latest_ratios = ratios[0] if isinstance(ratios[0], dict) else {}
            
            # Helper function to get value with multiple possible key names (handles FMP API naming variations)
            def get_ratio_value(ratios_dict, possible_keys):
                """Get value from ratios dict trying multiple possible key names."""
                for key in possible_keys:
                    value = ratios_dict.get(key)
                    if value is not None and not pd.isna(value) if isinstance(value, (int, float)) else value:
                        return value
                return None
            
            def get_metric_value(metrics_dict, possible_keys):
                """Get value from metrics dict trying multiple possible key names."""
                for key in possible_keys:
                    value = metrics_dict.get(key)
                    if value is not None and not pd.isna(value) if isinstance(value, (int, float)) else value:
                        return value
                return None
            
            # Extract ratios with multiple key name variations (FMP API uses different naming)
            # Note: P/E and P/B ratios are in ratios as priceToEarningsRatio and priceToBookRatio
            # Check ratios first (most common location), then metrics
            pe_ratio = get_ratio_value(latest_ratios, ['priceToEarningsRatio', 'priceEarningsRatio', 'peRatio', 'pe', 'Price Earnings Ratio', 'PE Ratio', 'priceToEarnings'])
            if not pe_ratio:
                pe_ratio = get_metric_value(latest_metrics, ['peRatio', 'pe', 'priceEarningsRatio', 'priceToEarningsRatio', 'Price Earnings Ratio', 'PE Ratio', 'priceToEarnings'])
            
            pb_ratio = get_ratio_value(latest_ratios, ['priceToBookRatio', 'priceBookRatio', 'pbRatio', 'pb', 'Price to Book Ratio', 'PB Ratio', 'priceToBook'])
            if not pb_ratio:
                pb_ratio = get_metric_value(latest_metrics, ['pbRatio', 'pb', 'priceToBookRatio', 'Price to Book Ratio', 'PB Ratio', 'priceToBook'])
            
            # ROE and ROA are in key_metrics as returnOnEquity and returnOnAssets
            roe = get_metric_value(latest_metrics, ['returnOnEquity', 'Return on Equity', 'ROE', 'roe'])
            if not roe:
                roe = get_ratio_value(latest_ratios, ['returnOnEquity', 'Return on Equity', 'ROE', 'roe'])
            
            roa = get_metric_value(latest_metrics, ['returnOnAssets', 'Return on Assets', 'ROA', 'roa'])
            if not roa:
                roa = get_ratio_value(latest_ratios, ['returnOnAssets', 'Return on Assets', 'ROA', 'roa'])
            
            # Debt to Equity - check both
            debt_to_equity = get_metric_value(latest_metrics, ['debtToEquity', 'debtEquityRatio', 'Debt to Equity', 'Debt/Equity', 'netDebtToEBITDA'])
            if not debt_to_equity:
                debt_to_equity = get_ratio_value(latest_ratios, ['debtEquityRatio', 'Debt to Equity', 'Debt/Equity', 'debtToEquity'])
            
            # Current Ratio - in key_metrics
            current_ratio = get_metric_value(latest_metrics, ['currentRatio', 'Current Ratio', 'current', 'Current'])
            if not current_ratio:
                current_ratio = get_ratio_value(latest_ratios, ['currentRatio', 'Current Ratio', 'current', 'Current'])
            
            # Dividend Yield - typically in ratios
            dividend_yield = get_ratio_value(latest_ratios, ['dividendYield', 'Dividend Yield', 'dividend', 'Dividend', 'dividendYieldPercentage'])
            if not dividend_yield:
                # Check if it's in profile
                dividend_yield = profile.get('dividendYield') or quote.get('dividendYield')
            
            # Get analyst rating data - extract all rating columns (excluding numeric ones for display)
            rating_data = ticker_data.get('rating', {})
            if not isinstance(rating_data, dict):
                rating_data = {}
            
            # Extract all rating columns (non-numeric for display)
            analyst_rating = rating_data.get('rating') or rating_data.get('ratingRecommendation') or ''
            rating_recommendation = rating_data.get('ratingRecommendation') or ''
            rating_details_dcf_recommendation = rating_data.get('ratingDetailsDCFRecommendation') or ''
            rating_details_roe_recommendation = rating_data.get('ratingDetailsROERecommendation') or ''
            rating_details_roa_recommendation = rating_data.get('ratingDetailsROARecommendation') or ''
            rating_details_de_recommendation = rating_data.get('ratingDetailsDERecommendation') or ''
            rating_details_pe_recommendation = rating_data.get('ratingDetailsPERecommendation') or ''
            rating_details_pb_recommendation = rating_data.get('ratingDetailsPBRecommendation') or ''
            
            # Extract numeric rating scores (for calculation)
            rating_score = rating_data.get('ratingScore') or None
            rating_details_dcf_score = rating_data.get('ratingDetailsDCFScore') or None
            rating_details_roe_score = rating_data.get('ratingDetailsROEScore') or None
            rating_details_roa_score = rating_data.get('ratingDetailsROAScore') or None
            rating_details_de_score = rating_data.get('ratingDetailsDEScore') or None
            rating_details_pe_score = rating_data.get('ratingDetailsPEScore') or None
            rating_details_pb_score = rating_data.get('ratingDetailsPBScore') or None
            
            # Get sentiment data if available
            sentiment_data = ticker_data.get('sentiment', {})
            if not isinstance(sentiment_data, dict):
                sentiment_data = {}
            sentiment_score = sentiment_data.get('avg_compound') or sentiment_data.get('compound') or sentiment_data.get('sentimentScore') or 0
            sentiment_positive = sentiment_data.get('avg_pos') or sentiment_data.get('positive') or 0
            sentiment_negative = sentiment_data.get('avg_neg') or sentiment_data.get('negative') or 0
            sentiment_neutral = sentiment_data.get('avg_neu') or sentiment_data.get('neutral') or 0
            
            # Calculate analyst sentiment metrics
            analyst_estimates_data = ticker_data.get('analyst_estimates', [])
            if isinstance(analyst_estimates_data, pd.DataFrame) and not analyst_estimates_data.empty:
                analyst_estimates_score = len(analyst_estimates_data)  # Count of estimates
            elif isinstance(analyst_estimates_data, list) and len(analyst_estimates_data) > 0:
                analyst_estimates_score = len(analyst_estimates_data)
            else:
                analyst_estimates_score = 0
            
            # Calculate earnings surprises metrics
            earnings_surprises_data = ticker_data.get('earnings_surprises', [])
            earnings_surprises_count = 0
            earnings_positive_surprises = 0
            if isinstance(earnings_surprises_data, pd.DataFrame) and not earnings_surprises_data.empty:
                earnings_surprises_count = len(earnings_surprises_data)
                # Count positive surprises (actual > estimated)
                if 'actualEarningResult' in earnings_surprises_data.columns and 'estimatedEarning' in earnings_surprises_data.columns:
                    earnings_positive_surprises = len(earnings_surprises_data[
                        earnings_surprises_data['actualEarningResult'] > earnings_surprises_data['estimatedEarning']
                    ])
            elif isinstance(earnings_surprises_data, list) and len(earnings_surprises_data) > 0:
                earnings_surprises_count = len(earnings_surprises_data)
                # Try to count positive surprises from list
                for surprise in earnings_surprises_data:
                    if isinstance(surprise, dict):
                        actual = surprise.get('actualEarningResult', 0) or 0
                        estimated = surprise.get('estimatedEarning', 0) or 0
                        if actual > estimated:
                            earnings_positive_surprises += 1
            
            # Calculate analyst sentiment from analyst grades
            analyst_grades_data = ticker_data.get('analyst_grades', [])
            analyst_buy_count = 0
            analyst_hold_count = 0
            analyst_sell_count = 0
            avg_analyst_sentiment = 0
            analyst_sentiment_count = 0
            
            if isinstance(analyst_grades_data, pd.DataFrame) and not analyst_grades_data.empty:
                # Count by grade
                if 'gradingCompany' in analyst_grades_data.columns:
                    grades = analyst_grades_data['gradingCompany'].str.upper()
                    analyst_buy_count = len(grades[grades.isin(['BUY', 'STRONG BUY', 'OUTPERFORM', 'OVERWEIGHT'])])
                    analyst_hold_count = len(grades[grades.isin(['HOLD', 'NEUTRAL', 'EQUAL-WEIGHT'])])
                    analyst_sell_count = len(grades[grades.isin(['SELL', 'STRONG SELL', 'UNDERPERFORM', 'UNDERWEIGHT'])])
                analyst_sentiment_count = len(analyst_grades_data)
                # Average sentiment (buy=1, hold=0, sell=-1)
                if analyst_sentiment_count > 0:
                    avg_analyst_sentiment = (analyst_buy_count - analyst_sell_count) / analyst_sentiment_count
            elif isinstance(analyst_grades_data, list) and len(analyst_grades_data) > 0:
                for grade_item in analyst_grades_data:
                    if isinstance(grade_item, dict):
                        grade = str(grade_item.get('gradingCompany', '')).upper()
                        if grade in ['BUY', 'STRONG BUY', 'OUTPERFORM', 'OVERWEIGHT']:
                            analyst_buy_count += 1
                        elif grade in ['HOLD', 'NEUTRAL', 'EQUAL-WEIGHT']:
                            analyst_hold_count += 1
                        elif grade in ['SELL', 'STRONG SELL', 'UNDERPERFORM', 'UNDERWEIGHT']:
                            analyst_sell_count += 1
                analyst_sentiment_count = len(analyst_grades_data)
                if analyst_sentiment_count > 0:
                    avg_analyst_sentiment = (analyst_buy_count - analyst_sell_count) / analyst_sentiment_count
            
            # Calculate institutional holdings net change percentage for scoring
            # Use the same calculation as Company Overview.ipynb (1-10)
            institutional_holders_data = ticker_data.get('institutional_holders', [])
            institutional_net_change_pct = 0
            if isinstance(institutional_holders_data, pd.DataFrame) and not institutional_holders_data.empty:
                # Calculate sum of shares and net change (from Company Overview.ipynb cell 1-10)
                total_shares = institutional_holders_data['shares'].sum() if 'shares' in institutional_holders_data.columns else 0
                net_change = institutional_holders_data['change'].sum() if 'change' in institutional_holders_data.columns else 0
                
                # Calculate rate of change as percentage for scoring
                if total_shares > 0 and net_change != 0:
                    institutional_net_change_pct = (net_change / total_shares) * 100  # Percentage
            
            # Calculate growth metrics from financial statements
            income_data = ticker_data.get('income_statement', pd.DataFrame())
            cashflow_data = ticker_data.get('cashflow', pd.DataFrame())
            
            fcf_growth = 0
            net_income_growth = 0
            operating_margin = 0
            avg_shares_dil_growth = 0
            
            if isinstance(income_data, pd.DataFrame) and not income_data.empty:
                # Net Income Growth
                if 'netIncome' in income_data.columns and len(income_data) >= 2:
                    net_income_values = income_data['netIncome'].head(2).values
                    if len(net_income_values) == 2 and net_income_values[1] != 0:
                        net_income_growth = (net_income_values[0] - net_income_values[1]) / abs(net_income_values[1])
                
                # Operating Margin
                if 'operatingIncome' in income_data.columns and 'revenue' in income_data.columns:
                    latest_operating = income_data['operatingIncome'].iloc[0] if len(income_data) > 0 else 0
                    latest_revenue = income_data['revenue'].iloc[0] if len(income_data) > 0 else 0
                    if latest_revenue != 0:
                        operating_margin = latest_operating / latest_revenue
                
                # Average Shares Diluted Growth
                if 'weightedAverageShsOutDil' in income_data.columns and len(income_data) >= 2:
                    shares_values = income_data['weightedAverageShsOutDil'].head(2).values
                    if len(shares_values) == 2 and shares_values[1] != 0:
                        avg_shares_dil_growth = (shares_values[0] - shares_values[1]) / shares_values[1]
            
            if isinstance(cashflow_data, pd.DataFrame) and not cashflow_data.empty:
                # Free Cash Flow Growth
                if 'freeCashFlow' in cashflow_data.columns and len(cashflow_data) >= 2:
                    fcf_values = cashflow_data['freeCashFlow'].head(2).values
                    if len(fcf_values) == 2 and fcf_values[1] != 0:
                        fcf_growth = (fcf_values[0] - fcf_values[1]) / abs(fcf_values[1])
            
            # Get volume and beta from quote/profile
            volume = quote.get('volume', 0) or 0
            beta = profile.get('beta', 0) or quote.get('beta', 0) or 0
            
            row = {
                'symbol': symbol,
                'companyName': profile.get('companyName', ''),
                'sector': profile.get('sector', 'Unknown'),
                'industry': profile.get('industry', 'Unknown'),
                'marketCap': profile.get('mktCap') or profile.get('marketCap') or quote.get('marketCap') or 0,
                'price': quote.get('price') or profile.get('price') or 0,
                'volume': volume,
                'beta': beta,
                # Rating columns (non-numeric for display)
                'rating': analyst_rating,
                'ratingRecommendation': rating_recommendation,
                'ratingDetailsDCFRecommendation': rating_details_dcf_recommendation,
                'ratingDetailsROERecommendation': rating_details_roe_recommendation,
                'ratingDetailsROARecommendation': rating_details_roa_recommendation,
                'ratingDetailsDERecommendation': rating_details_de_recommendation,
                'ratingDetailsPERecommendation': rating_details_pe_recommendation,
                'ratingDetailsPBRecommendation': rating_details_pb_recommendation,
                # Rating scores (numeric for calculation)
                'ratingScore': rating_score or 0,
                'ratingDetailsDCFScore': rating_details_dcf_score or 0,
                'ratingDetailsROEScore': rating_details_roe_score or 0,
                'ratingDetailsROAScore': rating_details_roa_score or 0,
                'ratingDetailsDEScore': rating_details_de_score or 0,
                'ratingDetailsPEScore': rating_details_pe_score or 0,
                'ratingDetailsPBScore': rating_details_pb_score or 0,
                # Sentiment and analyst metrics
                'sentimentScore': sentiment_score,
                'avgAnalystSentiment': avg_analyst_sentiment,
                'analystSentimentCount': analyst_sentiment_count,
                'analystBuyCount': analyst_buy_count,
                'analystHoldCount': analyst_hold_count,
                'analystSellCount': analyst_sell_count,
                'analystEstimatesScore': analyst_estimates_score,
                'earningsSurprisesScore': earnings_surprises_count,
                'earningsPositiveSurprises': earnings_positive_surprises,
                'institutionalNetChangePct': institutional_net_change_pct,
                # Financial ratios (for criteria checking)
                'roe': roe or 0,
                'debtToEquity': debt_to_equity or 0,
                'currentRatio': current_ratio or 0,
                'dividendYield': dividend_yield or 0,
                'fcfGrowth': fcf_growth,
                'netIncomeGrowth': net_income_growth,
                'operatingMargin': operating_margin,
                'avgSharesDilGrowth': avg_shares_dil_growth,
                # Keep these for reference
                'peRatio': pe_ratio,
                'pbRatio': pb_ratio,
                'roa': roa,
                'sentimentPositive': sentiment_positive,
                'sentimentNegative': sentiment_negative,
                'sentimentNeutral': sentiment_neutral,
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _save_cache(self, index_symbol: str, index_name: str,
                   validated_tickers: List[str],
                   sector_groups: Dict[str, List[str]],
                   industry_groups: Dict[str, List[str]],
                   enriched_data: List[Dict],
                   coverage: float,
                   market_intelligence: Dict) -> bool:
        """Save cache to disk."""
        try:
            index_slug = index_symbol.replace('^', '').lower()
            index_dir = self.output_dir / index_slug
            index_dir.mkdir(parents=True, exist_ok=True)
            datasets_dir = index_dir / 'datasets'
            datasets_dir.mkdir(exist_ok=True)
            merged_dir = index_dir / 'merged'
            merged_dir.mkdir(exist_ok=True)
            
            # Save metadata
            meta = {
                'index_symbol': index_symbol,
                'index_name': index_name,
                'constituent_count': len(validated_tickers),
                'coverage': coverage,
                'build_date': datetime.now().isoformat(),
                'sector_count': len(sector_groups),
                'industry_count': len(industry_groups),
                'market_intelligence': market_intelligence,
            }
            
            with open(index_dir / 'meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Save sector and industry groups
            with open(index_dir / 'sector_groups.json', 'w') as f:
                json.dump(sector_groups, f, indent=2)
            
            with open(index_dir / 'industry_groups.json', 'w') as f:
                json.dump(industry_groups, f, indent=2)
            
            # Collect and save datasets
            print("  Collecting datasets...")
            profiles = []
            quotes = []
            ratings = []
            key_metrics_list = []
            ratios_list = []
            income_list = []
            balance_list = []
            cashflow_list = []
            analyst_list = []
            earnings_list = []
            dcf_list = []
            
            for ticker_data in enriched_data:
                symbol = ticker_data['symbol']
                
                if 'profile' in ticker_data:
                    profile = ticker_data['profile'].copy()
                    profile['symbol'] = symbol
                    profiles.append(profile)
                
                if 'quote' in ticker_data:
                    quote = ticker_data['quote'].copy()
                    quote['symbol'] = symbol
                    quotes.append(quote)
                
                if 'rating' in ticker_data:
                    rating = ticker_data['rating'].copy()
                    rating['symbol'] = symbol
                    ratings.append(rating)
                
                if 'key_metrics' in ticker_data:
                    metrics = ticker_data['key_metrics']
                    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
                        metrics = metrics.copy()
                        metrics['symbol'] = symbol
                        key_metrics_list.append(metrics)
                
                if 'ratios' in ticker_data:
                    ratios = ticker_data['ratios']
                    if isinstance(ratios, pd.DataFrame) and not ratios.empty:
                        ratios = ratios.copy()
                        ratios['symbol'] = symbol
                        ratios_list.append(ratios)
                
                if 'income_statement' in ticker_data:
                    income = ticker_data['income_statement']
                    if isinstance(income, pd.DataFrame) and not income.empty:
                        income = income.copy()
                        income['symbol'] = symbol
                        income_list.append(income)
                
                if 'balance_sheet' in ticker_data:
                    balance = ticker_data['balance_sheet']
                    if isinstance(balance, pd.DataFrame) and not balance.empty:
                        balance = balance.copy()
                        balance['symbol'] = symbol
                        balance_list.append(balance)
                
                if 'cashflow' in ticker_data:
                    cashflow = ticker_data['cashflow']
                    if isinstance(cashflow, pd.DataFrame) and not cashflow.empty:
                        cashflow = cashflow.copy()
                        cashflow['symbol'] = symbol
                        cashflow_list.append(cashflow)
                
                if 'analyst_estimates' in ticker_data:
                    analyst = ticker_data['analyst_estimates'].copy()
                    analyst['symbol'] = symbol
                    analyst_list.append(analyst)
                
                if 'earnings_surprises' in ticker_data:
                    earnings = ticker_data['earnings_surprises'].copy()
                    earnings['symbol'] = symbol
                    earnings_list.append(earnings)
                
                if 'dcf' in ticker_data:
                    dcf = ticker_data['dcf'].copy()
                    dcf['symbol'] = symbol
                    dcf_list.append(dcf)
            
            # Save as parquet files
            print("  Saving datasets to disk...")
            
            if profiles:
                df_profiles = pd.DataFrame(profiles)
                df_profiles.to_parquet(datasets_dir / 'profile.parquet', index=False)
            
            if quotes:
                df_quotes = pd.DataFrame(quotes)
                df_quotes.to_parquet(datasets_dir / 'quote.parquet', index=False)
            
            if ratings:
                df_ratings = pd.DataFrame(ratings)
                df_ratings.to_parquet(datasets_dir / 'rating.parquet', index=False)
            
            if key_metrics_list:
                df_metrics = pd.concat(key_metrics_list, ignore_index=True)
                df_metrics.to_parquet(datasets_dir / 'key_metrics.parquet', index=False)
            
            if ratios_list:
                df_ratios = pd.concat(ratios_list, ignore_index=True)
                df_ratios.to_parquet(datasets_dir / 'ratios.parquet', index=False)
            
            if income_list:
                df_income = pd.concat(income_list, ignore_index=True)
                df_income.to_parquet(datasets_dir / 'income_statement_annual.parquet', index=False)
            
            if balance_list:
                df_balance = pd.concat(balance_list, ignore_index=True)
                df_balance.to_parquet(datasets_dir / 'balance_sheet_annual.parquet', index=False)
            
            if cashflow_list:
                df_cashflow = pd.concat(cashflow_list, ignore_index=True)
                df_cashflow.to_parquet(datasets_dir / 'cashflow_annual.parquet', index=False)
            
            if analyst_list:
                df_analyst = pd.concat(analyst_list, ignore_index=True)
                df_analyst.to_parquet(datasets_dir / 'analyst_estimates.parquet', index=False)
            
            if earnings_list:
                df_earnings = pd.concat(earnings_list, ignore_index=True)
                df_earnings.to_parquet(datasets_dir / 'earnings_surprises.parquet', index=False)
            
            if dcf_list:
                df_dcf = pd.concat(dcf_list, ignore_index=True)
                df_dcf.to_parquet(datasets_dir / 'dcf.parquet', index=False)
            
            # Build and save universe features
            universe_features = self._build_universe_features(enriched_data)
            universe_features.to_parquet(merged_dir / 'universe_features.parquet', index=False)
            
            # Save constituents
            constituents_df = pd.DataFrame({'symbol': validated_tickers})
            constituents_df.to_parquet(index_dir / 'constituents.parquet', index=False)
            
            print(f"[OK] Cache saved to {index_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False
    
    def build_all(self, min_coverage: float = 0.9, force_rebuild: bool = False):
        """Build cache for all discovered indexes."""
        print("="*60)
        print("INDEX CACHE BUILDER - Agno Framework")
        print("="*60)
        
        # Discover indexes
        discovered_indexes = self.discover_indexes()
        
        if not discovered_indexes:
            print("[ERROR] No indexes discovered!")
            print("\nTroubleshooting:")
            print("1. Check your FMP_API_KEY in .env file")
            print("2. Verify API key is valid at https://financialmodelingprep.com/developer/docs/")
            print("3. Index constituent endpoints require a subscription upgrade (HTTP 402)")
            print("   Your current plan may not include access to these endpoints.")
            print("\nOptions:")
            print("  a) Upgrade your FMP subscription to include index endpoints")
            print("  b) Use the Stock Detail page for individual stock analysis (this works!)")
            print("  c) Manually create index cache files if you have constituent lists from other sources")
            return
        
        print(f"\n[OK] Discovered {len(discovered_indexes)} indexes\n")
        
        # Build cache for each index
        for index_info in discovered_indexes:
            success = self.build_index_cache(
                index_info['symbol'],
                index_info['name'],
                min_coverage,
                force_rebuild=force_rebuild
            )
            
            if not success:
                print(f"[WARN] Skipping {index_info['name']} due to errors\n")
        
        print("\n" + "="*60)
        print("BUILD COMPLETE")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Build index cache using Agno framework')
    parser.add_argument('--force-rebuild', action='store_true', 
                       help='Force rebuild even if cache already exists')
    parser.add_argument('--out', '--output', dest='output', default='data/index_cache',
                       help='Output directory for cached data')
    parser.add_argument('--min-coverage', type=float, default=0.9,
                       help='Minimum coverage ratio (default: 0.9)')
    parser.add_argument('--api-key', help='Deprecated: API key parameter (ignored, kept for compatibility)')
    
    args = parser.parse_args()
    
    orchestrator = IndexCacheOrchestratorAgno(api_key=args.api_key, output_dir=args.output)
    orchestrator.build_all(min_coverage=args.min_coverage, force_rebuild=args.force_rebuild)


if __name__ == '__main__':
    main()

