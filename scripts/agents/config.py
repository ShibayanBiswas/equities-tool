"""
Configuration for agentic team - API keys and model settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDsnOc_CS91rMAxYuIyUpBM2iBsr0CD8Xc")
FMP_API_KEY = os.getenv("FMP_API_KEY", "QF6RyyMvY1VWRbOeyNdxkiSaxCun5wZN")
EXA_API_KEY = os.getenv("EXA_API_KEY", "008b5100-269c-47c8-9e1e-0d9cd83f6d7d")

# Model Configuration - Using Gemini (free tier models)
# Note: gemini-1.5-flash is free tier and fastest, gemini-1.5-pro requires paid tier
GEMINI_MODEL = "gemini-1.5-flash"  # Free tier - Fastest model for most operations
GEMINI_MODEL_FAST = "gemini-1.5-flash"  # Free tier - Fastest model for sub-agents
GEMINI_MODEL_PRO = "gemini-1.5-pro"  # Paid tier - Use only for complex operations if needed

# Backward compatibility aliases (deprecated - use GEMINI_* directly)
# These are kept only for any remaining legacy code
OPENAI_API_KEY = GEMINI_API_KEY  # Deprecated: Use GEMINI_API_KEY
OPENAI_MODEL = GEMINI_MODEL  # Deprecated: Use GEMINI_MODEL
OPENAI_MODEL_FAST = GEMINI_MODEL_FAST  # Deprecated: Use GEMINI_MODEL_FAST

# Exa Search Configuration
EXA_INCLUDE_DOMAINS = [
    "cnbc.com",
    "reuters.com",
    "bloomberg.com",
    "finance.yahoo.com",
    "wsj.com",
    "ft.com",
    "seekingalpha.com",
    "thestreet.com",
    "marketwatch.com",
    "fool.com",
    "barrons.com",
    "economist.com",
    "forbes.com",
    "businessinsider.com",
]

# Date range for searches (2 years)
from datetime import datetime, timedelta
today = datetime.now()
two_years_ago = today - timedelta(days=730)
EXA_START_DATE = two_years_ago.strftime("%Y-%m-%d")
EXA_END_DATE = today.strftime("%Y-%m-%d")

# Export for convenience
__all__ = [
    'GEMINI_API_KEY', 'OPENAI_API_KEY',  # OPENAI_API_KEY for backward compatibility
    'FMP_API_KEY', 'EXA_API_KEY',
    'GEMINI_MODEL', 'OPENAI_MODEL',  # OPENAI_MODEL for backward compatibility
    'GEMINI_MODEL_FAST', 'OPENAI_MODEL_FAST',  # OPENAI_MODEL_FAST for backward compatibility
    'EXA_INCLUDE_DOMAINS', 'EXA_START_DATE', 'EXA_END_DATE'
]

