"""
Asset Universe Configuration for AlphaTraderLab Step 3.2

This module defines the canonical list of assets (tickers) used for multi-asset
training and evaluation in Step 3.2.

Each asset is tested with the same single-agent PPO setup, and results are
aggregated into a multi-asset comparison table.
"""

# Canonical asset universe for Step 3.2
# These are the default tickers used when running multi-asset experiments
DEFAULT_ASSETS = [
    "BTC-USD",   # Bitcoin (cryptocurrency)
    "SPY",       # S&P 500 ETF (US large-cap stocks)
    "QQQ",       # Nasdaq-100 ETF (US tech stocks)
    "IWM",       # Russell 2000 ETF (US small-cap stocks)
]

# Asset metadata (optional, for documentation and display)
ASSET_METADATA = {
    "BTC-USD": {
        "name": "Bitcoin",
        "type": "Cryptocurrency",
        "description": "Leading cryptocurrency, highly volatile"
    },
    "SPY": {
        "name": "S&P 500 ETF",
        "type": "Equity Index",
        "description": "Tracks the S&P 500 index (US large-cap stocks)"
    },
    "QQQ": {
        "name": "Nasdaq-100 ETF",
        "type": "Equity Index",
        "description": "Tracks the Nasdaq-100 index (US tech-heavy stocks)"
    },
    "IWM": {
        "name": "Russell 2000 ETF",
        "type": "Equity Index",
        "description": "Tracks the Russell 2000 index (US small-cap stocks)"
    },
}


def normalize_ticker_to_slug(ticker: str) -> str:
    """
    Convert a ticker symbol to a filesystem-friendly slug.
    
    Examples:
    - "BTC-USD" -> "btc_usd"
    - "SPY" -> "spy"
    - "^GSPC" -> "gspc"
    
    Parameters:
    -----------
    ticker : str
        Raw ticker symbol.
    
    Returns:
    --------
    slug : str
        Normalized slug for use in filenames.
    """
    # Remove special characters and convert to lowercase
    slug = ticker.lower()
    slug = slug.replace("-", "_")
    slug = slug.replace("^", "")
    slug = slug.replace(".", "_")
    return slug


def get_asset_display_name(ticker: str) -> str:
    """
    Get human-readable display name for a ticker.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol.
    
    Returns:
    --------
    name : str
        Display name (falls back to ticker if not in metadata).
    """
    if ticker in ASSET_METADATA:
        return ASSET_METADATA[ticker]["name"]
    return ticker
