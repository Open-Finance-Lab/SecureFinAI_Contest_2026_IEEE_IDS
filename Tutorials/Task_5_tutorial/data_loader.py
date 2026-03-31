"""
Task 5 Data Loader - Alpaca Crypto + yfinance

Loads market data for ETH and MSFT.
"""

def load_market_data(asset="ETH", start_date="2025-08-01", end_date=None):
    """
    Load market data for crypto (Alpaca) or stocks (yfinance).
    """
    # All Alpaca-supported crypto symbols (20 coins)
    crypto_map = {
        "BTC": "BTC/USD",
        "ETH": "ETH/USD",
        "SOL": "SOL/USD",
        "XRP": "XRP/USD",
        "DOGE": "DOGE/USD",
        "ADA": "ADA/USD",
        "AVAX": "AVAX/USD",
        "LINK": "LINK/USD",
        "DOT": "DOT/USD",
        "UNI": "UNI/USD",
        "LTC": "LTC/USD",
        "BCH": "BCH/USD",
        "AAVE": "AAVE/USD",
        "YFI": "YFI/USD",
        "CRV": "CRV/USD",
        "XTZ": "XTZ/USD",
        "GRT": "GRT/USD",
        "BAT": "BAT/USD",
        "SUSHI": "SUSHI/USD",
        "SHIB": "SHIB/USD",
    }
    
    if asset in crypto_map:
        return load_alpaca_crypto(crypto_map[asset], start_date, end_date)
    else:
        # For non-crypto, try yfinance
        return load_yfinance(asset, start_date, end_date)


def load_alpaca_crypto(symbol="ETH/USD", start_date="2025-08-01", end_date=None):
    """Load crypto data from Alpaca"""
    import pandas as pd
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        API_KEY = "PKS7VQVXN4KGDCOEO656XK7QYY"
        SECRET_KEY = "HJW9fPTNcjDavkBAWiBXETp5ouMTd16S3vqWhydieZ36"
        
        client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
        
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = client.get_crypto_bars(request)
        df = bars.df
        
        # Flatten multi-index if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Rename columns to match expected format (Alpaca uses lowercase)
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        print(f"✅ Loaded {len(df)} bars for {symbol} from Alpaca")
        return df
        
    except Exception as e:
        print(f"❌ Alpaca error: {e}")
        return pd.DataFrame()


def load_yfinance(asset="MSFT", start_date="2025-08-01", end_date=None):
    """Load stock data from yfinance"""
    import yfinance as yf
    import pandas as pd
    
    ticker = f"{asset}-USD" if asset == "BTC" else asset
    print(f"Loading {ticker} from yfinance...")
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        print(f"✅ Loaded {len(data)} rows from yfinance")
        return data
    except Exception as e:
        print(f"yfinance error: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    import pandas as pd
    
    print("="*50)
    print("Testing data loaders...")
    print("="*50)
    
    print("\n--- ETH from Alpaca ---")
    eth = load_market_data("ETH", "2025-08-01")
    print(eth.tail() if len(eth) > 0 else "No data")
    
    print("\n--- MSFT from yfinance ---")
    msft = load_market_data("MSFT", "2025-08-01")
    print(msft.tail() if len(msft) > 0 else "No data")
