import requests
import pandas as pd
import time
import logging
from typing import Optional
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CoinGeckoAPIError(Exception):
    """Custom exception for CoinGecko API errors"""
    pass


def fetch_market_data(
    coin: str = "bitcoin",
    vs_currency: str = "usd",
    days: int = 30,
    max_retries: int = 3,
    retry_delay: int = 5,
    output_file: str = "crypto_prices.csv"
) -> pd.DataFrame:
    """
    Fetch cryptocurrency market data from CoinGecko API with retry logic.
    
    Args:
        coin: Cryptocurrency ID (default: bitcoin)
        vs_currency: Target currency (default: usd)
        days: Number of days of historical data
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        output_file: Output CSV filename
        
    Returns:
        DataFrame with timestamp and price columns
        
    Raises:
        CoinGeckoAPIError: If all retry attempts fail
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "hourly"
    }
    
    logger.info(f"Fetching {days} days of {coin} data from CoinGecko")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response structure
            if "prices" not in data:
                raise CoinGeckoAPIError(f"Invalid response structure: {data}")
            
            prices = data["prices"]
            
            if not prices:
                raise CoinGeckoAPIError("No price data returned")
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Validate data
            if df["price"].isnull().any():
                logger.warning("Found null prices, filling with forward fill")
                df["price"] = df["price"].fillna(method='ffill')
            
            if (df["price"] <= 0).any():
                logger.warning("Found invalid prices (<=0), removing")
                df = df[df["price"] > 0]
            
            # Add metadata columns
            df["coin"] = coin
            df["vs_currency"] = vs_currency
            df["fetched_at"] = datetime.utcnow()
            
            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            
            logger.info(f"Successfully fetched {len(df)} records and saved to {output_file}")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
            
            return df
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = retry_delay * (attempt + 1) * 2
                logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"HTTP error {e.response.status_code}: {e}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt + 1}: {e}")
            
        except (KeyError, ValueError) as e:
            logger.error(f"Data parsing error: {e}")
            raise CoinGeckoAPIError(f"Failed to parse API response: {e}")
        
        # Wait before retry (except on last attempt)
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    # All retries failed
    error_msg = f"Failed to fetch data after {max_retries} attempts"
    logger.error(error_msg)
    raise CoinGeckoAPIError(error_msg)


def fetch_multiple_coins(
    coins: list[str] = ["bitcoin", "ethereum", "binancecoin"],
    vs_currency: str = "usd",
    days: int = 30
) -> pd.DataFrame:
    """
    Fetch data for multiple cryptocurrencies.
    
    Args:
        coins: List of coin IDs
        vs_currency: Target currency
        days: Days of historical data
        
    Returns:
        Combined DataFrame for all coins
    """
    all_data = []
    
    for coin in coins:
        try:
            output_file = f"{coin}_prices.csv"
            df = fetch_market_data(
                coin=coin,
                vs_currency=vs_currency,
                days=days,
                output_file=output_file
            )
            all_data.append(df)
            time.sleep(1.5)  # Rate limiting - CoinGecko free tier
            
        except CoinGeckoAPIError as e:
            logger.error(f"Failed to fetch {coin}: {e}")
            continue
    
    if not all_data:
        raise CoinGeckoAPIError("Failed to fetch data for any coin")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv("all_crypto_prices.csv", index=False)
    
    logger.info(f"Combined data for {len(all_data)} coins saved to all_crypto_prices.csv")
    return combined_df


def get_current_price(coin: str = "bitcoin", vs_currency: str = "usd") -> float:
    """
    Get current price for a cryptocurrency.
    
    Args:
        coin: Cryptocurrency ID
        vs_currency: Target currency
        
    Returns:
        Current price as float
    """
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": coin,
        "vs_currencies": vs_currency
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        price = data[coin][vs_currency]
        logger.info(f"Current {coin} price: ${price:,.2f}")
        return price
        
    except Exception as e:
        logger.error(f"Failed to get current price: {e}")
        raise


def validate_data_file(file_path: str = "crypto_prices.csv") -> bool:
    """
    Validate that the data file exists and contains valid data.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ["timestamp", "price"]
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return False
        
        # Check data types
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["price"] = pd.to_numeric(df["price"])
        
        # Check for sufficient data
        if len(df) < 24:  # At least 24 hours
            logger.error(f"Insufficient data: only {len(df)} records")
            return False
        
        # Check for data gaps
        df = df.sort_values("timestamp")
        time_diffs = df["timestamp"].diff()
        max_gap = time_diffs.max()
        if max_gap > pd.Timedelta(hours=3):
            logger.warning(f"Large time gap detected: {max_gap}")
        
        logger.info(f"Data validation passed: {len(df)} records")
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch crypto market data from CoinGecko")
    parser.add_argument("--coin", default="bitcoin", help="Cryptocurrency ID")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--output", default="crypto_prices.csv", help="Output file")
    parser.add_argument("--multi", action="store_true", help="Fetch multiple coins")
    
    args = parser.parse_args()
    
    try:
        if args.multi:
            df = fetch_multiple_coins(days=args.days)
        else:
            df = fetch_market_data(
                coin=args.coin,
                days=args.days,
                output_file=args.output
            )
        
        # Validate the fetched data
        if validate_data_file(args.output):
            logger.info("✓ Data ingestion completed successfully")
            sys.exit(0)
        else:
            logger.error("✗ Data validation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"✗ Data ingestion failed: {e}")
        sys.exit(1)