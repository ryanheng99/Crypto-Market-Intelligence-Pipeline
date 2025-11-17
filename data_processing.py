import pandas as pd
import numpy as np
import logging
import sys
from typing import Tuple
from scipy import stats

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


class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass


def load_data(file_path: str = "crypto_prices.csv") -> pd.DataFrame:
    """
    Load data from CSV with validation.
    
    Args:
        file_path: Path to input CSV
        
    Returns:
        Loaded DataFrame
        
    Raises:
        DataProcessingError: If file cannot be loaded or validated
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate columns
        if "timestamp" not in df.columns or "price" not in df.columns:
            raise DataProcessingError(f"Missing required columns. Found: {df.columns.tolist()}")
        
        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Validate prices
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        
        if df["price"].isnull().all():
            raise DataProcessingError("All price values are null")
        
        logger.info(f"Loaded {len(df)} records")
        return df
        
    except FileNotFoundError:
        raise DataProcessingError(f"File not found: {file_path}")
    except Exception as e:
        raise DataProcessingError(f"Failed to load data: {e}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for processing.
    
    Operations:
    - Remove duplicates
    - Handle missing values
    - Remove outliers
    - Sort by timestamp
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    initial_len = len(df)
    
    logger.info("Starting data cleaning")
    
    # 1. Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"])
    dup_removed = initial_len - len(df)
    if dup_removed > 0:
        logger.info(f"Removed {dup_removed} duplicate records")
    
    # 2. Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 3. Handle missing prices
    null_count = df["price"].isnull().sum()
    if null_count > 0:
        logger.warning(f"Found {null_count} null prices, filling with interpolation")
        df["price"] = df["price"].interpolate(method='linear')
        df["price"] = df["price"].fillna(method='bfill').fillna(method='ffill')
    
    # 4. Remove invalid prices
    invalid_count = (df["price"] <= 0).sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count} invalid prices (<=0)")
        df = df[df["price"] > 0]
    
    # 5. Remove statistical outliers (z-score > 4)
    z_scores = np.abs(stats.zscore(df["price"]))
    outliers = z_scores > 4
    outlier_count = outliers.sum()
    
    if outlier_count > 0:
        logger.warning(f"Removing {outlier_count} outliers")
        df = df[~outliers]
    
    logger.info(f"Cleaning complete: {initial_len} → {len(df)} records")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and time-based features.
    
    Features:
    - Moving averages
    - Returns (percentage change)
    - Volatility
    - Time-based features
    
    Args:
        df: DataFrame with price column
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    logger.info("Adding features")
    
    # 1. Moving averages
    df["sma_24"] = df["price"].rolling(window=24, min_periods=1).mean()
    df["sma_168"] = df["price"].rolling(window=168, min_periods=1).mean()  # 7 days
    
    # 2. Exponential moving average
    df["ema_24"] = df["price"].ewm(span=24, adjust=False).mean()
    
    # 3. Returns
    df["returns_1h"] = df["price"].pct_change(periods=1)
    df["returns_24h"] = df["price"].pct_change(periods=24)
    
    # 4. Volatility (rolling std)
    df["volatility_24h"] = df["price"].rolling(window=24).std()
    
    # 5. Price momentum
    df["momentum"] = df["price"] - df["price"].shift(24)
    
    # 6. Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # 7. Lag features for ARIMA
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"price_lag_{lag}"] = df["price"].shift(lag)
    
    # Fill NaN values created by rolling/shift operations
    df = df.fillna(method='bfill')
    
    logger.info(f"Added {len(df.columns) - 2} features")
    return df


def resample_data(
    df: pd.DataFrame,
    freq: str = "6H",
    agg_method: str = "mean"
) -> pd.DataFrame:
    """
    Resample time series data to lower frequency.
    
    Args:
        df: DataFrame with timestamp index
        freq: Resampling frequency (e.g., '6H', '1D')
        agg_method: Aggregation method ('mean', 'median', 'ohlc')
        
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    
    logger.info(f"Resampling data to {freq} using {agg_method}")
    
    # Set timestamp as index
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    
    # Resample based on method
    if agg_method == "ohlc":
        resampled = df["price"].resample(freq).ohlc()
        resampled.columns = ["open", "high", "low", "close"]
        resampled["price"] = resampled["close"]
    elif agg_method == "median":
        resampled = df.resample(freq).median()
    else:  # mean
        resampled = df.resample(freq).mean()
    
    # Drop NaN rows
    resampled = resampled.dropna()
    
    logger.info(f"Resampled from {len(df)} to {len(resampled)} records")
    
    return resampled


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets chronologically.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
    
    return train_df, test_df


def check_stationarity(series: pd.Series) -> dict:
    """
    Check if time series is stationary using Augmented Dickey-Fuller test.
    
    Args:
        series: Time series data
        
    Returns:
        Dictionary with test results
    """
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(series.dropna())
    
    is_stationary = result[1] < 0.05  # p-value < 0.05
    
    logger.info(f"ADF Test - p-value: {result[1]:.4f}, Stationary: {is_stationary}")
    
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": is_stationary,
        "critical_values": result[4]
    }


def make_stationary(df: pd.DataFrame, method: str = "diff") -> pd.DataFrame:
    """
    Transform data to make it stationary.
    
    Args:
        df: DataFrame with price column
        method: 'diff' or 'log_diff'
        
    Returns:
        Transformed DataFrame
    """
    df = df.copy()
    
    if method == "log_diff":
        df["price_stationary"] = np.log(df["price"]).diff()
        logger.info("Applied log differencing")
    else:  # diff
        df["price_stationary"] = df["price"].diff()
        logger.info("Applied first differencing")
    
    df = df.dropna()
    return df


def process_data(
    input_file: str = "crypto_prices.csv",
    output_file: str = "processed_prices.csv",
    resample_freq: str = "6H",
    add_features_flag: bool = True
) -> pd.DataFrame:
    """
    Main data processing pipeline.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
        resample_freq: Resampling frequency
        add_features_flag: Whether to add technical features
        
    Returns:
        Processed DataFrame
    """
    try:
        # 1. Load data
        df = load_data(input_file)
        
        # 2. Clean data
        df = clean_data(df)
        
        # 3. Add features (optional)
        if add_features_flag:
            df = add_features(df)
        
        # 4. Resample
        df_resampled = resample_data(df, freq=resample_freq)
        
        # 5. Check stationarity
        stationarity = check_stationarity(df_resampled["price"])
        
        if not stationarity["is_stationary"]:
            logger.warning("Data is non-stationary. Consider differencing for ARIMA.")
        
        # 6. Save processed data
        df_resampled.to_csv(output_file)
        logger.info(f"✓ Processed data saved to {output_file}")
        
        # 7. Generate summary statistics
        logger.info("\n=== Data Summary ===")
        logger.info(f"Records: {len(df_resampled)}")
        logger.info(f"Date range: {df_resampled.index.min()} to {df_resampled.index.max()}")
        logger.info(f"Price range: ${df_resampled['price'].min():.2f} - ${df_resampled['price'].max():.2f}")
        logger.info(f"Mean price: ${df_resampled['price'].mean():.2f}")
        logger.info(f"Std dev: ${df_resampled['price'].std():.2f}")
        
        return df_resampled
        
    except Exception as e:
        logger.error(f"✗ Data processing failed: {e}")
        raise DataProcessingError(f"Processing failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process crypto market data")
    parser.add_argument("--input", default="crypto_prices.csv", help="Input file")
    parser.add_argument("--output", default="processed_prices.csv", help="Output file")
    parser.add_argument("--freq", default="6H", help="Resample frequency")
    parser.add_argument("--no-features", action="store_true", help="Skip feature engineering")
    
    args = parser.parse_args()
    
    try:
        df = process_data(
            input_file=args.input,
            output_file=args.output,
            resample_freq=args.freq,
            add_features_flag=not args.no_features
        )
        
        logger.info("✓ Data processing completed successfully")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"✗ Data processing failed: {e}")
        sys.exit(1)