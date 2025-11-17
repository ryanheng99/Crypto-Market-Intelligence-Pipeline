import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import json
import logging
import sys
from datetime import datetime
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

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


class ModelTrainingError(Exception):
    """Custom exception for model training errors"""
    pass


def load_training_data(file_path: str = "processed_prices.csv") -> pd.DataFrame:
    """
    Load processed data for model training.
    
    Args:
        file_path: Path to processed CSV
        
    Returns:
        DataFrame with timestamp index
    """
    try:
        logger.info(f"Loading training data from {file_path}")
        df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
        
        if df.empty:
            raise ModelTrainingError("Empty dataset")
        
        if len(df) < 50:
            raise ModelTrainingError(f"Insufficient data: {len(df)} records (need at least 50)")
        
        logger.info(f"Loaded {len(df)} records for training")
        return df
        
    except FileNotFoundError:
        raise ModelTrainingError(f"File not found: {file_path}")
    except Exception as e:
        raise ModelTrainingError(f"Failed to load data: {e}")


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    """
    Split data chronologically into train and test sets.
    
    Args:
        df: DataFrame with price data
        test_size: Proportion for test set
        
    Returns:
        Tuple of (train_series, test_series)
    """
    split_idx = int(len(df) * (1 - test_size))
    
    train = df["price"].iloc[:split_idx]
    test = df["price"].iloc[split_idx:]
    
    logger.info(f"Split: {len(train)} train, {len(test)} test samples")
    
    return train, test


def find_best_arima_order(
    train_data: pd.Series,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5
) -> Tuple[int, int, int]:
    """
    Find best ARIMA parameters using grid search on AIC.
    
    Args:
        train_data: Training time series
        max_p: Maximum AR order
        max_d: Maximum differencing order
        max_q: Maximum MA order
        
    Returns:
        Tuple of best (p, d, q)
    """
    logger.info("Searching for best ARIMA parameters...")
    
    best_aic = np.inf
    best_order = None
    
    # Grid search
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        
                except Exception:
                    continue
    
    if best_order is None:
        logger.warning("Grid search failed, using default (5,1,0)")
        best_order = (5, 1, 0)
    else:
        logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
    
    return best_order


def train_arima_model(
    train_data: pd.Series,
    order: Tuple[int, int, int] = None,
    auto_order: bool = False
) -> ARIMA:
    """
    Train ARIMA model.
    
    Args:
        train_data: Training time series
        order: ARIMA (p,d,q) order
        auto_order: Automatically find best order
        
    Returns:
        Fitted ARIMA model
    """
    if auto_order:
        order = find_best_arima_order(train_data)
    elif order is None:
        order = (5, 1, 0)
    
    logger.info(f"Training ARIMA{order} model")
    
    try:
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        
        logger.info(f"Model trained successfully")
        logger.info(f"AIC: {fitted_model.aic:.2f}")
        logger.info(f"BIC: {fitted_model.bic:.2f}")
        
        return fitted_model
        
    except Exception as e:
        raise ModelTrainingError(f"Failed to train ARIMA model: {e}")


def train_sarima_model(
    train_data: pd.Series,
    order: Tuple[int, int, int] = (5, 1, 0),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)
) -> SARIMAX:
    """
    Train SARIMA model (ARIMA with seasonality).
    
    Args:
        train_data: Training time series
        order: Non-seasonal (p,d,q) order
        seasonal_order: Seasonal (P,D,Q,s) order
        
    Returns:
        Fitted SARIMA model
    """
    logger.info(f"Training SARIMA{order}x{seasonal_order} model")
    
    try:
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False)
        
        logger.info(f"SARIMA model trained successfully")
        logger.info(f"AIC: {fitted_model.aic:.2f}")
        
        return fitted_model
        
    except Exception as e:
        logger.warning(f"SARIMA training failed: {e}, falling back to ARIMA")
        return train_arima_model(train_data, order=order)


def evaluate_model(
    model,
    test_data: pd.Series,
    steps: int = None
) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Fitted model
        test_data: Test time series
        steps: Number of steps to forecast
        
    Returns:
        Dictionary of metrics
    """
    if steps is None:
        steps = len(test_data)
    
    logger.info(f"Evaluating model on {steps} test samples")
    
    try:
        # Make predictions
        forecast = model.forecast(steps=steps)
        
        # Calculate metrics
        mae = np.mean(np.abs(forecast - test_data.values[:steps]))
        rmse = np.sqrt(np.mean((forecast - test_data.values[:steps]) ** 2))
        mape = np.mean(np.abs((test_data.values[:steps] - forecast) / test_data.values[:steps])) * 100
        
        # Directional accuracy
        actual_direction = np.sign(test_data.values[1:steps] - test_data.values[:steps-1])
        pred_direction = np.sign(forecast[1:] - forecast[:-1])
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "directional_accuracy": float(directional_accuracy),
            "test_samples": steps
        }
        
        logger.info("=== Model Performance ===")
        logger.info(f"MAE: ${mae:.2f}")
        logger.info(f"RMSE: ${rmse:.2f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "error": str(e)
        }


def save_model(
    model,
    model_path: str = "arima_model.pkl",
    metadata_path: str = "model_metadata.json",
    metrics: Dict = None
) -> None:
    """
    Save trained model and metadata.
    
    Args:
        model: Fitted model
        model_path: Path to save model
        metadata_path: Path to save metadata
        metrics: Performance metrics
    """
    try:
        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"✓ Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            "model_type": type(model).__name__,
            "trained_at": datetime.utcnow().isoformat(),
            "version": "1.0",
            "metrics": metrics or {},
            "model_params": {
                "order": getattr(model, 'order', None),
                "seasonal_order": getattr(model, 'seasonal_order', None)
            }
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to {metadata_path}")
        
    except Exception as e:
        raise ModelTrainingError(f"Failed to save model: {e}")


def load_model(model_path: str = "arima_model.pkl"):
    """
    Load trained model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return model
        
    except FileNotFoundError:
        raise ModelTrainingError(f"Model file not found: {model_path}")
    except Exception as e:
        raise ModelTrainingError(f"Failed to load model: {e}")


def train_model(
    input_file: str = "processed_prices.csv",
    model_path: str = "arima_model.pkl",
    model_type: str = "arima",
    auto_order: bool = False,
    evaluate: bool = True
) -> None:
    """
    Main model training pipeline.
    
    Args:
        input_file: Path to processed data
        model_path: Path to save model
        model_type: 'arima' or 'sarima'
        auto_order: Automatically find best ARIMA order
        evaluate: Whether to evaluate on test set
    """
    try:
        # 1. Load data
        df = load_training_data(input_file)
        
        # 2. Train-test split
        train_data, test_data = train_test_split(df, test_size=0.2)
        
        # 3. Train model
        if model_type.lower() == "sarima":
            model = train_sarima_model(train_data)
        else:
            model = train_arima_model(train_data, auto_order=auto_order)
        
        # 4. Evaluate (optional)
        metrics = None
        if evaluate and len(test_data) > 0:
            metrics = evaluate_model(model, test_data)
        
        # 5. Retrain on full data for production
        logger.info("Retraining on full dataset for production")
        full_data = df["price"]
        
        if model_type.lower() == "sarima":
            final_model = train_sarima_model(full_data, order=model.specification['order'])
        else:
            final_model = train_arima_model(full_data, order=model.specification['order'])
        
        # 6. Save model
        save_model(final_model, model_path, metrics=metrics)
        
        logger.info("✓ Model training completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        raise ModelTrainingError(f"Training failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train crypto price forecasting model")
    parser.add_argument("--input", default="processed_prices.csv", help="Input file")
    parser.add_argument("--output", default="arima_model.pkl", help="Model output file")
    parser.add_argument("--type", choices=["arima", "sarima"], default="arima", help="Model type")
    parser.add_argument("--auto-order", action="store_true", help="Auto-find best ARIMA order")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    
    args = parser.parse_args()
    
    try:
        train_model(
            input_file=args.input,
            model_path=args.output,
            model_type=args.type,
            auto_order=args.auto_order,
            evaluate=not args.no_eval
        )
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        sys.exit(1)