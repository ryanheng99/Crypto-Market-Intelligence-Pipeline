from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import pickle
import logging
import json
from datetime import datetime
from typing import List, Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API",
    description="Real-time Bitcoin price forecasting using ARIMA",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model caching
MODEL_CACHE = None
METADATA_CACHE = None
LAST_PREDICTION = None


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    steps: int = 1
    confidence_level: float = 0.95


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[float]
    timestamp: str
    model_type: str
    confidence_intervals: Optional[List[dict]] = None
    metadata: Optional[dict] = None


def load_model_with_cache(model_path: str = "arima_model.pkl"):
    """
    Load model with caching to avoid repeated disk I/O.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model
    """
    global MODEL_CACHE
    
    if MODEL_CACHE is None:
        try:
            with open(model_path, "rb") as f:
                MODEL_CACHE = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise HTTPException(status_code=503, detail="Model not available. Please train model first.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
    
    return MODEL_CACHE


def load_metadata(metadata_path: str = "model_metadata.json") -> dict:
    """
    Load model metadata with caching.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Metadata dictionary
    """
    global METADATA_CACHE
    
    if METADATA_CACHE is None:
        try:
            with open(metadata_path, "r") as f:
                METADATA_CACHE = json.load(f)
            logger.info(f"Metadata loaded from {metadata_path}")
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_path}")
            METADATA_CACHE = {}
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            METADATA_CACHE = {}
    
    return METADATA_CACHE


def get_current_data(file_path: str = "processed_prices.csv") -> pd.DataFrame:
    """
    Load current price data.
    
    Args:
        file_path: Path to processed data
        
    Returns:
        DataFrame with price data
    """
    try:
        df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise HTTPException(status_code=500, detail=f"Data loading error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model and metadata on startup"""
    logger.info("Starting API service...")
    try:
        load_model_with_cache()
        load_metadata()
        logger.info("✓ API service started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Crypto Price Prediction API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "/predict": "Get price predictions",
            "/health": "Check service health",
            "/model/info": "Get model information",
            "/data/latest": "Get latest price data"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model = load_model_with_cache()
        
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/predict", response_model=PredictionResponse)
async def get_prediction(steps: int = 1, include_confidence: bool = True):
    """
    Get price predictions.
    
    Args:
        steps: Number of steps to forecast (default: 1)
        include_confidence: Include confidence intervals
        
    Returns:
        Prediction response with forecasted prices
    """
    global LAST_PREDICTION
    
    try:
        # Validate input
        if steps < 1 or steps > 100:
            raise HTTPException(status_code=400, detail="Steps must be between 1 and 100")
        
        # Load model
        model = load_model_with_cache()
        metadata = load_metadata()
        
        # Make prediction
        logger.info(f"Generating {steps}-step forecast")
        forecast = model.forecast(steps=steps)
        
        predictions = [float(p) for p in forecast]
        
        # Get confidence intervals (if available)
        confidence_intervals = None
        if include_confidence and hasattr(model, 'get_forecast'):
            try:
                forecast_obj = model.get_forecast(steps=steps)
                conf_int = forecast_obj.conf_int()
                
                confidence_intervals = [
                    {
                        "step": i + 1,
                        "lower": float(conf_int.iloc[i, 0]),
                        "upper": float(conf_int.iloc[i, 1])
                    }
                    for i in range(steps)
                ]
            except Exception as e:
                logger.warning(f"Could not compute confidence intervals: {e}")
        
        # Cache last prediction
        LAST_PREDICTION = {
            "predictions": predictions,
            "timestamp": datetime.utcnow().isoformat(),
            "steps": steps
        }
        
        # Build response
        response = PredictionResponse(
            predictions=predictions,
            timestamp=datetime.utcnow().isoformat(),
            model_type=metadata.get("model_type", "ARIMA"),
            confidence_intervals=confidence_intervals,
            metadata={
                "trained_at": metadata.get("trained_at"),
                "metrics": metadata.get("metrics", {})
            }
        )
        
        logger.info(f"✓ Prediction successful: ${predictions[0]:.2f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def post_prediction(request: PredictionRequest):
    """
    POST endpoint for predictions with custom parameters.
    
    Args:
        request: Prediction request with steps
        
    Returns:
        Prediction response
    """
    return await get_prediction(
        steps=request.steps,
        include_confidence=request.confidence_level > 0
    )


@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model metadata and statistics
    """
    try:
        metadata = load_metadata()
        
        # Get data info
        try:
            df = get_current_data()
            data_info = {
                "total_records": len(df),
                "date_range": {
                    "start": df.index.min().isoformat(),
                    "end": df.index.max().isoformat()
                },
                "latest_price": float(df["price"].iloc[-1]),
                "price_range": {
                    "min": float(df["price"].min()),
                    "max": float(df["price"].max()),
                    "mean": float(df["price"].mean())
                }
            }
        except Exception as e:
            logger.warning(f"Could not load data info: {e}")
            data_info = None
        
        return {
            "model": metadata,
            "data": data_info,
            "last_prediction": LAST_PREDICTION
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/latest")
async def get_latest_data(limit: int = 10):
    """
    Get latest price data.
    
    Args:
        limit: Number of records to return
        
    Returns:
        Latest price records
    """
    try:
        df = get_current_data()
        
        latest = df.tail(limit)
        
        return {
            "data": [
                {
                    "timestamp": idx.isoformat(),
                    "price": float(row["price"])
                }
                for idx, row in latest.iterrows()
            ],
            "count": len(latest)
        }
        
    except Exception as e:
        logger.error(f"Failed to get latest data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """
    Reload model and metadata (useful after retraining).
    
    Returns:
        Status message
    """
    global MODEL_CACHE, METADATA_CACHE
    
    try:
        MODEL_CACHE = None
        METADATA_CACHE = None
        
        load_model_with_cache()
        load_metadata()
        
        logger.info("✓ Model reloaded successfully")
        
        return {
            "status": "success",
            "message": "Model reloaded",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """
    Get model performance metrics.
    
    Returns:
        Model evaluation metrics
    """
    try:
        metadata = load_metadata()
        
        metrics = metadata.get("metrics", {})
        
        if not metrics:
            raise HTTPException(status_code=404, detail="No metrics available")
        
        return {
            "metrics": metrics,
            "trained_at": metadata.get("trained_at"),
            "model_type": metadata.get("model_type")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )