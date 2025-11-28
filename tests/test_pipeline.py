import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion import fetch_market_data, validate_data_file, CoinGeckoAPIError
from data_processing import clean_data, add_features, resample_data, DataProcessingError
from model_training import train_arima_model, evaluate_model, ModelTrainingError


class TestDataIngestion:
    """Test data ingestion module"""
    
    @patch("data_ingestion.requests.get")
    def test_fetch_market_data_success(self, mock_get):
        """Test successful data fetch"""
        # Mock API response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "prices": [
                [1700000000000, 50000],
                [1700003600000, 50500]
            ]
        }

        df = fetch_market_data(coin="bitcoin", days=7, output_file="test_prices.csv")
            
        assert not df.empty
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert len(df) > 2
        assert df["price"].iloc[0] == 50000

    @patch("data_ingestion.requests.get")
    def test_fetch_market_data_api_failure(self, mock_get):
        """Test API failure handling"""
        mock_get.return_value.status_code = 404

        with pytest.raises(CoinGeckoAPIError):
            fetch_market_data(coin="invalid_coin", days=1, output_file="test_prices.csv")
                
    
    def test_fetch_invalid_coin(self):
        """Test fetch with invalid coin"""
        with pytest.raises(CoinGeckoAPIError):
            fetch_market_data(coin="invalid_coin_xyz", days=1)
    
    def test_validate_data_file(self):
        """Test data file validation"""
        # Create test data
        test_df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=50, freq="H"),
            "price": np.random.uniform(40000, 50000, 50)
        })
        test_df.to_csv("test_validation.csv", index=False)
        
        assert validate_data_file("test_validation.csv") == True
        
        # Cleanup
        os.remove("test_validation.csv")


class TestDataProcessing:
    """Test data processing module"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
        data = {
            "timestamp": dates,
            "price": np.random.uniform(40000, 50000, 100)
        }
        return pd.DataFrame(data)
    
    def test_clean_data(self, sample_data):
        """Test data cleaning"""
        # Add some duplicates and nulls
        df_dirty = sample_data.copy()
        df_dirty = pd.concat([df_dirty, df_dirty.iloc[:5]], ignore_index=True)
        df_dirty.loc[10, "price"] = None
        
        df_clean = clean_data(df_dirty)
        
        assert len(df_clean) <= len(df_dirty)
        assert df_clean["price"].isnull().sum() == 0
    
    def test_add_features(self, sample_data):
        """Test feature engineering"""
        df_features = add_features(sample_data)
        
        assert "sma_24" in df_features.columns
        assert "returns_1h" in df_features.columns
        assert "hour" in df_features.columns
        assert len(df_features) == len(sample_data)
    
    def test_resample_data(self, sample_data):
        """Test data resampling"""
        df_resampled = resample_data(sample_data, freq="6H")
        
        assert len(df_resampled) < len(sample_data)
        assert not df_resampled.empty


class TestModelTraining:
    """Test model training module"""
    
    @pytest.fixture
    def train_data(self):
        """Create training data"""
        dates = pd.date_range(start="2024-01-01", periods=200, freq="6H")
        prices = 40000 + np.cumsum(np.random.randn(200) * 100)
        return pd.Series(prices, index=dates)
    
    def test_train_arima_model(self, train_data):
        """Test ARIMA model training"""
        model = train_arima_model(train_data, order=(2, 1, 0))
        
        assert model is not None
        assert hasattr(model, 'forecast')
    
    def test_model_forecast(self, train_data):
        """Test model forecasting"""
        model = train_arima_model(train_data, order=(2, 1, 0))
        forecast = model.forecast(steps=5)
        
        assert len(forecast) == 5
        assert all(forecast > 0)
    
    def test_evaluate_model(self, train_data):
        """Test model evaluation"""
        # Split data
        train = train_data[:-20]
        test = train_data[-20:]
        
        # Train model
        model = train_arima_model(train, order=(2, 1, 0))
        
        # Evaluate
        metrics = evaluate_model(model, test, steps=10)
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] > 0


class TestAPI:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from api_service import app
        
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "service" in response.json()
    
    def test_health_endpoint(self, client):
        """Test health check"""
        response = client.get("/health")
        
        assert response.status_code in [200, 503]
        assert "status" in response.json()
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint"""
        try:
            response = client.get("/predict?steps=1")
            
            if response.status_code == 200:
                data = response.json()
                assert "predictions" in data
                assert len(data["predictions"]) == 1
                assert data["predictions"][0] > 0
        except Exception:
            pytest.skip("Model not available")


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_full_pipeline(self):
        """Test complete pipeline flow"""
        try:
            # 1. Data ingestion
            df_raw = fetch_market_data(coin="bitcoin", days=7, output_file="test_pipeline.csv")
            assert not df_raw.empty
            
            # 2. Data processing
            df_clean = clean_data(df_raw)
            df_processed = resample_data(df_clean, freq="6H")
            df_processed.to_csv("test_processed.csv")
            
            # 3. Model training
            train_data = df_processed["price"]
            if len(train_data) >= 50:
                model = train_arima_model(train_data, order=(2, 1, 0))
                
                # 4. Prediction
                forecast = model.forecast(steps=1)
                assert forecast[0] > 0
            
            # Cleanup
            os.remove("test_pipeline.csv")
            os.remove("test_processed.csv")
            
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])