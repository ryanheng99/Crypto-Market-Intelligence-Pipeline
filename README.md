## 🧠 Project Name: Crypto Market Intelligence Pipeline
## 🎯 Objective
```
Build a full-stack data engineering system that:

Ingests real-time Bitcoin market data from a free API (CoinGecko)
Processes and resamples the data
Trains a time-series forecasting model (ARIMA)
Serves predictions via a FastAPI web service
Automates the entire workflow using CI/CD with GitHub Actions
```

## 🔧 Components Explained
```
1. Data Ingestion (data_ingestion.py)

Fetches hourly Bitcoin price data for the past 30 days using CoinGecko’s free API.
Saves the data to a CSV file (crypto_prices.csv).

2. Data Processing (data_processing.py)

Reads the raw CSV file.
Resamples the data to 6-hour intervals to smooth out noise.
Saves the cleaned data to processed_prices.csv.

3. Model Training (model_training.py)

Loads the processed data.
Trains an ARIMA model to forecast future prices.
Saves the trained model as a .pkl file for reuse.

4. API Service (api_service.py)

Uses FastAPI to create a REST endpoint (/predict).
When accessed, it loads the trained model and returns the next predicted price.

5. CI/CD Pipeline (.github/workflows/ci_cd.yml)

GitHub Actions automates:

Dependency installation
Running ingestion, processing, and training scripts
Starting the API and testing the /predict endpoint



6. Dockerfile

Containerizes the FastAPI app for deployment.
Ensures consistent environments across development and production.
```

## 📦 Tech Stack
```
Python: Core language
FastAPI: Web framework
ARIMA (statsmodels): Time-series forecasting
GitHub Actions: CI/CD automation
Docker: Containerization
CoinGecko API: Free crypto data source
```

## ✅ Why This Project Is Valuable
```
Demonstrates real-world data engineering and ML deployment skills.
Uses free tools and APIs—ideal for portfolio or learning.
Fully automated pipeline with CI/CD—industry best practice.
Easily extendable to other coins, models, or alerting systems.
```