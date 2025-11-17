# 🧠 Crypto Market Intelligence Pipeline

[![CI/CD Pipeline](https://github.com/ryanheng99/Crypto-Market-Intelligence-Pipeline/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/ryanheng99/Crypto-Market-Intelligence-Pipeline/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, full-stack data engineering system that ingests real-time Bitcoin market data, processes it with advanced time-series techniques, trains forecasting models, and serves predictions via a RESTful API—all automated through CI/CD.

## 🎯 Objective

Build an end-to-end ML pipeline that:
- 📊 **Ingests** real-time Bitcoin market data from CoinGecko API
- 🔄 **Processes** and engineers features with technical indicators
- 🤖 **Trains** time-series forecasting models (ARIMA/SARIMA) with automatic hyperparameter tuning
- 🚀 **Serves** predictions via a high-performance FastAPI web service
- ⚙️ **Automates** the entire workflow using CI/CD with GitHub Actions
- 📦 **Containerizes** for deployment with Docker

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  CoinGecko API  │────▶│  Data Ingestion  │────▶│  Raw CSV Data   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Predictions    │◀────│  ARIMA Model     │◀────│  Data Processing│
│  (API Response) │     │  Training        │     │  & Feature Eng. │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐     ┌──────────────────┐
│  FastAPI        │     │  Model Artifacts │
│  /predict       │     │  (.pkl + meta)   │
│  /health        │     └──────────────────┘
│  /model/info    │
└─────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Docker Container                │
│  ┌────────────────────────────────┐    │
│  │  CI/CD Pipeline (GitHub Actions)│   │
│  │  • Lint & Test                  │   │
│  │  • Data Ingestion               │   │
│  │  • Model Training               │   │
│  │  • API Testing                  │   │
│  │  • Docker Build & Deploy        │   │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

---

## 🔧 Components

### 1. **Data Ingestion** (`data_ingestion.py`)
- ✅ Fetches hourly Bitcoin price data for the past 30 days from CoinGecko's free API
- ✅ **Robust error handling** with exponential backoff retry logic
- ✅ **Data validation** to ensure quality (null checks, outlier detection)
- ✅ **Multi-coin support** (Bitcoin, Ethereum, Binance Coin, etc.)
- ✅ Saves raw data to `crypto_prices.csv`

**Key Features:**
```python
# Automatic retry on API failures
fetch_market_data(coin="bitcoin", days=30, max_retries=3)

# Multi-coin ingestion
fetch_multiple_coins(["bitcoin", "ethereum", "binancecoin"])
```

### 2. **Data Processing** (`data_processing.py`)
- 🔄 Cleans and validates raw data (removes duplicates, handles missing values)
- 📈 **Feature engineering**: Moving averages (SMA, EMA), RSI, volatility, momentum
- 🔍 **Stationarity checks** using Augmented Dickey-Fuller test
- 📊 Resamples data to 6-hour intervals to reduce noise
- 💾 Saves processed data to `processed_prices.csv`

**Technical Indicators:**
- Simple & Exponential Moving Averages (SMA, EMA)
- Rate of Change (ROC)
- Volatility (rolling standard deviation)
- Relative Strength Index (RS