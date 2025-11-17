# Multi-stage build for smaller image size
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY data_ingestion.py .
COPY data_processing.py .
COPY model_training.py .
COPY api_service.py .

# Copy data files (if they exist)
COPY crypto_prices.csv* ./
COPY processed_prices.csv* ./
COPY arima_model.pkl* ./
COPY model_metadata.json* ./

# Create directory for logs
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API service
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]