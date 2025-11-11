import requests
import pandas as pd

def fetch_market_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "30", "interval": "hourly"}
    response = requests.get(url, params=params)
    data = response.json()
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.to_csv("crypto_prices.csv", index=False)

if __name__ == "__main__":
    fetch_market_data()
