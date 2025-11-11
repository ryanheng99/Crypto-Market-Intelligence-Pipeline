import pandas as pd

def process_data(file_path="crypto_prices.csv"):
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.resample("6H").mean().dropna()
    df.to_csv("processed_prices.csv")

if __name__ == "__main__":
    process_data()
    