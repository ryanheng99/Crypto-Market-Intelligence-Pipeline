import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle

def train_model(file_path="processed_prices.csv"):
    df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
    model = ARIMA(df["price"], order=(5,1,0))
    model_fit = model.fit()
    with open("arima_model.pkl", "wb") as f:
        pickle.dump(model_fit, f)

if __name__ == "__main__":
    train_model()