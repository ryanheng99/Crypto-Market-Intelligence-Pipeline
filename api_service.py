from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

@app.get("/predict")
def get_prediction():
    df = pd.read_csv("processed_prices.csv", index_col="timestamp", parse_dates=True)
    with open("arima_model.pkl", "rb") as f:
        model = pickle.load(f)
    forecast = model.forecast(steps=1)
    return {"next_price_prediction": forecast[0]}