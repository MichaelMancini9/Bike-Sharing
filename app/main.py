from fastapi import FastAPI
import pandas as pd
import joblib
import math

from app.schemas import BikeInput

app = FastAPI(title="Bike Demand Prediction API")

model = joblib.load("models/bike_model.joblib")
feature_cols = joblib.load("models/feature_cols.joblib")


@app.get("/")
def root():
    return {"message": "Bike Demand Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(data: BikeInput):
    hr_sin = math.sin(2 * math.pi * data.hr / 24)
    hr_cos = math.cos(2 * math.pi * data.hr / 24)

    input_data = {
        "season": data.season,
        "yr": data.yr,
        "mnth": data.mnth,
        "holiday": data.holiday,
        "weekday": data.weekday,
        "workingday": data.workingday,
        "weathersit": data.weathersit,
        "temp": data.temp,
        "hum": data.hum,
        "windspeed": data.windspeed,
        "hr_sin": hr_sin,
        "hr_cos": hr_cos
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_cols]

    prediction = model.predict(input_df)[0]

    return {
        "predicted_count": round(float(prediction), 2)
    }