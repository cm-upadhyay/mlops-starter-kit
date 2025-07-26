from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="ML Prediction API")

MODEL_PATH = 'iris_model.pkl'
model = joblib.load(MODEL_PATH)

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(request: IrisRequest):
    data = pd.DataFrame([request.dict()])
    prediction = model.predict(data)[0]
    return {"species_prediction": prediction}

@app.get("/")
def read_root():
    return {"message": "API is running. POST to /predict for predictions."}