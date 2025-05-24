from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Charger le modèle et le scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

class PatientData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: PatientData):
    scaled_features = scaler.transform([data.features])
    prediction = model.predict(scaled_features)
    return {"prediction": prediction[0]}