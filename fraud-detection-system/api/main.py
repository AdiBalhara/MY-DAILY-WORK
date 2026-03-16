import pandas as pd
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("src/models/best_fraud_model.pkl")


@app.get("/")
def home():

    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame([data])

    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    
    y_prob = model.predict_proba(input_data)[:,1]

    y_pred = (y_prob > 0.8).astype(int)

    return {
        "fraud_probability": float(y_prob),
        "prediction": int(y_pred),
        "label": "Fraud" if int(y_pred) == 1 else "Legitimate"
    }