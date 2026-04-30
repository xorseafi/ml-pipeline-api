# This file creates a REST API to use the trained model

from fastapi import FastAPI  # Framework for building APIs
import pickle
import numpy as np  # For handling numerical data

# Create API instance
app = FastAPI()

# Load trained model and scaler when API starts
with open("../model/model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# To test endpoint)
@app.get("/")
def home():
    return {"message": "ML Model API is running"}

# Prediction route
@app.post("/predict")
def predict(features: list):
    # Convert input list into proper format
    data = np.array(features).reshape(1, -1)

    # Apply same scaling used during training
    data = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data)

    # Return result
    return {"prediction": int(prediction[0])}
