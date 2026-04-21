import joblib
import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# uvicorn main:app --reload

# 1. Load the model once at startup
model = joblib.load("linear_regression_model.joblib")
app = FastAPI()

# 2. Define the input schema
class RegInput(BaseModel):
    experience: float

# 3. Prediction endpoint
@app.post("/predict")
def predict(data: RegInput):
    features = np.array([[data.experience]])
    prediction = model.predict(features)
    return {"prediction": prediction[0]}

# 4. Dummy users endpoint
@app.get("/users")
def get_users():
    url = "https://jsonplaceholder.typicode.com/users"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": "Failed to fetch users"}

    return response.json()
