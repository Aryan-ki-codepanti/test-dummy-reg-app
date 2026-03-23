import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# uvicorn main:app --reload

# 1. Load the model once at startup
model = joblib.load("linear_regression_model.joblib")
app = FastAPI()

# 2. Define the input schema


class RegInput(BaseModel):
    experience: float

# 3. Create the prediction endpoint


@app.post("/predict")
def predict(data: RegInput):
    # Convert Pydantic object to NumPy array
    features = np.array([[data.experience]])

    # Get prediction
    prediction = model.predict(features)
    return {"prediction": prediction[0]}
