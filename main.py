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



# --------- Get User By ID ---------
@app.get("/users/{user_id}")
def get_user(user_id: int):
    response = requests.get(f"{BASE_URL}/{user_id}")
    
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="User not found")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error fetching user")

    return response.json()

# --------- Update User ---------
@app.put("/users/{user_id}")
def update_user(user_id: int, user: UserUpdate):
    response = requests.put(
        f"{BASE_URL}/{user_id}",
        json=user.dict(exclude_none=True)
    )

    if response.status_code not in [200, 201]:
        raise HTTPException(status_code=500, detail="Failed to update user")

    return response.json()
