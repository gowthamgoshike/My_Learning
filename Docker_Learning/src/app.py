from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialize App
app = FastAPI()

# Load Model (Global variable)
# In Docker, we will ensure the model file is copied to the right place
model_path = "Docker_Learning/model/housing_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Define input data format
class HouseFeatures(BaseModel):
    size_sqft: float

@app.get("/")
def home():
    return {"message": "Housing Price Prediction API Container"}

@app.post("/predict")
def predict(features: HouseFeatures):
    if not model:
        return {"error": "Model not found"}
    
    # Reshape input for sklearn
    data_in = np.array([[features.size_sqft]])
    prediction = model.predict(data_in)
    
    return {"predicted_price": prediction[0]}