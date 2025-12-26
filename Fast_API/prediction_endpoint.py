# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# --- 1. Initialize the App ---
app = FastAPI(
    title="Day 16 Prediction Endpoint",
    description="A production-ready API to serve my ML Pipeline."
)

# --- 2. Load the Pipeline ---

# Ensure "final_pipeline_v1.pkl" is in the same folder
try:
    pipeline = joblib.load("final_pipeline_v1.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    pipeline = None

# --- 3. Define Input Schema (Pydantic) ---

class InputData(BaseModel):
    sepal_length: float  
    sepal_width: float   #
    petal_length: float  
    petal_width: float   

# --- 4. The Prediction Endpoint ---
@app.post("/predict")
def predict_endpoint(data: InputData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Step A: Convert input JSON to DataFrame
        # We use [data.dict()] to make it a list (one row)
        input_df = pd.DataFrame([data.dict()])
        
        # Step B: Predict
        # The pipeline will handle Scaling/Encoding automatically!
        prediction = pipeline.predict(input_df)
        
        class_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        # Step C: Return JSON
        return {
            "flower_class": int(prediction[0]),
            "flower_name": class_names[int(prediction[0])]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 5. Run the Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)