import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import pandas as pd

# Pull in our manual inference routing logic
from src.predict import run_standalone_inference

# Simple, single-line log layout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(title="Workforce Attrition API Portal", version="1.0.0")

# Permit the local Streamlit container network card to connect securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to cloud database logs cluster
MONGO_URI = os.getenv("MONGO_URI")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["workforce_db"]
    logs_collection = db["prediction_logs"]
    client.admin.command('ping')
    logger.info("MongoDB Atlas connected successfully.")
except Exception as e:
    logger.error(f"MongoDB connection dropped: {e}")
    logs_collection = None

# Input typing validator mapping
class EmployeeProfileInput(BaseModel):
    Department: str
    Base_Salary: float
    Years_At_Company: int
    Satisfaction_Score: int
    Avg_Monthly_Overtime_Hours: float
    Comp_Ratio: float

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_attrition(profile: EmployeeProfileInput):
    """Processes frontend telemetry records and drops them into MongoDB."""
    try:
        # Convert request parameters into standard python dictionary mapping
        input_data = profile.model_dump()
        
        # Pass input data over to our local feature engineering layer
        response_payload = run_standalone_inference(input_data)
        
        # Append history dump data straight into Atlas
        if logs_collection is not None:
            logs_collection.insert_one({
                "timestamp": datetime.utcnow(),
                "employee_inputs": input_data,
                "prediction_outputs": response_payload
            })
        return response_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))