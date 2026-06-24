import os
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Set up simple logging to see what's running
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Get paths right relative to where this script sits
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(BASE_DIR, "models")
os.makedirs(models_path, exist_ok=True)

# The exact columns our model will expect to see after one-hot encoding
FEATURES = [
    "Base_Salary", "Years_At_Company", "Satisfaction_Score", 
    "Avg_Monthly_Overtime_Hours", "Comp_Ratio",
    "Department_Engineering", "Department_Finance", 
    "Department_HR", "Department_Marketing", "Department_Sales"
]

def make_data(n_samples=300, seed=42):
    """Generates synthetic employee data with realistic, correlated feature logic."""
    np.random.seed(seed)
    depts = ["Engineering", "Sales", "HR", "Marketing", "Finance"]
    
    # 1. Base Feature Generation
    departments = np.random.choice(depts, size=n_samples)
    years_at_company = np.random.randint(0, 15, size=n_samples)
    satisfaction_scores = np.random.randint(1, 6, size=n_samples)
    overtime_hours = np.random.uniform(0.0, 80.0, size=n_samples)
    
    # 2. Establish Department-Based Salary Baselines (Realistic Market Ranges)
    # Engineering/Finance typically make more than HR/Marketing
    dept_market_averages = {
        "Engineering": 95000.0,
        "Finance": 88000.0,
        "Sales": 75000.0,
        "Marketing": 70000.0,
        "HR": 65000.0
    }
    
    base_salaries = []
    comp_ratios = []
    
    for dept in departments:
        market_avg = dept_market_averages[dept]
        # Generate salary centered around that department's market average
        salary = np.random.normal(loc=market_avg, scale=15000.0)
        salary = np.clip(salary, 30000.0, 150000.0) # Keep within bounds
        
        base_salaries.append(round(salary, 2))
        comp_ratios.append(round(salary / market_avg, 2))
        
    df = pd.DataFrame({
        "Department": departments,
        "Base_Salary": base_salaries,
        "Years_At_Company": years_at_company,
        "Satisfaction_Score": satisfaction_scores,
        "Avg_Monthly_Overtime_Hours": overtime_hours,
        "Comp_Ratio": comp_ratios
    })
    
    # 3. INTERACTION-DRIVEN RISK SCORING (The Core Fix)
    # - High overtime is offset if satisfaction is high or compensation is excellent.
    # - Low satisfaction hurts more if the salary is also below market average (Comp_Ratio < 1.0).
    
    overtime_penalty = df["Avg_Monthly_Overtime_Hours"] / 35.0
    compensation_buffer = (df["Comp_Ratio"] - 1.0) * 3.0  # Positive buffer reduces risk
    satisfaction_multiplier = df["Satisfaction_Score"] / 5.0
    
    # Combine them into a complex interaction score
    risk_score = (overtime_penalty * 2.0) - compensation_buffer - (satisfaction_multiplier * 3.5)
    
    # Add a bit of natural variance/noise so it isn't a perfect mathematical split
    noise = np.random.normal(0, 0.5, n_samples)
    final_score = risk_score + noise
    
    # Map to binary class via standard Sigmoid activation
    df["Left_Company"] = (1 / (1 + np.exp(-final_score)) > 0.5).astype(int)
    return df
    
def prepare_and_scale(df, is_train=True):
    """Handles dummy encoding and continuous data scaling."""
    # Convert text categories into binary columns
    df_encoded = pd.get_dummies(df)
    # Reindex to ensure columns always match our master list shape
    df_final = df_encoded.reindex(columns=FEATURES, fill_value=0)
    
    scaler = StandardScaler()
    if is_train:
        scaled_data = scaler.fit_transform(df_final)
        # Save the trained scaler state for the web app to use later
        joblib.dump(scaler, os.path.join(models_path, "feature_scaler.pkl"))
    else:
        scaler = joblib.load(os.path.join(models_path, "feature_scaler.pkl"))
        scaled_data = scaler.transform(df_final)
        
    return pd.DataFrame(scaled_data, columns=FEATURES)

if __name__ == "__main__":
    logger.info("Generating training dataset...")
    data = make_data()
    
    X = data.drop(columns=["Left_Company"])
    y = data["Left_Company"]
    
    # Save our column scheme out so predict.py knows the correct layout order
    joblib.dump(FEATURES, os.path.join(models_path, "model_features.pkl"))
    
    X_scaled = prepare_and_scale(X, is_train=True)
    
    logger.info("Training the Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    
    # Save the trained model weights
    joblib.dump(model, os.path.join(models_path, "attrition_gbm_model.pkl"))
    logger.info("Model and preprocessing assets saved successfully.")