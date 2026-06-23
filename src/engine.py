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
    """Generates synthetic employee data to train the model."""
    np.random.seed(seed)
    depts = ["Engineering", "Sales", "HR", "Marketing", "Finance"]
    
    df = pd.DataFrame({
        "Department": np.random.choice(depts, size=n_samples),
        "Base_Salary": np.random.randint(30000, 140000, size=n_samples).astype(float),
        "Years_At_Company": np.random.randint(0, 15, size=n_samples),
        "Satisfaction_Score": np.random.randint(1, 6, size=n_samples),
        "Avg_Monthly_Overtime_Hours": np.random.uniform(0.0, 80.0, size=n_samples),
        "Comp_Ratio": np.random.uniform(0.6, 1.4, size=n_samples)
    })
    
    # Simple equation to create a realistic target logic for who leaves
    risk_score = ((df["Avg_Monthly_Overtime_Hours"] / 35.0) * 1.5 + (1.2 - df["Comp_Ratio"]) * 2.0 - (df["Satisfaction_Score"] * 0.8))
    df["Left_Company"] = (1 / (1 + np.exp(-risk_score)) > 0.5).astype(int)
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