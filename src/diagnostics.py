import os
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from engine import prepare_and_scale

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline_diagnostics():
    """Generates standalone blind validation inputs to verify classification data accuracy."""
    try:
        model = joblib.load("models/attrition_gbm_model.pkl")
    except Exception as e:
        logger.error(f"Binaries missing from path. Run engine.py first. Trace: {e}")
        return
        
    # Build validation data using a completely separate seed to ensure honest checking
    np.random.seed(99)
    depts = ["Engineering", "Sales", "HR", "Marketing", "Finance"]
    test_df = pd.DataFrame({
        "Department": np.random.choice(depts, size=100),
        "Base_Salary": np.random.randint(30000, 140000, size=100).astype(float),
        "Years_At_Company": np.random.randint(0, 15, size=100),
        "Satisfaction_Score": np.random.randint(1, 6, size=100),
        "Avg_Monthly_Overtime_Hours": np.random.uniform(0.0, 80.0, size=100),
        "Comp_Ratio": np.random.uniform(0.6, 1.4, size=100)
    })
    
    # Mathematical ground truth rule generation
    risk_score = ((test_df["Avg_Monthly_Overtime_Hours"] / 35.0) * 1.5 + (1.2 - test_df["Comp_Ratio"]) * 2.0 - (test_df["Satisfaction_Score"] * 0.8))
    y_true = (1 / (1 + np.exp(-risk_score)) > 0.5).astype(int)
    
    # Process inputs without retraining scaler parameters
    X_scaled = prepare_and_scale(test_df, is_train=False)
    y_pred = model.predict(X_scaled)
    
    # Output testing reports
    print("\nCONSOLIDATED ENGINE DIAGNOSTICS:")
    print(f"Accuracy Metric Score: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    run_pipeline_diagnostics()