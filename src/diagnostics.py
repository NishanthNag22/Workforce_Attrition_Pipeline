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
        
    np.random.seed(99)
    depts = ["Engineering", "Sales", "HR", "Marketing", "Finance"]
    
    # 1. Re-generate test data matching your updated engine constraints
    departments = np.random.choice(depts, size=100)
    years_at_company = np.random.randint(0, 15, size=100)
    satisfaction_scores = np.random.randint(1, 6, size=100)
    overtime_hours = np.random.uniform(0.0, 80.0, size=100)
    
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
        salary = np.random.normal(loc=market_avg, scale=15000.0)
        salary = np.clip(salary, 30000.0, 150000.0)
        base_salaries.append(round(salary, 2))
        comp_ratios.append(round(salary / market_avg, 2))
        
    test_df = pd.DataFrame({
        "Department": departments,
        "Base_Salary": base_salaries,
        "Years_At_Company": years_at_company,
        "Satisfaction_Score": satisfaction_scores,
        "Avg_Monthly_Overtime_Hours": overtime_hours,
        "Comp_Ratio": comp_ratios
    })
    
    # 2. MATCH THE NEW RISK EQUATION
    overtime_penalty = test_df["Avg_Monthly_Overtime_Hours"] / 35.0
    compensation_buffer = (test_df["Comp_Ratio"] - 1.0) * 3.0
    satisfaction_multiplier = test_df["Satisfaction_Score"] / 5.0
    
    risk_score = (overtime_penalty * 2.0) - compensation_buffer - (satisfaction_multiplier * 3.5)
    
    # Re-apply the deterministic noise block
    np.random.seed(99) # Lock noise generation state
    noise = np.random.normal(0, 0.5, 100)
    final_score = risk_score + noise
    
    y_true = (1 / (1 + np.exp(-final_score)) > 0.5).astype(int)
    
    # Process inputs without retraining scaler parameters
    X_scaled = prepare_and_scale(test_df, is_train=False)
    y_pred = model.predict(X_scaled)
    
    # Output testing reports
    print("\nCONSOLIDATED ENGINE DIAGNOSTICS:")
    print(f"Accuracy Metric Score: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(classification_report(y_true, y_pred))
    
if __name__ == "__main__":
    run_pipeline_diagnostics()