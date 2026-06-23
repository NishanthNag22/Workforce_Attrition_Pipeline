import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(BASE_DIR, "models")

def run_standalone_inference(user_inputs: dict) -> dict:
    """Takes user inputs from the UI web form, matches the model features, and runs the prediction."""
    
    # Turn raw inputs into a basic single-row DataFrame
    raw_df = pd.DataFrame([{
        "Base_Salary": user_inputs["Base_Salary"],
        "Years_At_Company": user_inputs["Years_At_Company"],
        "Satisfaction_Score": user_inputs["Satisfaction_Score"],
        "Avg_Monthly_Overtime_Hours": user_inputs["Avg_Monthly_Overtime_Hours"],
        "Comp_Ratio": user_inputs["Comp_Ratio"]
    }])
    
    # Read the column structure we saved during model training
    features_list = joblib.load(os.path.join(models_path, "model_features.pkl"))
    
    # Set the department columns manually based on what the user picked
    for col in features_list:
        if col.startswith("Department_"):
            current_dept = col.split("Department_")[1]
            raw_df[col] = 1 if user_inputs["Department"].lower() == current_dept.lower() else 0
            
    # Force the DataFrame columns to match the training feature list order exactly
    final_df = raw_df.reindex(columns=features_list, fill_value=0)
    
    # Scale inputs using our pre-saved scaler weights
    scaler = joblib.load(os.path.join(models_path, "feature_scaler.pkl"))
    scaled_data = scaler.transform(final_df)
    
    # Load model and score the risk probability
    model = joblib.load(os.path.join(models_path, "attrition_gbm_model.pkl"))
    ml_probability = float(model.predict_proba(scaled_x := scaled_data)[0][1])
    
    # Calculate custom exploitation indicator logic
    overtime_ratio = user_inputs["Avg_Monthly_Overtime_Hours"] / 40.0
    underpay_margin = 1.0 - user_inputs["Comp_Ratio"]
    exploitation_index = max(0.0, overtime_ratio * underpay_margin * 100)
    
    # Check simple business logic flags
    bad_satisfaction = user_inputs["Satisfaction_Score"] <= 2
    high_overtime = user_inputs["Avg_Monthly_Overtime_Hours"] >= 50.0
    low_pay = user_inputs["Comp_Ratio"] < 0.8
    
    # Combine the ML model score with direct condition overrides
    if ml_probability >= 0.70 or (bad_satisfaction and (high_overtime or low_pay)):
        risk_label = "HIGH RISK (Flight Threat)"
        final_prob = max(ml_probability, 0.785)
    elif ml_probability >= 0.35 or bad_satisfaction or (high_overtime and user_inputs["Comp_Ratio"] <= 1.0):
        risk_label = "MEDIUM RISK"
        final_prob = max(ml_probability, 0.462)
    else:
        risk_label = "LOW RISK"
        final_prob = min(ml_probability, 0.241)
        
    return {
        "risk_classification": risk_label,
        "attrition_probability": f"{final_prob * 100:.2f}%",
        "computed_exploitation_index": round(exploitation_index, 2)
    }