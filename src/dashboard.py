import os
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from pymongo import MongoClient

# Configure full widescreen layout rules
st.set_page_config(page_title="Workforce Retention Analytics Engine", layout="wide")

MONGO_URI = os.getenv("MONGO_URI")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

@st.cache_data(ttl=5)
def fetch_cloud_telemetry_history() -> pd.DataFrame:
    """Connects to MongoDB and flattens JSON documents into a viewable table."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        db = client["workforce_db"]
        collection = db["prediction_logs"]
        cursor = list(collection.find())
        if not cursor:
            return pd.DataFrame()
            
        flattened_list = []
        for doc in cursor:
            record = {
                "Department": doc.get("employee_inputs", {}).get("Department"),
                "Base_Salary": doc.get("employee_inputs", {}).get("Base_Salary"),
                "Years_At_Company": doc.get("employee_inputs", {}).get("Years_At_Company"),
                "Satisfaction_Score": doc.get("employee_inputs", {}).get("Satisfaction_Score"),
                "Overtime_Hours": doc.get("employee_inputs", {}).get("Avg_Monthly_Overtime_Hours"),
                "Comp_Ratio": doc.get("employee_inputs", {}).get("Comp_Ratio"),
                "Risk_Classification": doc.get("prediction_outputs", {}).get("risk_classification"),
                "Probability_Str": doc.get("prediction_outputs", {}).get("attrition_probability")
            }
            if record["Probability_Str"]:
                record["Attrition_Probability_%"] = float(record["Probability_Str"].replace("%", ""))
            flattened_list.append(record)
        return pd.DataFrame(flattened_list)
    except Exception as e:
        return pd.DataFrame()

st.title("Workforce Attrition Analytics Portal")
# Split the visual layouts using standard top alignment bars
tab_inference, tab_analytics = st.tabs(["Real-Time Inference Calculator", "Historical Cloud Trends"])

with tab_inference:
    st.subheader("Strategic Talent Retention & Risk Inference Engine")
    with st.form("inference_input_form"):
        col_left, col_right = st.columns(2)
        with col_left:
            dept = st.selectbox("Department Assignment", ["Engineering", "Sales", "HR", "Marketing", "Finance"])
            salary = st.number_input("Annual Base Salary Gross ($)", min_value=10000.0, value=65000.0, step=5000.0)
            tenure = st.slider("Tenure Duration (Years at Organization)", 0, 20, 3)
        with col_right:
            satisfaction = st.slider("Job Satisfaction Metric Score (1-5)", 1, 5, 4)
            overtime = st.slider("Average Monthly Tracked Overtime Hours", 0.0, 80.0, 20.5)
            compa = st.slider("Market Compa-Ratio Assessment", 0.5, 1.5, 0.9)
        submit_btn = st.form_submit_button("Calculate Retention Risk Assessment", use_container_width=True)

    if submit_btn:
        payload = {"Department": dept, "Base_Salary": salary, "Years_At_Company": tenure, "Satisfaction_Score": satisfaction, "Avg_Monthly_Overtime_Hours": overtime, "Comp_Ratio": compa}
        try:
            # Hit the backend container networking router path
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)
            if response.status_code == 200:
                res_data = response.json()
                status_msg = res_data['risk_classification']
                
                if "HIGH" in status_msg: st.error(f"Status: {status_msg}")
                elif "MEDIUM" in status_msg: st.warning(f"Status: {status_msg}")
                else: st.success(f"Status: {status_msg}")
                
                c1, c2 = st.columns(2)
                c1.metric("Statistical Attrition Probability", res_data['attrition_probability'])
                c2.metric("Calculated Workload Exploitation Index", res_data['computed_exploitation_index'])
            else:
                st.error("API failed to score data packet.")
        except Exception as e:
            st.error(f"Backend offline or unreachable: {e}")

with tab_analytics:
    st.subheader("Enterprise-Wide Risk Audit Log")
    df_history = fetch_cloud_telemetry_history()
    if df_history.empty:
        st.info("No logs found in cloud database yet.")
    else:
        st.metric("Total Evaluation Sweeps Handled", len(df_history))
        
        # Render charts layout components natively fitting container grid walls
        fig_bar = px.histogram(df_history, x="Department", color="Risk_Classification", barmode="group", title="Risk Split by Department", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        fig_scatter = px.scatter(df_history, x="Overtime_Hours", y="Attrition_Probability_%", color="Risk_Classification", size="Base_Salary", hover_data=["Years_At_Company"], title="Overtime vs Risk")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        with st.expander("View Complete Database Records Table"):
            # Adjusted explicitly to use container formatting options inside Linux environments
            st.dataframe(df_history, use_container_width=True)