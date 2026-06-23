# 📊 Workforce Attrition Analytics & Risk Inference Pipeline

An enterprise-grade, containerized Machine Learning Engineering (MLE) and analytics pipeline designed to detect, analyze, and predict employee flight risks. By shifting away from manual spreadsheets and siloed local code, this system establishes a self-healing, automated architecture that pairs a predictive Machine Learning layer with deterministic business logic overrides.

---

# 🌟 Key Features

## 🧠 Gradient Boosting Intelligence
High-fidelity operational attrition modeling using an optimized Scikit-Learn GradientBoostingClassifier framework.

## 🎛️ Hybrid Inference Logic
Combines statistical probability metrics with strict, deterministic business rule overrides to catch high-risk telemetry blind spots.

## 🐳 Containerized Isolation
Multi-container architecture deployed natively through Docker Compose to separate processing and interface pathways.

## 🔄 Automated Telemetry Loop
Asynchronous transmission pipelines that log all incoming payload inferences directly to a persistence tier for audit tracking.

## 📊 Comprehensive Reporting
- Interactive UI Metrics: Live, clean interactive inputs via Streamlit to visually score flight risks instantly.
- Persistence Audits: Real-time persistence sync directly into a cloud infrastructure database.

---

# 🛠️ Technology Stack

| Layer | Technology |
|------|-------------|
| Backend | FastAPI, Uvicorn, Python 3.11-slim |
| Frontend | Streamlit Framework, Plotly Core |
| Intelligence | Scikit-Learn 1.5.0, NumPy, Pandas |
| Persistence | MongoDB Atlas Cloud Persistence Cluster |
| Deployment | Docker, Docker Compose Orchestration |

---

# 📋 Installation & Setup

## 1. Prerequisites
- Docker Desktop installed and running
- MongoDB Atlas Cloud Database URI string

---

## 2. Environment Configuration
Create a .env file in your root folder workspace directory containing your persistence credentials:

```env
MONGO_URI=your_actual_mongodb_atlas_connection_string_here
```
---

## 3. Launch Containerized Services
Compile and execute your isolated environment services using the unified compose orchestrator:

```bash
# Clean active instances and spin up services detached
docker compose down
docker compose up --build -d
```
---

## 4. Calibrate and Train Model Binaries
To completely eliminate package compilation version leaks, execute the training execution pipeline directly inside the active container runtime:
```bash
docker compose exec backend python src/engine.py
```
---

## 5. Accessing the Applications

Once the initialization parameters settle, operations are accessible across the following endpoints:
```text
- Interactive Frontend UI Dashboard: http://localhost:8501
- FastAPI Programmatic Swagger Docs: http://localhost:8000/docs
```
---

# 📂 Project Structure
```text
Workforce_Attrition/
├── dashboard/
│   └── Dockerfile             # Streamlit optimized multi-stage container file
│
├── models/
│   ├── attrition_gbm_model.pkl # Frozen Gradient Boosting weights
│   ├── encoder_dept.pkl       # Category mapping binaries
│   └── scaler.pkl             # Fitted standardization boundaries
│
├── src/
│   ├── app.py                 # FastAPI microservice routing backend
│   ├── dashboard.py           # Streamlit UI visual mapping script
│   ├── diagnostics.py         # Standalone blind validation test harness
│   ├── engine.py              # Synthetic data simulator & core training pipeline
│   └── predict.py             # Hybrid override inference logic controller
│
├── .env                       # Cloud database global variables (Hidden)
├── Dockerfile                 # Unified Python backend environment definition
├── docker-compose.yml         # Network orchestration manifest
└── requirements.txt           # Verified package dependency manifest
```
---

# 🛡️ Security & Privacy

## 🔐 API Protection
The configuration .env file containing database validation strings is isolated from your commits via explicit repository exclusions.

## 💾 Transient Calculations
User metrics used during runtime scoring transactions are passed through local memory variables and never persisted on unencrypted physical disk states.

---

# 📄 License

This project is licensed under the MIT License.  
See the `LICENSE` file for details.