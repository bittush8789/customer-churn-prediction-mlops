# Customer Churn Prediction Platform

An end-to-end web application and machine learning engine designed to predict telecom customer churn risk, forecast customer lifetime value (CLV), segment users, and generate automated retention rules.

---

## Directory Structure

```text
customer-churn-prediction-platform/
│
├── frontend/
│   ├── index.html                   # Landing page
│   ├── dashboard.html               # KPI metrics and Chart.js analytics dashboard
│   ├── predict.html                 # Score new customer risk and recommendations form
│   ├── reports.html                 # Past customer prediction logs
│   ├── css/
│   │   └── style.css                # Premium styling (Outfit/Plus Jakarta fonts, grids, cards)
│   └── js/
│       ├── app.js                   # Handles event listening, API fetches, form binding
│       └── charts.js                # Configurations for Chart.js rendering
│
├── backend/
│   ├── app.py                       # Flask Web API, serving endpoints & static frontend
│   ├── predict.py                   # Loads model.pkl and executes inference
│   └── utils.py                     # Retention recommendations and aggregate stats
│
├── ml/
│   ├── train.py                     # Trains classifiers/regressors, outputs model.pkl
│   ├── preprocess.py                # Handles imputations, scaling, and feature engineering
│   └── model.pkl                    # Bundled preprocessors, classifier, and regressor
│
├── data/
│   ├── customer_churn.csv           # Raw dataset
│   └── processed_data.csv           # Imputed, scaled, and encoded dataframe
│
├── reports/
│   └── churn_report.csv             # History of scored predictions (audit logs)
│
├── tests/
│   ├── test_api.py                  # Integration and unit tests for Flask API
│   └── test_unit.py                 # Core ML logic and engine unit tests
│
├── requirements.txt                 # Dependencies
├── Dockerfile                       # Multi-stage image containerizing the platform
├── .gitignore                       # Git patterns to ignore caches and outputs
└── README.md                        # Documentation
```

---

## Local Setup & Quick Start

### 1. Installation
Install the python requirements:
```bash
pip install -r requirements.txt
```

### 2. Model Training
Fit preprocessors and train model classifiers/regressors to generate `ml/model.pkl`:
```bash
python ml/train.py
```

### 3. Running Tests
Validate Flask REST endpoints and ML utility calculations:
```bash
# Run API Integration tests
python tests/test_api.py

# Run Core ML Unit tests
python tests/test_unit.py
```

### 4. Running the Web Platform
Launch the Flask development server:
```bash
python backend/app.py
```
Open `http://localhost:5000` in your web browser to browse the platform.

---

## Containerized Deployment (Docker)

To build and run using Docker:
```bash
# Build the container (automatically trains models)
docker build -t churn-platform .

# Run the container
docker run -p 5000:5000 churn-platform
```
Open `http://localhost:5000` to interact.
