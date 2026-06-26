import os
import csv
import sys
from flask import Flask, request, jsonify, send_from_directory

# Resolve paths
sys.path.append(os.path.dirname(__file__))

from predict import ChurnPredictorAPI
from utils import get_dashboard_stats

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# Initialize Predictor
predictor = None
try:
    predictor = ChurnPredictorAPI()
except Exception as e:
    print(f"Warning: Could not load model: {e}. Make sure to run ml/train.py first.")

# Create reports audit CSV path
AUDIT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "churn_report.csv"))
os.makedirs(os.path.dirname(AUDIT_CSV), exist_ok=True)

# Ensure file exists with headers if not created
if not os.path.exists(AUDIT_CSV):
    with open(AUDIT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CustomerID", "ChurnProbability", "ChurnPrediction", "RiskLevel", "PredictedCLV", "EstimatedRevenueLoss", "CustomerSegment"])

# Static HTML routes
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/dashboard.html")
def dashboard_page():
    return send_from_directory(app.static_folder, "dashboard.html")

@app.route("/predict.html")
def predict_page():
    return send_from_directory(app.static_folder, "predict.html")

@app.route("/reports.html")
def reports_page():
    return send_from_directory(app.static_folder, "reports.html")

# API endpoints
@app.route("/api/dashboard-stats", methods=["GET"])
def api_dashboard_stats():
    try:
        stats = get_dashboard_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded. Please train models."}), 500
        
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400
        
    try:
        # Perform inference
        res = predictor.predict(data)
        
        # Save to reports/churn_report.csv for audit history
        with open(AUDIT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                res["CustomerID"],
                res["ChurnProbability"],
                res["ChurnPrediction"],
                res["RiskLevel"],
                res["PredictedCLV"],
                res["EstimatedRevenueLoss"],
                res["CustomerSegment"]
            ])
            
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/reports", methods=["GET"])
def api_get_reports():
    try:
        reports = []
        if os.path.exists(AUDIT_CSV):
            with open(AUDIT_CSV, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    reports.append(row)
        # return reversed to show latest first
        return jsonify(list(reversed(reports))[:50]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
