import os
import sys
import unittest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.api.api import app

class ChurnPlatformFastAPITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)
        self.assertIn("Model", data[0])

    def test_predict_endpoint(self):
        test_payload = {
            "Gender": "Male",
            "Age": 35,
            "Tenure": 10,
            "MonthlyCharges": 70.0,
            "TotalCharges": 700.0,
            "ContractType": "One year",
            "InternetService": "DSL",
            "PaymentMethod": "Credit card",
            "SupportTickets": 1,
            "AverageUsageHours": 180.0
        }
        response = self.client.post(
            "/predict",
            json=test_payload
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("churn_probability", data)
        self.assertIn("risk_level", data)
        self.assertIn("predicted_clv", data)
        self.assertIn("top_churn_drivers", data)
        self.assertIn("retention_recommendations", data)

if __name__ == '__main__':
    unittest.main()
