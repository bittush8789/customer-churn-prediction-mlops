import os
import sys
import unittest
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from ml.preprocess import ChurnPreprocessor
from backend.utils import get_segment, RecommendationEngine

class ChurnPlatformUnitTests(unittest.TestCase):
    def setUp(self):
        # Create a tiny mock dataframe
        self.mock_df = pd.DataFrame({
            "CustomerID": ["C-001", "C-002"],
            "Gender": ["Male", "Female"],
            "Age": [30, 60],
            "Tenure": [5, 48],
            "MonthlyCharges": [50.0, 100.0],
            "TotalCharges": [250.0, 4800.0],
            "ContractType": ["Month-to-month", "Two year"],
            "InternetService": ["DSL", "Fiber optic"],
            "PaymentMethod": ["Electronic check", "Credit card"],
            "SupportTickets": [4, 0],
            "AverageUsageHours": [120.0, 240.0]
        })
        self.preprocessor = ChurnPreprocessor()

    def test_feature_engineering(self):
        engineered = self.preprocessor.engineer_features(self.mock_df)
        self.assertIn("CustomerLifetimeValue", engineered.columns)
        self.assertIn("RevenuePerCustomer", engineered.columns)
        self.assertIn("ComplaintFrequency", engineered.columns)
        self.assertIn("UsageScore", engineered.columns)
        self.assertIn("TenureCategory", engineered.columns)
        
        # Check specific values
        self.assertEqual(engineered.loc[0, "TenureCategory"], "New")
        self.assertEqual(engineered.loc[1, "TenureCategory"], "Long-term")

    def test_preprocessor_fit_transform(self):
        processed = self.preprocessor.fit_transform(self.mock_df)
        # Expected outputs include scaling and encoding
        self.assertIsNotNone(processed)
        self.assertTrue(len(processed.columns) > 0)

    def test_segment_classification(self):
        cust_new = {"Tenure": 6, "SupportTickets": 1, "MonthlyCharges": 40}
        cust_at_risk = {"Tenure": 24, "SupportTickets": 4, "MonthlyCharges": 90}
        cust_high_val = {"Tenure": 24, "SupportTickets": 0, "MonthlyCharges": 95}
        cust_loyal = {"Tenure": 36, "SupportTickets": 1, "MonthlyCharges": 50}
        
        self.assertEqual(get_segment(cust_new), "New Customers")
        self.assertEqual(get_segment(cust_at_risk), "At Risk Customers")
        self.assertEqual(get_segment(cust_high_val), "High Value Customers")
        self.assertEqual(get_segment(cust_loyal), "Loyal Customers")

    def test_recommendation_rules(self):
        engine = RecommendationEngine()
        cust = {
            "CustomerLifetimeValue": 2500,
            "MonthlyCharges": 95.0,
            "SupportTickets": 4,
            "ContractType": "Month-to-month"
        }
        recs = engine.generate(cust, 0.85)
        actions = [r["Action"] for r in recs]
        
        self.assertIn("Premium retention discount", actions)
        self.assertIn("Enroll in VIP Loyalty program", actions)
        self.assertIn("Supervisor queue routing", actions)

if __name__ == '__main__':
    unittest.main()
