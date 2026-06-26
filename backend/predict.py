import os
import sys
import pickle
import pandas as pd
import numpy as np

# Add ml folder to path so unpickler can locate preprocess modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import RecommendationEngine, get_segment

class ChurnPredictorAPI:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "model.pkl"))
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run ml/train.py first.")
            
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
            
        self.preprocessor = payload["preprocessor"]
        self.classifier = payload["classifier"]
        self.regressor = payload["regressor"]
        self.selected_features = payload["selected_features"]
        
        self.re = RecommendationEngine()

    def predict(self, customer_data: dict) -> dict:
        """Executes full prediction pipeline for a single customer profile."""
        df = pd.DataFrame([customer_data])
        
        # 1. Feature Engineering & Preprocess
        df_processed = self.preprocessor.transform(df)
        
        # 2. Select Features (handling dummy columns that might not exist in OHE step)
        X_selected = pd.DataFrame(0.0, index=[0], columns=self.selected_features)
        for col in self.selected_features:
            if col in df_processed.columns:
                X_selected[col] = df_processed[col].values
                
        # 3. Predict Churn Probability
        churn_prob = float(self.classifier.predict_proba(X_selected)[0, 1])
        churn_pred = "Yes" if churn_prob >= 0.5 else "No"
        
        risk_level = "Low"
        if churn_prob >= 0.8:
            risk_level = "High"
        elif churn_prob >= 0.5:
            risk_level = "Medium"
            
        # 4. Predict CLV
        predicted_clv = float(self.regressor.predict(X_selected)[0])
        revenue_loss = churn_prob * predicted_clv
        
        # 5. Segment & Recommendation
        segment = get_segment(customer_data)
        
        # Attach engineered fields to recommendations input
        engineered_row = self.preprocessor.engineer_features(df).iloc[0].to_dict()
        rec_input = {**customer_data, **engineered_row}
        recommendations = self.re.generate(rec_input, churn_prob)
        
        return {
            "CustomerID": customer_data.get("CustomerID", "UNKNOWN"),
            "ChurnProbability": round(churn_prob, 4),
            "ChurnPrediction": churn_pred,
            "RiskLevel": risk_level,
            "PredictedCLV": round(predicted_clv, 2),
            "EstimatedRevenueLoss": round(revenue_loss, 2),
            "CustomerSegment": segment,
            "Recommendations": recommendations
        }
