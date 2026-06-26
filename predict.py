import os
import sys
import pickle
import pandas as pd

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.explainability.explain import ChurnExplainer
from src.utils.retention import BusinessRetentionEngine

class ProductionPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        
        # Load artifacts
        with open(os.path.join(models_dir, "model.pkl"), "rb") as f:
            self.model_payload = pickle.load(f)
        with open(os.path.join(models_dir, "preprocessor.pkl"), "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(os.path.join(models_dir, "feature_pipeline.pkl"), "rb") as f:
            self.feature_pipeline = pickle.load(f)
            
        self.classifier = self.model_payload["classifier"]
        self.regressor = self.model_payload["regressor"]
        self.selected_features = self.model_payload["selected_features"]
        
        # Setup Explainer
        # Create tiny training sample for SHAP
        df_dummy = pd.DataFrame(0.0, index=[0], columns=self.selected_features)
        self.explainer = ChurnExplainer(self.classifier, df_dummy)
        
        # Setup Retention Engine
        self.retention_engine = BusinessRetentionEngine()

    def predict(self, customer_data: dict) -> dict:
        """Runs end-to-end production model inference on a raw customer dictionary."""
        df = pd.DataFrame([customer_data])
        
        # 1. Feature Engineering
        df_eng = self.feature_pipeline.transform(df)
        
        # 2. Preprocess
        df_proc = self.preprocessor.transform(df_eng)
        
        # 3. Selected Feature Alignment
        X_selected = pd.DataFrame(0.0, index=[0], columns=self.selected_features)
        for col in self.selected_features:
            if col in df_proc.columns:
                X_selected[col] = df_proc[col].values
                
        # 4. Predict Churn Probability & Class
        prob = float(self.classifier.predict_proba(X_selected)[0, 1])
        
        # 5. Predict CLV
        clv = float(self.regressor.predict(X_selected)[0])
        
        # 6. Evaluate Business Metrics & Retention Strategy
        metrics = self.retention_engine.evaluate_customer(customer_data, prob, clv)
        
        # 7. Extract Top Churn Drivers using SHAP
        top_drivers = self.explainer.explain_instance(X_selected)
        
        return {
            "churn_probability": round(prob, 4),
            "risk_level": metrics["risk_level"],
            "predicted_clv": round(clv, 2),
            "estimated_revenue_loss": metrics["estimated_revenue_loss"],
            "top_churn_drivers": top_drivers,
            "retention_recommendations": metrics["recommendations"],
            # Extra keys for dashboard auditing
            "health_score": metrics["health_score"],
            "retention_cost": metrics["retention_cost"],
            "retention_roi": metrics["retention_roi"]
        }

if __name__ == "__main__":
    # verification test
    pass
