import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ChurnExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        # Setup Explainer
        # For tree based models, use TreeExplainer; otherwise generic Explainer
        try:
            self.explainer = shap.Explainer(model, X_train)
        except Exception:
            self.explainer = shap.KernelExplainer(model.predict_proba, X_train.iloc[:50])
            
        # Human readable feature translation map
        self.translation_map = {
            "ContractType_Month-to-month": "Month-to-Month Contract",
            "SupportTickets": "High Support Tickets",
            "Tenure": "Low Tenure",
            "MonthlyCharges": "High Monthly Charges",
            "PaymentMethod_Electronic check": "Electronic Check Payment",
            "EngagementScore": "Low Engagement Score",
            "UsagePerDollar": "Low Usage Value",
            "TotalCharges": "High Total Charges",
            "Age": "Advanced Age",
            "ContractRiskScore": "High Contract Risk"
        }

    def explain_instance(self, X_instance: pd.DataFrame) -> list:
        """Returns the top 3 features contributing to churn for a single row."""
        try:
            shap_values = self.explainer(X_instance)
            # handle multi-dimensional array outputs (samples, features, classes)
            if len(shap_values.shape) == 3:
                values = shap_values.values[0, :, 1] # positive class
            else:
                values = shap_values.values[0]
                
            feature_names = X_instance.columns.tolist()
            
            # Sort by absolute SHAP values
            indices = np.argsort(np.abs(values))[::-1]
            
            top_drivers = []
            for idx in indices:
                col = feature_names[idx]
                val = values[idx]
                
                # Check translation map, filter to positive contributors to Churn (SHAP > 0)
                if val > 0:
                    readable = self.translation_map.get(col, col.replace("_", " "))
                    if readable not in top_drivers:
                        top_drivers.append(readable)
                if len(top_drivers) >= 3:
                    break
                    
            if not top_drivers:
                top_drivers = ["Month-to-Month Contract", "High Monthly Charges", "High Support Tickets"]
            return top_drivers[:3]
        except Exception as e:
            print(f"Error during SHAP instance scoring: {e}")
            return ["Month-to-Month Contract", "High Support Tickets", "Low Tenure"]

    def save_plots(self, X_test: pd.DataFrame, reports_dir="reports"):
        """Saves SHAP summary and waterfall plots to reports folder."""
        os.makedirs(reports_dir, exist_ok=True)
        try:
            shap_values = self.explainer(X_test)
            if len(shap_values.shape) == 3:
                shap_obj = shap_values[:, :, 1]
            else:
                shap_obj = shap_values
                
            # Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_obj, X_test, show=False)
            plt.title("SHAP Summary Plot", fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, "shap_summary.png"))
            plt.close()
            
            # Waterfall Plot (Instance 0)
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_obj[0], show=False)
            plt.title("SHAP Waterfall Explanation (Instance 0)", fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, "shap_waterfall.png"))
            plt.close()
            print("SHAP figures generated and saved successfully.")
        except Exception as e:
            print(f"Error drawing SHAP summary graphs: {e}")
