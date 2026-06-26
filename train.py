import os
import pickle
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.training.train_pipeline import run_ml_pipeline
from src.explainability.explain import ChurnExplainer

def main():
    # 1. Execute ML pipeline
    run_ml_pipeline()
    
    # 2. Generate SHAP plots for metrics dashboard
    print("\nGenerating SHAP plots for reports/...")
    try:
        with open("models/model.pkl", "rb") as f:
            payload = pickle.load(f)
        with open("models/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
            
        clf = payload["classifier"]
        selected_features = payload["selected_features"]
        
        # Load processed dataframe
        df_processed = pd.read_csv("data/processed_data.csv")
        X_selected = df_processed[selected_features]
        
        explainer = ChurnExplainer(clf, X_selected)
        explainer.save_plots(X_selected, reports_dir="reports")
        print("SHAP generation finished.")
    except Exception as e:
        print(f"SHAP generation skipped or encountered error: {e}")
        
    # Create copy of the model for CI/CD S3 upload compatibility
    try:
        import shutil
        if os.path.exists("models/model.pkl"):
            shutil.copy("models/model.pkl", "models/churn_model.pkl")
            print("Successfully copied models/model.pkl to models/churn_model.pkl")
    except Exception as e:
        print(f"Error copying model file: {e}")
        
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
