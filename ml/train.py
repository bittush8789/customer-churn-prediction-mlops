import os
import shutil
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score
from xgboost import XGBClassifier, XGBRegressor

# Import preprocessor
from preprocess import ChurnPreprocessor

def main():
    print("Starting ML Pipeline Training...")
    
    # 1. Ensure raw/processed data paths
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    raw_src = "data/raw/customer_churn.csv"
    dest_csv = "data/customer_churn.csv"
    
    if not os.path.exists(dest_csv):
        if os.path.exists(raw_src):
            shutil.copy(raw_src, dest_csv)
            print(f"Copied dataset to {dest_csv}")
        else:
            print("No source dataset found. Please generate synthetic data first.")
            return

    # Load dataset
    df = pd.read_csv(dest_csv)
    
    # 2. Initialize and Fit Preprocessor
    print("Fitting Preprocessor...")
    preprocessor = ChurnPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    
    # Save processed data to processed_data.csv as requested in directory structure
    df_processed.to_csv("data/processed_data.csv", index=False)
    print("Saved processed_data.csv")
    
    # Define Target & Features
    # Encode target
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df_processed
    
    # We will use RFE to select the top features
    # Select best 10 features using a simple RandomForest classifier
    print("Running feature selection...")
    from sklearn.feature_selection import RFE
    rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)
    rfe_selector.fit(X, y)
    selected_features = X.columns[rfe_selector.support_].tolist()
    print(f"Selected features: {selected_features}")
    
    X_selected = X[selected_features]
    
    # 3. Train Churn Classifier
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Comparing classifiers...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    
    best_clf = None
    best_score = -1
    best_name = ""
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        print(f" - {name} ROC-AUC: {auc:.4f}")
        if auc > best_score:
            best_score = auc
            best_clf = clf
            best_name = name
            
    print(f"Selected Best Classifier: {best_name} (ROC-AUC: {best_score:.4f})")
    
    # 4. Train CLV Regressor
    # Target: CustomerLifetimeValue
    # Feature engineering computed CustomerLifetimeValue, we can extract it from engineered dataset
    df_eng = preprocessor.engineer_features(df)
    y_reg = df_eng["CustomerLifetimeValue"]
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_selected, y_reg, test_size=0.2, random_state=42)
    
    print("Training CLV Regressor models...")
    reg_models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42)
    }
    
    best_reg = None
    best_r2 = -1
    best_reg_name = ""
    
    for name, reg in reg_models.items():
        reg.fit(X_train_reg, y_train_reg)
        preds = reg.predict(X_test_reg)
        r2 = r2_score(y_test_reg, preds)
        print(f" - {name} R2 Score: {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_reg = reg
            best_reg_name = name
            
    print(f"Selected Best Regressor: {best_reg_name} (R2: {best_r2:.4f})")
    
    # 5. Save all model artifacts to ml/model.pkl
    model_payload = {
        "preprocessor": preprocessor,
        "classifier": best_clf,
        "regressor": best_reg,
        "selected_features": selected_features,
        "classifier_name": best_name,
        "regressor_name": best_reg_name
    }
    
    with open("ml/model.pkl", "wb") as f:
        pickle.dump(model_payload, f)
        
    print("Saved comprehensive model artifacts to ml/model.pkl")

if __name__ == "__main__":
    main()
