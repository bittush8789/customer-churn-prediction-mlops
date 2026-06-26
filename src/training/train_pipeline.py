import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Custom modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ingestion.ingestion import DataIngestion
from feature_engineering.feature_pipeline import FeaturePipeline
from preprocessing.preprocessing import PreprocessorPipeline

def run_ml_pipeline():
    print("=========================================")
    print("Initializing Production Training Pipeline")
    print("=========================================")
    
    # 1. Ingestion
    ingestor = DataIngestion("data/customer_churn.csv")
    df_raw = ingestor.load_data()
    
    # 2. Feature Engineering
    fe_pipeline = FeaturePipeline()
    df_eng = fe_pipeline.fit_transform(df_raw)
    
    # 3. Preprocessing
    preprocessor = PreprocessorPipeline()
    df_processed = preprocessor.fit_transform(df_eng)
    
    # Save processed dataframe for verification
    df_processed.to_csv("data/processed_data.csv", index=False)
    print("Saved data/processed_data.csv")
    
    # Define Target & Features
    y = df_eng["Churn"].map({"Yes": 1, "No": 0})
    X = df_processed
    
    # Feature Selection (Select K Best or correlation threshold)
    # Drop highly correlated variables
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
    X_selected = X.drop(columns=to_drop)
    selected_features = X_selected.columns.tolist()
    print(f"Features selected for classification: {selected_features}")
    
    # 4. Handle Class Imbalance via SMOTE
    print(f"Original target ratio: Churn=Yes ({(y==1).sum()}), Churn=No ({(y==0).sum()})")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)
    print(f"Resampled target ratio: Churn=Yes ({(y_resampled==1).sum()}), Churn=No ({(y_resampled==0).sum()})")
    
    # 5. Stratified 5-Fold Cross Validation comparison
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_seed=42)
    }
    
    comparison_results = []
    
    # MLflow experiment setup
    mlflow.set_experiment("Subscribers Churn Analytics")
    
    best_mean_auc = -1
    best_model_name = ""
    best_base_model = None
    
    print("\nStarting 5-Fold Stratified CV Model Evaluations...")
    for name, model in models.items():
        acc_scores, f1_scores, auc_scores = [], [], []
        prec_scores, rec_scores = [], []
        
        # We run cross-validation on balanced sampling (SMOTE applied to train folds to avoid leakage)
        for train_idx, val_idx in skf.split(X_selected, y):
            X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Apply SMOTE only on training fold
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            
            # Train
            model.fit(X_tr_sm, y_tr_sm)
            
            # Predict
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else preds
            
            acc_scores.append(accuracy_score(y_val, preds))
            f1_scores.append(f1_score(y_val, preds))
            prec_scores.append(precision_score(y_val, preds, zero_division=0))
            rec_scores.append(recall_score(y_val, preds, zero_division=0))
            auc_scores.append(roc_auc_score(y_val, probs))
            
        mean_acc = np.mean(acc_scores)
        mean_f1 = np.mean(f1_scores)
        mean_prec = np.mean(prec_scores)
        mean_rec = np.mean(rec_scores)
        mean_auc = np.mean(auc_scores)
        
        print(f" - {name:20s} | Acc: {mean_acc:.3f} | Prec: {mean_prec:.3f} | Rec: {mean_rec:.3f} | F1: {mean_f1:.3f} | AUC: {mean_auc:.3f}")
        
        comparison_results.append({
            "Model": name,
            "Accuracy": mean_acc,
            "Precision": mean_prec,
            "Recall": mean_rec,
            "F1": mean_f1,
            "ROC-AUC": mean_auc
        })
        
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_model_name = name
            best_base_model = model
            
    # Save comparison table
    df_compare = pd.DataFrame(comparison_results).set_index("Model")
    os.makedirs("reports", exist_ok=True)
    df_compare.to_csv("reports/churn_report.csv")
    print("\nSaved models comparison report to reports/churn_report.csv")
    
    # 6. Hyperparameter Optimization on the selected best classifier
    print(f"\nFine-Tuning Hyperparameters for {best_model_name}...")
    
    # Setup search grids based on best model choice
    param_grids = {
        "Logistic Regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"]
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },
        "XGBoost": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "LightGBM": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "num_leaves": [15, 31, 63]
        },
        "CatBoost": {
            "iterations": [100, 150],
            "learning_rate": [0.05, 0.1],
            "depth": [4, 6]
        }
    }
    
    grid = param_grids.get(best_model_name, {"C": [1.0]})
    search = RandomizedSearchCV(
        estimator=best_base_model,
        param_distributions=grid,
        n_iter=5,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit random search on the full SMOTE resampled training data
    search.fit(X_resampled, y_resampled)
    best_clf = search.best_estimator_
    print(f"Optimal parameters: {search.best_params_}")
    
    # 7. Train CLV Regressor Model
    print("\nTraining CLV Regressor...")
    y_reg = df_eng["CustomerLifetimeValue"]
    regressor = XGBRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_selected, y_reg)
    reg_r2 = r2_score(y_reg, regressor.predict(X_selected))
    print(f"CLV Regressor Fitted. R2 Score: {reg_r2:.4f}")
    
    # 8. Log run metrics & model artifacts via MLflow
    print("\nLogging best model metrics into MLflow...")
    with mlflow.start_run(run_name=f"{best_model_name}_optimized") as run:
        mlflow.log_params(search.best_params_)
        
        # Log CV comparison scores for this model
        best_metrics = df_compare.loc[best_model_name].to_dict()
        mlflow.log_metrics(best_metrics)
        mlflow.log_metric("clv_reg_r2", reg_r2)
        
        # Log models
        mlflow.sklearn.log_model(best_clf, "best_churn_classifier")
        print(f"MLflow Run tracked: {run.info.run_id}")
        
    # 9. Model Persistence
    os.makedirs("models", exist_ok=True)
    
    # Save target files requested: model.pkl, preprocessor.pkl, feature_pipeline.pkl
    # model.pkl: bundles best classifier, regressor, and selected features list
    model_payload = {
        "classifier": best_clf,
        "regressor": regressor,
        "selected_features": selected_features,
        "classifier_name": best_model_name,
        "metrics": best_metrics
    }
    
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model_payload, f)
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open("models/feature_pipeline.pkl", "wb") as f:
        pickle.dump(fe_pipeline, f)
        
    print("\nModel Persistence completed.")
    print("Saved models/model.pkl, models/preprocessor.pkl, models/feature_pipeline.pkl")
    
if __name__ == "__main__":
    run_ml_pipeline()
