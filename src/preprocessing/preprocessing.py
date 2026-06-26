import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

class PreprocessorPipeline:
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.binary_encoders = {}
        
        # Categorizations
        self.num_cols = [
            "Age", "Tenure", "MonthlyCharges", "TotalCharges", "AverageUsageHours", "SupportTickets",
            "ChargePerMonth", "TicketRatio", "UsagePerDollar", 
            "CustomerValueScore", "ContractRiskScore", "EngagementScore"
        ]
        self.binary_cols = ["Gender"]
        self.cat_cols = ["ContractType", "InternetService", "PaymentMethod", "Segment"]
        self.feature_names = []
        self.outlier_bounds = {}

    def treat_outliers_fit(self, df: pd.DataFrame):
        """Calculates IQR limits for numerical fields."""
        for col in self.num_cols:
            if col in df.columns:
                q25 = df[col].quantile(0.25)
                q75 = df[col].quantile(0.75)
                iqr = q75 - q25
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                self.outlier_bounds[col] = (lower, upper)

    def treat_outliers_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Caps outliers based on calculated bounds."""
        df_capped = df.copy()
        for col, bounds in self.outlier_bounds.items():
            if col in df_capped.columns:
                df_capped[col] = np.clip(df_capped[col], bounds[0], bounds[1])
        return df_capped

    def fit(self, df: pd.DataFrame):
        """Fits core estimators on training input."""
        df_clean = df.copy()
        
        # Impute missing numeric columns
        self.num_imputer.fit(df_clean[self.num_cols])
        imputed_num = self.num_imputer.transform(df_clean[self.num_cols])
        
        # Treat Outliers
        df_imputed_num = pd.DataFrame(imputed_num, columns=self.num_cols, index=df.index)
        self.treat_outliers_fit(df_imputed_num)
        capped_num = self.treat_outliers_transform(df_imputed_num)
        
        # Fit Scaler
        self.scaler.fit(capped_num)
        
        # Impute categoricals
        imputed_cat = self.cat_imputer.fit_transform(df_clean[self.cat_cols + self.binary_cols])
        df_imputed_cat = pd.DataFrame(imputed_cat, columns=self.cat_cols + self.binary_cols, index=df.index)
        
        # Fit LabelEncoder for binary variables
        for col in self.binary_cols:
            le = LabelEncoder()
            le.fit(df_imputed_cat[col])
            self.binary_encoders[col] = le
            
        # Fit One-Hot Encoder on categoricals
        self.ohe.fit(df_imputed_cat[self.cat_cols])
        
        # Compile column names
        ohe_cols = self.ohe.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + self.binary_cols + ohe_cols
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted preprocessors to target dataset."""
        df_clean = df.copy()
        
        # 1. Impute and Scale numericals
        imputed_num = self.num_imputer.transform(df_clean[self.num_cols])
        df_imputed_num = pd.DataFrame(imputed_num, columns=self.num_cols, index=df.index)
        capped_num = self.treat_outliers_transform(df_imputed_num)
        scaled_num = self.scaler.transform(capped_num)
        df_num = pd.DataFrame(scaled_num, columns=self.num_cols, index=df.index)
        
        # 2. Impute and Encode categoricals
        imputed_cat = self.cat_imputer.transform(df_clean[self.cat_cols + self.binary_cols])
        df_imputed_cat = pd.DataFrame(imputed_cat, columns=self.cat_cols + self.binary_cols, index=df.index)
        
        # Label Encoding
        df_bin = pd.DataFrame(index=df.index)
        for col in self.binary_cols:
            le = self.binary_encoders[col]
            classes_dict = {c: i for i, c in enumerate(le.classes_)}
            df_bin[col] = df_imputed_cat[col].map(lambda x: classes_dict.get(x, 0))
            
        # One-Hot Encoding
        ohe_arr = self.ohe.transform(df_imputed_cat[self.cat_cols])
        ohe_cols = self.ohe.get_feature_names_out(self.cat_cols)
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)
        
        df_final = pd.concat([df_num, df_bin, df_ohe], axis=1)
        return df_final

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
