import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

class ChurnPreprocessor:
    def __init__(self, scaling_strategy="standard"):
        self.scaling_strategy = scaling_strategy
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.binary_encoders = {}
        
        self.num_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges", "AverageUsageHours", "SupportTickets", 
                         "CustomerLifetimeValue", "RevenuePerCustomer", "ComplaintFrequency", "UsageScore"]
        self.binary_cols = ["Gender"]
        self.cat_cols = ["ContractType", "InternetService", "PaymentMethod", "TenureCategory"]
        self.feature_names = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering transformations to input DataFrame."""
        df_feat = df.copy()
        
        # 1. Customer Lifetime Value (CLV)
        if "TotalCharges" in df_feat.columns:
            tc = df_feat["TotalCharges"].fillna(df_feat["MonthlyCharges"] * df_feat["Tenure"])
            df_feat["CustomerLifetimeValue"] = tc
        else:
            df_feat["CustomerLifetimeValue"] = df_feat["MonthlyCharges"] * df_feat["Tenure"]
            
        # 2. Revenue Per Customer
        df_feat["RevenuePerCustomer"] = df_feat["CustomerLifetimeValue"] / (df_feat["Tenure"] + 1)
        
        # 3. Complaint Frequency
        df_feat["ComplaintFrequency"] = df_feat["SupportTickets"] / (df_feat["Tenure"] + 1)
        
        # 4. Usage Score
        usage = df_feat["AverageUsageHours"].fillna(150.0)
        df_feat["UsageScore"] = usage / (df_feat["Age"] + 1)
        
        # 5. Tenure Category
        def get_tenure_cat(t):
            if t <= 12:
                return "New"
            elif t <= 36:
                return "Medium-term"
            else:
                return "Long-term"
        df_feat["TenureCategory"] = df_feat["Tenure"].apply(get_tenure_cat)
        
        return df_feat

    def fit(self, df: pd.DataFrame):
        """Fits all preprocessing structures."""
        df_eng = self.engineer_features(df)
        
        # Fit numerical imputer
        self.num_imputer.fit(df_eng[self.num_cols])
        imputed_num = self.num_imputer.transform(df_eng[self.num_cols])
        
        # Fit scaler
        self.scaler.fit(imputed_num)
        
        # Fit binary encoders
        for col in self.binary_cols:
            le = LabelEncoder()
            # handle possible missing value
            val = df_eng[col].fillna(df_eng[col].mode()[0] if not df_eng[col].empty else "Male")
            le.fit(val)
            self.binary_encoders[col] = le
            
        # Fit One-Hot Encoder
        imputed_cat = self.cat_imputer.fit_transform(df_eng[self.cat_cols])
        self.ohe.fit(imputed_cat)
        
        # Record output feature names list
        ohe_features = self.ohe.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + self.binary_cols + ohe_features
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a raw DataFrame."""
        df_eng = self.engineer_features(df)
        
        # 1. Numerical imputer & scaler
        imputed_num = self.num_imputer.transform(df_eng[self.num_cols])
        scaled_num = self.scaler.transform(imputed_num)
        df_num = pd.DataFrame(scaled_num, columns=self.num_cols, index=df.index)
        
        # 2. Binary variables
        df_binary = pd.DataFrame(index=df.index)
        for col in self.binary_cols:
            le = self.binary_encoders[col]
            classes_dict = {c: i for i, c in enumerate(le.classes_)}
            # Map values with fallback
            df_binary[col] = df_eng[col].fillna(le.classes_[0]).map(lambda x: classes_dict.get(x, 0))
            
        # 3. Categorical variables
        imputed_cat = self.cat_imputer.transform(df_eng[self.cat_cols])
        ohe_arr = self.ohe.transform(imputed_cat)
        ohe_cols = self.ohe.get_feature_names_out(self.cat_cols)
        df_cat = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)
        
        # Combine
        df_final = pd.concat([df_num, df_binary, df_cat], axis=1)
        return df_final

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
