import pandas as pd
import numpy as np

class FeaturePipeline:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering formulas to raw columns."""
        df_eng = df.copy()
        
        # Avoid division by zero by adding small epsilon
        tenure_eps = df_eng["Tenure"].replace(0, 1)
        charges_eps = df_eng["MonthlyCharges"].replace(0, 1)
        
        # 1. Base business ratios
        df_eng["CustomerLifetimeValue"] = df_eng["TotalCharges"].fillna(df_eng["MonthlyCharges"] * df_eng["Tenure"])
        df_eng["ChargePerMonth"] = df_eng["CustomerLifetimeValue"] / tenure_eps
        df_eng["TicketRatio"] = df_eng["SupportTickets"] / tenure_eps
        df_eng["UsagePerDollar"] = df_eng["AverageUsageHours"].fillna(150.0) / charges_eps
        
        # 2. Scores
        # Customer Value Score: scaled total value
        df_eng["CustomerValueScore"] = df_eng["Tenure"] * df_eng["MonthlyCharges"]
        
        # Contract Risk Score: Higher score = Higher risk
        contract_map = {"Month-to-month": 3.0, "One year": 2.0, "Two year": 1.0}
        df_eng["ContractRiskScore"] = df_eng["ContractType"].map(lambda x: contract_map.get(x, 2.0))
        
        # Engagement Score: High usage and low tickets = higher score
        df_eng["EngagementScore"] = df_eng["AverageUsageHours"].fillna(150.0) / (df_eng["SupportTickets"] + 1)
        
        # 3. Customer Segments (Personas)
        segments = []
        for idx, row in df_eng.iterrows():
            tenure = row["Tenure"]
            tickets = row["SupportTickets"]
            monthly = row["MonthlyCharges"]
            contract = row["ContractType"]
            
            if tickets >= 3:
                seg = "At Risk Customer"
            elif tenure > 24 and monthly > 85 and tickets < 2:
                seg = "VIP Customer"
            elif tenure > 36 and tickets < 2:
                seg = "Loyal Customer"
            elif tenure <= 12:
                seg = "New Customer"
            else:
                seg = "Regular Customer"
            segments.append(seg)
            
        df_eng["Segment"] = segments
        print("Production Feature Engineering completed.")
        return df_eng

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)
