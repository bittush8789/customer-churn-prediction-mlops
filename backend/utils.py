import os
import pandas as pd
import numpy as np

class RecommendationEngine:
    def __init__(self):
        pass

    def generate(self, customer: dict, churn_prob: float) -> list:
        recs = []
        if churn_prob >= 0.8:
            recs.append({
                "Action": "Premium retention discount",
                "Details": "Customer is at extreme risk (>=80%). Offer a 30% contract discount or 6 months of free streaming addons."
            })
        elif churn_prob >= 0.5:
            recs.append({
                "Action": "Proactive support follow-up",
                "Details": "Customer is at moderate risk (50%-80%). Schedule an outbound support call to review account health."
            })
            
        clv = customer.get("CustomerLifetimeValue", customer.get("TotalCharges", 0))
        monthly = customer.get("MonthlyCharges", 0)
        if clv > 1800 or monthly > 85:
            recs.append({
                "Action": "Enroll in VIP Loyalty program",
                "Details": "High value customer. Offer priority service queuing and premium device upgrades."
            })
            
        tickets = customer.get("SupportTickets", 0)
        if tickets >= 3:
            recs.append({
                "Action": "Supervisor queue routing",
                "Details": "Frequent complaints. Route next calls to senior support specialists and apply a $20 goodwill credit."
            })
            
        contract = customer.get("ContractType", "Month-to-month")
        if contract == "Month-to-month" and churn_prob > 0.4:
            recs.append({
                "Action": "Convert to annual contract",
                "Details": "Offer contract conversion with a 15% discount for a 12-month commitment."
            })
            
        if not recs:
            recs.append({
                "Action": "Standard maintenance",
                "Details": "Low risk customer. Keep active with normal newsletter updates."
            })
            
        return recs

def get_segment(row: dict) -> str:
    """Helper to classify a single customer dictionary into one of 4 segments."""
    tenure = float(row.get("Tenure", 0))
    tickets = int(row.get("SupportTickets", 0))
    monthly = float(row.get("MonthlyCharges", 0))
    
    if tenure <= 12:
        return "New Customers"
    elif tickets >= 3:
        return "At Risk Customers"
    elif monthly >= 80:
        return "High Value Customers"
    else:
        return "Loyal Customers"

def get_dashboard_stats() -> dict:
    """Computes stats from the dataset for dashboard visualizations."""
    csv_path = "data/customer_churn.csv"
    if not os.path.exists(csv_path):
        return {
            "total_customers": 0,
            "churn_rate": 0.0,
            "avg_tenure": 0.0,
            "avg_charges": 0.0,
            "segments": {"Loyal Customers": 0, "New Customers": 0, "High Value Customers": 0, "At Risk Customers": 0},
            "churn_by_contract": {}
        }
        
    df = pd.read_csv(csv_path)
    total = len(df)
    
    # Target distribution
    churn_yes = int((df["Churn"] == "Yes").sum())
    churn_rate = float(churn_yes / total * 100) if total > 0 else 0.0
    
    # Means
    avg_tenure = float(df["Tenure"].mean())
    avg_charges = float(df["MonthlyCharges"].mean())
    
    # Classify segments
    segments = {"Loyal Customers": 0, "New Customers": 0, "High Value Customers": 0, "At Risk Customers": 0}
    for _, row in df.iterrows():
        seg = get_segment(row.to_dict())
        segments[seg] = segments.get(seg, 0) + 1
        
    # Churn rate by contract type
    contract_churn = df.groupby("ContractType")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).to_dict()
    
    return {
        "total_customers": total,
        "churn_rate": round(churn_rate, 2),
        "avg_tenure": round(avg_tenure, 1),
        "avg_charges": round(avg_charges, 2),
        "segments": segments,
        "churn_by_contract": {k: round(v, 2) for k, v in contract_churn.items()}
    }
