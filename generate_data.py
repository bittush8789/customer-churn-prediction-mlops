import os
import pandas as pd

def main():
    print("Generating training dataset...")
    # Check if raw data exists, if not create dummy data or verify
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "customer_churn.csv")
    
    if os.path.exists(raw_path):
        print(f"Dataset already exists at {raw_path}")
    else:
        # Create a tiny dummy dataset to ensure pipeline compatibility in test environments
        dummy_data = {
            "Gender": ["Male", "Female"] * 10,
            "Age": [34, 45] * 10,
            "Tenure": [12, 6] * 10,
            "MonthlyCharges": [85.5, 55.0] * 10,
            "TotalCharges": [1026.0, 330.0] * 10,
            "ContractType": ["Month-to-month", "One year"] * 10,
            "InternetService": ["Fiber optic", "DSL"] * 10,
            "PaymentMethod": ["Electronic check", "Mailed check"] * 10,
            "SupportTickets": [3, 1] * 10,
            "AverageUsageHours": [140.5, 90.0] * 10,
            "Churn": ["No", "Yes"] * 10
        }
        df = pd.DataFrame(dummy_data)
        df.to_csv(raw_path, index=False)
        print(f"Created sample dataset at {raw_path}")

if __name__ == "__main__":
    main()
