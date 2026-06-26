import os
import pandas as pd

class DataIngestion:
    def __init__(self, filepath="data/customer_churn.csv"):
        self.filepath = filepath
        self.required_cols = [
            "Gender", "Age", "Tenure", "MonthlyCharges", "TotalCharges", 
            "ContractType", "InternetService", "PaymentMethod", "SupportTickets", "AverageUsageHours"
        ]

    def load_data(self) -> pd.DataFrame:
        """Loads and validates target CSV data."""
        if not os.path.exists(self.filepath):
            # If not in data/, check root directory or return a FileNotFoundError
            raise FileNotFoundError(f"Subscribers dataset not found at {self.filepath}.")
            
        df = pd.read_csv(self.filepath)
        print(f"Dataset read successfully: {df.shape}")
        
        # Verify required columns exist
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Schema missing critical feature columns: {missing}")
            
        # Clean duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"Dropped {dup_count} duplicate subscriber rows.")
            
        return df

if __name__ == "__main__":
    ingestor = DataIngestion()
    df = ingestor.load_data()
