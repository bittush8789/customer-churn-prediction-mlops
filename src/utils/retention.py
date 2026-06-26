class BusinessRetentionEngine:
    def __init__(self):
        pass

    def evaluate_customer(self, customer_data: dict, churn_prob: float, predicted_clv: float) -> dict:
        """Computes advanced retention and business metrics for a customer."""
        tickets = int(customer_data.get("SupportTickets", 0))
        monthly_charges = float(customer_data.get("MonthlyCharges", 0))
        
        # 1. Churn Risk Level
        if churn_prob >= 0.8:
            risk_level = "High"
        elif churn_prob >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        # 2. Customer Health Score (0 to 100)
        # Inversely proportional to churn prob and support ticket counts
        health = (1 - churn_prob) * 100 - (tickets * 3)
        health_score = max(0.0, min(100.0, health))
        
        # 3. Estimated Revenue Loss
        revenue_loss = churn_prob * predicted_clv
        
        # 4. Retention Cost
        # High Risk: 30% discount for 1 year (MonthlyCharges * 0.3 * 12) + loyalty onboarding ($50)
        # Medium Risk: 15% discount or upgrade plan incentive
        # Low Risk: General newsletter/maintenance
        if risk_level == "High":
            retention_cost = (monthly_charges * 0.3 * 12) + 50.0
        elif risk_level == "Medium":
            retention_cost = (monthly_charges * 0.15 * 6) + 20.0
        else:
            retention_cost = 5.0 # flat marketing cost
            
        # 5. Retention ROI (Return on Investment)
        # Expected Value Saved: (ChurnProbability * CLV) - RetentionCost
        saved_value = revenue_loss
        net_benefit = saved_value - retention_cost
        retention_roi = (net_benefit / (retention_cost + 1e-5)) * 100 # percentage
        
        # 6. Recommendation Rules
        recs = []
        if risk_level == "High":
            recs = [
                "Offer 30% Discount on Plan",
                "Convert to Annual Contract",
                "Assign Senior Support Agent",
                "Provide Loyalty Benefits (e.g. Free streaming add-on)"
            ]
        elif risk_level == "Medium":
            recs = [
                "Offer Upgrade Plan with Extra Benefits",
                "Send Targeted Retention Campaign (e.g. Email / Call)"
            ]
        else:
            recs = [
                "Maintain Monthly Engagement (e.g. Newsletters)",
                "VIP Rewards Program Enrollment (if High Value)"
            ]
            
        return {
            "risk_level": risk_level,
            "health_score": round(health_score, 1),
            "estimated_revenue_loss": round(revenue_loss, 2),
            "retention_cost": round(retention_cost, 2),
            "retention_roi": round(retention_roi, 1),
            "recommendations": recs
        }
