""" Street Vendor Survival Prediction
    Phase 5 - Feature Engineering """
# Importing Libraries
import pandas as pd
import numpy as np
import os

# Loading Cleaned Dataset
dataset_path = ("C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/Minor Project/Street-Vendor-Survival-Prediction/datasets/processed_dataset/street_vendor_survival_cleaned.csv")

df = pd.read_csv(dataset_path)
print("Cleaned Dataset Loaded Successfully")
print(f"Original Shape : {df.shape}")

# Creating a Copy
# Keeping the cleaned dataset unchanged
feature_df = df.copy()

""" Feature 1: Estimated Monthly Revenue
    avg_daily_revenue_inr is daily revenue.
    Multiplying it by 30 gives an estimated monthly revenue."""

feature_df["estimated_monthly_revenue"] = (feature_df["avg_daily_revenue_inr"] * 30)
print("\nEstimated Monthly Revenue Created")

""" Feature 2: Revenue Per Customer
    This shows approximately how much revenue is generated per customer."""
feature_df["revenue_per_customer"] = np.where(
    feature_df["avg_daily_customers"] > 0,
    feature_df["avg_daily_revenue_inr"] /
    feature_df["avg_daily_customers"],
    0)
print("Revenue Per Customer Created")

""" Feature 3: Rent to Revenue Ratio
    This represents the percentage of estimated monthly revenue spent on stall rent."""
feature_df["rent_to_revenue_ratio"] = np.where(
    feature_df["estimated_monthly_revenue"] > 0,
    feature_df["monthly_stall_rent_inr"] /
    feature_df["estimated_monthly_revenue"],
    0)
print("Rent to Revenue Ratio Created")

""" Feature 4: Revenue Per Hour
    Measures the average revenue generated for each hour of operation."""
feature_df["revenue_per_hour"] = np.where(
    feature_df["hours_open_per_day"] > 0,
    feature_df["avg_daily_revenue_inr"] /
    feature_df["hours_open_per_day"],
    0)
print("Revenue Per Hour Created")

""" Feature 5: Customers Per Hour
    Measures customer flow during operating hours."""
feature_df["customers_per_hour"] = np.where(
    feature_df["hours_open_per_day"] > 0,
    feature_df["avg_daily_customers"] /
    feature_df["hours_open_per_day"],
    0)
print("Customers Per Hour Created")

""" Feature 6: Vendor Age Category
    Categorizes vendors based on their age."""
feature_df["vendor_age_category"] = pd.cut(
    feature_df["vendor_age_years"],
    bins=[0, 25, 40, 60, np.inf],
    labels=["Young", "Adult", "Middle_Aged", "Senior"],
    include_lowest=True)
print("Vendor Age Category Created")

""" Feature 7: Business Experience Category
    Categorizes vendors based on their business experience."""
feature_df["business_experience_category"] = pd.cut(
    feature_df["years_in_business"],
    bins=[-1, 2, 5, 10, np.inf],
    labels=["New", "Early_Stage", "Established", "Experienced"])
print("Business Experience Category Created")

""" Feature 8: Revenue Category
    Categorizes vendors based on their revenue."""
feature_df["revenue_category"] = pd.qcut(
    feature_df["avg_daily_revenue_inr"],
    q=4,
    labels=["Low", "Medium", "High", "Very_High"],
    duplicates="drop")
print("Revenue Category Created")

# Feature 9: Competition Category
feature_df["competition_category"] = pd.cut(
    feature_df["competition_within_100m"],
    bins=[-1, 3, 7, np.inf],
    labels=["Low", "Medium", "High"])
print("Competition Category Created")

""" Feature 10: Inspection Category
    The cleaned dataset uses 0 when an inspection score was missing, so 0 is treated as "No Inspection Data"."""
feature_df["inspection_category"] = pd.cut(
    feature_df["monthly_health_inspection_score"],
    bins=[-1, 0, 50, 75, 100],
    labels=[
        "No_Inspection_Data",
        "Low",
        "Moderate",
        "Good"
    ])
print("Inspection Category Created")

# Feature 11: Complaint Category
feature_df["complaint_category"] = pd.cut(
    feature_df["customer_complaint_rate"],
    bins=[-np.inf, 0.10, 0.20, np.inf],
    labels=["Low", "Moderate", "High"])
print("Complaint Category Created")

""" Feature 12: Operational Load
    Combines customer volume and nearby competition.
    Higher values indicate a busier and more competitive operating environment."""
feature_df["operational_load"] = (
    feature_df["avg_daily_customers"] +
    feature_df["competition_within_100m"])
print("Operational Load Created")

# Checking New Features
new_features = [
    "estimated_monthly_revenue",
    "revenue_per_customer",
    "rent_to_revenue_ratio",
    "revenue_per_hour",
    "customers_per_hour",
    "vendor_age_category",
    "business_experience_category",
    "revenue_category",
    "competition_category",
    "inspection_category",
    "complaint_category",
    "operational_load"
]
print("\nNew Features Created:")
print(feature_df[new_features].head())

# Checking Missing Values
print("\nMissing Values After Feature Engineering:")
print(feature_df[new_features].isnull().sum())

""" Encoding Categorical Features
    Convert categorical features into numerical values so that ML algorithms can process them."""
categorical_columns = [
    "city",
    "zone_type",
    "food_category",
    "license_status",
    "season_of_observation",
    "vendor_age_category",
    "business_experience_category",
    "revenue_category",
    "competition_category",
    "inspection_category",
    "complaint_category"]

feature_df = pd.get_dummies(
    feature_df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int)
print("\nCategorical Features Encoded")

""" Removing Vendor ID
    vendor_id is only an identifier.
    It does not provide useful predictive information."""
if "vendor_id" in feature_df.columns:
    feature_df.drop(columns=["vendor_id"], inplace=True)
print("Vendor ID Removed")

# Checking Dataset
print("\nFeature Engineered Dataset Shape:")
print(feature_df.shape)
print("\nFeature Engineered Dataset Columns:")
print(feature_df.columns.tolist())

# Separating Features and Target
X = feature_df.drop(columns=["vendor_survived"])
y = feature_df["vendor_survived"]
print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)

# Checking Target Distribution
print("\nTarget Distribution:")
print(y.value_counts())
print("\nTarget Percentage:")
print((y.value_counts(normalize=True) * 100).round(2))

# Saving ML Dataset
output_path = (
    r"C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/"
    r"Minor Project/Street-Vendor-Survival-Prediction/"
    r"datasets/processed_dataset"
)

# Create the processed dataset folder if it does not exist
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(
    output_path,
    "street_vendor_survival_feature_engineered.csv"
)

feature_df.to_csv(
    output_file,
    index=False
)

print("\nFeature Engineered Dataset Saved Successfully")
print(f"Final Shape : {feature_df.shape}")
print(f"Location : {output_file}")