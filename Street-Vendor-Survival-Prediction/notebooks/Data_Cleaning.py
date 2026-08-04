""" Street Vendor Survival Prediction
    Phase 2 - Data Cleaning """
# Importing Libraries
import pandas as pd
import numpy as np
import os

# Loading Dataset
dataset_path = (r"C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/Minor Project/Street-Vendor-Survival-Prediction/datasets/raw_dataset/urban_street_food_vendor_survival_dataset.csv")
df = pd.read_csv(dataset_path)
print("Dataset Loaded Successfully")
print(f"Original Shape : {df.shape}")

# Removing Duplicate Records
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate Records Found : {duplicate_count}")
df.drop_duplicates(inplace=True)
print(f"Shape After Removing Duplicates : {df.shape}")

# Standardizing Text Columns
categorical_columns = [
    "city",
    "zone_type",
    "food_category",
    "license_status",
    "season_of_observation" ]

for column in categorical_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.title())
print("\nText Standardization Completed")

# Handling Missing Values
print("\nMissing Values Before Cleaning")
print(df.isnull().sum())

# ---------- Numerical Columns ----------
median_columns = [
    "vendor_age_years",
    "years_in_business",
    "avg_daily_revenue_inr",
    "avg_daily_customers",
    "monthly_stall_rent_inr",
    "hours_open_per_day",
    "competition_within_100m",
    "customer_complaint_rate"]

for column in median_columns:
    df[column] = df[column].fillna(df[column].median())

# Health inspection score
# Missing value may indicate vendor not inspected
df["monthly_health_inspection_score"] = df["monthly_health_inspection_score"].fillna(0)

# ---------- Categorical Columns ----------
df["license_status"] = df["license_status"].fillna("Unknown")

# ---------- Binary Columns ----------
df["had_fine_last_year"] = df["had_fine_last_year"].fillna(0)
df["has_online_presence"] = df["has_online_presence"].fillna(0)
print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# Correcting Data Types
binary_columns = [
    "had_fine_last_year",
    "has_online_presence",
    "vendor_survived"]

print("\nChecking binary columns before type conversion")
print(df[binary_columns].isnull().sum())

df[binary_columns] = df[binary_columns].astype("int64")
df["num_helpers"] = df["num_helpers"].astype(int)

# Remove Impossible Values
df = df[df["vendor_age_years"] > 15]
df = df[df["years_in_business"] >= 0]
df = df[df["avg_daily_revenue_inr"] >= 0]
df = df[df["avg_daily_customers"] >= 0]
df = df[df["monthly_stall_rent_inr"] >= 0]
df = df[df["hours_open_per_day"] > 0]
print(f"\nShape After Removing Invalid Records : {df.shape}")

# Checking Remaining Missing Values
print("\nFinal Missing Values")
print(df.isnull().sum())

# Saving Cleaned Dataset
processed_path = ("C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/Minor Project/Street-Vendor-Survival-Prediction/datasets/processed_dataset")
os.makedirs(processed_path, exist_ok=True)

output_file = os.path.join(processed_path, "street_vendor_survival_cleaned.csv")

df.to_csv(output_file, index=False)
print("\nCleaned Dataset Saved Successfully")
print(f"\nFinal Shape : {df.shape}")
print(f"\nLocation : {output_file}")