# Importing Required Libraries
import pandas as pd

# Loading the  Dataset
df = pd.read_csv('C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/Minor Project/Street-Vendor-Survival-Prediction/datasets/raw_dataset/urban_street_food_vendor_survival_dataset.csv')
print("\nFirst Five Rows")
print(df.head())

# Dataset Dimensions
print("\nDataset Shape")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Column Names
print("\nColumn Names")
print(df.columns.tolist())

# Data Types
print("\nData Types")
print(df.dtypes)

# Dataset Information
print("\nDataset Information")
df.info()

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Duplicate Records
print("\nDuplicate Records")
print(df.duplicated().sum())

# Target Variable
print("\nTarget Column")
target = "vendor_survived"
print(target)

print("\nTarget Distribution")
print(df[target].value_counts())

data_dictionary = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

data_dictionary.to_csv(
    "C:/Users/HP/3D Objects/Desktop/College/College 5th Sem/Minor Project/Street-Vendor-Survival-Prediction/reports/data_dictionary.csv",
    index=False)
print("Data Dictionary Created Successfully.")