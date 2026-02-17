# ==========================================
# PHASE 1 - EXPLORATION & DATA ANALYSIS
# Dataset: COVID-19 Symptoms
# Objective: Predict patient outcome
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------

print("Loading dataset...")

df = pd.read_csv("data/dataset_symps_covid19.csv")

print("Dataset loaded successfully!\n")

# ------------------------------------------
# 2. Basic Information
# ------------------------------------------

print("First 5 rows:")
print(df.head(), "\n")

print("Dataset Shape:")
print(df.shape, "\n")

print("Data Types:")
print(df.dtypes, "\n")

print("Missing Values:")
print(df.isnull().sum(), "\n")

print("Duplicate Rows:")
print(df.duplicated().sum(), "\n")

# ------------------------------------------
# 3. Data Cleaning
# ------------------------------------------

print("Cleaning data...")

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
# For target variable (outcome), replace NaN with "Unknown"
if "outcome" in df.columns:
    df["outcome"] = df["outcome"].fillna("Unknown")

print("Cleaning completed!\n")

# ------------------------------------------
# 4. Target Variable Analysis
# ------------------------------------------

TARGET_COLUMN = "outcome"

print(f"Target column: {TARGET_COLUMN}")
print(df[TARGET_COLUMN].value_counts(), "\n")

# ------------------------------------------
# 5. Class Distribution Plot
# ------------------------------------------

plt.figure()
sns.countplot(x=TARGET_COLUMN, data=df)
plt.title("Outcome Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/class_distribution.png")
plt.close()

print("Class distribution plot saved.\n")

# ------------------------------------------
# 6. Correlation Matrix (Numeric Features Only)
# ------------------------------------------

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=False)
plt.title("Correlation Matrix (Numeric Features Only)")
plt.tight_layout()
plt.savefig("data/correlation_matrix.png")
plt.close()

print("Correlation matrix saved.\n")

# ------------------------------------------
# 7. Statistical Summary
# ------------------------------------------

print("Statistical Summary (Numeric Columns):")
print(numeric_df.describe(), "\n")

# ------------------------------------------
# 8. Save Cleaned Dataset
# ------------------------------------------

df.to_csv("data/cleaned_covid19.csv", index=False)

print("Cleaned dataset saved as cleaned_covid19.csv")

print("\nPhase 1 completed successfully!")
