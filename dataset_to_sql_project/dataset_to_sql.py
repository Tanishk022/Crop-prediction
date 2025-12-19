# dataset_to_sql.py
# ----------------------------------------
# This script loads a crop recommendation dataset (CSV)
# and directly inserts it into a MySQL database.
#
# REQUIREMENTS:
# pip install pandas sqlalchemy pymysql
#
# DATABASE:
# Database name: crop_analysis
# Table name: crop_data (auto-created)
# ----------------------------------------

import pandas as pd
from sqlalchemy import create_engine

# STEP 1: Load dataset (CSV file)
# Make sure this CSV file is present in the SAME folder
DATASET_FILE = "Crop_recommendation.csv"
df = pd.read_csv(DATASET_FILE)

print("Dataset Loaded Successfully")
print(df.head())

# STEP 2: MySQL connection details
# CHANGE user, password if needed
engine = create_engine(
    "mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/crop_analysis"
)

# STEP 3: Insert dataset into MySQL
df.to_sql(
    name="crop_data",
    con=engine,
    if_exists="replace",
    index=False
)

print("SUCCESS: Dataset inserted into MySQL database (crop_analysis -> crop_data)")
