import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "winequality-red.csv"

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# A. Read CSV (remember the separator)
df = pd.read_csv(DATA_PATH, sep=";")

# B. Display first 5 rows
first_5_rows = df.head(5)
print(f"The first 5 rows are:\n{first_5_rows}")

# C. Display number of rows and columns
rows, cols = df.shape
print(f"Rows: {rows}, Columns: {cols}")

# D. Count missing values in each column
print(f"Missing values in each column:\n{df.isna().sum()}")

# E. Display information about the dataset
print(f"Information about the dataset:")
df.info()

# F. Display the mean, min, and max values for alcohol content
mean_alcohol = df["alcohol"].mean()
min_alcohol = df["alcohol"].min()
max_alcohol = df["alcohol"].max()
print(f"Mean alcohol content: {mean_alcohol}, Min alcohol content: {min_alcohol}, Max alcohol content: {max_alcohol}")

# G. Display quality distribution
quality_counts = df["quality"].value_counts()
sorted_quality_counts = quality_counts.sort_index()
print(f"Quality distribution:\n{sorted_quality_counts}")

# H. Display the correlation between alcohol content and quality
correlation_alcohol_quality = df["alcohol"].corr(df["quality"])
print(f"Correlation between alcohol content and quality: {correlation_alcohol_quality}")

# I. Display three highest correlation with quality (excluding quality itself), sorted in descending order
correlations = df.corr(numeric_only=True)["quality"]
top_3_correlations = correlations.drop("quality").sort_values(ascending=False).head(3)
print(f"Top 3 correlations with quality:\n{top_3_correlations}")

# J. Display the total rows in the dataset
total_rows = len(df)
print(f"Total rows in the dataset: {total_rows}")

# K. Display the duplicate rows 
duplicate_rows = df.duplicated().sum()
print(f"Duplicate rows in the dataset: {duplicate_rows}")

# L. Display the rate of duplicate rows
duplicate_percentage = duplicate_rows / total_rows * 100
print(f"Rate of duplicate rows: {duplicate_percentage:.2f}%")