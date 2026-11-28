import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "winequality-red.csv"

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Read CSV (remember the separator)
df = pd.read_csv(DATA_PATH, sep=";")

# Display the original dataset shape
print("Original dataset shape:", df.shape)

# Calculate and display the duplicated data
print("Duplicate rows:", df.duplicated().sum())

# Drop the duplicate data and reset its index
clean_df = df.drop_duplicates().reset_index(drop=True)
print("Cleaned dataset shape:", clean_df.shape)

# Feature & Label Separation
X = clean_df.drop(columns="quality")
y = clean_df["quality"]
print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# Dataset Inspection
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining label distribution:\n", y_train.value_counts())
print("\nTesting label distribution:\n", y_test.value_counts())

print("\nTraining label proportions:\n", y_train.value_counts(normalize=True))
print("\nTesting label proportions:\n", y_test.value_counts(normalize=True))

# Build a Dummy Classifier
model_dummy = DummyClassifier(strategy="most_frequent")
model_dummy.fit(X_train, y_train)
y_pred = model_dummy.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nPrediction shape:", y_pred.shape)
print("\nPredicted label distribution:\n", pd.Series(y_pred).value_counts())
print("\nBaseline accuracy:", accuracy)