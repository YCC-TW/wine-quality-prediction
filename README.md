# Wine Quality Prediction

## Project Overview

This project aims to analyze red wine quality using Python and machine learning.

The project focuses on exploratory data analysis (EDA), feature relationships, predictive modeling, and model evaluation.

**Current Status:** Baseline Model Completed — Predictive Modeling and Evaluation in Progress

## Dataset

**Source:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

The dataset contains physicochemical measurements of red wine samples and their corresponding quality scores.

- **Total Samples:** 1,599
- **Input Features:** 11
- **Target Variable:** `quality`
- **Total Columns:** 12
- **Missing Values:** 0

## Tools and Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Exploratory Data Analysis

### 1. Data Inspection

Loaded the red wine dataset using pandas and examined its structure.

- Verified the dataset dimensions.
- Inspected the first five rows.
- Checked for missing values.
- Examined feature data types and basic descriptive statistics.

### 2. Quality Distribution

The quality scores range from 3 to 8.

Most samples have quality scores of 5 or 6, accounting for approximately 82.5% of the dataset.

### 3. Feature Correlation

Calculated Pearson correlation coefficients between numerical features and wine quality.

The three features with the highest positive correlations to quality were:

| Feature | Pearson Correlation |
|---|---:|
| Alcohol | 0.4762 |
| Sulphates | 0.2514 |
| Citric Acid | 0.2264 |

Alcohol showed a moderate positive linear correlation with wine quality. However, correlation does not imply causation.

### 4. Duplicate Analysis

Checked the dataset for duplicate observations using pandas `duplicated()`.

The analysis produced the following results:

| Metric | Value |
|---|---:|
| Total Observations | 1,599 |
| Duplicate Observations | 240 |
| Duplicate Rate | 15.01% |

Approximately 15.01% of the dataset consists of duplicate observations.

However, duplicate observations do not necessarily indicate data errors, as different wine samples may share identical physicochemical measurements and quality scores.

## Data Preprocessing

The original dataset contains 1,599 observations, including 240 exact duplicate rows. After removing these duplicates and resetting the index, 1,359 observations remain.

Exact duplicate rows were removed before the train/test split to mitigate the risk of data leakage, as it is unclear whether identical observations represent independent wine samples or repeated records of the same sample.

## Baseline Model

Wine quality prediction is formulated as a multiclass classification problem with six quality classes ranging from 3 to 8.

A `DummyClassifier` with the `most_frequent` strategy was selected as the baseline model. This classifier predicts the most frequent class in the training set without learning relationships between the input features and the target variable.

The dataset was divided into training and testing sets using an 80/20 split, with `random_state=42` for reproducibility and `stratify=y` to preserve the approximate class distribution across both sets.

## Baseline Results

| Metric | Result |
|---|---:|
| Training Samples | 1,087 |
| Testing Samples | 272 |
| Predicted Class | Quality 5 |
| Correct Predictions | 116 |
| Incorrect Predictions | 156 |
| Baseline Accuracy | 42.65% |

The baseline model predicted Quality 5 for all 272 test samples because it was the most frequent class in the training set.

Since 116 of the 272 test samples belong to Quality 5, the resulting accuracy is approximately 42.65%. This result establishes an initial performance benchmark for evaluating future predictive models.

## Limitations

- **Class Imbalance:** Approximately 81.82% of the cleaned dataset consists of samples with quality scores of 5 or 6. This imbalance may make it difficult for predictive models to identify minority classes effectively.

- **Duplicate Removal:** Exact duplicate rows do not necessarily represent erroneous or repeated records. Different wine samples may have identical physicochemical measurements and quality scores. Removing these observations may reduce the amount of useful training data and affect the representativeness of the dataset.

- **Evaluation Metrics:** Accuracy alone may not adequately reflect model performance, particularly when the dataset is imbalanced. Additional evaluation metrics, including precision, recall, F1-score, and confusion matrices, will be considered in future model comparisons.

## Project Progress

- [x] Project setup and data loading
- [x] Initial data inspection
- [x] Missing value analysis
- [x] Basic descriptive statistics
- [x] Target distribution analysis
- [x] Initial correlation analysis
- [x] Duplicate analysis
- [x] Data preprocessing
- [x] Stratified train/test split
- [x] Baseline model implementation
- [x] Baseline model evaluation
- [ ] Data visualization
- [ ] Outlier analysis
- [ ] Predictive model implementation
- [ ] Model comparison
- [ ] Advanced evaluation metrics
- [ ] Error analysis
- [ ] Final results and conclusions

## How to Run

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Place `winequality-red.csv` inside the `data/` directory.

### Exploratory Data Analysis

Run the following command to inspect the dataset and perform initial exploratory data analysis:

```bash
python src/inspect_data.py
```

### Baseline Model

Run the following command to perform data preprocessing, split the dataset, train the baseline classifier, and evaluate its accuracy:

```bash
python src/train_baseline.py
```

## References

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.

Dataset: UCI Machine Learning Repository — Wine Quality.

Dataset licensed under CC BY 4.0.