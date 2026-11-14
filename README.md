# Wine Quality Prediction

## Project Overview

This project aims to analyze red wine quality using Python and machine learning.

The project focuses on exploratory data analysis (EDA), feature relationships, predictive modeling, and model evaluation.

**Current Status:** Exploratory Data Analysis (In Progress)

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

## Project Progress

- [x] Project setup and data loading
- [x] Initial data inspection
- [x] Missing value analysis
- [x] Basic descriptive statistics
- [x] Target distribution analysis
- [x] Initial correlation analysis
- [ ] Data visualization
- [ ] Duplicate and outlier analysis
- [ ] Train/test split
- [ ] Baseline model
- [ ] Model evaluation
- [ ] Final results and conclusions

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place `winequality-red.csv` inside the `data/` directory, then run:

```bash
python src/inspect_data.py
```

## References

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.

Dataset: UCI Machine Learning Repository — Wine Quality.

Dataset licensed under CC BY 4.0.