# E-commerce Consumer Behavior Analysis

## Overview
This project analyzes customer purchasing behavior using an e-commerce dataset and builds a simple predictive model to understand how age relates to purchase amount. The work includes data cleaning, exploratory analysis, visualizations, and regression modeling.

## Assignment 3 Update
This repository reflects the final state of the assignment work completed today. It includes the full data analysis workflow, validation tests, generated visualizations, and a working GitHub Actions CI setup.

## Tasks Completed 

### 1. Data loading and validation
- Confirmed the import path for the dataset and checked that the CSV existed and was not empty.
- Loaded the dataset into pandas for inspection and transformation.
- Verified the data shape and quality before applying the modeling steps.

### 2. Data cleaning and preparation
- Converted purchase amounts from strings such as "$500" and "$1,500" into numeric values.
- Filled missing categorical values with a placeholder label such as "Unknown".
- Removed incomplete rows in required numeric columns to maintain consistent model input.
- Reviewed duplicate entries and identified high-spending customers for additional inspection.

### 3. Exploratory data analysis
- Inspected the dataset schema and summary statistics.
- Reviewed missing data and overall data quality.
- Computed age-based spending patterns to explore relationships in the dataset.
- Compared age with purchase value to assess visible trends.

### 4. Visualization
- Created an age distribution plot to show the customer demographic spread.
- Generated the main regression visualization comparing purchase amount and age trends.
- Saved the remaining output images to the project directory for interpretation and reporting.

### 5. Regression modeling
- Built a linear regression model using age as the predictor variable.
- Split the data into training and testing sets.
- Evaluated the model with the following metrics:
  - Mean Squared Error (MSE): 17474.19
  - Root Mean Squared Error (RMSE): 132.19
  - Mean Absolute Error (MAE): 115.86
  - R-squared: -0.0035
- The model coefficients were:
  - Intercept: 279.96
  - Age coefficient: -0.1914

### 6. Testing and automation
- Added unit tests in `test_ecommerce_analysis.py` for data cleaning and filtering logic.
- Implemented CI in `.github/workflows/tests.yml` to run tests automatically on pushes and pull requests.
- Installed and validated the required Python packages for analysis and test execution.

## Key Result
The model shows that age alone is not a strong predictor of purchase amount in this dataset. The R-squared value is negative, which indicates the model performs worse than a simple mean-based baseline. This suggests that other variables such as income, engagement, purchase channel, or purchase intent likely contribute more strongly to purchasing behavior.

## Project Files
- `ecommerce_analysis.py` — data cleaning, exploratory analysis, plotting, and regression model
- `test_ecommerce_analysis.py` — pytest-based validation of core functions
- `.github/workflows/tests.yml` — automated testing pipeline
- `age_distribution.png` — age distribution visualization
- `purchase_amount_vs_age.png` — regression visualization

## How to Run

### Install dependencies
```bash
python -m pip install pandas pytest polars matplotlib seaborn scikit-learn
```

### Run the analysis script
```bash
python ecommerce_analysis.py
```

### Run tests
```bash
pytest -v
```

## Verification
The project was verified with the current repository state:
- Test result: 8 passed in 1.41s
- Analysis execution successfully produced the model metrics and the remaining output plots

## Final Status
The repository is now complete with a working analysis workflow, validated tests, remaining output visualizations, and CI configuration. It is ready for review and potential extension with additional behavioral features for improved predictive modeling.
