# E-commerce Consumer Behavior Analysis
[![Python Tests](https://github.com/tsaonetapologo/IDS706-ASSIGNMENT-2--TSAONE-TAPOLOGO/actions/workflows/tests.yml/badge.svg)](https://github.com/tsaonetapologo/IDS706-ASSIGNMENT-2--TSAONE-TAPOLOGO/actions/workflows/tests.yml)

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
- Generated the final age-vs-purchase regression visualization used for model interpretation.
- Saved the output plots in the project directory for reporting and review.

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
- Implemented GitHub Actions CI in `.github/workflows/tests.yml` to automatically
  run the test suite on pushes, pull requests, and a weekly schedule. The test
  job covers Python 3.11 and 3.12.
- Installed and validated the required Python packages for analysis and test execution.

### 7. Polars analysis
- Loaded `Purchase_Amount` as text so currency-formatted values can be cleaned consistently.
- Removed currency symbols and commas, trimmed surrounding whitespace, and cast the values to `Float64`.
- Printed the parsed values, null count, summary statistics, and average purchase amount as diagnostics.
- With the included dataset, the Polars average is approximately `275.06` and no purchase amounts fail conversion.

## Key Result
The regression results indicate that age alone provides very little explanatory power for purchase amount in this dataset. The negative R² indicates that the model performs worse than a simple mean-based baseline.

## Project Files
- `ecommerce_analysis.py` — data cleaning, exploratory analysis, plotting, and regression model
- `test_ecommerce_analysis.py` — pytest-based validation of core functions
- `.github/workflows/tests.yml` — automated testing pipeline
- `Dockerfile` — container image for running the analysis
- `requirements.txt` — Python dependencies
- `Ecommerce_Consumer_Behavior_Analysis_Data-2.csv` — source dataset used for the analysis
- `age_distribution.png` — age distribution visualization
- `purchase_amount_vs_age.png` — regression visualization

## How to Run

### Select the dataset
The script searches for `Ecommerce_Consumer_Behavior_Analysis_Data-2.csv` in this order:
1. The path set by `ECOMMERCE_CSV_PATH`.
2. The project directory.
3. The project's `data/` directory.
4. `/Users/tsaonetapologo/Downloads`.

Set `ECOMMERCE_CSV_PATH` to the full CSV file path when using a dataset stored elsewhere.

### Install dependencies
```bash
python -m pip install -r requirements.txt
```

### Run the analysis script
```bash
python ecommerce_analysis.py
```

### Run tests
```bash
python -m pytest -v
```

### Run with Docker
Build the image from the project directory, which includes the bundled dataset:
```bash
docker build -t ecommerce-analysis .
docker run --rm ecommerce-analysis
```
## Refactoring

### What I changed
I refactored the analysis script by extracting reusable helpers for CSV discovery, purchase normalization, plotting, and model fitting. This removed duplicated cleaning logic from `main()` and made the workflow easier to follow and test.

I also added a small, purpose-built regression test for the CSV resolution helper, introduced a consistent `Makefile` for local automation, and updated the CI workflow to check formatting and linting before running the test suite.

### Why I changed it
The original script had a large `main()` function that combined loading, cleaning, analysis, and visualization in one place. Separating those responsibilities made the code easier to read, debug, and extend without changing the underlying analysis results.

### How I verified it
I verified the refactor with fresh project checks:

```bash
python -m black --check .
python -m flake8 .
python -m pytest -q
python ecommerce_analysis.py
```

These checks passed successfully, and the end-to-end analysis still produced the expected model metrics and output plots.

### Refactoring diff

![Refactoring diff](refactor_diff.png)

## Final Status
The repository is now complete with a working analysis workflow, validated tests, the current output visualizations, CI configuration, linting, and a cleaner structure that is easier to maintain. It is ready for review and potential extension with additional behavioral features for improved predictive modeling.

## Real-World Business Interpretation
This analysis is useful for a retail or e-commerce team that wants to understand whether age alone is a useful indicator of purchase spending. The weak relationship in the regression suggests that spending is influenced more by other behavioral and demographic factors, such as income, product category, channel preference, or marketing engagement. In practice, a stronger model would likely combine age with features like purchase intent, income level, and platform behavior to support targeted promotions and customer segmentation.

## Conclusion
This project demonstrates a complete basic data-analysis workflow, including data cleaning, exploratory analysis, visualization, regression, alternative dataframe processing with Polars, automated testing, Docker, and continuous integration through GitHub Actions.
