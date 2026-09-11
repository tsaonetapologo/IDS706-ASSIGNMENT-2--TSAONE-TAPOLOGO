
 ## Ecommerce Consumer Behavior Analysis

## Overview

This project analyzes a synthetic ecommerce customer dataset to understand customer purchasing patterns and model the relationship between customer age and purchase amount. The workflow includes data cleaning, exploratory analysis, visualization, and a simple linear regression model.

## Assignment Problem

The objective is to examine how customer attributes relate to purchasing behavior and to build a basic predictive model that estimates purchase amount using age as the explanatory variable.

## Data

The analysis uses the dataset:

- `Ecommerce_Consumer_Behavior_Analysis_Data.csv`

Expected location:

- `/Users/tsaonetapologo/Downloads/Ecommerce_Consumer_Behavior_Analysis_Data.csv`

## Methodology

The script performs the following steps:

1. Loads the CSV file and validates that it exists and is not empty.
2. Cleans the `Purchase_Amount` column by converting currency strings to numeric values.
3. Prints a dataset preview, column names, summary statistics, and missing values.
4. Checks for duplicate rows and filters high-spending customers.
5. Groups average purchase amount by age.
6. Produces exploratory visualizations for age distribution and age versus purchase amount.
7. Trains a linear regression model using age to predict purchase amount.
8. Evaluates the model with:
   - Mean Squared Error (MSE)
   - R-squared ($R^2$)
9. Saves relevant visual output files to the working directory.

## Tools Used

- Python
- pandas
- matplotlib
- seaborn
- scikit-learn

## Requirements

Install the required packages in the project environment:

```bash
cd /Users/tsaonetapologo/data-processing-frameworks-demo
. .venv/bin/activate
python -m pip install pandas matplotlib seaborn scikit-learn
```

## Run the Script

```bash
/Users/tsaonetapologo/data-processing-frameworks-demo/.venv/bin/python /Users/tsaonetapologo/Downloads/ecommerce_analysis.py
```

## Generated Outputs

The following files are created in the working directory:

- `age_distribution.png`


## Results Summary

The analysis shows that the dataset has no missing values in most columns, with only a few categorical fields containing missing entries. The average purchase amount varies by age, and the regression model estimates a weak relationship between age and purchase amount in this dataset.

## Conclusion

This project demonstrates a basic end-to-end data analysis and machine learning workflow using a consumer behavior dataset. It provides a foundation for more advanced modeling, feature engineering, and deeper behavioral analysis in future work.

## How to Run the Project

1. Clone the GitHub repository.
2. Install the required Python libraries:
