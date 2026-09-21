# Ecommerce Consumer Behavior Analysis

## Overview

This project analyzes a synthetic ecommerce customer dataset to understand customer purchasing patterns and model the relationship between customer age and purchase amount (that is how much customer's spent purchasing items on the website). The workflow includes data cleaning, exploratory analysis, visualization and lastly, a simple linear regression model.

## Assignment Problem

The objective is to examine how customer attributes relate to purchasing behavior and to build a basic predictive model that estimates purchase amount using age as the explanatory variable.

## Data

The analysis uses the dataset:

- `Ecommerce_Consumer_Behavior_Analysis_Data.csv` from Kaggle

Expected location:

- `/Users/tsaonetapologo/Downloads/Ecommerce_Consumer_Behavior_Analysis_Data.csv`

## Methodology (how the analysis was carried out)

1. I loaded the dataset and validated that the file exists and is not empty.
2. Converted the `Purchase_Amount` field from currency strings into numeric values for my code run 
   easily with no errors.
3. Printed a preview of the dataset, column names, summary statistics, and missing values.
4. Identified duplicate rows and filters high-spending customers for additional review.
5. Grouped average purchase amount by age to examined spending patterns by demographic segment.
6. Generated exploratory visualizations for age distribution and age versus purchase amount.
7. Trained a simple linear regression model using age as the predictor variable.
8. Evaluated the model using:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R-squared ($R^2$)
9. Saved the generated plots to the working directory for interpretation and reporting.

## Tools Used

- Python
- pandas
- polars
- matplotlib
- seaborn
- scikit-learn
- Rust (for performance comparison and discussion)

## Generated Outputs

The following output files are retained in the project directory:

- `age_distribution.png` — a histogram showing the distribution of customer ages in the dataset.
- `purchase_amount_vs_age_model.png` — a scatter plot comparing actual and predicted purchase amounts against age for the regression analysis.
Keeping both plots is useful because they answer different questions: the histogram describes the data distribution, while the scatter plot shows the relationship being modeled and the model's fit.

## Results 

The dataset contains mostly complete records, with only a small number of missing values in categorical fields. These were filled with a placeholder value to maintain consistency in the analysis, and rows with missing values in numeric fields required for regression were removed before modeling.

The exploratory analysis suggests that customer age has only a limited relationship with purchase amount in this dataset. The linear regression model provides a useful baseline for prediction, but its explanatory power is modest, indicating that additional behavioral or demographic features would likely improve predictive accuracy. The script prints MSE, RMSE, MAE, and $R^2$ for the final model evaluation.

## Conclusion

This project demonstrates a basic end-to-end data analysis and machine learning workflow using a consumer behavior dataset. It provides a foundation for more advanced modeling, feature engineering, and deeper behavioral analysis in future work. The main lesson is that age alone is not a strong predictor of purchase amount, and a more complete model would likely benefit from incorporating additional behavioral and demographic features.


## QUESTION 2
## Rust and Pandas Update

This assignment also highlights the practical distinction between pandas and Rust in data processing workflows. Pandas is highly effective for rapid exploratory analysis, data cleaning, and prototyping because it provides an intuitive API and integrates seamlessly with Python-based tools for plotting, statistical analysis, and machine learning. In this project, pandas was used to load the dataset, clean the purchase data, summarize the data, and build the regression model.

Rust, by contrast, becomes valuable when performance, memory efficiency, and low-level control are more important, especially for larger datasets or compute-intensive pipelines. Although Rust has a steeper learning curve and a more verbose syntax, it offers faster execution and stronger guarantees around memory safety. In a larger analytics workflow, pandas is often ideal for rapid iteration and exploration, while Rust or another compiled engine may be more appropriate for performance-sensitive production pipelines.

This comparison illustrates an important principle in data engineering: Python libraries such as pandas are excellent for readability and experimentation, whereas Rust is especially useful when speed, concurrency, and system-level efficiency are the primary objectives.

## Requirements

Install the required packages in the project environment:

```bash
cd /Users/tsaonetapologo/data-processing-frameworks-demo
. .venv/bin/activate
python -m pip install pandas matplotlib seaborn scikit-learn polars
```

## Run the Script

```bash
/Users/tsaonetapologo/data-processing-frameworks-demo/.venv/bin/python /Users/tsaonetapologo/Downloads/ecommerce_analysis.py
```
## Visualizations
