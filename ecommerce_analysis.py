import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

matplotlib.use("Agg")

CATEGORICAL_COLUMNS = [
    "Social_Media_Influence",
    "Engagement_with_Ads",
    "Gender",
    "Income_Level",
    "Purchase_Category",
    "Purchase_Channel",
    "Time_of_Purchase",
    "Shipping_Preference",
    "Purchase_Intent",
    "Location",
    "Occupation",
    "Marital_Status",
    "Education_Level",
    "Payment_Method",
    "Device_Used_for_Shopping",
]

NUMERIC_COLUMNS = [
    "Age",
    "Purchase_Amount",
    "Frequency_of_Purchase",
    "Time_to_Decision",
]


def normalize_purchase_amounts(series):
    """Convert currency-formatted text values into numeric purchase amounts."""
    cleaned = series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def clean_data(df):
    """Clean the e-commerce dataset."""
    cleaned_df = df.copy()

    if "Purchase_Amount" in cleaned_df.columns:
        cleaned_df["Purchase_Amount"] = normalize_purchase_amounts(
            cleaned_df["Purchase_Amount"]
        )

    for column in CATEGORICAL_COLUMNS:
        if column in cleaned_df.columns:
            cleaned_df[column] = cleaned_df[column].fillna("Unknown")

    required_columns = [
        column for column in NUMERIC_COLUMNS if column in cleaned_df.columns
    ]
    if required_columns:
        cleaned_df = cleaned_df.dropna(subset=required_columns)

    return cleaned_df


def get_high_spenders(df, threshold=1000):
    """Return customers whose purchase amount exceeds the threshold."""
    return df[df["Purchase_Amount"] > threshold].copy()


def average_spending_by_age(df):
    """Calculate average purchase amount for each age."""
    return df.groupby("Age")["Purchase_Amount"].mean()


def resolve_csv_path(project_root, download_dir=None, env_path=None):
    """Choose the dataset path in a predictable order: explicit env var first."""
    download_dir = download_dir or Path("/Users/tsaonetapologo/Downloads")
    candidate_paths = [
        Path(env_path).expanduser() if env_path else None,
        project_root / "Ecommerce_Consumer_Behavior_Analysis_Data-2.csv",
        project_root / "data" / "Ecommerce_Consumer_Behavior_Analysis_Data-2.csv",
        download_dir / "Ecommerce_Consumer_Behavior_Analysis_Data-2.csv",
    ]
    return next(
        (path for path in candidate_paths if path is not None and path.exists()),
        None,
    )


def validate_csv_path(csv_path):
    """Ensure the selected CSV file exists and is not empty."""
    if csv_path is None:
        raise FileNotFoundError(
            "CSV file not found. Set ECOMMERCE_CSV_PATH or place the dataset in the "
            "project folder or Downloads."
        )

    if csv_path.stat().st_size == 0:
        raise ValueError(
            f"CSV file is empty: {csv_path}. Please add the data or use the "
            "correct dataset file."
        )

    return csv_path


def print_dataset_summary(df):
    """Print the cleaned dataset overview and quality checks."""
    print("Dataset preview:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nDataset information:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())

    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        print(f"\nFound {len(duplicate_rows)} duplicate rows:")
        print(duplicate_rows.head())

    df_high_spenders = get_high_spenders(df, threshold=1000)
    print(f"\nNumber of high spenders: {len(df_high_spenders)}")
    if not df_high_spenders.empty:
        print("\nSample of high spenders:")
        print(df_high_spenders.head())

    avg_spending_by_age = average_spending_by_age(df)
    print("\nAverage spending by age:")
    print(avg_spending_by_age.head(10))


def save_age_distribution_plot(df, output_path):
    """Save a histogram of customer age distribution."""
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Age", bins=15, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fit_age_regression(df):
    """Train and evaluate a simple age-based linear regression model."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split

    model_df = df.dropna(subset=["Age", "Purchase_Amount"]).copy()
    X = model_df[["Age"]]
    y = model_df["Purchase_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        },
    }


def save_purchase_vs_age_plot(X_test, y_test, y_pred, output_path):
    """Save a regression scatter plot comparing actual and predicted spend."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test["Age"], y_test, color="blue", label="Actual Amounts")
    plt.scatter(X_test["Age"], y_pred, color="red", label="Predicted Amounts")
    plt.title("Purchase Amount vs Age")
    plt.xlabel("Age")
    plt.ylabel("Purchase Amount")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()


def perform_polars_analysis(csv_path):
    """Run the equivalent purchase-value conversion check using Polars."""
    polars_df = pl.read_csv(csv_path, schema_overrides={"Purchase_Amount": pl.String})

    print("\nRAW PURCHASE AMOUNT VALUES:")
    print(polars_df["Purchase_Amount"].head(10))
    print("RAW DTYPE:", polars_df["Purchase_Amount"].dtype)

    polars_df = polars_df.with_columns(
        pl.col("Purchase_Amount")
        .str.replace_all(r"[$,]", "")
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias("Purchase_Amount")
    )

    print("\nFirst 5 rows using Polars:")
    print(polars_df.head())
    print("\nDataset shape:")
    print(polars_df.shape)
    print("\nSummary statistics:")
    print(polars_df.describe())

    print("\nPurchase Amount diagnostic:")
    print(polars_df["Purchase_Amount"].head(10))
    print(polars_df["Purchase_Amount"].dtype)
    print("Null count:", polars_df["Purchase_Amount"].null_count())

    if "Purchase_Amount" in polars_df.columns:
        average_purchase = polars_df["Purchase_Amount"].mean()
        print(f"\nAverage Purchase Amount: {average_purchase}")


def main():
    project_root = Path(__file__).resolve().parent
    download_dir = Path("/Users/tsaonetapologo/Downloads")
    csv_path = resolve_csv_path(
        project_root,
        download_dir=download_dir,
        env_path=os.environ.get("ECOMMERCE_CSV_PATH"),
    )
    csv_path = validate_csv_path(csv_path)

    df = pd.read_csv(csv_path)
    df_clean = clean_data(df)

    print_dataset_summary(df_clean)
    save_age_distribution_plot(df_clean, "age_distribution.png")

    model_result = fit_age_regression(df_clean)
    model = model_result["model"]
    X_test = model_result["X_test"]
    y_test = model_result["y_test"]
    y_pred = model_result["y_pred"]
    metrics = model_result["metrics"]

    print("\nModel Coefficients:")
    print(f"Coefficient for Age: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    print(
        "\nRegression equation: "
        f"Purchase_Amount = {model.intercept_:.2f} + ({model.coef_[0]:.4f} * Age)"
    )

    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {metrics['mse']}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']}")
    print(f"R-Squared: {metrics['r2']}")
    print(
        "\nInterpretation: Age alone provides limited explanatory power for purchase "
        "amount in this dataset, so the relationship is weak and additional features "
        "would likely improve prediction quality."
    )

    save_purchase_vs_age_plot(X_test, y_test, y_pred, "purchase_amount_vs_age.png")
    perform_polars_analysis(csv_path)


if __name__ == "__main__":
    main()
