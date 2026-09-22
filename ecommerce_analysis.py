import os
from pathlib import Path

import pandas as pd
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def clean_data(df):
    """Clean the e-commerce dataset."""
    df = df.copy()

    if "Purchase_Amount" in df.columns:
        df["Purchase_Amount"] = pd.to_numeric(
            df["Purchase_Amount"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )

    categorical_columns = [
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

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")

    numeric_columns = [
        "Age",
        "Purchase_Amount",
        "Frequency_of_Purchase",
        "Time_to_Decision",
    ]

    required_columns = [column for column in numeric_columns if column in df.columns]

    if required_columns:
        df = df.dropna(subset=required_columns)

    return df


def get_high_spenders(df, threshold=1000):
    """Return customers whose purchase amount exceeds the threshold."""
    return df[df["Purchase_Amount"] > threshold].copy()


def average_spending_by_age(df):
    """Calculate average purchase amount for each age."""
    return df.groupby("Age")["Purchase_Amount"].mean()


def main():
    project_root = Path(__file__).resolve().parent
    download_dir = Path("/Users/tsaonetapologo/Downloads")

    candidate_paths = [
        (
            Path(os.environ.get("ECOMMERCE_CSV_PATH", "")).expanduser()
            if os.environ.get("ECOMMERCE_CSV_PATH")
            else None
        ),
        project_root / "Ecommerce_Consumer_Behavior_Analysis_Data-2.csv",
        project_root / "data" / "Ecommerce_Consumer_Behavior_Analysis_Data-2.csv",
        download_dir / "Ecommerce_Consumer_Behavior_Analysis_Data-2.csv",
    ]

    csv_path = next(
        (path for path in candidate_paths if path is not None and path.exists()), None
    )

    if csv_path is None:
        raise FileNotFoundError(
            "CSV file not found. Set ECOMMERCE_CSV_PATH or place the dataset in the project folder or Downloads."
        )

    if csv_path.stat().st_size == 0:
        raise ValueError(
            f"CSV file is empty: {csv_path}. Please add the data or use the correct dataset file."
        )

    df = pd.read_csv(csv_path)

    df["Purchase_Amount"] = pd.to_numeric(
        df["Purchase_Amount"].replace(r"[$,]", "", regex=True), errors="coerce"
    )

    for col in [
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
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    numeric_cols = [
        "Age",
        "Purchase_Amount",
        "Frequency_of_Purchase",
        "Time_to_Decision",
    ]
    df_clean = df.dropna(subset=numeric_cols).copy()

    print("Dataset preview:")
    print(df_clean.head())
    print("\nColumns:")
    print(df_clean.columns.tolist())
    print("\nDataset information:")
    print(df_clean.info())
    print("\nSummary statistics:")
    print(df_clean.describe())
    print("\nMissing values per column:")
    print(df_clean.isnull().sum())

    duplicate_rows = df_clean[df_clean.duplicated()]
    if not duplicate_rows.empty:
        print(f"\nFound {len(duplicate_rows)} duplicate rows:")
        print(duplicate_rows.head())

    df_high_spenders = df_clean[df_clean["Purchase_Amount"] > 1000]
    print(f"\nNumber of high spenders: {len(df_high_spenders)}")
    if not df_high_spenders.empty:
        print("\nSample of high spenders:")
        print(df_high_spenders.head())

    avg_spending_by_age = df_clean.groupby("Age")["Purchase_Amount"].mean()
    print("\nAverage spending by age:")
    print(avg_spending_by_age.head(10))

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_clean, x="Age", bins=15, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("age_distribution.png")
    plt.close()

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df_cleaned = df_clean.dropna(subset=["Age", "Purchase_Amount"]).copy()
    X = df_cleaned[["Age"]]
    y = df_cleaned["Purchase_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel Coefficients:")
    print(f"Coefficient for Age: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    print(
        "\nRegression equation: "
        f"Purchase_Amount = {model.intercept_:.2f} + ({model.coef_[0]:.4f} * Age)"
    )

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-Squared: {r2}")
    print(
        "\nInterpretation: Age alone provides limited explanatory power for purchase amount "
        "in this dataset, so the relationship is weak and additional features would likely improve prediction quality."
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test["Age"], y_test, color="blue", label="Actual Amounts")
    plt.scatter(X_test["Age"], y_pred, color="red", label="Predicted Amounts")
    plt.title("Purchase Amount vs Age")
    plt.xlabel("Age")
    plt.ylabel("Purchase Amount")
    plt.legend()
    plt.grid()
    plt.savefig("purchase_amount_vs_age.png")
    plt.close()

    print("\nPOLARS ANALYSIS")

    pl_df = pl.read_csv(csv_path)
    pl_df = pl_df.with_columns(
        pl.col("Purchase_Amount")
        .str.replace_all(r"[$,]", "", literal=False)
        .cast(pl.Float64, strict=False)
        .alias("Purchase_Amount")
    )

    print("\nFirst 5 rows using Polars:")
    print(pl_df.head())
    print("\nDataset shape:")
    print(pl_df.shape)
    print("\nSummary statistics:")
    print(pl_df.describe())

    if "Purchase_Amount" in pl_df.columns:
        average_purchase = pl_df["Purchase_Amount"].mean()
        print(f"\nAverage Purchase Amount: {average_purchase}")


if __name__ == "__main__":
    main()
