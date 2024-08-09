import pandas as pd
import numpy as np
import pickle


def temporal_split(
    df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15
):
    #min_date = pd.to_datetime(min_date) if isinstance(min_date, str) else min_date
   # max_date = pd.to_datetime(max_date) if isinstance(max_date, str) else max_date
    # Convert min_date and max_date to pandas Timestamp if they are strings
   # min_date = pd.Timestamp(min_date)
   # max_date = pd.Timestamp(max_date)

    # Define the end dates for train and validation sets based on proportions
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

    # Initialize an empty list to hold the split labels
    split_labels = []

    # Assign split labels based on date ranges
    for date in df["Date"]:
        if date <= train_end:
            split_labels.append("Train")
        elif date <= val_end:
            split_labels.append("Validation")
        else:
            split_labels.append("Test")

    # Add the 'Split' column to the DataFrame with the computed labels
    df["Split"] = split_labels

    return df


def clean_dataframe_from_inf_and_nan(df):
    """
    Cleans the DataFrame by replacing infinite values and NaNs.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame with NaNs and infinite values handled.
    """
    # Convert categorical columns to object type temporarily
    cat_cols = df.select_dtypes(include=["category"]).columns
    df[cat_cols] = df[cat_cols].astype("object")

    # Replace +-inf with NaN 
    df = df.replace([np.inf, -np.inf], np.nan)
     # Fill NaN values with 0
    df = df.fillna(0)


    # Convert object columns back to categorical
    df[cat_cols] = df[cat_cols].astype("category")

    return df


def split_and_cleaning():
    df_with_all_dummies = pd.read_csv("df_with_all_dummies.csv")

    with open("column_lists.pkl", "rb") as f:
        data = pickle.load(f)
        NUMERICAL = data["NUMERICAL"]
        GROWTH = data["GROWTH"]
        DUMMIES = data["DUMMIES"]
        TO_PREDICT = data["TO_PREDICT"]
        CUSTOM_NUMERICAL = data["CUSTOM_NUMERICAL"]
        TECHNICAL_INDICATORS = data["TECHNICAL_INDICATORS"]
    

    # Ensure 'Date' column is in datetime format
    df_with_all_dummies["Date"] = pd.to_datetime(df_with_all_dummies["Date"], utc=True)

    # Get minimum and maximum dates from the DataFrame
    min_date_df = df_with_all_dummies.Date.min()
    max_date_df = df_with_all_dummies.Date.max()

    # Apply the temporal_split function to the DataFrame
    df_with_all_dummies = temporal_split(
        df_with_all_dummies, min_date=min_date_df, max_date=max_date_df
    )

    # Print the proportion of each split
    print(df_with_all_dummies["Split"].value_counts() / len(df_with_all_dummies))

    # Create a copy of the DataFrame for further analysis
    new_df = df_with_all_dummies.copy()

    # Group by 'Split' and aggregate statistics on the 'Date' column
    print(new_df.groupby(["Split"])["Date"].agg(["min", "max", "count"]))

    # Set pandas display option to show all columns
    pd.set_option("display.max_columns", None)

    # Print the last row of the DataFrame
    print(new_df.tail(1))

    # Replace with actual dummy features if any
    features_list = NUMERICAL + DUMMIES

    # Print the features list
    print(f"features list: {features_list}")

    to_predict = "Is_Positive_Growth_1h_Future"

    # Split the DataFrame into training, validation, and test sets
    train_df = new_df[new_df.Split.isin(["Train"])].copy(deep=True)
    valid_df = new_df[new_df.Split.isin(["Validation"])].copy(deep=True)
    train_valid_df = new_df[new_df.Split.isin(["Train", "Validation"])].copy(deep=True)
    test_df = new_df[new_df.Split.isin(["Test"])].copy(deep=True)

    # Separate features and target variable for training and testing sets
    X_train = train_df[features_list + [to_predict]]
    X_valid = valid_df[features_list + [to_predict]]
    X_train_valid = train_valid_df[features_list + [to_predict]]
    X_test = test_df[features_list + [to_predict]]

    # Predictions and join to the original DataFrame
    X_all = new_df[features_list + [to_predict]].copy(deep=True)

    # Print shapes of the datasets
    print(
        f"length: X_train {X_train.shape}, X_validation {X_valid.shape}, X_test {X_test.shape}, X_train_valid = {X_train_valid.shape}, all combined: X_all {X_all.shape}"
    )

    # Clean the DataFrames from infinite values and NaNs
    X_train = clean_dataframe_from_inf_and_nan(X_train)
    X_valid = clean_dataframe_from_inf_and_nan(X_valid)
    X_train_valid = clean_dataframe_from_inf_and_nan(X_train_valid)
    X_test = clean_dataframe_from_inf_and_nan(X_test)
    X_all = clean_dataframe_from_inf_and_nan(X_all)

    # Extract target variables
    y_train = X_train[to_predict]
    y_valid = X_valid[to_predict]
    y_train_valid = X_train_valid[to_predict]
    y_test = X_test[to_predict]
    y_all = X_all[to_predict]

    # Remove target variable from feature DataFrames
    del X_train[to_predict]
    del X_valid[to_predict]
    del X_train_valid[to_predict]
    del X_test[to_predict]
    del X_all[to_predict]

    with open("data_split.pkl", "wb") as f:
        pickle.dump(
            {
                "X_train": X_train,
                "X_valid": X_valid,
                "X_train_valid": X_train_valid,
                "X_test": X_test,
                "X_all": X_all,
                "y_train": y_train,
                "y_valid": y_valid,
                "y_train_valid": y_train_valid,
                "y_test": y_test,
                "y_all": y_all                   
            },
            f,
        )
   # new_df = new_df.drop(columns=["Date"])

    new_df.to_csv("prepared_df.csv", index=False)

if __name__ == "__main__":
    split_and_cleaning()
