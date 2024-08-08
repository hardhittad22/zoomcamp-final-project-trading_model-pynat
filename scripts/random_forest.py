import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score


def fit_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test):
    precision_matrix = {}
    best_precision = 0
    best_depth = 0
    best_estimators = 1

    # Hyperparameter tuning for Random Forest
    for depth in [15, 16, 17, 18, 19, 20]:
        for estimators in [50, 100, 200, 500]:
            print(
                f"Working with HyperParams: depth = {depth}, estimators = {estimators}"
            )
            start_time = time.time()

            # Initialize and train Random Forest
            rf = RandomForestClassifier(
                n_estimators=estimators, max_depth=depth, random_state=42, n_jobs=-1
            )

            rf.fit(X_train, y_train)

            # Validate on validation and test sets
            y_pred_valid = rf.predict(X_valid)
            precision_valid = precision_score(y_valid, y_pred_valid)
            y_pred_test = rf.predict(X_test)
            precision_test = precision_score(y_test, y_pred_test)
            print(
                f"  Precision on test is {precision_test}, (precision on valid is {precision_valid} - tends to overfit)"
            )

            precision_matrix[(depth, estimators)] = round(precision_test, 4)

            elapsed_time = time.time() - start_time
            print(
                f"Time for training: {elapsed_time:.2f} seconds, or {elapsed_time / 60:.2f} minutes"
            )

            # Track the best model parameters
            if precision_test >= best_precision:
                best_precision = round(precision_test, 4)
                best_depth = depth
                best_estimators = estimators
                print(
                    f"New best precision found for depth={depth}, estimators = {estimators}"
                )

            print("------------------------------")

    print(f"Matrix of precisions: {precision_matrix}")
    print(
        f"The best precision is {best_precision} and the best depth is {best_depth} and best estimators = {best_estimators}"
    )

    # Train the final model with the best parameters
    rf_model = RandomForestClassifier(
        max_depth=best_depth, n_estimators=best_estimators, random_state=42
    )
    rf_model.fit(X_train, y_train)

    return rf_model, best_depth, best_estimators


def tpr_fpr_dataframe(y_true, y_pred, only_even=False):
    scores = []

    # Generate thresholds for TPR/FPR calculation
    if only_even == False:
        thresholds = np.linspace(0, 1, 101)
    else:
        thresholds = np.linspace(0, 1, 51)

    # Calculate TPR, FPR, precision, recall, and accuracy for each threshold
    for t in thresholds:
        actual_positive = y_true == 1
        actual_negative = y_true == 0

        predict_positive = y_pred >= t
        predict_negative = y_pred < t

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))

    columns = [
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "accuracy",
        "f1_score",
    ]
    df_scores = pd.DataFrame(scores, columns=columns)

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    df_scores["tpr"] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores["fpr"] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores


def analyze_rf_model(new_df, rf_model, X_all, X_test, y_test):
    # Predict probabilities on the test set
    y_pred_test = rf_model.predict_proba(X_test)
    y_pred_test_class1 = [k[1] for k in y_pred_test]

    # Show the distribution of predicted probabilities
    y_pred_test_class1_df = pd.DataFrame(
        y_pred_test_class1, columns=["Class1_probability"]
    )
    print(y_pred_test_class1_df.head())

    sns.histplot(y_pred_test_class1)
    plt.title("The distribution of predictions for Random Forest")
    plt.show()

    # Generate TPR/FPR dataframe
    df_scores = tpr_fpr_dataframe(y_test, y_pred_test_class1, only_even=True)
    print(df_scores[(df_scores.threshold >= 0.32) & (df_scores.threshold <= 0.50)])

    # Plot precision, recall, and F1 score vs. thresholds
    df_scores.plot.line(
        x="threshold",
        y=["precision", "recall", "f1_score"],
        title="Precision vs. Recall for the Best Model (Random Forest)",
    )
    plt.show()

    # Predict probabilities on the entire dataset
    y_pred_all = rf_model.predict_proba(X_all)
    y_pred_all_class1 = [k[1] for k in y_pred_all]
    y_pred_all_class1_array = np.array(y_pred_all_class1)

    # Add predictions based on various thresholds to the dataframe
    new_df["proba_pred15"] = y_pred_all_class1_array
    new_df["pred15_rf_best_rule_32"] = (y_pred_all_class1_array >= 0.32).astype(int)

    new_df["proba_pred16"] = y_pred_all_class1_array
    new_df["pred16_rf_best_rule_35"] = (y_pred_all_class1_array >= 0.35).astype(int)

    new_df["proba_pred17"] = y_pred_all_class1_array
    new_df["pred17_rf_best_rule_37"] = (y_pred_all_class1_array >= 0.37).astype(int)

    new_df["proba_pred18"] = y_pred_all_class1_array
    new_df["pred18_rf_best_rule_39"] = (y_pred_all_class1_array >= 0.39).astype(int)

    new_df["proba_pred19"] = y_pred_all_class1_array
    new_df["pred19_rf_best_rule_41"] = (y_pred_all_class1_array >= 0.41).astype(int)

    new_df["proba_pred20"] = y_pred_all_class1_array
    new_df["pred20_rf_best_rule_43"] = (y_pred_all_class1_array >= 0.43).astype(int)

    new_df["proba_pred21"] = y_pred_all_class1_array
    new_df["pred21_rf_best_rule_45"] = (y_pred_all_class1_array >= 0.45).astype(int)

    new_df["proba_pred22"] = y_pred_all_class1_array
    new_df["pred22_rf_best_rule_47"] = (y_pred_all_class1_array >= 0.47).astype(int)

    new_df["proba_pred23"] = y_pred_all_class1_array
    new_df["pred23_rf_best_rule_49"] = (y_pred_all_class1_array >= 0.49).astype(int)

    new_df["proba_pred24"] = y_pred_all_class1_array
    new_df["pred24_rf_best_rule_50"] = (y_pred_all_class1_array >= 0.50).astype(
        int
    )  # best one

    return new_df


def train_random_forest_model():
    # Load preprocessed dataset and additional data
    new_df = pd.read_csv("updated_predictions.csv")

    with open("get_predictions_correctness.pkl", "rb") as f:
        get_predictions_correctness = pickle.load(f)

    with open("column_lists.pkl", "rb") as f:
        data = pickle.load(f)
        NUMERICAL = data["NUMERICAL"]
        GROWTH = data["GROWTH"]
        DUMMIES = data["DUMMIES"]
        TO_PREDICT = data["TO_PREDICT"]

    with open("data_split.pkl", "rb") as f:
        split_data = pickle.load(f)
        X_valid = split_data["X_valid"]
        y_valid = split_data["y_valid"]
        X_test = split_data["X_test"]
        y_test = split_data["y_test"]
        y_train_valid = split_data["y_train_valid"]
        X_all = split_data["X_all"]
        X_train_valid = split_data["X_train_valid"]

    # Load lists of columns
    with open("column_lists.pkl", "rb") as f:
        data = pickle.load(f)
        NUMERICAL = data["NUMERICAL"]
        TO_PREDICT = data["TO_PREDICT"]

    # Load split datasets
    with open("data_split.pkl", "rb") as f:
        split_data = pickle.load(f)
        X_valid = split_data["X_valid"]
        y_valid = split_data["y_valid"]
        X_test = split_data["X_test"]
        y_test = split_data["y_test"]
        y_train = split_data["y_train"]
        X_all = split_data["X_all"]
        X_train = split_data["X_train"]

    # Prepare training and validation data
    X_train_valid = new_df[new_df.Split.isin(["Train", "Validation"])].drop(
        columns=[TO_PREDICT]
    )
    y_train_valid = new_df[new_df.Split.isin(["Train", "Validation"])][TO_PREDICT]

    # Fit the Random Forest model and find the best hyperparameters
    rf_model, best_depth, best_estimators = fit_random_forest(
        X_train=X_train_valid,
        y_train=y_train_valid,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )

    # Analyze model performance and generate predictions based on various thresholds
    new_df = analyze_rf_model(new_df, rf_model, X_all, X_test, y_test)

    # Output the first few rows of the updated dataframe
    print(new_df.head())

    new_df.to_csv("updated_predictions2.csv", index=False)


if __name__ == "__main__":
    train_random_forest_model()
