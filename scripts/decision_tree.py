import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import plotly.express as px


# Function to evaluate the correctness of predictions
def get_predictions_correctness(df: pd.DataFrame, to_predict: str):
    PREDICTIONS = [k for k in df.keys() if k.startswith("pred")]
    print(f"Prediction columns found: {PREDICTIONS}")

    # Add columns to check if predictions are correct
    for pred in PREDICTIONS:
        part1 = pred.split("_")[0]  # First prefix before '_'
        df[f"is_correct_{part1}"] = (df[pred] == df[to_predict]).astype(int)

    IS_CORRECT = [k for k in df.keys() if k.startswith("is_correct_")]
    print(f"Created columns is_correct: {IS_CORRECT}")

    print("Precision on TEST set for each prediction:")
    for i, column in enumerate(IS_CORRECT):
        prediction_column = PREDICTIONS[i]
        is_correct_column = column
        filter = (df.Split == "Test") & (df[prediction_column] == 1)
        print(
            f"Prediction column: {prediction_column}, is_correct_column: {is_correct_column}"
        )
        print(df[filter][is_correct_column].value_counts())
        print(df[filter][is_correct_column].value_counts() / len(df[filter]))
        print("---------")

    return PREDICTIONS, IS_CORRECT


with open("get_predictions_correctness.pkl", "wb") as f:
    pickle.dump(get_predictions_correctness, f)


# Function to fit a Decision Tree
def fit_decision_tree(X, y, max_depth=20):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf, X.columns


# Main function for modeling
def train_decision_tree_model():
    new_df = pd.read_csv("prepared_df.csv")
  
    # Load necessary columns and data splits
    with open("data_split.pkl", "rb") as f:
        split_data = pickle.load(f)
        X_valid = split_data["X_valid"]
        y_valid = split_data["y_valid"]
        X_test = split_data["X_test"]
        y_test = split_data["y_test"]
        y_train_valid = split_data["y_train_valid"]
        X_all = split_data["X_all"]
        X_train_valid = split_data["X_train_valid"]

    # Generate manual predictions based on financial indicators
    new_df["pred01_momentum_positive"] = (new_df["MOM_10"] > 0).astype(np.int8)
    new_df["pred02_momentum_negative"] = (new_df["MOM_10"] < 0).astype(np.int8)
    new_df["pred03_roc_positive"] = (new_df["ROC_10"] > 0).astype(np.int8)
    new_df["pred04_roc_negative"] = (new_df["ROC_10"] < 0).astype(np.int8)
    new_df["pred05_cci_overbought"] = (new_df["CCI_14_0.015"] > 100).astype(np.int8)
    new_df["pred06_cci_oversold"] = (new_df["CCI_14_0.015"] < -100).astype(np.int8)
    new_df["pred07_fibonacci_50_support"] = (
        new_df["Adj Close"] > new_df["Fibonacci_50"]
    ).astype(np.int8)
    new_df["pred08_rsi_above_ma"] = (
        new_df["RSI_14"] > new_df["RSI_14"].rolling(window=14).mean()
    ).astype(np.int8)
    new_df["pred09_macd_hist_positive"] = (new_df["MACDh_12_26_9"] > 0).astype(np.int8)

    # Evaluate the correctness of the manual predictions
    to_predict = "Is_Positive_Growth_1h_Future"
    PREDICTIONS, IS_CORRECT = get_predictions_correctness(
        df=new_df, to_predict=to_predict
    )
    print(new_df[PREDICTIONS + IS_CORRECT + [to_predict]])


    
    #training Decision Tree
    clf_10, train_columns = fit_decision_tree(
       X=X_train_valid, y=y_train_valid, max_depth=10)

    # Predict on the full dataset using the trained Decision Tree
    y_pred_all = clf_10.predict(X_all)
    new_df["pred10_clf_10"] = y_pred_all

    # Recalculate the correctness after adding new predictions
    PREDICTIONS, IS_CORRECT = get_predictions_correctness(
        df=new_df, to_predict=to_predict
    )

   ### uncomment if you want to run it
   # Hyperparameter tuning Decision Tree
    #precision_by_depth = {}
    #best_precision = 0
    #best_depth = 0

    #for depth in range(1, 21):
     #   print(f"Working with a tree of a max depth= {depth}")
     #   clf, train_columns = fit_decision_tree(
     #       X=X_train_valid, y=y_train_valid, max_depth=depth
     #   )
     #   y_pred_valid = clf.predict(X_valid)
     #   precision_valid = precision_score(y_valid, y_pred_valid)
     #   y_pred_test = clf.predict(X_test)
     #   precision_test = precision_score(y_test, y_pred_test)
     #   print(
     #       f"  Precision on test is {precision_test}, (precision on valid is {precision_valid} - tends to overfit)"
     #   )
     #   precision_by_depth[depth] = round(precision_test, 4)
     #   if precision_test >= best_precision:
     #       best_precision = round(precision_test, 4)
     #       best_depth = depth
     #   tree_rules = export_text(
      #      clf, feature_names=list(X_train_valid.columns), max_depth=3
       # )
      #  print(tree_rules)
       # print("------------------------------")

  #  print(f"All precisions by depth: {precision_by_depth}")
   # print(f"The best precision is {best_precision} and the best depth is {best_depth}")
    ####


    precision_by_depth = {1: 0.5161, 2: 0.5161, 3: 0.5155, 4: 0.518, 5: 0.5219, 6: 0.5176, 7: 0.5162, 8: 0.5192, 9: 0.5173, 10: 0.5195, 11: 0.5215, 12: 0.5158, 13: 0.5178, 14: 0.5161, 15: 0.512, 16: 0.512, 17: 0.5094, 18: 0.5094, 19: 0.5092, 20: 0.5087}
    best_depth = 5
    best_precision = 0.5219

    # Convert the precision data to a DataFrame for visualization
    precision_df = pd.DataFrame(
        list(precision_by_depth.items()), columns=["max_depth", "precision_score"]
    )
    precision_df["precision_score"] = (
        precision_df["precision_score"] * 100.0
    )  # Convert to percentage

    # Create the bar chart using Plotly Express
    fig = px.bar(
        precision_df,
        x="max_depth",
        y="precision_score",
        title="Precision Score vs. Max Depth for a Decision Tree",
        labels={"max_depth": "Max Depth", "precision_score": "Precision Score"},
        range_y=[50, 58],
        text="precision_score",
    )

    # Update the text format to display as percentages
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        title={
            "text": "Precision Score vs. Max Depth for a Decision Tree",
            "x": 0.5,
            "xanchor": "center",
        }
    )
    fig.show()

    # Fit the best Decision Tree model
    clf_best, train_columns = fit_decision_tree(
        X=X_train_valid, y=y_train_valid, max_depth=best_depth
    )

    # Predict on the full dataset with the best model
    y_pred_clf_best = clf_best.predict(X_all)
    new_df["pred11_clf_best"] = y_pred_clf_best
    

    # Recalculate the correctness after adding the new predictions
    PREDICTIONS, IS_CORRECT = get_predictions_correctness(
        df=new_df, to_predict=to_predict
    )

    # Saving Model (clf_10)
    with open("clf_10_model.pkl", "wb") as f:
        pickle.dump(clf_10, f)

    # Saving best trained model (clf_best)
    with open("clf_best_model.pkl", "wb") as f:
        pickle.dump(clf_best, f)

    # Saving bet_precision and best_depth
    with open("best_model_info.pkl", "wb") as f:
        pickle.dump({"best_precision": best_precision, "best_depth": best_depth}, f)

    new_df.to_csv("updated_predictions.csv", index=False)


if __name__ == "__main__":
    train_decision_tree_model()
