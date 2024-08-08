# Import all Functions from scripts
from scripts.get_stocks import top_stocks
from scripts.get_crypto import get_coins # needs to run locally in vscode not colab due to binance settings otherwise error 451
from scripts.all_calculations import calculations
from scripts.prepare_and_dummies import prepare
from scripts.temporal_split_and_cleaning import split_and_cleaning
from scripts.decision_tree import train_decision_tree_model
from scripts.random_forest import train_random_forest_model
from scripts.logistic_regression import train_logistic_regression_model
from scripts.trading_simulation import trading_simulation


def main():
    """
    Main function to execute the end-to-end data workflow.
    """
    print("Starting data download and preparation...")

    # Calling functions to download and prepare data
    top_stocks()
    get_coins()

    # Calling functions to prepare the data and add features
    print("Performing all calculations and simulations...")
    calculations()

    # Calling functions to prepare the DataFrame for modeling
    print("Performing data preparation and transformation...")
    prepare()

    # Splitting the DataFrame
    print("Splitting and cleaning data...")
    split_and_cleaning()

    # Training Data for Decision Tree, adding predictions
    print("Performing modeling and predictions...")
    train_decision_tree_model()

    # Trainging Data for Random Forest, adding predictions and new tresholds
    print("Performing modeling and predictions...")
    train_random_forest_model()

    # Trainging Data for Logistic Regressoin, adding predictions and new tresholds
    print("Performing modeling and predictions...")
    train_logistic_regression_model()

    # Simulate different strategies, printing all predictions for the best strategy
    print("Performing all calculations and simulations...")
    trading_simulation()

    print("Workflow completed successfully!")


if __name__ == "__main__":
    main()
