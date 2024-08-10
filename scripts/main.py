# Import all Functions from scripts
from data_preparation import get_data
from modeling import modeling
from trading_simulation import trading_simulation


def main():
    """
    Main function to execute the end-to-end data workflow.
    """
    print("Starting data download and preparation...")

    # Calling functions to download and prepare data
    # This includes merging them to one DF and creatng dummies 
    
    get_data()
    

    # Training Data for Decision Tree, adding predictions
    # This file takes some time, if you want to reduce the time spent, comment feature importance in rf out
    print("Performing modeling and predictions...")
    modeling()


    # Simulate different strategies, printing all predictions for the best strategy
    print("Performing all calculations and simulations...")
    trading_simulation()

    print("Workflow completed successfully!")


if __name__ == "__main__":
    main()
