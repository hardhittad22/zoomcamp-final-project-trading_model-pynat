import pandas as pd
import numpy as np
import plotly.express as px
import pickle


def load_data():
    # Load data from CSV files
    new_df = pd.read_csv("updated_predictions3.csv")
    with open("get_predictions_correctness.pkl", "rb") as f:
        get_predictions_correctness = pickle.load(f)
    TO_PREDICT = "Is_Positive_Growth_1h_Future"
    PREDICTIONS, IS_CORRECT = get_predictions_correctness(new_df, TO_PREDICT)

    return new_df, TO_PREDICT, PREDICTIONS, IS_CORRECT


def trading_simulation():
    # Load data
    new_df, TO_PREDICT, PREDICTIONS = load_data()

    # Prediction
    pred = "pred14_lr"

    # Print total number of investment opportunities and days
    print(new_df[new_df.Split == "Test"].Close.count())
    print(new_df[new_df.Split == "Test"].Date.nunique())
    print(new_df[["Growth_Future_1h", "Is_Positive_Growth_1h_Future", pred]])

    # Define stop-loss and take-profit thresholds
    stop_loss_threshold = -0.02
    take_profit_threshold = 0.07

    # Strategy1 Generate trade signal by combining multiple prediction models
    new_df["Trade_Signal1"] = (
        (new_df["pred11_clf_best"] == 1)  # Model 11: Best classifier prediction
        & (new_df["pred12_rf"] == 1)  # Model 12: Random forest prediction
        & (
            new_df["pred30_lr_best_rule_50"] == 1
        )  # Model 30: Logistic regression with custom rule
        & (
            new_df["pred13_rf_reduced"] == 1
        )  # Model 13: Reduced random forest prediction
        & (
            new_df["pred09_macd_hist_positive"] == 1
        )  # MACD histogram is positive (bullish signal)
        & (
            new_df["pred05_cci_overbought"] == 0
        )  # CCI indicates not overbought (avoiding potential reversal)
    )

    # Calculate gross revenue with clipping to account for stop loss and take profit limits
    new_df["sim1_gross_rev_pred14"] = (
        new_df["Trade_Signal1"]
        * 150
        * (
            new_df["Growth_Future_1h"].clip(
                lower=(1 + stop_loss_threshold), upper=(1 + take_profit_threshold)
            )
            - 1
        )
    )

    # Calculate trading fees (assumed 0.2% per trade)
    new_df["sim1_fees_pred14"] = new_df["Trade_Signal1"].abs() * 0.002
    # Calculate net revenue by subtracting fees from gross revenue
    new_df["sim1_net_rev_pred14"] = (
        new_df["sim1_gross_rev_pred14"] - new_df["sim1_fees_pred14"]
    )
    # Output the total net revenue from all trades
    print(new_df["sim1_net_rev_pred14"].sum())

    # show it for all predictions
    COLUMNS_FIN_RESULT = (
        ["Date", "Ticker", "Close"]
        + TO_PREDICT
        + [pred, "sim1_gross_rev_pred14", "sim1_fees_pred14", "sim1_net_rev_pred14"]
    )
    new_df[(new_df.Split == "Test") & (new_df[pred] == 1)][COLUMNS_FIN_RESULT]

    # DAILY INVESTMENTS COUNT
    filter_test_and_positive_pred = (new_df.Split == "Test") & (new_df[pred] == 1)
    df_investments_count_daily = pd.DataFrame(
        new_df[filter_test_and_positive_pred].groupby("Date")[pred].count()
    )
    sim1_avg_investments_per_day = df_investments_count_daily[pred].quantile(0.75)
    print(sim1_avg_investments_per_day)

    # Distribution: how many times do we trade daily (for the current Prediction)?
    df_investments_count_daily.describe().T
    new_df[(new_df.Split == "Test") & (new_df[pred] == 1)][
        ["sim1_gross_rev_pred14", "sim1_fees_pred14", "sim1_net_rev_pred14"]
    ].sum()

    ###

    # Strategy2 enerate trade signal based on selected features (as indicated by the random forest model)
    new_df["Trade_Signal2"] = (
        new_df["BBP_5_2.0"] > 0.7
    ) & (  # Bollinger Band Position above 0.7 (indicating price is near the upper band)
        new_df["BOP"] > 0
    )  # Positive Balance of Power (indicating bullish momentum)

    # Calculate gross revenue with clipping to account for stop loss and take profit limits
    new_df["sim2_gross_rev_pred14"] = (
        new_df["Trade_Signal2"]
        * 150
        * (
            new_df["Growth_Future_1h"].clip(
                lower=(1 + stop_loss_threshold), upper=(1 + take_profit_threshold)
            )
            - 1
        )
    )
    # Calculate trading fees (assumed 0.2% per trade)
    new_df["sim2_fees_pred14"] = new_df["Trade_Signal2"].abs() * 0.002
    # Calculate net revenue by subtracting fees from gross revenue
    new_df["sim2_net_rev_pred14"] = (
        new_df["sim2_gross_rev_pred14"] - new_df["sim2_fees_pred14"]
    )

    # Output the total net revenue from all trades
    print(new_df["sim2_net_rev_pred14"].sum())
    COLUMNS_FIN_RESULT = (
        ["Date", "Ticker", "Close"]
        + TO_PREDICT
        + [pred, "sim2_gross_rev_pred14", "sim2_fees_pred14", "sim2_net_rev_pred14"]
    )
    new_df[(new_df.Split == "Test") & (new_df[pred] == 1)][COLUMNS_FIN_RESULT]

    # DAILY INVESTMENTS COUNT
    filter_test_and_positive_pred = (new_df.Split == "Test") & (new_df[pred] == 1)
    df_investments_count_daily = pd.DataFrame(
        new_df[filter_test_and_positive_pred].groupby("Date")[pred].count()
    )
    sim2_avg_investments_per_day = df_investments_count_daily[pred].quantile(0.75)
    print(sim2_avg_investments_per_day)

    new_df[(new_df.Split == "Test") & (new_df[pred] == 1)][
        ["sim2_gross_rev_pred14", "sim2_fees_pred14", "sim2_net_rev_pred14"]
    ].sum()

    ###

    # Strategy3 Volatility strategy using ATR and Bollinger Bands
    new_df["Trade_Signal3"] = (
        new_df["ATRr_14"] > 0.015
    ) & (  # High volatility indicated by ATR
        new_df["BBP_5_2.0"] < 0.2
    )  # Price near the lower Bollinger Band (potentially oversold)

    # Calculate gross revenue with stop-loss and take-profit limits applied
    new_df["sim3_gross_rev_pred14"] = (
        new_df["Trade_Signal3"]
        * 150
        * (
            new_df["Growth_4h"].clip(
                lower=(1 + stop_loss_threshold), upper=(1 + take_profit_threshold)
            )
            - 1
        )
    )

    # Calculate trading fees (assumed 0.2% per trade)
    new_df["sim3_fees_pred14"] = new_df["Trade_Signal3"].abs() * 0.002
    # Calculate net revenue by subtracting fees from gross revenue
    new_df["sim3_net_rev_pred14"] = (
        new_df["sim3_gross_rev_pred14"] - new_df["sim3_fees_pred14"]
    )
    # Output the total net revenue from all trades
    print(new_df["sim3_net_rev_pred14"].sum())

    COLUMNS_FIN_RESULT = (
        ["Date", "Ticker", "Close"]
        + TO_PREDICT
        + [pred, "sim3_gross_rev_pred14", "sim3_fees_pred14", "sim3_net_rev_pred14"]
    )
    new_df[(new_df.Split == "Test") & (new_df[pred] == 1)][COLUMNS_FIN_RESULT]

    # DAILY INVESTMENTS COUNT
    filter_test_and_positive_pred = (new_df.Split == "Test") & (new_df[pred] == 1)
    df_investments_count_daily = pd.DataFrame(
        new_df[filter_test_and_positive_pred].groupby("Date")[pred].count()
    )
    sim3_avg_investments_per_day = df_investments_count_daily[pred].quantile(
        0.75
    )  # 75% case - how many $100 investments per day do we have?
    print(sim3_avg_investments_per_day)

    new_df[(new_df.Split == "Test") & (new_df[pred] == 1)][
        ["sim3_gross_rev_pred14", "sim3_fees_pred14", "sim3_net_rev_pred14"]
    ].sum()

    ###

    # Strategy4
    new_df["Trade_Signal_3"] = (
        new_df["SMA_Cross"] == 1
    ) & (  # SMA-Kreuzung als Trendwechsel
        new_df["ADX_14"] > 25
    )  # Starker Trend (ADX über 25)

    # Berechnung des Bruttogewinns mit Stop-Loss und Take-Profit
    new_df["sim4_gross_rev_pred14"] = (
        new_df["Trade_Signal4"]
        * 150
        * (
            new_df["Growth_24h"].clip(
                lower=(1 + stop_loss_threshold), upper=(1 + take_profit_threshold)
            )
            - 1
        )
    )

    new_df["sim4_fees_pred14"] = new_df["Trade_Signal4"].abs() * 0.002
    new_df["sim4_net_rev_pred14"] = (
        new_df["sim4_gross_rev_pred14"] - new_df["sim4_fees_pred14"]
    )
    print(new_df["sim4_net_rev_pred14"].sum())
    COLUMNS_FIN_RESULT = (
        ["Date", "Ticker", "Close"]
        + TO_PREDICT
        + [pred, "sim4_gross_rev_pred14", "sim4_fees_pred14", "sim4_net_rev_pred14"]
    )
    new_df[(new_df.Split == "Test") & (new_df[pred] == 1)][COLUMNS_FIN_RESULT]

    # DAILY INVESTMENTS COUNT
    filter_test_and_positive_pred = (new_df.Split == "Test") & (new_df[pred] == 1)
    df_investments_count_daily = pd.DataFrame(
        new_df[filter_test_and_positive_pred].groupby("Date")[pred].count()
    )
    sim4_avg_investments_per_day = df_investments_count_daily[pred].quantile(
        0.75
    )  # 75% case - how many $100 investments per day do we have?
    print(sim4_avg_investments_per_day)

    new_df[(new_df.Split == "Test") & (new_df[pred] == 1)][
        ["sim4_gross_rev_pred14", "sim4_fees_pred14", "sim4_net_rev_pred14"]
    ].sum()

    # Calculate final results for all fields
    sim3_results = []  # initialize array

    # Strategie definieren
    new_df["Trade_Signal3"] = (new_df["ATRr_14"] > 0.015) & (  # high volatilität
        new_df["BBP_5_2.0"] < 0.2
    )  # touching lower bb

    # calculating revenue for all PREDICTIONS
    for pred in PREDICTIONS:
        print(f"Calculating simulation for prediction {pred}:")

        gross_rev_column = f"sim3_gross_rev_{pred}"
        fees_column = f"sim3_fees_{pred}"
        net_rev_column = f"sim3_net_rev_{pred}"

        # for trade signal 'Trade_Signal3'
        if pred == "Trade_Signal3":
            new_df[gross_rev_column] = (
                new_df["Trade_Signal3"]
                * 150
                * (
                    new_df["Growth_4h"].clip(
                        lower=(1 + stop_loss_threshold),
                        upper=(1 + take_profit_threshold),
                    )
                    - 1
                )
            )
        else:
            new_df[gross_rev_column] = new_df[pred] * 150 * (new_df["Growth_4h"] - 1)

        # calculating fees
        new_df[fees_column] = -abs(new_df[pred]) * 0.002
        new_df[net_rev_column] = new_df[gross_rev_column] + new_df[fees_column]

        # calculate results
        filter_test_and_positive_pred = (new_df.Split == "Test") & (new_df[pred] == 1)
        sim3_count_investments = len(new_df[filter_test_and_positive_pred])
        sim3_gross_rev = new_df[filter_test_and_positive_pred][gross_rev_column].sum()
        sim3_fees = new_df[filter_test_and_positive_pred][fees_column].sum()
        sim3_net_rev = new_df[filter_test_and_positive_pred][net_rev_column].sum()

        if sim3_gross_rev > 0:
            sim3_fees_percentage = -sim3_fees / sim3_gross_rev
        else:
            sim3_fees_percentage = None

        if sim3_count_investments > 0:
            sim3_average_net_revenue = sim3_net_rev / sim3_count_investments
        else:
            sim3_average_net_revenue = None

        # capital and CAGR
        df_investments_count_daily = pd.DataFrame(
            new_df[filter_test_and_positive_pred].groupby("Date")[pred].count()
        )
        sim3_avg_investments_per_day = df_investments_count_daily[pred].mean()
        sim3_q75_investments_per_day = df_investments_count_daily[pred].quantile(0.75)
        sim3_capital = 150 * 5 * sim3_q75_investments_per_day
        sim3_CAGR = ((sim3_capital + sim3_net_rev) / sim3_capital) ** (
            1 / 4
        )  # -1 laut chatgpt

        # safe to array
        sim3_results.append(
            (
                pred,
                sim3_count_investments,
                sim3_gross_rev,
                sim3_fees,
                sim3_net_rev,
                sim3_fees_percentage,
                sim3_average_net_revenue,
                sim3_avg_investments_per_day,
                sim3_capital,
                sim3_CAGR,
            )
        )

        # print results
        if sim3_count_investments > 1:
            print(
                f" Financial Result: \n {new_df[filter_test_and_positive_pred][[gross_rev_column, fees_column, net_rev_column]].sum()}"
            )
            print(f" Count Investments in 4 years (on TEST): {sim3_count_investments}")
            print(f" Gross Revenue: ${int(sim3_gross_rev)}")
            print(f" Fees (0.2% for buy+sell): ${int(-sim3_fees)}")
            print(f" Net Revenue: ${int(sim3_net_rev)}")
            print(
                f" Fees are {int(-10.0 * sim3_fees / sim3_gross_rev)} % from Gross Revenue"
            )
            print(f" Capital Required : ${int(sim3_capital)} (Vbegin)")
            print(
                f" Final value (Vbegin + Net_revenue) : ${int(sim3_capital + sim3_net_rev)} (Vfinal)"
            )
            print(
                f" Average CAGR on TEST (4 years) : {np.round(sim3_CAGR, 3)}, or {np.round(10.0 * (sim3_CAGR - 1), 1)}% "
            )
            print(f" Average daily stats: ")
            print(
                f" Average net revenue per investment: ${np.round(sim3_net_rev / sim3_count_investments, 2)} "
            )
            print(
                f" Average investments per day: {int(np.round(sim3_avg_investments_per_day))} "
            )
            print(
                f" Q75 investments per day: {int(np.round(sim3_q75_investments_per_day))} "
            )
            print("=============================================+")

    # save to df
    columns_simulation = [
        "prediction",
        "sim3_count_investments",
        "sim3_gross_rev",
        "sim3_fees",
        "sim3_net_rev",
        "sim3_fees_percentage",
        "sim3_average_net_revenue",
        "sim3_avg_investments_per_day",
        "sim3_capital",
        "sim3_CAGR",
    ]

    df_sim3_results = pd.DataFrame(sim3_results, columns=columns_simulation)

    df_sim3_results["sim3_growth_capital_4y"] = (
        df_sim3_results.sim3_net_rev + df_sim3_results.sim3_capital
    ) / df_sim3_results.sim3_capital
    print(df_sim3_results)

    # Create the scatter plot
    fig = px.scatter(
        df_sim3_results.dropna(),
        x="sim3_avg_investments_per_day",
        y="sim3_CAGR",
        size="sim3_growth_capital_4y",  # Use the 'size' parameter for sim1_CAGR
        text="prediction",
        title="Compound Annual Growth vs. Time spent (Average investments per day)",
        labels={
            "sim3_capital": "Initial Capital Requirement",
            "growth_capital_4y": "4-Year Capital Growth",
        },
    )

    # Update the layout to improve readability of the annotations
    fig.update_traces(textposition="top center")

    # Show the plot
    fig.show()


if __name__ == "__main__":
    trading_simulation()
