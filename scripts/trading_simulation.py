# imports
import numpy as np
import pandas as pd
import pickle

#finance
import yfinance as yf

# visualisation
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# time
from datetime import date, datetime, timedelta

# ML models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree, export_text, DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

#from Modeling import get_predictions_correctness # hier muss ich funktion importieren

def trading_simulation():

    #Imports from other files
    new_df = pd.read_csv("modeled_df.csv")
    with open("column_lists.pkl", "rb") as f:
            data = pickle.load(f)
            TO_PREDICT = data["TO_PREDICT"]
    
    with open('lists_pred.pkl', 'rb') as f:
        PREDICTIONS, IS_CORRECT = pickle.load(f)
        

    """# 4. Trading Simulations

    ## 4.1 Investing $150 on every positive prediction    

    fees = 0.1% for each buy and sell operation ==> 0.2% for buy+sell operation
    """

    pred = 'pred14_lr'

    # Total Number of Investment Opportunities
    new_df[new_df.Split=='Test']['Adj Close'].count()

    # Total Number of Days
    new_df[new_df.Split=='Test'].Date.nunique()

    # check actual future growth
    print(TO_PREDICT)

    new_df[['Growth_Future_1h','Is_Positive_Growth_1h_Future',pred]]

    """### 4.1.1 Strategy 1"""

    # Define thresholds for stop loss and take profit
    stop_loss_threshold = -0.02  # Stop loss at -2%
    take_profit_threshold = 0.07  # Take profit at +7%

    # Generate trade signal by combining multiple prediction models
    new_df['Trade_Signal_1'] = (
        (new_df['pred11_clf_best'] == 1) &       # Model 11: Best classifier prediction
        (new_df['pred12_rf'] == 1) &             # Model 12: Random forest prediction
        (new_df['pred30_lr_best_rule_50'] == 1) & # Model 30: Logistic regression with custom rule
        (new_df['pred13_rf_reduced'] == 1) &     # Model 13: Reduced random forest prediction
        (new_df['pred09_macd_hist_positive'] == 1) & # MACD histogram is positive (bullish signal)
        (new_df['pred05_cci_overbought'] == 0)   # CCI indicates not overbought (avoiding potential reversal)
    )

    # Calculate gross revenue with clipping to account for stop loss and take profit limits
    new_df['sim1_gross_rev_pred14'] = new_df['Trade_Signal_1'] * 150 * (
        new_df['Growth_Future_1h'].clip(lower=(1 + stop_loss_threshold), upper=(1 + take_profit_threshold)) - 1
    )

    # Calculate trading fees (assumed 0.2% per trade)
    new_df['sim1_fees_pred14'] = new_df['Trade_Signal_1'].abs() * 0.002

    # Calculate net revenue by subtracting fees from gross revenue
    new_df['sim1_net_rev_pred14'] = new_df['sim1_gross_rev_pred14'] - new_df['sim1_fees_pred14']

    # Output the total net revenue from all trades
    print(new_df['sim1_net_rev_pred14'].sum())

    #show it for all predictions
    COLUMNS_FIN_RESULT = ['Date','Ticker','Adj Close']+TO_PREDICT+ [pred, 'sim1_gross_rev_pred14','sim1_fees_pred14','sim1_net_rev_pred14']
    new_df[(new_df.Split=='Test')&(new_df[pred]==1)][COLUMNS_FIN_RESULT]

    # DAILY INVESTMENTS COUNT
    filter_test_and_positive_pred = (new_df.Split=='Test')&(new_df[pred]==1)
    df_investments_count_daily = pd.DataFrame(new_df[filter_test_and_positive_pred].groupby('Date')[pred].count())
    sim1_avg_investments_per_day = df_investments_count_daily[pred].quantile(0.75)
    print(sim1_avg_investments_per_day)

    # Distribution: how many times do we trade daily (for the current Prediction)?
    df_investments_count_daily.describe().T

    new_df[(new_df.Split=='Test')&(new_df[pred]==1)][['sim1_gross_rev_pred14','sim1_fees_pred14','sim1_net_rev_pred14']].sum()

    """### 4.1.2 Strategy 2"""

    # Generate trade signal based on selected features (as indicated by the random forest model)

    new_df['Trade_Signal_2'] = (
        (new_df['BBP_5_2.0'] > 0.7) &  # Bollinger Band Position above 0.7 (indicating price is near the upper band)
        (new_df['BOP'] > 0)  # Positive Balance of Power (indicating bullish momentum)
    )

    # Define thresholds for stop loss and take profit
    stop_loss_threshold = -0.02  # Stop loss at -2%
    take_profit_threshold = 0.07  # Take profit at +7%

    # Calculate gross revenue with clipping to account for stop loss and take profit limits
    new_df['sim2_gross_rev_pred14'] = new_df['Trade_Signal_2'] * 150 * (
        new_df['Growth_Future_1h'].clip(lower=(1 + stop_loss_threshold), upper=(1 + take_profit_threshold)) - 1
    )

    # Calculate trading fees (assumed 0.2% per trade)
    new_df['sim2_fees_pred14'] = new_df['Trade_Signal_2'].abs() * 0.002

    # Calculate net revenue by subtracting fees from gross revenue
    new_df['sim2_net_rev_pred14'] = new_df['sim2_gross_rev_pred14'] - new_df['sim2_fees_pred14']

    # Output the total net revenue from all trades
    print(new_df['sim2_net_rev_pred14'].sum())

    COLUMNS_FIN_RESULT = ['Date','Ticker','Adj Close']+TO_PREDICT+ [pred,  'sim2_gross_rev_pred14','sim2_fees_pred14','sim2_net_rev_pred14']
    new_df[(new_df.Split=='Test')&(new_df[pred]==1)][COLUMNS_FIN_RESULT]

    # DAILY INVESTMENTS COUNT
    filter_test_and_positive_pred = (new_df.Split=='Test')&(new_df[pred]==1)
    df_investments_count_daily = pd.DataFrame(new_df[filter_test_and_positive_pred].groupby('Date')[pred].count())
    sim2_avg_investments_per_day = df_investments_count_daily[pred].quantile(0.75)
    print(sim2_avg_investments_per_day)

    new_df[(new_df.Split=='Test')&(new_df[pred]==1)][['sim2_gross_rev_pred14','sim2_fees_pred14','sim2_net_rev_pred14']].sum()

    """### 4.1.3 Strategy 3"""

    # Volatility strategy using ATR and Bollinger Bands

    stop_loss_threshold = -0.02  # 2% Stop-Loss
    take_profit_threshold = 0.07  # 7% Take-Profit

    # Generate trade signal based on volatility and Bollinger Bands
    new_df['Trade_Signal_3'] = (
        (new_df['ATRr_14'] > 0.015) &  # High volatility indicated by ATR
        (new_df['BBP_5_2.0'] < 0.2)    # Price near the lower Bollinger Band (potentially oversold)
    )

    # Calculate gross revenue with stop-loss and take-profit limits applied
    new_df['sim3_gross_rev_pred14'] = new_df['Trade_Signal_3'] * 150 * (
        new_df['Growth_4h'].clip(lower=(1 + stop_loss_threshold), upper=(1 + take_profit_threshold)) - 1
    )

    # Calculate trading fees (assumed 0.2% per trade)
    new_df['sim3_fees_pred14'] = new_df['Trade_Signal_3'].abs() * 0.002

    # Calculate net revenue by subtracting fees from gross revenue
    new_df['sim3_net_rev_pred14'] = new_df['sim3_gross_rev_pred14'] - new_df['sim3_fees_pred14']

    # Output the total net revenue from all trades
    print(new_df['sim3_net_rev_pred14'].sum())

    COLUMNS_FIN_RESULT = ['Date','Ticker','Adj Close']+TO_PREDICT+ [pred, 'sim3_gross_rev_pred14','sim3_fees_pred14','sim3_net_rev_pred14']
    new_df[(new_df.Split=='Test')&(new_df[pred]==1)][COLUMNS_FIN_RESULT]

    # DAILY INVESTMENTS COUNT
    filter_test_and_positive_pred = (new_df.Split=='Test')&(new_df[pred]==1)
    df_investments_count_daily = pd.DataFrame(new_df[filter_test_and_positive_pred].groupby('Date')[pred].count())
    sim3_avg_investments_per_day = df_investments_count_daily[pred].quantile(0.75)  # 75% case - how many $100 investments per day do we have?
    print(sim3_avg_investments_per_day)

    new_df[(new_df.Split=='Test')&(new_df[pred]==1)][['sim3_gross_rev_pred14','sim3_fees_pred14','sim3_net_rev_pred14']].sum()

    """### 4.1.4 Strategy 4"""

    stop_loss_threshold = -0.02  # 3% Stop-Loss
    take_profit_threshold = 0.06  # 6% Take-Profit

    new_df['Trade_Signal_4'] = (
        (new_df['SMA_Cross'] == 1) &  # SMA-Kreuzung als Trendwechsel
        (new_df['ADX_14'] > 25)       # Starker Trend (ADX über 25)
    )

    # Berechnung des Bruttogewinns mit Stop-Loss und Take-Profit
    new_df['sim4_gross_rev_pred14'] = new_df['Trade_Signal_4'] * 150 * (
        new_df['Growth_24h'].clip(lower=(1 + stop_loss_threshold), upper=(1 + take_profit_threshold)) - 1
    )

    new_df['sim4_fees_pred14'] = new_df['Trade_Signal_4'].abs() * 0.002
    new_df['sim4_net_rev_pred14'] = new_df['sim4_gross_rev_pred14'] - new_df['sim4_fees_pred14']

    # print
    print(new_df['sim4_net_rev_pred14'].sum())

    COLUMNS_FIN_RESULT = ['Date','Ticker','Adj Close']+TO_PREDICT+ [pred,  'sim4_gross_rev_pred14','sim4_fees_pred14','sim4_net_rev_pred14']
    new_df[(new_df.Split=='Test')&(new_df[pred]==1)][COLUMNS_FIN_RESULT]

    # DAILY INVESTMENTS COUNT
    filter_test_and_positive_pred = (new_df.Split=='Test')&(new_df[pred]==1)
    df_investments_count_daily = pd.DataFrame(new_df[filter_test_and_positive_pred].groupby('Date')[pred].count())
    sim4_avg_investments_per_day = df_investments_count_daily[pred].quantile(0.75)  # 75% case - how many $100 investments per day do we have?
    print(sim4_avg_investments_per_day)

    new_df[(new_df.Split=='Test')&(new_df[pred]==1)][['sim4_gross_rev_pred14','sim4_fees_pred14','sim4_net_rev_pred14']].sum()

    """## 4.2 Calculate final results for all fields"""

    sim3_results = []  # initialize array

    stop_loss_threshold = -0.02  # 2% stop-loss
    take_profit_threshold = 0.07  # 7% take-profit

    # Strategie definieren
    new_df['Trade_Signal_3'] = (
        (new_df['ATRr_14'] > 0.015) &  # high volatilität
        (new_df['BBP_5_2.0'] < 0.2)   # touching lower bb
    )

    # calculating revenue for all PREDICTIONS
    for pred in PREDICTIONS:
        print(f'Calculating simulation for prediction {pred}:')

        gross_rev_column = f'sim3_gross_rev_{pred}'
        fees_column = f'sim3_fees_{pred}'
        net_rev_column = f'sim3_net_rev_{pred}'

        # for trade signal 'Trade_Signal3'
        if pred == 'Trade_Signal_3':
            new_df[gross_rev_column] = new_df['Trade_Signal3'] * 150 * (
                new_df['Growth_4h'].clip(lower=(1 + stop_loss_threshold), upper=(1 + take_profit_threshold)) - 1
            )
        else:
            new_df[gross_rev_column] = new_df[pred] * 150 * (new_df['Growth_4h'] - 1)

        # calculating fees
        new_df[fees_column] = -abs(new_df[pred]) * 0.002
        new_df[net_rev_column] = new_df[gross_rev_column] + new_df[fees_column]

        # calculate results
        filter_test_and_positive_pred = (new_df.Split == 'Test') & (new_df[pred] == 1)
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
        df_investments_count_daily = pd.DataFrame(new_df[filter_test_and_positive_pred].groupby('Date')[pred].count())
        sim3_avg_investments_per_day = df_investments_count_daily[pred].mean()
        sim3_q75_investments_per_day = df_investments_count_daily[pred].quantile(0.75)
        sim3_capital = 150 * 5 * sim3_q75_investments_per_day
        sim3_CAGR = ((sim3_capital + sim3_net_rev) / sim3_capital) ** (1 / 4) # -1 laut chatgpt

        # safe to array
        sim3_results.append((pred, sim3_count_investments, sim3_gross_rev, sim3_fees, sim3_net_rev, sim3_fees_percentage,
                            sim3_average_net_revenue, sim3_avg_investments_per_day, sim3_capital, sim3_CAGR))

        # print results
        if sim3_count_investments > 1:
            print(f" Financial Result: \n {new_df[filter_test_and_positive_pred][[gross_rev_column, fees_column, net_rev_column]].sum()}")
            print(f" Count Investments in 4 years (on TEST): {sim3_count_investments}")
            print(f" Gross Revenue: ${int(sim3_gross_rev)}")
            print(f" Fees (0.2% for buy+sell): ${int(-sim3_fees)}")
            print(f" Net Revenue: ${int(sim3_net_rev)}")
            print(f" Fees are {int(-10.0 * sim3_fees / sim3_gross_rev)} % from Gross Revenue")
            print(f" Capital Required : ${int(sim3_capital)} (Vbegin)")
            print(f" Final value (Vbegin + Net_revenue) : ${int(sim3_capital + sim3_net_rev)} (Vfinal)")
            print(f" Average CAGR on TEST (4 years) : {np.round(sim3_CAGR, 3)}, or {np.round(10.0 * (sim3_CAGR - 1), 1)}% ")
            print(f" Average daily stats: ")
            print(f" Average net revenue per investment: ${np.round(sim3_net_rev / sim3_count_investments, 2)} ")
            print(f" Average investments per day: {int(np.round(sim3_avg_investments_per_day))} ")
            print(f" Q75 investments per day: {int(np.round(sim3_q75_investments_per_day))} ")
            print('=============================================+')

    # save to df
    columns_simulation = ['prediction', 'sim3_count_investments', 'sim3_gross_rev', 'sim3_fees', 'sim3_net_rev', 'sim3_fees_percentage',
                        'sim3_average_net_revenue', 'sim3_avg_investments_per_day', 'sim3_capital', 'sim3_CAGR']

    df_sim3_results = pd.DataFrame(sim3_results, columns=columns_simulation)

    #Calculating simulation for prediction pred06_cci_oversold:
    # Financial Result:
    # sim3_gross_rev_pred06_cci_oversold    33547.335938
    #sim3_fees_pred06_cci_oversold         -3502.800000
    #sim3_net_rev_pred06_cci_oversold      30044.538867
    #dtype: float64
    # Count Investments in 4 years (on TEST): 11676
    # Gross Revenue: $33547
    # Fees (0.2% for buy+sell): $3502
    # Net Revenue: $30044
    # Fees are 1 % from Gross Revenue
    # Capital Required : $3000 (Vbegin)
    # Final value (Vbegin + Net_revenue) : $33044 (Vfinal)
    # Average CAGR on TEST (4 years) : 1.822, or 8.2%
    # Average daily stats:
    # Average net revenue per investment: $2.57
    # Average investments per day: 3
    # Q75 investments per day: 4

    # Loading Historical Data DAX
    ticker = "^GDAXI"  # DAX Index
    start_date = "2019-08-04"
    end_date = "2023-08-04"
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    # Calculate CAGR
    start_value = data['Adj Close'].iloc[0]
    end_value = data['Adj Close'].iloc[-1]
    n_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25

    cagr_dax = (end_value / start_value) ** (1 / n_years) - 1

    print(f"CAGR for DAX for the last 4 years {start_date} bis {end_date} beträgt: {cagr_dax:.2%}")

    # comparing profability

    cagr_strategy = 0.082  # 8.2% as decimal

    print(f"Defined strategy has a CAGR of {cagr_strategy:.2%}")
    print(f"CAGR of DAX is {cagr_dax:.2%}")

    if cagr_strategy > cagr_dax:
        print("The defined strategy performs better than DAX!")
    else:
        print("DAX has a higher performance.")

    df_sim3_results['sim3_growth_capital_4y'] = (df_sim3_results.sim3_net_rev+df_sim3_results.sim3_capital) / df_sim3_results.sim3_capital

    # final comparison
    df_sim3_results

    # Create the scatter plot
    fig = px.scatter(
        df_sim3_results.dropna(),
        x='sim3_avg_investments_per_day',
        y='sim3_CAGR',
        size='sim3_growth_capital_4y',  # Use the 'size' parameter for sim1_CAGR
        text='prediction',
        title='Compound Annual Growth vs. Time spent (Average investments per day)',
        labels={'sim3_capital': 'Initial Capital Requirement', 'growth_capital_4y': '4-Year Capital Growth'}
    )

    # Update the layout to improve readability of the annotations
    fig.update_traces(textposition='top center')

    # Show the plot
    fig.show()


if __name__ == "__main__":
    trading_simulation()