# Introduction
This project aims to develop an algorithmic trading strategy for swing trades with low capital in the crypto market and just a few trades per day. By leveraging historical data and machine learning, I aim to predict future price movements and automate trading decisions to maximize profitability and minimize risk with a set take profit and stop loss. Note that this model is in its early stages.  

## Problem Description    
Many individuals face the challenge of preparing for retirement with limited capital, requiring strategies that navigate high-risk, high-return markets like cryptocurrency. This project uses the crypto volatility for swing trading. By analyzing historical market data, stock prices, and technical indicators, I aim to develop an automated strategy for informed trading decisions. This strategy is easily reproducible by those with basic knowledge of the crypto market.

# Methodology and Implementation   

## Data Collection   
Step: Collect historical price data and volume from platforms like Binance, Yahoo Finance, Fred. Consolidate and clean the data and handling missing values and outliers. Create features like moving averages, RSI, Bollinger Bands, and Fibonacci levels and dummies.
Reproduction: Use the provided code, you can change the coins for your strategy.
Findings: Issues with Binance providing all the desired coins, API Limits. Talib library issues, needed to be replaced by pandas_ta. Some manual created predictions perform ver well.


## Model Training    
Step: Train models using Decision Trees, Logistic Regression, and Random Forests. Used Hyperparameter Tuning to optimize the model parameters to enhance performance.
Reproduction: Use Scikit-learn for model training and evaluation, along with the provided files.
Findings: Initial models yielded modest results, with Logistic Regression performing the best. However, tuning is necessary. The next step is to test the model with fewer indicators to improve its performance. Some indicators appear to be irrelevant based on feature importance calculations. The next step will be to eliminate these less impactful features to streamline the model and potentially improve its accuracy. Additionally, it would be valuable to test the model with different coins, as the current version includes coins from various sectors (e.g., AI, Gaming, etc.). 

## Trading Simulation    
Step: Simulate different trading strategies using historical data and volatility.
Reproduction: Use the provided files for simulation.
Findings: The strategy that leverages volatility proved to be the most profitable in the simulation.

I implemented a volatility strategy using the Average True Range (ATR) and Bollinger Bands:

High Volatility: ATR (14) > 0.015
Lower Bollinger Band Touch: BBP < 0.2
This strategy was tested over 4 years, producing the following results:
Net Revenue: $30,044
Gross Revenue: $33,547
Fees (0.2%): $3,502
Investments Count: 11,676
Capital Required: $3,000
Final Value (Vbegin + Net Revenue): $33,044
Average CAGR (4 years): 8.2%
Average Net Revenue per Investment: $2.57
Average Investments per Day: 3
75th Percentile Investments per Day: 4

## Automation   
Step: Automate the entire workflow from data collection to strategy simulation.
Reproduction: Convert Jupyter notebooks to scripts and set up cron jobs or use provied py files.
Findings: Achieved a fully automated system capable of periodic predictions and simulations.

# Important Findings and Difficulties
Findings: There are ongoing difficulties accessing Binance's APIs, suggesting the need to explore alternative exchanges to ensure data reliability and continuity. Capitalizing on market volatility can provide a strategic advantage. It is possible to achieve profitability without a significant initial investment, allowing for gradual capital growth over time. For traders with limited capital, engaging in swing trading—averaging around three trades per day—can be a practical approach to incrementally build wealth.

Difficulties: As this was my first experience developing a RF, LR and DT model, I invested significant time in research and self-education to build the necessary foundation. Due to limited time, I was unable to integrate certain factors, such as sentiment analysis, which I plan to incorporate in future iterations.
Defining critical events, particularly "black swan" events, and determining appropriate time frames for their impact proved to be challenging. Despite the challenges, the process was highly educational and enjoyable. I'm excited to continue refining the model and expanding its capabilities.   
#### WAGMI    


## Future Work
Adding sentiment analysis   
Testing with different coins, maybe more volatile   
Refining prediction parameters for higher accuracy   
Backtesting and Papertrading    
Incorporating real-time data for live trading   

## Requirements
- Python 3.x
- Google Colab or Local environment
- install the required packages listed in requirements.txt

## Access Data    
Required data can be downloaded from:     
File 1: [Crypto Data](https://drive.google.com/file/d/1-09LDYhQIjgorvsxgqVkhd1T3QvP3akU/view?usp=sharing)
File 2: [Stocks Data](https://drive.google.com/file/d/1PjkT11UkqvOw7Sl7AhrDjGHI3W7-f-Bh/view?usp=sharing)
File 3: [Both combined with calculations](https://drive.google.com/file/d/1fSoSmx0lkBawGWfySwqaAOTTK43-9nBM/view?usp=sharing)
File 4: [DataFrame after modeling](https://drive.google.com/file/d/1MpfzmSU5ixSQE_ZS8ZNjpMDFjVP76Fmy/view?usp=sharing)

### Automatically Load Data in Colab     
from google.colab import drive
import pandas as pd 

### Mount Google Drive
drive.mount('/content/drive')

### Load Crypto Data
crypto_path = '/content/drive/MyDrive/My Drive/crypto.csv'
data1 = pd.read_csv(crypto_path)

### Load Stocks Data
stocks_path = '/content/drive/MyDrive/My Drive/stocks.csv'
data2 = pd.read_csv(stocks_path)

### Load Combined Data with Calculations
combined_path = '/content/drive/MyDrive/My Drive/everything_data.csv'
data3 = pd.read_csv(combined_path)


## Setup
Install Python and pip
Download and install Python from python.org. pip usually comes with Python.
Create and activate a virtual environment (optional but recommended)
python -m venv venv

Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

Install dependencies
pip install -r requirements.txt

Run the project
python main.py
