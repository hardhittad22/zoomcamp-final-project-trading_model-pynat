# imports
import numpy as np
import pandas as pd
import requests
import pickle

# finance

# visualisation
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# time
import time
from datetime import date, datetime, timedelta

# ML models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree, export_text, DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler



def modeling():

  #import DataFrame
  df_with_all_dummies = pd.read_csv("df_with_all_dummies.csv")

  # Ensure 'Date' column is in datetime format
  df_with_all_dummies["Date"] = pd.to_datetime(df_with_all_dummies["Date"], utc=True)

  with open("column_lists.pkl", "rb") as f:
          data = pickle.load(f)
          NUMERICAL = data["NUMERICAL"]
          DUMMIES = data["DUMMIES"]
          TO_PREDICT = data["TO_PREDICT"]
      

  """## 1.3 Temporal Split"""

  def temporal_split(
      df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15
  ):
      min_date = pd.to_datetime(min_date) if isinstance(min_date, str) else min_date
      max_date = pd.to_datetime(max_date) if isinstance(max_date, str) else max_date
      # Convert min_date and max_date to pandas Timestamp if they are strings
      min_date = pd.Timestamp(min_date)
      max_date = pd.Timestamp(max_date)

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

  min_date_df = df_with_all_dummies.Date.min()
  max_date_df = df_with_all_dummies.Date.max()

  df_with_all_dummies = temporal_split(df_with_all_dummies,
                                  min_date = min_date_df,
                                  max_date = max_date_df)

  df_with_all_dummies['Split'].value_counts()/len(df_with_all_dummies)

  new_df = df_with_all_dummies.copy()

  new_df.groupby(['Split'])['Date'].agg({'min','max','count'})

  new_df.info()

  pd.set_option('display.max_columns', None)
  print(new_df.tail(1))

  new_df[TO_PREDICT].head(3)

  corr_is_positive_growth_1h_future = new_df[TO_PREDICT].corr()['Is_Positive_Growth_1h_Future']
  corr_is_positive_growth_1h_future_df = pd.DataFrame(corr_is_positive_growth_1h_future)
  corr_is_positive_growth_1h_future_df.sort_values(by='Is_Positive_Growth_1h_Future').head(5)

  corr_is_positive_growth_24h_future = new_df[TO_PREDICT].corr()['Is_Positive_Growth_24h_Future']
  corr_is_positive_growth_24h_future_df = pd.DataFrame(corr_is_positive_growth_24h_future)
  corr_is_positive_growth_24h_future_df.sort_values(by='Is_Positive_Growth_24h_Future').head(5)

  """## 1.4. Defining and Cleaning dataframe for Modeling (ML)"""

  features_list = NUMERICAL+DUMMIES

  print(features_list)

  to_predict = 'Is_Positive_Growth_1h_Future'

  train_df = new_df[new_df.Split.isin(['Train'])].copy(deep=True)
  valid_df = new_df[new_df.Split.isin(['Validation'])].copy(deep=True)
  train_valid_df = new_df[new_df.Split.isin(['Train','Validation'])].copy(deep=True)

  test_df =  new_df[new_df.Split.isin(['Test'])].copy(deep=True)

  # ONLY numerical Separate features and target variable for training and testing sets
  X_train = train_df[features_list+[to_predict]]
  X_valid = valid_df[features_list+[to_predict]]

  X_train_valid = train_valid_df[features_list+[to_predict]]

  X_test = test_df[features_list+[to_predict]]

  # Predictions and Join to the original dataframe new_df
  X_all =  new_df[features_list+[to_predict]].copy(deep=True)

  print(f'length: X_train {X_train.shape},  X_validation {X_valid.shape}, X_test {X_test.shape}, X_train_valid = {X_train_valid.shape},  all combined: X_all {X_all.shape}')

  def clean_dataframe_from_inf_and_nan(df):
      # Convert categorical columns to object type temporarily
      cat_cols = df.select_dtypes(include=['category']).columns
      df[cat_cols] = df[cat_cols].astype('object')

      # Replace +-inf with NaN 
      df = df.replace([np.inf, -np.inf], np.nan)
      # Fill NaN values with 0
      df = df.fillna(0)

      # Convert object columns back to categorical
      df[cat_cols] = df[cat_cols].astype('category')

      return df

  X_train = clean_dataframe_from_inf_and_nan(X_train)
  X_valid = clean_dataframe_from_inf_and_nan(X_valid)
  X_train_valid = clean_dataframe_from_inf_and_nan(X_train_valid)
  X_test = clean_dataframe_from_inf_and_nan(X_test)
  X_all = clean_dataframe_from_inf_and_nan(X_all)

  y_train = X_train[to_predict]

  y_valid = X_valid[to_predict]

  y_train_valid = X_train_valid[to_predict]
  y_test = X_test[to_predict]
  y_all =  X_all[to_predict]

  # remove y_train, y_test from X_ dataframes
  del X_train[to_predict]
  del X_valid[to_predict]
  del X_train_valid[to_predict]

  del X_test[to_predict]

  del X_all[to_predict]

  """# 2. Modeling

  ## 2.1 Manual 'rule of thumb' predictions
  """

  # generate manual predictions
  new_df['pred01_momentum_positive'] = (new_df['MOM_10'] > 0).astype(np.int8)
  # Momentum Indicator (MOM)
  # Positive Momentum: Helps identify the speed of price changes. Positive momentum indicates potential buying opportunities.

  new_df['pred02_momentum_negative'] = (new_df['MOM_10'] < 0).astype(np.int8)
  # Negative Momentum: Helps identify the speed of price changes. Negative momentum signals potential selling opportunities.

  new_df['pred03_roc_positive'] = (new_df['ROC_10'] > 0).astype(np.int8)
  # Rate of Change (ROC)
  # Positive ROC: Indicates the percentage change in price over a specified period. Positive ROC suggests an uptrend.

  new_df['pred04_roc_negative'] = (new_df['ROC_10'] < 0).astype(np.int8)
  # Negative ROC: Indicates the percentage change in price over a specified period. Negative ROC suggests a downtrend.

  new_df['pred05_cci_overbought'] = (new_df['CCI_14_0.015'] > 100).astype(np.int8)
  # Commodity Channel Index (CCI)
  # Overbought CCI: CCI above 100 can indicate an overbought condition, which could be a signal to sell.

  new_df['pred06_cci_oversold'] = (new_df['CCI_14_0.015'] < -100).astype(np.int8)
  # Oversold CCI: CCI below -100 can indicate an oversold condition, which could be a signal to buy.

  new_df['pred07_fibonacci_50_support'] = (new_df['Adj Close'] > new_df['Fibonacci_50']).astype(np.int8)
  # Price above Fibonacci 50% level: Indicates potential support if the price is above this level.

  new_df['pred08_rsi_above_ma'] = (new_df['RSI_14'] > new_df['RSI_14'].rolling(window=14).mean()).astype(np.int8)
  # RSI with Moving Average
  # RSI above its own 14-period moving average: Indicates bullish conditions as RSI is trending higher.

  new_df['pred09_macd_hist_positive'] = (new_df['MACDh_12_26_9'] > 0).astype(np.int8)
  # MACD Histogram Positive: Indicates that the MACD line is above the signal line, suggesting upward momentum.

  # Example Output manual predictions:
  new_df[['pred05_cci_overbought', 'pred07_fibonacci_50_support', 'pred08_rsi_above_ma']]

  # Function to find all predictions (starting from 'pred'), generate is_correct (correctness of each prediction)
  # and precision on TEST dataset (assuming there is df["split"] column with values 'train','validation','test'

  # returns 2 lists of features: PREDICTIONS and IS_CORRECT

  def get_predictions_correctness(df:pd.DataFrame, to_predict:str):
    PREDICTIONS = [k for k in df.keys() if k.startswith('pred')]
    print(f'Prediction columns founded: {PREDICTIONS}')

    # add columns is_correct_
    for pred in PREDICTIONS:
      part1 = pred.split('_')[0] # first prefix before '_'
      df[f'is_correct_{part1}'] =  (new_df[pred] == new_df[to_predict]).astype(int)

    # IS_CORRECT features set
    IS_CORRECT =  [k for k in df.keys() if k.startswith('is_correct_')]
    print(f'Created columns is_correct: {IS_CORRECT}')

    print('Precision on TEST set for each prediction:')
    # Define "Precision" for ALL predictions on a Test Data
    for i,column in enumerate(IS_CORRECT):
      prediction_column = PREDICTIONS[i]
      is_correct_column = column
      filter = (new_df.Split=='Test') & (new_df[prediction_column]==1)
      print(f'Prediction column:{prediction_column} , is_correct_column: {is_correct_column}')
      print(new_df[filter][is_correct_column].value_counts())
      print(new_df[filter][is_correct_column].value_counts()/len(new_df[filter]))
      print('---------')

    return PREDICTIONS, IS_CORRECT

  PREDICTIONS, IS_CORRECT = get_predictions_correctness(df = new_df, to_predict= to_predict)

  new_df[PREDICTIONS+IS_CORRECT+[to_predict]]



  """## 2.2 Decision Tree Classifier

  ### 2.2.1 Defining Functions to clean_df(), fit_decision_tree(), predict_decision_tree()
  """

  # max_depth is hyperParameter
  def fit_decision_tree(X, y, max_depth=20):
  # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth=max_depth,
                                random_state=42)

    # Fit the classifier to the training data
    clf.fit(X, y)
    return clf, X.columns

  # Iterating
  for column, dtype in X_train_valid.dtypes.items():
      # printing columns and types
      print(f"{column}: {dtype}")


  clf_10, train_columns = fit_decision_tree(X=X_train_valid,
                                            y=y_train_valid,
                                            max_depth=10)

  # predict on a full dataset
  y_pred_all = clf_10.predict(X_all)

  # defining a new prediction vector is easy now, as the dimensions will match
  new_df['pred10_clf_10'] = y_pred_all

  # new prediction is added --> need to recalculate the correctness
  PREDICTIONS, IS_CORRECT = get_predictions_correctness(df = new_df, to_predict = to_predict)



  """### 2.2.2 Hyperparams Tuning Decision Tree Classifier"""
  #precision_by_depth = {}
  #best_precision = 0
  #best_depth = 0

  #for depth in range(1, 21):
  #    print(f'Working with a tree of a max depth= {depth}')
  #    clf, train_columns = fit_decision_tree(X=X_train_valid,
  #                                           y=y_train_valid,
  #                                           max_depth=depth)
  #    y_pred_valid = clf.predict(X_valid)
  #    precision_valid = precision_score(y_valid, y_pred_valid)
  #    y_pred_test = clf.predict(X_test)
  #    precision_test = precision_score(y_test, y_pred_test)
  #    print(f'  Precision on test is {precision_test}, (precision on valid is {precision_valid} - tend to overfit)')
  #    precision_by_depth[depth] = round(precision_test, 4)
  #    if precision_test >= best_precision:
  #        best_precision = round(precision_test, 4)
  #        best_depth = depth
  #    tree_rules = export_text(clf, feature_names=list(X_train), max_depth=3)
  #    print(tree_rules)
  #    print('------------------------------')

  #print(f'All precisions by depth: {precision_by_depth}')
  #print(f'The best precision is {best_precision} and the best depth is {best_depth}')

  """FOR JUST CRYPTO Positive Growth 1h Future:    
  All precisions by depth: {1: 0.5192, 2: 0.5192, 3: 0.5227, 4: 0.5297, 5: 0.5316, 6: 0.5258, 7: 0.5286, 8: 0.5228, 9: 0.5205, 10: 0.5237, 11: 0.5233, 12: 0.5249, 13: 0.5241, 14: 0.5238, 15: 0.5213, 16: 0.5198, 17: 0.5185, 18: 0.5186, 19: 0.5165, 20: 0.5154}
  The best precision is 0.5316 and the best depth is 5

  FOR STOCKS AND CRYPTO Positive Growth 1h Future:      
  All precisions by depth: {1: 0.5192, 2: 0.5192, 3: 0.5227, 4: 0.5206, 5: 0.5337, 6: 0.5241, 7: 0.5283, 8: 0.5246, 9: 0.5237, 10: 0.5206, 11: 0.5222, 12: 0.5204, 13: 0.5223, 14: 0.5212, 15: 0.5199, 16: 0.5163, 17: 0.5149, 18: 0.5141, 19: 0.5121, 20: 0.5137}
  The best precision is 0.5337 and the best depth is 5

  FOR STOCKS AND CRYPTO Positive Growth24h Future:     
  All precisions by depth: {1: 0.5276, 2: 0.5649, 3: 0.5532, 4: 0.5534, 5: 0.5572, 6: 0.5367, 7: 0.5278, 8: 0.5396, 9: 0.5298, 10: 0.5261, 11: 0.5178, 12: 0.519, 13: 0.5105, 14: 0.5181, 15: 0.5192, 16: 0.5283, 17: 0.5112, 18: 0.5212, 19: 0.5194, 20: 0.5219}
  The best precision is 0.5649 and the best depth is 2

  """

  # Working with 1h growth prediction as lr is performing the best out of them and we use the 1h strategy
  # All precisions by depth: {1: 0.5192, 2: 0.5192, 3: 0.5227, 4: 0.5206, 5: 0.5337, 6: 0.5241, 7: 0.5283, 8: 0.5246, 9: 0.5237, 10: 0.5206, 11: 0.5222, 12: 0.5204, 13: 0.5223, 14: 0.5212, 15: 0.5199, 16: 0.5163, 17: 0.5149, 18: 0.5141, 19: 0.5121, 20: 0.5137}
  # The best precision is 0.5337 and the best depth is 5
  precision_by_depth = {1: 0.5192, 2: 0.5192, 3: 0.5227, 4: 0.5206, 5: 0.5337, 6: 0.5241, 7: 0.5283, 8: 0.5246, 9: 0.5237, 10: 0.5206, 11: 0.5222, 12: 0.5204, 13: 0.5223, 14: 0.5212, 15: 0.5199, 16: 0.5163, 17: 0.5149, 18: 0.5141, 19: 0.5121, 20: 0.5137}
  best_depth = 5
  best_precision = 0.5337

  # Convert the dictionary to a DataFrame
  df = pd.DataFrame(list(precision_by_depth.items()), columns=['max_depth', 'precision_score'])
  df.loc[:,'precision_score'] = df.precision_score*100.0 # need for % visualisation

  # Create the bar chart using Plotly Express
  fig = px.bar(df,
              x='max_depth',
              y='precision_score',
              title='Precision Score vs. Max Depth for a Decision Tree',
              labels={'max_depth': 'Max Depth', 'precision_score': 'Precision Score'},
              range_y=[50, 58],
              text='precision_score')

  # Update the text format to display as percentages
  fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
  fig.update_layout(title={'text': 'Precision Score vs. Max Depth for a Decision Tree', 'x': 0.5, 'xanchor': 'center'})
  fig.show()

  clf_best,train_columns = fit_decision_tree(X=X_train_valid,
                                            y=y_train_valid,
                                            max_depth=best_depth)

  # Get the number of nodes and leaves in the tree
  n_nodes = clf_best.tree_.node_count
  n_leaves = clf_best.get_n_leaves()

  print(f"Number of nodes: {n_nodes}")
  print(f"Number of leaves: {n_leaves}")

  clf_best

  # predict on a full dataset
  y_pred_clf_best = clf_best.predict(X_all)

  # defining a new prediction vector is easy now, as the dimensions will match
  new_df['pred11_clf_best'] = y_pred_clf_best

  # new prediction is added --> need to recalculate the correctness
  PREDICTIONS, IS_CORRECT = get_predictions_correctness(df = new_df, to_predict = to_predict)



  """## 2.3 Random Forest

  ### 2.3.1 Hyperparams Tuning
  """

  #precision_matrix = {}
  #best_precision = 0
  #best_depth = 0
  #best_estimators = 1

  #for depth in [15, 16, 17, 18, 19, 20]:
  #    for estimators in [50, 100, 200, 500]:
  #        print(f'Working with HyperParams: depth = {depth}, estimators = {estimators}')

  #        start_time = time.time()

  #        rf = RandomForestClassifier(n_estimators=estimators,
  #                                    max_depth=depth,
  #                                    random_state=42,
  #                                    n_jobs=-1)
  #
  #        rf = rf.fit(X_train_valid, y_train_valid)

  #        y_pred_valid = rf.predict(X_valid)
  #        precision_valid = precision_score(y_valid, y_pred_valid)
  #        y_pred_test = rf.predict(X_test)
  #        precision_test = precision_score(y_test, y_pred_test)
  #        print(f'  Precision on test is {precision_test}, (precision on valid is {precision_valid} - tend to overfit)')

  #        precision_matrix[depth, estimators] = round(precision_test, 4)

  #        elapsed_time = time.time() - start_time
  #        print(f'Time for training: {elapsed_time:.2f} seconds, or {elapsed_time/60:.2f} minutes')

  #        if precision_test >= best_precision:
  #            best_precision = round(precision_test, 4)
  #            best_depth = depth
  #            best_estimators = estimators
  #            print(f'New best precision found for depth={depth}, estimators = {estimators}')

  #        print('------------------------------')

  #print(f'Matrix of precisions: {precision_matrix}')
  #print(f'The best precision is {best_precision} and the best depth is {best_depth} ')

  # Result For 1h Future Growth
  # Matrix of precisions: {(15, 50): 0.5453, (15, 100): 0.5487, (15, 200): 0.5541, (15, 500): 0.5558, (16, 50): 0.5405, (16, 100): 0.5461, (16, 200): 0.5499, (16, 500): 0.5558, (17, 50): 0.5448, (17, 100): 0.5473, (17, 200): 0.554, (17, 500): 0.5561, (18, 50): 0.5469, (18, 100): 0.5515, (18, 200): 0.5524, (18, 500): 0.5549, (19, 50): 0.5424, (19, 100): 0.5468, (19, 200): 0.5506, (19, 500): 0.555, (20, 50): 0.5467, (20, 100): 0.5453, (20, 200): 0.553, (20, 500): 0.555}
  # The best precision is 0.5561 and the best depth is 17

  # Result For 24h Future Growth
  # Matrix of precisions: {(15, 50): 0.5327, (15, 100): 0.5277, (15, 200): 0.529, (15, 500): 0.5301, (16, 50): 0.5295, (16, 100): 0.53, (16, 200): 0.5284, (16, 500): 0.5296, (17, 50): 0.5257, (17, 100): 0.5303, (17, 200): 0.5294, (17, 500): 0.5293, (18, 50): 0.5272, (18, 100): 0.5313, (18, 200): 0.5299, (18, 500): 0.527, (19, 50): 0.5339, (19, 100): 0.5337, (19, 200): 0.5328, (19, 500): 0.5294, (20, 50): 0.5305, (20, 100): 0.5296, (20, 200): 0.529, (20, 500): 0.5287}
  # The best precision is 0.5339 and the best depth is 19

  """### 2.3.2 Working with RF + Feature Importance"""

  # working with better RF model
  best_precision_matrix_random_forest = {(15, 50): 0.5453, (15, 100): 0.5487, (15, 200): 0.5541, (15, 500): 0.5558, (16, 50): 0.5405, (16, 100): 0.5461, (16, 200): 0.5499, (16, 500): 0.5558, (17, 50): 0.5448, (17, 100): 0.5473, (17, 200): 0.554, (17, 500): 0.5561, (18, 50): 0.5469, (18, 100): 0.5515, (18, 200): 0.5524, (18, 500): 0.5549, (19, 50): 0.5424, (19, 100): 0.5468, (19, 200): 0.5506, (19, 500): 0.555, (20, 50): 0.5467, (20, 100): 0.5453, (20, 200): 0.553, (20, 500): 0.555}
  best_precision = 0.5561
  best_depth = 17
  best_estimators = 500

  # training Random Forest with best hyperparam
  best_max_depth = 17
  best_n_estimators = 500
  rf_model = RandomForestClassifier(max_depth=best_max_depth, n_estimators=best_n_estimators, random_state=42)
  rf_model.fit(X_train, y_train)


  # feature importances extracting
  feature_importances = rf_model.feature_importances_
  # convert DataFrame
  features_df = pd.DataFrame({
      'Feature': X_train.columns,
      'Importance': feature_importances
  })
  # sort importance
  features_df = features_df.sort_values(by='Importance', ascending=False)

  # visualisation
  plt.figure(figsize=(20, 25))

  plt.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
  plt.xlabel('Feature Importance', fontsize=14)
  plt.ylabel('Features', fontsize=14)
  plt.title('Feature Importances in Random Forest Model', fontsize=16)
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  plt.gca().invert_yaxis()
  plt.tight_layout()

  plt.show()

  # all features no treshhold
  y_pred_all = rf_model.predict(X_all)

  # all features no treshhold
  new_df['pred12_rf'] = y_pred_all

  # new prediction is added --> need to recalculate the correctness
  PREDICTIONS, IS_CORRECT = get_predictions_correctness(df = new_df, to_predict = to_predict)

  # feature reduction
  # treshhold for feature importance
  # please be patient this takes some time before its printed
  importance_threshold = 0.005

  # identify important features
  important_features = features_df[features_df['Importance'] > importance_threshold]['Feature']

  # reducing data
  X_train_reduced = X_train[important_features]
  X_test_reduced = X_test[important_features]

  # traing model with reduced features
  rf_model_reduced = RandomForestClassifier(max_depth=best_max_depth, n_estimators=best_n_estimators, random_state=42)
  rf_model_reduced.fit(X_train_reduced, y_train)

  # new predictions
  y_pred = rf_model_reduced.predict(X_test_reduced)
  accuracy = accuracy_score(y_test, y_pred)
  print(f'New model accuracy: {accuracy}')

  # Reducing X_all to important features
  X_all_reduced = X_all[important_features]

  #reduced treshhold
  y_pred_all = rf_model_reduced.predict(X_all_reduced)
  new_df['pred13_rf_reduced'] = y_pred_all

  # new prediction is added --> need to recalculate the correctness
  PREDICTIONS, IS_CORRECT = get_predictions_correctness(df = new_df, to_predict = to_predict)

  # Convert data to DataFrame
  df = pd.DataFrame.from_dict(best_precision_matrix_random_forest, orient='index', columns=['precision_score']).reset_index()

  # Rename the columns for clarity
  df.columns = ['max_depth_and_metric', 'precision_score']

  # Separate the tuple into two columns
  df[['max_depth', 'n_estimators']] = pd.DataFrame(df['max_depth_and_metric'].tolist(), index=df.index)

  # Drop the combined column
  df = df.drop(columns=['max_depth_and_metric'])

  # Create line plot using Plotly Express
  fig = px.line(df, x='max_depth', y='precision_score', color='n_estimators',
                labels={'max_depth': 'Max Depth', 'precision_score': 'Precision Score', 'n_estimators': 'Number of Estimators'},
                title='Random Forest Models: Precision Score vs. Max Depth for Different Number of Estimators')

  # Adjust x-axis range
  fig.update_xaxes(range=[15, 20])

  # Show the figure
  fig.show()

  """## 2.4 Logistic Regression"""

  #Hyperparametertuning for Logistic Regression
  # precision_matrix = {}
  #best_precision = 0
  #best_C = 0
  #best_iter = 0

  # for c in [1, 0.1, 0.01]:
  #    for iter in [50, 100, 200]:
  #        print(f'Working with HyperParams: C = {c} (positive float, smaller = stronger regularization), max_iter={iter}')
          # Fitting the Tree on X_train, y_train
          # HyperParam C should be between 0 and 1
  #        lr = LogisticRegression(C=c,
  #                                random_state=42,
  #                                max_iter=iter,
  #                                solver='sag',
  #                                n_jobs=-1)

  #        lr = lr.fit(X_train_valid, y_train_valid)

          # Getting Predictions for TEST and Accuracy Acore
  #        y_pred_valid = lr.predict(X_valid)
  #        precision_valid = precision_score(y_valid, y_pred_valid)
  #        y_pred_test = lr.predict(X_test)
  #        precision_test = precision_score(y_test, y_pred_test)
  #        print(f'  Precision on test is {precision_test}, (precision on valid is {precision_valid} - tend to overfit)')
          # Saving to Dict
  #        precision_matrix[(c, iter)] = round(precision_test, 4)
          # Updating the best Precision
  #        if precision_test >= best_precision:
  #            best_precision = round(precision_test, 4)
  #            best_C = c
  #            best_iter = iter
  #            print(f'New best precision found for C={c}')
  #            print('------------------------------')

  #print(f'Matrix of precisions: {precision_matrix}')
  #print(f'The best precision is {best_precision} and the best C is {best_C}, best max_iter = {best_iter}')

  # Matrix of precisions: {(1, 50): 0.5608, (1, 100): 0.5698, (1, 200): 0.5809, (0.1, 50): 0.5608, (0.1, 100): 0.5698, (0.1, 200): 0.5809, (0.01, 50): 0.5608, (0.01, 100): 0.5698, (0.01, 200): 0.5809}
  # The best precision is 0.5809 and the best C is 1, best max_iter = 200

  best_precision_matrix_logistic_regression = {(1, 50): 0.5608, (1, 100): 0.5698, (1, 200): 0.5809, (0.1, 50): 0.5608, (0.1, 100): 0.5698, (0.1, 200): 0.5809, (0.01, 50): 0.5608, (0.01, 100): 0.5698, (0.01, 200): 0.5809}

  # Beste Hyperparameter aus dem Hypertuning
  best_C = 1
  best_max_iter = 200

  # Modell mit den besten Hyperparametern erstellen
  best_lr = make_pipeline(RobustScaler(), LogisticRegression(C=best_C, max_iter=best_max_iter, solver='liblinear'))

  # Modell trainieren
  best_lr.fit(X_train, y_train)

  y_pred_all = best_lr.predict(X_all)
  new_df['pred14_lr'] = y_pred_all

  # new prediction is added --> need to recalculate the correctness
  PREDICTIONS, IS_CORRECT = get_predictions_correctness(df = new_df, to_predict = to_predict)

  # Visualise Precision Scores

  # Prepare Data
  rows = []
  for key, value in list(best_precision_matrix_logistic_regression.items()):
      C, max_iter = key
      combination_label = f'C={C}, max_iter={max_iter}'
      rows.append({'Combination': combination_label, 'Precision': value})

  df = pd.DataFrame(rows)
  df.loc[:,'Precision'] = df.Precision*100.0 # need for % visualisation

  # Create Bar Chart
  fig = px.bar(df,
              x='Combination',
              y='Precision',
              text='Precision'
              )

  # Customize Layout for better Readability
  fig.update_layout(
      xaxis_title='Hyperparams combinations of <C, Max Iterations>',
      yaxis_title='Precision Score',
      xaxis_tickangle=-45,
      title={
          'text': 'Precision Scores for Various Logistic Regression Hyperparameter Combinations',
          'y':0.95,
          'x':0.5,
          'xanchor': 'center',
          'yanchor': 'top'
      }
  )


  # Update Text Position
  fig.update_traces(texttemplate='%{text:.2f}%',
                    textposition='inside',
                    textfont_color='white')

  # Show Figure
  fig.show()

  """# 3. Different Decision Rules to improve Precision

  ## 3.1 Predict Probalility

  ### 3.1.1 Logistic Regression
  """

  # Predict Probability
  y_pred_test = best_lr.predict_proba(X_test)
  y_pred_test_class1 = [k[1] for k in y_pred_test]

  # Plot
  plt.hist(y_pred_test_class1, bins=30, edgecolor='k', alpha=0.7)
  plt.title("The distribution of predictions for the best logistic regression model")
  plt.xlabel("Predicted Probability for Class 1")
  plt.ylabel("Count")
  plt.show()

  sns.histplot(y_pred_test_class1)

  # Add title
  plt.title('The distribution of predictions for Logistic Regression')

  # Show plot
  plt.show()

  # tpr (True Positive Rate) vs. fpr (False Positive Rate) dataframe
  # tp = True Positive
  # tn = True Negative
  # fp = False Positive
  # fn = False Negative
  # Decision Rule :  "y_pred>= Threshold" for Class "1"

  # when only_even=True --> we'll have a step ==0.02 and leave only even records

  def tpr_fpr_dataframe(y_true, y_pred, only_even=False):
      scores = []

      if only_even==False:
        thresholds = np.linspace(0, 1, 101) #[0, 0.01, 0.02, ...0.99,1.0]
      else:
        thresholds = np.linspace(0, 1, 51) #[0, 0.02, 0.04,  ...0.98,1.0]

      for t in thresholds:

          actual_positive = (y_true == 1)
          actual_negative = (y_true == 0)

          predict_positive = (y_pred >= t)
          predict_negative = (y_pred < t)

          tp = (predict_positive & actual_positive).sum()
          tn = (predict_negative & actual_negative).sum()

          fp = (predict_positive & actual_negative).sum()
          fn = (predict_negative & actual_positive).sum()

          if tp + fp > 0:
            precision = tp / (tp + fp)

          if tp + fn > 0:
            recall = tp / (tp + fn)

          if precision+recall > 0:
            f1_score = 2*precision*recall / (precision+recall)

          accuracy = (tp+tn) / (tp+tn+fp+fn)

          scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))

      columns = ['threshold', 'tp', 'fp', 'fn', 'tn','precision','recall', 'accuracy','f1_score']
      df_scores = pd.DataFrame(scores, columns=columns)

      df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
      df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

      return df_scores

  df_scores = tpr_fpr_dataframe(y_test,
                                y_pred_test_class1,
                                only_even=True)

  # full df_scores
  df_scores

  df_scores[(df_scores.threshold>=0.20) & (df_scores.threshold<=0.30)]

  """### 3.1.2 Decision Tree"""

  # predicting probability
  y_pred_test = clf_best.predict_proba(X_test)
  y_pred_test_class1 = [k[1] for k in y_pred_test] # k[1] is the second element in the list of Class predictions

  # Unconditional probability of a positive growth is 51.6%
  y_test.sum()/y_test.count()

  sns.histplot(y_pred_test_class1)

  # Add a title
  plt.title('The distribution of predictions for the current model (Decision Tree with max_depth=15)')

  # Show the plot
  plt.show()

  # Precision score points

  df_scores.plot.line(x='threshold',
                      y=['precision','recall', 'f1_score'],
                      title = 'Precision vs. Recall for the Best Model')



  """### 3.1.3 Random Forest"""

  # predicting probability
  y_pred_test = rf_model.predict_proba(X_test)
  y_pred_test_class1 = [k[1] for k in y_pred_test]  # k[1] is the second element in the list of Class predictions

  # example prediction of probabilities
  y_pred_test

  #without reduction
  y_pred_test_class1_df = pd.DataFrame(y_pred_test_class1, columns=['Class1_probability'])
  y_pred_test_class1_df.head()

  y_pred_test_class1_df.describe().T

  # Unconditional probability of a positive growth is 50.1% with reduction
  y_test.sum()/y_test.count()

  sns.histplot(y_pred_test_class1)

  # Add title
  plt.title('The distribution of predictions for Random Forest')

  # Show plot
  plt.show()

  # tpr (True Positive Rate) vs. fpr (False Positive Rate) dataframe
  # tp = True Positive
  # tn = True Negative
  # fp = False Positive
  # fn = False Negative
  # Decision Rule :  "y_pred>= Threshold" for Class "1"

  # when only_even=True --> we'll have a step ==0.02 and leave only even records

  def tpr_fpr_dataframe(y_true, y_pred, only_even=False):
      scores = []

      if only_even==False:
        thresholds = np.linspace(0, 1, 101) #[0, 0.01, 0.02, ...0.99,1.0]
      else:
        thresholds = np.linspace(0, 1, 51) #[0, 0.02, 0.04,  ...0.98,1.0]

      for t in thresholds:

          actual_positive = (y_true == 1)
          actual_negative = (y_true == 0)

          predict_positive = (y_pred >= t)
          predict_negative = (y_pred < t)

          tp = (predict_positive & actual_positive).sum()
          tn = (predict_negative & actual_negative).sum()

          fp = (predict_positive & actual_negative).sum()
          fn = (predict_negative & actual_positive).sum()

          if tp + fp > 0:
            precision = tp / (tp + fp)

          if tp + fn > 0:
            recall = tp / (tp + fn)

          if precision+recall > 0:
            f1_score = 2*precision*recall / (precision+recall)

          accuracy = (tp+tn) / (tp+tn+fp+fn)

          scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))

      columns = ['threshold', 'tp', 'fp', 'fn', 'tn','precision','recall', 'accuracy','f1_score']
      df_scores = pd.DataFrame(scores, columns=columns)

      df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
      df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

      return df_scores

  df_scores = tpr_fpr_dataframe(y_test,
                                y_pred_test_class1,
                                only_even=True)

  # full df_scores
  df_scores

  df_scores[(df_scores.threshold>=0.32) & (df_scores.threshold<=0.50)]

  # Try to find high Precision score points

  df_scores.plot.line(x='threshold',
                      y=['precision','recall', 'f1_score'],
                      title = 'Precision vs. Recall for the Best Model (Decision Tree with max_depth=15)')



  """## 3.2 Defining new columns with Predictions in new_df"""

  # adding Random Forest predictors to the dataset for new rules: Threshold = 0.32 and 0.50

  y_pred_all = rf_model.predict_proba(X_all)
  y_pred_all_class1 = [k[1] for k in y_pred_all] #list of predictions for class "1"
  y_pred_all_class1_array = np.array(y_pred_all_class1) # (Numpy Array) np.array of predictions for class "1" , converted from a list

  new_df['proba_pred15'] = y_pred_all_class1_array
  new_df['pred15_rf_best_rule_32'] = (y_pred_all_class1_array >= 0.32).astype(int)

  new_df['proba_pred16'] = y_pred_all_class1_array
  new_df['pred16_rf_best_rule_35'] = (y_pred_all_class1_array >= 0.35).astype(int)

  new_df['proba_pred17'] = y_pred_all_class1_array
  new_df['pred17_rf_best_rule_37'] = (y_pred_all_class1_array >= 0.37).astype(int)

  new_df['proba_pred18'] = y_pred_all_class1_array
  new_df['pred18_rf_best_rule_39'] = (y_pred_all_class1_array >= 0.39).astype(int)

  new_df['proba_pred19'] = y_pred_all_class1_array
  new_df['pred19_rf_best_rule_41'] = (y_pred_all_class1_array >= 0.41).astype(int)

  new_df['proba_pred20'] = y_pred_all_class1_array
  new_df['pred20_rf_best_rule_43'] = (y_pred_all_class1_array >= 0.43).astype(int)

  new_df['proba_pred21'] = y_pred_all_class1_array
  new_df['pred21_rf_best_rule_45'] = (y_pred_all_class1_array >= 0.45).astype(int)

  new_df['proba_pred22'] = y_pred_all_class1_array
  new_df['pred22_rf_best_rule_47'] = (y_pred_all_class1_array >= 0.47).astype(int)

  new_df['proba_pred23'] = y_pred_all_class1_array
  new_df['pred23_rf_best_rule_49'] = (y_pred_all_class1_array >= 0.49).astype(int)

  new_df['proba_pred24'] = y_pred_all_class1_array
  new_df['pred24_rf_best_rule_50'] = (y_pred_all_class1_array >= 0.50).astype(int) # best one


  # adding Logistic Regression
  y_pred_all = best_lr.predict_proba(X_all)
  y_pred_all_class1 = [k[1] for k in y_pred_all]
  y_pred_all_class1_array = np.array(y_pred_all_class1)

  new_df['proba_pred25'] = y_pred_all_class1_array
  new_df['pred25_lr_best_rule_25'] = (y_pred_all_class1_array >= 0.25).astype(int)

  new_df['proba_pred26'] = y_pred_all_class1_array
  new_df['pred26_lr_best_rule_26'] = (y_pred_all_class1_array >= 0.26).astype(int)

  new_df['proba_pred27'] = y_pred_all_class1_array
  new_df['pred27_lr_best_rule_27'] = (y_pred_all_class1_array >= 0.27).astype(int) 

  new_df['proba_pred28'] = y_pred_all_class1_array
  new_df['pred28_lr_best_rule_28'] = (y_pred_all_class1_array >= 0.28).astype(int)

  new_df['proba_pred29'] = y_pred_all_class1_array
  new_df['pred29_lr_best_rule_30'] = (y_pred_all_class1_array >= 0.30).astype(int)

  new_df['proba_pred30'] = y_pred_all_class1_array
  new_df['pred30_lr_best_rule_50'] = (y_pred_all_class1_array >= 0.50).astype(int)

  """## 3.3 Aggregated Stats an all perdictions

  """

  PREDICTIONS, IS_CORRECT = get_predictions_correctness(new_df, to_predict='Is_Positive_Growth_1h_Future')

  PREDICTIONS

  # best one
  new_df[(new_df.Split=='Test')&(new_df.pred14_lr==1)].Date.hist()

  # another strong one
  new_df[(new_df.Split=='Test')&(new_df.pred06_cci_oversold==1)].Date.hist()

  # Pred 14: How many positive prediction per day
  pred14_daily_positive_count = pd.DataFrame(new_df[(new_df.Split=='Test')&(new_df.pred14_lr==1)].groupby('Date')['pred14_lr'].count())

  # Pred 06: How many positive prediction per day
  pred06_daily_positive_count = pd.DataFrame(new_df[(new_df.Split=='Test')&(new_df.pred06_cci_oversold==1)].groupby('Date')['pred06_cci_oversold'].count())

  # Unique trading days on Test
  new_df[(new_df.Split=='Test')].Date.nunique()

  pred14_daily_positive_count

  pred14_daily_positive_count.hist()

  pred14_daily_positive_count.describe().T

  """The statistics for predictions with a threshold of 0.47 show an average of about 5 positive predictions per day, with values ranging from 1 to 20. Most days have between 4 and 7 positive predictions, indicating moderate variability in daily prediction counts."""

  pred06_daily_positive_count.hist()

  # all predictions on MODELS (not-manual predictions)
  PREDICTIONS_ON_MODELS = [p for p in PREDICTIONS if int(p.split('_')[0].replace('pred', ''))>=5]
  PREDICTIONS_ON_MODELS

  # all predictions on Models - correctness
  IS_CORRECT_ON_MODELS = [p for p in IS_CORRECT if int(p.replace('is_correct_pred', ''))>=5]
  IS_CORRECT_ON_MODELS

  # predictions on models
  # pred10_rf_best_rule_60: ONLY 2% of TEST cases predicted with high confidence of growth
  print(new_df.groupby('Split')[PREDICTIONS_ON_MODELS].agg(['count','sum','mean']).T)

  #Export for next file
  # Saving Model (clf_10)
  with open("clf_10_model.pkl", "wb") as f:
          pickle.dump(clf_10, f)

  # Saving best trained model (clf_best)
  with open("clf_best_model.pkl", "wb") as f:
          pickle.dump(clf_best, f)

  # Saving best model (rf)
  with open("rf_model_model.pkl", "wb") as f:
          pickle.dump(rf_model, f)

  # Saving best reduced by feature importance model (rf)
  with open("rf_model_reduced.pkl", "wb") as f:
          pickle.dump(rf_model_reduced, f)

  # Saving best trained model (best_lr)
  with open("logistic_regression_model.pkl", "wb") as f:
          pickle.dump(best_lr, f)

  with open('lists_pred.pkl', 'wb') as f:
      pickle.dump((PREDICTIONS, IS_CORRECT), f)

  new_df.to_csv("modeled_df.csv", index=False)

if __name__ == "__main__":
    modeling()