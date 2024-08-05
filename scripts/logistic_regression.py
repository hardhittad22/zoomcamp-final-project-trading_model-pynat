import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def fit_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test):
    precision_matrix = {}
    best_precision = 0
    best_C = 0
    best_iter = 0

    for c in [1, 0.1, 0.01]:
        for iter in [50, 100, 200]:
            print(f'Working with HyperParams: C = {c} (positive float, smaller = stronger regularization), max_iter={iter}')
            lr = LogisticRegression(C=c,
                                    random_state=42,
                                    max_iter=iter,
                                    solver='sag',
                                    n_jobs=-1)

            lr.fit(X_train, y_train)

            y_pred_valid = lr.predict(X_valid)
            precision_valid = precision_score(y_valid, y_pred_valid)
            y_pred_test = lr.predict(X_test)
            precision_test = precision_score(y_test, y_pred_test)
            print(f'  Precision on test is {precision_test}, (precision on valid is {precision_valid} - tends to overfit)')

            precision_matrix[(c, iter)] = round(precision_test, 4)

            if precision_test >= best_precision:
                best_precision = round(precision_test, 4)
                best_C = c
                best_iter = iter
                print(f'New best precision found for C={c}, max_iter={iter}')
                print('------------------------------')

    print(f'Matrix of precisions: {precision_matrix}')
    print(f'The best precision is {best_precision} and the best C is {best_C}, best max_iter = {best_iter}')

    return best_C, best_iter, precision_matrix

def tpr_fpr_dataframe(y_true, y_pred, only_even=False):
    scores = []

    if not only_even:
        thresholds = np.linspace(0, 1, 101)  # [0, 0.01, 0.02, ...0.99,1.0]
    else:
        thresholds = np.linspace(0, 1, 51)  # [0, 0.02, 0.04,  ...0.98,1.0]

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

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'accuracy', 'f1_score']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores

def train_logistic_regression_model():
    new_df = pd.read_csv('updated_predictions2.csv')

    with open('data_split.pkl', 'rb') as f:
        split_data = pickle.load(f)
        X_valid = split_data['X_valid']
        y_valid = split_data['y_valid']
        y_train = split_data['y_train']
        X_test = split_data['X_test']
        y_test = split_data['y_test']
        X_all = split_data['X_all']
        X_train = split_data['X_train']

    best_C, best_iter, precision_matrix_lr = fit_logistic_regression(X_train=X_train, y_train=y_train,
                                                                     X_valid=X_valid, y_valid=y_valid,
                                                                     X_test=X_test, y_test=y_test)

    # Train the Logistic Regression model with the best hyperparameters
    best_lr = make_pipeline(RobustScaler(), LogisticRegression(C=best_C, max_iter=best_iter, solver='liblinear'))
    best_lr.fit(X_train, y_train)

    y_pred_all = best_lr.predict(X_all)
    new_df['pred14_lr'] = y_pred_all

    # Predict Probability
    y_pred_test = best_lr.predict_proba(X_test)
    y_pred_test_class1 = [k[1] for k in y_pred_test]

    # Plot the distribution of predictions
    plt.hist(y_pred_test_class1, bins=30, edgecolor='k', alpha=0.7)
    plt.title("The distribution of predictions for the best logistic regression model")
    plt.xlabel("Predicted Probability for Class 1")
    plt.ylabel("Count")
    plt.show()

    sns.histplot(y_pred_test_class1)
    plt.title('The distribution of predictions for Logistic Regression')
    plt.show()

    # Generate TPR vs FPR dataframe
    df_scores = tpr_fpr_dataframe(y_test, y_pred_test_class1, only_even=True)
    print(df_scores[(df_scores.threshold >= 0.20) & (df_scores.threshold <= 0.30)])

    # Plot Precision, Recall, and F1-Score vs Threshold
    df_scores.plot.line(x='threshold',
                        y=['precision', 'recall', 'f1_score'],
                        title='Precision vs. Recall for the Best Model Logistic Regression')
    plt.show()

    # Add probability predictions to dataframe
    y_pred_all = best_lr.predict_proba(X_all)
    y_pred_all_class1 = [k[1] for k in y_pred_all]
    y_pred_all_class1_array = np.array(y_pred_all_class1)

    # Add predictions with different thresholds to the dataframe
    for threshold in [0.25, 0.26, 0.27, 0.28, 0.30, 0.50]:
        prob_col = f'proba_pred{int(threshold * 100)}'
        pred_col = f'pred{int(threshold * 100)}_lr_best_rule_{int(threshold * 100)}'
        new_df[prob_col] = y_pred_all_class1_array
        new_df[pred_col] = (y_pred_all_class1_array >= threshold).astype(int)

    # Visualize Precision Scores for Logistic Regression
    rows = []
    for key, value in precision_matrix_lr.items():
        C, max_iter = key
        combination_label = f'C={C}, max_iter={max_iter}'
        rows.append({'Combination': combination_label, 'Precision': value})

    df = pd.DataFrame(rows)
    df['Precision'] = df.Precision * 100.0  # Convert to percentage for visualization

    fig = px.bar(df,
                 x='Combination',
                 y='Precision',
                 text='Precision')

    fig.update_layout(
        xaxis_title='Hyperparams combinations of <C, Max Iterations>',
        yaxis_title='Precision Score',
        xaxis_tickangle=-45,
        title={
            'text': 'Precision Scores for Various Logistic Regression Hyperparameter Combinations',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    fig.update_traces(texttemplate='%{text:.2f}%',
                      textposition='inside',
                      textfont_color='white')

    fig.show()

    new_df.to_csv('updated_predictions3.csv', index=False)

if __name__ == "__main__":
    train_logistic_regression_model()
