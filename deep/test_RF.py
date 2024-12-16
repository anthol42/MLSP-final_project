from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from datetime import datetime
from backtest.data import JSONCache, FetchCharts, Cache, FilterNoneCharts, CausalImpute
from pipes import Finviz, RmTz, FromFile
import pandas as pd
import numpy as np
from data import annotate_chart_change, split_data, sample_tickers, ImageDataset
from metrics import accuracy, custom_precision, precision_2d
import json
import pandas_ta as ta
import pytz

DATASET = "huge"
HSEARCH = False
FRACT = 0.1

if '__main__' == __name__:

    if DATASET == "huge":
        pipe = Finviz("https://finviz.com/screener.ashx?v=111&f=ipodate_more10", True) | JSONCache() | \
               FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
               Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
    elif DATASET == "small":
        pipe = FromFile("tw50.json") | JSONCache() | \
               FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
               Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
        pipe.set_id(10)
    elif DATASET == "smallUS":
        pipe = FromFile("us50.json") | JSONCache() | \
               FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
               Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
        pipe.set_id(20)
    else:
        raise ValueError(f"Invalid dataset: {DATASET}")
    start, end = datetime(2000, 1, 1), datetime(2020, 1, 1)
    data = pipe.get(start, end)
    if FRACT < 1.:
        tickers = sample_tickers(data, FRACT, seed=42)
        data = ImageDataset.sample_data(data, tickers)

    processed_data = {}
    annotation = {}

    for ticker, chart in data.items():

        chart['rsi'] = ta.rsi(chart['Close'])
        chart['SO'] = ta.stoch(high = chart['High'], low = chart['Low'], close = chart['Close']).iloc[:, 0]
        chart['Will_R'] = ta.willr(high = chart['High'], low = chart['Low'], close = chart['Close'])
        chart['MACD'] = ta.macd(close = chart['Close']).iloc[:, 0]
        chart['PRoC'] = ta.roc(close = chart['Close'])
        chart['OBV'] = ta.obv(close = chart['Close'], volume = chart['Volume'])
        chart['anno'] = annotate_chart_change(data[ticker].values)
        chart.dropna(subset=['anno'], inplace = True)
        processed_data[ticker] = chart


    # Spliting the data into train/validation/test data/labels
    # train_start = datetime(2000, 1, 1).replace(tzinfo=pytz.UTC)
    validate_start = datetime(2016, 12, 31) # .replace(tzinfo=pytz.UTC)
    test_start = datetime(2018, 6, 13) # .replace(tzinfo=pytz.UTC)
    # test_end = datetime(2020, 1, 1).replace(tzinfo=pytz.UTC)

    # X_train = {}
    # X_validate = {}
    # X_test = {}
    X_train = split_data(data, start=start, end=validate_start)
    X_validate = split_data(data, start=validate_start, end=test_start)
    X_test = split_data(data, start=test_start, end=end)
    # for tick, data in processed_data.items():
    #     X_train[tick] = data.loc[:validate_start]
    #     X_validate[tick] = data.loc[validate_start:test_start]
    #     X_test[tick] = data.loc[test_start:]

    X_train = pd.concat(X_train.values(), axis=0)
    X_validate = pd.concat(X_validate.values(), axis=0)
    X_test = pd.concat(X_test.values(), axis=0)

    Y_train = X_train['anno']
    Y_validate = X_validate['anno']
    Y_test = X_test['anno']

    X_train.drop('anno', axis=1, inplace=True)
    X_validate.drop('anno', axis=1, inplace=True)
    X_test.drop('anno', axis=1, inplace=True)

    print('Data preparation complete')


    # # Initiating - Training - Testing
    if HSEARCH:
        rf = RandomForestClassifier(random_state=42)
        print('Setting done')

        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        randomized_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=50,  # Number of parameter settings sampled
            scoring='accuracy',
            cv=3,  # 3-fold cross-validation
            verbose=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        print("Starting hyperparameter optimization...")
        randomized_search.fit(X_train, Y_train)

        print("Best parameters found:")
        print(randomized_search.best_params_)

        # Train the final model with the best parameters
        best_rf = randomized_search.best_estimator_
    else:
        best_rf = RandomForestClassifier(random_state=42, max_depth=10, n_jobs=-1)

    # Get label ditstibution
    _, counts_train = np.unique(Y_train, return_counts=True)
    _, counts_val = np.unique(Y_validate, return_counts=True)
    _, counts_test = np.unique(Y_test, return_counts=True)
    print(f"Train label distribution: 0: {counts_train[0]}, 1: {counts_train[1]}, total: {len(Y_train)}")
    print(f"Validation label distribution: 0: {counts_val[0]}, 1: {counts_val[1]}, total: {len(Y_validate)}")
    print(f"Test label distribution: 0: {counts_test[0]}, 1: {counts_test[1]}, total: {len(Y_test)}")
    print("-----------------------------------------------------------------------------------------------")
    print(f"Train label positive ratio: {counts_train[1] / len(Y_train)}")
    print(f"Validation label positive ratio: {counts_val[1] / len(Y_validate)}")
    print(f"Test label positive ratio: {counts_test[1] / len(Y_test)}")
    best_rf.fit(X_train, Y_train)
    print('Fitting done')

    # Prediction
    pred = best_rf.predict(X_test)
    print('Prediction done')
    print(Y_test.values, pred)

    # Evaluating with custom metrics
    print("Accuracy on test data:",
          accuracy_score(y_true=Y_test.values, y_pred=pred))