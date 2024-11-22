from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
from backtest.data import FetchCharts, Cache
import pandas as pd
import numpy as np
from data import annotate_chart_change, split_data
from metrics import accuracy, custom_precision, precision_2d
import json
import pandas_ta as ta
import pytz


if '__main__' == __name__:

    with open('./us50.json', 'r') as f:
        data = json.load(f)
        pipi = FetchCharts(data) | Cache()
        us50 = pipi.get(datetime(2000, 1, 1),datetime(2020, 1, 1))

    processed_data = {}
    annotation = {}

    for ticker, chart in us50.items():

        chart['rsi'] = ta.rsi(chart['Close'])
        chart['SO'] = ta.stoch(high = chart['High'], low = chart['Low'], close = chart['Close']).iloc[:, 0]
        chart['Will_R'] = ta.willr(high = chart['High'], low = chart['Low'], close = chart['Close'])
        chart['MACD'] = ta.macd(close = chart['Close']).iloc[:, 0]
        chart['PRoC'] = ta.roc(close = chart['Close'])
        chart['OBV'] = ta.obv(close = chart['Close'], volume = chart['Volume'])
        chart['anno'] = annotate_chart_change(us50[ticker].values)
        chart.dropna(subset=['anno'], inplace = True)
        processed_data[ticker] = chart


    # Spliting the data into train/validation/test data/labels
    train_start = datetime(2000, 1, 1).replace(tzinfo=pytz.UTC)
    validate_start = datetime(2016, 12, 31).replace(tzinfo=pytz.UTC)
    test_start = datetime(2018, 6, 13).replace(tzinfo=pytz.UTC)
    test_end = datetime(2020, 1, 1).replace(tzinfo=pytz.UTC)

    X_train = {}
    X_validate = {}
    X_test = {}

    for tick, data in processed_data.items():
        X_train[tick] = data.loc[train_start:validate_start]
        X_validate[tick] = data.loc[validate_start:test_start]
        X_test[tick] = data.loc[test_start:test_end]

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
rf = RandomForestClassifier(n_estimators=100, verbose=1, random_state=42)
print('Setting done')
rf.fit(X_train, Y_train)
print('Fitting done')
pred = rf.predict(X_test)
print('Prediction done')
print(Y_test.values, pred)
# Evaluating with custom metrics
print(accuracy_score(y_true=Y_test.values, y_pred=pred))