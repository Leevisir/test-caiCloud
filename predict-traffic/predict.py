import pandas as pd
import xgboost as xgb
import logging
logger = logging.getLogger(__name__)

import utils as ut
import preprocess as ft

# XGBoost configuration
XGBOOST_PARAMS = {
    'max_depth': 2,
    'eta': .005,
    'verbosity': 0,
    'objective': 'reg:squarederror',
    'nthread': 5,
    'eval_metric': 'rmse'
}
NUMROUND = 100
FREQUENCY = 'D'
EARLY_STOPPING_ROUNDS = 10


def train_test_split(X,Y,split):
    X_train = X.loc[X.index <= split].copy()
    Y_train =  Y.loc[Y.index <= split].copy()
    X_test = X.loc[X.index > split].copy()
    Y_test =  Y.loc[Y.index > split].copy()

    return X_train,Y_train, X_test, Y_test

def train(X_train,Y_train,X_test,Y_test):
    train_data = xgb.DMatrix(X_train, label=Y_train)
    test_data = xgb.DMatrix(X_test, label=Y_test)
    eval_list = [(test_data, 'eval'), (train_data, 'train')]
    model = xgb.train(
        XGBOOST_PARAMS, 
        train_data, 
        NUMROUND, 
        eval_list,
        verbose_eval=False, 
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )

    return model

# Get fitted values and align time series to historical values
def get_fitted_and_align(model,X, idx_all):
    Y_hat = model.predict(xgb.DMatrix(X))
    Y_hat = Y_hat.tolist()
    X_idx = get_date_from_col(X.index.tolist())
    X_idx_head = X_idx[0]
    
    # Pad start of predictions
    for i in idx_all:
        if i == X_idx_head:
            break
        else:
            Y_hat.insert(0, None)

    # Pad end of predictions
    padding = len(idx_all) - len(Y_hat)
    for i in range(padding):
        Y_hat.append(None)

    fitted = ut.TimeSeries(
        x = idx_all,
        y = Y_hat
    )

    return fitted

def get_date_from_col(col):
    return list(map(lambda x: str(x).split(' ')[0], list(col)))

def predict_timeseries(data):
    # Initialize values, copy some data
    start = str(data.iloc[0]['datetime'])
    end = str(data.iloc[-1]['datetime'])
    historical = ut.TimeSeries(
        x = get_date_from_col(data['datetime']),
        y = data['y'].tolist()
    )

    # Construct features
    dtindex = pd.date_range(start=start, end=end, freq=FREQUENCY)
    data.set_index('datetime', inplace=True)
    data, isencoded = ft.fillinmissing(data=data, dtindex=dtindex, fillin=0, indicator=True)
    datalag = ft.get_lag(data=data, lags=range(14, 30), unit='D')
    fulllist = [data, datalag]
    fulldf = pd.concat(fulllist, axis=1).dropna()
    forecastdf = pd.concat(fulllist, axis=1).iloc[-14:, 1:]

    # Make train/test sets
    splitpoint = int(0.8 * len(dtindex))
    splitdt = str(dtindex[splitpoint])
    Y = fulldf['y']
    X = fulldf.drop(columns='y')
    X_train, Y_train, X_test, Y_test = train_test_split(X,Y,splitdt)
    
    # Fit model and forecast values
    model = train(X_train, Y_train, X_test, Y_test)
    prediction = model.predict(xgb.DMatrix(forecastdf))
    predicted = ut.TimeSeries(
        x = get_date_from_col(forecastdf.index.tolist()),
        y = prediction.tolist()
    )    

    # Get fitted values, align with historical data
    fitted = get_fitted_and_align(model,X_train,historical.x)

    return historical, fitted, predicted
