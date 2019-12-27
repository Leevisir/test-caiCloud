import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _validate_format(data):
    if not isinstance(data, pd.DataFrame):
        logger.error("Expected a DataFrame")

    #validate date
    if len(data.columns) != 2:
        raise ValueError(
            'Data format not correct!'
        )

def _validate_dates(data):
    if data['datetime'].dt.tz is not None:
        raise ValueError(
            'Column date has timezone specified, which is not supported. '
            'Remove timezone.'
        )

    if data['datetime'].isnull().any():
        raise ValueError('Found NaN in column date.')


def _validate_response(data):
    if np.isinf(data['y'].values).any():
        raise ValueError('Found infinity in column y.')


def process_input(data):
    '''
    :param data: a DataFrame where first column contains date information, second contains response
    '''
    # Check that input data is in the correct format
    _validate_format(data)

    data = data.drop_duplicates()
    data = data.dropna()
    data = data.reset_index(drop=True)
    data.columns = ['datetime', 'y']

    if data['datetime'].dtype == np.int64:
        data['datetime'] = data['datetime'].astype(str)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Validates the pre-processed data
    _validate_dates(data)
    _validate_response(data)

    # extract date
    data['datetime'] = data['datetime'].dt.date

    return data

if __name__ == "__main__":
    data = pd.read_csv('./test_data.csv')
    pd = preprocess(data)