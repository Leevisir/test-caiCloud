import pandas as pd
import xgboost as xgb
import logging
logger = logging.getLogger(__name__)


def fillinmissing(data, dtindex, fillin=None, indicator=False):
    '''This function takes a data frame that is indexed by standard datetime index.
    It completes the data frame by encoding values to missing records.
        Args:
            data: a data frame that is indexed by datetime index with missing records to be filled in.
            dtindex: a full datetime index list as a reference to locate the missing records.
            fillin: indicate what value should be filled in.
            indicator: if is True. The function will add an additional column indicts which row is newly filled in.
        Returns:
            A data frame without missing records.
    '''
    fulldata = pd.DataFrame(index=dtindex)
    fulldata = fulldata.join(data)

    if indicator is True:
        ismissing = pd.notna(fulldata)
        fulldata = fulldata.fillna(fillin)
        return fulldata, ismissing

    return fulldata


def get_lag(data, lags, unit):
    out = pd.DataFrame(index=data.index)
    for lag in lags:
        ld = data.shift(lag, freq=unit)
        colnames = ["%slag%s%s" % (cn,str(lag),unit) for cn in data.columns]
        ld.columns = colnames
        out = pd.merge(out,ld,how='outer',left_index=True,right_index=True)
    return out.dropna()