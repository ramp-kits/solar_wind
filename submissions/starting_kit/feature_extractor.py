import numpy as np
import pandas as pd
from scipy import constants

from joblib import Memory


memory = Memory(cachedir='./cache', verbose=1)


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        return _transform(X_df)


@memory.cache
def _transform(X_df):
    """
    Cached version of the transform method.
    """
    X_df_new = X_df.copy()
    #X_df_new = computeBeta(X_df_new)
    #X_df_new = computeRollingStd(X_df_new, '15min', 'Beta')
    X_df_new = computeRollingStd(X_df_new, '2h', 'Beta')

    # for now impute missing values (need to deal with burn in period)
    X_df_new.fillna(X_df_new.median(), inplace=True)

    return X_df_new


def computeBeta(data):
    """
    Compute the evolution of the Beta for data
    data is a Pandas dataframe
    The function assume data already has ['Np','B','Vth'] features
    """
    try:
        data['Beta'] = 1e6 * data['Vth'] * data['Vth'] * constants.m_p * data[
            'Np'] * 1e6 * constants.mu_0 / (1e-18 * data['B'] * data['B'])
    except KeyError:
        ValueError('Error computing Beta,B,Vth or Np'
                   ' might not be loaded in dataframe')
    return data


def computeRollingStd(data, time_window, feature, center=False):
    """
    For a given dataframe, compute the standard dev over
    a defined period of time (timeWindow) of a defined feature*
    ARGS :
    data : dataframe
    feature : feature in the dataframe we wish to compute the rolling mean from
                (string format)
    timeWindow : string that defines the length of the timeWindow (see pds doc)
    center : boolean to indicate if the point of the dataframe considered is
    center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    data[name] = data[feature].rolling(time_window, center=center).std()
    return data
