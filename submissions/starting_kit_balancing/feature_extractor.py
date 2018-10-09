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
    # X_df_new = compute_Beta(X_df_new)
    # X_df_new = compute_rolling_std(X_df_new, '15min', 'Beta')
    X_df_new = compute_rolling_std(X_df_new, 'Beta', '2h')

    return X_df_new


def compute_Beta(data):
    """
    Compute the evolution of the Beta for data.

    The function assume data already has ['Np','B','Vth'] features.
    """
    try:
        data['Beta'] = 1e6 * data['Vth'] * data['Vth'] * constants.m_p * data[
            'Np'] * 1e6 * constants.mu_0 / (1e-18 * data['B'] * data['B'])
    except KeyError:
        ValueError('Error computing Beta,B,Vth or Np'
                   ' might not be loaded in dataframe')
    return data


def compute_rolling_std(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    time_indow : str
        string that defines the length of the time window passed to `rolling`
    center : boolean
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    data[name] = data[feature].rolling(time_window, center=center).std()
    data[name] = data[name].ffill().bfill()
    data[name] = data[name].astype(data[feature].dtype)
    return data
