from __future__ import division, print_function
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import rampwf as rw


problem_title = 'Solar wind classification'
Predictions = rw.prediction_types.make_multiclass(label_names=[0, 1, 2])
# # The time-series feature extractor step (the first step of the ElNino)
# # workflow takes two additional parameters the parametrize the lookahead
# # check. You may want to change these at the backend or add more check
# # points to avoid that people game the setup.
# workflow = rw.workflows.ElNino(check_sizes=[132], check_indexs=[13])

score_types = [
    rw.score_types.Accuracy(name='acc', precision=3),
]

# # We do an 8-fold block cv. With the given parameters, we have
# # length of common block: 300 months = 25 years
# # length of validation block: 288 months = 24 years
# # length of each cv block: 36 months = 3 years
# cv = rw.cvs.TimeSeries(
#     n_cv=8, cv_block_size=0.5, period=12, unit='month', unit_2='year')
# get_cv = cv.get_cv
cv = KFold(n_splits=10)
get_cv = cv.split


def _read_data(path, type_):

    fname = 'data_{}.parquet'.format(type_)
    fp = os.path.join(path, 'data', fname)

    data = pd.read_parquet(fp)

    y = data['label']
    del data['label']

    # for the "quick-test" mode, use less data
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        N_small = 100000
        data = data[:N_small]
        y = y[:N_small]
        # add some random to ensure each CV fold has all classes
        y[np.random.randint(N_small, size=100)] = 1
        y[np.random.randint(N_small, size=100)] = 2

    return data, y


def get_train_data(path='.'):
    return _read_data(path, 'train')


def get_test_data(path='.'):
    return _read_data(path, 'test')


class TimeSeriesClassifier(object):
    def __init__(self, workflow_element_names=[
            'ts_feature_extractor', 'classifier']):
        self.element_names = workflow_element_names
        self.feature_extractor_workflow = rw.workflows.FeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = rw.workflows.Classifier(
            [self.element_names[1]])

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_df, y_array, train_is)
        X_train_array = self.feature_extractor_workflow.test_submission(
            fe, X_df.iloc[train_is])
        clf = self.classifier_workflow.train_submission(
            module_path, X_train_array, y_array[train_is])
        return fe, clf

    def test_submission(self, trained_model, X_df):
        fe, clf = trained_model
        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_df)
        y_proba = self.classifier_workflow.test_submission(clf, X_test_array)
        return y_proba


workflow = TimeSeriesClassifier()


# import numpy as np
# from rampwf.workflows.ts_feature_extractor import TimeSeriesFeatureExtractor
# from rampwf.workflows import Classifier


# class TimeSeriesClassifier(object):
#     def __init__(self, check_sizes, check_indexs, workflow_element_names=[
#             'ts_feature_extractor', 'classifier']):
#         self.element_names = workflow_element_names
#         self.ts_feature_extractor_workflow = TimeSeriesFeatureExtractor(
#             check_sizes, check_indexs, [self.element_names[0]])
#         self.classifier_workflow = Classifier([self.element_names[1]])

#     def train_submission(self, module_path, X_df, y_array, train_is=None):
#         """
#         Train a time series feature extractor + regressor workflow.

#         `X_ds` is `n_burn_in` longer than `y_array` since `y_array` contains
#         targets without the initial burn in period. `train_is` are wrt
#         `y_array`, so `X_ds` has to be _extended_ by `n_burn_in` when sent to
#         the time series feature extractor.

#         """
#         if train_is is None:
#             # slice doesn't work here because of the way `extended_train_is`
#             # is computed below
#             train_is = np.arange(len(y_array))
#         ts_fe = self.ts_feature_extractor_workflow.train_submission(
#             module_path, X_ds, y_array)

#         n_burn_in = X_ds.n_burn_in
#         # X_ds contains burn-in so it needs to be extended by n_burn_in
#         # timesteps. This assumes that train_is is a block of consecutive
#         # time points.
#         burn_in_range = np.arange(train_is[-1], train_is[-1] + n_burn_in)
#         extended_train_is = np.concatenate((train_is, burn_in_range))
#         X_train_ds = X_ds.isel(time=extended_train_is)
#         # At this point X_train_ds is n_burn_in longer than y_array[train_is]

#         # ts_fe.transform should return an array corresponding to time points
#         # without burn in, so X_train_array and y_array[train_is] should now
#         # have the same length.
#         X_train_array = self.ts_feature_extractor_workflow.test_submission(
#             ts_fe, X_train_ds)

#         reg = self.regressor_workflow.train_submission(
#             module_path, X_train_array, y_array[train_is])
#         return ts_fe, reg

#     def test_submission(self, trained_model, X_ds):
#         ts_fe, reg = trained_model
#         X_test_array = self.ts_feature_extractor_workflow.test_submission(
#             ts_fe, X_ds)
#         y_pred = self.regressor_workflow.test_submission(reg, X_test_array)
#         return y_pred
