from __future__ import division, print_function
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score

import rampwf as rw
from rampwf.score_types.classifier_base import ClassifierBaseScoreType


problem_title = 'Solar wind classification'

Predictions = rw.prediction_types.make_multiclass(label_names=[0, 1])

workflow = rw.workflows.FeatureExtractorClassifier()


class PointWisePrecision(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='pw_prec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = precision_score(y_true_label_index, y_pred_label_index)
        return score


class PointWiseRecall(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='pw_rec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = recall_score(y_true_label_index, y_pred_label_index)
        return score


score_types = [
    # log-loss
    rw.score_types.NegativeLogLikelihood(name='pw_ll'),
    # point-wise (for each time step) precision and recall
    PointWisePrecision(),
    PointWiseRecall()
]


def get_cv(X, y):
    # 10 fold cross-validation based on 5 splits
    n_splits = 5
    cv = KFold(n_splits=n_splits)
    splits = list(cv.split(X, y))
    # 5 folds, each point is in test set 4x
    # set k to a lower number if you want less folds
    pattern = [
        ([2, 3, 4], [0, 1]), ([0, 1, 4], [2, 3]), ([0, 2, 3], [1, 4]),
        ([0, 1, 3], [2, 4]), ([1, 2, 4], [0, 3]), ([0, 1, 2], [3, 4]),
        ([0, 2, 4], [1, 3]), ([1, 2, 3], [0, 4]), ([0, 3, 4], [1, 2]),
        ([1, 3, 4], [0, 2])
    ]
    for ps in pattern:
        yield (np.hstack([splits[p][1] for p in ps[0]]),
               np.hstack([splits[p][1] for p in ps[1]]))


def _read_data(path, type_):

    fname = 'data_{}.parquet'.format(type_)
    fp = os.path.join(path, 'data', fname)
    data = pd.read_parquet(fp)

    fname = 'labels_{}.csv'.format(type_)
    fp = os.path.join(path, 'data', fname)
    labels = pd.read_csv(fp)

    # convert labels into continuous array

    labels['begin'] = pd.to_datetime(
        labels['begin'], format="%Y-%m-%d %H:%M:%S")
    labels['end'] = pd.to_datetime(labels['end'], format="%Y-%m-%d %H:%M:%S")

    # problem with identical begin / end previous label with reindexing method
    mask = labels['begin'] == pd.Timestamp('2000-11-11 04:10:00')
    labels.loc[mask, 'begin'] += pd.Timedelta('20min')

    labels['end'] = labels['end'] + pd.Timedelta('10min')
    labels.columns.name = 'label'
    labels = labels[['begin', 'end']].stack().reset_index(name='time')
    labels['label'] = labels['label'].replace({'begin': 1, 'end': 0})
    labels = labels.set_index('time')['label']

    y = labels.reindex(data.index, method='ffill')
    # remaining NaNs at beginning of series
    y = y.fillna(0).astype(int)

    # easier but slow method
    # y = pd.Series(0, index=data.index)
    # for begin, end in labels[['begin', 'end']].itertuples(index=False):
    #     y.loc[begin:end] = 1

    # for the "quick-test" mode, use less data
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        N_small = 50000
        data = data[:N_small]
        y = y[:N_small]

    return data, y


def get_train_data(path='.'):
    return _read_data(path, 'train')


def get_test_data(path='.'):
    return _read_data(path, 'test')
