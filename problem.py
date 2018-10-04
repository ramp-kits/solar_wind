from __future__ import division, print_function
import os
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows.feature_extractor import FeatureExtractor
from rampwf.workflows.classifier import Classifier


problem_title = 'Solar wind classification'


# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------


class FeatureExtractorClassifier(object):

    def __init__(self):
        self.element_names = ['feature_extractor', 'classifier']
        self.feature_extractor_workflow = FeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = Classifier([self.element_names[1]])

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
        y_df_proba = pd.DataFrame(y_proba, index=X_df.index)
        return y_df_proba


workflow = FeatureExtractorClassifier()


# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------


BaseMultiClassPredictions = rw.prediction_types.make_multiclass(
    label_names=[0, 1])


class Predictions(BaseMultiClassPredictions):

    def __init__(self, y_pred=None, y_true=None, n_samples=None):
        # override init to not convert y_pred/y_true to arrays
        if y_pred is not None:
            self.y_pred = y_pred
        elif y_true is not None:
            self._init_from_pred_labels(y_true)
            self.y_pred = pd.DataFrame(self.y_pred, y_true.index)
        elif n_samples is not None:
            print("Here I should never be !!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.y_pred = np.empty((n_samples, self.n_columns), dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    @property
    def y_pred_label_index(self):
        """Multi-class y_pred is the index of the predicted label."""
        return np.argmax(self.y_pred.values, axis=1)


# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------


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


class EventWisePrecision(BaseScoreType):
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='ev_prec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        event_true = turnPredictionToEventList(y_true[1])
        event_pred = turnPredictionToEventList(y_pred[1])
        FP = [x for x in event_pred
              if max(overlapWithList(x, event_true, percent=True)) < 0.4]
        FP_too_short = [x for x in FP
                        if x.duration < datetime.timedelta(hours=2.5)]
        for event in FP_too_short:
            FP.remove(event)
        score = 1-len(FP)/len(event_pred)
        return score


class EventWiseRecall(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='ev_rec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        event_true = turnPredictionToEventList(y_true[1])
        event_pred = turnPredictionToEventList(y_pred[1])
        FN = 0
        for event in event_true:
            corresponding = find(event, event_pred, 0.5, 'best')
            if corresponding is None:
                FN += 1
        score = 1-FN/len(event_true)
        return score


class Event:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.duration = self.end-self.begin

    def __str__(self):
        return "{} ---> {}".format(self.begin, self.end)

    def __repr__(self):
        return "Event({} ---> {})".format(self.begin, self.end)


def overlap(event1, event2):
    '''return the time overlap between two events as a timedelta'''
    delta1 = min(event1.end, event2.end)
    delta2 = max(event1.begin, event2.begin)
    return max(delta1 - delta2, datetime.timedelta(0))


def overlapWithList(ref_event, event_list, percent=False):
    '''
    return the list of the overlaps between an event and the elements of
    an event list
    Have the possibility to have it as the percentage of fthe considered event
    in the list
    '''
    if percent:
        return [overlap(ref_event, elt) / elt.duration for elt in event_list]
    else:
        return [overlap(ref_event, elt) for elt in event_list]


def isInList(ref_event, event_list, thres):
    '''
    returns True if ref_event is overlapped thres percent of its duration by
    at least one elt in event_list
    '''
    return max(overlapWithList(ref_event,
                               event_list)) > thres*ref_event.duration


def merge(event1, event2):
    return Event(event1.begin, event2.end)


def choseEventFromList(ref_event, event_list, choice='first'):
    '''
    return an event from even_list according to the choice adopted
    first return the first of the lists
    last return the last of the lists
    best return the one with max overlap
    merge return the combination of all of them
    '''
    if choice == 'first':
        return event_list[0]
    if choice == 'last':
        return event_list[-1]
    if choice == 'best':
        return event_list[np.argmax(overlapWithList(ref_event, event_list))]
    if choice == 'merge':
        return merge(event_list[0], event_list[-1])


def find(ref_event, event_list, thres, choice='best'):
    '''
    Return the event in event_list that overlap ref_event for a given threshold
    if it exists
    Choice give the preference of returned :
    first return the first of the lists
    Best return the one with max overlap
    merge return the combination of all of them
    '''
    if isInList(ref_event, event_list, thres):
        return(choseEventFromList(ref_event, event_list, choice))
    else:
        return None


def turnPredictionToEventList(y, thres=0.5):
    '''
    Consider y as a pandas series, returns a list of Events corresponding to
    the requested label (int), works for both smoothed and expected series
    Delta corresponds to the series frequency (in our basic case with random
    index, we consider this value to be equal to 2)
    '''
    listOfPosLabel = y[y > thres]
    deltaBetweenPosLabel = listOfPosLabel.index[1:] - listOfPosLabel.index[:-1]
    deltaBetweenPosLabel.insert(0, datetime.timedelta(0))
    endOfEvents = np.where(deltaBetweenPosLabel >
                           datetime.timedelta(minutes=10))[0]
    indexBegin = 0
    eventList = []
    for i in endOfEvents:
        end = i
        eventList.append(Event(listOfPosLabel.index[indexBegin],
                         listOfPosLabel.index[end]))
        indexBegin = i + 1
    eventList.append(Event(listOfPosLabel.index[indexBegin],
                           listOfPosLabel.index[-1]))
    i = 0
    eventList = [evt for evt in eventList
                 if evt.duration > datetime.timedelta(0)]
    while i < len(eventList) - 1:
        if (eventList[i+1].begin-eventList[i].end
                < datetime.timedelta(hours=thres)):
            eventList[i] = merge(eventList[i], eventList[i+1])
            eventList.remove(eventList[i+1])
        else:
            i += 1
    return eventList


score_types = [
    # log-loss
    rw.score_types.NegativeLogLikelihood(name='pw_ll'),
    # point-wise (for each time step) precision and recall
    PointWisePrecision(),
    PointWiseRecall(),
    # event-based precision and recall
    EventWisePrecision(),
    EventWiseRecall()
]


# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------


def get_cv(X, y):
    # using 5 folds as default
    k = 5
    # up to 10 fold cross-validation based on 5 splits, using two parts for
    # testing in each fold
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
    for ps in pattern[:k]:
        yield (np.hstack([splits[p][1] for p in ps[0]]),
               np.hstack([splits[p][1] for p in ps[1]]))


# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


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
