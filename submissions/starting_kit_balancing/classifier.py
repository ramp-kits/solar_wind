from __future__ import division

from collections import Counter

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier


def determine_ratio(y):
    """
    Balancing the backround class by subsampling it, so it is 5 times more
    prevalent than the target class (instead of 100 times  in the data).
    """
    target_stats = Counter(y)
    max_ = max(target_stats.values())
    ratio = min(target_stats.values()) / max(target_stats.values()) * 2
    ratio = min(ratio, 1.)

    return {key: int(value * ratio)
            for key, value in target_stats.items() if value == max_}


class Classifier(BaseEstimator):
    def __init__(self):
        # mimicking balanced random forest with the BalancedBaggingClassifier
        # and DecisionTreeClassifier combination
        self.bbc = BalancedBaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_features='auto'),
            ratio=determine_ratio, random_state=0, n_estimators=50, n_jobs=1)

    def fit(self, X, y):
        self.bbc.fit(X, y)

    def predict_proba(self, X):
        return self.bbc.predict_proba(X)
