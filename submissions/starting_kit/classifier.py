from __future__ import division

from collections import Counter

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier


def determine_ratio(y):
    target_stats = Counter(y)
    max_ = max(target_stats.values())
    ratio = min(target_stats.values()) / max(target_stats.values()) * 5

    return {key: int(value * ratio)
            for key, value in target_stats.items() if value == max_}


class Classifier(BaseEstimator):
    def __init__(self):
        self.bbc = BalancedBaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_features='auto'),
            ratio=determine_ratio, random_state=0)

    def fit(self, X, y):
        self.bbc.fit(X, y)

    def predict_proba(self, X):
        return self.bbc.predict_proba(X)
