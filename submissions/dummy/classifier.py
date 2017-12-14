import numpy as np
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        # just return majority class (class 0)
        y_pred = np.zeros((len(X), 3))
        y_pred[:, 0] = 1
        return y_pred
