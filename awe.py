# @Author: Jenny Hsiao
# @Date:   2018-12-20T14:24:23+08:00
# @Email:  jenny.hsiao@asmpt.com
# @Filename: awe.py
# @Last modified by:   Jenny Hsiao
# @Last modified time: 2018-12-20T14:53:51+08:00

import copy as cp
from sortedcontainers import SortedDict
import numpy as np
from sklearn.tree import DecisionTreeClassifier

rng = np.random.RandomState(42)

def mse_prob(y_pred_prob, y_true):
    err = [1-p[c]for p, c in zip(y_pred_prob, y_true)]
    return np.square(err).sum()/len(err)

def get_class_distribution(Y):
    all_p = 0
    n = len(Y)
    for c in set(Y):
        n_c = len(Y[Y==c])
        p = (n_c/n)
        all_p = all_p + (p * (1-p)**2)
    return all_p


class AWE():
    """
        Implement Mining Concept-Drifting Data Streams using Ensemble Classifiers
        Accuracy-weighted ensemble classifier

        Parameters
        ----------
        base_estimator: StreamModel or sklearn model
            This is the ensemble learner type, each ensemble model is a copy of
            this one.

        window_size (int)
            The size of the training window (batch), in other words, how many instances are kept for training.

        n_estimators (int)
            Number of estimators in the ensemble.

    """

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=10, window_size=100):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        # The ensemble
        self.ensemble = []
        self.weights = []
        self.i = -1
        self.X_batch = None
        self.y_batch = None

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
        N, D = X.shape
        X_batch = X
        y_batch = y
        # if self.i < 0:
        #     # No models yet -- initialize
        #     self.X_batch = np.zeros((self.window_size, D))
        #     self.y_batch = np.zeros(self.window_size, dtype='int')
        #     self.i = 0

        # for n in range(N):
        #     # For each instance ...
        #     # TODO not very pythonic at the moment
        #     self.X_batch[self.i] = X[n]
        #     self.y_batch[self.i] = y[n]
        #     self.i = self.i + 1
        #     if self.i == self.window_size:
        # A new model
        h = cp.deepcopy(self.base_estimator)
        # Train it
        h.fit(X=X_batch, y=y_batch)
        # calculate the accuracy weight
        y_pred_prob = h.predict_proba(X_batch)
        mse_i = mse_prob(y_pred_prob, y_batch)
        mse_r = get_class_distribution(y_batch)
        w = max(mse_r - mse_i, 0.0001)

        # Add it
        self.ensemble.append(h)
        self.weights.append(w)

        # Get rid of the smallest weighted model
        if len(self.ensemble) > self.n_estimators:
            idx = np.argmin(self.weights)
            self.ensemble.pop(idx)
            self.weights.pop(idx)

        # Reset the window
        self.i = 0


        return self

    def predict_proba(self, X):
        N, _ = X.shape
        votes = np.zeros(N)
        if len(self.ensemble) <= 0:
            # No models yet, just predict zeros
            raise "data is not sufficient to train a model."

        # weighted probability
        w_sum = sum(self.weights)
        weighted_prob = [ (h_i.predict_proba(X)*w/w_sum) for w, h_i in zip(self.weights, self.ensemble)]
        votes = np.sum(weighted_prob, axis=0)


        return votes

    def predict(self, X):
        weighted_prob = self.predict_proba(X)
        weighted_prob = np.reshape(weighted_prob, (len(weighted_prob), -1))
        # majority voting
        votes = np.argmax(weighted_prob, axis=1)
        return votes

    def score(self, X, y):
        raise NotImplementedError

    def reset(self):
        self.ensemble = []
        self.weights = []
        self.i = -1
        self.X_batch = None
        self.y_batch = None

    def get_info(self):
        return 'Accuracy Weighted Ensemble Classifier:' \
               ' - base_estimator: {}'.format(type(self.base_estimator).__name__) + \
               ' - window_size: {}'.format(self.window_size) + \
               ' - n_estimators: {}'.format(self.n_estimators)
