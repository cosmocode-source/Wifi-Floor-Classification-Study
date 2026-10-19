"""Exact PCA–kNN with deterministic neighbours and equal-weight voting."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from Cleaned_Dataset.clean_data import SignalEncoder, FLOORS

PCA_COMPONENTS = 200

def neighbour_labels(X_train, y_train, X_test, limit, batch_size=128):
    """Exact Euclidean neighbours, with equal-distance ties in training-row order.

    Stable ordering makes the k-search and final prediction use the same policy.
    Batching bounds memory; no approximate-neighbour algorithm is used.
    """
    limit = min(int(limit), len(X_train))
    output = np.empty((len(X_test), limit), dtype=np.int8)
    for start in range(0, len(X_test), batch_size):
        stop = min(start + batch_size, len(X_test))
        squared = euclidean_distances(X_test[start:stop], X_train, squared=True)
        nearest = np.argsort(squared, axis=1, kind='stable')[:, :limit]
        output[start:stop] = np.asarray(y_train, dtype=np.int8)[nearest]
    return output


def predict_all_k(neighbours):
    """Column k-1 is the equal-weight majority prediction for that k.

    A tied class vote chooses the smaller floor label, matching sorted-class
    majority voting. Odd k is not assumed to eliminate multiclass ties.
    """
    votes = np.zeros((len(neighbours), len(FLOORS)), dtype=np.int32)
    predictions = np.empty_like(neighbours)
    rows = np.arange(len(neighbours))
    for index in range(neighbours.shape[1]):
        votes[rows, neighbours[:, index]] += 1
        predictions[:, index] = votes.argmax(axis=1)
    return predictions


class ExactFloorKNN(ClassifierMixin, BaseEstimator):
    """Small exact kNN estimator, compatible with sklearn Pipeline and joblib."""
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float64).copy()
        self.y_ = np.asarray(y, dtype=np.int8).copy()
        self.classes_ = np.unique(self.y_)
        self.n_features_in_ = self.X_.shape[1]
        if not 1 <= self.n_neighbors <= len(self.X_):
            raise ValueError('k must be between 1 and the number of fitted rows.')
        return self

    def predict(self, X):
        neighbours = neighbour_labels(self.X_, self.y_, np.asarray(X), self.n_neighbors)
        return predict_all_k(neighbours)[:, -1].astype(int)


def knn_pipeline(k):
    return Pipeline([
        ('encode', SignalEncoder()),
        ('pca', PCA(n_components=PCA_COMPONENTS, svd_solver='full', whiten=False)),
        ('classifier', ExactFloorKNN(n_neighbors=int(k)))])


