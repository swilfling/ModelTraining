import numpy as np
from sklearn.base import TransformerMixin
from ...interfaces import PickleInterface

class OffsetComp(TransformerMixin, PickleInterface):
    offset = None
    remove_offset = True

    def __init__(self, remove_offset=True, **kwargs):
        self.remove_offset = remove_offset

    def fit(self, X, y=None, **fit_params):
        self.offset = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        if self.remove_offset:
            return X - self.offset
        return X

    def inverse_transform(self, X):
        if self.remove_offset:
            return X + self.offset
        return X