from abc import abstractmethod

import numpy as np
from scipy import signal as sig
from sklearn.base import TransformerMixin, _OneToOneFeatureMixin

from ....datamodels.datamodels.wrappers.feature_extension.store_interface import StoreInterface


class OffsetComp(TransformerMixin, StoreInterface):
    offset = None

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, **fit_params):
        self.offset = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.offset

    def inverse_transform(self, X):
        return X + self.offset


class NaNComp(TransformerMixin, StoreInterface):
    mask_nan = None

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, **fit_params):
        self.mask_nan = np.isnan(X)
        return self

    def transform(self, X):
        return np.nan_to_num(X)

    def inverse_transform(self, X):
        X_tr = X
        X_tr[self.mask_nan is True] = np.nan
        return X_tr


class Filter(TransformerMixin, StoreInterface):
    """
    Signal filter - based on sklearn TransformerMixin. Can be stored to pickle file (StoreInterface).
    Options:
        - keep_nans: Filtered signal still keeps NaN values from original signals
        - remove_offset: Remove offset from signal before filtering, apply offset afterwards
    """
    keep_nans = False
    remove_offset = False
    offset_comp = None
    nan_comp = None
    coef_ = [[0], [0]]
    features_to_filter=None

    def __init__(self, remove_offset=False, keep_nans=False, features_to_filter=None, **kwargs):
        self._set_attrs(remove_offset=remove_offset, keep_nans=keep_nans)
        self.offset_comp = OffsetComp()
        self.nan_comp = NaNComp()
        self.features_to_filter = features_to_filter

    def fit(self, X, y=None, **fit_params):
        self.coef_ = self._fit(X, y, **fit_params)
        return self

    def transform(self, X):
        """
        Filter signal
        @param x: Input feature vector (n_samples, n_features)
        """
        # Select features to filter
        X_to_filter = X[:, self.features_to_filter] if self.features_to_filter is not None else X
        # Remove offset
        if self.remove_offset:
            X_to_filter = self.offset_comp.fit_transform(X_to_filter)
        # Keep NaN values
        if self.keep_nans:
            X_to_filter = self.nan_comp.fit_transform(X_to_filter)
        # Transform features
        x_filt = self._transform(X_to_filter)
        # Apply NaNs
        if self.keep_nans:
            x_filt = self.nan_comp.inverse_transform(x_filt)
        # Add offset
        if self.remove_offset:
            x_filt = self.offset_comp.inverse_transform(x_filt)
        # Add non-selected features
        if self.features_to_filter is not None:
            x_filt_new = X
            x_filt_new[:, self.features_to_filter] = x_filt
            return x_filt_new
        return x_filt

    def _transform(self, X):
        """
        Filter signal. Override if necessary.
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        return sig.lfilter(*self.coef_, X, axis=0)

    def get_coef(self):
        """
        Get filter coefficients.
        """
        return self.coef_

    @abstractmethod
    def _fit(self, X, y=None, **fit_params):
        """
        Override this method to create filter coeffs.
        """
        raise NotImplementedError

    def get_feature_names_out(self, feature_names=None):
        """
        Get feature names - passthrough for pipeline
        @param feature_names: Input feature names
        @return: input feature names
        """
        return feature_names