from dataclasses import dataclass

import numpy as np
import pandas as pd

from .featurecreator import FeatureCreator


@dataclass
class CyclicFeatureInfo:
    name: str = "daytime"
    T: float = 3600


class CyclicFeatures(FeatureCreator):
    """
    Create cyclic encoding for features.
    Currently supported: daytime, weekday, month, day
    """
    all_time_vals = {"hour":CyclicFeatureInfo("hour", 24),
                     "weekday":CyclicFeatureInfo("weekday", 7),
                     "month":CyclicFeatureInfo("month", 12),
                     "day":CyclicFeatureInfo("day", 30)}

    def __init__(self, selected_feats=["hour", "weekday"], **kwargs):
        super().__init__(selected_feats=selected_feats, **kwargs)
        pass

    def transform(self, X: pd.DataFrame):
        """
        Add cyclic features
        @param X: input data
        @return: transformed data
        """
        X_t = X.copy()
        # Add labels to data
        for label in self.selected_feats:
            cycl_info = self.all_time_vals[label]
            X[label] = getattr(X.index, cycl_info.name, [])
            X_t[f'{label}_sin'], X_t[f'{label}_cos'] = self.calc_sin_cos(X[label], cycl_info.T)
        return X_t

    def calc_sin_cos(self, time, T:float=1):
         return np.sin(time * 2 * np.pi / T), np.cos(time * 2 * np.pi / T)

    def get_additional_feat_names(self):
        """
        Get names of cyclic features
        @return: list of feature names
        """
        list_feat_names = []
        for label in self.selected_feats:
            list_feat_names += [f"{label}_sin", f"{label}_cos"]
        return list_feat_names


class CyclicFeaturesSampleTime(CyclicFeatures):
    """
    Create cyclic encoding for features based on sampling time.
    Currently supported: daytime, weekday, month, day
    """
    sample_time = 3600
    all_time_vals = {"daytime":CyclicFeatureInfo("hour", 3600 * 24),
                     "weekday":CyclicFeatureInfo("weekday", 3600 * 24 * 7),
                     "month":CyclicFeatureInfo("month", 3600 * 24 * 365),
                     "day":CyclicFeatureInfo("day", 3600 * 24 * 30)}

    def __init__(self, sample_time=3600, selected_feats=["hour", "weekday"], **kwargs):
        super().__init__(selected_feats=selected_feats, **kwargs)
        self.sample_time = sample_time
        pass

    def transform(self, X: pd.DataFrame):
        """
        Add cyclic features
        @param X: input data
        @return: transformed data
        """
        X_t = X.copy()
        # Add labels to data
        for label in self.selected_feats:
            cycl_info = self.all_time_vals[label]
            X_t[label] = np.array([(val - X.index[0]).total_seconds() for val in X.index]) / self.sample_time
            X_t[f'{label}_sin'], X_t[f'{label}_cos'] = self.calc_sin_cos(X_t[label], cycl_info.T / self.sample_time)
        return X_t