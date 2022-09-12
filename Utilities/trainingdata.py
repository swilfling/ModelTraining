from dataclasses import dataclass
from ..feature_engineering.featureengineering.interfaces.storage import PickleInterface
import numpy as np
import os
import pandas as pd
from typing import List

@dataclass
class TrainingData(PickleInterface):
    """
    This class contains the results for training and test set.
    """
    train_index: np.ndarray = None
    train_target: np.ndarray = None
    train_input: np.ndarray = None
    train_prediction: np.ndarray = None
    test_index: np.ndarray = None
    test_target: np.ndarray = None
    test_prediction: np.ndarray = None
    test_input: np.ndarray = None
    target_feat_names: List[str] = None

    def test_results_to_csv(self, dir, filename="test_results.csv", index_label='t'):
        """
        Store test results in csv file
        @param dir: output dir
        @param filename: output filename
        @param index_label: index label
        """
        self.test_result_df().to_csv(os.path.join(dir, filename), index_label=index_label)

    def train_results_to_csv(self, dir, filename="train_results.csv", index_label='t'):
        """
        Store train results in csv file
        @param dir: output dir
        @param filename: output filename
        @param index_label: index label
        """
        df = pd.DataFrame(index=self.train_index, data=np.hstack((self.train_target, self.train_prediction)), columns=['GroundTruth', 'Prediction'])
        df.to_csv(os.path.join(dir, filename), index_label=index_label)

    def test_result_df(self, feat=None, col_names=[]):
        """
        Create dataframe from test results
        @param feat: optional: get only results for this target feature
        @param col_names: optional: define column names for dataframe
        @return: dataframe containing results
        """
        if self.test_target.ndim == 1:
            self.test_target = np.expand_dims(self.test_target, axis=1)
        if self.test_target.ndim == 3:
            self.test_target = np.reshape(self.test_target, (self.test_target.shape[0], self.test_target.shape[1] * self.test_target.shape[2]))

        if self.test_prediction is None:
            self.test_prediction = np.zeros(self.test_target.shape)

        self.test_prediction = np.reshape(self.test_prediction, self.test_target.shape)

        if feat is None:
            return pd.DataFrame(index=self.test_index, data=np.hstack((self.test_target, self.test_prediction)),
                                columns=self._get_df_cols() if col_names == [] else col_names)
        else:
            feat_ind = self.target_feat_names.index(feat)
            return pd.DataFrame(index=self.test_index, data=np.vstack((self.test_target[:,feat_ind], self.test_prediction[:,feat_ind])).T,
                                columns=[f'GroundTruth_{feat}', f'Prediction_{feat}'])

    def test_target_vals(self, feat=None):
        """
        Get target vals for feature
        @param feat: optional: get only results for this target feature
        @return: target vals for feature, if no feature name is passed, returns all target vals
        """
        test_target = self.test_target.values if isinstance(self.test_target, pd.DataFrame) else self.test_target
        if feat is None:
            return test_target
        if isinstance(feat, str):
            return test_target[:, self.target_feat_names.index(feat)].reshape(test_target.shape[0],1)
        if isinstance(feat, list):
            return np.concatenate([test_target[:, self.target_feat_names.index(name)].reshape(test_target.shape[0],1) for name in feat], axis=1)

    def test_pred_vals(self, feat=None):
        """
        Get prediction vals for feature
        @param feat: optional: get only results for this target feature
        @return: prediction vals for feature, if no feature name is passed, returns all prediction vals
        """
        test_pred = self.test_prediction.values if isinstance(self.test_prediction, pd.DataFrame) else self.test_prediction
        if feat is None:
            return test_pred
        if isinstance(feat, str):
            return test_pred[:, self.target_feat_names.index(feat)].reshape(self.test_target.shape[0],1)
        if isinstance(feat, list):
            return np.concatenate([test_pred[:, self.target_feat_names.index(name)].reshape(self.test_target.shape[0],1) for name in feat], axis=1)

    def _get_df_cols(self):
        """
        Helper function - get columns for dataframe
        """
        if self.target_feat_names is None:
            num_target_feats = self.test_target.shape[1]
            return [f'GroundTruth_{i}' for i in range(num_target_feats)] + [f'Prediction_{i}' for i in range(num_target_feats)]
        else:
            return [f'GroundTruth_{feat}' for feat in self.target_feat_names] + [f'Prediction_{feat}' for feat in self.target_feat_names]

    def get_feat_ind(self):
        """
        Helper function - get indices for feats
        """
        if self.target_feat_names is None:
            return range(int(len(self._get_df_cols()) / 2))
        else:
            return self.target_feat_names