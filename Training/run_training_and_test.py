import logging
import os
from typing import List

import pandas as pd
from sklearn.model_selection import GridSearchCV
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.Training.predict import predict_with_history, predict_gt
from ModelTraining.Training.ModelCreation import create_model
from ModelTraining.Training.GridSearch import prepare_data_for_fit, create_pipeline
from ModelTraining.Preprocessing.FeatureSelection.FeatureSelector import FeatureSelector
import ModelTraining.Preprocessing.FeatureSelection.feature_selection as feat_select
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults
from ModelTraining.Preprocessing.FeatureSelection.feature_selection_params import FeatureSelectionParams
import ModelTraining.Preprocessing.noise_generation as noise_gen


def run_training_and_test(data, list_training_parameters: List[TrainingParams],
                          results_dir_path, prediction_type="History", **kwargs):
    models, results, list_selectors = [], [], []
    # Get optional arguments
    model_parameters = kwargs.get('model_parameters', {})
    expander_parameters = kwargs.get('expander_parameters', {})

    for training_params in list_training_parameters:
        target_features = training_params.target_features

        # Get data and reshape
        index, x, y, feature_names = train_utils.extract_training_and_test_set(data, training_params)
        index_train, x_train, y_train, index_test, x_test, y_test = train_utils.split_into_training_and_test_set(
            index, x, y, training_params.training_split)

        # Create model
        logging.info(f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}")
        model = create_model(training_params, expander_parameters=expander_parameters, feature_names=feature_names)

        # Select features + Grid Search
        search = GridSearchCV(model.model, model_parameters,
                              scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], refit='r2',
                              verbose=4)
        selectors = [FeatureSelector.from_params(params) for params in
                     kwargs.get('feature_select_params', [FeatureSelectionParams()])]
        pipeline = create_pipeline(model.expanders, selectors, search)
        pipeline.fit(*prepare_data_for_fit(model, x_train, y_train))
        # Set grid search params
        print(f"Best score for model {model.__class__.__name__} is: {search.best_score_}")
        print(f"Best parameters are {search.best_params_}")
        # Configure model
        model.model.set_params(**search.best_params_)
        feat_select.configure_feature_select(model.expanders, selectors)
        list_selectors.append(selectors)
        # Train model
        model.train(x_train, y_train)
        models.append(model)
        # Save Model
        train_utils.save_model_and_parameters(os.path.join(results_dir_path,
                                                           f"Models/{training_params.model_name}/{training_params.model_type}_{training_params.expansion[-1]}"),
                                              model, training_params)
        # Predict test data
        predict_function = predict_with_history if prediction_type == 'History' else predict_gt
        result_prediction = predict_function(model, index_test, x_test, y_test, training_params)
        test_prediction = result_prediction[[f"predicted_{feature}" for feature in target_features]].to_numpy()
        results.append(TrainingResults(train_index=index_train, train_target=y_train,
                                       test_index=index_test, test_target=y_test, test_prediction=test_prediction,
                                       test_input=x_test))

    return models, [test_prediction, list_selectors]


def train_predict(data, training_params, num_training_sample, test_starting_idx, expander_parameters, model_parameters,
          target_features, **kwargs):
    index, x, y, feature_names = train_utils.extract_training_and_test_set(data, training_params)

    index_train, x_train, y_train, index_test, x_test, y_test = train_utils.split_training_and_test_specified(
        index, x, y, training_params.training_split, test_starting_idx)

    # Create model
    model = create_model(training_params, expander_parameters=expander_parameters, feature_names=feature_names)

    # Select features + Grid Search
    search = GridSearchCV(model.model, model_parameters,
                          scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], refit='r2',
                          verbose=4)
    selectors = [FeatureSelector.from_params(params) for params in
                 kwargs.get('feature_select_params', [FeatureSelectionParams()])]
    pipeline = create_pipeline(model.expanders, selectors, search)
    pipeline.fit(*prepare_data_for_fit(model, x_train, y_train))
    # Set grid search params
    print(f"Best score for model {model.__class__.__name__} is: {search.best_score_}")
    print(f"Best parameters are {search.best_params_}")
    models, results, list_selectors = [], [], []

    # Configure model
    model.model.set_params(**search.best_params_)
    feat_select.configure_feature_select(model.expanders, selectors)
    list_selectors.append(selectors)
    # Train model
    model.train(x_train, y_train)
    models.append(model)

    # Predict test data
    result_prediction = predict_gt(model, index_test, x_test, y_test, training_params)
    test_prediction = result_prediction[[f"predicted_{feature}" for feature in target_features]].to_numpy()
    results.append(TrainingResults(train_index=index_train, train_target=y_train,
                                   test_index=index_test, test_target=y_test, test_prediction=test_prediction,
                                   test_input=x_test))

    return models, result_prediction, selectors, y_test


def run_noise_test(data, list_training_parameters: List[TrainingParams],
                   results_dir_path,expansion, **kwargs):
    models, results, list_selectors = [], [], []
    test_prediction_df = pd.DataFrame()
    # Get optional arguments
    model_parameters = kwargs.get('model_parameters', {})
    expander_parameters = kwargs.get('expander_parameters', {})
    training_split = kwargs.get('training_split', 0.7)
    test_starting_idx = kwargs.get('test_starting_idx', "2019-07-02 17:00:00")
    noise = noise_gen.change_range(data["temperature"])
    for training_params in list_training_parameters:
        target_features = training_params.target_features

        _, clean_preds, _, y_true = train_predict(data, training_params, training_split, test_starting_idx,
                                          expander_parameters, model_parameters, target_features)
        for i in range(len(noise)):
            data["temperature"] = noise[i]

            models, preds, _, y_true = train_predict(data, training_params, training_split, test_starting_idx,
                                              expander_parameters, model_parameters, target_features)
            test_prediction_df[f'result_prediction_{i}'] = preds["predicted_energy"]

        noise_gen.plot_robust_interval(test_prediction_df, clean_preds,  model_type=training_params.model_type, expansion_type=expansion,result_path=results_dir_path, y_true=y_true)