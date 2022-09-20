#%%
import ModelTraining.Training.TrainingUtilities.training_utils as train_utils
from ModelTraining.Training.TrainingUtilities.trainingparams_expanded import TrainingParamsExpanded
from ModelTraining.Utilities import TrainingData
from ModelTraining.feature_engineering.featureengineering.featureselectors import FeatureSelector
import ModelTraining.Training.TrainingUtilities.training_utils_expanded as train_utils_exp
from ModelTraining.Utilities.MetricsExport import MetricsCalc, ResultExport, metr_utils
from ModelTraining.Training.TrainingUtilities.training_utils import load_from_json
from ModelTraining.Data.DataImport.featureset.featureset import FeatureSet
from ModelTraining.datamodels.datamodels.wrappers.expandedmodel import ExpandedModel, TransformerSet
from ModelTraining.feature_engineering.featureengineering.featureexpanders import FeatureExpansion
import os
import argparse
from ModelTraining.datamodels.datamodels.processing import datascaler
from ModelTraining.datamodels import datamodels
from sklearn.model_selection import GridSearchCV

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--usecase_names", type=str, default='Solarhouse2')
    parser.add_argument("--model_types", type=str, default='RandomForestRegression')
    args = parser.parse_args()
    model_types = model_names = args.model_types.split(",")
    list_usecases = args.usecase_names.split(",")
    data_dir_path = "./Data/"
    root_dir = "./"
    plot_enabled = False

    # basic training params
    config_path = os.path.join(root_dir, 'Configuration')
    # Model parameters and expansion parameters
    params_dir =os.path.join(root_dir, 'Configuration/GridSearchParameters')
    parameters_full = {model_type: load_from_json(os.path.join(params_dir, f'parameters_{model_type}.json')) for model_type in model_types}
    parameters_thresh = {"rthreshold__thresh": [0.001, 0.01, 0.02, 0.05, 0.1,0.2]}

    parameters_thresh = {"rthreshold__thresh":[0.05]}
    parameters_full = {"RandomForestRegression":{"criterion":["squared_error"]}}

    selected_thresh = "rdc"
    transf_cfg_files = [f"train_params_{selected_thresh}_0_05_{expansion_type}_r_0_05.json" for expansion_type in
                        ['basic', 'poly']]
    list_train_params = [TrainingParamsExpanded.load(os.path.join(root_dir, "Configuration","TrainingParameters", file)) for file in
                               transf_cfg_files]
    params_names = [f'{selected_thresh}-value_0.05_R-value_0.05']
    # Use cases
    dict_usecases = [load_from_json(os.path.join(config_path,"UseCaseConfig", f"{name}.json")) for name in
                     list_usecases]
    # Results output
    timestamp = metr_utils.create_file_name_timestamp()
    results_path = os.path.join(root_dir, 'results', timestamp)
    os.makedirs(results_path, exist_ok=True)
    metrics_path = os.path.join(root_dir, 'results', timestamp, 'Metrics')
    os.makedirs(metrics_path, exist_ok=True)

    metrics_names = {'FeatureSelect': ['selected_features', 'all_features'], 'Metrics': ['R2_SKLEARN', 'CV-RMS', 'MAPE', 'RA_SKLEARN'], 'pvalues': ['pvalue_lm', 'pvalue_f']}
    gridsearch_scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']


#%%
    # Main loop
    print('Starting Training')
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        results_path_dataset = os.path.join(results_path, usecase_name)
        # Get data and feature set
        dataimport_cfg_path = os.path.join(data_dir_path, "Configuration", "DataImport")
        data = train_utils_exp.import_data(dataimport_cfg_path, data_dir_path, dict_usecase)
        if usecase_name == "Solarhouse2":
            data["Vd_Solar_inv"] = 0
            data["Vd_Solar_inv"][data["Vd_Solar"] != 0] = (1 / data["Vd_Solar"][data["Vd_Solar"] != 0])

        feature_set = FeatureSet(os.path.join(root_dir, "Data", "Configuration", "FeatureSet", dict_usecase['fmu_interface']))
        # Main loop
        for params_name in params_names:
            os.makedirs(os.path.join(results_path, dict_usecase['name'], params_name), exist_ok=True)
            results_path_thresh = os.path.join(results_path_dataset, params_name)
            for training_params_cfg in list_train_params:
                if isinstance(training_params_cfg, TrainingParamsExpanded):
                    train_utils_exp.set_train_params_transformers(training_params_cfg, dict_usecase)
                for model_type in model_types:
                    for feature in feature_set.get_output_feature_names():
                        training_params = train_utils.set_train_params_model(training_params_cfg, feature_set, feature, model_type)
                        create_df = True if isinstance(training_params, TrainingParamsExpanded) else False
                        index, x, y, feature_names = train_utils.extract_training_and_test_set(data, training_params,
                                                                                               create_df=create_df)
                        train_data = train_utils.create_train_data(index, x, y,
                                                                   training_split=training_params.training_split)
                        index_train, x_train, y_train = train_data.train_index, train_data.train_input, train_data.train_target
                        index_test, x_test, y_test = train_data.test_index, train_data.test_input, train_data.test_target
                        model = getattr(datamodels, training_params.model_type)(
                            x_scaler_class=getattr(datascaler, training_params.normalizer),
                            name=training_params.str_target_feats(),
                            parameters={})
                        # Create expanded model wrapper
                        if isinstance(training_params, TrainingParamsExpanded):
                            model = ExpandedModel(
                                transformers=TransformerSet.from_list_params(training_params.transformer_params),
                                model=model, feature_names=feature_names)
                        # Select features + Grid Search
                        estimator_name = model.get_estimator().__class__.__name__.lower()
                        gridsearch_params = {f"{estimator_name}__{key}":vals for key, vals in parameters_full[model_type].items()}
                        gridsearch_params.update(parameters_thresh)
                       # print(gridsearch_params)
                        search = GridSearchCV(model.get_full_pipeline(), gridsearch_params, cv=3, scoring=gridsearch_scoring,
                                              refit='r2', verbose=4)
                        # Transform x train
                        search.fit(*model.scale(x_train, y_train))
                        print(search.best_params_)
                        for k, val in search.best_params_.items():
                           if k.split("__")[0] == estimator_name:
                               setattr(model.get_estimator(), k.split("__")[1], val)
                           else:
                               model.transformers.set_transformer_attributes(k.split("__")[0], {k.split("__")[1]: val})

                        # Train final model
                        model.train(x_train, y_train)
                        train_data.test_prediction = model.predict(x_test)
                        # Save models
                        model_dir = os.path.join(f"{training_params.model_name}",f"{training_params.model_type}_{training_params.str_expansion(range=[2,-1])}")
                        result_main_dir = os.path.join(results_path_thresh, "Models", model_dir)
                        train_utils.save_model_and_params(model, training_params, result_main_dir)
                        train_data.save_pkl(results_path_thresh, f'results_{model_type}_{training_params.str_target_feats()}_{training_params.str_expansion(range=[2,-1])}.pkl')
    print('Experiments finished')

# %%
    print('Analyzing results')
    metr_exp = MetricsCalc(metr_names=metrics_names)
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        feature_set = FeatureSet(
            os.path.join(root_dir, "Data", "Configuration", "FeatureSet", dict_usecase['fmu_interface']))
        for params_name in params_names:
            result_exp = ResultExport(results_root=os.path.join(results_path, usecase_name, params_name),
                                      plot_enabled=True)
            for training_params in list_train_params:
                for model_type in model_types:
                    for feat in feature_set.get_output_feature_names():
                        # Load results
                        result = TrainingData.load_pkl(result_exp.results_root,
                                                          f'results_{model_type}_{feat}_{training_params.str_expansion(range=[2,-1])}.pkl') # TODO fix this
                        model_dir = os.path.join(result_exp.results_root,
                                                 f'Models/{feat}/{model_type}_{training_params.str_expansion(range=[2,-1])}/{feat}')
                        model = ExpandedModel.load_pkl(model_dir, "expanded_model.pkl")
                        # Export model properties
                        result_exp.export_result_full(model, result, training_params.str_expansion(range=[2,-1]))
                        # Calculate metrics
                        selectors = model.transformers.get_transformers_of_type(FeatureSelector)
                        metr_vals = metr_exp.calc_all_metrics(result, selectors, model.get_num_predictors())
                        # Set metrics identifiers
                        for metr_val in metr_vals:
                            metr_val.set_metr_properties(model_type, model.name, model.transformers.type_last_transf(FeatureExpansion),
                                                         params_name, usecase_name)
                        metr_exp.add_metr_vals(metr_vals)
    metr_exp.store_all_metrics(results_path=metrics_path, timestamp=timestamp)
    print('Result analysis finished')