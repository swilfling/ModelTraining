from pathlib import Path
import pandas as pd
from .training_utils import save_model_and_params
from ...Training.TrainingUtilities.trainingparams_expanded import TrainingParamsExpanded
from ...feature_engineering.featureengineering.featureexpanders import PolynomialExpansion
from ...Data.DataImport.dataimport import DataImport
from ...Utilities.MetricsExport import ResultExport, MetricsVal, MetricsCalc
from ...feature_engineering.featureengineering.featureselectors import FeatureSelector
from ...datamodels.datamodels.wrappers.expandedmodel import TransformerParams



########################### Data import #################################################


def import_data(cfg_path: str, data_dir:str, dict_usecase):
    """
    Import data using configuration
    :param cfg_path: Path to configuration files for data import
    :param data_dir: path to datasets
    :param dict_usecase: dict containing information about dataset directory and filename
    :return: data
    """
    dataimport_cfg_path = Path(cfg_path) / dict_usecase['dataset_dir'] / f"{dict_usecase['dataset_filename']}.json"
    data_import = DataImport.load(str(dataimport_cfg_path))
    use_processed_file = dict_usecase.get('use_processed',False)
    filename = dict_usecase['dataset_filename'] if not use_processed_file else f"{dict_usecase['dataset_filename']}_proc"
    data = data_import.import_data(str(Path(data_dir) / "Data" / dict_usecase['dataset_dir'] / filename))
    return data


def set_train_params_transformers(training_params: TrainingParamsExpanded, dict_usecase):
    """
    Set training parameters for transformers based on use case
    :param training_params:
    :param dict_usecase:
    :return: modified training params
    """
    tr_params = training_params.transformer_params
    for cfg in TransformerParams.get_params_of_type(tr_params, "CategoricalFeatures"):
        cfg.params['selected_feats'] = dict_usecase.get('cyclical_feats', [])
    for cfg in TransformerParams.get_params_of_type(tr_params, "CyclicFeatures"):
        cfg.params['selected_feats'] = dict_usecase.get('onehot_feats', [])

    for cfg in TransformerParams.get_params_of_type(tr_params, 'Transformer_MaskFeats', 'transformer_params',
                                                         'DynamicFeatures'):
        cfg.params['mask_params']['features_to_transform'] = training_params.dynamic_input_features
    for cfg in TransformerParams.get_params_of_type(tr_params, 'Transformer_MaskFeats', 'transformer_type',
                                                          'StatisticalFeaturesNumpy'):
        cfg.params['mask_params']['features_to_transform'] = dict_usecase['stat_feats']

    for cfg in TransformerParams.get_params_of_type(tr_params, 'Transformer_MaskFeats', 'InverseTransform'):
        cfg.params['mask_params']['features_to_transform'] = dict_usecase.get('to_invert', [])


def expand_features(data: pd.DataFrame, feature_names, expander_parameters={}):
    """
    Expand features through polynomial expansion
    @param data: input data
    @param feature_names: names of features to expand
    @param expander_parameters: Parameters for polynomial expansion
    @return: dataframe containing expanded data
    """
    expander = PolynomialExpansion(**expander_parameters)
    data_expanded = expander.fit_transform(data[feature_names])
    feature_names_expanded = expander.get_feature_names_out(feature_names)
    return pd.DataFrame(data_expanded, columns=feature_names_expanded)

"""
Replace the ground truth by the predicted values for all the target features in the dataset and save the new one
Assuming that each csv file contains values of one column (each model predicts one target feature without replicas)
"""

def replace_dataset(data, list_training_parameters, first_train_results_path, lookback_horizon):
    new_dataset = data[lookback_horizon + 1:]

    """
    Predicted dataset starts from (lookback horizon + 1)
    """
    for training_parameters in list_training_parameters:

        for feature in training_parameters.target_features:
            val = pd.read_csv(f"{first_train_results_path}/{training_parameters.model_type}_{feature}_predictions.csv", sep=',', engine='python').set_index('date')
            new_dataset = new_dataset.drop(feature, axis=1)
            new_dataset[feature] = val['predicted']

    return new_dataset


def store_results(model,train_data, training_params, metr_exp: MetricsCalc, best_params, result_dir_model, result_id, experiment_name, usecase_name):
    model_dir = Path(result_dir_model) / "Models"/f"{training_params.model_name}"/f"{training_params.model_type}"
    save_model_and_params(model, training_params, str(model_dir))
    # Calculate and export metrics
    train_data.save_pkl(result_dir_model, "result_data.pkl")
    # Calc metrics
    for k, val in best_params.items():
            metr_exp.add_metr_val(MetricsVal(model_type=model.model.model_type,
                                             model_name=model.name,
                                             featsel_thresh=k.split("__")[0],
                                             expansion_type=result_id,
                                             metrics_type="best_params", metrics_name=k, val=val,
                                             usecase_name=usecase_name))

    metr_vals = metr_exp.calc_all_metrics(train_data, model.transformers.get_transformers_of_type(FeatureSelector))
    # metr_vals = metr_exp.calc_perf_metrics(train_data, model.get_num_predictors())
    # Set metrics identifiers
    transformer_name = model.transformers.get_transformer_by_index(-1).__class__.__name__
    for metr_val in metr_vals:
        metr_val.set_metr_properties(model.model.model_type, model.name, result_id, transformer_name, usecase_name)
    # Store metrics separately
    metr_exp.add_metr_vals(metr_vals)
    for type in ['Metrics', 'pvalues', 'FeatureSelect']:
        filename = Path(result_dir_model)/ f"{type}_{experiment_name}.csv"
        metr_exp.store_metr_df(metr_exp.get_metr_df(type), str(filename))
    # Export results
    exp = ResultExport(results_root=result_dir_model, plot_enabled=True)
    exp.export_result(train_data, result_id, show_fig=False)
    exp.plot_enabled = False
    exp.export_model_properties(model, result_id)
    exp.export_featsel_metrs(model, result_id)


