# %%

from ModelTraining.Training.TrainingUtilities.training_utils import load_from_json
from ModelTraining.Data.DataImport.featureset.featureset import FeatureSet
import ModelTraining.Preprocessing.data_preprocessing as dp_utils
from ModelTraining.Data.DataImport.dataimport import DataImport
import os
import numpy as np


if __name__ == '__main__':
    # %%
    root_dir = "../../"
    data_dir = "../../Data/"
    dataimport_config_path = os.path.join(root_dir, "Data", "Configuration", "DataImport")
    # Added: Preprocessing - Smooth features
    config_path = os.path.join(root_dir, 'Configuration')
    list_usecases = ['CPS-Data', 'SensorA6', 'SensorB2', 'SensorC6', 'Solarhouse1', 'Solarhouse2']
    list_usecases += ['Beyond_B20_LR_dyn', 'Beyond_B12_LR_dyn']

    dict_usecases = [load_from_json(os.path.join(config_path, "UseCaseConfig", f"{name}.json")) for name in
                     list_usecases]

    interaction_only = True
    matrix_path = "./Figures/Correlation"
    vif_path = './Figures/Correlation/VIF'
    os.makedirs(vif_path, exist_ok=True)
    float_format = "%.2f"
    expander_parameters = {'degree': 2, 'interaction_only': True, 'include_bias': False}

    # %% correlation matrices
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        # Get data and feature set
        data_import = DataImport.load(
            os.path.join(root_dir, "Data", "Configuration","DataImport", dict_usecase['dataset_dir'], f"{dict_usecase['dataset_filename']}.json"))
        data = data_import.import_data(
            os.path.join(data_dir, "Data", dict_usecase['dataset_dir'], dict_usecase['dataset_filename']))
        feature_set = FeatureSet(
            os.path.join(root_dir, "Data", "Configuration", "FeatureSet", dict_usecase['fmu_interface']))
        # Data preprocessing
        data = dp_utils.preprocess_data(data, dict_usecase['dataset_filename'])

        filename = os.path.join(data_dir, "Data", dict_usecase['dataset_dir'],dict_usecase['dataset_filename'])
        filename_proc = f"{filename}_proc"
        data_import.data_to_file(data, filename_proc)
        data_reimported = data_import.import_data(filename_proc)
        print(f"Checking file: {dict_usecase['dataset_filename']}")
        assert(np.all(data_reimported.columns == data.columns))
        assert(np.all(data_reimported.index == data.index))
        assert(np.all(data_reimported.values == data.values))


