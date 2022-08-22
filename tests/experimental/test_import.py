from ModelTraining.Data.DataImport.dataimport import CSVImport, ExcelImport, HDFImport, DataImport, TXTImport_Octave
import os

if __name__ == "__main__":
#%%
    data_filename_solarhouse1 = "../../Data/Data/AEE/Solarhouse1/Resampled15min"
    solarhouse1_import = CSVImport(freq='15T', sep=";", index_col='Zeitraum')
    #solarhouse1_import.to_file("../Data/Configuration/AEE/Solarhouse1/Resampled15min.json")
    #solarhouse1_import_2 = DataImport.load("../Configuration/DataImport/Resampled15min.json")
    data = solarhouse1_import.import_data(data_filename_solarhouse1)

#%%
    data_filename_solarhouse2 = "../../Data/Data/AEE/Solarhouse2/P2017_20_Solarhouse_2"
    solarhouse2_import = HDFImport(freq='15T', index_col='Zeitraum', cols_to_rename={'T_P_oo ': 'T_P_top'}, cols_to_drop=["P_Recool"])
    data = solarhouse2_import.import_data(data_filename_solarhouse2)
    solarhouse2_import.cols_to_rename = {label: label.split(" ")[0] for label in data.columns}
    solarhouse2_import.cols_to_rename.update({'T_P_oo ': 'T_P_top'})
    #solarhouse2_import.to_file(f"../Data/Configuration/AEE/Solarhouse2/P2017_20_Solarhouse_2.json")
    data_2 = solarhouse2_import.import_data(data_filename_solarhouse2)
    #data_3 = DataImport.load(f"../Configuration/DataImport/P2017_20_Solarhouse_2.json").import_data()
    print(data_2.columns)

    data_filename_cps = "../../Data/Data/Inffeldgasse/Datasets_Occupancy_WeatherData/cps_data"
    cps_import = ExcelImport(freq='1H', index_col='datetime', cols_to_rename={"time":"daytime"}, cols_to_drop=['Unnamed: 0'])
    cps_import.import_data(data_filename_cps)
    #cps_import.to_file(f"../Data/Configuration/Inffeldgasse/Datasets_Occupancy_WeatherData/cps_data.json")

#%%
    for name in ['A6','B2','C6']:
        data_filename_sensor = f"../../Data/Data/Inffeldgasse/Datasets_Occupancy_WeatherData/sensor_{name}"
        sensor_import = ExcelImport(freq='1H', index_col='datetime',
                                     cols_to_rename={"time": "daytime", 'consumption [kWh]': 'energy'})
        data_2 = sensor_import.import_data(data_filename_sensor)
        #sensor_import.to_file(f"../Data/Configuration/Inffeldgasse/Datasets_Occupancy_WeatherData/sensor_{name}.json")

    #filename = f"../Data/Data/Beyond/T24/T24_full"
    #t24_import = CSVImport(freq='1H', index_col='t')
    #t24_import.to_file(f"../Data/Configuration/Beyond/T24/T24_full.json")
    #data = t24_import.import_data(filename=filename)

#%%
    for name in ['B12','B20']:
        filename = f"../../Data/Data/Beyond/REFIT/{name}/Beyond_{name}_full"
        t24_import = CSVImport(freq='1H', index_col='dt')
        #t24_import.to_file(f"../Data/Configuration/Beyond/{name}/Beyond_{name}_full.json")
        data = t24_import.import_data(filename=filename)

#%%
    dataset_name = "processed-H01-Accounts-3-31-temperature"
    filename = f"../../Data/Data/Beyond/LEEDR/LEEDR_data_minute_temperature/{dataset_name}-CLEAN"

    data_import = TXTImport_Octave(freq='1min', use_octave_header=True, octave_header_file=dataset_name,
                            index_col='timestamp', index_type='datetime',datetime_fmt='posix')
    data_import.to_file(f"../../Data/Configuration/DataImport/Beyond/LEEDR/{dataset_name}")
    data = data_import.import_data(filename)
    print(data.columns)
