# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:10:02 2021

@author: Basak
"""
from pathlib import Path
from ModelTraining.DatasetAnalysis.Datasets.DataProcessingInffeldgasse.EnergoPlus.VirtualMeasurements.VirtualMeasurementFunctions import *
import Data.Plotting.plot_data as plt_utils
import Data.Plotting.plot_distributions as plt_dist

if __name__ == "__main__":
    root_dir = (Path(__file__).parent).parent.parent.parent.parent
    filename = root_dir / "Data/Inffeldgasse/EnergoPlus_Data/AllSensors.csv"
    all_data = pd.read_csv(filename, delimiter=',')
    all_data = all_data.set_index(pd.to_datetime(all_data['Timestamp'], format='%Y-%m-%d %H:%M:%S'))
    all_data = all_data.drop(['Timestamp'], axis=1)

    create_missing_value_plot = True

    if create_missing_value_plot:
        plt_dist.plot_missing_values(all_data, "../Figures/", "Inffeld_MissingValues.png", fig_title="Missing values")

    # Drop Nans that are common in all columns
    #new_all_data = all_data.dropna(axis=0, how='all')
    new_all_data = all_data

    print("Available data:")
    print(list(new_all_data.columns))


    # Create virtual measurements and add them to data frame
    df_key_extension = "_MWh"
    df_virtual_measurements = get_measurements_based_on_csv('calculations.csv', new_all_data, df_key_extension)
    data_full = pd.concat([new_all_data, df_virtual_measurements],axis=1)

    ########### Calculate year consumption
    #starttime = datetime.datetime(2019,1,1)
    #endtime = datetime.datetime(2019, 12, 31)
    year = "2019"
    data_full_year = data_full[year]
    sum_data = data_full_year.sum(axis=0)
    print(sum_data)

    #################### Plotting ########################################

    #data_full = data_full[0:10000]
    list_inffeld10 = ["IN10-H_MELFT_MWh"]
    list_inffeld11 = ["IN11-HGES_MWh", "IN11-H72_MWh", "IN11-H73_MWh"]
    list_inffeld12 = ["IN12-HGES_MWh"]
    list_inffeld13 = ["IN13-HGES_MWh", "IN13-HSRWP1_MWh", "IN13-HSRWP2_MWh"]
    list_inffeld18 = ["IN18-HGES_MWh"]
    list_inffeld19 = ["IN19-HGES_MWh"]
    list_inffeld21 = ["IN21+HGES_MWh"]
    list_inffeld21ab = ["IN21AB-HGES_MWh"]
    list_inffeld23 = ["IN21AB-HGES_MWh"]
    list_inffeld25 = ["MAF+HGES2_MWh"]


    list_data = [list_inffeld10, list_inffeld11, list_inffeld12, list_inffeld13,
                 list_inffeld18, list_inffeld19, list_inffeld21, list_inffeld21ab, list_inffeld23, list_inffeld25]

    data = [data_full[values] for values in list_data]

    plt_utils.plt_subplots(data, path="../Figures/", filename="Inffeld_Buildings", fig_title="Inffeld Buildings",
                           figsize=(20, 20))
