# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:10:02 2021

@author: Basak, Sandra
"""
import os
from pathlib import Path
from parse_excel_files import parse_excel_files
from ModelTraining.DatasetAnalysis.Datasets.DataProcessingInffeldgasse.EnergoPlus.VirtualMeasurements.VirtualMeasurementFunctions import *
import Data.Plotting.plot_data as plt_utils
import Data.Plotting.plot_distributions as plt_dist


if __name__ == "__main__":

    ####################### Read data ######################################


    file_extension = ".xls"
    create_missing_value_plot = True
    read_excel_files = True
    write_to_csv = True
    read_csv_file = True
    csv_filename ="AllSensors.csv"
    # Recommendation: run script once with excel file reading and store to csv enabled,
    # then disable excel file reading and enable csv reading

    # Paths to data directory
    root_dir = (Path(__file__).parent).parent.parent.parent.parent
    WorkingDirectory = root_dir / "Data" / "Inffeldgasse" / "EnergoPlus_Data"

    print("Starting Data Readout")

    # Parse Excel files
    if read_excel_files:
        print("Parsing Excel Files")
        all_data = parse_excel_files(WorkingDirectory, file_extension)
        # Store data to csv
        if write_to_csv:
            print("Storing raw data to CSV file")
            all_data.to_csv(WorkingDirectory / csv_filename, index=True, index_label='Timestamp')

    # Read CSV files
    if read_csv_file:
        print("Reading CSV file")
        all_data = pd.read_csv(WorkingDirectory / csv_filename, delimiter=',')

    print("Data readout finished")

    ############################ Cleanup data, plot missing values ##########

    print(os.linesep + "Cleaning up data")

    if all_data is None:
        print("Error: No data read out")
        exit(1)

    # Set index
    all_data = all_data.set_index(pd.to_datetime(all_data.get('Timestamp'), format='%Y-%m-%d %H:%M:%S'))

    # Create missing value plot - checks for NaNs
    if create_missing_value_plot:
        plt_dist.plot_missing_values(all_data, "../Figures/", "Inffeld_MissingValues", fig_title="Missing values")

    # Drop Nans that are common in all columns
    new_all_data = all_data.dropna(axis=0, how='all')
    print("Available data:")
    print(list(new_all_data.columns))


    #####################################################################

    print(os.linesep + "Adding virtual measurements (calculated from existing sensor data)")

    # Create virtual measurements and add them to the data frame
    # Virtual measurements for consumption values - in MWh
    df_virtual_measurements_MWh = get_measurements_based_on_csv('calculations.csv', new_all_data, "_MWh")
    data_full = pd.concat([new_all_data, df_virtual_measurements_MWh],axis=1)

    # Virtual measurements for load values - in MW
    df_virtual_measurements_MW = get_measurements_based_on_csv('calculations.csv', new_all_data, "_MW")
    data_full = pd.concat([data_full, df_virtual_measurements_MW], axis=1)

    ########### Calculate year consumption #############################

    print(os.linesep + "Calculating year consumption")
    # Alternative method - give start and end time
    #starttime = datetime.datetime(2019,1,1)
    #endtime = datetime.datetime(2019, 12, 31)

    # Consumption for year 2019
    year = "2019"
    data_full_year = data_full[year]
    sum_data = data_full_year.sum(axis=0)
    print("Year consumption: ")
    print(sum_data)

    #################### Plotting ########################################

    print(os.linesep + "Now plotting data")

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

    plt_utils.plt_subplots(data, path="../Figures/", filename="Inffeld Buildings", figsize=(20, 20))
