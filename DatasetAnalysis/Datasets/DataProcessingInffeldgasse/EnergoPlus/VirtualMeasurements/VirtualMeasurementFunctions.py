# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:10:02 2021

@author: Basak
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


# This function creates the following list from a csv line:
# {'<Dataframe label>': '<+/->', '<Dataframe label>': '<+/->'}
# {'IN21-HBT1': '+', 'IN21-HBT2': '+'}
def parse_csv_line(line):
    measurement = line["Virtual Measurement"]
    list_items = {}
    # Get all not-NaN items
    for item in line[1:]:
        if not pd.isna(item):
            list_items.update({item[1:]: '-'} if item[0] == '-' else {item:'+'})
    return {measurement: list_items}

# Parse CSV file and create dictionary of measurements
def parse_csv(csv_filename):
    csv_lines = pd.read_csv(csv_filename)
    dict = {}
    for _, line in csv_lines.iterrows():
        dict.update(parse_csv_line(line))
    return dict


# This function calculates a measurement based on the following variables list:
# {'<Dataframe label>': '<+/->', '<Dataframe label>': '<+/->'}
# Example:
# {'IN21-HBT1': '+', 'IN21-HBT2': '+'}
def calculate_measurement(measurement_key, list_variables, df, df_key_extension):
    meas_values = np.zeros((df.shape[0],1))
    for key, pos_neg_value in list_variables.items():
        # Get dataframe column
        if key + df_key_extension in df.columns:
            df_values = np.array(df[key + df_key_extension]).reshape((df.shape[0],1))
            meas_values += (-df_values if pos_neg_value == '-' else df_values)
        else:
            print("Calculation Error in measurement " + measurement_key + ": Key " + key + " is not in dataframe.")
    return meas_values


# Parse CSV and get measurements
def get_measurements_based_on_csv(csv_filename, df, df_key_extension):
    measurements = parse_csv(csv_filename)
    #print(measurements)
    df_new = calculate_virtual_measurements(measurements, df, df_key_extension)
    return df_new


# Calculate measurements based on dictionary
def calculate_virtual_measurements(measurements, df, df_key_extension):
    df_new = pd.DataFrame(index=df.index)
    for key, values in measurements.items():
        meas_values = calculate_measurement(key, values,df,df_key_extension)
        df_new[key + df_key_extension] = meas_values.flatten()
    return df_new
