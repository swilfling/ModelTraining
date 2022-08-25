# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:32:56 2021

@author: Basak
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import Utilities.DataProcessing.data_import

import DymolaPythonInterface.TestbenchCreation.TestbenchUtilities.TestbenchUtilities
import Utilities.file_utilities as file_utils
import Utilities.DataProcessing.signal_processing_utils as sigutils

if __name__ == "__main__":
    save_figures = True

    # File paths
    hybridcosim_path = os.environ.get("HYBRIDCOSIM_REPO_PATH", "../../")
    csv_file_path = os.path.join(hybridcosim_path, "Data/AEE/Resampled15min.csv")
    fig_save_path = os.path.join(hybridcosim_path, "DataAnalysis","DataProcessingAEE","Solarhouse1", 'Figures')
    file_utils.create_dir(fig_save_path)
    df = Utilities.DataProcessing.data_import.import_data(csv_file_path)

    ########################### Outside temperature: Add mean per day ##################################

    current_data = df["TAussen"]
    num_samples_per_day = 4 * 24
    day_means = []
    datarange = range(0, len(current_data))

    for i in range(0, len(current_data), num_samples_per_day):
        mean_day = np.mean(current_data[i:i + num_samples_per_day])
        day_means.append(mean_day)

    mean_values_all_days = []
    for i in datarange:
        day = int(i / num_samples_per_day)
        mean_values_all_days.append(day_means[day])

    df["TAussen_Mean_Day"] = mean_values_all_days

    ############################# Storage Tank #####################################################

    df['deltaT_TPuffero'] = sigutils.diff(df['TPuffero'])
    df['deltaT_TPuffermo'] = sigutils.diff(df['TPuffermo'])
    df['deltaT_TPuffermu'] = sigutils.diff(df['TPuffermu'])
    df['deltaT_TPufferu'] = sigutils.diff(df['TPufferu'])


    #################### Solar Collector: Delta Q ###########################
    current_data = df["QSolar"]
    # THIS IS AN EMPIRICAL VALUE ! through 1 year we have peak value 5
    # higher values are interpreted as faulty
    threshold_delta_q_solar = 15
    new_data = current_data.diff(periods=1).fillna(0)
    deltaQ = sigutils.thresh_max(new_data, threshold_delta_q_solar)
    deltaQ = sigutils.thresh_min(deltaQ, 0)

    df["DeltaQSolar"] = deltaQ
    plt.figure()
    plt.title('DeltaQSolar')
    plt.stem(df["DeltaQSolar"])
    plt.show()

    ####################### Solar Collector: PSolar ###########################

    psolar = df["PSolar"]
    current_data = psolar
    psolar_new = sigutils.thresh_min(psolar, 0)
    df["PSolar"] = psolar_new
    plt.figure()
    plt.title('PSolar')
    plt.plot(psolar_new)
    plt.show()

    ######################## Solar Delta T Solar #########################
    tsolarvl = df["TSolarVL"]
    tsolarrl = df["TSolarRL"]
    deltatsolar = tsolarvl - tsolarrl
    df["DeltaTSolar"] = deltatsolar

    ######################## Plots #####################################

    plt.figure()
    plt.title("TSolarVL")
    plt.tight_layout()
    plt.plot(tsolarvl)
    plt.show()

    sglobal = df["SGlobal"]
    plt.figure()
    plt.title("SGlobal")
    plt.plot(sglobal)
    plt.show()

    taussen = df["TAussen"]
    plt.figure()
    plt.title("TAussen")
    plt.tight_layout()
    plt.plot(taussen)
    plt.show()

    starttime = pd.Timestamp(datetime.datetime(2019,10,20))
    stoptime = pd.Timestamp(datetime.datetime(2019,11,30))
    tsolarrl = df["TSolarRL"][starttime:stoptime]

    #time_index = df.index[starttime:stoptime]
    plt.figure()
    plt.title("TSolarVL, TSolarRL")
    plt.plot(tsolarrl)
    plt.plot(df["TSolarVL"][starttime:stoptime])
    plt.legend(["TSolarVL","TSolarRL"])
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("TOfenVL, TPuffero")
    plt.plot(df["TOfenVL"][starttime:stoptime])
    plt.plot(df["TPuffero"][starttime:stoptime])
    plt.tight_layout()
    plt.show()

    starttime = pd.Timestamp(datetime.datetime(2019, 2, 1))
    stoptime = pd.Timestamp(datetime.datetime(2019, 2, 20))

    plt.figure()
    plt.title("deltaTSolar")
    plt.plot(deltatsolar[starttime:stoptime])
    plt.tight_layout()
    plt.show()

    # plt.figure()
    # qsolar = df["QSolar"]
    # qsolar = sig_utils.fft_threshold(qsolar)
    # plt.plot(qsolar)
    # plt.show()

    DymolaPythonInterface.SimulationUtilities.SimulationUtilities.store_to_csv(df, csv_file_path)