#%%

from ModelTraining.Data.DataImport.dataimport import ExcelImport
import os
import numpy as np
import ModelTraining.Data.Plotting.plot_data as plt_utils
import ModelTraining.DatasetAnalysis.Analysis.signal_processing_utils as sigutils
import tikzplotlib

#%%

if __name__=='__main__':
    root_dir = "../../../../"
    csv_file_path = os.path.join(root_dir, "Data/Data/Inffeldgasse/Datasets_Occupancy_WeatherData/cps_data.xlsx")

    csv_file_path = os.path.join(root_dir, "Data/Data/Inffeldgasse/Datasets_Occupancy_WeatherData/sensor_A6.xlsx")

    fig_save_path = "./Figures"

    # Create output dir, read csv
    os.makedirs(fig_save_path, exist_ok=True)

    df_orig = ExcelImport(cols_to_rename={"consumption [kWh]":"energy", "time":"daytime"}).import_data(csv_file_path)

    print(df_orig.columns)
    df_orig['holiday_weekend'] = np.logical_or(df_orig['holiday'], (df_orig['weekday'] == 5) + (df_orig['weekday'] == 6))

    df = df_orig[0:1000]

#%%
    plt_utils.plt_subplots([df[['temperature']]], fig_save_path, "Temperature")

    plt_utils.plt_subplots([df[['energy']]], fig_save_path, "EnergyDemand")

#%%

    energy_demod = sigutils.calculate_envelope(df['energy'], T=24)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(energy_demod)
    plt.show()

#%%

    plt.figure()
    offset = 3*24
    num_days = 5
    end = 24*num_days+offset
    energy_demod = sigutils.calculate_envelope(df['energy'][offset:end], T=24)
    plt.plot(df.index[offset:end],energy_demod)
    plt.plot(df.index[offset:end],df['energy'][offset:end])
    plt.show()

#%%

    energy = df_orig['energy']
    sig_zm, _ = sigutils.remove_offset(energy)
    fftfreqs,spec = sigutils.fft_abs(sig_zm)
    # spectrum
    plt.figure(figsize=(10,5))
    plt.stem(24*fftfreqs,spec, label='Energy Demand')
    import pandas as pd
    vals = pd.Series(index=24*fftfreqs, data=spec, name='W')
    vals.to_csv('./Figures/spec_energydemand.csv', index_label='f')
    plt.plot([1/100,1/100], [0,1.1*max(spec)], color='lightgreen', label = 'T = 100 days')
    plt.plot([1/30,1/30], [0,1.1*max(spec)], color='yellow', label = 'T = 1 month')
    plt.plot([1/7,1/7], [0,1.1*max(spec)], color='orange', label = 'T = 1 week')
    plt.plot([1,1], [0,1.1*max(spec)], color='red', label = 'T = 1 day')
    #plt.plot([12,12], [0,1.1*max(spec)], color='gray', label = '2h Trend')
    plt.ylim([0,1.1*max(spec)])
    plt.xlim([2/365,12])
    plt.gca().set_xscale('log')
    plt.ylabel('Magnitude Spectrum - normalized')
    plt.xlabel('Frequency logarithmic [1/d]')
    plt.legend()
    plt.title(f'Dataset CPS-Data - Energy Demand Magnitude Spectrum - Zero mean')
    plt.grid('both')
    plt.tight_layout()
    plt.savefig(f'./Figures/Spectrum_energydemand_log.png')

    tikzplotlib.save('./Figures/spec_energydemand.tex')
    plt.show()

#%%

    plt.figure(figsize=(10,2))
    plt.tight_layout()
    #plt.plot(df_orig['uni_holiday'])
    plt.plot(df_orig['holiday'])
    #plt.legend(['Holiday'])
    plt.yticks([0,1])
    plt.ylabel('holiday')
    plt.savefig("././Figures/Holiday.png")
    plt.figure(figsize=(10,2))
    plt.tight_layout()
    plt.plot(df_orig['holiday_weekend'])
    plt.yticks([0,1])
    plt.ylabel('holiday_weekend')
    #plt.legend(['Holiday or weekend'])
    plt.savefig("././Figures/Holiday_weekend.png")
    percentage = np.sum(df_orig['holiday']) / df_orig.shape[0]
    print(np.sum(df_orig['holiday_weekend']) / df_orig.shape[0])
    print(percentage)
    df_orig['weekday'] = df_orig['weekday']+1
    df_orig['daytime'] = df_orig['daytime']+1
    columns = df_orig.columns[:7]
    sparsity = np.array([np.sum(df_orig[feature] != 0)  / df_orig.shape[0] for feature in columns])

    plt_utils.barplot(pd.Series(index=columns, data=sparsity),path='./Figures', fig_title='Density Analysis - Dataset CPS-Data', figsize=(8,4))


#%%


