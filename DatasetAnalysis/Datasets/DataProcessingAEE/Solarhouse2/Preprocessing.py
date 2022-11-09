import pandas as pd
import os
import ModelTraining.DatasetAnalysis.Preprocessing.data_preprocessing as dp_utils
import ModelTraining.Data.Plotting.plot_distributions as plt_utils


if __name__ == "__main__":
    filename = "P2017_20_Solarhouse_2"
    hybridcosim_path = os.environ.get("HYBRIDCOSIM_REPO_PATH", "../../")
    path_to_dir = os.path.join("../../../../", "Data/Data/AEE/Solarhouse2")
    path_full = os.path.join(path_to_dir, filename)
    datahouse = pd.read_hdf(path_full + ".hd5")
    path_csv_output = os.path.join(path_to_dir, filename + ".csv")

    '''Missing Data Analysis'''
    plt_utils.plot_missing_values(datahouse, filename=filename)


    '''Filling the missing data'''
    Resampled = datahouse
    # Remove spaces from labels
    for label in Resampled.columns:
        newlabel = label.split(" ")[0]
        Resampled = Resampled.rename(columns={label: newlabel})
    # Rename T_P_oo label
    Resampled = Resampled.rename(columns={'T_P_oo':'T_P_top'})

    # Filter modulated signals
    # Add new columns:
    # T_FBH_VL_filt, T_FBH_VL_env, T_FBH_RL_filt, T_FBH_RL_env,
    # Vd_FBH_filt, Vd_FBH_env
    T_FBH_modulation = 20 # this is not a temperature, this is a period
    fbh_labels = ['T_FBH_VL', 'T_FBH_RL', 'Vd_FBH']
    Resampled = dp_utils.demod_signals(Resampled, fbh_labels, T_FBH_modulation)
    plt_utils.plot_missing_values(Resampled, filename=filename)

    keep_nans = False
    if not keep_nans:
        Resampled = Resampled.where(Resampled.isna()==False)
        '''Filling the missing data'''
        Resampled = Resampled.copy()
        Resampled = Resampled.groupby(Resampled.index.time).ffill()

    plt_utils.plot_missing_values(Resampled, filename=filename)
    Resampled.to_csv(path_csv_output, sep=';', index=True, header=True, index_label='Zeitraum')

    #fig, (ax5) = plt.subplots(1, sharex=True)
    #ax5.plot(df.index, df['T_P_oo '], 'r', alpha=0.4, label='Top Raw filled')
    #ax5.plot(Resampled.index, Resampled['T_P_oo '], 'darkorange', label='Top Raw Resampled')
    #ax5.plot(datahouse.index, datahouse['T_P_oo '], 'r', linestyle='--', label='Top Raw Original')
    #plt.legend()
    #plt.show()

    #data = df['T_P_oo '][0:50000]
    #plot_spectra(data)

    #df_fftthresh = fft_threshold(data, 0.05)
    #plt.figure(figsize=(10, 5))
    #plt.title("Time Series - Denoising - Thresholded FFT")
    #plt.plot(data.values)
    #plt.plot(df_fftthresh)
    #plt.show()