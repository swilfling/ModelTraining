#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
from ModelTraining.feature_engineering.featureengineering.filters import ButterworthFilter, Envelope_MA
import ModelTraining.DatasetAnalysis.Analysis.signal_processing_utils as sigutils
import Data.Plotting.plot_data as plt_utils

#%%

if __name__ == "__main__":

    hybridcosim_path = os.environ.get('HYBRIDCOSIM_REPO_PATH')
    beyond_data_path = os.path.join(hybridcosim_path, 'Data', 'Data','Beyond')

    files = ['','Beyond_-_B20_cleaned_data.csv','Beyond_-_REFIT_climate_data.csv']

    climate_data = pd.read_csv(os.path.join(beyond_data_path,files[2]), index_col='dt')
    climate_data.index = pd.DatetimeIndex(climate_data.index, freq='1H')
    climate_data = climate_data.drop(climate_data.columns[0],axis=1)

    dict_data = {}
    for dataset in ['B12','B20']:

        b20_data = pd.read_csv(os.path.join(beyond_data_path,dataset,f'Beyond_-_{dataset}_cleaned_data.csv'),encoding='UTF-8', index_col='dt')
        b20_data = b20_data.drop(b20_data.columns[0],axis=1)
        b20_data.index = pd.DatetimeIndex(b20_data.index)

        data_full = pd.merge(climate_data, b20_data, left_index=True, right_index=True)
        data_full.index = pd.DatetimeIndex(data_full.index, freq='1H')
        #print(data_full.columns)

        data_full = data_full.dropna(axis=0)
        room1_label = f'{dataset} - BR1 - Temperature (C)'
        plt.figure(figsize=(10,5))
        plt.plot(data_full[room1_label])
        plt.grid('both')
        plt.xlabel('Time')
        plt.ylabel(room1_label)
        plt.show()

        plt.figure(figsize=(10,5))
        signal = data_full[room1_label][pd.Timestamp(2014,3,1):pd.Timestamp(2014,3,31)]
        plt.plot(signal)
        sig_zm, offset = sigutils.remove_offset(signal)
        plt.plot(signal.index,ButterworthFilter(T=5).fit_transform(sig_zm) + offset)
        #plt.plot(signal.index,sigutils.filter_smoothe_signal(signal,T=100))
        plt.grid('both')
        #plt.ylim([19,25])
        plt.xlabel('Time')
        plt.ylabel(room1_label)
        plt.show()

        plt.figure(figsize=(10,5))
        plt.plot(data_full[room1_label][pd.Timestamp(2014,3,15):pd.Timestamp(2014,3,16)])
        plt.grid('both')
        plt.xlabel('Time')
        plt.ylabel(room1_label)
        plt.show()

        sig_zm, _ = sigutils.remove_offset(signal)
        fftfreqs, spec = sigutils.fft_abs(sig_zm)

        plt.figure(figsize=(10,5))
        plt.stem(24*fftfreqs,spec, label='Room Temperature')
        #plt.plot([1/75,1/75], [0,1.1*max(spec)], color='lightgreen', label = 'T = 75 days')
        plt.plot([1/29,1/29], [0,1.1*max(spec)], color='yellow', label = 'T = 1 month')
        plt.plot([1/7,1/7], [0,1.1*max(spec)], color='orange', label = 'T = 1 week')
        plt.plot([1/4,1/4], [0,1.1*max(spec)], color='magenta', label = 'T = 5 days')
        plt.plot([1,1], [0,1.1*max(spec)], color='red', label = 'T = 1 day')
        #plt.plot([12,12], [0,1.1*max(spec)], color='gray', label = '2h Trend')
        plt.plot([2,2], [0,1.1*max(spec)], color='gray', label = '12h Trend')
        plt.ylim([0,1.1*max(spec)])
        plt.xlim([2/365,12])
        plt.gca().set_xscale('log')
        plt.ylabel('Magnitude Spectrum - normalized')
        plt.xlabel('Frequency logarithmic [1/d]')
        plt.legend()
        plt.title(f'Dataset Beyond {dataset} BR1 - Room Temperature Magnitude Spectrum - Zero mean')
        plt.grid('both')
        plt.tight_layout()
        plt.savefig((f'./Figures/{dataset}/Spectrum_temp_log.png'))

        tikzplotlib.save('./Figures/spec_temp_log.tex')
        plt.show()


        data_full = data_full.rename(columns={'Outdoor temperature (C)':'TAmbient',
                                      'Global Horizontal solar Irradiance (W/m2)':'SGlobalH',
                                      'Outdoor Relative Humidity (%)':'humidity',
                                      'Wind Direction (degrees)':'dWind',
                                      'Wind Speed (m/s)':'vWind',
                                      'Total Rainfall (mm)':'rain',
                                      f'{dataset} - Gas Consumption (m3)':f'{dataset}Gas',
                                      f'{dataset} - BR1 - Temperature (C)':f'T{dataset}BR1',
                                      f'{dataset} - BR2 - Temperature (C)':f'T{dataset}BR2',
                                      f'{dataset} - BR3 - Temperature (C)':f'T{dataset}BR3',
                                      f'{dataset} - LR - Illuminance (lux)':f'lum{dataset}LR',
                                      f'{dataset} - LR - Temperature (C)':f'T{dataset}LR'})

        if dataset == 'B20':
            data_full = data_full.rename(columns={f'{dataset} - BR1 - Illuminance (lux)':f'lum{dataset}BR1'})

        data_full.to_csv(os.path.join(beyond_data_path,dataset, f'Beyond_{dataset}_full.csv'))
        columns = ['TAmbient', 'SGlobalH', 'humidity', 'vWind', 'rain', f'{dataset}Gas' ]
        sparsity = np.array([np.sum(data_full[feature] != 0)  / data_full.shape[0] for feature in columns])
        plt_utils.barplot(pd.Series(index=columns, data=sparsity),fig_save_path=f'./Figures/{dataset}', fig_title=f'Density Analysis - Dataset Beyond {dataset} BR1', figsize=(8,4))


        data_1 = data_full[[f'T{dataset}LR']]
        data_2 = data_full[[f'T{dataset}BR1']]
        data_3 = data_full[[f'T{dataset}BR2']]
        data_4 = data_full[[f'T{dataset}BR3']]

        # Energy Demand
        data_full = data_full.dropna(axis=0)
        date_range = pd.date_range(pd.Timestamp(2014, 4, 13), pd.Timestamp(2014, 4, 24), freq='1H')
        data_full_new = data_full
        for date in date_range:
            if date in data_full.index:
                data_full_new = data_full_new.drop(date, axis=0)
        print(data_full_new.shape[0])
        dict_data.update({dataset:data_full_new})
        room1_label = f'T{dataset}LR'

        signal = data_full_new[room1_label]
        sig_zm,_ = sigutils.remove_offset(signal)
        fftfreqs, spec = sigutils.fft_abs(sig_zm)
        plt.figure(figsize=(10,5))
        plt.stem(24*fftfreqs,spec, label=room1_label)
        #plt.plot([1/75,1/75], [0,1.1*max(spec)], color='lightgreen', label = 'T = 75 days')
        plt.plot([1/29,1/29], [0,1.1*max(spec)], color='yellow', label = 'T = 1 month')
        plt.plot([1/7,1/7], [0,1.1*max(spec)], color='orange', label = 'T = 1 week')
        plt.plot([1/4,1/4], [0,1.1*max(spec)], color='magenta', label = 'T = 5 days')
        plt.plot([1,1], [0,1.1*max(spec)], color='red', label = 'T = 1 day')
        #plt.plot([12,12], [0,1.1*max(spec)], color='gray', label = '2h Trend')
        plt.plot([2,2], [0,1.1*max(spec)], color='gray', label = '12h Trend')
        plt.ylim([0,1.1*max(spec)])
        plt.xlim([2/365,12])
        plt.gca().set_xscale('log')
        plt.ylabel('Magnitude Spectrum - normalized')
        plt.xlabel('Frequency logarithmic [1/d]')
        plt.legend()
        plt.title(f'Dataset Beyond {dataset} - {room1_label} Magnitude Spectrum - Zero mean')
        plt.grid('both')
        plt.tight_layout()
        plt.savefig((f'./Figures/{dataset}/Spectrum_gas_temp_log.png'))

        plt.figure(figsize=(10,5))
        plt.stem(24*fftfreqs,spec, label=room1_label)
        #plt.plot([1/75,1/75], [0,1.1*max(spec)], color='lightgreen', label = 'T = 75 days')
        #plt.plot([1,1], [0,1.1*max(spec)], color='red', label = 'T = 1 day')
        plt.plot([12,12], [0,1.1*max(spec)], color='gray', label = '2h Trend')
        plt.plot([2,2], [0,1.1*max(spec)], color='gray', label = '12h Trend')
        plt.ylim([0,1.1*max(spec)])
        plt.xlim([2/365,12])
        plt.ylabel('Magnitude Spectrum - normalized')
        plt.xlabel('Frequency logarithmic [1/d]')
        plt.legend()
        plt.title(f'Dataset Beyond {dataset} - {room1_label} Magnitude Spectrum - Zero mean')
        plt.grid('both')
        plt.tight_layout()
        plt.savefig((f'./Figures/{dataset}/Spectrum_gas_temp_log.png'))

#%%

        import scipy.signal as sig
        signal,_ = sigutils.remove_offset(data_full[room1_label])
        #signal = signal.to_numpy()
        print(signal.shape)
        highpass = ButterworthFilter(filter_type='highpass', T=np.array([24*30]), order=3)
        lowpass = ButterworthFilter(filter_type='lowpass', T=np.array([48]), order=5)
        bandpass = ButterworthFilter(filter_type='bandpass', T=np.array([12,24]), order=2)
        highpass.fit(signal)
        lowpass.fit(signal)
        bandpass.fit(signal)
        envelope = Envelope_MA(T=24*7)
        plt.figure()
        w1, spec1 = sig.freqz(*highpass.get_coef())
        w2, spec2 = sig.freqz(*lowpass.get_coef())
        w3, spec3 = sig.freqz(*bandpass.get_coef())
        plt.plot(w1, np.abs(spec1))
        plt.plot(w2, np.abs(spec2))
        plt.plot(w3, np.abs(spec3))
        plt.show()
        plt.figure()
        plt.plot(w1, np.rad2deg(np.angle(spec1)))
        plt.plot(w2, np.rad2deg(np.angle(spec2)))
        plt.plot(w3, np.rad2deg(np.angle(spec3)))
        plt.show()

        print(np.std(signal))
        plt.figure(figsize=(20,5))
        plt.plot(signal)
        plt.plot(lowpass.fit_transform(signal))
        print(np.std(lowpass.fit_transform(signal)))
        plt.show()

        plt.figure(figsize=(20,5))
        plt.plot(signal)
        plt.plot(highpass.fit_transform(signal))
        plt.show()

        plt.figure(figsize=(20,5))
        plt.plot(signal)
        plt.title('Lowpass + Highpass')
        plt.plot(highpass.fit_transform(signal)+ lowpass.fit_transform(signal))
        plt.show()

        plt.figure(figsize=(20,5))
        plt.title('Bandpass')
        plt.plot(lowpass.fit_transform(highpass.fit_transform(signal)))
        plt.show()

        env = envelope.fit_transform(signal)
        signal_2 = data_full['TAmbient']
        env_2 = envelope.fit_transform(signal_2)
        plt.title('Envelope')
        plt.figure(figsize=(20,5))
        plt.plot(signal.index,env)
        plt.plot(signal.index,env_2)
        plt.show()
        env = env.dropna()
        env_2 = env_2.dropna()
        import sklearn
        print(sklearn.feature_selection.r_regression(env.to_numpy().reshape(-1, 1), env_2.to_numpy().reshape(-1, 1)))

        #plt.figure(figsize=(20,5))
        #plt.plot(lowpass.fit_transform(highpass.fit_transform(signal[600:700])))
        #plt.show()

        plt.figure(figsize=(20,5))
        fft = np.abs(np.fft.fft(lowpass.fit_transform(signal)))
        #plt.stem(fft[700:])
        plt.stem(fft)
        plt.show()

#%%

        plt.figure(figsize=(20,5))
        plt.plot(signal[5000:])
        plt.plot(data_full['TAmbient'][5000:] - np.mean(data_full['TAmbient'][5000:]))
        plt.plot(data_full[f'{dataset}Gas'][5000:]*20)
        plt.legend(['Tindoor', 'TAmbient',f'{dataset}Gas'])

        plt.figure(figsize=(20,5))
        plt.plot(signal[5000:])
        plt.plot(data_full['TAmbient'][5000:] - np.mean(data_full['TAmbient'][5000:]))
        plt.plot(data_full[f'{dataset}Gas'][5000:]*20)

        #print(sklearn.feature_selection.r_regression(data_full['TAmbient'][5000:].to_numpy().reshape(-1, 1), signal[5000:].to_numpy().reshape(-1, 1)))
        plt.show()

        plt.figure(figsize=(20,5))
        plt.plot(data_full['vWind'][5000:], label='wind')
        plt.plot(data_full['humidity'][5000:]/100, label='hum')
        plt.plot(data_full['rain'][5000:], label='rain')
        plt.legend()
        plt.show()


        plt.figure(figsize=(20,5))
        plt.plot(data_full[f'lum{dataset}LR'][5000:])
        plt.show()

        plt.figure(figsize=(20,5))
        plt.plot(data_full[f'{dataset}Gas'][5000:])
        plt.show()

#%%

        plt.figure(figsize=(20,5))
        plt.plot(data_full[f'T{dataset}LR'][5000:5500])
        plt.legend([f'T{dataset}LR'])
        plt.show()

        plt.figure(figsize=(20,5))
        plt.plot(data_full['TAmbient'][5000:5500])
        plt.plot(data_full[f'T{dataset}LR'][5000:5500])
        plt.legend(['TAmbient', f'T{dataset}LR'])
        plt.show()

        plt.figure(figsize=(20,5))
        plt.plot(data_full[f'T{dataset}LR'][5000:5500] - data_full['TAmbient'][5000:5500])
        plt.legend(['deltaT'])
        plt.show()

        plt.figure(figsize=(20,5))
        plt.plot(data_full['SGlobalH'][5000:5500])
        plt.plot(data_full[f'lum{dataset}LR'][5000:5500])
        plt.legend(['SGlobal (Outddor Luminance)', 'Indoor Luminance'])
        plt.show()

        plt.figure(figsize=(20,5))
        plt.plot(data_full['rain'][5000:5500])
        plt.legend(['Rain'])
        plt.show()

        plt.figure(figsize=(20,5))
        plt.plot(data_full['vWind'][5000:5500])
        plt.legend(['Wind'])
        plt.show()

#%%
    data_full_new = dict_data['B20']
    plt.figure(figsize=(20,5))
    #plt.plot(data_full_new['TB20LR'])
    plt.plot(data_full_new[f'TB20LR'][data_full_new.index.week == 30])
    plt.show()

    plt.figure(figsize=(20,5))
    #plt.plot(data_full_new['TB20LR'])
    plt.plot(data_full_new['TB20LR'][data_full_new.index.week == 29])
    plt.show()

#%%
    plt.figure(figsize=(20,5))
    plt.plot(data_full_new['TB20LR'][int(data_full.shape[0]*0.64):int(data_full.shape[0]*0.8)])
    plt.show()
    print(set(data_full_new.index.week[int(data_full.shape[0]*0.64):int(data_full.shape[0]*0.8)]))


#%%
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=3)
    print(data_full_new.columns)
    dbscan.fit(data_full_new[['TAmbient','SGlobalH','humidity', 'vWind', 'lumB20LR']])
    weights = np.ones(data_full_new.shape[0])
    weights[dbscan.core_sample_indices_] = 2
    print(dbscan.core_sample_indices_)
    print(np.sum(weights == 2) / data_full_new.shape[0])


