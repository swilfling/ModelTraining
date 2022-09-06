#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
import Data.Plotting.plot_data as plt_utils


#%%

if __name__ == "__main__":

    beyond_data_path = os.path.join("..", "..", 'Data','Beyond')
    os.makedirs('Figures/B12', exist_ok=True)
    os.makedirs('Figures/B20', exist_ok=True)
    # Merge climate and building data
    climate_data = pd.read_csv(os.path.join(beyond_data_path,'Beyond_-_REFIT_climate_data.csv'), index_col='dt')
    climate_data.index = pd.DatetimeIndex(climate_data.index)
    climate_data = climate_data.drop(climate_data.columns[0],axis=1)

    for dataset in ['B12','B20']:

        data = pd.read_csv(os.path.join(beyond_data_path,dataset,f'Beyond_-_{dataset}_cleaned_data.csv'),encoding='UTF-8', index_col='dt')
        # Drop first column
        data = data.drop(data.columns[0],axis=1)
        # Set index to Datetimeindex
        data.index = pd.DatetimeIndex(data.index)

        # Merge climate data and
        data_full = pd.merge(climate_data, data, left_index=True, right_index=True)
        data_full = data_full.dropna(axis=0)

        # Rename column
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

        # Plot input features
        data_1 = data_full[['TAmbient']]
        data_2 = data_full[['SGlobalH']]
        data_3 = data_full[['humidity']]
        data_4 = data_full[['vWind']]
        data_5 = data_full[['rain']]
        data_7 = data_full[[f'lum{dataset}LR']]
        data_8 = data_full[[f'{dataset}Gas']]

        # Clean up data
        # Wind speed
        # Beaufort Scale 12 - 32.7 m/s
        # average wind speed in Austria: 3 m/s
        # Outliers: vWind > 10 m/s
        data_full['vWind'][data_full['vWind'] > 10 ] = 0

        # Store data to csv
        data_full.to_csv(os.path.join(beyond_data_path,dataset, f'Beyond_{dataset}_full.csv'))
        plt_utils.plt_subplots([data_1,data_2,data_3,data_4,data_5,data_8],f"./Figures/{dataset}","InputFeatures", figsize=(20,20), show_ylabel=True)

        # Possible target features
        data_1 = data_full[[f'T{dataset}LR']]
        data_2 = data_full[[f'T{dataset}BR1']]
        data_3 = data_full[[f'T{dataset}BR2']]
        data_4 = data_full[[f'T{dataset}BR3']]
        plt_utils.plt_subplots([data_1,data_2,data_3,data_4],f"./Figures/{dataset}",f"TargetFeatures_{dataset}", figsize=(20,20), show_ylabel=True)

        # Data cleanup
        data_full = data_full.dropna(axis=0)

        # Plot different timespans of target feature
        target_feature = f'{dataset}Gas'
        plt.figure(figsize=(10,5))
        plt.plot(data_full[target_feature])
        plt.grid('both')
        plt.xlabel('Time')
        plt.ylabel(target_feature)
        plt.show()

        plt.figure(figsize=(10,5))
        plt.plot(data_full[target_feature][pd.Timestamp(2014,4,1):pd.Timestamp(2014,4,30)])
        plt.grid('both')
        plt.xlabel('Time')
        plt.ylabel(target_feature)
        plt.show()

