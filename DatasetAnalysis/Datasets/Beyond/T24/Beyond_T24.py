import os
import pandas as pd

import Data.Plotting.plot_data as plt_utils

if __name__ == "__main__":

    beyond_data_path = os.path.join("..", "..", 'Data','Beyond', 'T24')
    os.makedirs('Figures/T24', exist_ok=True)
    # Merge climate and building data
    climate_data = pd.read_csv(os.path.join(beyond_data_path,'Weather London_Kew Gardens_SRC ID 723.csv'), index_col='t')
    climate_data.index = pd.DatetimeIndex(climate_data.index)

    print(climate_data.columns)

    data = pd.read_csv(os.path.join(beyond_data_path,f't24.csv'),encoding='UTF-8', index_col='t')
    # Drop first column
    data = data.drop(data.columns[0],axis=1)
    # Set index to Datetimeindex
    data.index = pd.DatetimeIndex(data.index)

    print(data.columns)
    # Merge climate data
    data_full = pd.merge(climate_data, data, left_index=True, right_index=True)
    data_full = data_full.dropna(axis=0)

    data_full.to_csv(os.path.join(beyond_data_path, "T24_full.csv"), index_label="t")

    # # Plot input features
    data_1 = data_full[['Text']]
    data_2 = data_full[['GHI']]
    data_3 = data_full[['Tint']]
    plt_utils.plt_subplots([data_1, data_2, data_3], "./Figures/T24", "Features", fig_title="Features", figsize=(20, 20), show_ylabel=True, show_legend=True)

