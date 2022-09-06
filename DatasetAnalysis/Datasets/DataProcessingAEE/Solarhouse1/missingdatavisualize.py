import pandas as pd
import numpy as np
import seaborn as sns
import os

folder_relative_path = "../Data"
filename = "Resampled15min.csv"
hybridcosim_path = os.environ.get("HYBRIDCOSIM_REPO_PATH", "../../")
csv_file_path = os.path.join(hybridcosim_path, "Data", "AEE", "Resampled15min.csv")

#Thelables that we are interested in are selected and the names are changed as a second csv file,
datahouse = pd.read_csv(csv_file_path, sep=';', encoding='latin-1', header=0, low_memory=False)
datahouse['Zeitraum']= pd.to_datetime(datahouse['Zeitraum'], format='%d.%m.%Y %H:%M')
datahouse = datahouse.set_index('Zeitraum')
 
datahouse.info()
datahouse.dtypes

#data preprocessing if there are nulls 
generatetime= pd.DataFrame(columns=['NULL'],index=pd.date_range(datahouse.index[0], datahouse.index[-1], freq='1T'))
datahouse = datahouse.reindex(generatetime.index, fill_value=np.nan)

#we need to change the index format otherwise seaborn gives UTC000000 length labels. 
datahouse = datahouse.set_index(generatetime.index.strftime('%d.%m.%Y'))

# sns.set(font_scale=1.3)
ax = sns.heatmap(datahouse.isna(), xticklabels=1, cbar=False)
ax.set_ylabel('Time Period', fontsize=20)
ax.set_xlabel('Sensors', fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

