import os
from typing import List

import pandas as pd

from ModelTraining.Preprocessing.dataimport.data_import import import_data
import DataAnalysis.signal_processing_utils as sigutils
import datetime
from Plotting import plot_data as plt_utils


def rename_columns(df: pd.DataFrame, labels: List[str]):
    """
    Rename dataframe columns
    @param df: pd.Dataframe
    @param labels: list of new labels
    @return: renamed dataframe
    """
    return df.copy().rename({col: label for col, label in zip(list(df.columns), labels)}, axis=1) if labels else df


if __name__ == "__main__":
    filename = "P2017_20_Solarhouse_2.csv"
    hybridcosim_path = os.environ.get("HYBRIDCOSIM_REPO_PATH", "../../")
    path_to_dir = os.path.join(hybridcosim_path, "Data/AEE/")
    path_full = os.path.join(path_to_dir, filename)
    #datahouse = pd.read_hdf(path_full)
    #datahouse = pd.read_csv(path_full, sep=";")
    fig_save_path = "./Figures/"
    #datahouse.info()
    ##datahouse.dtypes

    # 01.01 15.03
    # 05.06 27.07
    # 29.12 07.03

    '''Missing Data Analysis'''
    # data preprocessing if there are nulls
    # generatetime= pd.DataFrame(columns=['NULL'],index=pd.date_range(datahouse.index[0], datahouse.index[-1], freq='1T'))
    # datahousemissing = datahouse.reindex(generatetime.index, fill_value=np.nan)

    lab = ['T_P_top', 'T_P_o', 'T_P_mo', 'T_P_mu', 'T_P_u', 'T_Holzkessel', 'T_Wohnraumofen',
           'T_Solar_VL', 'T_Solar_RL', 'Vd_Solar',
           'T_Nachheizung_VL', 'T_Nachheizung_RL', 'Vd_Nachheizung',
           'T_WW_VL', 'T_WW_RL', 'Vd_WW',
           'T_FBH_VL', 'T_FBH_RL', 'Vd_FBH',
           'Qel_Technik', 'Pel_Technik', 'Qel_Haushalt', 'Pel_Haushalt',
           'T_Raum', 'P_Holzkessel_calc', 'P_Ofen_calc', 'P_Recool']

    # we need to change the index format otherwise seaborn gives UTC000000 length labels.
    #plot_missing_values(datahouse, 'missingdata.png')
    #'''Filling the missing data'''
    df = import_data(path_full)
    # df.info()
    #df = datahouse.iloc[0:12000]
    year = 2019
    month = 5
    start_date=datetime.datetime(year,month,1)

    end_date = datetime.datetime(year, month,28)
    df = df[start_date:end_date]

#    datahouse['Zeitraum'] = pd.to_datetime(datahouse['Zeitraum'], format='%d.%m.%Y %H:%M')
#     datahouse.index=datahouse["Zeitraum"]
#     datahouse=datahouse.drop(["Zeitraum"], axis=1)

    # starttime = pd.Timestamp(datetime.datetime(2019,1,1))
    # stoptime = pd.Timestamp(datetime.datetime(2019,11,30))
    # df = datahouse[starttime]
    #df = df[df.index[df['Zeitraum'] == str(start_date)].tolist()[0]: df.index[df['Zeitraum'] == str(end_date)].tolist()[0]]
    #print(datahouse.iloc(datahouse["Zeitraum"]==start))
    # df = df.groupby(df.index.time).ffill()
    # df.info()

    # #Resample every 15 min get rid of the noise
    # Resampled = df.resample('15T').first()
    # Resampled.to_csv (r'..\Resampled15minStreit.csv', sep=';', index = True, header=True)

    # fig, (ax5) = plt.subplots(1, sharex=True)
    # ax5.plot(df.index, df['T_P_oo '], 'r', alpha=0.4, label='Top Raw filled')
    # ax5.plot(datahouse.index, datahouse['T_P_oo '], 'r', linestyle='--', label='Top Raw Original')
    # ax5.plot(Resampled.index, Resampled['T_P_oo '], 'darkorange', label='Top Raw Resampled')

    colors = ['red', 'blue', 'orange', 'purple']
    '''Storage Tank Labels'''
    storage_tank_labels = ['T_P_top', 'T_P_o', 'T_P_mo', 'T_P_mu', 'T_P_u']
    storage_tank_legends = ['Storage top top', 'Storage top', 'Storage middle top', 'Storage middle bottom', 'Storage bottom' ]

    '''Temperature Labels'''
    temperature_labels = ['T_Solar_VL', 'T_WW_RL', 'T_FBH_RL', 'T_Nachheizung_VL']
    temperature_legends = ['Solar supply', 'Domestic hot water return', 'Floor Heating return', 'Stove/Boiler supply']

    ''' Solar labels'''
    df["DeltaT_Solar"] = df["T_Solar_pri_VL"] - df["T_Solar_pri_RL"]
    solar_labels = ['T_Solar_VL', 'T_Solar_RL', 'T_Solar_pri_VL','T_Solar_pri_RL']
    solar_colors = ['red', 'blue', 'red', 'blue']
    linestyles = ['-', '-', '--','--']
    solar_legends = ['Solar supply', 'Solar return', 'Solar Supply Primary (collector side)', 'Solar return primary']

    ''' Y Labels '''
    ylabel_temperature = 'Temperature [°C]'
    ylabel_massflow = 'Volume flow rate'

    '''Storage Tank Temperatures'''

    # normalize mass flows
    massflow_labels = ['Vd_Solar', 'Vd_WW', 'Vd_FBH', 'Vd_Nachheizung']
    for label in massflow_labels:
        sigutils.scale_curve(df, label)

    scaled_massflow_labels = ['Vd_Solar_scaled', 'Vd_FBH_scaled', 'Vd_WW_scaled', 'Vd_Nachheizung_scaled']
    scaled_massflow_legends = ['Scaled Solar Volume flow', 'Scaled Warm Water Volume flow',
                               'Scaled Floor Heating Volume flow', 'Scaled Boiler Volume flow' ]
    data_massflow = rename_columns(df[scaled_massflow_labels], scaled_massflow_legends)
    storage_tank_data = rename_columns(df[storage_tank_labels], storage_tank_legends)


    plt_utils.plt_subplots([data_massflow, storage_tank_data], path=fig_save_path, filename='Storage Tank Temperatures and Volume flows', linestyle='--', figsize=(20, 10))

    '''Temperatures'''
    temperatures_data = rename_columns(df[temperature_labels], temperature_legends)
    plt_utils.plt_subplots([storage_tank_data, temperatures_data], path=fig_save_path, filename='Temperatures', linestyle='--', figsize=(20, 10))
    #plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
    #ax2.set_xlim([df.index.min(), df.index.max()])
    #ax2.grid(True)

    '''Solar'''
    solar_data_temp = rename_columns(df[solar_labels], solar_legends)
    solar_data_Vd = [rename_columns(df[['Vd_Solar']], ['Volume flow Solar']), rename_columns(df[['Pu_Sol']], ['Pump Solar activation'])]
    solar_data_rad = [rename_columns(df[['R_Global']], ['Solar radiation'])]
    solar_data_deltaT = [rename_columns(df[['DeltaT_Solar']], ['Delta T'])]
    plt_utils.plt_subplots([solar_data_temp, solar_data_Vd, solar_data_rad, solar_data_deltaT], path=fig_save_path, filename='Solar', linestyle='--', figsize=(20, 10))

    '''Domestic Hot water'''
    warm_water_labels = ['T_WW_VL', 'T_WW_RL']
    warm_water_legends = ['DomesticWater Supply', 'DomesticWater Return']
    warm_water_data = rename_columns(df[warm_water_labels], warm_water_legends)
    warm_water_data_massflows = [
        rename_columns(df[['Vd_WW']], ['Volume flow Warm Water']), rename_columns(df[['P_WW']], ['P_WW'])]
    plt_utils.plt_subplots([storage_tank_data, warm_water_data, warm_water_data_massflows],
                                                            path=fig_save_path, filename='Warm Water',
                                                            linestyle='--', figsize=(20, 10))

    '''Boiler'''
    boiler_temperature_labels = ['T_Nachheizung_VL', 'T_Nachheizung_RL']
    boiler_temperature_legends = ['Boiler Supply', 'Boiler Return']
    boiler_temperature_data = rename_columns(df[boiler_temperature_labels], boiler_temperature_legends)
    boiler_massflow_data = [
        rename_columns(df[['Vd_Nachheizung']], ['Volume flow Boiler']), rename_columns(df[['VEN_NH_RL']], ['Boiler return enable'])]
    plt_utils.plt_subplots([storage_tank_data, boiler_temperature_data, boiler_massflow_data],
                                                            path=fig_save_path, filename='Boiler',
                                                            linestyle='--', figsize=(20, 10))

    '''Floor Heating'''
    fbh_temperature_labels = ['T_FBH_VL', 'T_FBH_RL']
    fbh_temperature_legends = ['Floor Heating Supply', 'Floor Heating Return']
    fbh_temperature_data = rename_columns(df[fbh_temperature_labels], fbh_temperature_legends)
    fbh_massflow_data = [
        rename_columns(df[['Vd_FBH']], ['Volume flow Floor Heating']),
        rename_columns(df[['VEN_FBH_VL', 'Pu_FBH']], ['Pump Floor Heating', 'Floor Heating Supply Enable'])]
    plt_utils.plt_subplots([storage_tank_data, fbh_temperature_data, fbh_massflow_data],
                                                            path=fig_save_path, filename='Floor Heating',
                                                            linestyle='--', figsize=(20, 10))

