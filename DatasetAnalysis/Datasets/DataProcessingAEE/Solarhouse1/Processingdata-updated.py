# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:32:56 2021

@author: Basak
"""
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

#Load the dataset
hybridcosim_path = os.environ.get("HYBRIDCOSIM_REPO_PATH", "../../")
csv_file_path = os.path.join(hybridcosim_path, "Data/AEE/Resampled15min.csv")
datahouse = pd.read_csv(csv_file_path, sep=';', encoding='latin-1', header=0, low_memory=False)

#prepare datetime
datahouse['Zeitraum']= pd.to_datetime(datahouse['Zeitraum'], format='%d.%m.%Y %H:%M')
datahouse.info()
datahouse.dtypes

df = datahouse.copy()
df = df.groupby(df['Zeitraum'].dt.time).ffill()
df = df.set_index(df['Zeitraum'])

datahouse = datahouse.set_index(df['Zeitraum'])
#Generate the whole datetime 
generatetime= pd.DataFrame(columns=['NULL'],index=pd.date_range(datahouse.index[0], datahouse.index[-1], freq='1T'))
#Now we fill Nans to missing spots
datahouse = datahouse.reindex(generatetime.index, fill_value=np.nan)
# #check for missing datetimeindex values based on reference index (with all values)
# missing_dates = datahouse.index[np.where(datahouse.isna())[0]]
# #Grab the Boolean
# bool_cols = [col for col in datahouse if np.isin(datahouse[col].dropna(), [0, 1]).all() and (datahouse[col].max() == 1 or datahouse[col].max() == 0)]
# rest_cols = list(set(datahouse.columns).difference(bool_cols))

df = df.reindex(generatetime.index, fill_value=np.nan)
df = df.groupby(df.index.time).ffill()
df.info()


#Example to plot the difference
fig, (ax5, ax3) = plt.subplots(2, sharex=True)
ax5.plot(df.index, df['TPuffero'], 'r', label='Storage top Processed Data')
ax5.plot(df.index, df['TPufferu'], 'blue', label='Storage bottom Processed Data')
ax5.plot(datahouse.index, datahouse['TPuffero'], 'darkorange', linestyle="--", label='Storage top Un-processed Data')
ax5.plot(datahouse.index, datahouse['TPufferu'], 'c', linestyle="--", label='Storage bottom Un-processed Data')
ax5.set_ylabel('Temperature \n Levels in \n Tank [°C]')
ax5.legend(loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax5.grid(True)

ax3.plot(df.index, df['PSolar'], 'b', label='Power Solar Processed Data')
ax3.plot(datahouse.index, datahouse['PSolar'], 'c', linestyle="--", label='Power Solar Un-processed Data')
ax33 = ax3.twinx()
ax33.plot(df.index, df['VDSolar'], 'black', alpha=0.6, label ='Volume flow rate Solar Processed Data')
ax33.plot(datahouse.index, datahouse['VDSolar'], 'black', linestyle="--", label='Power Solar Un-processed Data')
ax33.legend(loc=0)
ax33.set_ylabel('Volume flow [l/hr]')
ax3.set_ylabel('Power Solar [kW]')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax3.grid(True)

fig.suptitle('Solar Part')
plt.show()

#Resample every 15 min get rid of the noise
Resampled = df.resample('15T').first()
Resampled = Resampled.drop(['Zeitraum'], axis=1)

#Add the calculated values in the Resampled 15 min data
#VDSolarStorage, msolartorage, VDSolarPool, msolarpool, VDStoragePool,mstoragePool, VDPoolTot, mpooltot
Resampled['VDSolarStorage'] = Resampled['VDSolar']*np.logical_xor(df['GSolarPufferPool'],1).astype(int)
Resampled['msolartorage'] = Resampled['VDSolarStorage'] *0.00028
Resampled['VDSolarPool'] = Resampled['VDSolar'] - Resampled['VDSolarStorage']
Resampled['msolarpool'] = Resampled['VDSolarPool'] *0.00028
Resampled['VDStoragePool'] = Resampled['VDSchwimmbad']*Resampled['PuPufferPool']
Resampled['mstoragePool'] = Resampled['VDStoragePool'] *0.00028
Resampled['VDPoolTot'] = Resampled['VDSolarPool'] + Resampled['VDStoragePool']
Resampled['mpooltot'] = Resampled['VDPoolTot'] *0.00028

#Assign load/unload conditions: choose the mass flow that enters the tank, if they are all equal (true) change it to 0 
#Load = 1, unload = 0
Flows = ['VDWP', 'VDSolarStorage', 'VDPoolTot', 'VDOfen', 'VDFBH']
Resampled['PCondition'] = np.logical_xor(Resampled[Flows].eq(0).all(1).astype(int),1).astype(int)

#save the processed 

Resampled.to_csv (r'..\Resampled15min.csv', sep=';', index = True, header=True)

#Example to plot the difference
fig, (ax5) = plt.subplots(1, sharex=True)
ax5.plot(df.index, df['TPuffero'], 'r', label='Storage top Processed Data')
ax5.plot(datahouse.index, datahouse['TPuffero'], 'darkorange', linestyle="--", label='Storage top Un-processed Data')
ax5.plot(Resampled.index, Resampled['TPuffero'], 'purple', linestyle="--", label='Storage top Processed Data Resampled')
ax5.set_ylabel('Temperature \n Levels in \n Tank [°C]')
ax5.legend(loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax5.grid(True)
fig.suptitle('Solar Part')
plt.show()

# test = datahouse.loc[datahouse.isnull().any(axis=1)]

# #sum the Nans 
# numberofnan = datahouse.isna().sum()

# datahouse.describe()

# #Grab the Boolean
# bool_cols = [col for col in datahouse if np.isin(datahouse[col].dropna(), [0, 1]).all() and (datahouse[col].max() == 1 or datahouse[col].max() == 0)]
# rest_cols = list(set(datahouse.columns).difference(bool_cols))


# df = datahouse.copy()

# #IMPUTING the Nans
# #Logic if the missingness is more than one day then copy paste the previous data behavior

# datahouse.fillna(datahouse.mean(), inplace=True)
# #datahousepro = datahouse.interpolate() Interpolation doesnt change 

# #Resample every 15 min
# Resampled = datahouse.resample('15T').first()
# #save the processed 
# Resampled.to_csv (r'..\Resampled15min.csv', sep=';', index = True, header=True)
# #Resampled = pd.DataFrame()
# #Resampled = datahouse.resample('15T').mean()
# #Resampled.to_csv (r'..\Resampled.csv', sep=';', index = True, header=True)
# df = Resampled.copy()
# df = test.copy()
# df = df[1000:5000] # 11.02 - 25.03
# df = df[0:27880] # 18.11.2019 
# df = df[7050:7800] # 16.04 - 23.04 data window to check the data, the whole year is too detail 

# '''**********************************************************************************************************************************'''
# ''' Switches'''
# fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)

# ax1.plot(df.index, df['PuSolarKoll'], 'darkorange')
# ax1.set_ylabel('PuSolarKoll')
# ax1.set_xlim([df.index.min(), df.index.max()])
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax1.grid(True)

# ax2.plot(df.index, df['PuSolar'], 'r')
# ax2.set_ylabel('PuSolar')
# ax2.set_xlim([df.index.min(), df.index.max()])
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax2.grid(True)

# ax3.plot(df.index, df['GSolarPufferPool'], 'g')
# ax3.set_ylabel('GSolarPufferPool')
# ax3.set_xlim([df.index.min(), df.index.max()])
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax3.grid(True)

# ax4.plot(df.index, df['PuPool'], 'c', alpha=0.6, linestyle='--')
# ax4.set_ylabel('PuPool')
# ax4.set_xlim([df.index.min(), df.index.max()])
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax4.grid(True)

# ax5.plot(df.index, df['PuPufferPool'], 'black', alpha=0.6, linestyle='--' )
# ax5.set_ylabel('PuPufferPool')
# ax5.set_xlim([df.index.min(), df.index.max()])
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax5.grid(True)

# ax6.plot(df.index, df['PuHolzkessel'], 'b', alpha=0.6, linestyle='--')
# ax6.set_ylabel('PuHolzkessel')
# ax6.set_xlim([df.index.min(), df.index.max()])
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax6.grid(True)

# fig.suptitle('Switches')
# plt.show()
'''**********************************************************************************************************************************'''
# ''' Solar'''

# df=df[300000:370000]
# fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)

# ax1.plot(df.index, df['SGlobal'], 'darkorange')
# ax11 = ax1.twinx()
# ax11.plot(df.index, df['PuSolarKoll'], color='#006656', label='Pump Collector')
# ax11.plot(df.index, df['PuSolar'], color='#00FFFF', linestyle='-.')
# ax11.plot(df.index, df['GSolarPufferPool'], 'b', alpha=0.6, linestyle='-.', label ='Collector to Pool' )
# ax11.set_ylabel('Switches')
# ax11.legend(['Pump Solar Collector', 'Pump Solar', 'Switch Collector to Puffer/Pool'], loc=0,fontsize = 'small')
# ax1.set_xlim([df.index.min(), df.index.max()])
# ax1.set_ylabel('Solar Global \n Radiation [W/m2]')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax1.grid(True)

# ax2.plot(df.index, df['TSolarVLpri'], 'r')
# ax2.plot(df.index, df['TSolarRLpri'], 'b')
# ax2.plot(df.index, df['TSolarVL'], 'darkorange', linestyle='--')
# ax2.plot(df.index, df['TSolarRL'], 'c', linestyle='--')
# ax2.set_ylabel('Solar Collector [°C]')
# ax2.legend(['Supply Temp Primary Side (after HEX)', 'Return Temp Primary Side (after HEX)', 'Supply Temp Secondary Side (before HEX)', 'Return Temp Secondary Side (before HEX)'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax2.grid(True)

# ax3.plot(df.index, df['PSolar'], 'b')
# ax33 = ax3.twinx()
# ax33.plot(df.index, df['VDSolar'], 'black', alpha=0.6, label ='Volume flow rate Solar')
# ax33.legend(loc=0)
# ax33.set_ylabel('Volume flow [l/hr]')
# ax3.set_ylabel('Power Solar [kW]')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax3.grid(True)

# ax4.plot(df.index, df['TSchwimmbadRaum'], color='red')
# ax44 = ax4.twinx()
# ax44.plot(df.index, df['VDSchwimmbad'], 'black', alpha=0.6, label ='Volume flow to Pool')
# ax44.legend(loc=0)
# ax44.set_ylabel('Volume flow [l/hr]')
# ax4.set_xlim([df.index.min(), df.index.max()])
# ax4.set_ylabel('Temperature \n at Pool [°C]')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax4.grid(True)

# ax5.plot(df.index, df['TPuffero'], 'r')
# ax5.plot(df.index, df['TPuffermo'], 'darkorange', linestyle='--')
# ax5.plot(df.index, df['TPuffermu'], 'b')
# ax5.plot(df.index, df['TPufferu'], 'c', linestyle='--')
# ax55 = ax5.twinx()
# ax55.plot(df.index, df['PuPufferPool'], 'black', alpha=0.6, label='Pump  Puffer to Pool')
# ax55.legend(loc=0)
# ax55.set_ylabel('Storage to pool')
# ax5.set_ylabel('Temperature \n Levels in \n Tank [°C]')
# ax5.legend(['Top (Oben)', 'Middle Top', 'Middle Bottom', 'Bottom'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax5.grid(True)

# Psolarbuffer = df['PSolar']*np.logical_xor(df['GSolarPufferPool'],1).astype(int)
# Psolarpool = df['PSolar']*df['GSolarPufferPool']*(df['TSolarVL']>df['TSolarRL'])
# Psolarbufferrest = (df['PSolar']*df['GSolarPufferPool'] - (df['TSolarVL']-df['TSchwimmbadRL'])*df['VDSchwimmbad']*1.16/1000).clip(lower=0)

# ax6.plot(df.index, Psolarbuffer, 'b')
# ax6.plot(df.index, Psolarpool, 'green')
# ax6.plot(df.index, Psolarbufferrest, 'cyan')
# ax6.plot(df.index, df['PSolar'], 'orange', alpha=0.7, linestyle='-.')
# ax6.legend(['Psolarbuffer', 'Psolarpool', 'Psolarbufferrest', 'PSolar'], loc='upper right',fontsize = 'small')
# ax6.set_ylabel('Power \n Distribution')
# ax6.set_xlim([df.index.min(), df.index.max()])
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax6.grid(True)

# #fig.tight_layout()
# fig.suptitle('Solar Part')
# plt.show()

# #When the switch is 0 or 1, what are the max and min number of the Solar Global Radiation
# df.SGlobal.loc[df.groupby('PuSolarKoll').SGlobal.idxmax()]
# df.SGlobal.loc[df.groupby('PuSolarKoll').SGlobal.idxmax()]

# '''**********************************************************************************************************************************'''
# ''' Pool'''
# fig, (ax1, ax4, ax5, ax6) = plt.subplots(4, sharex=True)

# ax1.plot(df.index, df['GSolarPufferPool'], 'blue')
# ax1.plot(df.index, df['PuPool'], 'darkorange', linestyle='--')
# ax11 = ax1.twinx()
# ax11.plot(df.index, df['PuPufferPool'], 'green', linestyle='--', label ='Pump Storage to Pool')
# ax11.legend(loc=0)
# ax1.legend(['Switch Collector to Puffer/Pool', 'Pump Pool'], loc=0, fontsize = 'small')
# ax1.set_xlim([df.index.min(), df.index.max()])
# ax1.set_ylabel('Pool')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax1.grid(True)

# ax4.plot(df.index, df['TSchwimmbadVL'], color='red')
# ax4.plot(df.index, df['TSchwimmbadRL'], color='blue')
# ax4.legend(['Supply', 'Return'], loc=0, fontsize = 'small')
# ax44 = ax4.twinx()
# ax44.plot(df.index, df['VDSchwimmbad'], 'black', alpha=0.6, label ='Volume flow to Pool')
# ax44.legend(loc=0)
# ax44.set_ylabel('Volume flow [l/hr]')
# ax4.set_xlim([df.index.min(), df.index.max()])
# ax4.set_ylabel('Temperature \n Pool [°C]')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax4.grid(True)

# ax5.plot(df.index, df['TPuffero'], 'r')
# ax5.plot(df.index, df['TPuffermo'], 'darkorange', linestyle='--')
# ax5.plot(df.index, df['TPuffermu'], 'b', linestyle='--')
# ax5.plot(df.index, df['TPufferu'], 'c')
# ax5.set_ylabel('Temperature \n Levels in \n Tank [°C]')
# ax5.legend(['Top', 'Middle Top', 'Middle Bottom', 'Bottom'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax5.grid(True)

# Psolarbuffer = df['PSolar']*np.logical_xor(df['GSolarPufferPool'],1).astype(int)
# Psolarpool = df['PSolar']*df['GSolarPufferPool']*(df['TSolarVL']>df['TSolarRL'])
# Psolarbufferrest = (df['PSolar']*df['GSolarPufferPool'] - (df['TSolarVL']-df['TSchwimmbadRL'])*df['VDSchwimmbad']*1.16/1000).clip(lower=0)

# ax6.plot(df.index, Psolarbuffer, 'b')
# ax6.plot(df.index, Psolarpool, 'green')
# ax6.plot(df.index, Psolarbufferrest, 'cyan')
# ax6.legend(['Psolarbuffer', 'Psolarpool', 'Psolarbufferrest'], loc='upper right',fontsize = 'small')
# ax6.set_ylabel('Power \n Distribution')
# ax6.set_xlim([df.index.min(), df.index.max()])
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax6.grid(True)

# fig.tight_layout()
# fig.suptitle('Pool Part')
# plt.show()

# '''**********************************************************************************************************************************'''
# '''HeatPump'''
# fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

# ax1.plot(df.index, df['TWPVL'], 'r')
# ax1.plot(df.index, df['TWPRL'], 'b', linestyle='--')
# ax1.set_ylabel('Temperature [°C]')
# ax1.legend(['Supply Temp', 'Return Temp'], loc='upper left',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax1.grid(True)

# ax2.plot(df.index, df['PWP'], 'r')
# ax22 = ax2.twinx()
# ax22.plot(df.index, df['VDWP'], 'black', alpha=0.6, label ='Volume flow rate HP' )
# ax22.legend(loc='upper right')
# ax22.set_ylabel('[kg/s]')
# ax2.set_ylabel('[W]')
# ax2.legend(['Power HeatPump', 'Flow Rate HeatPump'], loc='upper left')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax2.grid(True)

# ax3.plot(df.index, df['QStromWP'], 'darkorange', alpha=0.6)
# ax3.set_xlabel('Date')
# ax3.set_ylabel('[W]')
# ax3.set_xlim([df.index.min(), df.index.max()])
# ax3.legend(['Electricity input of HP'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax3.grid(True)

# fig.suptitle('HeatPump Part')
# plt.show()

# '''**********************************************************************************************************************************'''
# '''Storage'''
# fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

# ax1.plot(df.index, df['TPuffero'], 'r')
# ax1.plot(df.index, df['TPuffermo'], 'darkorange', linestyle='--')
# ax1.plot(df.index, df['TPuffermu'], 'b')
# ax1.plot(df.index, df['TPufferu'], 'c', linestyle='--')
# ax1.set_ylabel('Temperature Levels in Tank [°C]')
# ax1.legend(['Top (Oben)', 'Middle Top', 'Middle Bottom', 'Bottom'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax1.grid(True)

# ax2.plot(df.index, df['TVLOfenn'], 'darkorange', alpha=0.6)
# ax2.plot(df.index, df['TOfenVL'], 'r', alpha=0.6)
# ax2.plot(df.index, df['TOfenRL'], 'b')
# ax2.set_ylabel('[°C]')
# ax2.set_xlim([df.index.min(), df.index.max()])
# ax2.legend(['Furnace supply temperature (inlet stroge after mix)', 'Furnace supply temperature (before mix)', 'Furnace return temperature (after mix)'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax2.grid(True)

# ax3.plot(df.index, df['PuPufferPool'], 'r')
# ax33 = ax3.twinx()
# ax33.plot(df.index, df['GSolarPufferPool'], 'black', alpha=0.6, label ='Switch Solar Puffer to Pool' )
# ax33.legend(loc='upper right')
# ax33.set_ylabel('Switch')
# ax3.set_xlabel('Date')
# ax3.set_ylabel('[W]')
# ax3.legend(['Pump Storage-Pool', 'Flow Rate HeatPump'], loc='upper left',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax3.grid(True)

# fig.suptitle('Storage Part')
# plt.show()

# '''**********************************************************************************************************************************'''
# '''House - Floor Heating'''
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)

# ax1.plot(df.index, df['TFBHVLn'], 'r')
# ax1.plot(df.index, df['TFBHRLn'], 'b')
# ax1.plot(df.index, df['TFBHVL'], 'r', linestyle='--')
# ax1.plot(df.index, df['TFBHRL'], 'b', linestyle='--')
# ax1.set_ylabel('[°C]')
# ax1.legend(['Floor Heating Supply (after mixing valve)', 'Floor Heating Return (after mixing valve)','Floor Heating Supply (before mixing valve) Tpo-TPmo', 'Floor Heating Return (before mixing valve) TPu'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax1.grid(True)

# ax3.plot(df.index, df['PuFBH'], 'r')
# ax3.set_ylabel('[W]')
# ax3.legend(['Pump Floow Heating'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax3.grid(True)

# ax4.plot(df.index, df['GSommergarten'], 'black', alpha=0.6, label='Summergarden')
# ax4.plot(df.index, df['GEssen'], 'darkorange', alpha=0.6, label='Kitchen')
# ax4.plot(df.index, df['GWohnen'], 'black', alpha=0.6, label='Swimming pool room')
# ax4.plot(df.index, df['GSchlafzimmer'], 'darkorange', alpha=0.6, label='Bedroom')
# ax4.plot(df.index, df['GArbeitszimmer'], 'black', alpha=0.6, label='Workingroom')
# ax4.plot(df.index, df['GKinderzimmer'], 'darkorange', alpha=0.6, label='Childrenroom')
# ax4.plot(df.index, df['GGarage'], 'darkorange', alpha=0.6, label='Garage')
# ax4.set_xlim([df.index.min(), df.index.max()])
# ax4.set_xlabel('Date')
# ax4.set_ylabel('Temperature (C)')
# ax4.legend(loc=0)
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax4.grid(True)

# ax5.plot(df.index, df['TSommergarten'], 'black', alpha=0.6, label='Summergarden')
# ax5.plot(df.index, df['TEssen'], 'darkorange', alpha=0.6, label='Kitchen')
# ax5.plot(df.index, df['TSchwimmbadRaum'], 'black', alpha=0.6, label='Swimming pool room')
# ax5.plot(df.index, df['TSchlafzimmer'], 'darkorange', alpha=0.6, label='Bedroom')
# ax5.plot(df.index, df['TArbeitszimmer'], 'black', alpha=0.6, label='Workingroom')
# ax5.plot(df.index, df['TKinderzimmer'], 'darkorange', alpha=0.6, label='Childrenroom')
# ax5.plot(df.index, df['TGarage'], 'darkorange', alpha=0.6, label='Garage')
# ax5.set_xlim([df.index.min(), df.index.max()])
# ax5.set_xlabel('Date')
# ax5.set_ylabel('Temperature (C)')
# ax5.legend(loc=0)
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax5.grid(True)

# fig.suptitle('House - Floor Heating')
# plt.show()

# '''**********************************************************************************************************************************'''
# '''House - Hot Water'''
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)

# ax1.plot(df.index, df['TFBHVLn'], 'r')
# ax1.plot(df.index, df['TFBHRLn'], 'b')
# ax1.set_ylabel('[°C]')
# ax1.legend(['Floor Heating Supply (after mixing valve)', 'Floor Heating Return (after mixing valve)'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax1.grid(True)

# ax2.plot(df.index, df['TFBHVL'], 'r')
# ax2.plot(df.index, df['TFBHRL'], 'b')
# ax2.set_ylabel('[°C]')
# ax2.legend(['Floor Heating Supply (before mixing valve) Tpo-TPmo', 'Floor Heating Return (before mixing valve) TPu'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax2.grid(True)

# ax3.plot(df.index, df['PuFBH'], 'r')
# ax33 = ax3.twinx()
# ax33.plot(df.index, df['PFBH'], 'black', alpha=0.6, label ='Switch Floor Heating' )
# ax33.legend(loc=0)
# ax33.set_ylabel('Power')
# ax3.set_ylabel('[W]')
# ax3.legend(['Pump Floow Heating'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax3.grid(True)

# ax4.plot(df.index, df['QStromHaus'], 'darkorange')
# ax4.set_xlim([df.index.min(), df.index.max()])
# ax4.set_ylabel('[W]')
# ax4.legend(['Electrictiy of House'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax4.grid(True)

# ax5.plot(df.index, df['PuSolarKoll'], 'black', alpha=0.6)
# ax5.plot(df.index, df['PuSolar'], 'darkorange', alpha=0.6)
# ax55 = ax5.twinx()
# ax55.plot(df.index, df['GSolarPufferPool'], 'black', alpha=0.6, label ='Switch Collector to Pool' )
# ax55.legend(loc=0)
# ax55.set_ylabel('Switch')
# ax5.set_xlim([df.index.min(), df.index.max()])
# ax5.set_xlabel('Date')
# ax5.set_ylabel('Pump switches')
# ax5.legend(['Pump Solar Collector', 'Pump solar'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax5.grid(True)

# fig.suptitle('House - Hot Water')
# plt.show()