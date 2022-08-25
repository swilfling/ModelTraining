# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:32:56 2021

@author: Basak
"""
import matplotlib.pyplot as plt

from Utilities.DataProcessing.data_import import import_data
import Utilities.file_utilities as file_utils
import os
save_figures = True

# File paths

hybridcosim_path = os.environ.get("HYBRIDCOSIM_REPO_PATH", "../../")
csv_file_path = os.path.join(hybridcosim_path, "Data/AEE/Resampled15min.csv")
fig_save_path = os.path.join(hybridcosim_path, "DataAnalysis","DataProcessingAEE","Solarhouse1", 'Figures')
file_utils.create_dir(fig_save_path)
df = import_data(csv_file_path)

df = df[7050:7800] # 16.04 - 23.04 data window to check the data, the whole year is too detail 

'''**********************************************************************************************************************************'''
''' Switches'''
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, figsize= (8,8))

plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.0)

ax1.plot(df.index, df['PuSolarKoll'], 'darkorange')
ax1.set_ylabel('PuSolarKoll', fontsize=8)
ax1.set_xlim([df.index.min(), df.index.max()])
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax1.grid(True)

ax2.plot(df.index, df['PuSolar'], 'r')
ax2.set_ylabel('PuSolar', fontsize=8)
ax2.set_xlim([df.index.min(), df.index.max()])
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax2.grid(True)

ax3.plot(df.index, df['GSolarPufferPool'], 'g')
ax3.set_ylabel('GSolarPufferPool', fontsize=8)
ax3.set_xlim([df.index.min(), df.index.max()])
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax3.grid(True)

ax4.plot(df.index, df['PuPool'], 'c', alpha=0.6, linestyle='--')
ax4.set_ylabel('PuPool', fontsize=8)
ax4.set_xlim([df.index.min(), df.index.max()])
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax4.grid(True)

ax5.plot(df.index, df['PuPufferPool'], 'black', alpha=0.6, linestyle='--' )
ax5.set_ylabel('PuPufferPool', fontsize=8)
ax5.set_xlim([df.index.min(), df.index.max()])
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax5.grid(True)

ax6.plot(df.index, df['PuHolzkessel'], 'b', alpha=0.6, linestyle='--')
ax6.set_ylabel('PuHolzkessel', fontsize=8)
ax6.set_xlim([df.index.min(), df.index.max()])
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax6.grid(True)

fig_title = 'Switches'
fig.suptitle(fig_title)
if save_figures:
    plt.savefig(fig_save_path + fig_title + '.png', format='png')
plt.show()
'''**********************************************************************************************************************************'''
''' Solar'''
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, figsize= (12,12))

plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.0)

ax1.plot(df.index, df['SGlobal'], 'darkorange')
ax11 = ax1.twinx()
ax11.plot(df.index, df['PuSolarKoll'], color='#006656', label='Pump Collector')
ax11.plot(df.index, df['PuSolar'], color='#00FFFF', linestyle='-.')
ax11.plot(df.index, df['GSolarPufferPool'], 'b', alpha=0.6, linestyle='-.', label ='Collector to Pool' )
ax11.set_ylabel('Switches', fontsize=8)
ax11.legend(['Pump Solar Collector', 'Pump Solar', 'Switch Collector to Puffer/Pool'], loc=0,fontsize = 'small')
ax1.set_xlim([df.index.min(), df.index.max()])
ax1.set_ylabel('Solar Global \n Radiation [W/m2]')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax1.grid(True)

ax2.plot(df.index, df['TSolarVLpri'], 'r')
ax2.plot(df.index, df['TSolarRLpri'], 'b')
ax2.plot(df.index, df['TSolarVL'], 'darkorange', linestyle='--')
ax2.plot(df.index, df['TSolarRL'], 'c', linestyle='--')
ax2.set_ylabel('Solar Collector [°C]')
ax2.legend(['Supply Temp Primary Side (after HEX)', 'Return Temp Primary Side (after HEX)', 'Supply Temp Secondary Side (before HEX)', 'Return Temp Secondary Side (before HEX)'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax2.grid(True)

ax3.plot(df.index, df['PSolar'], 'b')
ax33 = ax3.twinx()
ax33.plot(df.index, df['VDSolar'], 'black', alpha=0.6, label ='Volume flow rate Solar')
ax33.legend(loc=0)
ax33.set_ylabel('Volume flow [l/hr]', fontsize=8)
ax3.set_ylabel('Power Solar [kW]', fontsize=8)
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax3.grid(True)

ax4.plot(df.index, df['TSchwimmbadRaum'], color='red')
ax44 = ax4.twinx()
ax44.plot(df.index, df['VDSchwimmbad'], 'black', alpha=0.6, label ='Volume flow to Pool')
ax44.legend(loc=0)
ax44.set_ylabel('Volume flow [l/hr]')
ax4.set_xlim([df.index.min(), df.index.max()])
ax4.set_ylabel('Temperature \n at Pool [°C]', fontsize=8)
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax4.grid(True)

ax5.plot(df.index, df['TPuffero'], 'r')
ax5.plot(df.index, df['TPuffermo'], 'darkorange', linestyle='--')
ax5.plot(df.index, df['TPuffermu'], 'b')
ax5.plot(df.index, df['TPufferu'], 'c', linestyle='--')
ax55 = ax5.twinx()
ax55.plot(df.index, df['PuPufferPool'], 'black', alpha=0.6, label='Pump  Puffer to Pool')
ax55.legend(loc=0)
ax55.set_ylabel('Storage to pool')
ax5.set_ylabel('Temperature \n Levels in \n Tank [°C]', fontsize=8)
ax5.legend(['Top (Oben)', 'Middle Top', 'Middle Bottom', 'Bottom'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax5.grid(True)

Psolarbuffer = df['PSolar']*np.logical_xor(df['GSolarPufferPool'],1).astype(int)
Psolarpool = df['PSolar']*df['GSolarPufferPool']*(df['TSolarVL']>df['TSolarRL'])
Psolarbufferrest = (df['PSolar']*df['GSolarPufferPool'] - (df['TSolarVL']-df['TSchwimmbadRL'])*df['VDSchwimmbad']*1.16/1000).clip(lower=0)

ax6.plot(df.index, Psolarbuffer, 'b')
ax6.plot(df.index, Psolarpool, 'green')
ax6.plot(df.index, Psolarbufferrest, 'cyan')
ax6.plot(df.index, df['PSolar'], 'orange', alpha=0.7, linestyle='-.')
ax6.legend(['Psolarbuffer', 'Psolarpool', 'Psolarbufferrest', 'PSolar'], loc='upper right',fontsize = 'small')
ax6.set_ylabel('Power \n Distribution', fontsize=8)
ax6.set_xlim([df.index.min(), df.index.max()])
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax6.grid(True)

fig_title = 'Solar'
fig.suptitle(fig_title)
if save_figures:
    plt.savefig(fig_save_path + fig_title + '.png')
plt.show()

#When the switch is 0 or 1, what are the max and min number of the Solar Global Radiation
df.SGlobal.loc[df.groupby('PuSolarKoll').SGlobal.idxmax()]
df.SGlobal.loc[df.groupby('PuSolarKoll').SGlobal.idxmax()]

'''**********************************************************************************************************************************'''
''' Pool'''
fig, (ax1, ax4, ax5, ax6) = plt.subplots(4, sharex=True, figsize= (12,12))
plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.0)

ax1.plot(df.index, df['GSolarPufferPool'], 'blue')
ax1.plot(df.index, df['PuPool'], 'darkorange', linestyle='--')
ax11 = ax1.twinx()
ax11.plot(df.index, df['PuPufferPool'], 'green', linestyle='--', label ='Pump Storage to Pool')
ax11.legend(loc=0)
ax1.legend(['Switch Collector to Puffer/Pool', 'Pump Pool'], loc=0, fontsize = 'small')
ax1.set_xlim([df.index.min(), df.index.max()])
ax1.set_ylabel('Pool', fontsize='small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax1.grid(True)

ax4.plot(df.index, df['TSchwimmbadVL'], color='red')
ax4.plot(df.index, df['TSchwimmbadRL'], color='blue')
ax4.legend(['Supply', 'Return'], loc=0, fontsize = 'small')
ax44 = ax4.twinx()
ax44.plot(df.index, df['VDSchwimmbad'], 'black', alpha=0.6, label ='Volume flow to Pool')
ax44.legend(loc=0)
ax44.set_ylabel('Volume flow [l/hr]', fontsize='small')
ax4.set_xlim([df.index.min(), df.index.max()])
ax4.set_ylabel('Temperature \n Pool [°C]', fontsize='small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax4.grid(True)

ax5.plot(df.index, df['TPuffero'], 'r')
ax5.plot(df.index, df['TPuffermo'], 'darkorange', linestyle='--')
ax5.plot(df.index, df['TPuffermu'], 'b', linestyle='--')
ax5.plot(df.index, df['TPufferu'], 'c')
ax5.set_ylabel('Temperature \n Levels in \n Tank [°C]', fontsize=8)
ax5.legend(['Top', 'Middle Top', 'Middle Bottom', 'Bottom'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax5.grid(True)

Psolarbuffer = df['PSolar']*np.logical_xor(df['GSolarPufferPool'],1).astype(int)
Psolarpool = df['PSolar']*df['GSolarPufferPool']*(df['TSolarVL']>df['TSolarRL'])
Psolarbufferrest = (df['PSolar']*df['GSolarPufferPool'] - (df['TSolarVL']-df['TSchwimmbadRL'])*df['VDSchwimmbad']*1.16/1000).clip(lower=0)

ax6.plot(df.index, Psolarbuffer, 'b')
ax6.plot(df.index, Psolarpool, 'green')
ax6.plot(df.index, Psolarbufferrest, 'cyan')
ax6.legend(['Psolarbuffer', 'Psolarpool', 'Psolarbufferrest'], loc='upper right',fontsize = 'small')
ax6.set_ylabel('Power \n Distribution', fontsize='small')
ax6.set_xlim([df.index.min(), df.index.max()])
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax6.grid(True)

fig_title = 'Pool Part'
fig.suptitle(fig_title)
if save_figures:
    plt.savefig(fig_save_path + fig_title + '.png')
plt.show()

'''**********************************************************************************************************************************'''
'''HeatPump'''
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize= (8,8))
plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.0)

ax1.plot(df.index, df['TWPVL'], 'r')
ax1.plot(df.index, df['TWPRL'], 'b', linestyle='--')
ax1.set_ylabel('Temperature [°C]', fontsize=8)
ax1.legend(['Supply Temp', 'Return Temp'], loc='upper left',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax1.grid(True)

ax2.plot(df.index, df['PWP'], 'r')
ax22 = ax2.twinx()
ax22.plot(df.index, df['VDWP'], 'black', alpha=0.6, label ='Volume flow rate HP' )
ax22.legend(loc='upper right')
ax22.set_ylabel('[kg/s]')
ax2.set_ylabel('[W]', fontsize=8)
ax2.legend(['Power HeatPump', 'Flow Rate HeatPump'], loc='upper left')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax2.grid(True)

ax3.plot(df.index, df['QStromWP'], 'darkorange', alpha=0.6)
ax3.set_xlabel('Date')
ax3.set_ylabel('[W]', fontsize=8)
ax3.set_xlim([df.index.min(), df.index.max()])
ax3.legend(['Electricity input of HP'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax3.grid(True)

fig_title = 'HeatPump Part'
fig.suptitle(fig_title)
if save_figures:
    plt.savefig(fig_save_path + fig_title + '.png')
plt.show()

'''**********************************************************************************************************************************'''
'''Storage'''
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize= (10,10))
plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.0)

ax1.plot(df.index, df['TPuffero'], 'r')
ax1.plot(df.index, df['TPuffermo'], 'darkorange', linestyle='--')
ax1.plot(df.index, df['TPuffermu'], 'b')
ax1.plot(df.index, df['TPufferu'], 'c', linestyle='--')
ax1.set_ylabel('Temperature Levels in Tank [°C]', fontsize=8)
ax1.legend(['Top (Oben)', 'Middle Top', 'Middle Bottom', 'Bottom'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax1.grid(True)

ax2.plot(df.index, df['TVLOfenn'], 'darkorange', alpha=0.6)
ax2.plot(df.index, df['TOfenVL'], 'r', alpha=0.6)
ax2.plot(df.index, df['TOfenRL'], 'b')
ax2.set_ylabel('[°C]', fontsize=8)
ax2.set_xlim([df.index.min(), df.index.max()])
ax2.legend(['Furnace supply temperature (inlet stroge after mix)', 'Furnace supply temperature (before mix)', 'Furnace return temperature (after mix)'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax2.grid(True)

ax3.plot(df.index, df['PuPufferPool'], 'r')
ax33 = ax3.twinx()
ax33.plot(df.index, df['GSolarPufferPool'], 'black', alpha=0.6, label ='Switch Solar Puffer to Pool' )
ax33.legend(loc='upper right')
ax33.set_ylabel('Switch')
ax3.set_xlabel('Date')
ax3.set_ylabel('[W]', fontsize=8)
ax3.legend(['Pump Storage-Pool', 'Flow Rate HeatPump'], loc='upper left',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax3.grid(True)

fig_title = 'Storage Part'
fig.suptitle(fig_title)
plt.savefig(fig_save_path + fig_title + '.png')
plt.show()

'''**********************************************************************************************************************************'''
'''House - Floor Heating'''
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize= (10,10))
plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.0)

ax1.plot(df.index, df['TFBHVLn'], 'r')
ax1.plot(df.index, df['TFBHRLn'], 'b')
ax1.set_ylabel('[°C]', fontsize=8)
ax1.legend(['Floor Heating Supply (after mixing valve)', 'Floor Heating Return (after mixing valve)'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax1.grid(True)

ax2.plot(df.index, df['TFBHVL'], 'r')
ax2.plot(df.index, df['TFBHRL'], 'b')
ax2.set_ylabel('[°C]', fontsize=8)
ax2.legend(['Floor Heating Supply (before mixing valve) Tpo-TPmo', 'Floor Heating Return (before mixing valve) TPu'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax2.grid(True)

ax3.plot(df.index, df['PuFBH'], 'r')
ax3.set_ylabel('[W]', fontsize=8)
ax3.legend(['Pump Floow Heating'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax3.grid(True)

ax4.plot(df.index, df['QStromHaus'], 'darkorange')
ax4.set_xlim([df.index.min(), df.index.max()])
ax4.set_ylabel('[W]', fontsize=8)
ax4.legend(['Electrictiy of House'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax4.grid(True)

ax5.plot(df.index, df['PuSolarKoll'], 'black', alpha=0.6)
ax5.plot(df.index, df['PuSolar'], 'darkorange', alpha=0.6)
ax55 = ax5.twinx()
ax55.plot(df.index, df['GSolarPufferPool'], 'black', alpha=0.6, label ='Switch Collector to Pool' )
ax55.legend(loc=0)
ax55.set_ylabel('Switch', fontsize=8)
ax5.set_xlim([df.index.min(), df.index.max()])
ax5.set_xlabel('Date')
ax5.set_ylabel('Pump switches', fontsize=8)
ax5.legend(['Pump Solar Collector', 'Pump solar'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax5.grid(True)


fig_title = 'House - Floor Heating'
fig.suptitle(fig_title)
if save_figures:
    plt.savefig(fig_save_path + fig_title + '.png')
plt.show()

'''**********************************************************************************************************************************'''
'''House - Hot Water'''
#fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize= (10,10))
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize= (10,10))
plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.0)
ax1.plot(df.index, df['Wasserzaehler'], 'r')
ax1.set_ylabel('[]', fontsize=8)
ax1.legend(['Water consumption'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax1.grid(True)

ax2.plot(df.index, df['QWW'], 'b')
ax2.set_ylabel('[]', fontsize=8)
ax2.legend(['Hot water flow'], loc='upper right',fontsize = 'small')
plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
ax2.grid(True)

# ax1.plot(df.index, df['TFBHVLn'], 'r')
# ax1.plot(df.index, df['TFBHRLn'], 'b')
# ax1.set_ylabel('[°C]', fontsize=8)
# ax1.legend(['Floor Heating Supply (after mixing valve)', 'Floor Heating Return (after mixing valve)'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax1.grid(True)

# ax2.plot(df.index, df['TFBHVL'], 'r')
# ax2.plot(df.index, df['TFBHRL'], 'b')
# ax2.set_ylabel('[°C]', fontsize=8)
# ax2.legend(['Floor Heating Supply (before mixing valve) Tpo-TPmo', 'Floor Heating Return (before mixing valve) TPu'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax2.grid(True)
#
# ax3.plot(df.index, df['PuFBH'], 'r')
# ax33 = ax3.twinx()
# ax33.plot(df.index, df['PFBH'], 'black', alpha=0.6, label ='Switch Floor Heating' )
# ax33.legend(loc=0)
# ax33.set_ylabel('Power', fontsize=8)
# ax3.set_ylabel('[W]')
# ax3.legend(['Pump Floow Heating'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax3.grid(True)
#
# ax4.plot(df.index, df['QStromHaus'], 'darkorange')
# ax4.set_xlim([df.index.min(), df.index.max()])
# ax4.set_ylabel('[W]', fontsize=8)
# ax4.legend(['Electrictiy of House'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax4.grid(True)
#
# ax5.plot(df.index, df['PuSolarKoll'], 'black', alpha=0.6)
# ax5.plot(df.index, df['PuSolar'], 'darkorange', alpha=0.6)
# ax55 = ax5.twinx()
# ax55.plot(df.index, df['GSolarPufferPool'], 'black', alpha=0.6, label ='Switch Collector to Pool' )
# ax55.legend(loc=0)
# ax55.set_ylabel('Switch', fontsize=8)
# ax5.set_xlim([df.index.min(), df.index.max()])
# ax5.set_xlabel('Date')
# ax5.set_ylabel('Pump switches', fontsize=8)
# ax5.legend(['Pump Solar Collector', 'Pump solar'], loc='upper right',fontsize = 'small')
# plt.rc('grid', linestyle="--", alpha=0.6, color='gray')
# ax5.grid(True)


fig_title = 'House - Hot Water'
fig.suptitle(fig_title)
if save_figures:
    plt.savefig(fig_save_path + fig_title + '.png')
plt.show()