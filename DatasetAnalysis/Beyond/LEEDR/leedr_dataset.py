from ModelTraining.Data.DataImport.dataimport import DataImport
import matplotlib.pyplot as plt
from ModelTraining.Data.Plotting.plot_distributions import plot_missing_values
import numpy as np


if __name__ == '__main__':
    # %%
    root_dir = "../"
    data_dir = "Beyond/LEEDR"
    filename = "processed-H01-Accounts-3-31-temperature"
    data_import = DataImport.load(f"../../../Data/Configuration/DataImport/{data_dir}/{filename}.json")
    data = data_import.import_data(f"../../../Data/Data/{data_dir}/LEEDR_data_minute_temperature/{filename}-CLEAN.txt")

    print(data.columns)
    plt.figure()
    plt.plot(data["\"Loughborough31,LBORO-DOR-431,00-0D-6F-00-00-C1-2F-64,Outdoor(Boiler)\""])
    plt.show()

    # missing values
    data[data == -99] = np.nan
    plot_missing_values(data, "./Figures", "missing_vals")


