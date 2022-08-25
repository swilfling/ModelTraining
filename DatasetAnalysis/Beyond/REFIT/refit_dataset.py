import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":
    data = pd.read_csv("../../../Data/Data/Beyond/REFIT/REFIT_TIME_SERIES_VALUES.csv")
    tb20lr = data["data"][data[data.columns[0]]=="TimeSeriesVariable257"]
    tb20rad243_1 = data["data"][data[data.columns[0]] == "TimeSeriesVariable1177"]

    plt.figure(figsize=(20,7))
    plt.plot(tb20rad243_1.values)
    plt.plot(tb20lr.values)
    plt.ylabel("Temperature [°C]")
    plt.legend(["Radiator temperature","indoor temperature"])
    plt.show()