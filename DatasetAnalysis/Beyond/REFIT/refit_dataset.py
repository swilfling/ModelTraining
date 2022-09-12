import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dir = "../../../Data/Data/Beyond/REFIT"
    data = pd.read_csv(f"{dir}/REFIT_TIME_SERIES_VALUES.csv")

    columns = {"TB20LR":"TimeSeriesVariable257",
               #"TB20LR_2": "TimeSeriesVariable258",
               "lumB20LR":"TimeSeriesVariable516",
               "rhB20LR":"TimeSeriesVariable672",
               #"rhB20LR_2":"TimeSeriesVariable673",
               "TB20RAD243":"TimeSeriesVariable1177",
               #"TB20RAD243_2":"TimeSeriesVariable1178",
               #"TB20RAD243_3":"TimeSeriesVariable1179",
               "TB20RAD244": "TimeSeriesVariable1180",
               "TAmbient": "TimeSeriesVariable1573",
               "humidity": "TimeSeriesVariable1574",
               "vWind":"TimeSeriesVariable1575",
               "SHorizontal":"TimeSeriesVariable1578",
               "rain": "TimeSeriesVariable1582"
               #"TB20RAD244_2": "TimeSeriesVariable1181",
               #"TB20RAD244_3": "TimeSeriesVariable1182",
               }

    df = pd.DataFrame()
    for col, var in columns.items():
        tb20lr = data[["data","dateTime"]][data[data.columns[0]] == var]
        tb20lr = tb20lr.set_index(pd.to_datetime(tb20lr["dateTime"]))
        tb20lr = tb20lr.drop("dateTime",axis=1)
        tb20lr = tb20lr.rename({"data": col}, axis=1)
        if df.empty:
            df = tb20lr
        else:
            df = df.join(tb20lr, how="outer")

    df = df.resample("30min").first()
    df = df.dropna(axis=0)

    df.to_csv(f"{dir}/B20/B20_Indoor.csv", index_label="dt")


    #df_additional = pd.read_csv(f"{dir}/B20/Beyond_B20_full.csv", index_col="dt", infer_datetime_format=True)
    #df = df_additional.join(df, how="outer", rsuffix="_indoor")

    plt.figure(figsize=(20, 7))
    plt.plot(df["TB20LR"])
    plt.plot(df["TB20RAD243"])

    plt.ylabel("Temperature [°C]")
    plt.legend(df.columns)
    plt.show()

    plt.figure(figsize=(20, 7))
    plt.plot(df["TB20RAD243"] - df["TB20LR"])
    plt.ylabel("Temperature [°C]")
    plt.legend(["Temperature Difference - Radiator vs Indoor"])
    plt.show()