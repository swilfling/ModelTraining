from ModelTraining.Preprocessing.DataImport import data_import
import os
import random
import numpy as np
import matplotlib

def create_noise_serieses_gaussian(df, mean=0.0, sigma=1.0, size=[100, 1]):
    noisy_list = []
    for sample in enumerate(df):
        series = np.random.normal(mean, sigma, size)
        noisy_sample = [(sample + el)[1] for el in series]
        noisy_list.append(noisy_sample)

    return np.stack(noisy_list, axis=1)


def change_range(df):
    noisy_list = []
    mean = 1
    size = [100, 1]
    import random
    sigma = 1
    range = (df.max() - df.min()) / 10
    for sample in enumerate(df):
        series = np.random.normal(mean, sigma, size)
        noisy_sample = [(sample + el)[1] for el in series]
        # noisy_sample = [(sample+ (el * int(range)))[1] for el in series]
        noisy_list.append(noisy_sample)
    return np.stack(noisy_list, axis=1)


def plot_robust_interval(df, clean_df, model_type, expansion_type, result_path, y_true):
    lower, upper, mean_list = [], [], []
    for index, row in df.iterrows():
        lower.append(min(row))
        upper.append(max(row))
        mean_list.append(np.mean(row))
    import matplotlib.pyplot as plt
    # clean = clean_df.tolist()
    clean = clean_df.loc[:, "predicted_energy"]
    x = df.iloc[0:300].index
    y = mean_list[0:300]

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Mean of Robustness interval")

    ax.plot(clean.iloc[0:300], linestyle="--", linewidth=1, color="r", label="Clean data")
    ax.plot(x, y_true[0:300], linestyle="--", linewidth=1, color="y", label="True data")
    ax.set_ylim([5, 75])
    # ax.plot(x,clean, linestyle="--", linewidth=1, color="r", label="Clean data" )
    ax.fill_between(x, lower[0:300], upper[0:300], color='b', alpha=0.1)
    # ax.fill_between(x, lower, upper, color='b', alpha=0.1)
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Predictions", fontsize=10)
    matplotlib.rcParams['legend.fontsize'] = 10
    figname = f"Robustness interval for model {model_type} with {expansion_type}"
    plt.title(figname, fontsize=10)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(result_path, f"{figname}.png"))
