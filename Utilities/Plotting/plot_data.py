"""
Plotting functions
    Single plot
    - plot_data (uses pandas dataframe)
    - plot_training_results (uses TrainingResult structure)
    Subplots
    - plot_df_subplots (plots content of dataframe as subplots)
    - plot_subplots (plots list of dataframes as subplots)
"""

from typing import List
import pandas as pd
import os
from matplotlib import pyplot as plt
from . import utils as plt_utils
from ..Parameters import TrainingResults


############################# Single plot ##############################################################################


def plot_training_results(result: TrainingResults, path="./", filename="Result", fig_title: str="Result", fmt="png"):
    """
    Plot training results
    @param result: TrainingResults structure
    @param path: Output dir
    @param title: filename
    """
    day = 4 * 24
    int = (0 + day * 10, 0 + day * 12)
    print("starting", int[0])
    print(int[1])
    print('Plotting results in Interval:', result.test_index[int[0]], result.test_index[int[1]])
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.style.use('classic')
    ax.plot(result.test_index[int[0]:int[1]], result.test_target[int[0]:int[1]], color='blue', label='true')
    ax.plot(result.test_index[int[0]:int[1]], result.test_prediction[int[0]:int[1]], color='orange', label='prediction')
    ax.grid(linestyle="--", alpha=0.6, color='gray')
    ax.set_xlim([result.test_index[int[0]], result.test_index[int[1]]])
    ax.legend(loc='upper left', frameon=True)
    ax.set_ylabel('Temperature (°C)')
    fig.suptitle(fig_title, y=0.95)
    with open(os.path.join(path, f'{filename}.{fmt}'), "wb") as fp:
        plt.savefig(fp, dpi=300)
    plt.close()


def plot_data(data: pd.DataFrame, plot_path="./", filename='Result', store_to_csv=True, **kwargs):
    """
    Plot dataframe, store values to csv (optional)
    @param data: dataframe
    @param plot_path: Output path
    @param filename: filename
    @param store_to_csv: store results in csv vile
    Optional params: ylabel, figsize, fig_title, plot args
    """
    if data is not None:
        fig, ax, = plt_utils.create_figure(kwargs.pop('fig_title'), figsize=kwargs.pop('figsize', None))
        plt.xlabel('Time')
        if kwargs.get('ylim', None):
            plt.ylim(kwargs.pop('ylim'))
        if kwargs.get('ylabel', None):
            plt.ylabel(kwargs.pop('ylabel'))
        plt_utils.plot_df(ax, data, **kwargs)
        plt_utils.save_figure(plot_path, filename)
        if store_to_csv:
            data.to_csv(os.path.join(plot_path, f'{filename}.csv'))


############################ Subplots ##################################################################################


def plot_df_subplots(df, path="./", filename="Plots", **kwargs):
    """
    Plot subplots - take list of dataframes
    @param df: pd.Dataframe
    @param path: Directory to store to
    @param filename: filename
    Optional args: figsize, store_tikz, fig_title
    """
    store_tikz = kwargs.pop('store_tikz', False)
    fig, ax = plt_utils.create_figure(kwargs.pop('fig_title', ""), figsize=kwargs.pop('figsize', None))
    df.plot(subplots=True, ax=ax)
    plt_utils.save_figure(path, filename, store_tikz=store_tikz)
    plt.show()


def plt_subplots(list_df: List[List[pd.DataFrame]], path="./", filename="Plots", **kwargs):
    """
    Plot subplots - take list of dataframes or list of lists of dataframes (if twinx)
    @param list_df: List of pd.Dataframes
    @param path: Directory to store to
    @param filename: filename
    Optional args: figsize, ylim, store_tikz, fig_title
    """
    store_tikz = kwargs.pop('store_tikz', False)
    fig, ax = plt_utils.create_figure(kwargs.pop('fig_title', ""), figsize=kwargs.pop('figsize', None))
    if kwargs.get('ylim',None):
        plt.ylim(kwargs.pop('ylim'))
    for index, data in enumerate(list_df):
        ax = plt.subplot(len(list_df), 1, index + 1)
        plt.grid('on')
        # Twinx plot - two axes
        if type(data) == list:
            plt_utils.plot_df(ax, data[0])
            # Second axis
            if len(data) > 1:
                ax2 = plt.twinx()
                plt_utils.plot_df(ax2, data[1], **kwargs)
        # Single x axis plot
        elif type(data) == pd.DataFrame:
            plt_utils.plot_df(ax, data, **kwargs)
    plt_utils.save_figure(path, filename, store_tikz=store_tikz)
    plt.show()