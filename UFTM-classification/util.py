import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#TODO add threshold line formatting


def dist_bar_plot(df, metric, cutoff_line, name='dist_bar_plot'):
    plt.clf()
    bins = np.arange(0,5,0.5)
    if cutoff_line: # put earlier (after bins = ) create df['pop'].hist(label=...)
        df[metric].hist(grid=False, bins=bins, label=metric)
        plt.vlines(x=cutoff_line, ymin=0, ymax=max(df[metric]), \
        linestyles='dashed', colors='black', label = "Threshold")
        plt.legend()
    else:
        df[metric].hist(grid=False, bins=bins)
    plt.xticks(bins[::25])
    plt.ylabel("Number of news sources")
    plt.xlabel(metric)
    plt.title("News Source " + metric)
    plt.savefig(name + '.png')


def dist_curve_plot(df, metric, cutoff_line, name='dist_curve_plot'):
    plt.clf()
    df = df.sort_values(by=[metric])
    if cutoff_line:
        plt.plot(range(len(df)), df[metric], label = metric)
        plt.vlines(x=cutoff_line, ymin=0, ymax=max(df[metric]), \
        linestyles='dashed', colors='black', label = "Threshold")
        plt.legend()
    else:
        plt.plot(range(len(df)), df[metric])
    plt.xlabel("Number of news sources")
    plt.ylabel(metric)
    plt.title("News Source " + metric)
    plt.savefig(name + '.png')


def joint_plot(df, name):
    '''
    Specific to 2 variable plots. In this case, popularity score and 
    trustworthiness score
    '''
    plt.clf()
    sns.jointplot(x="trust_score", y="pop_score", edgecolor="white", data=df)
    plt.ylim(top=10)
    plt.savefig(name + '.png')


