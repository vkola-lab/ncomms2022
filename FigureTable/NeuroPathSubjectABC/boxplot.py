import matplotlib
import matplotlib.pyplot as plt
import csv
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.markers as mmarkers

from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', weight='bold', size=17)

pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * .5]
open_circle = matplotlib.path.Path(vert)
palette = ['green', 'blue', 'orange', 'red']
# palette = ['black'] * 4

def get_swarmplot_coordinate(thres, column):
    y_label = 'CNN_COG_score'
    df = pd.read_csv('../NeuroPathTable/ALL.csv')
    index_names = df[df['diff_days'] > thres * 365].index
    df.drop(index_names, inplace=True)
    value_to_dataname = {}
    for index, row in df.iterrows():
        value_to_dataname["{:.10f}".format(row[y_label])] = row['dataset']
    ax = sns.swarmplot(x=column, y=y_label, data=df, palette=palette, size=10)
    res = []
    for bin in range(4):
        bin_res = []
        for case in ax.collections[bin].get_offsets():
            key = "{:.10f}".format(case[1])
            if key in value_to_dataname:
                bin_res.append([case[0], case[1], value_to_dataname[key]])
            else:
                print(case, 'dataset not found')
        res.append(bin_res)
    print(len(res))
    plt.close()
    return res

def gen_plot_COG(thres, column):
    y_label = 'CNN_COG_score'
    df = pd.read_csv('../NeuroPathTable/ALL.csv')
    index_names = df[df['diff_days'] > thres * 365].index
    df.drop(index_names, inplace = True)
    print(df.shape[0])

    ans = get_swarmplot_coordinate(thres, column)

    ax = sns.boxplot(x=column, y=y_label, data=df, color='white', linewidth=2.5)
    for i, box in enumerate(ax.artists):
        box.set_edgecolor(palette[i])
        box.set_alpha(0.3)
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(palette[i])
            line.set_mfc(palette[i])
            line.set_mec(palette[i])
            line.set_alpha(0.3)

    size, lw = 45, 3
    for i in range(4):
        for case in ans[i]:
            if case[2] == 'ADNI':
                ax.scatter([case[0]], [case[1]], marker='o', edgecolors=palette[i], linewidths=lw, s=size, c='white', label='ADNI')
            elif case[2] == 'NACC':
                ax.scatter([case[0]], [case[1]], marker='s', edgecolors=palette[i], linewidths=lw, s=size-5, c='white', label='NACC')
            elif case[2] == 'FHS':
                ax.scatter([case[0]], [case[1]], marker='^', edgecolors=palette[i], linewidths=lw, s=size+2, c='white', label='FHS')

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='center left', bbox_to_anchor=(1, 0.5))

    sns.despine()
    ax.set_xlabel(column.replace('_', ' '), fontsize=20, fontweight='bold')
    ax.set_ylabel("DEMO score", fontsize=20, fontweight='bold')
    plt.savefig("boxplot/{}years_".format(thres) + column + '__' + y_label + ".png", bbox_inches='tight', dpi=300)
    plt.close()

def gen_plot_ADD(thres, column):
    y_label = 'CNN_ADD_prob'
    df = pd.read_csv('../NeuroPathTable/ALL.csv')
    index_names = df[df['diff_days'] > thres * 365].index
    df.drop(index_names, inplace = True)
    index_names = df[df['Group'] == 'NC'].index
    df.drop(index_names, inplace=True)
    index_names = df[df['Group'] == 'MCI'].index
    df.drop(index_names, inplace=True)

    df['ADD rank'] = df[y_label].rank()
    y_label = 'ADD rank'

    ax = sns.boxplot(x=column, y=y_label, data=df, color='white', linewidth=2.5)
    for i, box in enumerate(ax.artists):
        box.set_edgecolor(palette[i])
        box.set_alpha(0.3)
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(palette[i])
            line.set_mfc(palette[i])
            line.set_mec(palette[i])
            line.set_alpha(0.3)
    ax = sns.swarmplot(x=column, y=y_label, data=df, palette=palette,
                       marker=open_circle, size=6)
    sns.despine()

    ax.set_xlabel(column.replace('_', ' '), fontsize=15, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=15, fontweight='bold')
    plt.savefig("boxplot/{}years_".format(thres) + column + '__' + y_label + ".png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    for thres in [2]:
        gen_plot_COG(thres, 'C_score')
        gen_plot_COG(thres, 'B_score')
        gen_plot_COG(thres, 'A_score')
        # gen_plot_ADD(thres, 'C_score')
        # gen_plot_ADD(thres, 'B_score')
        # gen_plot_ADD(thres, 'A_score')






