import matplotlib
import matplotlib.pyplot as plt
import csv
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import copy

from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', weight='bold', size=13)
rcParams['hatch.linewidth'] = 2.0

thres = 365 * 2

def draw_COG_heatmap():
    data = []
    with open('../NeuroPathTable/ALL.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['diff_days'] and row['Group']:
                if int(row['diff_days']) < thres:
                    case = [row['Group'], row['dataset'], float(row['CNN_COG_score'])]
                    if row['A_score']:
                        case.append(int(row['A_score']))
                    else:
                        case.append(99)
                    if row['B_score']:
                        case.append(int(row['B_score']))
                    else:
                        case.append(99)
                    if row['C_score']:
                        case.append(int(row['C_score']))
                    else:
                        case.append(99)
                    if case[-1] == 99 and case[-2] == 99  and case[-3] == 99:
                        continue
                    data.append(case)

    data = sorted(data, key=lambda x : -x[2])
    heatmap = np.array([d[3:] for d in data])
    yticks = [d[0] if d[0] in ['NC', 'MCI'] else 'DEM' for d in data]
    dataset = [d[1][0] for d in data]

    cmap = matplotlib.cm.get_cmap('bwr', 4)
    cmap.set_over('white')

    fig, ax = plt.subplots(figsize=(3, len(yticks) / 4.0), dpi=200)
    ax = sns.heatmap(heatmap, cmap=cmap, vmin=-1, vmax=4, cbar=False, linecolor='white', linewidths=1.3)
    ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_yticks(ticks = [i+0.5 for i in range(len(yticks))])
    ax.set_yticklabels(yticks)
    for i in range(len(yticks)):
        plt.text(3, i+1, dataset[i])
    plt.savefig("COG_heatmap.png", bbox_inches='tight')
    plt.close()


def draw_ADD_heatmap():
    data = []
    with open('../NeuroPathTable/ALL.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['diff_days'] and row['Group']:
                if int(row['diff_days']) < thres and 'ADD' in row['Group']:
                    case = [row['Group'], float(row['CNN_ADD_prob'])]
                    if row['A_score']:
                        case.append(int(row['A_score']))
                    else:
                        case.append(99)
                    if row['B_score']:
                        case.append(int(row['B_score']))
                    else:
                        case.append(99)
                    if row['C_score']:
                        case.append(int(row['C_score']))
                    else:
                        case.append(99)
                    if case[-1] == 99 and case[-2] == 99 and case[-3] == 99:
                        continue
                    data.append(case)

    data = sorted(data, key=lambda x: -x[1])
    heatmap = np.array([d[2:] for d in data])
    yticks = [d[0] for d in data]

    cmap = matplotlib.cm.get_cmap('bwr', 4)
    cmap.set_over('white')

    fig, ax = plt.subplots(figsize=(3, len(yticks) / 4.0), dpi=200)
    ax = sns.heatmap(heatmap, cmap=cmap, vmin=-1, vmax=4, cbar=False, linecolor='white', linewidths=1.3)
    ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_yticks([])
    plt.savefig("ADD_heatmap.png", bbox_inches='tight')
    plt.close()
    

def draw_ADD_heatmap_labels():
    data = []
    with open('../NeuroPathTable/ALL.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['diff_days']:
                if int(row['diff_days']) < thres and 'ADD' in row['Group']:
                    case = [row['Group'], float(row['CNN_ADD_prob'])]
                    if row['A_score']:
                        case.append(int(row['A_score']))
                    else:
                        case.append(99)
                    if row['B_score']:
                        case.append(int(row['B_score']))
                    else:
                        case.append(99)
                    if row['C_score']:
                        case.append(int(row['C_score']))
                    else:
                        case.append(99)
                    if case[-1] == 99 and case[-2] == 99 and case[-3] == 99:
                        continue
                    case.append(row['dataset'])
                    data.append(case)

    data = sorted(data, key=lambda x: -x[1])
    
    fig, ax = plt.subplots(figsize=(.5, len(data) * .5), dpi=200)
    
    lst_hatch = ['xx', '.', 'O']
    lst_dset  = ['NACC', 'ADNI', 'FHS']
    hmp_hatch = dict(zip(lst_dset, lst_hatch))

    lst_color = [(0, 0, 1),
                 (1, 0, 0)]
    lst_label = ['nADD', 'ADD']
    hmp_color = dict(zip(lst_label, lst_color))
    
    for i, itm in enumerate(data):
        xs = (0, 2)
        ys_top = (-i, -i)
        ys_bot = (-i-1, -i-1)
        plt.fill_between(xs, ys_top, ys_bot,
                          facecolor='grey',
                          edgecolor='white',
                          linewidth = 1,
                          hatch=hmp_hatch[itm[-1]])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.box(False)
    plt.axis('scaled')
    fig.savefig('./ADD_heatmap_labels.png', bbox_inches='tight', dpi=200, transparent=True)


if __name__ == "__main__":
    draw_ADD_heatmap()
    draw_COG_heatmap()
    draw_ADD_heatmap_labels()






