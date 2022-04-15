import csv
from sklearn.metrics import cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib

from matplotlib import rc
rc('axes', linewidth=1.0)
rc('font', weight='bold', size=18)

def get_diags(csvfile, mode):
    res = []
    if mode == 'all':
        map = {'Mild Cognitive Impairment' : 1,
               'Normal Cognition' : 0,
               'Dementia _ Alzheimer\'s Disease Dementia' : 2,
               'Dementia _ not Alzheimer\'s Disease Dementia': 3}
    elif mode == 'NC':
        map = {'Mild Cognitive Impairment': 1,
               'Normal Cognition': 0,
               'Dementia _ Alzheimer\'s Disease Dementia': 1,
               'Dementia _ not Alzheimer\'s Disease Dementia': 1}
    elif mode == 'MCI':
        map = {'Mild Cognitive Impairment': 0,
               'Normal Cognition': 1,
               'Dementia _ Alzheimer\'s Disease Dementia': 1,
               'Dementia _ not Alzheimer\'s Disease Dementia': 1}
    elif mode == 'DE':
        map = {'Mild Cognitive Impairment': 1,
               'Normal Cognition': 1,
               'Dementia _ Alzheimer\'s Disease Dementia': 0,
               'Dementia _ not Alzheimer\'s Disease Dementia': 0}
    with open(csvfile, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            res.append(map[row['Diagnosis Label']])
    return res


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    code from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=22, fontweight='black')

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, rotation=60)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    plt.setp(ax.get_xticklabels())

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def agreement(mode):
    agree = np.ones((17, 17)) * 100
    average = []
    for i in range(17):
        for j in range(i):
            resA = get_diags('neurologists/n{}.csv'.format(i + 1), mode)
            resB = get_diags('neurologists/n{}.csv'.format(j + 1), mode)
            agree[i, j] = cohen_kappa_score(resA, resB)
            average.append(agree[i, j])
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = copy.copy(matplotlib.cm.get_cmap("PiYG"))
    cmap.set_over('white')
    im, cbar = heatmap(agree,
                       ['neuro. {}'.format(i) for i in range(1, 18)],
                       [i for i in range(1, 18)],
                       ax=ax, vmin=-0.2, vmax=1,
                       cmap=cmap, cbarlabel="Cohen's Kappa")
    average = sum(average) / len(average)
    plt.savefig('kappa/kappa_{}_avg{:.3f}.png'.format(mode, average), dpi=200, bbox_inches='tight')
    plt.close()
    print(np.mean(agree))

if __name__ == "__main__":
    agreement('all')
    agreement('NC')
    agreement('MCI')
    agreement('DE')




