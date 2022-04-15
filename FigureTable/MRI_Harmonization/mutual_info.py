import csv
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

def mutual_info(matrix):
    true = []
    pred = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            N = matrix[i, j]
            true += [i] * N
            pred += [j] * N
    return normalized_mutual_info_score(true, pred)

adc_diag = {}
scanner_diag = {}
adc_set = set()
scanner_set = set()

with open('tSNE_table.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['scanner_brand'] and row['diag']:
            scanner, diag = row['scanner_brand'], row['diag']
            if scanner not in scanner_diag:
                scanner_diag[scanner] = collections.defaultdict(int)
            scanner_diag[scanner][diag] += 1
            scanner_set.add(scanner)
        if row['ADC_id'] and row['diag']:
            adc, diag = row['ADC_id'], row['diag']
            if adc not in adc_diag:
                adc_diag[adc] = collections.defaultdict(int)
            adc_diag[adc][diag] += 1
            adc_set.add(adc)

adcs = sorted(list(adc_diag.keys()))
scanners = ['GE', 'Siemens', 'Philips']
diags = ['NC', 'MCI', 'AD', 'nADD']

scanner_matrix = [[0 for _ in range(len(scanners))] for _ in range(4)]
for i in range(len(scanner_matrix)):
    for j in range(len(scanner_matrix[0])):
        diag = diags[i]
        scanner = scanners[j]
        scanner_matrix[i][j] = scanner_diag[scanner][diag]

adc_matrix = [[0 for _ in range(len(adcs))] for _ in range(4)]
for i in range(len(adc_matrix)):
    for j in range(len(adc_matrix[0])):
        diag = diags[i]
        adc = adcs[j]
        adc_matrix[i][j] = adc_diag[adc][diag]

scanner_matrix = np.array(scanner_matrix)
adc_matrix = np.array(adc_matrix)

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
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=12, fontweight='black')

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

from matplotlib import rc, rcParams
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams.update({'font.size': 18})

n, m = len(scanner_matrix), len(scanner_matrix[0])
fig, ax = plt.subplots(dpi=300)
im, cbar = heatmap(scanner_matrix, diags, scanners, ax=ax,
                   cmap="YlGn", cbarlabel="Count")
texts = annotate_heatmap(im, valfmt="{x:.0f}")
fig.tight_layout()
plt.savefig('scanner_heatmap.png')

n, m = len(adc_matrix), len(adc_matrix[0])
fig, ax = plt.subplots(dpi=300, figsize=(m, n))
im, cbar = heatmap(adc_matrix, diags, adcs, ax=ax,
                   cmap="YlGn", cbarlabel="Count")
texts = annotate_heatmap(im, valfmt="{x:.0f}")
fig.tight_layout()
plt.savefig('Adc_id_heatmap.png')

print('scanner nutual info', mutual_info(scanner_matrix))
print('adc nutual info', mutual_info(adc_matrix))


