import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

FEATURES = ["age", "gender", "education", "race",
            "trailA", "trailB", "boston", "digitB", "digitBL", "digitF",
            "digitFL", "animal", "gds", "lm_imm", "lm_del", "mmse",
            "npiq_DEL", "npiq_HALL", "npiq_AGIT", "npiq_DEPD", "npiq_ANX", "npiq_ELAT", "npiq_APA",
            "npiq_DISN", "npiq_IRR", "npiq_MOT", "npiq_NITE", "npiq_APP",
            "faq_BILLS", "faq_TAXES", "faq_SHOPPING", "faq_GAMES", "faq_STOVE",
            "faq_MEALPREP", "faq_EVENTS", "faq_PAYATTN", "faq_REMDATES", "faq_TRAVEL",
            "his_NACCFAM", "his_CVHATT", "his_CVAFIB", "his_CVANGIO", "his_CVBYPASS", "his_CVPACE",
            "his_CVCHF", "his_CVOTHR", "his_CBSTROKE", "his_CBTIA", "his_SEIZURES", "his_TBI",
            "his_HYPERTEN", "his_HYPERCHO", "his_DIABETES", "his_B12DEF", "his_THYROID", "his_INCONTU", "his_INCONTF",
            "his_DEP2YRS", "his_DEPOTHR", "his_PSYCDIS", "his_ALCOHOL",
            "his_TOBAC100", "his_SMOKYRS", "his_PACKSPER", "his_ABUSOTHR"]

FEATURES_FULLNAME = [
    "Age",
    "Gender",
    "Education",
    "Race",
    "Trail Making Test Part A",
    "Trail Making Test Part B",
    "Boston Naming Test (30)",
    "Digit span backward trials correct",
    "Digit span backward length",
    "Digit span forward trials correct",
    "Digit span forward length",
    "Animals",
    "Total GDS Score",
    "Logical memory immediate recall",
    "Logical memory delayed recall",
    "Total MMSE score",
    "NPIQ delusions",
    "NPIQ hallucinations",
    "NPIQ agitation or aggression",
    "NPIQ depression or dysphoria",
    "NPIQ anxiety",
    "NPIQ elation or euphoria",
    "NPIQ apathy or indifference",
    "NPIQ disinhibition",
    "NPIQ irritability or lability",
    "NPIQ motor disturbance",
    "NPIQ nighttime behaviors",
    "NPIQ appetite",
    "FAQ bills",
    "FAQ taxes",
    "FAQ shopping",
    "FAQ games",
    "FAQ stove",
    "FAQ meal prep",
    "FAQ events",
    "FAQ pay attention",
    "FAQ remdates",
    "FAQ travel",
    "History family cognitive impairment",
    "History heart attack/cardiac arrest",
    "History atrial fibrillation",
    "History angioplasty/endarterectomy/stent",
    "History cardiac bypass procedure",
    "History pacemaker",
    "History congestive heart failure",
    "History other cardiovascular disease",
    "History stroke",
    "History transient ischemic attack",
    "History seizures",
    "History traumatic brain injury",
    "History hypertension",
    "History hypercholesterolemia",
    "History diabetes",
    "History vitamin B12 deficiency",
    "History Thyroid disease",
    "History incontinence-urinary",
    "History incontinence-bowel",
    "History active depression in the last two years",
    "History depression episodes more than two years ago",
    "History other psychiatric disorder",
    "History alcohol abuse",
    "History smoked more than 100 cigarettes in life",
    "History total years smoked cigarettes",
    "History average number of packs smoked per day",
    "History other abused substances"
]

def get_missing_rate(filenames, FEATURES, labels):
    data = {} # data[feature][diag] = [exist, total]
    for feature in FEATURES:
        data[feature] = {}
        for diag in labels:
            data[feature][diag] = [0, 0]
    for filename in filenames:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['COG'] == '0': # NC
                    for f in FEATURES:
                        if f in row and row[f]:
                            data[f]['NC'][0] += 1
                        data[f]['NC'][1] += 1
                elif row['COG'] == '1': # MCI
                    for f in FEATURES:
                        if f in row and row[f]:
                            data[f]['MCI'][0] += 1
                        data[f]['MCI'][1] += 1
                elif row['COG'] == '2':
                    if row['AD'] == '1': # AD
                        for f in FEATURES:
                            if f in row and row[f]:
                                data[f]['AD'][0] += 1
                            data[f]['AD'][1] += 1
                    else: # nADD
                        for f in FEATURES:
                            if f in row and row[f]:
                                data[f]['nADD'][0] += 1
                            data[f]['nADD'][1] += 1
    return data

def convert(data, labels):
    matrix = [[0] * len(labels) for i in range(len(FEATURES))]
    for i, f in enumerate(FEATURES):
        for j, diag in enumerate(labels):
            matrix[i][j] = 1
            if data[f][diag][1]:
                matrix[i][j] -= data[f][diag][0] / data[f][diag][1]
    return matrix

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
    im = ax.imshow(data, vmin=0, vmax=1, **kwargs)

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
rcParams.update({'font.size': 12})

csv_map = {
    "NACC":  ["../../lookupcsv/dataset_table/NACC_ALL/NACC.csv"],
    "OASIS": ["../../lookupcsv/dataset_table/OASIS/OASIS.csv"],
    "ADNI":  ["../../lookupcsv/dataset_table/ADNI1/ADNI1.csv",
             "../../lookupcsv/dataset_table/ADNI2/ADNI2.csv",
             "../../lookupcsv/dataset_table/ADNI3/ADNI3.csv",
             "../../lookupcsv/dataset_table/ADNIGO/ADNIGO.csv"],
    "AIBL":  ["../../lookupcsv/dataset_table/AIBL/AIBL.csv"],
    "FHS":   ["../../lookupcsv/dataset_table/FHS/FHS.csv"],
    "PPMI":  ["../../lookupcsv/dataset_table/PPMI/PPMI.csv"],
    "NIFD":  ["../../lookupcsv/dataset_table/NIFD/NIFD.csv"],
    "LBDSU": ["../../lookupcsv/dataset_table/Stanford/Stanford.csv"]
}

labels = {
    'NACC': ['NC', 'MCI', 'AD', 'nADD'],
    'OASIS': ['NC', 'MCI', 'AD', 'nADD'],
    'ADNI': ['NC', 'MCI', 'AD'],
    'FHS': ['NC', 'MCI', 'AD', 'nADD'],
    'NIFD': ['NC', 'nADD'],
    'PPMI': ['NC', 'MCI'],
    'LBDSU': ['NC', 'MCI', 'nADD'],
    'AIBL': ['NC', 'MCI', 'AD'],
}

for dataset in ['NACC', 'OASIS', 'ADNI', 'FHS', 'NIFD', 'PPMI', 'LBDSU', 'AIBL']:
    data = get_missing_rate(csv_map[dataset], FEATURES, labels[dataset])
    matrix = np.array(convert(data, labels[dataset]))

    fig, ax = plt.subplots(figsize=(8, 40), dpi=200)
    im, cbar = heatmap(matrix, FEATURES_FULLNAME, labels[dataset], ax=ax,
                       cmap="YlGnBu", cbarlabel="Feature Missing Rate")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    fig.tight_layout()
    plt.savefig('{}_features.png'.format(dataset))
    plt.close()



