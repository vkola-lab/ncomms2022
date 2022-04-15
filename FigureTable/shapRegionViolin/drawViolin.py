import matplotlib
import matplotlib.pyplot as plt
import csv
import collections
import numpy as np
import seaborn as sns
import pandas as pd

def plot_violin_shap_regions():
    scale = 1
    factor = 170.0 * 206.0 * 170.0 / 43.0 / 52.0 / 43.0 / scale
    # load raw data and regions
    content = []
    regions = []
    with open('../brainNetwork/Hammers_mith_atlases_n30r95_label_indices_SPM12_20170315.xml', 'r') as f:
        for row in f:
            regions.append(row.split("<name>")[1].split("</name>")[0])
    # combine left and right for the same region of ADD cases
    data = np.load('../brainNetwork/regional95_avgScore_ADD.npy') / factor
    ADD_new_regions = collections.defaultdict(list)
    for i, region in enumerate(regions):
        if region[-2:] not in [' R', ' L']:
            ADD_new_regions[region].append(data[:, i])
        else:
            ADD_new_regions[region[:-2]].append(data[:, i])
    for key in ADD_new_regions:
        ADD_new_regions[key] = np.sum(np.stack(ADD_new_regions[key]), axis=0)
        for j in range(ADD_new_regions[key].shape[0]):
            content.append({'region':abbre(key), 'Group':'AD', 'shap':ADD_new_regions[key][j]})

    print(ADD_new_regions.keys())

    # combine left and right for the same region of nADD cases
    data = np.load('../brainNetwork/regional95_avgScore_nADD.npy') / factor
    nADD_new_regions = collections.defaultdict(list)
    for i, region in enumerate(regions):
        if region[-2:] not in [' R', ' L']:
            nADD_new_regions[region].append(data[:, i])
        else:
            nADD_new_regions[region[:-2]].append(data[:, i])
    for key in nADD_new_regions:
        nADD_new_regions[key] = np.sum(np.stack(nADD_new_regions[key]), axis=0)
        for j in range(nADD_new_regions[key].shape[0]):
            content.append({'region': abbre(key), 'Group': 'nADD', 'shap': nADD_new_regions[key][j]})

    # write data into a csv, where columns are "ADD", "region", "shap"
    f = open("violin_shap.csv", 'w')
    writer = csv.DictWriter(f, fieldnames=['Group', 'shap', 'region'])
    writer.writeheader()
    for row in content:
        writer.writerow(row)
    f.close()

    def sort_regions(regions, ADD_new_regions, nADD_new_regions):
        rank = []
        for r in regions:
            value = np.var(np.concatenate((ADD_new_regions[r], nADD_new_regions[r])))
            rank.append((r, value))
        rank.sort(key=lambda x:x[1])
        return [a[0] for a in rank][::-1]

    TL = ['TL hippocampus', 'TL posterior temporal lobe', 'TL amygdala', 'TL anterior temporal lobe medial part', 'TL anterior temporal lobe lateral part',
          'TL parahippocampal and ambient gyrus', 'TL superior temporal gyrus middle part', 'TL middle and inferior temporal gyrus',
          'TL fusiform gyrus', 'TL superior temporal gyrus anterior part']
    TL = sort_regions(TL, ADD_new_regions, nADD_new_regions)

    FL = ['FL straight gyrus', 'FL anterior orbital gyrus', 'FL inferior frontal gyrus', 'FL precentral gyrus',
          'FL superior frontal gyrus', 'FL lateral orbital gyrus', 'FL posterior orbital gyrus','FL subcallosal area',
          'FL pre-subgenual frontal cortex', 'FL middle frontal gyrus', 'FL medial orbital gyrus','FL subgenual frontal cortex']
    FL = sort_regions(FL, ADD_new_regions, nADD_new_regions)

    PL = ['PL angular gyrus','PL postcentral gyrus','PL superior parietal gyrus','PL supramarginal gyrus']
    PL = sort_regions(PL, ADD_new_regions, nADD_new_regions)

    OL = ['OL lingual gyrus', 'OL cuneus','OL lateral remainder occipital lobe']
    OL = sort_regions(OL, ADD_new_regions, nADD_new_regions)

    insula = ['insula anterior short gyrus', 'insula middle short gyrus', 'insula posterior short gyrus', 'insula anterior inferior cortex',
              'insula anterior long gyrus', 'insula posterior long gyrus']
    insula = sort_regions(insula, ADD_new_regions, nADD_new_regions)

    ventricle = ['Lateral ventricle excluding temporal horn', 'Lateral ventricle temporal horn', 'Third ventricle']
    ventricle = sort_regions(ventricle, ADD_new_regions, nADD_new_regions)

    other = ['cerebellum', 'brainstem excluding substantia nigra', 'substantia nigra', 'CG anterior cingulate gyrus', 'CG posterior cingulate gyrus',
              'caudate nucleus', 'nucleus accumbens', 'putamen', 'thalamus', 'pallidum', 'corpus callosum']
    other = sort_regions(other, ADD_new_regions, nADD_new_regions)

    orders = TL + FL + PL + OL + ventricle + other + insula
    assert(len(orders) == len(list(ADD_new_regions.keys())))
    orders = [abbre(a) for a in orders]
    print(orders)

    from matplotlib import rc, rcParams
    rc('axes', linewidth=1)
    rc('font', weight='bold', size=10)
    sns.set_style("darkgrid", rc={"axes.facecolor": "#f7f7f7"})
    df = pd.read_csv('violin_shap.csv')
    fig, ax = plt.subplots(dpi=300, figsize=(20, 5))
    for i in range(49):
        plt.plot((i, i), (-0.1, 0.1), color='#e4d1ff', linestyle='dashed', linewidth=1, zorder=0)
    ax1 = sns.violinplot(data=df, x="region", y="shap", hue="Group", cut=0, order=orders, legend=False,
                   split=True, inner="quart", linewidth=1, palette={'nADD': "#42c5f5", 'AD': "#f54242"})
    ax1.legend(loc='upper right')
    sns.despine(left=True)
    ax.set_ylabel('Shap value', fontsize=15, fontweight='bold')
    ax.set_xlabel('')
    ax.set_yticks(ticks=[-0.1, -0.05, 0, 0.05, 0.1])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.ylim([-0.1, 0.1])
    plt.savefig("violin_VD.png", bbox_inches='tight')
    plt.close()


def abbre(text):
    replacements = {'temporal': 'temp', 'occipital': 'occi', 'ventricle': 'vent', 'brainstem': 'stem', 'excluding': 'ex.',
                    'middle': 'mid', 'posterior': 'post', 'anterior': 'ant', 'inferior': 'inf', 'superior': 'sup', 'Lateral': 'lat',
                    'lateral': 'lat', 'frontal': 'fron', 'and': '&', 'parahippocampal': 'parahippo', 'medial': 'med',
                    'TL ': '', 'FL ': '', 'PL ': '', 'OL ': '', 'CG ': ''}
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


if __name__ == "__main__":
    plot_violin_shap_regions()