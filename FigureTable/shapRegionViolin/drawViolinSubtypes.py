import matplotlib
import matplotlib.pyplot as plt
import csv
import collections
import numpy as np
import seaborn as sns
import pandas as pd

def plot_violin_shap_regions(subtype):
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

    # combine left and right for the same region of subtype cases
    data = np.load('regional95_avgScore_{}.npy'.format(subtype)) / factor
    nADD_new_regions = collections.defaultdict(list)
    for i, region in enumerate(regions):
        if region[-2:] not in [' R', ' L']:
            nADD_new_regions[region].append(data[:, i])
        else:
            nADD_new_regions[region[:-2]].append(data[:, i])
    for key in nADD_new_regions:
        nADD_new_regions[key] = np.sum(np.stack(nADD_new_regions[key]), axis=0)
        for j in range(nADD_new_regions[key].shape[0]):
            content.append({'region': abbre(key), 'Group': subtype, 'shap': nADD_new_regions[key][j]})

    # write data into a csv, where columns are "ADD", "region", "shap"
    f = open("violin_shap_{}.csv".format(subtype), 'w')
    writer = csv.DictWriter(f, fieldnames=['Group', 'shap', 'region'])
    writer.writeheader()
    for row in content:
        writer.writerow(row)
    f.close()

    orders = ['hippocampus', 'post temp lobe', 'sup temp gyrus mid part', 'parahippo & ambient gyrus', 'amygdala', 'mid & inf temp gyrus', 'ant temp lobe med part',
     'fusiform gyrus', 'sup temp gyrus ant part', 'ant temp lobe lat part', 'sup fron gyrus', 'mid fron gyrus', 'precentral gyrus', 'inf fron gyrus', 'straight gyrus', 
     'post orbital gyrus', 'subgenual fron cortex', 'med orbital gyrus', 'lat orbital gyrus', 'ant orbital gyrus', 'subcallosal area', 'pre-subgenual fron cortex',
      'sup parietal gyrus', 'postcentral gyrus', 'supramarginal gyrus', 'angular gyrus', 'lat remainder occi lobe', 'lingual gyrus', 'cuneus', 'lat vent ex. temp horn', 
      'lat vent temp horn', 'Third vent', 'corpus callosum', 'cerebellum', 'caudate nucleus', 'thalamus', 'stem ex. substantia nigra', 'ant cingulate gyrus', 
      'post cingulate gyrus', 'putamen', 'nucleus accumbens', 'pallidum', 'substantia nigra', 'insula post long gyrus', 'insula ant inf cortex', 'insula ant long gyrus', 
      'insula post short gyrus', 'insula ant short gyrus', 'insula mid short gyrus']
    

    from matplotlib import rc, rcParams
    rc('axes', linewidth=1)
    rc('font', weight='bold', size=10)
    sns.set_style("darkgrid", rc={"axes.facecolor": "#f7f7f7"})
    df = pd.read_csv('violin_shap_{}.csv'.format(subtype))
    fig, ax = plt.subplots(dpi=300, figsize=(20, 2))
    for i in range(49):
        plt.plot((i, i), (-0.1, 0.1), color='#e4d1ff', linestyle='dashed', linewidth=1, zorder=0)
    ax1 = sns.violinplot(data=df, x="region", y="shap", hue="Group", cut=0, order=orders, legend=False,
                   split=True, inner="quart", linewidth=1, palette={'nADD': "#42c5f5", 'AD': "#f54242", 'VD': "#9966cc", "FTD": "#00ff49", "LBD": "#42c5f5"})
    ax1.legend(loc='upper right')
    sns.despine(left=True)
    ax.set_ylabel('Shap value', fontsize=15, fontweight='bold')
    ax.set_xlabel('')
    ax.set_yticks(ticks=[-0.1, -0.05, 0, 0.05, 0.1])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.ylim([-0.1, 0.1])
    plt.savefig("violin_{}.png".format(subtype), bbox_inches='tight')
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
    plot_violin_shap_regions('VD')
    plot_violin_shap_regions('FTD')
    plot_violin_shap_regions('LBD')