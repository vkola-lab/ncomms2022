import csv
from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
import math
import copy
import os
from collections import defaultdict

def boxplot(shap, neuropath, name, folder):
    x_scores = [0, 1, 2, 3]
    all_data = [[] for _ in range(len(x_scores))]
    for data, score in zip(shap, neuropath):
        all_data[x_scores.index(score)].append(data)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    ax.boxplot(all_data,
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=x_scores,  # will be used to label x-ticks
               zorder=0,
               showfliers=False)
    for x, data in zip(x_scores, all_data):
        ax.plot(np.random.normal(x+1, 0.1, size=len(data)), data, 'r.',
                alpha=0.6, zorder=10, markersize=8)
    c, p = stats.spearmanr(shap, neuropath)
    ax.set_title('corr={:.4f}'.format(c))
    ax.set_xlabel(name)
    ax.set_ylabel('Shap value')
    plt.savefig(folder+name+'.png', dpi=200, bbox_inches='tight')
    plt.close()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75, orientation="horizontal", **cbar_kw)
    cbar.set_label(cbarlabel, rotation=0, fontsize=12, fontweight='black')
    # cbar.ax.set_ylabel(cbarlabel, rotation=0, va="center", fontsize=12, fontweight='black')
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

def plot_corr_heatmap(regions, region_names, stains, corre, annot, filename, folder):
    from matplotlib import rc, rcParams
    rc('axes', linewidth=1)
    rc('font', weight='bold')
    rcParams.update({'font.size': 7})
    hm = np.zeros((len(stains), len(regions)))
    an = np.zeros((len(stains), len(regions)))
    for i in range(len(stains)):
        for j in range(len(regions)):
            hm[i, j] = corre[regions[j]][stains[i]]
            an[i, j] = annot[regions[j]][stains[i]]
    fig, ax = plt.subplots(figsize=(6, 18))
    cmap = copy.copy(matplotlib.cm.get_cmap("bwr"))
    cmap.set_over('grey')
    im, cbar = heatmap(hm, stains, region_names, ax=ax, vmin=-1, vmax=1,
                       cmap=cmap, cbarlabel="Correlation")
    plt.savefig(folder + 'corrmap_{}.png'.format(filename), dpi=200, bbox_inches='tight')
    plt.close()

def file_interval_info(type):
    time_diff = {}
    with open('../NeuroPathTable/ALL.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if type == 'ADD':
                if row['Group'] in ['NC', 'MCI']:
                    continue
            time_diff[row['filename']]  = int(row['diff_days']) if row['diff_days'] else 100000
    return time_diff

def get_correlation(col_name, indexes, thres, interval, folder, type='ADD', layer='block2conv', missing=100):
    sub_shap = {}
    with open('shap_csvfiles/ADNI_shap_{}_region_scores_{}.csv'.format(type, layer), 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['filename'] in interval and interval[row['filename']] <= thres:
                rid = row['filename'].split('_')[3]
                sub_shap[rid] = 0
                for idx in indexes:
                    sub_shap[rid] += float(row[str(idx)])
    with open('shap_csvfiles/FHS_shap_{}_region_scores_{}.csv'.format(type, layer), 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['filename'] in interval and interval[row['filename']] <= thres:
                id = row['filename'].split('_')[0]
                sub_shap[id] = 0
                for idx in indexes:
                    sub_shap[id] += float(row[str(idx)])
    sub_np = {}
    with open('../NeuroPathTable/ALL.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row[col_name] and row[col_name].isdecimal() and row[col_name] not in ['9', '9.0']:
                sub_np[row['id']] = float(row[col_name])
    vec1, vec2 = [], []
    for key in sub_shap:
        if key in sub_np:
            vec1.append(sub_shap[key])
            vec2.append(sub_np[key])
    # print('vec1 is', vec1)
    # print('vec2 is', vec2)
    # print('-------------------------------')
    if len(vec1) < 2 or len(set(vec2)) == 1:
        return missing, 0
    boxplot(vec1, vec2, col_name, folder)
    c, p = stats.spearmanr(vec1, vec2)
    return c, len(vec1)

prefixes = ['CG_1', 'FL_mfg_7', 'FL_pg_10', 'TL_stg_32', 'PL_ag_20', 'Amygdala_24',
            'TL_hippocampus_28', 'TL_parahippocampal_30', 'c_37', 'bs_35', 'sn_48', 'th_49',
            'pal_46', 'na_45X', 'cn_36C', 'OL_17_18_19OL']

regions = ['CG anterior cingulate gyrus', 'FL middle frontal gyrus',
           'FL precentral gyrus', 'TL superior temporal gyrus anterior part',
           'PL angular gyrus', 'Amygdala', 'Hippocampus', 'parahippocampal and ambient gyrus',
           'cerebellum', 'brainstem excluding substantia nigra', 'substantia nigra', 'thalamus',
           'pallidum', 'nucleus accumbens', 'caudate nucleus',
           'OL cuneus, lateral remainder, lingual gyrus']

def abbre(text):
    replacements = {'temporal': 'temp', 'occipital': 'occi', 'ventricle': 'vent', 'brainstem': 'stem', 'excluding': 'ex.',
                    'middle': 'mid', 'posterior': 'post', 'anterior': 'ant', 'inferior': 'inf', 'superior': 'sup', 'Lateral': 'lat',
                    'lateral': 'lat', 'frontal': 'fron', 'and': '&', 'parahippocampal': 'parahippo', 'medial': 'med',
                    'TL ': '', 'FL ': '', 'PL ': '', 'OL ': '', 'CG ': ''}
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

regions = [abbre(reg) for reg in regions]
print(regions)
regions = ['Ant cingulate gyrus',
           'Mid fron gyrus',
           'Precentral gyrus',
           'Sup temp gyrus ant part',
           'Angular gyrus',
           'Amygdala',
           'Hippocampus',
           'Parahippocampus',
           'Cerebellum',
           'Brainstem',
           'Substantia nigra',
           'Thalamus',
           'Pallidum',
           'Nucleus accumbens',
           'Caudate nucleus',
           'Cuneus']

prefix_idx = {'CG_1':[24],
              'FL_mfg_7': [28],
              'FL_pg_10': [50],
              'TL_stg_32': [82],
              'PL_ag_20': [32],
              'Amygdala_24': [4],
              'TL_hippocampus_28': [2],
              'TL_parahippocampal_30': [10],
              'c_37': [18],
              'bs_35': [19],
              'sn_48': [74],
              'th_49': [40],
              'pal_46': [42],
              'na_45X': [36],
              'cn_36C': [34],
              'OL_17_18_19OL': [64, 66, 22]}

stains = ['AB_DP', 'TAU_NFT', 'TAU_NP', 'SILVER_NFT']

if __name__ == "__main__":
    years = 2
    time_threshold, type = 365 * years, 'COG'
    folder = type + '_correlation_{}_years/'.format(years)
    if not os.path.exists(folder):
        os.mkdir(folder)
    interval = file_interval_info(type)

    ordered_regions, ordered_prefixes = [], []

    for layername in ['block2conv', 'block2pooling', 'block2BN']:
        corre = collections.defaultdict(dict)
        annot = collections.defaultdict(dict)
        region_to_corre = defaultdict(list)
        for region in prefixes:
            for stain in stains:
                # print('found ', region + '_' + stain, ' with index ', prefix_idx[region])
                corr, n = get_correlation(region + '_' + stain, prefix_idx[region], time_threshold, interval, folder, type, layername)
                # print('correlation is ', corr)
                corre[region][stain] = corr
                annot[region][stain] = n
                if corr >= -1 and corr <= 1:
                    region_to_corre[region].append(corr)

        pool = [(-sum(region_to_corre[key])/len(region_to_corre[key]), key, region) for key, region in zip(prefixes, regions)]
        pool.sort(reverse=True)

        if not ordered_prefixes:
            ordered_prefixes = [p[1] for p in pool]
            ordered_regions = [p[2] for p in pool]

        plot_corr_heatmap(ordered_prefixes, ordered_regions, stains, corre, annot, '{}days_{}shap_{}'.format(time_threshold, type, layername), folder)

