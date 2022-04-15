import csv
import os
from collections import defaultdict
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy
import numpy as np

from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', weight='bold', size=20)

from correlation import *

pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * .5]
open_circle = matplotlib.path.Path(vert)

def scatter(radio_vec, shap_vec, name, corr):
    size, lw = 60, 3
    if not os.path.exists('scatter/'):
        os.mkdir('scatter/')
    fig, ax = plt.subplots(dpi=300, figsize=(3, 3))
    ax.scatter(radio_vec, shap_vec, marker='o', edgecolors='blue', linewidths=lw, s=size, c='white', alpha=0.6)
    ax.set_ylabel('SHAP', fontweight='bold')
    ax.set_xlabel('Averaged atrophy rating', fontweight='bold')
    ax.set_title('{}({:.2f})'.format(name, corr), fontweight='bold')
    plt.savefig('scatter/{}.png'.format(name), bbox_inches='tight')
    plt.close()

def scatter(radio_vec_l, shap_vec_l, radio_vec_r, shap_vec_r, name):
    size, lw = 60, 3
    if not os.path.exists('scatter/'):
        os.mkdir('scatter/')
    fig, ax = plt.subplots(dpi=300, figsize=(3, 3))
    ax.scatter(radio_vec_l, shap_vec_l, marker='o', edgecolors='blue', linewidths=lw, s=size, c='white', alpha=0.6, label='left')
    ax.scatter(radio_vec_r, shap_vec_r, marker='^', edgecolors='blue', linewidths=lw, s=size, c='white', alpha=0.6, label='right')
    ax.set_ylabel('SHAP', fontweight='bold')
    ax.set_xlabel('Averaged atrophy rating', fontweight='bold')
    ax.set_title('{}({:.2f})'.format(name, 0.2), fontweight='bold')
    plt.savefig('scatter/{}.png'.format(name), bbox_inches='tight')
    plt.close()

cache = {}
for region in regions:
    vec1 = get_averaged_radio_scores(region, team)
    vec2 = get_shap_scores(region, shap)
    c, p = stats.spearmanr(vec1, vec2)
    cache[region] = [vec1, vec2]

def create_lmplot(regions, name):
    # create the data frame, with the following columns
    # shap, rating, lr
    colors = ['#266dfc', '#fc5426']
    hue_order = ['left', 'right']
    y_min, y_max = 100, -100
    data = {'shap' : [], 'rating' : [], 'lr' : []}
    rate, shap = cache[regions[0]]
    y_min = min(y_min, min(shap))
    y_max = max(y_max, max(shap))
    l_stat = scipy.stats.pearsonr(rate, shap)
    for i in range(len(rate)):
        data['shap'].append(shap[i])
        data['rating'].append(rate[i])
        data['lr'].append('left')
    rate, shap = cache[regions[1]]
    y_min = min(y_min, min(shap))
    y_max = max(y_max, max(shap))
    if regions[0] == 'l_atl_l':
        y_max = 1.4
        y_min = -0.5
    r_stat = scipy.stats.pearsonr(rate, shap)
    for i in range(len(rate)):
        data['shap'].append(shap[i])
        data['rating'].append(rate[i])
        data['lr'].append('right')
    df = pd.DataFrame.from_dict(data)
    fig, ax = plt.subplots(dpi=300, figsize=(3, 4))

    # option1 with scatter and marginal distribution plot
    g = sns.jointplot(data=df, x='rating', y='shap', hue='lr', legend=False,
                      palette=colors, hue_order=hue_order, markers=['^', "o"])
    for n, gr in df.groupby('lr'):
        sns.regplot(x='rating', y='shap', data=gr, scatter=False, ax=g.ax_joint, truncate=False, color=colors[hue_order.index(n)])
    g.ax_joint.set_ylabel('SHAP', fontweight='bold', fontsize=20)
    g.ax_joint.set_ylim(y_min, (y_max - y_min) * 1.2 + y_min)  # leave some space for the p-value text
    g.ax_joint.set_xlabel('Avg. Atrophy Rating', fontweight='bold', fontsize=20)
    g.ax_marg_x.set_title(name, fontweight='bold', fontsize=20)

    # option2 only scatter plot
    # g = sns.lmplot(x="rating", y="shap", hue="lr", data=df, legend=False,
    #                   palette=colors, hue_order=['left', 'right'], markers=['^', "o"])
    # g.ax.set_ylabel('SHAP', fontweight='bold', fontsize=15)
    # g.ax.set_ylim(y_min, (y_max - y_min) * 1.2 + y_min) # leave some space for the p-value text
    # g.ax.set_xlabel('Avg. Atrophy Rating', fontweight='bold', fontsize=15)
    # g.ax.set_title(name, fontweight='bold', fontsize=15)

    plt.text(0.25, 0.375, format_stat(*l_stat),
             color=colors[0],
             fontsize=20,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    plt.text(0.25, 0.35, format_stat(*r_stat),
             color=colors[1],
             fontsize=20,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    plt.savefig("scatter/{}.png".format(name), bbox_inches='tight')
    plt.close()

def format_stat(r, p):
    p_str = "P={:.3f}".format(p) if p > 0.001 else "P<0.001"
    stat_str = "r={:.2f}; ".format(r) + p_str
    return stat_str

## Temporal lobe
create_lmplot(['l_atl_amyg', 'r_atl_amyg'], 'Amygdala')
create_lmplot(['l_mtl_hippo', 'r_mtl_hippo'], 'Hippocampus')
create_lmplot(['l_mtl_parahippo', 'r_mtl_parahippo'], 'Parahippocampus')
create_lmplot(['l_atl_m', 'r_atl_m'], 'Anter temp lobe medial')
create_lmplot(['l_atl_l', 'r_atl_l'], 'Anter temp lobe lateral')

## Patrietal lobe
create_lmplot(['l_pl', 'r_pl'], 'Sup parietal lobe')

## Frontal lobe
create_lmplot(['l_orbitofrontal', 'r_orbitofrontal'], 'Orbitofrontal lobe')
create_lmplot(['l_dorsolateral', 'r_dorsolateral'], 'Mid frontal lobe')
create_lmplot(['l_superior', 'r_superior'], 'Sup frontal lobe')
create_lmplot(['l_posterior', 'r_posterior'], 'Post frontal lobe')

## Other
create_lmplot(['l_latventricle_temph', 'r_latventricle_temph'], 'Lat vent temp horn')
create_lmplot(['l_latventricle', 'r_latventricle'], 'Lat vent')





