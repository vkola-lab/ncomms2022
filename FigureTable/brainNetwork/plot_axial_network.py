import matplotlib.pyplot as plt
import numpy as np
import cv2
from random import randint
import csv
import networkx as nx
from scipy.stats import spearmanr
import scipy
from plot_sagittal_network import get_region_idx, top_N, get_regions_from_key

def draw_node(ax, x, y, size, label, color):
    ax.scatter(x, y, c=color, s=size, linewidths=size**0.5 / 10, edgecolors='white', zorder=100)
    ax.text(x, y + 1, label, ha="center", va="center", fontweight='bold',
            fontsize=4, fontstyle='oblique', color='black', zorder=101)

def get_node_meta_info():
    node_meta_map = {}
    new_idx = 1
    with open('network_axi_combine_regions.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['X'] and row['Y']:
                row['new idx'] = new_idx
                new_idx += 1
                node_meta_map[row['region after combination']] = [int(row['X']), int(row['Y']), int(row['color']), row['new idx']]
    return node_meta_map

def load_combine_shap(node_meta_map, keyword):
    all_regions = get_region_idx()
    shap = np.load('regional95_avgScore_{}.npy'.format(keyword)) # shape is (N, 95)
    for node in node_meta_map:
        regions = get_regions_from_key(node, all_regions)
        node_shap = np.zeros(shap.shape[0])
        for reg in regions:
            node_shap += shap[:, all_regions[reg]-1]
        node_meta_map[node].append(node_shap)

def transform(weights, mode):
    print('there are {} edges'.format(len(weights)))
    if mode == 'linear':
        for i in range(len(weights)):
            weights[i] = weights[i] * 4
    elif mode == 'quadratic':
        for i in range(len(weights)):
            weights[i] = (weights[i] ** 2) * 40
    elif mode == 'cubic':
        for i in range(len(weights)):
            weights[i] = (weights[i] ** 3) * 20

def plot_network(keyword, mode, pvalue, N, corre_method):
    color_list = ['#b343d9', '#2fc40e', '#c4be0e', '#0ec4b5', '#f70c7d']
    fig, ax = plt.subplots(dpi=500)
    img = cv2.imread('glassbrain_background_axial.png')
    # img = np.rot90(img) # rotate rotate
    img = img[::-1, :]  # flip
    tall = img.shape[0]
    ax.imshow(img)
    node_meta_map = get_node_meta_info()
    load_combine_shap(node_meta_map, keyword)
    G = nx.Graph()
    regions = list(node_meta_map.keys())
    for i, reg in enumerate(regions):
        x, y, c, label, _ = node_meta_map[reg]
        # x, y = rotate_coordinate_90(x, y) # rotate rotate
        y = tall - y # flip
        G.add_node(label, pos=(float(x), float(y)), size=10, color=color_list[c])
    pos = nx.get_node_attributes(G, 'pos')
    max_corr = 0
    for i in range(len(regions) - 1):
        for j in range(i+1, len(regions)):
            if corre_method == 's':
                corr, p = spearmanr(node_meta_map[regions[i]][-1], node_meta_map[regions[j]][-1])
            elif corre_method == 'p':
                corr, p = scipy.stats.pearsonr(node_meta_map[regions[i]][-1], node_meta_map[regions[j]][-1])
            if p > pvalue: continue
            max_corr = max(corr, max_corr)
            color = 'r' if corr > 0 else 'b'
            G.add_edge(node_meta_map[regions[i]][3], node_meta_map[regions[j]][3], weight=abs(corr), color=color)
    print("max correlation is ", max_corr)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    colors = [G[u][v]['color'] for u, v in edges]
    threshold = top_N(weights, N)
    transform(weights, mode)
    for i, (u, v) in enumerate(edges):
        G[u][v]['weight'] = weights[i]
    node_size = [a[1] for a in nx.degree(G, weight='weight')]
    nx.draw(G, pos,
            width=weights,
            node_size=node_size,
            alpha=0.9,
            edge_color=colors,
            with_labels=False)
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        draw_node(ax, x, y, 20+3*G.degree(weight='weight')[node], node, G.nodes[node]['color'])
    w_min, w_max = min(weights), max(weights)
    add_legend(ax, w_min, w_max, threshold, 6)
    plt.savefig('plot/' + keyword + '_axi_network.png', bbox_inches='tight')
    plt.close()

def rotate_coordinate_90(x, y):
    x, y = y, x
    return x, y

def combine(filename):
    img1 = cv2.imread('plot/ADD_axi_network.png')
    img2 = cv2.imread('plot/nADD_axi_network.png')
    img = np.concatenate((img1, img2), axis=1)
    cv2.imwrite('plot/' + filename + '.png', img)

def add_legend(ax, vmin, vmax, thres, N):
    def corre_to_width(corr):
        return (corr-thres) * 4
    corr_min, corr_max = vmin / 4 + thres, vmax / 4 + thres
    corr_diff = (corr_max - corr_min) / (N-1)
    from matplotlib.lines import Line2D
    lines = []
    diff = (vmax - vmin) / (N-1)
    edges_weight_list1 = [vmin + i * diff for i in range(1, N)]
    corr_list1 = [-thres-corr_diff * i for i in range(1, N)]
    edges_weight_list2 = edges_weight_list1[::-1]
    corr_list2 = [-a for a in corr_list1[::-1]]

    corr_list2 = [0.8, 0.7, 0.6, 0.5, 0.4]
    corr_list1 = [-0.4, -0.5, -0.6, -0.7, -0.8]
    edges_weight_list2 = [corre_to_width(a) for a in corr_list2]
    edges_weight_list1 = edges_weight_list2[::-1]
    for i, width in enumerate(edges_weight_list2):
        lines.append(Line2D([], [], linewidth=width, color='r'))
    for i, width in enumerate(edges_weight_list1):
        lines.append(Line2D([], [], linewidth=width, color='b'))
    label_list = ["{:.1f}".format(a) for a in corr_list2+corr_list1]
    ax.legend(lines, label_list, loc='right', frameon=False, prop={'size': 6})


if __name__ == "__main__":
    mode, p, N, corre_method = 'linear', 0.05, 200, 'p'
    plot_network('ADD', mode, p, N, corre_method)
    plot_network('nADD', mode, p, N, corre_method)
    combine('combined_axial_network_{}_method={}'.format(mode, corre_method) + '_N={}'.format(N) + '_p={}'.format(p))





