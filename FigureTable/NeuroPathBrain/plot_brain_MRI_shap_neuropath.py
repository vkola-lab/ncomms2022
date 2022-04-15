import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import csv
from scipy.ndimage import zoom
from skimage.segmentation import find_boundaries
import os

from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', weight='bold', size=12)

def get_continuous_cmap(hex_list, float_list=None):
    def rgb_to_dec(value):
        return [v / 256 for v in value]
    def hex_to_rgb(value):
        value = value.strip("#")  # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def resize(mri, a=182.0, b=218.0):
    a, b = float(a), float(b)
    x, y = mri.shape
    return zoom(mri, (a*a/(b*x), a/y))

def upsample(heat, target_shape=(182, 218, 182), margin=0):
    background = np.zeros(target_shape)
    x, y, z = heat.shape
    X, Y, Z = target_shape
    X, Y, Z = X - 2 * margin, Y - 2 * margin, Z - 2 * margin
    data = zoom(heat, (float(X)/x, float(Y)/y, float(Z)/z), mode='nearest')
    background[margin:margin+data.shape[0], margin:margin+data.shape[1], margin:margin+data.shape[2]] = data
    return background

def parse_neuropath_regional_scores(row, stain):
    regions =  ['CG_1', 'FL_mfg_7', 'FL_pg_10', 'TL_stg_32', 'PL_ag_20', 'Amygdala_24',
                'TL_hippocampus_28', 'TL_parahippocampal_30', 'c_37', 'bs_35', 'sn_48', 'th_49',
                'pal_46', 'na_45X', 'cn_36C', 'OL_17_18_19OL']
    ans = {}
    for reg in regions:
        ans[reg] = row[reg + '_' + stain]
    return ans

def plot_MRI(ax, filedir, view='axi', idx=100, mri_vmin=0, mri_vmax=7):
    mri = np.load(filedir)
    if view == 'axi':
        ax.imshow(mri[:, :, idx].transpose((1, 0))[::-1, ::-1], cmap='gray', vmin=mri_vmin, vmax=mri_vmax)
    elif view == 'sag':
        ax.imshow(resize(np.rot90(mri[idx, :, :])), cmap='gray', vmin=mri_vmin, vmax=mri_vmax)
    elif view == 'cor':
        ax.imshow(np.rot90(mri[:, idx, :])[:, ::-1], cmap='gray', vmin=mri_vmin, vmax=mri_vmax)
    ax.axis('off')
    ax.set_title('MRI', fontweight='bold')

def plot_shap(ax, shapdir, seg_b, view='axi', idx=100, alpha=0.1, layer='block2conv'):
    factor = 0.3
    if layer == 'block2conv':
        shap = upsample(np.load(shapdir), margin=6)
    elif layer == 'block2pooling':
        shap = upsample(np.load(shapdir), margin=12)
    elif layer == 'block2BN':
        shap = upsample(np.load(shapdir), margin=16)
    if 'ADD' in shapdir:
        cmap = 'bwr'
        vmin, vmax = -6.9e-5, 6.9e-5
    elif 'COG' in shapdir:
        hex_list = ['#0066ff', '#ffffff', '#ff6600']
        cmap = get_continuous_cmap(hex_list)
        vmin, vmax = -1.5e-3, 1.5e-3
    if view == 'axi':
        img = shap[:, :, idx]
        range = max(np.amax(img), -np.amin(img)) * factor
        ax.imshow(shap[:, :, idx].transpose((1, 0))[::-1, ::-1], cmap=cmap, vmin=-range, vmax=range)
        ax.imshow(seg_b[:, :, idx].transpose((1, 0))[::-1, ::-1], cmap='binary', vmin=0, vmax=1, alpha=alpha)
    elif view == 'sag':
        img = shap[idx, :, :]
        range = max(np.amax(img), -np.amin(img)) * factor
        ax.imshow(resize(np.rot90(shap[idx, :, :])), cmap=cmap, vmin=-range, vmax=range)
        ax.imshow(resize(np.rot90(seg_b[idx, :, :])), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    elif view == 'cor':
        img = shap[:, idx, :]
        range = max(np.amax(img), -np.amin(img)) * factor
        ax.imshow(np.rot90(shap[:, idx, :])[:, ::-1], cmap=cmap, vmin=-range, vmax=range)
        ax.imshow(np.rot90(seg_b[:, idx, :])[:, ::-1], cmap='binary', vmin=0, vmax=1, alpha=alpha)
    ax.axis('off')
    ax.set_title('SHAP ' + layer, fontweight='bold')

def plot_neuropath(ax, seg_b, seg_c, view='axi', idx=100, alpha=0.1, title=None):
    vmin, vmax = -0.5, 3.5
    cmap = matplotlib.cm.get_cmap('bwr', 4)
    cmap.set_over('white')
    if view == 'axi':
        im = ax.imshow(seg_c[:, :, idx].transpose((1, 0))[::-1, ::-1], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.imshow(seg_b[:, :, idx].transpose((1, 0))[::-1, ::-1], cmap='binary', vmin=0, vmax=1, alpha=alpha)
    elif view == 'sag':
        im = ax.imshow(resize(np.rot90(seg_c[idx, :, :])), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.imshow(resize(np.rot90(seg_b[idx, :, :])), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    elif view == 'cor':
        im = ax.imshow(np.rot90(seg_c[:, idx, :])[:, ::-1], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.imshow(np.rot90(seg_b[:, idx, :])[:, ::-1], cmap='binary', vmin=0, vmax=1, alpha=alpha)
    ax.axis('off')
    ax.set_title(title, fontweight='bold')

def convertSeg(seg, np_data):
    prefix_idx = {'CG_1': [24,25,26,27],
                  'FL_mfg_7': [28,29,52,53,54,55,56,57,58,59,68,69,70,71,72,73,76,77,80,81],
                  'FL_pg_10': [50,51,78,79],
                  'TL_stg_32': [5,6,7,8,11,12,30,31,82,83],
                  'PL_ag_20': [32,33,60,61,62,63,84,85],
                  'Amygdala_24': [3,4],
                  'TL_hippocampus_28': [1,2],
                  'TL_parahippocampal_30': [9,10],
                  'c_37': [17,18],
                  'bs_35': [19],
                  'sn_48': [74,75],
                  'th_49': [40,41],
                  'pal_46': [42,43],
                  'na_45X': [36,37],
                  'cn_36C': [34,35],
                  'OL_17_18_19OL': [22,23,64,65,66,67]}
    # prefix_idx = {'CG_1': [24],
    #               'FL_mfg_7': [28],
    #               'FL_pg_10': [50],
    #               'TL_stg_32': [82],
    #               'PL_ag_20': [32],
    #               'Amygdala_24': [4],
    #               'TL_hippocampus_28': [2],
    #               'TL_parahippocampal_30': [10],
    #               'c_37': [18],
    #               'bs_35': [19],
    #               'sn_48': [74],
    #               'th_49': [40],
    #               'pal_46': [42],
    #               'na_45X': [36],
    #               'cn_36C': [34],
    #               'OL_17_18_19OL': [64, 66, 22]}
    mapping = {}
    for reg in np_data:
        if np_data[reg]:
            for idx in prefix_idx[reg]:
                mapping[idx] = int(np_data[reg])
    new_seg = np.ones(seg.shape) * 100
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                if seg[i, j, k] in mapping:
                    new_seg[i, j, k] = mapping[seg[i, j, k]]
    return new_seg

def plot(mri_dir, shap_dirs, seg_dir, np_data, label, view, slide_idx, id):
    if not os.path.exists("plot/{}_{}/".format(id, label)):
        os.mkdir("plot/{}_{}/".format(id, label))
    seg = nib.load(seg_dir).get_data()
    seg_b = find_boundaries(seg, mode='thick').astype(np.uint8)
    seg_converted = [convertSeg(seg, np_data[i]) for i in range(4)]
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1, 8, figsize=(24, 3), dpi=300)
    plot_MRI(ax1, mri_dir, view=view, idx=slide_idx)
    plot_shap(ax2, shap_dirs[0], seg_b, view=view, idx=slide_idx, layer='block2conv')
    plot_shap(ax3, shap_dirs[1], seg_b, view=view, idx=slide_idx, layer='block2pooling')
    plot_shap(ax4, shap_dirs[2], seg_b, view=view, idx=slide_idx, layer='block2BN')
    plot_neuropath(ax5, seg_b, seg_converted[0], view=view, idx=slide_idx, title=stains[0])
    plot_neuropath(ax6, seg_b, seg_converted[1], view=view, idx=slide_idx, title=stains[1])
    plot_neuropath(ax7, seg_b, seg_converted[2], view=view, idx=slide_idx, title=stains[2])
    plot_neuropath(ax8, seg_b, seg_converted[3], view=view, idx=slide_idx, title=stains[3])
    plt.savefig("plot/{}_{}/{}_{}.png".format(id, label, view, slide_idx), bbox_inches='tight')
    plt.close()

def plot_all(task, stains, layers):
    mri_id_map, shap_id_map, seg_id_map = {}, {}, {}
    np_id_data, label_id_data = {}, {}
    with open('../NeuroPathTable/ALL.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['dataset'] in ['ADNI', 'FHS'] and row['diff_days'] and int(row['diff_days']) < 365 * 2:
                mri_id_map[row['id']] = '/data_2/NEUROPATH/{}_neuroPath/npy/'.format(row['dataset']) + row['filename']
                shap_id_map[row['id']] = ['/data_2/NEUROPATH/{}_neuroPath/shap/{}/shap_{}_'.format(row['dataset'], layer, task) + row['filename'] for layer in layers]
                seg_id_map[row['id']] = '/data_2/NEUROPATH/{}_neuroPath/seg/'.format(row['dataset']) + row['filename'].replace('.npy', '.nii')
                np_id_data[row['id']] = [parse_neuropath_regional_scores(row, stain) for stain in stains]
                label_id_data[row['id']] = row['Group']
    # for id in ['0565', '0880', '0691', '4910', '4802', '0821', '0492', '1271', '5017', '4936', '4223']:
    #     for slide_idx in range(80, 140, 10):
    #         plot(mri_id_map[id], shap_id_map[id], seg_id_map[id], np_id_data[id], label_id_data[id], 'sag', slide_idx, id)
    #     for slide_idx in range(30, 130, 10):
    #         plot(mri_id_map[id], shap_id_map[id], seg_id_map[id], np_id_data[id], label_id_data[id], 'axi', slide_idx, id)
    #     for slide_idx in range(50, 150, 10):
    #         plot(mri_id_map[id], shap_id_map[id], seg_id_map[id], np_id_data[id], label_id_data[id], 'cor', slide_idx, id)
    id = '4910'
    plot(mri_id_map[id], shap_id_map[id], seg_id_map[id], np_id_data[id], label_id_data[id], 'axi', 50, id)
    plot(mri_id_map[id], shap_id_map[id], seg_id_map[id], np_id_data[id], label_id_data[id], 'cor', 130, id)
    plot(mri_id_map[id], shap_id_map[id], seg_id_map[id], np_id_data[id], label_id_data[id], 'sag', 110, id)


if __name__ == "__main__":
    stains = ['AB_DP', 'TAU_NFT', 'TAU_NP', 'SILVER_NFT']
    plot_all('COG', stains, ['block2conv', 'block2pooling', 'block2BN'])

