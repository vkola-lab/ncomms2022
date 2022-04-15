import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import os
import csv
from glob import glob
import matplotlib as mpl
from sklearn.manifold import TSNE
import matplotlib as mpl
import nibabel as nib
from scipy.ndimage import zoom
import scipy.ndimage as ndimage
import math
import json
from skimage.segmentation import find_boundaries
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import collections
from scipy.stats import spearmanr
from scipy import stats

mpl.rcParams['mpl_toolkits.legacy_colorbar'] = False

# -----------------------------------------------------
## customized colormap

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
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

def draw_colorbar():
    data = np.random.rand(100, 100)
    hex_list = ['#0066ff', '#ffffff', '#ff6600']
    COG_cmap = get_continuous_cmap(hex_list)
    plt.imshow(data, cmap=COG_cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([])
    plt.savefig('dummy colorbar custom', dpi=100)
    plt.close()

## customized colormap
# -----------------------------------------------------

# -----------------------------------------------------
## plot shap heatmaps and MRIs

def plot_CNN_shap(filenames, out_folder):
    mri_vmin, mri_vmax, ADD_shap_vmin, ADD_shap_vmax, COG_shap_vmin, COG_shap_vmax, alpha = 0, 5, -4, 4, -4, 4, 0.3
    mri = nib.load(filenames['MRI']).get_data()[6:-6, 6:-6, 6:-6]
    mri = mri / np.mean(mri)
    # mri = np.where(mri < 0.2, mri_vmax + 1, mri)

    seg = nib.load('MRI_process/MNI_segmentation_template.nii.gz').get_data()[6:-6, 6:-6, 6:-6]
    seg = find_boundaries(seg, mode='thick').astype(np.uint8)

    ADD = np.load(filenames['ADD'])
    nADD = np.load(filenames['nADD'])
    std = np.std(ADD)
    ADD_shap_vmin, ADD_shap_vmax = ADD_shap_vmin * std, ADD_shap_vmax * std
    print("ADD shap vmin is {}, vamx is {}".format(ADD_shap_vmin, ADD_shap_vmax))
    ADD_cmap = 'bwr'

    NC = np.load(filenames['NC'])
    MCI = np.load(filenames['MCI'])
    DE = np.load(filenames['DE'])
    std = np.std(NC)
    COG_shap_vmin, COG_shap_vmax = COG_shap_vmin * std, COG_shap_vmax * std
    print("COG shap vmin is {}, vamx is {}".format(COG_shap_vmin, COG_shap_vmax))
    hex_list = ['#0066ff', '#ffffff', '#ff6600']
    COG_cmap = get_continuous_cmap(hex_list)  # blue white orange

    ADD = upsample(ADD, mri.shape)
    nADD = upsample(nADD, mri.shape)
    NC = upsample(NC, mri.shape)
    MCI = upsample(MCI, mri.shape)
    DE = upsample(DE, mri.shape)

    for axi in range(25, 155, 10):
        plot_axial(mri, seg, NC, MCI, DE, ADD, nADD, COG_cmap, ADD_cmap, axi, out_folder, \
                   mri_vmin, mri_vmax, COG_shap_vmin, COG_shap_vmax, ADD_shap_vmin, ADD_shap_vmax, alpha)
    for sag in range(25, 155, 10):
        plot_sagittal(mri, seg, NC, MCI, DE, ADD, nADD, COG_cmap, ADD_cmap, sag, out_folder, \
                      mri_vmin, mri_vmax, COG_shap_vmin, COG_shap_vmax, ADD_shap_vmin, ADD_shap_vmax, alpha)
    for cor in range(45, 165, 10):
        plot_coronal(mri, seg, NC, MCI, DE, ADD, nADD, COG_cmap, ADD_cmap, cor, out_folder, \
                     mri_vmin, mri_vmax, COG_shap_vmin, COG_shap_vmax, ADD_shap_vmin, ADD_shap_vmax, alpha)

def plot_axial(mri, seg, NC, MCI, DE, ADD, nADD, COG_cmap, ADD_cmap, idx, out_folder,
               mri_vmin, mri_vmax, COG_shap_vmin, COG_shap_vmax, ADD_shap_vmin, ADD_shap_vmax, alpha):
    fig = plt.figure(dpi=500)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 7),
                     axes_pad=0.1,
                     aspect=True)
    i = 0
    grid[i].imshow(mri[:, :, idx].transpose((1, 0))[::-1, :], cmap='gray', vmin=mri_vmin, vmax=mri_vmax)
    grid[i].axis('off')
    i += 1
    grid[i].imshow(NC[:, :, idx].transpose((1, 0))[::-1, :], cmap=COG_cmap, vmin=COG_shap_vmin, vmax=COG_shap_vmax)
    grid[i].imshow(seg[:, :, idx].transpose((1, 0))[::-1, :], cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[i].axis('off')

    i += 1
    grid[i].imshow(MCI[:, :, idx].transpose((1, 0))[::-1, :], cmap=COG_cmap, vmin=COG_shap_vmin, vmax=COG_shap_vmax)
    grid[i].imshow(seg[:, :, idx].transpose((1, 0))[::-1, :], cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[i].axis('off')
    i += 1
    grid[i].imshow(DE[:, :, idx].transpose((1, 0))[::-1, :], cmap=COG_cmap, vmin=COG_shap_vmin, vmax=COG_shap_vmax)
    grid[i].imshow(seg[:, :, idx].transpose((1, 0))[::-1, :], cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[i].axis('off')
    i += 1

    grid[i].imshow(mri[:, :, idx].transpose((1, 0))[::-1, :], cmap='gray', vmin=mri_vmin, vmax=mri_vmax)
    grid[i].axis('off')
    i += 1
    grid[i].imshow(ADD[:, :, idx].transpose((1, 0))[::-1, :], cmap=ADD_cmap, vmin=ADD_shap_vmin, vmax=ADD_shap_vmax)
    grid[i].imshow(seg[:, :, idx].transpose((1, 0))[::-1, :], cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[i].axis('off')
    i += 1
    grid[i].imshow(nADD[:, :, idx].transpose((1, 0))[::-1, :], cmap=ADD_cmap, vmin=ADD_shap_vmin, vmax=ADD_shap_vmax)
    grid[i].imshow(seg[:, :, idx].transpose((1, 0))[::-1, :], cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[i].axis('off')

    fig.savefig(out_folder + "CNN_shap_axial_{}.jpeg".format(idx), bbox_inches='tight')
    plt.close(fig)

def plot_sagittal(mri, seg, NC, MCI, DE, ADD, nADD, COG_cmap, ADD_cmap, idx, out_folder,
                  mri_vmin, mri_vmax, COG_shap_vmin, COG_shap_vmax, ADD_shap_vmin, ADD_shap_vmax, alpha):
    fig = plt.figure(dpi=500)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 7),
                     axes_pad=0.1,
                     aspect=True)
    grid[0].imshow(resize(np.rot90(mri[idx, :, :])), cmap='gray', vmin=mri_vmin, vmax=mri_vmax)
    grid[0].axis('off')
    grid[1].imshow(resize(np.rot90(NC[idx, :, :])), cmap=COG_cmap, vmin=COG_shap_vmin, vmax=COG_shap_vmax)
    grid[1].imshow(resize(np.rot90(seg[idx, :, :])), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[1].axis('off')
    grid[2].imshow(resize(np.rot90(MCI[idx, :, :])), cmap=COG_cmap, vmin=COG_shap_vmin, vmax=COG_shap_vmax)
    grid[2].imshow(resize(np.rot90(seg[idx, :, :])), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[2].axis('off')
    grid[3].imshow(resize(np.rot90(DE[idx, :, :])), cmap=COG_cmap, vmin=COG_shap_vmin, vmax=COG_shap_vmax)
    grid[3].imshow(resize(np.rot90(seg[idx, :, :])), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[3].axis('off')

    grid[4].imshow(resize(np.rot90(mri[idx, :, :])), cmap='gray', vmin=mri_vmin, vmax=mri_vmax)
    grid[4].axis('off')
    grid[5].imshow(resize(np.rot90(ADD[idx, :, :])), cmap=ADD_cmap, vmin=ADD_shap_vmin, vmax=ADD_shap_vmax)
    grid[5].imshow(resize(np.rot90(seg[idx, :, :])), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[5].axis('off')
    grid[6].imshow(resize(np.rot90(nADD[idx, :, :])), cmap=ADD_cmap, vmin=ADD_shap_vmin, vmax=ADD_shap_vmax)
    grid[6].imshow(resize(np.rot90(seg[idx, :, :])), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[6].axis('off')

    fig.savefig(out_folder + "CNN_shap_sagittal_{}.jpeg".format(idx), bbox_inches='tight')
    plt.close(fig)

def plot_coronal(mri, seg, NC, MCI, DE, ADD, nADD, COG_cmap, ADD_cmap, idx, out_folder,
                 mri_vmin, mri_vmax, COG_shap_vmin, COG_shap_vmax, ADD_shap_vmin, ADD_shap_vmax, alpha):
    fig = plt.figure(dpi=500)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 7),
                     axes_pad=0.1,
                     aspect=True)
    grid[0].imshow(np.rot90(mri[:, idx, :]), cmap='gray', vmin=mri_vmin, vmax=mri_vmax)
    grid[0].axis('off')
    grid[1].imshow(np.rot90(NC[:, idx, :]), cmap=COG_cmap, vmin=COG_shap_vmin, vmax=COG_shap_vmax)
    grid[1].imshow(np.rot90(seg[:, idx, :]), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[1].axis('off')
    grid[2].imshow(np.rot90(MCI[:, idx, :]), cmap=COG_cmap, vmin=COG_shap_vmin, vmax=COG_shap_vmax)
    grid[2].imshow(np.rot90(seg[:, idx, :]), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[2].axis('off')
    grid[3].imshow(np.rot90(DE[:, idx, :]), cmap=COG_cmap, vmin=COG_shap_vmin, vmax=COG_shap_vmax)
    grid[3].imshow(np.rot90(seg[:, idx, :]), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[3].axis('off')

    grid[4].imshow(np.rot90(mri[:, idx, :]), cmap='gray', vmin=mri_vmin, vmax=mri_vmax)
    grid[4].axis('off')
    grid[5].imshow(np.rot90(ADD[:, idx, :]), cmap=ADD_cmap, vmin=ADD_shap_vmin, vmax=ADD_shap_vmax)
    grid[5].imshow(np.rot90(seg[:, idx, :]), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[5].axis('off')
    grid[6].imshow(np.rot90(nADD[:, idx, :]), cmap=ADD_cmap, vmin=ADD_shap_vmin, vmax=ADD_shap_vmax)
    grid[6].imshow(np.rot90(seg[:, idx, :]), cmap='binary', vmin=0, vmax=1, alpha=alpha)
    grid[6].axis('off')

    fig.savefig(out_folder + "CNN_shap_coronal_{}.jpeg".format(idx), bbox_inches='tight')
    plt.close(fig)

## plot shap heatmaps and MRIs
# -----------------------------------------------------


# -----------------------------------------------------
## tensor manipulations

def smooth(img):
    return ndimage.gaussian_filter(img, sigma=(3, 3, 3), order=0)

def upsample(heat, target_shape):
    x, y, z = heat.shape
    X, Y, Z = target_shape
    return zoom(heat, (float(X)/x, float(Y)/y, float(Z)/z), mode='nearest')

def resize(mri):
    x, y = mri.shape
    return zoom(mri, (170.0*170.0/(206.0*x), 170.0/y))

def overlay(mri, risk, filename, alpha):
    mri = cv2.imread(mri)
    risk = cv2.imread(risk)
    print(mri.shape, risk.shape)
    for i in range(mri.shape[0]):
        for j in range(mri.shape[1]):
            if mri[i, j, 0] < 10 and mri[i, j, 1] < 10 and mri[i, j, 2] < 10:
                risk[i, j, :] = mri[i, j, :]
            else:
                risk[i, j, :] = alpha * mri[i, j, :] + (1-alpha) * risk[i, j, :]
    cv2.imwrite(filename, risk)

## tensor manipulations
# -----------------------------------------------------

# -----------------------------------------------------
## shap statistics

# get regional sum of shap for each of the 95 regions for N true predicted suject,
# saved numpy array in shape (N, 95)
def region_ADD_shap_sum(tb_log_dir):
    VD, FTD, LBD = [], [], []
    with open('lookupcsv/dataset_table/NACC_ALL/NACC.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['AD'] == '1':
                continue
            if row['VD'] == '1':
                VD.append(row['filename'])
            if row['LBD'] == '1':
                LBD.append(row['filename'])
            if row['FTD'] == '1':
                FTD.append(row['filename'])
    print(len(VD), len(FTD), len(LBD))

    shap_dir = '/data_2/sq/shap_mid/'
    array_ADD, array_nADD, count_ADD, count_nADD = [], [], 0, 0
    array_VD, array_FTD, array_LBD = [], [], []
    for i in range(5):
        with open(tb_log_dir + 'cross{}/'.format(i) + 'test_eval.csv', 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                seg_dir = '/data_2/NACC_ALL/seg/' + row['filename'].replace('.npy', '.nii')
                if not os.path.exists(seg_dir):
                    continue
                if row['ADD'] in ['1', '1.0'] and row['ADD_pred'] in ['1', '1.0']:
                    continue
                    seg = nib.load(seg_dir).get_data()[6:-6, 6:-6, 6:-6]
                    shap = upsample(np.load(shap_dir + 'shap_ADD_' + row['filename']), seg.shape)
                    array_ADD.append(get_region_scores(shap, seg))
                    count_ADD += 1
                    print(count_ADD, count_nADD, array_ADD[-1].shape)
                elif row['ADD'] in ['0', '0.0'] and row['ADD_pred'] in ['0', '0.0']:
                    seg = nib.load(seg_dir).get_data()[6:-6, 6:-6, 6:-6]
                    shap = upsample(np.load(shap_dir + 'shap_ADD_' + row['filename']), seg.shape)
                    array_nADD.append(get_region_scores(shap, seg))
                    count_nADD += 1
                    print(count_ADD, count_nADD, array_nADD[-1].shape)
                    if row['filename'] in VD:
                        array_VD.append(array_nADD[-1])
                    if row['filename'] in FTD:
                        array_FTD.append(array_nADD[-1])
                    if row['filename'] in LBD:
                        array_LBD.append(array_nADD[-1])
                    
    array_ADD, array_nADD = np.array(array_ADD), np.array(array_nADD)
    array_VD = np.array(array_VD)
    array_FTD = np.array(array_FTD)
    array_LBD = np.array(array_LBD)
    print(array_ADD.shape, array_nADD.shape, array_VD.shape, array_FTD.shape, array_LBD.shape)
    #np.save('shap/regional95_avgScore_ADD.npy', array_ADD)
    np.save('shap/regional95_avgScore_nADD.npy', array_nADD)
    np.save('shap/regional95_avgScore_VD.npy', array_VD)
    np.save('shap/regional95_avgScore_FTD.npy', array_FTD)
    np.save('shap/regional95_avgScore_LBD.npy', array_LBD)

def region_COG_shap_sum(tb_log_dir):
    shap_dir = '/data_2/sq/shap_mid/'
    array_NC, array_MCI, array_DE, count_NC, count_MCI, count_DE = [], [], [], 0, 0, 0
    for i in range(5):
        with open(tb_log_dir + 'cross{}/'.format(i) + 'test_eval.csv', 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                seg_dir = '/data_2/NACC_ALL/seg/' + row['filename'].replace('.npy', '.nii')
                if not os.path.exists(seg_dir):
                    continue
                if row['COG'] in ['0', '0.0'] and row['COG_pred'] in ['0', '0.0']:
                    seg = nib.load(seg_dir).get_data()[6:-6, 6:-6, 6:-6]
                    shap = upsample(np.load(shap_dir + 'shap_COG_' + row['filename']), seg.shape)
                    array_NC.append(get_region_scores(shap, seg))
                    count_NC += 1
                    print(count_NC, count_MCI, count_DE)
                elif row['COG'] in ['1', '1.0'] and row['COG_pred'] in ['1', '1.0']:
                    seg = nib.load(seg_dir).get_data()[6:-6, 6:-6, 6:-6]
                    shap = upsample(np.load(shap_dir + 'shap_COG_' + row['filename']), seg.shape)
                    array_MCI.append(get_region_scores(shap, seg))
                    count_MCI += 1
                    print(count_NC, count_MCI, count_DE)
                elif row['COG'] in ['2', '2.0'] and row['COG_pred'] in ['2', '2.0']:
                    seg = nib.load(seg_dir).get_data()[6:-6, 6:-6, 6:-6]
                    shap = upsample(np.load(shap_dir + 'shap_COG_' + row['filename']), seg.shape)
                    array_DE.append(get_region_scores(shap, seg))
                    count_DE += 1
                    print(count_NC, count_MCI, count_DE)
    array_NC, array_MCI, array_DE = np.array(array_NC), np.array(array_MCI), np.array(array_DE)
    print(array_NC.shape, array_MCI.shape, array_DE.shape)
    np.save('shap/regional95_avgScore_NC.npy', array_NC)
    np.save('shap/regional95_avgScore_MCI.npy', array_MCI)
    np.save('shap/regional95_avgScore_DE.npy', array_DE)

def get_region_scores(heatmap, seg):
    pool = [[] for _ in range(96)]
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                pool[int(seg[i, j, k])].append(heatmap[i, j, k])
    ans = []
    for i in range(1, 96):
        ans.append(sum(pool[i]))
    return np.array(ans)

def get_region_sum_scores(heatmap, seg):
    region_idx = \
    [('hippocampus', [1, 2]),
     ('temporal', [3, 5, 7, 9, 11, 13, 15, 31, 83, 4, 6, 8, 10, 12, 14, 16, 30, 82]),
     ('cerebellum', [17, 18]),
     ('brainstem', [19, 74, 75]),
     ('insula', [21, 87, 89, 91, 93, 95, 20, 86, 88, 90, 92, 94]),
     ('occipital', [23, 65, 67, 22, 64, 66]),
     ('frontal', [29, 51, 53, 55, 57, 59, 69, 71, 73, 77, 79, 81, 28, 50, 52, 54, 56, 58, 68, 70, 72, 76, 78, 80]),
     ('parietal', [33, 61, 63, 85, 32, 60, 62, 84]),
     ('ventricle', [49, 48, 47, 46, 45])]
    pool = [[] for _ in range(96)]
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                pool[int(seg[i, j, k])].append(heatmap[i, j, k])
    ans = []
    for region, idxes in region_idx:
        Pool = []
        for idx in idxes:
            Pool += pool[idx]
        sum_ = sum(Pool)
        ans.append(sum_)
    return ans

def shap_region_scores_to_csv(data_dir, task):
    shap_dir = data_dir + 'shap_mid/'
    seg_dir = data_dir + 'seg/'
    f = open('shap_{}_region_scores.csv'.format(task), 'w')
    fieldnames = ['filename'] + get_95_region_names()
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for file in glob(seg_dir + '*.nii'):
        filename = file.split('/')[-1].replace('.nii', '.npy')
        seg = nib.load(file).get_data()[6:-6, 6:-6, 6:-6]
        shap = upsample(np.load(shap_dir + 'shap_{}_'.format(task) + filename), seg.shape)
        ans = get_region_scores(shap, seg)
        print(seg.shape, shap.shape, ans.shape)
        case = {'filename': filename}
        for i in range(1, 96):
            case[str(i)] = ans[i-1]
        writer.writerow(case)
    f.close()

def get_95_region_names():
    return [str(i) for i in range(1, 96)]

## shap statistics
# -----------------------------------------------------


if __name__ == "__main__":
    pass
    # plot_CNN_shap(
    #     {'MRI': 'MRI_process/MNI152_T1_1mm_brain.nii',
    #      'ADD': '/data_2/sq/shap_mid/ADD.npy',
    #      'nADD': '/data_2/sq/shap_mid/nADD.npy',
    #      'NC': '/data_2/sq/shap_mid/NC.npy',
    #      'MCI': '/data_2/sq/shap_mid/MCI.npy',
    #      'DE': '/data_2/sq/shap_mid/DE.npy',
    #      'COG_abs_mean': '/data_2/sq/shap_mid/COG_abs.npy',
    #      'ADD_abs_mean': '/data_2/sq/shap_mid/ADD_abs.npy'},
    #     out_folder='tb_log/CNN_baseline_new_cross0/shap/')

    region_ADD_shap_sum('tb_log/CNN_baseline_new_')
    # region_COG_shap_sum('tb_log/CNN_baseline_new_')

    # shap_region_scores_to_csv('/data_2/NACC_ALL/ADNI_neuroPath/', 'ADD')



