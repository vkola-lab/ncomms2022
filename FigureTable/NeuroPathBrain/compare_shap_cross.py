import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import csv
from scipy.ndimage import zoom
from skimage.segmentation import find_boundaries
from glob import glob

from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', weight='bold', size=13)

def plot_shap_block2conv(shaps, background_idx):
    vmin, vmax = -1.5e-3, 1.5e-3
    name = shaps[0].split('/')[-1].replace('.npy', '_{}.png'.format(background_idx))
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    data = []
    for shap in shaps:
        data.append(np.load(shap))
    data.append(sum(data)/5)
    print([d.shape for d in data])
    for i in range(6):
        axes[i].imshow(data[i][20, :, :], vmin=vmin, vmax=vmax, cmap='bwr')
    plt.savefig('plot/' + name)
    plt.close()


def plot_shap_block2BN(shaps, view='axi', idx=100, alpha=0.1):
    vmin, vmax = -1.5e-2, 1.5e-2
    name = shaps[0].split('/')[-1].replace('.npy', '.png')
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    data = []
    for shap in shaps:
        data.append(np.load(shap))
    data.append(sum(data)/5)
    print([d.shape for d in data])
    for i in range(6):
        axes[i].imshow(data[i][10, :, :], vmin=vmin, vmax=vmax, cmap='bwr')
    plt.savefig('plot/' + name)
    plt.close()


for file in glob('/data_2/NEUROPATH/ADNI_neuroPath/shap_block2conv_cross0_background1/shap_mid/shap_COG_*.npy'):
    pool1 = [file.replace('cross0', 'cross{}'.format(i)) for i in range(5)]
    pool2 = [f.replace('ground1', 'ground2') for f in pool1]
    pool3 = [f.replace('ground1', 'ground3') for f in pool1]
    plot_shap_block2conv(pool1, 1)
    plot_shap_block2conv(pool2, 2)
    plot_shap_block2conv(pool3, 3)


