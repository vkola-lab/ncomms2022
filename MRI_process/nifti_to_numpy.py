import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
import os
import numpy as np
from tqdm import tqdm

source = '/home/sq/FHS_NP_/scans_uniform/'
target = '/data_2/NEUROPATH/FHS_neuroPath/npy/'

if not os.path.exists(target):
    os.mkdir(target)

fileList = glob(source+'*.nii')

def normalize(data):
    data = data / np.mean(data)
    data = np.clip(data, 0, 8)
    return data

def show_scan(data, filename):
    image_name = filename.replace('npy', 'jpg')
    plt.subplot(1, 3, 1)
    plt.imshow(data[100, :, :], vmin=0, vmax=8)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(data[:, 100, :], vmin=0, vmax=8)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(data[:, :, 100], vmin=0, vmax=8)
    plt.colorbar()
    plt.savefig(target+image_name)
    plt.close()

for file in tqdm(fileList):
    data = nib.load(file).get_data()
    filename = file.split('/')[-1].replace('.nii', '.npy')
    data = normalize(data)
    np.save(target+filename, data)
    show_scan(data, filename)


