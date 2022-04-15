import csv
from collections import defaultdict
import random
import numpy as np

def get_dataset_name(path):
    if 'ADNI' in path:
        return 'ADNI'
    elif 'NACC' in path:
        return 'NACC'
    elif 'FHS' in path:
        return 'FHS'
    elif 'OASIS' in path:
        return "OASIS"
    elif 'AIBL' in path:
        return 'AIBL'
    elif 'PPMI' in path:
        return 'PPMI'
    elif 'NIFD' in path:
        return 'NIFD'
    elif "Stanford" in path:
        return "LBDSU"


count = defaultdict(list)
with open('../../lookupcsv/CrossValid/all.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        path, filename =row['path'], row['filename']
        dataname = get_dataset_name(path)
        count[dataname].append(path + filename)

row, col = 8, 7
scans = []
hist = defaultdict(list)
datasets = ["ADNI", "AIBL", "FHS", "OASIS", "PPMI", "NACC", "LBDSU", "NIFD"]
for dataname in datasets:
    sampled = random.sample(count[dataname], col)
    for f in sampled:
        hist[dataname].append(np.load(f.replace('.nii', '.npy')).ravel())
        data = np.load(f.replace('.nii', '.npy'))[100, :, :]
        scans.append(data)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=300, figsize=(row * 0.7, col))
for i in range(1, row * col + 1):
    ax = plt.subplot(row, col, i)
    ax.imshow(scans[i-1], cmap='gray', vmin=0, vmax=8)
    ax.axis('off')
plt.savefig("MRI_sites.png")

# fig, ax = plt.subplots(dpi=300, figsize=(row * 0.7, col))

# for dataname in datasets:
#     data = np.concatenate(hist[dataname])
#     print(data.shape)
#     plt.hist(data, histtype='stepfilled', alpha=0.2, density=True, bins=40, label=dataname, ec='k')
# plt.legend()
# ax.set_xlim(0.3, 8)
# ax.set_ylim(0, 0.3)
# plt.savefig("MRI_sites_hist.png")




