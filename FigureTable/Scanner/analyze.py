import csv
from collections import defaultdict
import random
import numpy as np

count = {}
with open('analyze_table.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        label, scanner = row['diag'], row['scanner']
        if label not in count:
            count[label] = defaultdict(list)
        count[label][scanner].append(row['dir'])

scans = []
for scanner in ["GE", "Siemens", "Philips"]:
    filenames = count['NC'][scanner]
    sampled = random.sample(filenames, 7)
    for f in sampled:
        data = np.load(f)[100, :, :]
        scans.append(data)

import matplotlib.pyplot as plt

row, col = 3, 7

# for i in range(1, 22):
#     ax = plt.subplot(3, 7, i)
#     ax.imshow(scans[i-1], cmap='gray', vmin=0, vmax=8)
#     ax.axis('off')
# plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(5.6, 2.7))
for i in range(1, row * col + 1):
    ax = plt.subplot(row, col, i)
    ax.imshow(scans[i-1], cmap='gray', vmin=0, vmax=8)
    ax.axis('off')
plt.savefig("MRI_scanners.png")