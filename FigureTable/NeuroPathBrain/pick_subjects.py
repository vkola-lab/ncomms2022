import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import csv
from scipy.ndimage import zoom
from skimage.segmentation import find_boundaries
from scipy import stats

from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', weight='bold', size=13)

prefix_idx = {'CG_1': [24],
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

def normalize():
    seg = nib.load('/data_2/NEUROPATH/FHS_neuroPath/seg/0-4556_20060726.nii').get_data()
    region_volume = {}
    for pre in prefix_idx:
        count = 0
        for idx in prefix_idx[pre]:
            count += np.count_nonzero(seg == idx)
        region_volume[pre] = count
    return region_volume

def rank_all(task, stain, layer):
    region_volume = normalize()
    shap = {}
    with open('../NeuroPathRegions/shap_csvfiles/ADNI_shap_{}_region_scores_{}.csv'.format(task, layer), 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            shap[row['filename']] = {i : row[str(i)] for i in range(1, 96)}
    with open('../NeuroPathRegions/shap_csvfiles/FHS_shap_{}_region_scores_{}.csv'.format(task, layer), 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            shap[row['filename']] = {i : row[str(i)] for i in range(1, 96)}
    cases = []
    with open('../NeuroPathTable/ALL.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['diff_days'] and int(row['diff_days']) < 365 * 2:
                c = get_corr(row, stain, shap, region_volume)
                if c != -1 or np.isnan(c):
                    cases.append([c, row['id'], row['Group']])
    cases.sort()
    for c in cases:
        print(c)

def get_corr(row, stain, shap, region_volume):
    vec1, vec2 = [], []
    for reg in prefix_idx:
        if row[reg + '_' + stain]:
            vec1.append(int(row[reg + '_' + stain]))
            shap_value = 0
            for idx in prefix_idx[reg]:
                shap_value += float(shap[row['filename']][idx])
            vec2.append(shap_value / region_volume[reg])
    assert len(vec1) == len(vec2), 'length not equal'
    if not vec1:
        return -1
    if len(set(vec1)) == 1:
        print('neuropath is constant')
        return -1
    c, p = stats.spearmanr(vec1, vec2)
    return c

if __name__ == "__main__":
    rank_all('COG', 'AB_DP', 'block2conv')

