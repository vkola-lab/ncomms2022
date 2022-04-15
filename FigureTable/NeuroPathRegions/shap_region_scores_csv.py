import csv
import nibabel as nib
from scipy.ndimage import zoom
from glob import glob
import numpy as np

def upsample(heat, target_shape=(182, 218, 182), margin=0):
    background = np.zeros(target_shape)
    x, y, z = heat.shape
    X, Y, Z = target_shape
    X, Y, Z = X - 2 * margin, Y - 2 * margin, Z - 2 * margin
    data = zoom(heat, (float(X)/x, float(Y)/y, float(Z)/z), mode='nearest')
    background[margin:margin+data.shape[0], margin:margin+data.shape[1], margin:margin+data.shape[2]] = data
    return background

def shap_region_scores_to_csv(data_dir, task, dataname, layername):
    shap_dir = data_dir + 'shap/' + layername + '/'
    seg_dir = data_dir + 'seg/'
    f = open('FigureTable/NeuroPathRegions/shap_csvfiles/' + dataname + '_shap_{}_region_scores_{}.csv'.format(task, layername), 'w')
    fieldnames = ['filename'] + get_95_region_names()
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for file in glob(seg_dir + '*.nii'):
        filename = file.split('/')[-1].replace('.nii', '.npy')
        seg = nib.load(file).get_data()
        if layername == 'block2conv':
            shap = upsample(np.load(shap_dir + 'shap_{}_'.format(task) + filename), margin=6)
        elif layername == 'block2pooling':
            shap = upsample(np.load(shap_dir + 'shap_{}_'.format(task) + filename), margin=12)
        elif layername == 'block2BN':
            shap = upsample(np.load(shap_dir + 'shap_{}_'.format(task) + filename), margin=16)
        ans = get_region_scores(shap, seg)
        print(seg.shape, shap.shape, ans.shape)
        case = {'filename': filename}
        for i in range(1, 96):
            case[str(i)] = ans[i-1]
        writer.writerow(case)
    f.close()

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

def get_95_region_names():
    return [str(i) for i in range(1, 96)]

if __name__ == "__main__":
    # shap_region_scores_to_csv('/data_2/NEUROPATH/NACC_neuroPath/', 'ADD', 'NACC')
    # shap_region_scores_to_csv('/data_2/NEUROPATH/NACC_neuroPath/', 'COG', 'NACC')

    for layername in ['block2conv', 'block2pooling', 'block2BN']:

        shap_region_scores_to_csv('/data_2/NEUROPATH/ADNI_neuroPath/', 'ADD', 'ADNI', layername)
        shap_region_scores_to_csv('/data_2/NEUROPATH/ADNI_neuroPath/', 'COG', 'ADNI', layername)

        shap_region_scores_to_csv('/data_2/NEUROPATH/FHS_neuroPath/', 'ADD', 'FHS', layername)
        shap_region_scores_to_csv('/data_2/NEUROPATH/FHS_neuroPath/', 'COG', 'FHS', layername)
