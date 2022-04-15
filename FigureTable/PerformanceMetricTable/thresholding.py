import csv
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
print(parentdir)
sys.path.append(parentdir)
from performance_eval import *
import numpy as np
from tqdm import tqdm
import random
random.seed(1001)

def downsample(pool):
    random.shuffle(pool)
    new_pool = [[], [], [], []]
    for p in pool:
        if p[0] == '2' and p[1] in ['1', '1.0']:
            new_pool[2].append(p)
        if p[0] == '2' and p[1] in ['0', '0.0']:
            new_pool[3].append(p)
        if p[0] == '1':
            new_pool[1].append(p)
        if p[0] == '0':
            new_pool[0].append(p)
    min_count = min(len(a) for a in new_pool)
    return new_pool[0][:min_count] + new_pool[1][:min_count] + new_pool[2][:min_count] + new_pool[3][:min_count]

def threshold(pool, cut = {'NC': 0.5, 'DE': 1.5, 'AD': 0.5}):
    new_pool = []
    for cog, add, cog_score, add_score in pool:
        cog_score, add_score = float(cog_score), float(add_score)
        if cog_score < cut['NC']:
            cog_pred = 0
        elif cut['NC'] <= cog_score < cut['DE']:
            cog_pred = 1
        else:
            cog_pred = 2
        if add_score < cut['AD']:
            add_pred = 0
        else:
            add_pred = 1
        new_pool.append({'COG':cog, 'ADD':add, 'COG_pred':cog_pred, 'ADD_pred':add_pred})
    return new_pool

def get_4ways_macro_accuracy(csv_files):
    metric_list = []
    for csv_file in csv_files:
        metric_list.append(get_accuracy(cm_4_4(csv_file)))
    metric_list = np.array(metric_list)
    return np.mean(metric_list)

def get_threshold(valid_csv, balanced=True, steps=20):
    # this function will get the optimal threshold from the validation data
    valid_data = []
    with open(valid_csv, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data = [row['COG'], row['ADD'], row['COG_score'], row['ADD_score']]
            valid_data.append(data)
    if balanced:
        # downsample from the validation set to make it class balanced
        downsampled_valid_data = []
        for _ in range(5):
            downsampled_valid_data += downsample(valid_data)
        valid_data = downsampled_valid_data
    cuts = []
    for nc_ in np.linspace(0.2, 1.0, steps):
        for de_ in np.linspace(1.1, 2.0, 20+steps):
            for ad_ in np.linspace(0.01, 0.99, steps):
                cut = {'NC': nc_, 'DE': de_, 'AD':ad_}
                cuts.append(cut)
    results = []
    for cut in tqdm(cuts):
        threshold_valid_data = threshold(valid_data, cut)
        with open('tmp/valid_thresholding.csv', 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['COG', 'COG_pred', 'ADD', 'ADD_pred'])
            writer.writeheader()
            for data in threshold_valid_data:
                writer.writerow(data)
        validation_accu = get_4ways_macro_accuracy(['tmp/valid_thresholding.csv'])
        results.append((validation_accu, cut['NC'], cut['DE'], cut['AD']))
    results.sort()
    cut = {'NC': float(results[-1][1]),
           'DE': float(results[-1][2]),
           'AD': float(results[-1][3])}
    print('the optimal cut is:', cut)
    return cut

def apply_threshold(csvfile, cut):
    # apply the optimal threshold on the testing data
    # save the prediction into a csv file with thres as suffix
    if "eval.csv" not in csvfile:
        print('eval.csv key word not found in', test_csv)
        return
    data = []
    with open(csvfile, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            d = [row['COG'], row['ADD'], row['COG_score'], row['ADD_score']]
            data.append(d)
    thresholded_data = threshold(data, cut)
    with open(csvfile.replace('eval.csv', 'eval_thres.csv'), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['COG', 'COG_pred', 'ADD', 'ADD_pred'])
        writer.writeheader()
        for data in thresholded_data:
            writer.writerow(data)

def threshold_crossValid(valid_files, test_files, oasis_files, balanced=True):
    for fold_idx, (valid_csv, test_csv, oasis_file) in enumerate(zip(valid_files, test_files, oasis_files)):
        cut = get_threshold(valid_csv, balanced, steps=20)
        apply_threshold(test_csv, cut)
        apply_threshold(oasis_file, cut)

if __name__ == '__main__':
    # valid_files = [parentdir + '/tb_log/CNN_3ways_special1_cross{}/valid_eval_after.csv'.format(i) for i in range(5)]
    # test_files = [parentdir + '/tb_log/CNN_3ways_special1_cross{}/neuro_test_eval_after.csv'.format(i) for i in range(5)]
    # valid_files = [parentdir + '/tb_log/CatBoost_CrossValid_nonImg_cross{}/valid_eval.csv'.format(i) for i in range(5)]
    # test_files = [parentdir + '/tb_log/CatBoost_CrossValid_nonImg_cross{}/test_eval.csv'.format(i) for i in range(5)]
    # oasis_files = [parentdir + '/tb_log/CatBoost_CrossValid_nonImg_cross{}/OASIS_eval.csv'.format(i) for i in range(5)]
    # valid_files = [parentdir + '/tb_log/CatBoost_Fusion_cross{}/valid_mri_eval.csv'.format(i) for i in range(5)]
    # test_files = [parentdir + '/tb_log/CatBoost_Fusion_cross{}/test_mri_eval.csv'.format(i) for i in range(5)]
    # oasis_files = [parentdir + '/tb_log/CatBoost_Fusion_cross{}/OASIS_mri_eval.csv'.format(i) for i in range(5)]
    # valid_files = [parentdir + '/tb_log/CNN_baseline_new_cross{}/valid_eval.csv'.format(i) for i in range(5)]
    # test_files = [parentdir + '/tb_log/CNN_baseline_new_cross{}/test_eval.csv'.format(i) for i in range(5)]
    # oasis_files = [parentdir + '/tb_log/CNN_baseline_new_cross{}/OASIS_eval.csv'.format(i) for i in range(5)]
    # external_files = [parentdir + '/tb_log/CNN_baseline_new_cross{}/exter_test_eval.csv'.format(i) for i in range(5)]

    # model_name = "_CatBoost"
    # valid_files = [parentdir + '/tb_log/{}_NonImg_cross{}/valid_eval.csv'.format(model_name, i) for i in range(5)]
    # test_files = [parentdir + '/tb_log/{}_NonImg_cross{}/test_eval.csv'.format(model_name, i) for i in range(5)]
    # oasis_files = [parentdir + '/tb_log/{}_NonImg_cross{}/OASIS_eval.csv'.format(model_name, i) for i in range(5)]
    # threshold_crossValid(valid_files, test_files, oasis_files, balanced=False)

    # model_name = "_XGBoost"
    # valid_files = [parentdir + '/tb_log/{}_Fusion_cross{}/valid_mri_eval.csv'.format(model_name, i) for i in range(5)]
    # test_files = [parentdir + '/tb_log/{}_Fusion_cross{}/test_mri_eval.csv'.format(model_name, i) for i in range(5)]
    # oasis_files = [parentdir + '/tb_log/{}_Fusion_cross{}/OASIS_mri_eval.csv'.format(model_name, i) for i in range(5)]
    # threshold_crossValid(valid_files, test_files, oasis_files, balanced=False)

    combinations = ["demo", "demo+np", "demo+func", "demo+his",
                    "demo+np+func", "demo+np+his", "demo+np+his+func",
                    "MRI+demo", "MRI+demo+np", "MRI+demo+func", "MRI+demo+his",
                    "MRI+demo+np+func", "MRI+demo+np+his", "MRI+demo+np+his+func"]
    for model_name in combinations:
        valid_files = [parentdir + '/tb_log/{}_cross{}/valid_mri_eval.csv'.format(model_name, i) for i in range(5)]
        test_files = [parentdir + '/tb_log/{}_cross{}/test_mri_eval.csv'.format(model_name, i) for i in range(5)]
        oasis_files = [parentdir + '/tb_log/{}_cross{}/OASIS_mri_eval.csv'.format(model_name, i) for i in range(5)]
        threshold_crossValid(valid_files, test_files, oasis_files, balanced=False)
