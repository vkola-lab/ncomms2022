import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np

import collections
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from matplotlib import rc
rc('axes', linewidth=1.5)
rc('font', weight='bold', size=15)

def get_subtype_pool(csv_file, sub_type, pool):
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[sub_type]=='1' and row['ADD']=='0':
                pool[row['filename']] = 0
            if row['ADD'] == '1':
                pool[row['filename']] = 1

def get_score_label(csv_file, pool):
    # get the raw scores and labels from the csv_file for the ROC PR curves
    score, label = [], []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['filename'] in pool:
                score.append(float(row['ADD_score']))
                label.append(pool[row['filename']])
    return score, label

def generate_roc(gt_files, csv_files, sub_type, color, out_file):
    """
    :param csv_files:       a list of csv files as above format
    :param positive_label:  if sub_type == 'FTD', the curve is about ADD vs FTD
                            if sub_type == 'VD',  the curve is about ADD vs VD
                            if sub_type == 'PDD', the curve is about ADD vs PDD
                            if sub_type == 'LBD', the curve is about ADD vs LBD
    :param color:           color of the roc curve
    :param out_file:        image filename you want to save as
    :return:
    """
    lw = 2
    text_size = 20
    fig, ax = plt.subplots(dpi=100)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    pool = {}
    for gt_file in gt_files:
        get_subtype_pool(gt_file, sub_type, pool)
    ADD_count, other_count = 0, 0
    for key in pool:
        if pool[key] == 1: ADD_count += 1
        if pool[key] == 0: other_count += 1
    print('ADD count {} and {} count {}'.format(ADD_count, sub_type, other_count))
    for csvfile in csv_files:
        scores, labels = get_score_label(csvfile, pool)
        fpr, tpr, thres = roc_curve(labels, scores, pos_label=1)
        AUC = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=lw / 2, alpha=0.15)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(AUC)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=color,
            label=r'AUC=%0.3f$\pm$%0.3f' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    legend_properties = {'weight': 'bold', 'size': text_size}
    ax.legend(loc="lower right", prop=legend_properties)
    ax.set_xlabel('False Positive Rate', fontsize=text_size, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=text_size, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.savefig(out_file, bbox_inches='tight')
    fig.clf()
    plt.close()
    return mean_auc

def generate_pr(gt_files, csv_files, sub_type, color, out_file):
    lw = 2
    text_size = 20
    fig, ax = plt.subplots(dpi=100)
    prs = []
    aucs = []
    pool = {}
    for gt_file in gt_files:
        get_subtype_pool(gt_file, sub_type, pool)
    mean_rc = np.linspace(0, 1, 100)
    Labels = []
    for csvfile in csv_files:
        scores, labels = get_score_label(csvfile, pool)
        Labels += labels
        pr, rc, thres = precision_recall_curve(labels, scores, pos_label=1)
        pr, rc = pr[::-1], rc[::-1]
        AUC = average_precision_score(labels, scores, pos_label=1)
        ax.plot(rc, pr, lw=lw/2, alpha=0.15)
        interp_pr = np.interp(mean_rc, rc, pr)
        prs.append(interp_pr)
        aucs.append(AUC)
    mean_pr = np.mean(prs, axis=0)
    mean_auc = np.mean(aucs) # is this right?
    std_auc = np.std(aucs)
    ax.plot(mean_rc, mean_pr, color=color,
            label=r'AP=%0.3f$\pm$%0.3f' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    count = collections.Counter(Labels)
    ratio = count[1] / (count[1] + count[0])
    ax.plot([0, 1], [ratio, ratio], 'k--', lw=lw)
    std_pr = np.std(prs, axis=0)
    prs_upper = np.minimum(mean_pr + std_pr, 1)
    prs_lower = np.maximum(mean_pr - std_pr, 0)
    ax.fill_between(mean_rc, prs_lower, prs_upper, color=color, alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    legend_properties = {'weight': 'bold', 'size': text_size}
    ax.legend(loc="lower left", prop=legend_properties)
    ax.set_xlabel('Recall', fontsize=text_size, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=text_size, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set(xlim=[-0.05, 1.05], ylim=[ratio-0.05, 1.001])
    fig.savefig(out_file, bbox_inches='tight')
    fig.clf()
    plt.close()
    return mean_auc

if __name__ == "__main__":
    # MRI model
    path = '../../tb_log/CNN_baseline_new_cross{}/test_eval.csv'
    csv_files = [path.format(i) for i in range(5)]
    gt_files = ['../../lookupcsv/CrossValid/cross{}/test.csv'.format(i) for i in range(5)]
    generate_roc(gt_files, csv_files, 'FTD', 'red', 'ADD_vs_FTD_NACC_roc.png')
    generate_roc(gt_files, csv_files, 'DLB', 'blue', 'ADD_vs_LBD_NACC_roc.png')
    generate_roc(gt_files, csv_files, 'VD', 'green', 'ADD_vs_VD_NACC_roc.png')
    generate_roc(gt_files, csv_files, 'PDD', 'purple', 'ADD_vs_PDD_NACC_roc.png')
    generate_pr(gt_files, csv_files, 'FTD', 'red', 'ADD_vs_FTD_NACC_pr.png')
    generate_pr(gt_files, csv_files, 'DLB', 'blue', 'ADD_vs_LBD_NACC_pr.png')
    generate_pr(gt_files, csv_files, 'VD', 'green', 'ADD_vs_VD_NACC_pr.png')
    generate_pr(gt_files, csv_files, 'PDD', 'purple', 'ADD_vs_PDD_NACC_pr.png')

    path = '../../tb_log/CNN_baseline_new_cross{}/exter_test_eval.csv'
    csv_files = [path.format(i) for i in range(5)]
    gt_files = ['../../lookupcsv/CrossValid/cross0/exter_test.csv']
    generate_roc(gt_files, csv_files, 'FTD', 'red', 'ADD_vs_FTD_exter_roc.png')
    generate_roc(gt_files, csv_files, 'DLB', 'blue', 'ADD_vs_LBD_exter_roc.png')
    generate_roc(gt_files, csv_files, 'VD', 'green', 'ADD_vs_VD_exter_roc.png')
    generate_roc(gt_files, csv_files, 'PDD', 'purple', 'ADD_vs_PDD_exter_roc.png')
    generate_pr(gt_files, csv_files, 'FTD', 'red', 'ADD_vs_FTD_exter_pr.png')
    generate_pr(gt_files, csv_files, 'DLB', 'blue', 'ADD_vs_LBD_exter_pr.png')
    generate_pr(gt_files, csv_files, 'VD', 'green', 'ADD_vs_VD_exter_pr.png')
    generate_pr(gt_files, csv_files, 'PDD', 'purple', 'ADD_vs_PDD_exter_pr.png')

    #################################################################################
    # Fusion model
    path = '../../tb_log/_CatBoost_Fusion_cross{}/test_mri_eval.csv'
    csv_files = [path.format(i) for i in range(5)]
    gt_files = ['../../lookupcsv/CrossValid_/cross{}/test.csv'.format(i) for i in range(5)]
    generate_roc(gt_files, csv_files, 'FTD', 'red', 'Fusion_ADD_vs_FTD_NACC_roc.png')
    generate_roc(gt_files, csv_files, 'DLB', 'blue', 'Fusion_ADD_vs_LBD_NACC_roc.png')
    generate_roc(gt_files, csv_files, 'VD', 'green', 'Fusion_ADD_vs_VD_NACC_roc.png')
    generate_roc(gt_files, csv_files, 'PDD', 'purple', 'Fusion_ADD_vs_PDD_NACC_roc.png')
    generate_pr(gt_files, csv_files, 'FTD', 'red', 'Fusion_ADD_vs_FTD_NACC_pr.png')
    generate_pr(gt_files, csv_files, 'DLB', 'blue', 'Fusion_ADD_vs_LBD_NACC_pr.png')
    generate_pr(gt_files, csv_files, 'VD', 'green', 'Fusion_ADD_vs_VD_NACC_pr.png')
    generate_pr(gt_files, csv_files, 'PDD', 'purple', 'Fusion_ADD_vs_PDD_NACC_pr.png')

    path = '../../tb_log/_CatBoost_Fusion_cross{}/OASIS_mri_eval.csv'
    csv_files = [path.format(i) for i in range(5)]
    gt_files = ['../../lookupcsv/CrossValid_/cross{}/OASIS.csv'.format(i) for i in range(5)]
    generate_roc(gt_files, csv_files, 'FTD', 'red', 'Fusion_ADD_vs_FTD_OASIS_roc.png')
    generate_roc(gt_files, csv_files, 'DLB', 'blue', 'Fusion_ADD_vs_LBD_OASIS_roc.png')
    generate_roc(gt_files, csv_files, 'VD', 'green', 'Fusion_ADD_vs_VD_OASIS_roc.png')
    generate_roc(gt_files, csv_files, 'PDD', 'purple', 'Fusion_ADD_vs_PDD_OASIS_roc.png')
    generate_pr(gt_files, csv_files, 'FTD', 'red', 'Fusion_ADD_vs_FTD_OASIS_pr.png')
    generate_pr(gt_files, csv_files, 'DLB', 'blue', 'Fusion_ADD_vs_LBD_OASIS_pr.png')
    generate_pr(gt_files, csv_files, 'VD', 'green', 'Fusion_ADD_vs_VD_OASIS_pr.png')
    generate_pr(gt_files, csv_files, 'PDD', 'purple', 'Fusion_ADD_vs_PDD_OASIS_pr.png')


