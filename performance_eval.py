import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from utils import plot_confusion_matrix

"""
this script contains the utils functions to evaluate model's performance

to use the functions in this script, we are expecting the users have already generated a csv
table in the following format
    ---------------------------------------------------------------------------
    | ID | filename | COG_score | ADD_score | COG_pred | ADD_pred | COG | ADD |
    ---------------------------------------------------------------------------
where the COG_score is the continuous regression score from the model
      the ADD_score is the probability of being ADD
      the COG_pred  is the predicted label (NC=0, MCI=1, DE=2)
      the ADD_pred  is the predicted label (ADD=1, nADD=0)
      the COG       is the true label (NC=0, MCI=1, DE=2)
      the ADD       is the true label (ADD=1, nADD=0)
"""

import numpy as np
import collections
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from matplotlib import rc
rc('axes', linewidth=1.5)
rc('font', weight='bold', size=15)

# generate confusion matrix from the a csv_file
# ----------------------------------------------------------------------------------
def cm_3_3(csv_file):
    # this function generate a 3 by 3 confusion matrix for the COG task which is a
    # 3 way classification between NC=0, MCI=1, DE=2
    pred, label = [], []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row['COG']: continue
            pred.append(int(row['COG_pred']))
            label.append(int(row['COG'][0]))
    return confusion_matrix(label, pred, labels=[0, 1, 2])

def cm_2_2(csv_file):
    # this function generate a 2 by 2 confusion matrix for the ADD task which is a
    # binary classification between ADD=1 and nADD=0
    pred, label = [], []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row['ADD']: continue
            pred.append(int(row['ADD_pred']))
            label.append(int(row['ADD'][0]))
    return confusion_matrix(label, pred, labels=[0, 1])

def cm_4_4(csv_file):
    # this function generate a 4 by 4 confusion matrix for the ADD task which is a
    # 4 ways classification between NC=0, MCI=1, ADD=2 and nADD=3
    pred, label = [], []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row['COG']: continue
            pred.append(to_4labels(row['COG_pred'], row['ADD_pred']))
            label.append(to_4labels(row['COG'], row['ADD']))
    return confusion_matrix(label, pred, labels=[0, 1, 2, 3])

def to_4labels(cog, add):
    if cog[0] == '0': return 0  # NC
    if cog[0] == '1': return 1  # MCI
    if cog[0] == '2':
        if add[0] == '1': return 2 # ADD
        if add[0] == '0': return 3 # nADD

def get_accuracy(cm):
    total, correct = 0, 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            total += cm[i, j]
            if i == j: correct += cm[i, j]
    return float(correct) / float(total)

# generate confusion matrix for a five fold cross validation experiment
# pass in csv_files which is a list of csv_file representing each fold

def crossValid_cm_3_3(csv_files):
    cm = np.zeros((len(csv_files), 3, 3))
    for i, csvfile in enumerate(csv_files):
        cm[i] = cm_3_3(csvfile)
    return cm.mean(axis=0)

def crossValid_cm_2_2(csv_files):
    cm = np.zeros((len(csv_files), 2, 2))
    for i, csvfile in enumerate(csv_files):
        cm[i] = cm_2_2(csvfile)
    return cm.mean(axis=0)

def crossValid_cm_4_4(csv_files):
    cm = np.zeros((len(csv_files), 4, 4))
    for i, csvfile in enumerate(csv_files):
        cm[i] = cm_4_4(csvfile)
    return cm.mean(axis=0)

def crossValid_cm(csv_files, stage):
    print('2 by 2 confusion matrix is:')
    cm = crossValid_cm_2_2(csv_files)
    fig = plot_confusion_matrix(cm, ['nADD', 'ADD'])
    fig.savefig(csv_files[0].replace(stage + '_eval.csv', stage + '_cm2.jpg'))
    print(cm)
    print('---------------------------------')
    print('3 by 3 confusion matrix is:')
    cm = crossValid_cm_3_3(csv_files)
    fig = plot_confusion_matrix(cm, ['NC', 'MCI', 'DE'])
    fig.savefig(csv_files[0].replace(stage + '_eval.csv', stage + '_cm3.jpg'))
    print(cm)
    print('---------------------------------')
    print('4 by 4 confusion matrix is:')
    cm = crossValid_cm_4_4(csv_files)
    fig = plot_confusion_matrix(cm, ['NC', 'MCI', 'ADD', 'nADD'])
    fig.savefig(csv_files[0].replace(stage + '_eval.csv', stage + '_cm4.jpg'))
    print(cm)
    print('---------------------------------')






# generate ROC and PR curves
# ----------------------------------------------------------------------------------

def generate_roc(csv_files, positive_label, color, out_file):
    """
    :param csv_files:       a list of csv files as above format
    :param positive_label:  if positive_label == 'NC', the curve is about NC vs not NC
                            if positive_label == 'DE', the curve is about DE vs not DE
                            if positive_label =='ADD', the curve is about ADD vs nADD
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
    for csvfile in csv_files:
        scores, labels = get_score_label(csvfile, positive_label)
        fpr, tpr, thres = roc_curve(labels, scores, pos_label=1)
        AUC = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=lw/2, alpha=0.15)
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


def generate_pr(csv_files, positive_label, color, out_file):
    lw = 2
    text_size = 20
    fig, ax = plt.subplots(dpi=100)
    prs = []
    aucs = []
    mean_rc = np.linspace(0, 1, 100)
    for csvfile in csv_files:
        scores, labels = get_score_label(csvfile, positive_label)
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
    count = collections.Counter(labels)
    ratio = count[1] / (count[1] + count[0])
    ax.plot([0, 1], [ratio, ratio], 'k--', lw=lw)
    std_pr = np.std(prs, axis=0)
    prs_upper = np.minimum(mean_pr + std_pr, 1)
    prs_lower = np.maximum(mean_pr - std_pr, 0)
    ax.fill_between(mean_rc, prs_lower, prs_upper, color=color, alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    legend_properties = {'weight': 'bold', 'size': text_size}
    ax.legend(loc="lower left", prop=legend_properties)
    ax.set_xlabel('Recall', fontsize=text_size, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=text_size, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.savefig(out_file, bbox_inches='tight')
    fig.clf()
    plt.close()
    return mean_auc


def get_score_label(csv_file, positive_label):
    # get the raw scores and labels from the csvfile for the ROC PR curves
    score, label = [], []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row['COG']: continue
            if positive_label == 'NC':
                score.append(-float(row['COG_score']))
                label.append(1 if row['COG'][0] == '0' else 0)
            if positive_label == 'DE':
                score.append(float(row['COG_score']))
                label.append(1 if row['COG'][0] == '2' else 0)
            if positive_label == 'ADD':
                if not row['ADD']: continue
                score.append(float(row['ADD_score']))
                label.append(int(row['ADD'][0]))
    return score, label


def ROC_PR_curves(csv_files, stage):
    nc_roc =  generate_roc(csv_files, 'NC', 'g', csv_files[0].replace(stage + '_eval.csv', stage + '_NC_roc.jpg'))
    de_roc =  generate_roc(csv_files, 'DE', 'darkorange', csv_files[0].replace(stage + '_eval.csv', stage + '_DE_roc.jpg'))
    add_roc = generate_roc(csv_files, 'ADD', 'r', csv_files[0].replace(stage + '_eval.csv', stage + '_ADD_roc.jpg'))
    nc_pr =  generate_pr(csv_files, 'NC', 'g', csv_files[0].replace(stage + '_eval.csv', stage + '_NC_pr.jpg'))
    de_pr =  generate_pr(csv_files, 'DE', 'darkorange', csv_files[0].replace(stage + '_eval.csv', stage + '_DE_pr.jpg'))
    add_pr = generate_pr(csv_files, 'ADD', 'r', csv_files[0].replace(stage + '_eval.csv', stage + '_ADD_pr.jpg'))
    return (nc_pr + nc_roc) + (de_roc + de_pr) + (add_roc + add_pr) * 2


# generate performance metrics table
# ----------------------------------------------------------------------------------

"""
a. NC (sensitivity; specificity, etc)
b. MCI (sensitivity; specificity, etc)
c. DE (sensitivity; specificity, etc)
d. ADD (sensitivity; specificity, etc)
e. nADD (sensitivity; specificity, etc)
f. ADD | DE (sensitivity; specificity, etc)
g. nADD | DE (sensitivity; specificity, etc)
"""

def perform_table(csv_files, output_name):
    ans = 0
    content = []
    cache = {}
    for metric in ['Accuracy', 'F-1', 'Sensitivity', 'Specificity', 'MCC']:
        row = [metric]
        cache[metric] = {}
        for task in ['NC', 'MCI', 'DE', 'ADD', 'nADD', 'ADD|DE', 'nADD|DE', 'macro-3ways', 'macro-4ways']:
            metric_list = []
            cache[metric][task] = {}
            for csv_file in csv_files:
                pd, gt = get_pd_gt(csv_file, task)
                if metric == 'Accuracy':
                    if task == 'macro-3ways':
                        metric_list.append(get_accuracy(cm_3_3(csv_file)))
                    elif task == 'macro-4ways':
                        metric_list.append(get_accuracy(cm_4_4(csv_file)))
                    else:
                        metric_list.append(accu_(gt, pd))
                elif metric == 'F-1':
                    if task == 'macro-3ways':
                        val = sum(cache[metric][key][csv_file] for key in ['NC', 'MCI', 'DE']) / 3.0
                    elif task == 'macro-4ways':
                        val = sum(cache[metric][key][csv_file] for key in ['NC', 'MCI', 'ADD', 'nADD']) / 4.0
                    else:
                        val = f1_(gt, pd)
                    metric_list.append(val)
                elif metric == 'Sensitivity':
                    if task == 'macro-3ways':
                        val = sum(cache[metric][key][csv_file] for key in ['NC', 'MCI', 'DE']) / 3.0
                    elif task == 'macro-4ways':
                        val = sum(cache[metric][key][csv_file] for key in ['NC', 'MCI', 'ADD', 'nADD']) / 4.0
                    else:
                        val = sens_(gt, pd)
                    metric_list.append(val)
                elif metric == 'Specificity':
                    if task == 'macro-3ways':
                        val = sum(cache[metric][key][csv_file] for key in ['NC', 'MCI', 'DE']) / 3.0
                    elif task == 'macro-4ways':
                        val = sum(cache[metric][key][csv_file] for key in ['NC', 'MCI', 'ADD', 'nADD']) / 4.0
                    else:
                        val = spec_(gt, pd)
                    metric_list.append(val)
                elif metric == 'MCC':
                    if task == 'macro-3ways':
                        val = sum(cache[metric][key][csv_file] for key in ['NC', 'MCI', 'DE']) / 3.0
                    elif task == 'macro-4ways':
                        val = sum(cache[metric][key][csv_file] for key in ['NC', 'MCI', 'ADD', 'nADD']) / 4.0
                    else:
                        val = mcc_(gt, pd)
                    metric_list.append(val)
                cache[metric][task][csv_file] = metric_list[-1]
            metric_list = np.array(metric_list)
            mean, std = np.mean(metric_list), np.std(metric_list)
            left, right = confidence_intervals(mean, std)
            row.append("{0:.3f}±{1:.3f} [{2:.3f}-{3:.3f}]".format(mean, std, left, right))
            if metric == 'MCC': ans += np.mean(metric_list)
        content.append(row)
    headers = [' ', 'NC', 'MCI', 'DE', 'ADD', 'nADD', 'ADD|DE', 'nADD|DE', 'macro-3ways', 'macro-4ways']
    print(tabulate(content,
                   headers=headers))
    print('-' * 147)
    with open(output_name + ".csv", "w") as f:
        f.write(tabulate(content, headers=headers, tablefmt="csv"))
    return ans

def confidence_intervals(mean, std, dof=4):
    z = 2.776
    if dof == 16: # for 17 neurologists
        z = 2.120
    elif dof == 6: # for 7 radiologists
        z = 2.447
    return mean - z * std / ((dof + 1) ** 0.5), mean + z * std / ((dof + 1) ** 0.5)

def get_pd_gt(csv_file, task):
    pred, label = [], []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row['COG']: continue
            if task == 'NC':
                pred.append(1 if row['COG_pred'] == '0' else 0)
                label.append(1 if row['COG'][0] == '0' else 0)
            elif task == 'MCI':
                pred.append(1 if row['COG_pred'] == '1' else 0)
                label.append(1 if row['COG'][0] == '1' else 0)
            elif task == 'DE':
                pred.append(1 if row['COG_pred'] == '2' else 0)
                label.append(1 if row['COG'][0] == '2' else 0)
            elif task == 'ADD':
                pred.append(1 if (row['COG_pred']=='2' and row['ADD_pred']=='1') else 0)
                label.append(1 if (row['COG'][0]=='2' and row['ADD'][0]=='1') else 0)
            elif task == 'nADD':
                pred.append(1 if (row['COG_pred']=='2' and row['ADD_pred']=='0') else 0)
                label.append(1 if (row['COG'][0]=='2' and row['ADD'][0]=='0') else 0)
            elif task == 'ADD|DE':
                if not row['ADD']: continue
                pred.append(1 if row['ADD_pred']=='1' else 0)
                label.append(1 if row['ADD'][0]=='1' else 0)
            elif task == 'nADD|DE':
                if not row['ADD']: continue
                pred.append(1 if row['ADD_pred']=='0' else 0)
                label.append(1 if row['ADD'][0]=='0' else 0)
    return pred, label

def accu_(gt, pd):
    cm = confusion_matrix(gt, pd, labels=[0, 1])
    tp, tn, fp, fn = cm[1][1], cm[0][0], cm[0][1], cm[1][0]
    return float(tp + tn) / (tp + fp + fn + tn)

def f1_(gt, pd):
    cm = confusion_matrix(gt, pd, labels=[0, 1])
    tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
    return float(tp) / (tp + 0.5 * (fp + fn) + 0.000001)

def sens_(gt, pd):
    cm = confusion_matrix(gt, pd, labels=[0, 1])
    tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
    return float(tp) / (tp + fn + 0.000001)

def spec_(gt, pd):
    cm = confusion_matrix(gt, pd, labels=[0, 1])
    tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
    return float(tn) / (tn + fp + 0.000001)

def mcc_(gt, pd):
    cm = confusion_matrix(gt, pd, labels=[0, 1])
    TP, FP, FN, TN = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
    upper = TP * TN - FP * FN
    lower = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    return float(upper) / (lower ** 0.5 + 0.000000001)


def whole_eval_package(model_name, stage, performTableFileName):
    print("evaluating the model performance of "+stage)
    common_path = 'tb_log/' + model_name
    name = '/{}_eval.csv'.format(stage)
    csv_files = [common_path + '_cross{}'.format(i) + name for i in range(5)]
    crossValid_cm(csv_files, stage)
    perform_table(csv_files, performTableFileName)
    ROC_PR_curves(csv_files, stage)



if __name__ == "__main__":
    pass











