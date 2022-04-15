import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
print(parentdir)
sys.path.append(parentdir)
from performance_eval import *
import csv
from collections import defaultdict

# get diagnoses from radiologists' rating form
team = defaultdict(dict) # key is initials of radiologist, content is rating
with open('ratings.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['rev_initials'] in ['SQ', 'PJ test', 'PJ-trial chart', 'test', 'test (JL)']: continue
        id = row['id'].split()[0]
        team[row['rev_initials'].lower()][id] = row['dem_dx'] if row['dem_dx'] == '1' else '0'

# for key in team:
#     print(key, len(team[key].keys()), team[key])
#     name_map = {'jac': 'n1', 'phl': 'n2', 'am': 'n3', 'abp': 'n4', 'jes': 'n5', 'mjs': 'n6', 'hy': 'n7'}
#     with open('kappa/' + name_map[key] + '.csv', 'w') as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=['Diagnosis Label'])
#         writer.writeheader()
#         for id in range(1, 51):
#             writer.writerow({'Diagnosis Label': team[key][str(id)]})

# get ground truth labels from 50cases_dummy.csv
label = {} # key is dummy id, content is ADD label (0 is non-ADD, 1 is ADD)
with open('../../lookupcsv/derived_tables/NACC_ALL/50cases_dummy.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        label[row['dummy']] = row['ADD']
# print(label)

def get_sensitivity_specificity(diag, label):
    tp, fp, tn, fn = 0, 0, 0, 0
    for key in diag:
        assert key in label, 'id not found'
        if diag[key] == '1' and label[key] == '1':
            tp += 1
        elif diag[key] == '1' and label[key] == '0':
            fp += 1
        elif diag[key] == '0' and label[key] == '0':
            tn += 1
        elif diag[key] == '0' and label[key] == '1':
            fn += 1
    sensi = tp / (tp + fn + 0.00000001)
    specf = tn / (tn + fp + 0.00000001)
    preci = tp / (tp + fp + 0.00000001)
    recal = sensi
    return sensi, specf, preci, recal

def generate_roc(csv_files, positive_label, color, out_file, team):
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
    text_size = 18
    fig, ax = plt.subplots(dpi=200)
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
    legend_properties = {'weight': 'bold', 'size': 15}
    ax.set_xlabel('False Positive Rate', fontsize=text_size, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=text_size, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    add_dots(ax, team, 'roc')
    ax.legend(loc="lower right", prop=legend_properties)
    fig.savefig(out_file, bbox_inches='tight')
    fig.clf()
    plt.close()
    return mean_auc


def generate_pr(csv_files, positive_label, color, out_file, team):
    lw = 2
    text_size = 18
    fig, ax = plt.subplots(dpi=200)
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
    legend_properties = {'weight': 'bold', 'size': 16}
    ax.set_xlabel('Recall', fontsize=text_size, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=text_size, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    add_dots(ax, team, 'pr')
    ax.legend(loc="lower left", prop=legend_properties)
    fig.savefig(out_file, bbox_inches='tight')
    fig.clf()
    plt.close()
    return mean_auc

def add_dots(ax, team, mode):
    y_list, x_list = [], []
    for ini in team:
        sen, spe, pre, rec = get_sensitivity_specificity(team[ini], label)
        print(ini + ' has sensitivity ', sen, ' has specificity ', spe)
        if mode == 'roc':
            y_list.append(sen)
            x_list.append(1-spe)
        if mode == 'pr':
            y_list.append(pre)
            x_list.append(rec)
    print(x_list, y_list)
    x_mean, x_std = np.array(x_list).mean(), np.array(x_list).std()
    y_mean, y_std = np.array(y_list).mean(), np.array(y_list).std()
    ax.scatter(x_list, y_list, color='b', marker='P', label='Radiologist',
               linewidths=1, edgecolors='k', s=7 ** 2, zorder=10)
    ax.errorbar(x_mean, y_mean, label='avg.Radiologist',
                xerr=x_std, yerr=y_std, fmt='o',
                markeredgewidth=1, markeredgecolor='k',
                markerfacecolor='green',
                markersize=7, marker='P',
                elinewidth=1.5, ecolor='green',
                capsize=3, zorder=11)


# draw model roc curve
csv_files = ['../../tb_log/CNN_3ways_special1_cross{}/neuro_test_eval_after.csv'.format(i) for i in range(5)]
generate_roc(csv_files, 'ADD', 'r', 'radioroc.png', team)
generate_pr(csv_files, 'ADD', 'r', 'radiopr.png', team)



