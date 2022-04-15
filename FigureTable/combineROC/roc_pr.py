import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
print(parentdir)
sys.path.append(parentdir)
from performance_eval import *
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from brokenaxes import brokenaxes

def generate_roc(group_csv_files, positive_label, colors, out_file):
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
    groups = ['MRI', 'NonImg', 'Fusion']
    mean_fpr = np.linspace(0, 1, 100)
    for i, csvfiles in enumerate(group_csv_files):
        groupName = groups[i]
        color = colors[i]
        tprs = []
        aucs = []
        for csvfile in csvfiles:
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
                label='{} AUC={:.3f}$\pm${:.3f}'.format(groupName, mean_auc, std_auc),
                lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    legend_properties = {'weight': 'bold', 'size': 16}
    ax.set_xlabel('False Positive Rate', fontsize=text_size, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=text_size, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc="lower right", prop=legend_properties)
    fig.savefig(out_file, bbox_inches='tight')
    fig.clf()
    plt.close()
    return

def generate_pr(group_csv_files, positive_label, colors, out_file, gap=0.5, yticks=None):
    """
    :param csv_files:       a list of csv files as above format
    :param positive_label:  if positive_label == 'NC', the curve is about NC vs not NC
                            if positive_label == 'DE', the curve is about DE vs not DE
                            if positive_label =='ADD', the curve is about ADD vs nADD
    :param color:           color of the roc curve
    :param out_file:        image filename you want to save as
    :return:
    """
    extra = 0.05 / 1.0 * (1 - gap)
    lw = 2
    text_size = 18
    fig, ax = plt.subplots(dpi=200)
    groups = ['MRI', 'NonImg', 'Fusion']
    mean_rc = np.linspace(0, 1, 100)
    for i, csvfiles in enumerate(group_csv_files):
        groupName = groups[i]
        color = colors[i]
        prs = []
        aucs = []
        labels = []
        for csvfile in csvfiles:
            scores, labels = get_score_label(csvfile, positive_label)
            pr, rc, thres = precision_recall_curve(labels, scores, pos_label=1)
            pr, rc = pr[::-1], rc[::-1]
            AUC = average_precision_score(labels, scores, pos_label=1)
            ax.plot(rc, pr, lw=lw / 2, alpha=0.15)
            interp_pr = np.interp(mean_rc, rc, pr)
            prs.append(interp_pr)
            aucs.append(AUC)
        mean_pr = np.mean(prs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax.plot(mean_rc, mean_pr, color=color,
                label='{} AP={:.3f}$\pm${:.3f}'.format(groupName, mean_auc, std_auc),
                lw=2, alpha=.8)
        count = collections.Counter(labels)
        ratio = count[1] / (count[1] + count[0])
        ax.plot([0, 1], [ratio, ratio], 'k--', lw=lw)
        std_pr = np.std(prs, axis=0)
        prs_upper = np.minimum(mean_pr + std_pr, 1)
        prs_lower = np.maximum(mean_pr - std_pr, 0)
        ax.fill_between(mean_rc, prs_lower, prs_upper, color=color, alpha=.2)
    legend_properties = {'weight': 'bold', 'size': 16}
    ax.set_yticks(yticks)
    ax.set(xlim=[-0.05, 1.05], ylim=[gap-extra, 1.0+extra])
    ax.set_xlabel('Recall', fontsize=text_size, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=text_size, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc="lower left", prop=legend_properties)
    fig.savefig(out_file, bbox_inches='tight')
    fig.clf()
    plt.close()
    return

def generate_pr_gap(group_csv_files, positive_label, colors, out_file, yticks, gap=0.5):
    """
    :param csv_files:       a list of csv files as above format
    :param positive_label:  if positive_label == 'NC', the curve is about NC vs not NC
                            if positive_label == 'DE', the curve is about DE vs not DE
                            if positive_label =='ADD', the curve is about ADD vs nADD
    :param color:           color of the roc curve
    :param out_file:        image filename you want to save as
    :return:
    """
    extra = 0.05 / 1.0 * (1 - gap)
    lw = 2
    text_size = 20
    r, c = 6, 5.5
    factor = 0.88
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(r * factor, c * factor), sharex=True, dpi=200, gridspec_kw={'height_ratios': [100, 1]})
    groups = ['MRI', 'NonImg', 'Fusion']
    mean_rc = np.linspace(0, 1, 100)
    for i, csvfiles in enumerate(group_csv_files):
        groupName = groups[i]
        color = colors[i]
        prs = []
        aucs = []
        labels = []
        for csvfile in csvfiles:
            scores, labels = get_score_label(csvfile, positive_label)
            pr, rc, thres = precision_recall_curve(labels, scores, pos_label=1)
            pr, rc = pr[::-1], rc[::-1]
            AUC = average_precision_score(labels, scores, pos_label=1)
            ax1.plot(rc, pr, lw=lw / 2, alpha=0.15)
            interp_pr = np.interp(mean_rc, rc, pr)
            prs.append(interp_pr)
            aucs.append(AUC)
        mean_pr = np.mean(prs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax1.plot(mean_rc, mean_pr, color=color,
                label='{} AP={:.3f}$\pm${:.3f}'.format(groupName, mean_auc, std_auc),
                lw=2, alpha=.8)
        count = collections.Counter(labels)
        ratio = count[1] / (count[1] + count[0])
        ax1.plot([0, 1], [ratio, ratio], 'k--', lw=lw)
        std_pr = np.std(prs, axis=0)
        prs_upper = np.minimum(mean_pr + std_pr, 1)
        prs_lower = np.maximum(mean_pr - std_pr, 0)
        ax1.fill_between(mean_rc, prs_lower, prs_upper, color=color, alpha=.2)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    f = d * 100
    ax2.plot((-d, +d), (1-f, 1+f), **kwargs)  # bottom-left diagonal
    ax2.plot((1-d, 1+d), (1-f, 1+f), **kwargs)  # bottom-right diagonal
    legend_properties = {'weight': 'bold', 'size': 16}
    ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 0.01])
    ax1.set(xlim=[-0.05, 1.05], ylim=[gap, 1 + extra])
    ax2.set_xlabel('Recall', fontsize=text_size, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=text_size, fontweight='bold')
    ax2.tick_params(axis='x', which='major', labelsize=16)
    ax2.set_yticks([])
    ax1.set_yticks(yticks)
    ax1.tick_params(axis='y', which='major', labelsize=16)
    ax1.legend(loc="lower left", prop=legend_properties)
    ax1.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        top=False)  # ticks along the top edge are off
    fig.tight_layout(pad=0.0)
    fig.savefig(out_file, bbox_inches='tight')
    fig.clf()
    plt.close()
    return

if __name__ == "__main__":
    # NACC
    mri_csv_files = ['../../tb_log/CNN_baseline_new_cross{}/test_eval.csv'.format(i) for i in range(5)]
    nonImg_csv_files = ['../../tb_log/_CatBoost_NonImg_cross{}/test_eval.csv'.format(i) for i in range(5)]
    fusion_csv_files = ['../../tb_log/_CatBoost_Fusion_cross{}/test_mri_eval.csv'.format(i) for i in range(5)]
    csv_files = [mri_csv_files, nonImg_csv_files, fusion_csv_files]
    colors1 = ['#222222', 'b', 'r']
    colors2 = colors1
    colors3 = colors1
    generate_roc(csv_files, 'NC', colors1, 'roc_test_nc.png')
    generate_pr(csv_files, 'NC', colors1, 'pr_test_nc.png', gap=0.5, yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    generate_roc(csv_files, 'DE', colors2, 'roc_test_de.png')
    generate_pr(csv_files, 'DE', colors2, 'pr_test_de.png', gap=0.2, yticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    generate_roc(csv_files, 'ADD', colors3, 'roc_test_add.png')
    generate_pr(csv_files, 'ADD', colors3, 'pr_test_add.png', gap=0.6, yticks=[0.6, 0.7, 0.8, 0.9, 1.0])

    # OASIS
    mri_csv_files = ['../../tb_log/CNN_baseline_new_cross{}/OASIS_eval.csv'.format(i) for i in range(5)]
    nonImg_csv_files = ['../../tb_log/_CatBoost_NonImg_cross{}/OASIS_eval.csv'.format(i) for i in range(5)]
    fusion_csv_files = ['../../tb_log/_CatBoost_Fusion_cross{}/OASIS_mri_eval.csv'.format(i) for i in range(5)]
    csv_files = [mri_csv_files, nonImg_csv_files, fusion_csv_files]
    generate_roc(csv_files, 'NC', colors1, 'roc_oasis_nc.png')
    generate_pr(csv_files, 'NC', colors1, 'pr_oasis_nc.png', gap=0.5, yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    generate_roc(csv_files, 'DE', colors2, 'roc_oasis_de.png')
    generate_pr(csv_files, 'DE', colors2, 'pr_oasis_de.png', gap=0.2, yticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    generate_roc(csv_files, 'ADD', colors3, 'roc_oasis_add.png')
    generate_pr(csv_files, 'ADD', colors3, 'pr_oasis_add.png', gap=0.6, yticks=[0.6, 0.7, 0.8, 0.9, 1.0])



