import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy import random
import json
import csv
import random
import os
import time
import itertools
from scipy.special import softmax
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle
from collections import OrderedDict, defaultdict
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, accuracy_score
import pandas as pd
import collections
from scipy.ndimage import zoom
from tabulate import tabulate
import shap
from glob import glob

#--------------------------------------------------------------------------------------------
# shap CNN heatmap utility functions
def shap_abs_mean(tb_log_dir, shap_dir, task):
    mean, count = np.zeros((43, 52, 43)), 0
    for i in range(5):
        with open(tb_log_dir + 'cross{}/'.format(i) + 'test_eval.csv', 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row[task]:
                    mean += np.abs(np.load(shap_dir + 'shap_{}_'.format(task) + row['filename']))
                    count += 1
    mean = mean / count
    print('averaged {} cases for the {} task'.format(count, task))
    np.save(shap_dir + '{}_abs.npy'.format(task), mean)

def average_ADD_shapmap(tb_log_dir, shap_dir):
    ADD, nADD, count_ADD, count_nADD = np.zeros((43, 52, 43)), np.zeros((43, 52, 43)), 0, 0
    for i in range(5):
        with open(tb_log_dir + 'cross{}/'.format(i) + 'test_eval.csv', 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['ADD'] in ['1', '1.0']:
                    ADD += np.load(shap_dir + 'shap_ADD_' + row['filename'])
                    count_ADD += 1
                elif row['ADD'] in ['0', '0.0']:
                    nADD += np.load(shap_dir + 'shap_ADD_' + row['filename'])
                    count_nADD += 1
    ADD = ADD / count_ADD
    nADD = nADD / count_nADD
    std = np.std(ADD)
    ADD, nADD = ADD / std, nADD / std
    print('averaged {} ADD cases and {} nADD cases'.format(count_ADD, count_nADD))
    np.save(shap_dir + 'ADD.npy', ADD)
    np.save(shap_dir + 'nADD.npy', nADD)
    return shap_dir + 'ADD.npy', shap_dir + 'nADD.npy'

def average_ADD_shapmap_truepred(tb_log_dir, shap_dir):
    ADD, nADD, count_ADD, count_nADD = np.zeros((43, 52, 43)), np.zeros((43, 52, 43)), 0, 0
    for i in range(5):
        with open(tb_log_dir + 'cross{}/'.format(i) + 'test_eval.csv', 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['ADD'] in ['1', '1.0'] and row['ADD_pred'] in ['1', '1.0']:
                    ADD += np.load(shap_dir + 'shap_ADD_' + row['filename'])
                    count_ADD += 1
                elif row['ADD'] in ['0', '0.0'] and row['ADD_pred'] in ['0', '0.0']:
                    nADD += np.load(shap_dir + 'shap_ADD_' + row['filename'])
                    count_nADD += 1
    ADD = ADD / count_ADD
    nADD = nADD / count_nADD
    # std = np.std(ADD)
    # ADD, nADD = ADD / std, nADD / std
    print('averaged {} ADD cases and {} nADD cases'.format(count_ADD, count_nADD))
    np.save(shap_dir + 'ADD.npy', ADD)
    np.save(shap_dir + 'nADD.npy', nADD)
    return shap_dir + 'ADD.npy', shap_dir + 'nADD.npy'

def average_COG_shapmap_truepred(tb_log_dir, shap_dir):
    NC, MCI, DE, count_NC, count_MCI, count_DE = np.zeros((43, 52, 43)), np.zeros((43, 52, 43)), np.zeros((43, 52, 43)), 0, 0, 0
    for i in range(5):
        with open(tb_log_dir + 'cross{}/'.format(i) + 'test_eval.csv', 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['COG'] in ['0', '0.0'] and row['COG_pred'] in ['0', '0.0']:
                    NC += np.load(shap_dir + 'shap_COG_' + row['filename'])
                    count_NC += 1
                elif row['COG'] in ['1', '1.0'] and row['COG_pred'] in ['1', '1.0']:
                    MCI += np.load(shap_dir + 'shap_COG_' + row['filename'])
                    count_MCI += 1
                elif row['COG'] in ['2', '2.0'] and row['COG_pred'] in ['2', '2.0']:
                    DE += np.load(shap_dir + 'shap_COG_' + row['filename'])
                    count_DE += 1
    NC = NC / count_NC
    MCI = MCI / count_MCI
    DE = DE / count_DE
    # std = np.std(NC)
    # NC, MCI, DE = NC / std, MCI / std, DE / std
    print('averaged {} NC cases and {} MCI cases and {} DE cases'.format(count_NC, count_MCI, count_DE))
    np.save(shap_dir + 'NC.npy', NC)
    np.save(shap_dir + 'MCI.npy', MCI)
    np.save(shap_dir + 'DE.npy', DE)


# shap meshgrid plot
def plot_shap_heatmap(models, tasks, stage):
    from matplotlib import rc, rcParams
    rc('axes', linewidth=2)
    rc('font', weight='bold')
    rcParams.update({'font.size': 14})
    heatmaps = [[] for _ in tasks]
    for model in models:
        common_path = 'tb_log/' + model + '_CrossValid_nonImg'
        for i, task in enumerate(tasks):
            name = 'shap_{}_shap_{}.csv'.format(stage, task)
            csv_files = [common_path + '_cross{}/'.format(j) + name for j in range(5)]
            mean, std, columns = shap_stat(csv_files)
            heatmaps[i].append(mean)
    for i, task in enumerate(tasks):
        heatmaps[i] = np.array(heatmaps[i])
        print(heatmaps[i].shape, columns)
        hm, feature_names = get_common_top_N(heatmaps[i], columns)
        for i, f in enumerate(feature_names):
            feature_names[i] = f.lower()
            if f == 'ADD_score': feature_names[i] = 'mri_add'
            if f == 'COG_score': feature_names[i] = 'mri_cog'
        for j in range(hm.shape[0]):
            hm[j, :] = hm[j, :] / np.max(hm[j, :])
        fig, ax = plt.subplots(figsize=(12, 6))
        im, cbar = heatmap(hm, models, feature_names, ax=ax, vmin=0, vmax=1,
                           cmap="cool", cbarlabel="Relative Importance")
        plt.savefig('shap_heatmap_{}.png'.format(task), dpi=200, bbox_inches='tight')
        plt.close()


def CNN_shap_regions_heatmap(corre_file, name):
    from matplotlib import rc, rcParams
    rc('axes', linewidth=2)
    rc('font', weight='bold')
    rcParams.update({'font.size': 14})
    with open(corre_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        column_names, hm = [], []
        for row in reader:
            for key in row:
                column_names = key.split()
                data = row[key].split()[1:]
                data = np.array(list(map(float, data)))
                hm.append(data)
        hm = np.array(hm)
        print(hm.shape)
        fig, ax = plt.subplots(figsize=(12, 12))
        im, cbar = heatmap(hm, column_names, column_names, ax=ax, vmin=-1, vmax=1,
                           cmap="hot", cbarlabel="Pearson Correlation")
        plt.savefig('regional_heatmap_{}.png'.format(name), dpi=200, bbox_inches='tight')
        plt.close()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    code from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=12, fontweight='black')

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def get_common_top_N(heatmap, columns):
    mean = np.mean(heatmap, axis=0)
    indexes = mean.argsort()
    new_columns = []
    for i in indexes[-15:][::-1]:
        new_columns.append(columns[i])
    return heatmap[:, indexes[-15:][::-1]], new_columns

# shap bar plot
def plot_shap_bar(path, model_name, stage, tasks, top):
    from matplotlib import rc, rcParams
    rc('axes', linewidth=2)
    rc('font', weight='bold', size=15)
    common_path = 'tb_log/' + model_name
    for task in tasks:
        name = 'shap_{}_{}.csv'.format(stage, task)
        csv_files = [common_path + '_cross{}/'.format(i) + name for i in range(5)]
        mean, std, columns = shap_stat(csv_files)
        for i, f in enumerate(columns):
            columns[i] = f.lower()
            if f == 'ADD_score': columns[i] = 'mri_add'
            if f == 'COG_score': columns[i] = 'mri_cog'
        pool = get_top_N(mean, std, columns, top)
        fig, ax = plt.subplots(figsize=(4, 10))
        plt.barh([a[2] for a in pool], [a[0] for a in pool], color='r', xerr=[a[1] for a in pool], capsize=5)
        ax.set_xlabel('Mean(|SHAP|)', fontsize=16, fontweight='black')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.spines['bottom'].set_position(('axes', 0.01))
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(path + 'shap_bar_{}.png'.format(task), dpi=200, bbox_inches='tight')
        plt.close()

def plot_shap_beeswarm(path, SHAP, DATA, tasks, stage):
    from matplotlib import rc, rcParams
    rc('axes', linewidth=2)
    rc('font', weight='bold', size=15)
    for i, task in enumerate(tasks):
        fig, ax = plt.subplots()
        shap_values = np.concatenate([s[i] for s in SHAP], axis=0)
        feature_values = pd.concat([d[i] for d in DATA])
        feature_values.rename(columns={'ADD_score': 'mri_add'}, inplace=True)
        feature_values.rename(columns={'COG_score': 'mri_cog'}, inplace=True)
        feature_values.columns = map(str.lower, feature_values.columns)
        print('shap_values\n', shap_values)
        print('feature_values\n', feature_values)
        shap.summary_plot(shap_values, feature_values, max_display=15)
        fig = plt.gcf()
        fig.set_size_inches(4, 10)
        cbar_ax = fig.axes[-1]
        cbar_ax.tick_params(labelsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('SHAP value', fontsize=16, fontweight='black')
        plt.savefig(path + '{}_shap_beeswarm_{}.png'.format(stage, task), dpi=200, bbox_inches='tight')
        plt.close()

def parse_csv(csv_files):
    res = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        res.append(df)
    return pd.concat(res)

def shap_stat(csv_files):
    shap_mag = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        columns = df.columns
        shap = df.to_numpy()
        shap_mean = np.mean(np.absolute(shap), axis=0)
        shap_mag.append(shap_mean)
    shap_mag = np.array(shap_mag)
    mean, std = np.mean(shap_mag, axis=0), np.std(shap_mag, axis=0)
    return mean, std, columns.to_list()

def get_top_N(mean, std, columns, N):
    pool = list(zip(mean, std, columns))
    pool.sort()
    return pool[-N:]

#--------------------------------------------------------------------------------------------
def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module' in k: name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

#-------------------------------------------------------------------------------------------------------
# util functions for model_wrappers.py to generate ROC, PR curves and confusion matrices during training
#-------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names, title=None):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    cmap = plt.cm.Blues
    xticks_rotation = 'horizontal'
    yticks_rotation = 'vertical'
    fig, ax = plt.subplots(dpi=100)
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    text_ = np.empty_like(cm, dtype=object)
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text_cm = format(cm[i, j], '.1f')
        if cm.dtype.kind != 'f':
            text_d = format(cm[i, j], 'd')
            if len(text_d) < len(text_cm):
                text_cm = text_d
        text_[i, j] = ax.text(
                        j, i, text_cm, fontsize=20, fontweight='bold',
                        ha="center", va="center",
                        color=color)
    display_labels = class_names
    cbar = fig.colorbar(im_, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation, fontsize=15, fontweight='bold')
    plt.setp(ax.get_yticklabels(), rotation=yticks_rotation, fontsize=15, fontweight='bold')
    ax.set_xlabel('Pred', fontsize=15, fontweight='bold')
    ax.set_ylabel('True', fontsize=15, fontweight='bold')
    ax.set_title(title,   fontsize=15, fontweight='bold')
    return fig

def plot_all_CM(epoch, mode, task, pd, gt, thres, path, class_names):
    """
    :param epoch: during training, which epoch the figure is about
    :param mode:  mode can be like "train", "valid", to indicate whether the figure is for training set or validation
    :param task:  task name
    :param pd:    prediction list
    :param gt:    ground truth list
    :param na:    name of datasets list
    :param thres: threshold for the raw score to discretize continuous score
    :param path:  path to save figure
    :param class_names: class name for labelling
    """
    labels = [i for i in range(len(class_names))]
    if task in thres:
        pd_prob = softmax(pd, axis=1)[:, 1]
        pd = np.where(pd_prob > thres[task], 1, 0)
    else:
        pd = np.argmax(pd, axis=1)
    cm = confusion_matrix(gt, pd, labels=labels)
    title = mode + '_' + task + '_ALL'
    fig = plot_confusion_matrix(cm, class_names, title)
    fig.savefig(path + mode + '_CM_' + task + '_all_{}.jpg'.format(epoch))
    fig.clf()
    plt.close()

def plot_all_CM_reg(epoch, mode, task, pd, gt, thres, path, class_names):
    """
    same parameter setting as the function above
    the reg surfix indicates this function will be used to generate confusion matrix for regression task
    when we set "COG" task as regression task, model is trained to regress: NC=0, MCI=1, DE=2
    thus NC_thres is needed to separate NC from MCI+DE
         DE_thres is needed to seperate DE from NC+MCI
    """
    labels = [0, 1] if task != 'COG' else [0, 1, 2]
    NC_thres = thres['NC'] if 'NC' in thres else 0.5
    DE_thres = thres['DE'] if 'DE' in thres else 1.5
    pd_ = []
    for p in pd:
        if p < NC_thres:
            pd_.append(0)
        elif NC_thres <= p < DE_thres:
            pd_.append(1)
        else:
            pd_.append(2)
    cm = confusion_matrix(gt, pd_, labels=labels)
    title = mode + '_' + task + '_ALL'
    fig = plot_confusion_matrix(cm, class_names, title)
    fig.savefig(path + mode + '_CM_' + task + '_all_{}.jpg'.format(epoch))
    fig.clf()
    plt.close()

def plot_roc_curve(epoch, mode, task, gt, pd, path, class_names):
    fig, ax = plt.subplots(dpi=100)
    pd = softmax(pd, axis=1)
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen']
    if len(class_names)==2: # binary classification
        y_score = pd[:, 1]
        fpr, tpr, thres = roc_curve(gt, y_score, pos_label=1)
        AUCs = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[1], lw=lw, label='{0} AUC = {1:0.2f}'.format(class_names[1], AUCs))
    else: # multi-way classification
        AUCs = 0
        for i in range(len(class_names)):
            y_score = pd[:, i]
            y_true = [1 if g==i else 0 for g in gt]
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
            AUC = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=lw, label='{0} AUC = {1:0.2f}'.format(class_names[i], AUC))
            AUCs += AUC
        AUCs /= len(class_names)
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlabel('False Positive Rate', fontsize=15, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=15, fontweight='bold')
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))
    ax.set_title(mode + '_' + task + '_ROC', fontsize=15, fontweight='bold')
    legend_properties = {'weight': 'bold', 'size': 15}
    ax.legend(loc="lower right", prop=legend_properties)
    fig.savefig(path + mode + '_ROC_' + task + '_{}.jpg'.format(epoch))
    fig.clf()
    plt.close()
    return AUCs

def plot_roc_curve_reg(epoch, mode, task, gt, pd, path, class_names):
    fig, ax = plt.subplots(dpi=100)
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen']
    if len(class_names) == 3:
        AUCs = 0
        for i in [0, 2]:
            y_score = -pd if i == 0 else pd
            y_true = [1 if g==i else 0 for g in gt]
            fpr, tpr, thres = roc_curve(y_true, y_score, pos_label=1)
            AUC = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=lw, label='{0} AUC = {1:0.2f}'.format(class_names[i], AUC))
            AUCs += AUC
        AUCs /= 2
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlabel('False Positive Rate', fontsize=15, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=15, fontweight='bold')
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))
    ax.set_title(mode + '_' + task + '_ROC', fontsize=15, fontweight='bold')
    legend_properties = {'weight': 'bold', 'size': 15}
    ax.legend(loc="lower right", prop=legend_properties)
    fig.savefig(path + mode + '_ROC_' + task + '_{}.jpg'.format(epoch))
    fig.clf()
    plt.close()
    return AUCs

def plot_pr_curve(epoch, mode, task, gt, pd, path, class_names):
    fig, ax = plt.subplots(dpi=100)
    pd = softmax(pd, axis=1)
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen']
    if len(class_names)==2: # binary classification
        y_score = pd[:, 1]
        pr, rc, thres = precision_recall_curve(gt, y_score, pos_label=1)
        APs = average_precision_score(gt, y_score, pos_label=1)
        ax.plot(rc, pr, color=colors[1], lw=lw, label='{0} AP = {1:0.2f}'.format(class_names[1], APs))
        count = collections.Counter(gt)
        ratio = count[1] / (count[1] + count[0])
        ax.plot([0, 1], [ratio, ratio], 'k--', lw=lw)
    else:
        APs = 0
        for i in range(len(class_names)):
            y_score = pd[:, i]
            y_true = [1 if g==i else 0 for g in gt]
            pr, rc, _ = precision_recall_curve(y_true, y_score, pos_label=1)
            AP = average_precision_score(y_true, y_score, pos_label=1)
            ax.plot(rc, pr, color=colors[i], lw=lw, label='{0} AP = {1:0.2f}'.format(class_names[i], AP))
            APs += AP
            count = collections.Counter(y_true)
            ratio = count[1] / (count[1] + count[0])
            ax.plot([0, 1], [ratio, ratio], 'k--', lw=lw)
        APs /= len(class_names)
    ax.set_xlabel('Recall', fontsize=15, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=15, fontweight='bold')
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))
    ax.set_title(mode + '_' + task + '_PR', fontsize=15, fontweight='bold')
    legend_properties = {'weight': 'bold', 'size': 15}
    ax.legend(loc="lower left", prop=legend_properties)
    fig.savefig(path + mode + '_PR_' + task + '_{}.jpg'.format(epoch))
    fig.clf()
    plt.close()
    return APs

def plot_pr_curve_reg(epoch, mode, task, gt, pd, path, class_names):
    gt = np.squeeze(gt)
    fig, ax = plt.subplots(dpi=100)
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen']
    if len(class_names) == 3:
        AUCs = 0
        for i in [0, 2]:
            y_score = -pd if i == 0 else pd
            y_true = [1 if g==i else 0 for g in gt]
            pr, rc, thres = precision_recall_curve(y_true, y_score, pos_label=1)
            ap = average_precision_score(y_true, y_score, pos_label=1)
            ax.plot(rc, pr, color=colors[i], lw=lw, label='{0} AP = {1:0.2f}'.format(class_names[i], ap))
            AUCs += ap
        AUCs /= 2
    count = collections.Counter(gt)
    ratio1, ratio2 = count[0] / len(gt), count[2] / len(gt)
    ax.plot([0, 1], [ratio1, ratio1], 'k--', lw=lw)
    ax.plot([0, 1], [ratio2, ratio2], 'k--', lw=lw)
    ax.set_xlabel('Recall', fontsize=15, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=15, fontweight='bold')
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))
    ax.set_title(mode + '_' + task + '_PR', fontsize=15, fontweight='bold')
    legend_properties = {'weight': 'bold', 'size': 15}
    ax.legend(loc="lower left", prop=legend_properties)
    fig.savefig(path + mode + '_PR_' + task + '_{}.jpg'.format(epoch))
    fig.clf()
    plt.close()
    return AUCs

#--------------------------------------------------------
# util functions for model_wrappers.py to do thresholding
#--------------------------------------------------------
def f1_threshold_reg(pd, gt):
    """
    this function select best thresholds for NC MCI DE regression problem using f-1 as metric
    """
    pd = np.concatenate(pd, axis=0)
    gt = np.concatenate(gt, axis=0)
    pool = []
    for nc in range(-100, 199):
        nc_thres = float(nc) / 100.0
        for de in range(nc, 200):
            de_thres = float(de) / 100.0
            pd_ = []
            for p in pd:
                if p < nc_thres:
                    pd_.append(0)
                elif nc_thres <= p < de_thres:
                    pd_.append(1)
                else:
                    pd_.append(2)
            if len(set(pd_)) == 3:
                f1 = f1_score(gt, pd_, average='macro')
                pool.append((f1, nc_thres, de_thres))
    pool.sort()
    return pool[-1][1], pool[-1][2]

def accu_threshold_reg(pd, gt):
    """
    this function select best thresholds for NC MCI DE regression problem using accuracy as metric
    """
    pd = np.concatenate(pd, axis=0)
    gt = np.concatenate(gt, axis=0)
    pool = []
    for nc in range(-100, 199):
        nc_thres = float(nc) / 100.0
        for de in range(nc, 200):
            de_thres = float(de) / 100.0
            pd_ = []
            for p in pd:
                if p < nc_thres:
                    pd_.append(0)
                elif nc_thres <= p < de_thres:
                    pd_.append(1)
                else:
                    pd_.append(2)
            if len(set(pd_)) == 3:
                cm = confusion_matrix(gt, pd_)
                accu = cm[0][0] + cm[1][1] + cm[2][2]
                pool.append((accu, nc_thres, de_thres))
    pool.sort()
    return pool[-1][1], pool[-1][2]

def get_threshold(pd, gt):
    """
    this function select best thresholds for binary classification task using MCC as metric
    """
    pd = np.concatenate(pd, axis=0)
    gt = np.concatenate(gt, axis=0)
    pd = softmax(pd, axis=1)
    score = pd[:, 1]
    import warnings
    warnings.filterwarnings('ignore')
    mcc, threshold = [], []
    for thres in range(1, 1000 * thresRange):
        thres = float(thres) / 1000
        threshold.append(thres)
        pred = score_threshold(score, thres)
        mcc.append(matthews_corrcoef(gt, pred))
    mcc = np.array(mcc)
    return threshold[np.argmax(mcc)]

def score_threshold(score, thres):
    pred = np.zeros(score.shape)
    for i in range(score.size):
        pred[i] = 0 if score[i] <= thres else 1
    return pred

#--------------------new version
def get_COG_score_label(csv_file):
    label, score = [], []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['COG'] and row['COG_score']:
                label.append(int(row['COG'][0]))
                score.append(float(row['COG_score']))
    return label, score

def get_ADD_score_label(csv_file):
    label, score = [], []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['ADD'] and row['ADD_score']:
                label.append(int(row['ADD'][0]))
                score.append(float(row['ADD_score']))
    return label, score

def COG_thresholding(csv_file):
    label, score = get_COG_score_label(csv_file)
    NC_count = label.count(0)
    MCI_count = label.count(1)
    score = sorted(score)
    return score[NC_count], score[NC_count+MCI_count]

def ADD_thresholding(csv_file):
    label, score = get_ADD_score_label(csv_file)
    nADD_count = label.count(0)
    score = sorted(score)
    return score[nADD_count]


if __name__ == "__main__":
    models = ['XGBoost', 'CatBoost', 'RandomForest', 'DecisionTree', 'Perceptron']
    tasks = ['COG', 'ADD']
    stage = 'test'
    plot_shap_heatmap(models, tasks, stage)

    # average_COG_shapmap_truepred('tb_log/CNN_baseline_new_', '/data_2/sq/shap_mid/')
    # ADD_shap, nADD_shap = average_ADD_shapmap_truepred('tb_log/CNN_baseline_new_', '/data_2/sq/shap_mid/')

    # CNN_shap_regions_heatmap('shap/ADD_correlation_top10.csv', 'ADD')
    # CNN_shap_regions_heatmap('shap/nADD_correlation_top10.csv', 'nADD')

    # shap_abs_mean('tb_log/CNN_baseline_new_', '/data_2/sq/shap_mid/', 'COG')

