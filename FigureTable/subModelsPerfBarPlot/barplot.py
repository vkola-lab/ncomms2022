import matplotlib
import matplotlib.pyplot as plt
import csv
import collections
import numpy as np
import pandas as pd

from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', weight='bold', size=10)

def barplot_nonimg(mean, std, y_min, y_max, metric):
    barWidth = 0.25
    r = np.arange(len(mean[0])) * 2.2
    r4 = [x + 3 * barWidth for x in r]
    patterns = ["\\\\\\", "---", "|||", "///", "\/\/", "o", "*"]
    colors = ['#bfe2e3', '#69869c', '#36896e', '#c22e00', '#c6d645', '#ffd3b6', '#b2b2b2']
    labels = ['demo', 'demo+np', 'demo+func', 'demo+his', 'demo+np+func', 'demo+np+his', 'demo+np+his+func']
    fig, ax = plt.subplots(dpi=300, figsize=(9, 4))
    for i in range(7):
        plt.bar(r, mean[i], hatch=patterns[i], yerr=std[i], color=colors[i], width=barWidth, edgecolor='white', label=labels[i])
        r = [x + barWidth for x in r]
    plt.bar([r[-1] + 2], [0])
    # Add xticks on the middle of the group bars
    plt.xticks(r4, ['NACC test', 'OASIS'])
    plt.ylabel('4-way '+metric, fontweight='bold')
    plt.ylim(y_min, y_max)
    ax.legend(loc='right')
    plt.savefig('bar_nonimg_{}.png'.format(metric), bbox_inches='tight')

def barplot_fusion(mean, std, y_min, y_max, metric):
    barWidth = 0.25
    r = np.arange(len(mean[0])) * 2.2
    r4 = [x + 3 * barWidth for x in r]
    patterns = ["\\\\\\", "---", "|||", "///", "\/\/", "o", "*"]
    colors = ['#bfe2e3', '#69869c', '#36896e', '#c22e00', '#c6d645', '#ffd3b6', '#b2b2b2']
    labels = ['MRI+demo', 'MRI+demo+np', 'MRI+demo+func', 'MRI+demo+his', 'MRI+demo+np+func', 'MRI+demo+np+his', 'MRI+demo+np+his+func']
    fig, ax = plt.subplots(dpi=300, figsize=(9, 4))
    for i in range(7):
        plt.bar(r, mean[i], hatch=patterns[i], yerr=std[i], color=colors[i], width=barWidth, edgecolor='white', label=labels[i])
        r = [x + barWidth for x in r]
    plt.bar([r[-1] + 2], [0])
    # Add xticks on the middle of the group bars
    plt.xticks(r4, ['NACC test', 'OASIS'])
    plt.ylabel('4-way '+metric, fontweight='bold')
    plt.ylim(y_min, y_max)
    ax.legend(loc='right')
    plt.savefig('bar_fusion_{}.png'.format(metric), bbox_inches='tight')

def get_data_and_plot_nonimg(metric, y_min, y_max, metrics):
    combinations = ['demo', 'demo+np', 'demo+func', 'demo+his', 'demo+np+func', 'demo+np+his', 'demo+np+his+func']
    MEAN, STD = [], []
    for comb in combinations:
        mean1, std1 = metrics['test'][comb][metric]
        mean2, std2 = metrics['OASIS'][comb][metric]
        MEAN.append([mean1, mean2])
        STD.append([std1, std2])
    barplot_nonimg(MEAN, STD, y_min, y_max, metric)

def get_data_and_plot_fusion(metric, y_min, y_max, metrics):
    combinations = ['MRI+demo', 'MRI+demo+np', 'MRI+demo+func', 'MRI+demo+his', 'MRI+demo+np+func', 'MRI+demo+np+his', 'MRI+demo+np+his+func']
    MEAN, STD = [], []
    for comb in combinations:
        mean1, std1 = metrics['test'][comb][metric]
        mean2, std2 = metrics['OASIS'][comb][metric]
        MEAN.append([mean1, mean2])
        STD.append([std1, std2])
    barplot_fusion(MEAN, STD, y_min, y_max, metric)

def collect_metrics():
    combinations = ['MRI+demo', 'MRI+demo+np', 'MRI+demo+func', 'MRI+demo+his', 'MRI+demo+np+func', 'MRI+demo+np+his', 'MRI+demo+np+his+func']
    combinations += ['demo', 'demo+np', 'demo+func', 'demo+his', 'demo+np+func', 'demo+np+his', 'demo+np+his+func']
    all_metrics = {'test' : {}, 'OASIS': {}}
    for model_name in combinations:
        filedir = '../../tb_log/{}_cross0/performance_test.csv'.format(model_name)
        all_metrics['test'][model_name] = read_metrics_from_csv(filedir)
        filedir = '../../tb_log/{}_cross0/performance_OASIS.csv'.format(model_name)
        all_metrics['OASIS'][model_name] = read_metrics_from_csv(filedir)
    return all_metrics

def read_metrics_from_csv(csvfile):
    ans = {}
    with open(csvfile, 'r') as f:
        for row in f:
            if 'Accuracy' in row:
                m = row.split()[-2]
                mean, std = m.split('±')
                ans['Accuracy'] = [float(mean), float(std)]
            elif 'F-1' in row:
                m = row.split()[-2]
                mean, std = m.split('±')
                ans['F-1'] = [float(mean), float(std)]
            elif 'Sensitivity' in row:
                m = row.split()[-2]
                mean, std = m.split('±')
                ans['Sensitivity'] = [float(mean), float(std)]
            elif 'Specificity' in row:
                m = row.split()[-2]
                mean, std = m.split('±')
                ans['Specificity'] = [float(mean), float(std)]
            elif 'MCC' in row:
                m = row.split()[-2]
                mean, std = m.split('±')
                ans['MCC'] = [float(mean), float(std)]
    return ans

if __name__ == "__main__":
    # accu_nim = [[0.444, 0.486],
    #             [0.720, 0.720],
    #             [0.671, 0.420],
    #             [0.491, 0.440],
    #             [0.752, 0.718],
    #             [0.730, 0.717],
    #             [0.753, 0.725]]
    # accu_err_nim = [[0.020, 0.011],
    #                 [0.020, 0.016],
    #                 [0.017, 0.097],
    #                 [0.016, 0.032],
    #                 [0.019, 0.027],
    #                 [0.020, 0.007],
    #                 [0.022, 0.025]]
    # accu_fus = [[0.599, 0.627],
    #             [0.746, 0.705],
    #             [0.721, 0.562],
    #             [0.626, 0.614],
    #             [0.772, 0.678],
    #             [0.746, 0.698],
    #             [0.773, 0.693]]
    # accu_err_fus = [[0.015, 0.018],
    #                 [0.009, 0.020],
    #                 [0.013, 0.089],
    #                 [0.014, 0.030],
    #                 [0.014, 0.039],
    #                 [0.010, 0.024],
    #                 [0.012, 0.025]]
    #
    # fusion_test = {'Accu': ['0.599+/-0.015', '0.748+/-0.009'],
    #                'F-1':  ['0.472+/-0.014', '0.611+/-0.019'],
    #                'Sens': ['0.469+/-0.012', '0.608+/-0.021'],
    #                'Spec': ['0.847+/-0.003', '0.905+/-0.003'],
    #                'MCC':  ['0.321+/-0.019', '0.518+/-0.021']}
    # fusion_oasis ={'Accu': ['0.627+/-0.018', ''],
    #                'F-1':  ['0.391+/-0.012', ''],
    #                'Sens': ['0.404+/-0.015', ''],
    #                'Spec': ['0.861+/-0.004', ''],
    #                'MCC':  ['0.265+/-0.015', '']}

    # barplot_fusion(accu_fus, accu_err_fus, 0.8, 'accuracy')
    # barplot_nonimg(accu_nim, accu_err_nim, 0.8, 'accuracy')

    # ans = get_metric_from_csv('demo_oasis.csv')
    # print(ans)

    # get_data_and_plot_fusion('Accuracy', 0, 0.8)
    # get_data_and_plot_nonimg('Accuracy', 0, 0.8)
    #
    # get_data_and_plot_fusion('F-1', 0, 0.7)
    # get_data_and_plot_nonimg('F-1', 0, 0.7)
    #
    # get_data_and_plot_fusion('Sensitivity', 0, 0.7)
    # get_data_and_plot_nonimg('Sensitivity', 0, 0.7)
    #
    # get_data_and_plot_fusion('Specificity', 0, 1)
    # get_data_and_plot_nonimg('Specificity', 0, 1)
    #
    # get_data_and_plot_fusion('MCC', 0, 0.7)
    # get_data_and_plot_nonimg('MCC', 0, 0.7)

    all_metrics = collect_metrics()
    print(all_metrics)

    get_data_and_plot_fusion('Accuracy', 0, 0.8, all_metrics)
    get_data_and_plot_nonimg('Accuracy', 0, 0.8, all_metrics)

    get_data_and_plot_fusion('F-1', 0, 0.7, all_metrics)
    get_data_and_plot_nonimg('F-1', 0, 0.7, all_metrics)

    get_data_and_plot_fusion('Sensitivity', 0, 0.7, all_metrics)
    get_data_and_plot_nonimg('Sensitivity', 0, 0.7, all_metrics)

    get_data_and_plot_fusion('Specificity', 0, 1, all_metrics)
    get_data_and_plot_nonimg('Specificity', 0, 1, all_metrics)

    get_data_and_plot_fusion('MCC', 0, 0.7, all_metrics)
    get_data_and_plot_nonimg('MCC', 0, 0.7, all_metrics)







