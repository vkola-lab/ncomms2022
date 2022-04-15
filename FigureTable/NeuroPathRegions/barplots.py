from correlate import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', weight='bold', size=10)

def barplots(prefixes, regions, stains, corre, error, name, folder, ylim):
    for stain in stains:
        barplot(prefixes, regions, stain, corre, error, name, folder, ylim)

def barplot(prefixes, regions, stain, corre, error, name, folder, ylim):
    colors = ['#bfe2e3', '#69869c', '#36896e', '#c22e00', '#c6d645', '#ffd3b6', '#b2b2b2', '#4724a9',
              '#9bc84d', '#7141ae', '#d2a782', '#933b61', '#435299', '#d88770', '#765aa8', '#719795']
    Val, Std = [], []
    for i, prefix in enumerate(prefixes):
        val, std = corre[prefix][stain], error[prefix][stain]
        Val.append(val)
        Std.append(std)
    fig, ax = plt.subplots(dpi=300, figsize=(6, 3))
    index = [i for i in range(len(regions))]
    ax.bar(index, Val, yerr=Std, capsize=2, color=colors[:len(prefixes)])
    ax.set_ylabel('Spearman\'s rank correlation', fontweight='bold')
    ax.set_ylim(ylim[0]-0.05, ylim[1]+0.05)
    ax.set_xticks(index)
    ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.grid(which='major', axis='both', linestyle='--')
    plt.savefig(folder + 'bar_{}_{}.png'.format(stain, name), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    years = 2
    layername = 'block2BN'
    time_threshold, type = 365*years, 'COG'
    folder = type + '_correlation_{}_years/'.format(years)
    if not os.path.exists(folder):
        os.mkdir(folder)
    interval = file_interval_info(type)

    y_lim = [0, 0]
    corre = collections.defaultdict(dict)
    error = collections.defaultdict(dict)
    pool = [[0, prefixes[i], regions[i]] for i in range(len(regions))]
    for i, region in enumerate(prefixes):
        for stain in stains:
            corr, std = get_correlation(region + '_' + stain, prefix_idx[region], time_threshold, interval, folder, type, layername, missing=0)
            corre[region][stain] = corr
            error[region][stain] = 0
            y_lim[1] = max(y_lim[1], corr)
            y_lim[0] = min(y_lim[0], corr)
            pool[i][0] -= corr
    pool.sort()
    prefixes = [p[1] for p in pool]
    regions = [p[2] for p in pool]
    barplots(prefixes, regions, stains, corre, error, '{}days_{}shap_{}'.format(time_threshold, type, layername), folder, y_lim)








