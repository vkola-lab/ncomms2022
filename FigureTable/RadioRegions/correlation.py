import csv
from collections import defaultdict
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', weight='bold', size=10)

"""
data field             meaning                                          atlas index
-----------------------------------------------------------------------------------
l_atl_amyg           : left anterior temporal lobe, amygdala           : 4
l_atl_m              : left anterior temporal lobe,                    : 6
l_atl_l              : left anterior temporal lobe,                    : 8
l_mtl_hippo          : left medial temporal lobe, hippocampus          : 2
l_mtl_parahippo      : left medial temporal lobe, parahippocampus      : 10
l_pl                 : parietal lobe                                   : 62
l_orbitofrontal      : Left Orbitofrontal                              : 54, 56
l_dorsolateral       : Left Dorsolateral                               : 28
l_superior           : left Superior                                   : 58
l_posterior          : Left Posterior                                  : 50
l_latventricle_temph :                                                 : 48
l_latventricle       : left Lateral Ventricle                          : 46
ventricle3           : 3rd Ventricle                                   : 49
ventricle4           : 4th Ventricle                                   :
genu                 : genu                                            : 76
body                 : body of corpus callosum                         : 44
splenium             : splenium                                        :
"""

anterior_temporal_regions = ['l_atl_amyg', 'r_atl_amyg', 'l_atl_m', 'r_atl_m', 'l_atl_l', 'r_atl_l']
medical_temporal_regions = ['l_mtl_hippo', 'r_mtl_hippo', 'l_mtl_parahippo', 'r_mtl_parahippo']
parietal_lobe_regions = ['l_pl', 'r_pl']
anterior_atrophy_regions = ['l_orbitofrontal', 'r_orbitofrontal', 'l_dorsolateral', 'r_dorsolateral', 'l_superior', 'r_superior']
posterior_atrophy_regions = ['l_posterior', 'r_posterior']
other_regions = ['l_latventricle_temph', 'r_latventricle_temph', 'l_latventricle', 'r_latventricle', 'ventricle3', 'genu', 'body']

region_to_atlas = {'l_atl_amyg': [4],
                   'r_atl_amyg': [3],
                   'l_atl_m': [6],
                   'r_atl_m': [5],
                   'l_atl_l': [8],
                   'r_atl_l': [7],
                   'l_mtl_hippo': [2],
                   'r_mtl_hippo': [1],
                   'l_mtl_parahippo': [10],
                   'r_mtl_parahippo': [9],
                   'l_pl': [62],
                   'r_pl': [63],
                   'l_orbitofrontal': [54, 56],
                   'r_orbitofrontal': [55, 57],
                   'l_dorsolateral': [28],
                   'r_dorsolateral': [29],
                   'l_superior': [58],
                   'r_superior': [59],
                   'l_posterior': [50],
                   'r_posterior': [51],
                   'l_latventricle_temph': [48],
                   'r_latventricle_temph': [47],
                   'l_latventricle': [46],
                   'r_latventricle': [45],
                   'ventricle3': [49],
                   'genu': [76, 77],
                   'body': [44]
                   }

region_to_names = {'l_atl_amyg': "TL amygdala L",
                   'r_atl_amyg': "TL amygdala R",
                   'l_atl_m': "TL anter. temp lobe med L",
                   'r_atl_m': "TL anter. temp lobe med R",
                   'l_atl_l': "TL anter. temp lobe lat L",
                   'r_atl_l': "TL anter. temp lobe lat R",
                   'l_mtl_hippo': "TL hippocampus L",
                   'r_mtl_hippo': "TL hippocampus R",
                   'l_mtl_parahippo': "TL parahippo. L",
                   'r_mtl_parahippo': "TL parahippo. R",
                   'l_pl': "PL sup parie gyrus L",
                   'r_pl': "PL sup parie gyrus R",
                   'l_orbitofrontal': "FL ant orbital, inf fron gyrus L",
                   'r_orbitofrontal': "FL ant orbital, inf fron gyrus R",
                   'l_dorsolateral': "FL mid fron gyrus L",
                   'r_dorsolateral': "FL mid fron gyrus R",
                   'l_superior': "FL sup fron gyrus L",
                   'r_superior': "FL sup fron gyrus R",
                   'l_posterior': "FL precentral gyrus L",
                   'r_posterior': "FL precentral gyrus R",
                   'l_latventricle_temph': "Lat vent temp horn L",
                   'r_latventricle_temph': "Lat vent temp horn R",
                   'l_latventricle': "Lat vent L",
                   'r_latventricle': "Lat vent R",
                   'ventricle3': "Third ventricle",
                   'genu': "FL subgenual fron cortex",
                   'body': "corpus callosum"
                   }


regions = anterior_temporal_regions + medical_temporal_regions + parietal_lobe_regions + \
          anterior_atrophy_regions + posterior_atrophy_regions + other_regions

team = defaultdict(dict) # key is initials of radiologist, content is rating
with open('ratings.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['rev_initials'] in ['SQ', 'PJ test', 'PJ-trial chart', 'test', 'test (JL)']: continue
        id = row['id'].split()[0]
        name = row['rev_initials'].lower()
        for reg in regions:
            if id not in team[name]:
                team[name][id] = {}
            if row[reg] == '':
                score = 0
            elif row[reg] == '99':
                score = -1 # Not available
            else:
                score = int(row[reg])
            team[name][id][reg] = score

for name in team:
    print(name, 'evaluated', len(team[name].keys()), 'subjects with', len(team[name]['1'].keys()), 'regions')

filename_dummyID = {}
with open('50cases_with_dummyID.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        filename_dummyID[row['filename']] = row['dummy']

shap = {}
with open('Radio_shap_COG_region_scores_block2conv.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['filename'] not in filename_dummyID: continue # NC or MCI subjects
        id = filename_dummyID[row['filename']]
        shap[id] = {}
        for region in regions:
            val = 0
            for reg_idx in region_to_atlas[region]:
                val += float(row[str(reg_idx)])
            shap[id][region] = val

def get_averaged_radio_scores(region, team):
    # since there are 50 subjects, the return will be a list of 50 numbers
    ans = []
    for id in range(1, 51):
        subject_score = []
        for name in team:
            if team[name][str(id)][region] != -1:
                subject_score.append(team[name][str(id)][region])
        ans.append((sum(subject_score)+0.000001) / len(subject_score))
    assert len(ans) == 50, "length wrong"
    return ans

def get_shap_scores(region, shap):
    # since there are 50 subjects, the return will be a list of 50 numbers
    ans = []
    for id in range(1, 51):
        ans.append(shap[str(id)][region])
    assert len(ans) == 50, "length wrong"
    return ans

if __name__ == "__main__":
    correlation = {}
    for region in regions:
        vec1 = get_averaged_radio_scores(region, team)
        vec2 = get_shap_scores(region, shap)
        c, p = stats.spearmanr(vec1, vec2)
        correlation[region] = c

    colors = ['#bfe2e3', '#69869c', '#36896e', '#c22e00', '#c6d645', '#ffd3b6', '#b2b2b2', '#4724a9',
              '#9bc84d', '#7141ae', '#d2a782', '#933b61', '#435299', '#d88770', '#765aa8', '#719795'] * 2

    pool = []
    for i in range(len(regions)):
        pool.append((correlation[regions[i]], regions[i]))
    pool.sort()
    pool = pool[::-1]

    regions = [p[1] for p in pool]
    Val = [p[0] for p in pool]
    regions = [region_to_names[reg] for reg in regions]
    fig, ax = plt.subplots(dpi=300, figsize=(9, 4))
    index = [i for i in range(len(regions))]
    ax.bar(index, Val, color=colors[:len(regions)])
    ax.set_ylabel('Spearman\'s rank correlation', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.set_xticks(index)
    ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.grid(which='major', axis='both', linestyle='--')
    plt.savefig('bar_block2conv.png', bbox_inches='tight')
    plt.close()


