import csv
import numpy as np
from tabulate import tabulate
from scipy.stats import f_oneway, chi2_contingency

def read_csv(csv_table, col_name):
    content = []
    with open(csv_table, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content.append(row[col_name])
    return content

def add(TABLES, vari):
    NC, MCI, AD, OT = [], [], [], []
    for table in TABLES:
        if table == 'NACC':
            csv_table = '../../lookupcsv/dataset_table/NACC_ALL/' + table + '.csv'
        else:
            csv_table = '../../lookupcsv/dataset_table/' + table + '/' + table + '.csv'
        with open(csv_table, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if vari not in row or row[vari] == '': continue
                if row['COG'] == '0':
                    NC.append(row[vari])
                elif row['COG'] == '1':
                    MCI.append(row[vari])
                elif row['AD'] == '1':
                    AD.append(row[vari])
                elif row['COG'] == '2':
                    OT.append(row[vari])
    return NC, MCI, AD, OT

def format_stat(list):
    import unicodedata
    list = [float(a) for a in list]
    list = sorted(list)
    median, mi, ma = np.median(list), min(list), max(list)
    mean, std = np.mean(list), np.std(list)
    return '{:.2f}'.format(mean)+chr(177)+'{:.2f}'.format(std)

def format_stat_cate(list):
    count_male = list.count('male')
    count_female = list.count('female')
    return str(count_male) + ' ({:.2f}%)'.format(100 * count_male / (count_male+count_female))

def format_race(list):
    ans, string = [], ''
    cate = ['whi', 'blk', 'ans', 'ind', 'haw', 'mix']
    for key in cate:
        ans.append(list.count(key))
    for key, count in zip(cate, ans):
        string += key+':'+str(count)+' '
    return string

def format_apoe(list):
    count_pos = list.count('1')
    count_neg = list.count('0')
    return str(count_pos) + ' ({:.3f}%)'.format(100 * count_pos / (count_pos + count_neg))

def initalize_table():
    content = {}
    for dataset in ['ADNI', 'NACC', 'NIFD', 'PPMI', 'AIBL', 'OASIS', 'FHS', 'Stanford']:
        if dataset not in content: content[dataset] = {}
        for stage in ['NC', 'MCI', 'AD', 'OT']:
            if stage not in content[dataset]: content[dataset][stage] = {}
            for variable in ['age', 'gender', 'education', 'race', 'apoe', 'mmse', 'cdr', 'moca']:
                content[dataset][stage][variable] = []
    return content

def add_content(groups, dataset, col, table):
    table[dataset]['NC'][col], table[dataset]['MCI'][col], table[dataset]['AD'][col], table[dataset]['OT'][col] = add(groups, col)

def anova(pool):
    if not pool: return 'NA'
    for i in range(len(pool)):
        pool[i] = [float(a) for a in pool[i]]
    _, p = f_oneway(*pool)
    if p < 0.001: return '<0.001'
    return '{:.3f}'.format(p)

def chi_square(pool):
    if not pool: return 'NA'
    element = set([])
    for p in pool:
        for a in p:
            element.add(a)
    print(element)
    matrix = []
    for p in pool:
        matrix.append([p.count(e) for e in element])
    print(matrix)
    chi2, p, dof, exp = chi2_contingency(np.array(matrix))
    if p < 0.001: return '<0.001'
    return '{:.3f}'.format(p)

def gene_table1():

    table = initalize_table()  # content[ADNI][NC][age] = [age1, age2, age3, .....]

    for dataset in ['ADNI', 'NACC', 'NIFD', 'PPMI', 'AIBL', 'OASIS', 'FHS', 'Stanford']:
        groups = ['ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO'] if dataset == 'ADNI' else [dataset]
        for vari in ['filename', 'age', 'gender', 'race', 'apoe', 'education', 'mmse', 'moca', 'cdr']:
            add_content(groups, dataset, vari, table)

    content = []
    # merge everything into one table
    for dataset in ['ADNI', 'NACC', 'NIFD', 'PPMI', 'AIBL', 'OASIS', 'FHS', 'Stanford']:
        for stage in ['NC', 'MCI', 'AD', 'OT']:
            row = [dataset + '_' + stage + "(n={})".format(len(table[dataset][stage]['filename']))]
            N = len(table[dataset][stage]['filename'])
            for variable in ['age', 'gender', 'education', 'race', 'apoe', 'mmse', 'cdr', 'moca']:
                if table[dataset][stage][variable]:
                    if variable == 'gender':
                        row.append(format_stat_cate(table[dataset][stage][variable]))
                    elif variable == 'race':
                        row.append(format_race(table[dataset][stage][variable]))
                    elif variable == 'apoe':
                        row.append(format_apoe(table[dataset][stage][variable]))
                    else:
                        row.append(format_stat(table[dataset][stage][variable]))
                    if len(table[dataset][stage][variable]) < N:
                        row[-1] += '^'
                else:
                    row.append('NA')
            content.append(row)
        # add p value
        row = ['p-val']
        for variable in ['age', 'gender', 'education', 'race', 'apoe', 'mmse', 'cdr', 'moca']:
            pool = []
            for stage in ['NC', 'MCI', 'AD', 'OT']:
                if table[dataset][stage][variable]: pool.append(table[dataset][stage][variable])
            if variable in ['age', 'education', 'mmse', 'cdr', 'moca']:
                p = anova(pool)
            else:
                print(dataset)
                p = chi_square(pool)
            row.append(p)
        content.append(row)
        content.append(['----'] * 9)

    print(tabulate(content, headers=["dataset", 'age', 'gender', 'education', 'race', 'apoe', 'mmse', 'cdr', 'moca']))

    with open("demographic.csv", "w") as f:
        f.write(tabulate(content, headers=["dataset", 'age', 'gender', 'education', 'race', 'apoe', 'mmse', 'cdr', 'moca'], tablefmt="csv"))

if __name__ == "__main__":

    gene_table1()
