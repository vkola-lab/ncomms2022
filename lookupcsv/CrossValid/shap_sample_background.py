import csv
import numpy as np
import random
import copy
from random import sample

def get_content(csv_file):
    content = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content.append(row)
    return content

def sample_N_cases_for_ADD_shap_background(content, N):
    ADD = get_vari_group(content, 'ADD')
    nADD = get_vari_group(content, 'nADD')
    print(len(nADD))
    ADD = sample(ADD, N)
    nADD = sample(nADD, N)
    print(len(ADD), len(nADD))
    return ADD + nADD

def sample_N_cases_for_COG_shap_background(content, N):
    NC = get_vari_group(content, 'NC')
    MCI = get_vari_group(content, 'MCI')
    DE = get_vari_group(content, 'DE')
    NC = sample(NC, N)
    MCI = sample(MCI, N)
    DE = sample(DE, N)
    print(len(NC), len(MCI), len(DE))
    return NC + MCI + DE

def create_csv(filename, content):
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        column_names = get_columns("cross0/train.csv")
        spamwriter.writerow(column_names)
        for row in content:
            spamwriter.writerow([row[col] if col in row else '' for col in column_names])

def get_columns(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            return row

def get_vari_group(content, group):
    res = []
    for case in content:
        if case['NC'] == '1' and group == 'NC':
            res.append(case)
        elif case['MCI'] == '1' and group == 'MCI':
            res.append(case)
        elif case['DE'] == '1' and group == 'DE':
            res.append(case)
        elif '1' in case['ADD'] and group == 'ADD':
            res.append(case)
        elif case['ADD'] and case['ADD'][0] == '0' and group == 'nADD':
            res.append(case)
    return res


if __name__ == "__main__":
    for i in range(5):
        content = get_content("cross{}/train.csv".format(i))
        cases = sample_N_cases_for_ADD_shap_background(content, 2)
        create_csv("cross{}/ADD_shap_background4.csv".format(i), cases)
        cases = sample_N_cases_for_COG_shap_background(content, 1)
        create_csv("cross{}/COG_shap_background4.csv".format(i), cases)

    # content = get_content("cross0/test.csv")
    # cases = sample_N_cases_for_shap_background(content, 35)
    # create_csv("cross0/ADD_test_shap.csv", cases)