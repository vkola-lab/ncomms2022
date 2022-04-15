import csv
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from utils import plot_confusion_matrix
from performance_eval import crossValid_cm_4_4

def get_label():
    # get ground truth labels from 100cases_dummy.csv
    label = {}  # key is dummy id, content is ground truth label
    with open('../../lookupcsv/derived_tables/NACC_ALL/100cases_dummy.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['COG'] == '0':
                label[int(row['dummy'])] = 0
            elif row['COG'] == '1':
                label[int(row['dummy'])] = 1
            elif row['ADD'] == '1':
                label[int(row['dummy'])] = 2
            elif row['ADD'] == '0':
                label[int(row['dummy'])] = 3
    return label

def get_team():
    map = {'Normal Cognition': 0,
          'Mild Cognitive Impairment': 1,
          'Dementia _ Alzheimer\'s Disease Dementia': 2,
          'Dementia _ not Alzheimer\'s Disease Dementia': 3}

    # get diagnoses from neurologists' rating form
    team = defaultdict(dict)  # key is initials of radiologist, content is rating
    d_id = 0 # doctor dummy id
    for i in range(1, 18):
        with open('neurologists/n{}.csv'.format(i), 'r') as csv_file:
            id = 1 # subject dummy id
            reader = csv.DictReader(csv_file)
            for row in reader:
                team[d_id][id] = map[row['Diagnosis Label']]
                id += 1
            d_id += 1
    return team

def cm_neurologists(team, Label):
    # this function generate averaged 4 by 4 confusion matrix from all neurologists
    CM = []
    for d_id in team:
        pred, label = [], []
        for id in Label:
            assert id in team[d_id], 'id not found from team'
            pred.append(team[d_id][id])
            label.append(Label[id])
        print(pred)
        print(label)
        cm = confusion_matrix(label, pred, labels=[0, 1, 2, 3])
        CM.append(cm)
    total = np.array(CM)
    print(total.shape)
    np.save("cm/all_neurologists.npy", total)
    CM = np.mean(np.array(CM), axis=0)
    print(CM.shape)
    fig = plot_confusion_matrix(CM, ['NC', 'MCI', 'ADD', 'nADD'])
    fig.savefig('cm/neurologists_cm.png')


def model_cm(csv_files, name):
    cm = crossValid_cm_4_4(csv_files)
    np.save(name + '_cm4.npy', cm)
    fig = plot_confusion_matrix(cm, ['NC', 'MCI', 'ADD', 'nADD'])
    fig.savefig(name + '_cm4.png')


if __name__ == "__main__":
    team = get_team()
    label = get_label()
    cm_neurologists(team, label)

    # for i in range(5):
    #     csv_files = ['../../tb_log/CNN_3ways_special2_cross{}/neuro_test_eval_after.csv'.format(i)]
    #     model_cm(csv_files, 'cm/MRI_{}'.format(i))
    #
    #     csv_files = ['../../tb_log/CatBoost_special1_nonImg_cross{}/neuro_test_eval.csv'.format(i)]
    #     model_cm(csv_files, 'cm/nonimg_{}'.format(i))
    #
    #     csv_files = ['../../tb_log/CatBoost_special1_Fusion_cross{}/neuro_test_mri_eval.csv'.format(i)]
    #     model_cm(csv_files, 'cm/fusion_{}'.format(i))

    for mode in ['MRI', 'fusion', 'nonimg']:
        datas = []
        for i in range(5):
            data = np.load('cm/{}_{}_cm4.npy'.format(mode, i))
            datas.append(data)
        np.save("{}_CM.npy".format(mode), np.array(datas))





