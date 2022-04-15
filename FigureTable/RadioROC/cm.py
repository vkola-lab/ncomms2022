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
from collections import defaultdict

# get diagnoses from radiologists' rating form
team = defaultdict(dict) # key is initials of radiologist, content is rating
with open('ratings.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['rev_initials'] in ['SQ', 'PJ test', 'PJ-trial chart', 'test', 'test (JL)']: continue
        id = row['id'].split()[0]
        team[row['rev_initials'].lower()][id] = row['dem_dx'] if row['dem_dx'] == '1' else '0'

Label = {} # key is dummy id, content is ADD label (0 is non-ADD, 1 is ADD)
with open('../../lookupcsv/derived_tables/NACC_ALL/50cases_dummy.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        Label[row['dummy']] = row['ADD']

print(Label)
print(team)

CM = []
for name in team:
    pred, label = [], []
    for id in Label:
        assert id in team[name], 'id not found from team: ' + name
        pred.append(team[name][id])
        label.append(Label[id])
    cm = confusion_matrix(label, pred, labels=['0', '1'])
    CM.append(cm)
CM = np.mean(np.array(CM), axis=0)
print(CM.shape)

fig = plot_confusion_matrix(CM, ['nADD', 'ADD'])
fig.savefig('cm/radiologists_cm.png')
