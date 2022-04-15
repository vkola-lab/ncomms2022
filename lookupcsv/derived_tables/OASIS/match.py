import csv
from collections import defaultdict
import datetime
import os
from glob import glob

diagTable = 'diagTable.csv'

diagMap = defaultdict(dict)
with open(diagTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id, date = row['UDS_D1DXDATA ID'].split('_')[0], row['UDS_D1DXDATA ID'].split('_')[2]
        diagresult = (row['NC'], row['MCI'], row['DE'], row['COG'], row['AD'], row['PD'], row['FTD'], row['VD'], row['LBD'], row['PDD'], row['DLB'], row['Other'])
        diagMap[id][date] = diagresult

print(len(diagMap))

MRI_list = glob("/data_2/OASIS/*_*/")
mriMap = defaultdict(dict)

for mri in MRI_list:
    id, date = mri.split('/')[-2].split('_')[0], mri.split('/')[-2].split('_')[2]
    mriMap[id][date] = mri

print(len(mriMap))

def find_closest(id, date):
    min_diff = 1000000
    ans = None
    mriDate = int(date.strip('d'))
    diagTimes = diagMap[id].keys()
    for time in diagTimes:
        diagDate = int(time.strip('d'))
        diff = diagDate - mriDate
        diff = abs(diff)
        if diff < min_diff and diff < 183:  # within +/- 6 months
            min_diff = diff
            ans = time
    return ans


content = []

for mri in MRI_list:
    row = [mri]
    id, date = mri.split('/')[-2].split('_')[0], mri.split('/')[-2].split('_')[2]
    diagDate = find_closest(id, date)
    row.append(id)
    if diagDate:
        row.extend(diagMap[id][diagDate])
    else:
        row.extend(['']*12)
    content.append(row)

content = sorted(content, key=lambda x:x[0])

with open('mri_diag_table_6months.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['folderName', 'ID', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other'])
    writer.writeheader()
    case = {}
    for row in content:
        case['folderName'] = row[0]
        case['ID'] = row[1]
        for i, vari in enumerate(['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other']):
            case[vari] = row[2+i]
        writer.writerow(case)

for id in diagMap:
    for time, diag in diagMap[id].items():
        if diag[5] == '1':
            print('PD found', diag, time, id, mriMap[id])

