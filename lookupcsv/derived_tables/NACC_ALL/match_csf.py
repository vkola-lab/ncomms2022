import csv
from collections import defaultdict
import datetime

mriTable = 'zip_id_data_6months.csv'
csfTable  = '../../raw_tables/NACC/kolachalama06022020csf.csv'
diagTable = 'diagTable.csv'

diagMap = defaultdict(dict)
with open(diagTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id = row['NACCID']
        time = (int(row['VISITYR']), int(row['VISITMO']), int(row['VISITDAY']))
        diagMap[id][time] = (row['COG'], row['AD'])

mriMap = defaultdict(dict)
with open(mriTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id = row['ID']
        time = (int(row['Year']), int(row['Month']), int(row['Day']))
        mriMap[id][time] = row['zipname']

csfMap = defaultdict(dict)
with open(csfTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id = row['NACCID']
        time = (int(row['CSFLPYR']), int(row['CSFLPMO']), int(row['CSFLPDY']))
        csfMap[id][time] = [row['CSFABETA'], row['CSFPTAU'], row['CSFTTAU']]

def find_closest(target_time, series):
    min_diff = 1000000
    ans = None
    target_date = datetime.date(*target_time)
    series_times = series.keys()
    for time in series_times:
        date = datetime.date(*time)
        diff = target_date - date
        diff = abs(diff.days)
        if diff < min_diff and diff < 183:  # within +/- 6 months
            min_diff = diff
            ans = time
    return ans

content = []

for id in diagMap:
    if id not in csfMap or id not in mriMap:
        continue
    diag = diagMap[id]
    csfs = csfMap[id]
    mris = mriMap[id]
    pairs = []
    for diag_time in diag:
        csf_time = find_closest(diag_time, csfs)
        mri_time = find_closest(diag_time, mris)
        if not csf_time: continue
        pairs.append((None, id, csfs[csf_time], diag[diag_time], csf_time))
    print(pairs)
    content.extend(pairs)

with open('diag_csf_table_6months.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['zipname', 'ID', 'CSFABETA', 'CSFPTAU', 'CSFTTAU', 'COG', 'AD'])
    writer.writeheader()
    case = {}
    for row in content:
        case['zipname'] = row[0]
        case['ID'] = row[1]
        for i, vari in enumerate(['CSFABETA', 'CSFPTAU', 'CSFTTAU']):
            case[vari] = row[2][i]
        for i, vari in enumerate(['COG', 'AD']):
            case[vari] = row[3][i]
        writer.writerow(case)
