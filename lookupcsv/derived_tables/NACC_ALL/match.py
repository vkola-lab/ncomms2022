import csv
from collections import defaultdict
import datetime

diagTable = 'diagTable.csv'
mriTable  = '../../raw_tables/NACC_ALL/kolachalama12042020mri.csv'

# mri hashmap id: time: zip
# diag hashmap id: time: results

mriMap = defaultdict(dict)
with open(mriTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id = row['NACCID']
        zipname = row['NACCMRFI']
        time = (int(row['MRIYR']), int(row['MRIMO']), int(row['MRIDY']))
        mriMap[id][time] = zipname

print(len(mriMap))

diagMap = defaultdict(dict)
with open(diagTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id = row['NACCID']
        time = (int(row['VISITYR']), int(row['VISITMO']), int(row['VISITDAY']))
        diagresult = (row['NC'], row['MCI'], row['DE'], row['COG'], row['AD'], row['PD'], row['FTD'], row['VD'], row['LBD'], row['PDD'], row['DLB'], row['Other'])
        diagMap[id][time] = diagresult

print(len(diagMap))

def find_closest(mri_time, diags):
    min_diff = 1000000
    ans = None
    mriDate = datetime.date(*mri_time)
    diagTimes = diags.keys()
    for time in diagTimes:
        diagDate = datetime.date(*time)
        diff = diagDate - mriDate
        diff = abs(diff.days)
        if diff < min_diff and diff < 183:  # within +/- 6 months
            min_diff = diff
            ans = time
    return ans

content = []

for id in mriMap:
    if id not in diagMap:
        continue
    mris = mriMap[id]
    diags = diagMap[id]
    print(id, '---------------------------------------------')
    print(mris)
    print(diags)
    pairs = []
    for mri_time in mris:
        diag_time = find_closest(mri_time, diags)
        if not diag_time: continue
        if diags[diag_time][0] == '': continue
        pairs.append((mris[mri_time], id, diags[diag_time], diag_time))
    print(pairs)
    content.extend(pairs)

with open('mri_diag_table_6months.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['zipname', 'ID', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other'])
    writer.writeheader()
    case = {}
    for row in content:
        case['zipname'] = row[0]
        case['ID'] = row[1]
        for i, vari in enumerate(['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other']):
            case[vari] = row[2][i]
        writer.writerow(case)

with open('zip_id_data_6months.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['zipname', 'ID', 'Day', 'Month', 'Year'])
    writer.writeheader()
    case = {}
    for row in content:
        case['zipname'] = row[0]
        case['ID'] = row[1]
        case['Year'], case['Month'], case['Day'] = row[3]
        writer.writerow(case)




