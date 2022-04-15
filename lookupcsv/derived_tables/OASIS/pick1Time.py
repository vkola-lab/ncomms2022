import csv

included = set()
content = {}
readMap = {}

with open('mri_diag_table_6months.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        readMap[row['ID']] = (row['folderName'], row['NC'], row['MCI'], row['DE'], row['COG'],
                              row['AD'], row['PD'], row['FTD'], row['VD'], row['DLB'], row['LBD'], row['PDD'], row['Other'])

for idx in [9, 8, 7, 6, 5, 3, 2, 1, 12]:
    for id in readMap:
        if id not in included and readMap[id][idx]=='1':
            included.add(id)
            content[id] = readMap[id]

print(len(content))

with open('unique_mri_diag_table_6months.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['folderName', 'ID', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'LBD', 'PDD', 'Other'])
    writer.writeheader()
    case = {}
    for id in content:
        case['folderName'] = content[id][0]
        case['ID'] = id
        for i, vari in enumerate(['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'LBD', 'PDD', 'Other']):
            case[vari] = content[id][i+1]
        writer.writerow(case)