import csv

def get_diag(cog, add):
    if cog == '0':
        return 'NC'
    elif cog == '1':
        return 'MCI'
    elif cog == '2':
        if add == '1':
            return 'AD'
        elif add == '0':
            return 'nADD'
        return ''
    return ''

def get_cohort(path):
    if 'ADNI' in path:
        return 'ADNI'
    elif 'NACC' in path:
        return 'NACC'
    elif 'FHS' in path:
        return 'FHS'
    elif 'OASIS' in path:
        return "OASIS"
    elif 'AIBL' in path:
        return 'AIBL'
    elif 'PPMI' in path:
        return 'PPMI'
    elif 'NIFD' in path:
        return 'NIFD'
    elif "Stanford" in path:
        return "LBDSU"


scanner_label = {"philips": '0', "ge": '1', "siemens": '2'}
scanner = {}
with open('../../lookupcsv/derived_tables/NACC_ALL/scanner.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        scanner[row['filename']] = (row['brand'], row['model'])
print('done reading scanner table')

centerID = {}
with open('../../lookupcsv/raw_tables/NACC_ALL/kolachalama12042020mri.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        key = row['NACCMRFI'].strip('.zip')
        centerID[key] = row['NACCADC']
with open('../../lookupcsv/raw_tables/NACC/kolachalama06022020mri.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        key = row['NACCMRFI'].strip('.zip')
        centerID[key] = row['NACCADC']

# content = []
# with open('../../lookupcsv/dataset_table/NACC_ALL/NACC.csv', 'r') as csv_file:
#     reader = csv.DictReader(csv_file)
#     for row in reader:
#         case = {}
#         if row['filename'] in scanner and row['COG'] == '0':
#             case['scanner'] = scanner[row['filename']][0]
#             case['diag'] = get_diag(row['COG'], row['ADD'])
#             case['dir'] = row['path'] + row['filename']
#             case['scanner_label'] = scanner_label.get(case['scanner'].lower(), '')    
#             if case['scanner_label']:
#                 content.append(case)
# print('done with content')

# with open('train_table.csv', 'w') as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=['dir', 'diag', 'scanner', 'scanner_label'])
#     writer.writeheader()
#     for case in content:
#         writer.writerow(case)

content = []
with open('../../lookupcsv/CrossValid/all.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        case = {}
        if row['filename'] in scanner:
            case['scanner_brand'] = scanner[row['filename']][0]
            case['scanner_model'] = scanner[row['filename']][1]
        case['diag'] = get_diag(row['COG'], row['ADD'])
        case['MRI_path'] = row['path']
        case['emb_path'] = '/home/sq/multi-task/t_SNE/early_embeddings/'
        case['cohort'] = get_cohort(row['path'])
        case['filename'] = row['filename']
        if row['filename'][:3] == 'mri':
            key = row['filename'].split('_')[0]
        elif row['filename'][:3] == 'NAC':
            key = "_".join(row['filename'].split('_')[:2])
        else:
            key = "None"
        print(key)
        if key in centerID:
            case['ADC_id'] = centerID[key]
        content.append(case)
print('done with content')

with open('tSNE_table.csv', 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['filename', 'MRI_path', 'emb_path', 'cohort', 'diag', 'scanner_brand', 'scanner_model', 'ADC_id'])
    writer.writeheader()
    for case in content:
        writer.writerow(case)

