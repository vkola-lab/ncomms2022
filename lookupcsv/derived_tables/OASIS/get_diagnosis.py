import csv
import collections

count_DLB = collections.defaultdict(int)
count_FTD = collections.defaultdict(int)
count_VASC = collections.defaultdict(int)
count_PD = collections.defaultdict(int)

table1 = '../../raw_tables/OASIS/UDS_D1_Clinician_Diagnosis.csv'
table2 = '../../raw_tables/OASIS/ADRC_Clinical_Data.csv'

content = []

def add_NC(row, case):
    if row['NORMCOG'] == '1': # NC case
        for vari in ['MCI', 'DE', 'COG', 'AD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other']:
            case[vari] = 0
        case['NC'] = 1
        if row['PARK'] == '1':
            case['PD'] = 1
        elif row['PARK'] == '0':
            case['PD'] = 0
    return case

def add_MCI(row, case):
    if '1' in [row['MCIAMEM'], row['MCIAPLUS'], row['MCIAPLAN'], row['MCIAPATT'], row['MCIAPEX'], row['MCIAPVIS']]: # MCI case
        for vari in ['NC', 'DE', 'AD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other']:
            case[vari] = 0
        case['MCI'] = 1
        case['COG'] = 1
        if row['PARK'] == '1':
            case['PD'] = 1
        elif row['PARK'] == '0':
            case['PD'] = 0
    return case

def add_DE(row, case):
    if row['DEMENTED'] == '1': # dementia case
        case['DE'] = 1
        case['COG'] = 2
        case['NC'] = 0
        case['MCI'] = 0
        # parkinson disease check
        if row['PARK'] == '1': # PD present
            case['PD'] = 1
            case['PDD'] = 1
        elif row['PARK']=='0': # no PD
            case['PD'] = 0
            case['PDD'] = 0
        # dementia with Lewy Bodies check
        if row['PARK'] == '0' and row['DLB'] == '1':
            case['DLB'] = 1
        else:
            case['DLB'] = 0
        # Lewy Body dementia check
        if row['DLB'] == '1' or row['PARK'] == '1':
            case['LBD'] = 1
        else:
            case['LBD'] = 0
        # Alzheimer's disease check
        if row['PROBAD'] == '1' or row['POSSAD'] == '1': # AD present
            case['AD'] = 1
        elif row['PROBAD'] == '0' or row['POSSAD'] == '0': # no AD
            case['AD'] = 0
        # Vascular dementia check
        if '1' in [row['VASC'], row['VASCPS']]: # VD present
            case['VD'] = 1
        elif '0' in [row['VASC'], row['VASCPS']]: # no LBD
            case['VD'] = 0
        # FTD check
        if '1' in [row['FTD'], row['PPAPH'], row['PNAPH'], row['SEMDEMAN'], row['SEMDEMAG'], row['PPAOTHR']]: # any type present
            case['FTD'] = 1
        elif '0' in [row['FTD'], row['PPAPH'], row['PNAPH'], row['SEMDEMAN'], row['SEMDEMAG'], row['PPAOTHR']]: # none of them exist
            case['FTD'] = 0
        # other dementia check
        if '1' in [row['ALCDEM'], row['DEMUN'], row['PSP'], row['CORT'], row['HUNT'], row['PRION'], row['MEDS'],
                   row['DYSILL'], row['DEP'], row['OTHPSY'], row['DOWNS'], row['STROKE'], row['HYCEPH'],
                   row['BRNINJ'], row['NEOP'], row['COGOTH'], row['COGOTH2'], row['COGOTH3']]:
            case['Other'] = 1
        else:
            case['Other'] = 0
    return case

with open(table1, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        case = {}
        for key in ['UDS_D1DXDATA ID']:
            case[key] = row[key]
        case = add_NC(row, case)
        case = add_MCI(row, case)
        case = add_DE(row, case)
        content.append(case)

columnNames = ['UDS_D1DXDATA ID', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other']

with open('diagTable.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columnNames)
    writer.writeheader()
    for case in content:
        writer.writerow(case)