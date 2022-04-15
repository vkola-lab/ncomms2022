import csv

content = []
columnNames = ['NACCID', 'VISITMO', 'VISITDAY', 'VISITYR', 'SEX', 'EDUC', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other']

def add_NC(row, case):
    if row['NACCUDSD'] == '1': # NC case
        for vari in ['MCI', 'DE', 'COG', 'AD', 'FTD', 'VD', 'LBD', 'PDD', 'DLB', 'Other']:
            case[vari] = 0
        case['NC'] = 1
        if row['PARK'] == '1':
            case['PD'] = 1
        elif row['PARK'] == '0':
            case['PD'] = 0
    return case

def add_MCI(row, case):
    if row['NACCUDSD'] == '3' or row['NACCTMCI'] in ['1', '2', '3', '4']: # MCI case
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
    if row['NACCUDSD'] == '4': # dementia case
        case['DE'] = 1
        case['COG'] = 2
        case['NC'] = 0
        case['MCI'] = 0
        # parkinson disease (dementia) check
        if row['PARK'] == '1': # PD present and PDD present
            case['PD'] = 1
            case['PDD'] = 1
        else: # no PD
            case['PD'] = 0
            case['PDD'] = 0
        # dementia with Lewy Bodies check
        if row['PARK'] == '0' and row['NACCLBDE'] == '1':
            case['DLB'] = 1
        else:
            case['DLB'] = 0
        # Lewy Body dementia check
        if row['NACCLBDE'] == '1' or row['PARK'] == '1':
            case['LBD'] = 1
        else:
            case['LBD'] = 0
        # NACCLBDP = 1 primary; 2 contributing (same for park)
        # Alzheimer's disease check (primary and contributing)
        if row['NACCALZD'] == '1': # AD present
            case['AD'] = 1
        elif row['NACCALZD'] == '0': # no AD
            case['AD'] = 0
        # NACCALZP = 1 primary; 2 contributing
        # Vascular dementia check (primary and contributing)
        if '1' in [row['VASC'], row['VASCPS'], row['CVD']]: # VD present
            case['VD'] = 1
        elif '0' in [row['VASC'], row['VASCPS'], row['CVD']]: # no LBD
            case['VD'] = 0
        # CVDIF 1 primary 2 contributing or
        # VASCIF 1 primary 2 contributing or
        # VASCPSIF 1 primary 2 contributing or
        # FTD check
        if '1' in [row['FTD'], row['FTLDNOS'], row['PPAPH'], row['NACCPPA']]: # any type present
            case['FTD'] = 1
        elif '0' in [row['FTD'], row['FTLDNOS'], row['PPAPH'], row['NACCPPA']]: # none of them exist
            case['FTD'] = 0
        # FTDIF 1 primary 2 contributing or
        # FTLDNOIF 1 primary 2 contributing or
        # PPAPHIF 1 primary 2 contributing or
        # NACCPRET
        # other dementia check
        if '1' in [row['MSA'], row['PSP'], row['CORT'], row['HYCEPH'], row['EPILEP'], row['NEOP'], row['HIV'], row['OTHCOG'],
                row['DEP'], row['BIPOLDX'], row['SCHIZOP'], row['ANXIET'], row['DELIR'], row['PTSDDX'], row['OTHPSY'],
                   row['ALCDEM'], row['IMPSUB'], row['DYSILL'], row['DEMUN'], row['COGOTH'], row['COGOTH2'], row['COGOTH3'],
                row['HUNT'], row['DOWNS'], row['PRION'], row['BRNINJ'], row['MEDS']]:
            case['Other'] = 1
        else:
            case['Other'] = 0
    return case


diagTable = '../../raw_tables/NACC_ALL/kolachalama12042020.csv'
with open(diagTable, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        case = {}
        for key in ['NACCID', 'VISITMO', 'VISITDAY', 'VISITYR', 'SEX', 'EDUC']:
            case[key] = row[key]
        case = add_NC(row, case)
        case = add_MCI(row, case)
        case = add_DE(row, case)
        content.append(case)


with open('diagTable.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columnNames)
    writer.writeheader()
    for case in content:
        writer.writerow(case)
