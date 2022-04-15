import csv


def get_label():
    # get ground truth labels from 100cases_dummy.csv
    label = {}  # key is dummy id, content is ground truth label
    with open('../../lookupcsv/derived_tables/NACC_ALL/100cases_dummy.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['COG'] == '0':
                label[int(row['dummy'])] = 'NC'
            elif row['COG'] == '1':
                label[int(row['dummy'])] = 'MCI'
            elif row['COG'] == '2':
                if row['AD'] == '1':
                    label[int(row['dummy'])] = 'AD'
                else:
                    label[int(row['dummy'])] = 'nADD'
    return label

def get_diag(csvfile):
    diags = []
    with open(csvfile, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            diag = row['Diagnosis Label']
            if diag == 'Normal Cognition':
                diag = 'NC'
            if diag == 'Mild Cognitive Impairment':
                diag = 'MCI'
            if diag == 'Dementia _ Alzheimer\'s Disease Dementia':
                diag = 'AD'
            if diag == 'Dementia _ not Alzheimer\'s Disease Dementia':
                diag = 'nADD'
            diags.append(diag)
    return diags

def create_new_table(csvfile):
    label = get_label()
    diags = get_diag(csvfile)
    fieldnames = ['COG', 'COG_pred', 'ADD', 'ADD_pred']
    content = []
    for i, d in enumerate(diags):
        case = {}
        l = label[i+1]
        if l == 'NC':
            case['COG'] = 0
        elif l == 'MCI':
            case['COG'] = 1
        elif l == 'AD':
            case['COG'] = 2
            case['ADD'] = 1
        elif l == 'nADD':
            case['COG'] = 2
            case['ADD'] = 0
        if d == 'NC':
            case['COG_pred'] = 0
        elif d == 'MCI':
            case['COG_pred'] = 1
        elif d == 'AD':
            case['COG_pred'] = 2
            case['ADD_pred'] = 1
        elif d == 'nADD':
            case['COG_pred'] = 2
            case['ADD_pred'] = 0
        content.append(case)
    with open(csvfile.replace('.csv', '_new.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in content:
            writer.writerow(data)

for i in range(1, 18):
    create_new_table('neurologists/n{}.csv'.format(i))



